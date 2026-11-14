"""Small from-zero, own-policy PPO pilot on the controlled pipeline diagnostic."""
import argparse
import csv
import dataclasses
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.native import load_cast_core
from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig
from agent.runtime import PaperRewardConfig,apply_paper_rewards
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,actor_from_joint_payload
from agent.runtime.shared_bit_mask_policy import SharedBitMaskActor
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.table_observation import TABLE_STATE_PROJECTION
from agent.runtime.state_collector import PHASE_FEATURE_FIELDS
from scripts.revision.probe_phase_pipeline import CONFIG,FixedWriterHooks,execute_pair,Probe
from scripts.revision.train_heterogeneous_phase_updates import make_manager,payload,actor_training_rows
from scripts.revision.train_joint_skew_domain import fresh_actor
from scripts.revision.train_ycsb_onpolicy import OnPolicySampler,verify_onpolicy
from scripts.revision.train_ycsb_mixed_prefix import checked_rows
from scripts.revision.run_audited_credit_pressure import AuditedManager,LinearScorePolicy,fit_linear_rewarded
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.audit_serial_history import verify_history


class LocalRuleHooks(FixedWriterHooks):
    """Shared local key counts; competitor retains identical fixed policy."""
    def __init__(self,manager,threshold):
        super().__init__(manager)
        self.item=ItemFrequencyHooks(manager,minimum_accesses=threshold,window_ms=1000.)

    def _select_and_apply(self,txn,state):
        # This baseline makes no actor/table calls; fixed writer makes none either.
        return None

    def before_read(self,txn,key):
        if txn.metadata.get('pipeline_fixed_writer'):
            self.item.observe_only(txn,key,write=False)
            return super().before_read(txn,key)
        return self.item.before_read(txn,key)

    def before_write(self,txn,key):
        if txn.metadata.get('pipeline_fixed_writer'):
            self.item.observe_only(txn,key,write=True)
            return super().before_write(txn,key)
        return self.item.before_write(txn,key)

    def before_commit(self,txn):
        if txn.metadata.get('pipeline_fixed_writer'):return super().before_commit(txn)
        return self.item.before_commit(txn)


class ObservedAccessHooks(FixedWriterHooks):
    """Shared experimental gate: first decision at issued access, not empty prefix."""
    def _select_and_apply(self,txn,state):
        if not txn.read_set and not txn.write_set and not txn.metadata.get('_pipeline_issued_access'):
            return None
        return super()._select_and_apply(txn,state)

    def before_read(self,txn,key):
        if not txn.metadata.get('pipeline_fixed_writer') and not txn.read_set and not txn.write_set:
            txn.metadata['_pipeline_issued_access']=str(key)
            try:self._select_and_apply(txn,self._state(txn))
            finally:txn.metadata.pop('_pipeline_issued_access',None)
        return super().before_read(txn,key)


def inputs(seed,n):
    rng=random.Random(seed)
    profiles=['refine','explore']*(n//2)
    rng.shuffle(profiles)
    return [dict(profile=p,config=dict(CONFIG,agent_writer=True,
        prefix_ms=rng.uniform(35.,45.),cheap_ms=rng.uniform(18.,22.),
        reason_ms=rng.uniform(280.,320.),dependent_tool_ms=rng.uniform(180.,220.))) for p in profiles]


def remove_pipeline_rounds(arm, plan):
    return arm == 'nophase' and not plan.get('label_only_nophase', False)


def pipeline_payload(actor, arm, generation, plan):
    if isinstance(actor, SharedBitMaskActor):
        return actor.joint_payload(generation=generation)
    return JointMaskTablePolicy.compile(actor, generation=generation,
        remove_phase_index_proxy=remove_pipeline_rounds(arm, plan)).to_dict()


def restore_training_actor(record):
    """Restore the parameterization used for gradients, not just its joint law."""
    if 'bit_before' in record:
        return SharedBitMaskActor.from_dict(record['bit_before'], seed=20261012)
    return actor_from_joint_payload(record['before'], seed=20261012)


def fresh_shared_pipeline_actor(arm):
    actor = SharedBitMaskActor(remove_phase=False, retain_explicit_phase=arm == 'full',
        residual_context='', seed=20261012)
    # Match the categorical pilot's uniform law over all legal complete masks.
    for row in actor.actor_weights:
        row[:] = [0.] * len(row)
    actor.actor_group_logits.clear()
    return actor


def run(core,runtime,policy,item,index,collect=False,threshold=None,observed_access=False,
        pre_expensive_validation=False,warmup_items=None,checkpoint_min_budget_ms=0.):
    gc.collect();m=make_manager(core,runtime,policy,collect)
    m.pre_expensive_validation_enabled=pre_expensive_validation
    m.policy_checkpoint_min_interval_ms=checkpoint_min_budget_ms
    warmup=[]
    if warmup_items:
        from agent.runtime.paper_policy import AtomicPolicyManager
        selected=m.paper_policy
        m.paper_policy=AtomicPolicyManager(Probe('hold_current'))
        m.interceptor.hooks=FixedWriterHooks(m)
        for j,warm in enumerate(warmup_items):
            warmup.append(execute_pair(m,None,warm['profile'],-j-1,warm['config']))
        m.paper_policy=selected
        if not m.is_hot('shift-key:0'):raise ValueError('Warmup failed to establish actual hot key')
    m.interceptor.hooks=(ObservedAccessHooks(m) if observed_access else FixedWriterHooks(m)) if threshold is None else LocalRuleHooks(m,threshold)
    r=execute_pair(m,policy if isinstance(policy,RecordingPolicy) else None,item['profile'],index,item['config'])
    r['pre_expensive_validation_enabled']=pre_expensive_validation
    r['checkpoint_min_budget_ms']=checkpoint_min_budget_ms
    if warmup:
        r['warmup_records']=warmup
        r['shared_key_hot_after_warmup']=True
    foreground={a['tid'] for a in r['tasks'][0]['attempts']}
    raw=[dataclasses.asdict(t) for t in m.trajectory_collector.snapshot() if t.txn_id in foreground] if collect else []
    if threshold is not None:
        r['item_decisions']=m.interceptor.hooks.item.decisions
    AuditedManager.instances.remove(m)
    if AuditedManager.instances:raise ValueError('Manager leak')
    return r,raw


def workflow_reward_rows(rows,episode_results):
    """End-to-end episodic return, including other-agent downstream work.

    All retries of a logical foreground task form one trajectory.  Only its
    last transition receives the measured negative pair makespan. No phase
    labels, chosen actions or handwritten target decisions enter the reward.
    """
    indices={}
    for i,row in enumerate(rows):
        episode=int(row.txn_id.split(':',1)[0])
        indices.setdefault(episode,[]).append(i)
    if set(indices)!=set(range(len(episode_results))):raise ValueError('Missing episode decisions')
    last={positions[-1]:episode for episode,positions in indices.items()}
    changed=[]
    for i,row in enumerate(rows):
        episode=int(row.txn_id.split(':',1)[0])
        terminal=i in last
        reward=-10.*episode_results[episode]['summary']['elapsed_seconds'] if terminal else 0.
        changed.append(dataclasses.replace(row,txn_id=f'workflow:{episode}',done=terminal,reward=reward))
    return changed,dict(objective='negative measured pair makespan / 100ms',
        linked_retry_episodes=len(indices),terminal_rewards=[r.reward for r in changed if r.done],
        reward_uses_phase=False,reward_uses_action=False,all_other_rewards_zero=True)


def parent_rewarded_rows(parent):
    plan=read(parent/'protocol.json');result=[]
    for arm in ('full','nophase'):
        for iteration in range(plan['updates_per_arm']):
            fit=read(parent/f'{arm}_update_{iteration}.json')
            rows=checked_rows(fit['transitions'],f'pipeline:{arm}:{iteration}')
            episodes=[read(parent/f'{arm}_train_{iteration}_{j}.json') for j in range(plan['episodes_per_update'])]
            rewarded,_=(workflow_reward_rows(rows,episodes) if plan.get('reward_objective')=='workflow-makespan'
                else apply_paper_rewards(rows,PaperRewardConfig(**plan['reward_config'])))
            result.extend(dataclasses.replace(t,source_id='parent:'+t.source_id) for t in rewarded)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observed-access-checkpoint',action='store_true')
    p.add_argument('--reward-objective',choices=('paper','workflow-makespan'),default='paper')
    p.add_argument('--training-episodes',type=int,choices=(16,32,64),default=16)
    p.add_argument('--episodes-per-update',type=int,choices=(8,16),default=8)
    p.add_argument('--training-seed',type=int,default=2026102101)
    p.add_argument('--test-seed',type=int,default=2026102102)
    p.add_argument('--test-episodes',type=int,choices=(8,16),default=8)
    p.add_argument('--read-timing-only',action='store_true',
        help='Shared {optimistic, read-protect} action ablation; removes equivalent unused write masks')
    p.add_argument('--curriculum-parent',type=Path,
        help='Paired learned parent as a frozen probability prior; child trains all 16 actions')
    p.add_argument('--prior-uniform-mix',type=float,default=.2)
    p.add_argument('--phase-feature-group-ablation',action='store_true',
        help='NoPhase removes the complete declared paper phase-feature group, not just labels/rounds')
    p.add_argument('--update-rate-scale',type=float,default=1.,
        help='Common actor/group/critic step multiplier for conservative full-space continuation')
    p.add_argument('--no-explicit-phase-label',action='store_true',
        help='Full retains paper phase-related features but receives no explicit phase label')
    p.add_argument('--label-only-nophase',action='store_true',
        help='NoLabel removes only phase, preserving rounds/progress in sampling, PPO and deployment')
    p.add_argument('--pre-expensive-validation',action='store_true',
        help='Give all training and test policies the same pre-call observed-version check')
    p.add_argument('--warmup-pairs',type=int,choices=(0,2),default=0,
        help='Common actual optimistic warmup before each training/test pair, excluded from reward')
    p.add_argument('--warmup-seed',type=int,default=2026102221)
    p.add_argument('--checkpoint-min-budget-ms',type=float,default=0.,
        help='Common caller-budget gate for training and deployment; prefix validation is retained')
    p.add_argument('--advantage-baseline',choices=('linear-critic','deployment-group'),default='linear-critic',
        help='Common state-only advantage baseline for both learned arms; defaults preserve old training')
    p.add_argument('--shared-bit-actor',action='store_true',
        help='Paired label-only experiment: four shared bit parameters, complete 16-mask law, uniform prior')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.shared_bit_actor and (not args.label_only_nophase or args.read_timing_only
            or args.curriculum_parent or args.advantage_baseline != 'linear-critic'):
        p.error('Shared bits require label-only, full masks, no curriculum and linear critic baseline')
    if args.label_only_nophase and (args.phase_feature_group_ablation or args.no_explicit_phase_label or args.curriculum_parent):
        p.error('Label-only requires explicit-phase Full, no group removal or inherited curriculum')
    if args.curriculum_parent and args.read_timing_only:p.error('Curriculum stage must restore full action space')
    if not math.isfinite(args.update_rate_scale) or args.update_rate_scale<=0:
        p.error('update-rate-scale must be positive and finite')
    if not math.isfinite(args.checkpoint_min_budget_ms) or args.checkpoint_min_budget_ms<0:
        p.error('checkpoint-min-budget-ms must be finite and nonnegative')
    schema=read(args.schema/'protocol.json');runtime=schema['runtime']
    cfg=PPOConfig(min_group_samples=4,initialize_from_behavior=False,
        learning_rate=.003*args.update_rate_scale,group_learning_rate=.03*args.update_rate_scale,
        critic_learning_rate=.005*args.update_rate_scale,advantage_baseline=args.advantage_baseline,
        conditioned_mask_gradient=args.shared_bit_actor)
    reward_cfg=PaperRewardConfig(**schema['reward_config'])
    plan=dict(scope='Constructed pipeline mechanism pilot, not YCSB or measured real LLM trace',
        actor_initialization='zero actor logits/weights/critic; no inherited learned support',
        training_input=inputs(args.training_seed,args.training_episodes),test_input=inputs(args.test_seed,args.test_episodes),
        training_seed=args.training_seed,test_seed=args.test_seed,
        updates_per_arm=args.training_episodes//args.episodes_per_update,episodes_per_update=args.episodes_per_update,
        runtime=runtime,config=dataclasses.asdict(cfg),reward_config=dataclasses.asdict(reward_cfg),
        reward_changed=args.reward_objective!='paper',reward_objective=args.reward_objective,
        state_projection=TABLE_STATE_PROJECTION,new_api_calls=0,
        observed_access_checkpoint=args.observed_access_checkpoint,
        phase_feature_group_ablation=args.phase_feature_group_ablation,
        no_explicit_phase_label=args.no_explicit_phase_label,
        label_only_nophase=args.label_only_nophase,
        shared_bit_actor=args.shared_bit_actor,
        pre_expensive_validation=args.pre_expensive_validation,
        checkpoint_min_budget_ms=args.checkpoint_min_budget_ms,
        warmup_input=inputs(args.warmup_seed,args.warmup_pairs) if args.warmup_pairs else [],
        warmup_seed=args.warmup_seed,
        removed_nophase_fields=['phase'] if args.label_only_nophase else
            ['phase',*PHASE_FEATURE_FIELDS] if args.phase_feature_group_ablation else ['phase','completed_rounds'],
        allowed_masks=[0,3] if args.read_timing_only else None,
        rule='Per-item 1000ms sliding access count including fixed competitor accesses, threshold 1 (always) or 3; reset each independent pair; no global risk',
        test_tuning=False,deployment='Frozen coefficient actor; pure-table compilation not part of this pilot',
        sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'scripts/revision/probe_phase_pipeline.py':sha(ROOT/'scripts/revision/probe_phase_pipeline.py'),
            'agent/runtime/item_frequency_hooks.py':sha(ROOT/'agent/runtime/item_frequency_hooks.py')})
    plan['sources']['agent/runtime/joint_mask_policy.py']=sha(ROOT/'agent/runtime/joint_mask_policy.py')
    plan['sources']['agent/runtime/shared_bit_mask_policy.py']=sha(ROOT/'agent/runtime/shared_bit_mask_policy.py')
    plan['sources']['agent/runtime/operation_interceptor.py']=sha(ROOT/'agent/runtime/operation_interceptor.py')
    if args.curriculum_parent:
        from scripts.revision.audit_phase_pipeline_pilot import audit as audit_parent
        audit_parent(args.curriculum_parent)
        plan.update(curriculum_parent=str(args.curriculum_parent.resolve()),prior_uniform_mix=args.prior_uniform_mix,
            actor_initialization='zero residual actor/critic atop own frozen learned parent; full 16-action continuation, not from scratch',
            parent_artifact_sha256={name:sha(args.curriculum_parent/name) for name in
                ('protocol.json','full_policy.json','nophase_policy.json','linear_policy.json')})
        for arm in ('full','nophase'):
            (args.output/f'parent_{arm}_policy.json').write_bytes((args.curriculum_parent/f'{arm}_policy.json').read_bytes())
        (args.output/'parent_linear_policy.json').write_bytes((args.curriculum_parent/'linear_policy.json').read_bytes())
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    core=load_cast_core();rewarded_all=parent_rewarded_rows(args.curriculum_parent) if args.curriculum_parent else []
    models={};timing={};proof=[]
    for arm in ('full','nophase'):
        actor=fresh_actor(JointMaskTablePolicy.from_dict(read(args.schema/f'{arm}_policy.json')),arm)
        if args.shared_bit_actor:
            if actor.growth_conditioned_groups or not actor.local_context_groups:
                raise ValueError('Shared-bit pilot requires matching unconditioned local context')
            actor=fresh_shared_pipeline_actor(arm)
        actor.remove_phase_index_proxy=remove_pipeline_rounds(arm,plan)
        actor.state_projection=TABLE_STATE_PROJECTION
        if args.no_explicit_phase_label:actor.phase_awareness='disabled'
        actor.remove_phase_features=bool(args.phase_feature_group_ablation and arm=='nophase')
        actor.allowed_masks=tuple(plan['allowed_masks']) if plan['allowed_masks'] is not None else None
        if args.curriculum_parent:
            initial=payload(actor,arm,2026102099)
            initial.update(prior_policy=read(args.output/f'parent_{arm}_policy.json'),prior_uniform_mix=args.prior_uniform_mix)
            actor=actor_from_joint_payload(initial,seed=20261012)
        sampler=OnPolicySampler(actor,remove_pipeline_rounds(arm,plan));tick=time.perf_counter()
        for update_index in range(plan['updates_per_arm']):
            before=pipeline_payload(actor,arm,2026102100+update_index,plan);critic_before=list(actor.critic_weights);raw=[]
            bit_before=actor.to_dict() if args.shared_bit_actor else None
            source=f'pipeline:{arm}:{update_index}';episode_results=[]
            size=plan['episodes_per_update']
            for j,item in enumerate(plan['training_input'][size*update_index:size*(update_index+1)]):
                r,transitions=run(core,runtime,sampler,item,j,True,observed_access=args.observed_access_checkpoint,
                    pre_expensive_validation=args.pre_expensive_validation,warmup_items=plan['warmup_input'],
                    checkpoint_min_budget_ms=args.checkpoint_min_budget_ms)
                for t in transitions:t.update(txn_id=f'{j}:'+t['txn_id'],source_id=source)
                raw.extend(transitions)
                episode_results.append(r)
                (args.output/f'{arm}_train_{update_index}_{j}.json').write_text(json.dumps(r,indent=2))
            rows=checked_rows(raw,source);propensity=verify_onpolicy(actor,rows,remove_pipeline_rounds(arm,plan))
            rewarded,reward=(workflow_reward_rows(rows,episode_results) if args.reward_objective=='workflow-makespan'
                else apply_paper_rewards(rows,reward_cfg))
            rewarded_all.extend(rewarded)
            fit=DiscretePPOTrainer(cfg).train(actor,actor_training_rows(actor,rewarded,remove_pipeline_rounds(arm,plan)))
            after=pipeline_payload(actor,arm,2026102101+update_index,plan)
            record=dict(before=before,critic_before=critic_before,transitions=raw,propensity=propensity,
                reward=reward,fit=fit,after=after,critic_after=list(actor.critic_weights))
            if args.shared_bit_actor:
                record.update(bit_before=bit_before,bit_after=actor.to_dict())
            if args.curriculum_parent:
                from agent.cc.atcc.ppo import valid_actions
                record['expanded_space']=dict(
                    all_legal_actions_positive=all(all(t.behavior_action_probabilities[a]>0 for a in valid_actions(t.state.current_action)) for t in rows),
                    sampled_masks=sorted({t.action for t in rows}),sampled_outside_parent_pair=sum(t.action not in (0,3) for t in rows))
            (args.output/f'{arm}_update_{update_index}.json').write_text(json.dumps(record,indent=2))
            # Independent numerical reconstruction, including critic and probabilities.
            clone=restore_training_actor(record);clone.critic_weights=critic_before[:]
            if verify_onpolicy(clone,rows,remove_pipeline_rounds(arm,plan))!=propensity:raise ValueError('Propensity mismatch')
            replay=DiscretePPOTrainer(cfg).train(clone,actor_training_rows(clone,rewarded,remove_pipeline_rounds(arm,plan)))
            if replay!=fit or pipeline_payload(clone,arm,2026102101+update_index,plan)!=after or clone.critic_weights!=actor.critic_weights:
                raise ValueError('PPO numerical replay mismatch')
            if args.shared_bit_actor and clone.to_dict()!=record['bit_after']:
                raise ValueError('Shared-bit parameter replay mismatch')
            proof.append(dict(arm=arm,update=update_index,rows=len(rows),numerical_replay=True))
            print(json.dumps(dict(stage='trained',**proof[-1])),flush=True)
        timing[arm+'_collect_update_and_replay_seconds']=time.perf_counter()-tick
        (args.output/f'{arm}_policy.json').write_text(json.dumps(after,indent=2))
        models[arm]=JointMaskTablePolicy.from_dict(after)
    tick=time.perf_counter();linear=fit_linear_rewarded(rewarded_all)
    timing['linear_fit_seconds']=time.perf_counter()-tick
    (args.output/'linear_policy.json').write_text(json.dumps(linear,indent=2))
    models['linear']=LinearScorePolicy(linear['weights'])
    models['hot_always']=models['full'];models['hot_threshold']=models['full']
    (args.output/'timing.json').write_text(json.dumps(timing,indent=2))
    arms=list(models);results=[]
    for i,item in enumerate(plan['test_input']):
        for arm in arms[i%5:]+arms[:i%5]:
            rec=RecordingPolicy(models[arm]);threshold={'hot_always':1,'hot_threshold':3}.get(arm)
            r,_=run(core,runtime,rec,item,i,threshold=threshold,observed_access=args.observed_access_checkpoint,
                pre_expensive_validation=args.pre_expensive_validation,warmup_items=plan['warmup_input'],
                checkpoint_min_budget_ms=args.checkpoint_min_budget_ms)
            r['summary'].update(arm=arm,episode=i)
            for d in r['decisions']:
                from agent.runtime.state_collector import phase_aware_state_from_dict
                if models[arm].select(phase_aware_state_from_dict(d['state'])).encoded!=d['action']:
                    raise ValueError('Frozen action differs')
            (args.output/f'{arm}_test_{i}.json').write_text(json.dumps(r,indent=2))
            results.append(r['summary']);print(json.dumps(dict(stage='tested',**r['summary'])),flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    aggregate=[]
    for arm in arms:
        rows=[r for r in results if r['arm']==arm];elapsed=sum(r['elapsed_seconds'] for r in rows)
        aggregate.append(dict(arm=arm,logical_tasks=2*len(rows),elapsed_seconds=elapsed,
            task_tps=2*len(rows)/elapsed,mean_logical_ms=sum(r['mean_logical_ms'] for r in rows)/len(rows),
            aborts=sum(r['aborts'] for r in rows),aborted_gap_ms=sum(r['aborted_gap_ms'] for r in rows),
            policy_calls=sum(r['policy_calls'] for r in rows),action_changes=sum(r['action_changes'] for r in rows)))
        aggregate[-1]['item_protection_changes']=sum(r.get('item_protection_changes',0) for r in rows)
    (args.output/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    with (args.output/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    (args.output/'training_audit.json').write_text(json.dumps(dict(passed=True,updates=proof,
        histories_verified_during_execution=True,test_actions_replayed=True,
        caveat='Small pilot; no stable P99 inference; timing includes audit replay separately labelled'),indent=2))
    print(json.dumps(dict(stage='complete',aggregate=aggregate)),flush=True)


if __name__=='__main__':main()
