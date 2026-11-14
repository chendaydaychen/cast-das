"""Four matched own-actor PPO updates; freeze before new-input evaluation."""
import argparse
import csv
import dataclasses
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.native import load_cast_core
from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig
from agent.runtime import PaperRewardConfig,apply_paper_rewards
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,actor_from_joint_payload
from agent.runtime.paper_policy import HotRulePolicy,CompiledPhasePolicy
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.state_collector import phase_aware_state_from_dict
from agent.runtime.table_observation import TABLE_STATE_PROJECTION
from scripts.revision.run_heterogeneous_phase_pilot import make_tasks,execute
from scripts.revision.run_frozen_ycsb_shift import summarize
from scripts.revision.train_joint_skew_domain import fresh_actor
from scripts.revision.train_ycsb_onpolicy import OnPolicySampler,training_rows,verify_onpolicy
from scripts.revision.train_ycsb_mixed_prefix import checked_rows
from scripts.revision.run_audited_credit_pressure import AuditedManager,LinearScorePolicy,fit_linear_rewarded
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.evaluate_target_skew_domain import compile_tables
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.audit_serial_history import verify_history

TRAIN_SEEDS=tuple(range(2026101720,2026101728))
TEST_SEED=2026101728


class SwapPhasePolicy:
    """Frozen-model label intervention; not a separately trained NoPhase."""
    def __init__(self,policy):self.policy=policy
    def select(self,state):
        return self.policy.select(dataclasses.replace(state,
            phase={'explore':'refine','refine':'explore'}.get(state.phase,state.phase)))


def make_manager(core,original,policy,collect=False):
    m=AuditedManager(storage_engines={'default':core.Dbx1000VersionedKVStore(),'audit':core.Dbx1000VersionedKVStore()},
        paper_policy=policy,record_traces=False,collect_trajectories=collect,
        performance_guards_enabled=False,low_conflict_occ_guard=False,
        delayed_write_apply_enabled=True,priority_enabled=True,
        policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
    configure_frozen_runtime(m,original);m.expensive_operation_checkpoint_enabled=True
    for key in range(32):
        m.register_object(f'shift-key:{key}','0',kind='row',storage_id='audit' if key%2 else 'default')
    return m


def payload(actor,arm,generation):
    return JointMaskTablePolicy.compile(actor,generation=generation,remove_phase_index_proxy=arm=='nophase').to_dict()


def actor_training_rows(actor, rows, remove_phase):
    rows=training_rows(rows,remove_phase)
    if not actor.state_projection and not getattr(actor,'remove_phase_features',False):
        return rows
    return [dataclasses.replace(t,state=actor.observation(t.state),
        next_state=actor.observation(t.next_state) if t.next_state else None) for t in rows]


def audit(output):
    plan=read(output/'protocol.json');models={};all_rewarded=[]
    if sha(output/'runner_executed.py')!=plan['source_sha256'][str(Path(__file__).relative_to(ROOT))]:
        raise ValueError('Saved executed runner differs from declaration')
    for arm in ('full','nophase'):
        prior=None;prior_critic=None
        for iteration in range(plan['updates_per_arm']):
            fit=read(output/f'{arm}_update_{iteration}.json')
            if prior and fit['before']!=prior:raise ValueError('Actor chain differs')
            if prior_critic is not None and fit['critic_before']!=prior_critic:raise ValueError('Critic chain differs')
            if iteration==0 and (any(v for row in fit['before']['weights'] for v in row)
                    or fit['before']['groups'] or fit['before']['support'] or any(fit['critic_before'])):
                raise ValueError('Initialization was not from zero')
            actor=actor_from_joint_payload(fit['before'],seed=20261012)
            actor.critic_weights=list(fit['critic_before'])
            rows=checked_rows(fit['transitions'],f'heterogeneous-updates:{arm}:{iteration}')
            if verify_onpolicy(actor,rows,arm=='nophase')!=fit['propensity']:raise ValueError('Propensities differ')
            rewarded,reward=apply_paper_rewards(rows,PaperRewardConfig(**plan['reward_config']))
            all_rewarded.extend(rewarded)
            update=DiscretePPOTrainer(PPOConfig(**plan['config'])).train(actor,actor_training_rows(actor,rewarded,arm=='nophase'))
            if (reward!=fit['reward'] or update!=fit['update'] or list(actor.critic_weights)!=fit['critic_after']
                    or payload(actor,arm,2026101720+iteration)!=fit['after']):
                raise ValueError('PPO chain numerical replay differs')
            prior=fit['after']
            prior_critic=fit['critic_after']
        if prior!=read(output/f'{arm}_policy.json'):raise ValueError('Final model differs')
        models[arm]=JointMaskTablePolicy.from_dict(prior)
    models['full_swapped']=SwapPhasePolicy(models['full'])
    if fit_linear_rewarded(all_rewarded)!=read(output/'linear_policy.json'):
        raise ValueError('Linear numerical fit differs')
    compiled={arm:load_pure_table(output/'compilation/audit.json',output,arm,4)[0] for arm in ('full','nophase')}
    models['full_table']=compiled['full'];models['nophase_table']=compiled['nophase']
    checks=[]
    aggregate=read(output/'results.json')
    if [r['arm'] for r in aggregate]!=plan['arms']:raise ValueError('Incomplete declared matrix')
    for path in sorted(output.glob('*_train_*.json')):
        r=read(path)
        if verify_history(**r['history'])!=r['oracle'] or r['oracle']['verdict']!='verified':raise ValueError('Invalid training history')
        if plan.get('retire_training_managers',False) and r.get('manager_registry')!={'before':0,'after':0}:
            raise ValueError('Training manager lifecycle differs')
    for arm in plan['arms']:
        r=read(output/f'{arm}_test.json');matches=0
        if r['summary']!=next(row for row in aggregate if row['arm']==arm):raise ValueError('Aggregate mismatch')
        actor_arm=arm.removesuffix('_table')
        if verify_history(**r['history'])!=r['oracle'] or r['oracle']['verdict']!='verified':raise ValueError('Invalid test history')
        expected=summarize(r['tasks'],r['summary']['workload_seconds'])
        if any(r['summary'][k]!=v for k,v in expected.items()):raise ValueError('Summary differs')
        for d in r['decisions']:
            s=phase_aware_state_from_dict(d['state'])
            if arm in models and models[arm].select(s).encoded!=d['action']:raise ValueError('Executed action mismatch')
            if actor_arm in compiled:
                matches+=compiled[actor_arm].select(s).encoded==models[actor_arm].select(s).encoded
            if d['action']&s.current_action!=s.current_action:raise ValueError('Released protection')
        checks.append(dict(arm=arm,queries=len(r['decisions']),actor_table_matches=matches if actor_arm in compiled else None,
            oracle='verified',commits=r['summary']['commits']))
    result=dict(passed=True,updates_reproduced=2*plan['updates_per_arm'],records=checks,
        linear_fit_reproduced=True,source_snapshot_verified=True,initialization_and_critic_chain_verified=True,
        training_test_disjoint=plan['test_seed'] not in plan['training_seeds'],
        protocol_sha256=sha(output/'protocol.json'),
        caveat='One four-task heldout batch; label intervention is not trained NoPhase, table performance separately measured')
    (output/'independent_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--batches-per-update',type=int,choices=(2,4,8),default=2,
        help='Same 32-task budget; optionally pool four cohorts before each update')
    p.add_argument('--training-cohorts',type=int,choices=(8,16),default=8,
        help='16 cohorts is a separately declared 64-task coverage experiment')
    p.add_argument('--table-state-projection',action='store_true',
        help='Fresh actors and pure tables share policy-input buckets, including restart-cost context')
    p.add_argument('--retire-training-managers',action='store_true',
        help='Release completed training managers for both arms before the next batch')
    args=p.parse_args()
    if args.audit_only:audit(args.output);return
    args.output.mkdir(parents=True,exist_ok=False)
    original=read(args.schema/'protocol.json')
    cfg=PPOConfig(min_group_samples=4,initialize_from_behavior=False)
    reward_cfg=PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
        terminal_cost_scaling=original['terminal_cost_scaling'])
    arms=['full','nophase','linear','hot_always','item_threshold3','full_swapped','full_table','nophase_table']
    train_seeds=tuple(range(2026101730,2026101746)) if args.training_cohorts==16 else TRAIN_SEEDS
    test_seed=2026101746 if args.training_cohorts==16 else (2026101729 if args.batches_per_update==4 else TEST_SEED)
    updates=len(train_seeds)//args.batches_per_update
    plan=dict(training_seeds=train_seeds,test_seed=test_seed,arms=arms,
        training_tasks_per_arm=4*len(train_seeds),updates_per_arm=updates,tasks_per_update=4*args.batches_per_update,
        batches_per_update=args.batches_per_update,
        config=dataclasses.asdict(cfg),reward_config=dataclasses.asdict(reward_cfg),runtime=original,
        scope='Four-worker synthetic heterogeneous workflow; 280ms prescribed total wait, not measured LLM calibration',
        initialization='zero logits; no inherited support/weights',no_test_tuning=True,carry_prior_retry_cost=False,
        state_projection=TABLE_STATE_PROJECTION if args.table_state_projection else '',
        retire_training_managers=args.retire_training_managers,
        input=[make_tasks(s,True) for s in train_seeds],test_input=make_tasks(test_seed,True),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'scripts/revision/run_heterogeneous_phase_pilot.py':sha(ROOT/'scripts/revision/run_heterogeneous_phase_pilot.py')})
    if args.table_state_projection:
        plan['projection_source_sha256']={name:sha(ROOT/name) for name in (
            'agent/runtime/table_observation.py','agent/runtime/joint_mask_policy.py',
            'agent/runtime/paper_policy.py','agent/cc/atcc/mask_closure.py')}
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    core=load_cast_core();all_rewarded=[];all_raw=[];timing={}
    for arm in ('full','nophase'):
        actor=fresh_actor(JointMaskTablePolicy.from_dict(read(args.schema/f'{arm}_policy.json')),arm)
        actor.state_projection=plan['state_projection']
        sampler=OnPolicySampler(actor,arm=='nophase')
        started=time.perf_counter()
        for iteration in range(updates):
            # Generation identifies the version, not a learned coefficient.
            before=payload(actor,arm,2026101719+iteration);critic_before=list(actor.critic_weights)
            raw=[]
            for j in range(args.batches_per_update):
                if args.retire_training_managers:gc.collect()
                registry_before=len(AuditedManager.instances)
                batch=args.batches_per_update*iteration+j;m=make_manager(core,original,sampler,True)
                r=execute(m,None,plan['input'][batch],batch)
                transitions=[dataclasses.asdict(t) for t in m.trajectory_collector.snapshot()]
                for row in transitions:
                    row['txn_id']=f'{arm}:{batch}:'+row['txn_id'];row['source_id']=f'heterogeneous-updates:{arm}:{iteration}'
                raw.extend(transitions)
                if args.retire_training_managers:
                    AuditedManager.instances.remove(m)
                    if AuditedManager.instances:
                        raise ValueError('Previous training manager retained')
                    m=None
                r['manager_registry']=dict(before=registry_before,after=len(AuditedManager.instances))
                (args.output/f'{arm}_train_{batch}.json').write_text(json.dumps(r,indent=2))
            rows=checked_rows(raw,f'heterogeneous-updates:{arm}:{iteration}')
            propensity=verify_onpolicy(actor,rows,arm=='nophase')
            rewarded,reward=apply_paper_rewards(rows,reward_cfg)
            update=DiscretePPOTrainer(cfg).train(actor,actor_training_rows(actor,rewarded,arm=='nophase'))
            after=payload(actor,arm,2026101720+iteration)
            record=dict(before=before,critic_before=critic_before,transitions=raw,propensity=propensity,
                reward=reward,update=update,after=after,critic_after=list(actor.critic_weights))
            (args.output/f'{arm}_update_{iteration}.json').write_text(json.dumps(record,indent=2))
            all_rewarded.extend(rewarded);all_raw.extend(raw)
            print(json.dumps(dict(stage='trained',arm=arm,iteration=iteration,rows=len(raw),
                groups=update['group_residual_training'])),flush=True)
        timing[arm+'_collect_update_seconds']=time.perf_counter()-started
        (args.output/f'{arm}_policy.json').write_text(json.dumps(after,indent=2))
    started=time.perf_counter();linear=fit_linear_rewarded(all_rewarded)
    timing['linear_fit_seconds']=time.perf_counter()-started
    (args.output/'linear_policy.json').write_text(json.dumps(linear,indent=2))
    (args.output/'training_transitions.json').write_text(json.dumps(dict(transitions=all_raw),indent=2))
    started=time.perf_counter();compile_tables(args.output,args.output/'compilation')
    timing['pure_table_compile_seconds']=time.perf_counter()-started
    timing['total_collect_train_fit_compile_seconds']=sum(timing.values())
    (args.output/'timing.json').write_text(json.dumps(timing,indent=2))
    policies={a:JointMaskTablePolicy.from_dict(read(args.output/f'{a}_policy.json')) for a in ('full','nophase')}
    policies.update(linear=LinearScorePolicy(linear['weights']),hot_always=HotRulePolicy(),item_threshold3=CompiledPhasePolicy([]))
    policies['full_swapped']=SwapPhasePolicy(policies['full'])
    for a in ('full','nophase'):policies[a+'_table']=load_pure_table(args.output/'compilation/audit.json',args.output,a,4)[0]
    results=[]
    for arm in arms:
        recorded=RecordingPolicy(policies[arm]);m=make_manager(core,original,recorded)
        if arm=='hot_always':m.first_hot_access_events=[];m.interceptor.hooks=FirstHotAccessHooks(m)
        if arm=='item_threshold3':m.interceptor.hooks=ItemFrequencyHooks(m,minimum_accesses=3,window_ms=1000.)
        r=execute(m,recorded,plan['test_input'],8);r['summary']['arm']=arm
        if arm=='item_threshold3':r['item_decisions']=m.interceptor.hooks.decisions
        (args.output/f'{arm}_test.json').write_text(json.dumps(r,indent=2));results.append(r['summary'])
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(dict(stage='tested',**r['summary'])),flush=True)
    with (args.output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    audit(args.output)


if __name__=='__main__':main()
