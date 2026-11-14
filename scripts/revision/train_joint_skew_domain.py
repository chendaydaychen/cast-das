"""One small paired from-zero PPO update on a declared target-domain skew input."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.native import load_cast_core
from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig,FEATURE_COUNT
from agent.runtime import PaperRewardConfig,apply_paper_rewards
from agent.runtime.joint_mask_policy import JointMaskActor,JointMaskTablePolicy,actor_from_joint_payload
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.run_joint_skew_transfer import make_skew_cohorts
from scripts.revision.run_frozen_ycsb_shift import run_cohort
from scripts.revision.run_audited_credit_pressure import AuditedManager,fit_linear
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.train_ycsb_onpolicy import OnPolicySampler,training_rows,verify_onpolicy
from scripts.revision.train_ycsb_mixed_prefix import checked_rows
from scripts.revision.audit_serial_history import verify_history

TRAIN_SEED=2026101703
TEST_SEED=2026101704


def fresh_actor(template,arm):
    actor=JointMaskActor(seed=20261012,phase_awareness='enabled' if arm=='full' else 'disabled',
        frozen_support={},remove_phase_index_proxy=arm=='nophase',
        growth_conditioned_groups=template.growth_conditioned_groups,
        local_context_groups=template.local_context_groups)
    actor.categorical_actor_weights=[[0.]*FEATURE_COUNT for _ in range(16)]
    actor.categorical_group_logits.clear()
    return actor


def audit(folder):
    plan=read(folder/'protocol.json')
    if plan['training_seed']!=TRAIN_SEED or plan['heldout_seed']!=TEST_SEED:
        raise ValueError('Declared split changed')
    if read(folder/'training_cohorts.json')!=make_skew_cohorts(seed=TRAIN_SEED):
        raise ValueError('Training input changed')
    for name,digest in plan['runtime_source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Training source changed: '+name)
    proofs=[]
    combined=[]
    for arm in ('full','nophase'):
        fit=read(folder/f'{arm}_fit.json')
        actor=actor_from_joint_payload(fit['before_actor'],seed=20261012)
        raw=read(folder/f'{arm}_rollout.json')['transitions']
        combined.extend(raw)
        for epoch in range(2):
            record=read(folder/f'{arm}_epoch{epoch}.json')
            if verify_history(**record['history'])!=record['oracle'] or record['oracle']['verdict']!='verified':
                raise ValueError('Invalid training history')
        rows=checked_rows(raw,'target-skew:'+arm)
        propensity=verify_onpolicy(actor,rows,arm=='nophase')
        rewarded,reward=apply_paper_rewards(rows,PaperRewardConfig(**fit['reward_config']))
        update=DiscretePPOTrainer(PPOConfig(**fit['config'])).train(actor,training_rows(rewarded,arm=='nophase'))
        after=JointMaskTablePolicy.compile(actor,generation=2026101704,remove_phase_index_proxy=arm=='nophase').to_dict()
        if (propensity!=fit['propensity'] or reward!=fit['reward'] or update!=fit['fit_report']
                or after!=fit['after_actor'] or after!=read(folder/f'{arm}_policy.json')
                or actor.critic_weights!=fit['critic_after']):
            raise ValueError('Target-domain PPO numerical replay differs')
        proofs.append(dict(arm=arm,transitions=len(rows),gradient_reproduced=True,
            changed_model=fit['before_actor']!=after,
            updates=sum(e['updates'] for e in update.get('epochs',[]))))
    if combined!=read(folder/'training_transitions.json')['transitions']:
        raise ValueError('Linear training batch differs')
    reproduced=fit_linear(folder/'training_transitions.json',folder/'training_report.json',prior_batches=())
    if reproduced!=read(folder/'linear_policy.json'):
        raise ValueError('Same-information Linear fit differs')
    result=dict(passed=True,arms=proofs,linear_reproduced=True,
        policy_sha256={a:sha(folder/f'{a}_policy.json') for a in ('full','nophase','linear')})
    (folder/'training_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',type=Path)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.audit_only:
        audit(args.output);return
    if args.schema is None or args.calibration is None:
        p.error('Schema and measured calibration required')
    original=read(args.schema/'protocol.json')
    calibration=read(args.calibration)['calibration']
    delays=[calibration['delay_ms'][k] for k in ('explore','refine','commit')]
    cfg=PPOConfig(min_group_samples=4,initialize_from_behavior=False)
    reward_cfg=PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
        terminal_cost_scaling=original['terminal_cost_scaling'])
    cohorts=make_skew_cohorts(seed=TRAIN_SEED)
    args.output.mkdir(parents=True,exist_ok=False)
    dependencies=('scripts/revision/train_joint_skew_domain.py','scripts/revision/run_joint_skew_transfer.py',
        'scripts/revision/run_frozen_ycsb_shift.py','scripts/revision/train_ycsb_onpolicy.py',
        'scripts/revision/train_ycsb_mixed_prefix.py','scripts/revision/run_audited_credit_pressure.py',
        'scripts/revision/run_agent_schedule_comparison_20261012.py','agent/cc/atcc/ppo.py',
        'agent/runtime/joint_mask_policy.py','agent/runtime/transaction.py','agent/runtime/paper_hooks.py',
        'agent/runtime/operation_interceptor.py','agent/runtime/priority.py','agent/runtime/trajectory.py')
    plan=dict(original,training_seed=TRAIN_SEED,heldout_seed=TEST_SEED,training_workers=4,
        actor_seed=20261012,training_tasks_per_arm=8,one_update=True,fresh_initialization=True,
        calibration_sha256=sha(args.calibration),delays_ms=delays,config=dataclasses.asdict(cfg),
        reward_config=dataclasses.asdict(reward_cfg),
        runtime_source_sha256={n:sha(ROOT/n) for n in dependencies},
        schema_only_parent=str(args.schema.resolve()),
        training_scope='Zero scores, uniform legal monotone masks, no inherited learned weights/support; one own-actor two-cohort rollout',
        evaluation_performed=False,new_api_calls=0)
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'training_cohorts.json').write_text(json.dumps(cohorts,indent=2))
    (args.output/'calibration.json').write_bytes(args.calibration.read_bytes())
    core=load_cast_core()
    all_raw,summaries=[],[]
    started=time.perf_counter()
    for arm in ('full','nophase'):
        template=JointMaskTablePolicy.from_dict(read(args.schema/f'{arm}_policy.json'))
        actor=fresh_actor(template,arm)
        before=JointMaskTablePolicy.compile(actor,generation=2026101703,remove_phase_index_proxy=arm=='nophase').to_dict()
        sampler=OnPolicySampler(actor,arm=='nophase')
        manager=AuditedManager(storage_engines={'default':core.Dbx1000VersionedKVStore(),
            'audit':core.Dbx1000VersionedKVStore()},paper_policy=sampler,record_traces=False,
            collect_trajectories=True,performance_guards_enabled=False,low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True,priority_enabled=True,
            policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
        configure_frozen_runtime(manager,original)
        manager.expensive_operation_checkpoint_enabled=original['before_reasoning_checkpoint']
        for key in range(128):
            manager.register_object(f'shift-key:{key}','0',kind='row',storage_id='audit' if key%2 else 'default')
        for cohort in cohorts:
            record=run_cohort(manager,None,cohort,system='paper-atcc',delays_ms=delays,
                max_attempts=6,carry_cost=original.get('carry_prior_restart_cost',False))
            (args.output/f'{arm}_epoch{cohort["epoch"]}.json').write_text(json.dumps(record,indent=2))
            if record['oracle']['verdict']!='verified':
                raise ValueError('Training committed history failed')
            print(json.dumps(dict(arm=arm,**record['summary'])),flush=True)
        raw=[dataclasses.asdict(t) for t in manager.trajectory_collector.snapshot()]
        for row in raw:
            row['source_id']='target-skew:'+arm
            row['txn_id']='target-skew:'+arm+':'+row['txn_id']
        (args.output/f'{arm}_rollout.json').write_text(json.dumps(dict(transitions=raw),indent=2))
        rows=checked_rows(raw,'target-skew:'+arm)
        propensity=verify_onpolicy(actor,rows,arm=='nophase')
        rewarded,reward=apply_paper_rewards(rows,reward_cfg)
        fit=DiscretePPOTrainer(cfg).train(actor,training_rows(rewarded,arm=='nophase'))
        after=JointMaskTablePolicy.compile(actor,generation=2026101704,remove_phase_index_proxy=arm=='nophase').to_dict()
        result=dict(before_actor=before,config=dataclasses.asdict(cfg),propensity=propensity,
            reward_config=dataclasses.asdict(reward_cfg),reward=reward,fit_report=fit,
            after_actor=after,critic_after=list(actor.critic_weights))
        (args.output/f'{arm}_fit.json').write_text(json.dumps(result,indent=2))
        (args.output/f'{arm}_policy.json').write_text(json.dumps(after,indent=2))
        all_raw.extend(raw)
        summaries.append(dict(arm=arm,transitions=len(rows),fit=fit))
    (args.output/'training_transitions.json').write_text(json.dumps(dict(transitions=all_raw),indent=2))
    _,reward=apply_paper_rewards(checked_rows(all_raw,'target-skew-combined'),reward_cfg)
    report=dict(arms=summaries,reward=reward,collect_fit_seconds=time.perf_counter()-started)
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2))
    linear=fit_linear(args.output/'training_transitions.json',args.output/'training_report.json',prior_batches=())
    (args.output/'linear_policy.json').write_text(json.dumps(linear,indent=2))
    report['total_seconds']=time.perf_counter()-started
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status='trained-not-evaluated',seconds=report['total_seconds'])),flush=True)


if __name__=='__main__':
    main()
