"""Predeclared heterogeneous phase pilot; no changes to historical runners."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
from pathlib import Path
import random
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc import LockConflict
from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig
from agent.native import load_cast_core
from agent.runtime import PaperRewardConfig,apply_paper_rewards
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,actor_from_joint_payload
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.paper_policy import HotRulePolicy,CompiledPhasePolicy
from scripts.revision.train_joint_skew_domain import fresh_actor
from scripts.revision.train_ycsb_onpolicy import OnPolicySampler,training_rows,verify_onpolicy
from scripts.revision.train_ycsb_mixed_prefix import checked_rows
from scripts.revision.run_audited_credit_pressure import AuditedManager,LinearScorePolicy,fit_linear_rewarded
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy
from scripts.revision.run_frozen_ycsb_shift import summarize
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.audit_nophase_recoverability import features,fit,score
from scripts.unified_trace.run_credit_review_experiment import transaction_metadata

TRAIN_SEEDS=(2026101711,2026101712)
TEST_SEED=2026101713
ARMS=('full','nophase','linear','hot_always','item_threshold3')


def make_tasks(seed,aligned=False):
    rng=random.Random(seed)
    profiles=['early','late']*2
    rng.shuffle(profiles)
    tasks=[]
    for profile in profiles:
        remaining=list(range(32));ops=[]
        for i in range(10):
            key=rng.choices(remaining,weights=[(k+1)**-1.2 for k in remaining])[0]
            remaining.remove(key)
            ops.append(dict(key=key,kind='write' if i in (3,7) else 'read'))
        # Phase describes different application workflows, not future values
        # inferred by the CC. Both profiles have the same total prescribed wait.
        tasks.append(dict(profile=profile,operations=ops,
            phase_before=({'3':'refine','8':'commit'} if profile=='early' else
                ({'3':'explore','8':'refine'} if aligned else {'6':'refine','9':'commit'})),
            gap_ms={'4':240,'7':20,'10':20} if profile=='early' else {'4':20,'7':240,'10':20}))
    return tasks


def execute(manager,recorded,tasks,epoch):
    barrier=threading.Barrier(len(tasks))
    def worker(item):
        index,task=item
        barrier.wait(timeout=30)
        started=time.perf_counter();attempts=[]
        for retry in range(6):
            metadata=transaction_metadata('paper-atcc',retry_count=retry,policy_commit_batch=False)
            metadata.update(workload='heterogeneous_phase_pilot',task_type='ycsb')
            txn=manager.begin(f'hetero:{epoch}:{index}',metadata,snapshot_object_ids=None,strategy='paper-atcc')
            paid=0.;observed=[]
            try:
                txn.enter_phase('explore')
                for i,op in enumerate(task['operations'],1):
                    txn.check_active()
                    if str(i) in task['phase_before']:txn.enter_phase(task['phase_before'][str(i)])
                    key=f'shift-key:{op["key"]}'
                    if op['kind']=='read':txn.read(key)
                    else:txn.write(key,str(((epoch+1)<<40)|((index+1)<<16)|i))
                    observed.append(op)
                    if str(i) in task['gap_ms']:
                        txn.check_active()
                        manager.interceptor.expensive_operation_checkpoint(txn,1)
                        tick=time.perf_counter();time.sleep(task['gap_ms'][str(i)]/1000.)
                        paid+=(time.perf_counter()-tick)*1000.
                        txn.check_active()
                result=txn.commit('paper-atcc');committed=bool(result.committed);reason=str(result.reason)
            except LockConflict as exc:
                if txn.result is None:txn.abort(exc.reason,strategy='paper-atcc',conflict_object_ids=exc.targets)
                committed=False;reason=str(exc.reason)
            except BaseException:
                if txn.result is None:txn.abort('heterogeneous execution error',strategy='paper-atcc')
                raise
            attempts.append(dict(committed=committed,reason=reason,paid_gap_ms=paid,observed=observed,
                action_changes=sum(e.kind=='action-change' for e in txn.events),
                item_changes=sum(e.kind=='item-protection-change' for e in txn.events)))
            if committed:break
            if retry<5:time.sleep(.010)
        return dict(worker=index,profile=task['profile'],committed=committed,
            latency_ms=(time.perf_counter()-started)*1000.,attempts=attempts)
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(worker,enumerate(tasks)))
    elapsed=time.perf_counter()-started
    history=manager.history();oracle=verify_history(**history)
    if oracle['verdict']!='verified':raise ValueError('Committed history invalid')
    summary=summarize(outcomes,elapsed)
    summary.update(mean_submitted_task_ms=sum(t['latency_ms'] for t in outcomes)/len(outcomes),
        policy_calls=len(recorded.records) if recorded else 0)
    return dict(summary=summary,tasks=outcomes,history=history,oracle=oracle,
        decisions=recorded.records if recorded else [])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--aligned-boundaries',action='store_true',help='Both profiles query at ops 3/8, but report different semantic phases')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    original=read(args.schema/'protocol.json')
    cfg=PPOConfig(min_group_samples=4,initialize_from_behavior=False)
    reward_cfg=PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
        terminal_cost_scaling=original['terminal_cost_scaling'])
    train_seeds=(2026101714,2026101715,2026101716,2026101717) if args.aligned_boundaries else TRAIN_SEEDS
    test_seed=2026101718 if args.aligned_boundaries else TEST_SEED
    plan=dict(training_seeds=train_seeds,test_seed=test_seed,arms=ARMS,
        training_tasks_per_arm=4*len(train_seeds),workers=4,keys=32,zipf=1.2,max_attempts=6,
        training_inputs=[make_tasks(s,args.aligned_boundaries) for s in train_seeds],
        test_input=make_tasks(test_seed,args.aligned_boundaries),aligned_boundaries=args.aligned_boundaries,
        runtime=original,config=dataclasses.asdict(cfg),reward_config=dataclasses.asdict(reward_cfg),
        prescribed_total_wait_ms=280,training_from_zero=True,own_actor_rollouts=True,
        one_ppo_update=True,no_test_tuning=True,new_api_calls=0,
        carry_prior_restart_cost=False,
        deployment='exact actor coefficients; not compressed table',
        scope='small synthetic heterogeneous workflow diagnostic; not original Figure12 or real agent replay',
        tail='sample maximum only; not population P99',source_sha256=sha(Path(__file__)))
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    core=load_cast_core()
    def manager(policy,collect=False):
        m=AuditedManager(storage_engines={'default':core.Dbx1000VersionedKVStore(),'audit':core.Dbx1000VersionedKVStore()},
            paper_policy=policy,record_traces=False,collect_trajectories=collect,
            performance_guards_enabled=False,low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True,priority_enabled=True,
            policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
        configure_frozen_runtime(m,original);m.expensive_operation_checkpoint_enabled=True
        for key in range(32):m.register_object(f'shift-key:{key}','0',kind='row',storage_id='audit' if key%2 else 'default')
        return m
    all_rewarded=[];policies={};phase_train=[];train_started=time.perf_counter()
    for arm in ('full','nophase'):
        template=JointMaskTablePolicy.from_dict(read(args.schema/f'{arm}_policy.json'))
        actor=fresh_actor(template,arm);sampler=OnPolicySampler(actor,arm=='nophase')
        before=JointMaskTablePolicy.compile(actor,generation=2026101711,remove_phase_index_proxy=arm=='nophase').to_dict()
        raw=[]
        for epoch,tasks in enumerate(plan['training_inputs']):
            m=manager(sampler,True);record=execute(m,None,tasks,epoch)
            (args.output/f'{arm}_train_{epoch}.json').write_text(json.dumps(record,indent=2))
            batch=[dataclasses.asdict(t) for t in m.trajectory_collector.snapshot()]
            for row in batch:
                row['txn_id']=f'{arm}:{epoch}:'+row['txn_id'];row['source_id']=f'hetero:{arm}:{epoch}'
            raw.extend(batch)
        rows=checked_rows(raw,'hetero:'+arm)
        probability=verify_onpolicy(actor,rows,arm=='nophase')
        rewarded,reward=apply_paper_rewards(rows,reward_cfg);all_rewarded.extend(rewarded)
        update=DiscretePPOTrainer(cfg).train(actor,training_rows(rewarded,arm=='nophase'))
        after=JointMaskTablePolicy.compile(actor,generation=2026101712,remove_phase_index_proxy=arm=='nophase').to_dict()
        # Numerical replay starts from the same zero actor, independent optimizer.
        replay=actor_from_joint_payload(before,seed=20261012)
        verify_onpolicy(replay,rows,arm=='nophase')
        replay_update=DiscretePPOTrainer(cfg).train(replay,training_rows(rewarded,arm=='nophase'))
        reproduced=JointMaskTablePolicy.compile(replay,generation=2026101712,remove_phase_index_proxy=arm=='nophase').to_dict()
        if update!=replay_update or after!=reproduced or actor.critic_weights!=replay.critic_weights:
            (args.output/f'{arm}_replay_failure.json').write_text(json.dumps(dict(before=before,after=after,
                reproduced=reproduced,update=update,replay_update=replay_update,
                critic=list(actor.critic_weights),replay_critic=list(replay.critic_weights)),indent=2))
            raise ValueError('PPO update does not reproduce')
        (args.output/f'{arm}_training.json').write_text(json.dumps(dict(transitions=raw,before=before,after=after,
            propensity=probability,reward=reward,update=update,numerical_replay=True),indent=2))
        policies[arm]=JointMaskTablePolicy.from_dict(after)
        if arm=='nophase':phase_train=[(features(r['state']),r['state']['phase']) for r in raw]
        print(json.dumps(dict(stage='trained',arm=arm,transitions=len(raw),update=update)),flush=True)
    linear=fit_linear_rewarded(all_rewarded)
    (args.output/'linear_policy.json').write_text(json.dumps(linear,indent=2))
    policies.update(linear=LinearScorePolicy(linear['weights']),hot_always=HotRulePolicy(),item_threshold3=CompiledPhasePolicy([]))
    training_seconds=time.perf_counter()-train_started
    results=[]
    for arm in ARMS:
        rec=RecordingPolicy(policies[arm]);m=manager(rec)
        if arm=='hot_always':m.first_hot_access_events=[];m.interceptor.hooks=FirstHotAccessHooks(m)
        if arm=='item_threshold3':m.interceptor.hooks=ItemFrequencyHooks(m,minimum_accesses=3,window_ms=1000.)
        record=execute(m,rec,plan['test_input'],2)
        record['summary']['arm']=arm
        if arm=='item_threshold3':record['item_decisions']=m.interceptor.hooks.decisions
        (args.output/f'{arm}_test.json').write_text(json.dumps(record,indent=2))
        results.append(record['summary'])
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(dict(stage='tested',**record['summary'])),flush=True)
    tree=fit(phase_train,depth=3,min_leaf=5)
    from collections import Counter
    majority=Counter(y for _,y in phase_train).most_common(1)[0][0]
    held=read(args.output/'nophase_test.json')['decisions']
    phase_test=[(features(d['state']),d['state']['phase']) for d in held]
    diagnostic=dict(tree=tree,training=score(tree,phase_train,majority),test=score(tree,phase_test,majority),
        caveat='Predictability diagnostic, not proof actor infers phase; correlated states')
    (args.output/'phase_recoverability.json').write_text(json.dumps(diagnostic,indent=2))
    import csv
    with (args.output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    audit=dict(all_histories_verified=True,numerical_ppo_replay=True,training_seconds_including_replay_audit=training_seconds,
        source_unchanged=sha(Path(__file__))==plan['source_sha256'],
        complete_test=all(r['commits']==4 for r in results),new_api_calls=0,
        files_sha256={p.name:sha(p) for p in args.output.glob('*.json')})
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(audit=audit,phase_diagnostic=diagnostic)),flush=True)


if __name__=='__main__':main()
