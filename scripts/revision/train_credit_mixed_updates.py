"""Paired own-actor continuation with training-only timed short updates.

Both actors receive two sequential two-task training rollouts. Linear receives
all the additional measured trajectories and the prior training information.
No evaluation state, timing or action is used for fitting.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig
from agent.runtime import PaperRewardConfig,apply_paper_rewards
from agent.runtime.residual_policy import ResidualTablePolicy
from scripts.revision.compare_frozen_credit_trace import run_system
from scripts.revision.run_frozen_credit_short_updates import MixedManager,ShortUpdateHooks
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import validate_inputs
from scripts.revision.frozen_actor_collection import actor_from_payload,training_ancestry_batches
from scripts.revision.train_ycsb_onpolicy import OnPolicySampler,verify_onpolicy,training_rows
from scripts.revision.train_ycsb_mixed_prefix import checked_rows
from scripts.revision.run_audited_credit_pressure import fit_linear
from scripts.revision.audit_serial_history import verify_history


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def training_input(prior):
    protocol=load(prior/'protocol.json')
    path=prior/'additional_train_capture.json'
    if sha(path)!=protocol['additional_training_capture_sha256']:
        raise ValueError('Training-only capture differs from its frozen provenance')
    source=load(path)
    companies={t['task']['company_id'] for t in source['tasks']}
    if len(source['tasks'])!=2 or companies!={221} or not all(t['committed'] for t in source['tasks']):
        raise ValueError('This pilot declares the two original company221 training intents')
    return source,path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prior','capture','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    original,_=validate_inputs(args.capture,args.prior)
    source,source_path=training_input(args.prior)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(original,sampling_model_folder=str(args.prior.resolve()),
        sampling_model_sha256={f'{a}_policy.json':sha(args.prior/f'{a}_policy.json') for a in ('full','nophase')},
        training_workers=2,training_repeat_factor=1,training_tasks_per_batch=2,
        training_iterations=2,collection_batches=2,on_policy_collection=True,
        training_exploration='own stochastic actor',replay_gradients=False,
        shared_behavior_data_for_all_learned_arms=False,
        training_scope='two sequential own-actor mixed-environment updates per arm; company221 only',
        mixed_training_input_sha256=sha(source_path),mixed_training_arrivals_seconds=[8.],
        mixed_training_company=221,mixed_training_key='credit:company:221:limit',
        linear_training_information='ancestry plus all four new measured actor rollouts',
        planned_evaluation_company=222,planned_evaluation_arrivals_seconds=[10.],
        planned_evaluation_arms=['full','nophase','linear','hot_always','item_threshold3','silo'],
        evaluation_performed=False,new_api_calls=0,
        experiment_role='exploratory mixed-workload continuation; existing evaluation inspected previously',
        runtime_source_sha256={name:sha(ROOT/name) for name in (
            'scripts/revision/train_credit_mixed_updates.py','scripts/revision/run_frozen_credit_short_updates.py',
            'scripts/revision/compare_frozen_credit_trace.py','agent/runtime/transaction.py',
            'agent/runtime/paper_hooks.py','agent/runtime/atcc_lock_manager.py','agent/runtime/priority.py',
            'agent/cc/atcc/ppo.py','agent/runtime/residual_policy.py','agent/runtime/state_collector.py',
            'agent/runtime/trajectory.py','scripts/revision/frozen_actor_collection.py',
            'scripts/revision/train_ycsb_onpolicy.py','scripts/revision/run_audited_credit_pressure.py')})
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    shutil.copyfile(source_path,args.output/'mixed_training_input.json')
    # Retain the inherited training lineage without pretending all its tasks
    # were re-executed in this continuation.
    for name in ('additional_train_capture.json','merged_training_source.json'):
        shutil.copyfile(args.prior/name,args.output/name)
    cfg=PPOConfig(conditioned_mask_gradient=True,logged_action_support=True,
        min_group_samples=4,ppo_reference_mode='behavior',initialize_from_behavior=False)
    reward_cfg=PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
        terminal_cost_scaling=original['terminal_cost_scaling'])
    actors={a:actor_from_payload(load(args.prior/f'{a}_policy.json'),seed=original['actor_seed'])
        for a in ('full','nophase')}
    all_rows=[]
    records=[]
    started=time.perf_counter()
    for iteration in range(2):
        for arm in ('full','nophase'):
            actor=actors[arm]
            before=ResidualTablePolicy.compile(actor,generation=2026101700+iteration,
                remove_phase_index_proxy=arm=='nophase').to_dict()
            critic_before=list(actor.critic_weights)
            sampler=OnPolicySampler(actor,arm=='nophase')
            holder=[]
            def factory(**kwargs):
                manager=MixedManager(**kwargs,arrival_seconds=[8.],update_key='credit:company:221:limit')
                configure_frozen_runtime(manager,original)
                manager.interceptor.hooks=ShortUpdateHooks(manager.interceptor.hooks)
                holder.append(manager)
                return manager
            raw,attempts=[],[]
            collected=time.perf_counter()
            try:
                result=run_system(source,'paper-atcc',workers=2,max_attempts=original['max_attempts'],
                    time_scale=1.,repeat_factor=1,policy=sampler,trajectory_sink=raw,attempt_sink=attempts,
                    manager_factory=factory,frozen_agent_schedule=True,policy_commit_batch=original['policy_commit_batch'],
                    policy_selected_write_admission_only=original['policy_selected_write_admission_only'],
                    before_reasoning_checkpoint=original['before_reasoning_checkpoint'],
                    carry_prior_restart_cost=original.get('carry_prior_restart_cost',False))
            finally:
                if holder:
                    holder[0].join_short_updates()
            manager=holder[0]
            history=manager.history()
            oracle=verify_history(**history)
            source_id=f'mixed-credit:{arm}:iteration{iteration}'
            for row in raw:
                row.update(source_id=source_id,txn_id=source_id+':'+row['txn_id'])
            record=dict(iteration=iteration,arm=arm,before_actor=before,critic_before=critic_before,
                result=result,attempts=attempts,transitions=raw,history=history,oracle=oracle,
                short_updates=manager.short_records,collect_seconds=time.perf_counter()-collected)
            raw_path=args.output/f'{arm}_iteration{iteration}.json'
            raw_path.write_text(json.dumps(record,indent=2))
            if oracle['verdict']!='verified':
                raise ValueError('Combined training history rejected; raw record retained')
            rows=checked_rows(raw,source_id)
            propensity=verify_onpolicy(actor,rows,arm=='nophase')
            rewarded,rewards=apply_paper_rewards(rows,reward_cfg)
            fit_started=time.perf_counter()
            fit=DiscretePPOTrainer(cfg).train(actor,training_rows(rewarded,arm=='nophase'))
            after=ResidualTablePolicy.compile(actor,generation=2026101701+iteration,
                remove_phase_index_proxy=arm=='nophase').to_dict()
            record.update(propensity=propensity,config=dataclasses.asdict(cfg),reward=rewards,
                fit_report=fit,after_actor=after,critic_after=list(actor.critic_weights),
                fit_compile_seconds=time.perf_counter()-fit_started)
            raw_path.write_text(json.dumps(record,indent=2))
            (args.output/f'{arm}_policy.json').write_text(json.dumps(after,indent=2))
            records.append(dict(arm=arm,iteration=iteration,result=result,
                transitions=len(rows),collect_seconds=record['collect_seconds'],
                fit_compile_seconds=record['fit_compile_seconds'],
                gradient_updates=sum(epoch['updates'] for epoch in fit.get('epochs',[])),
                residual_fit=fit.get('group_residual_training')))
            all_rows.extend(raw)
            (args.output/'progress.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(records[-1]),flush=True)
    path=args.output/'training_transitions.json'
    path.write_text(json.dumps(dict(transitions=all_rows),indent=2))
    _,combined=apply_paper_rewards(checked_rows(all_rows,'mixed-credit-combined'),reward_cfg)
    report=dict(arms=records,reward=combined,collect_fit_compile_seconds=time.perf_counter()-started)
    report_path=args.output/'training_report.json'
    report_path.write_text(json.dumps(report,indent=2))
    linear_started=time.perf_counter()
    linear=fit_linear(path,report_path,prior_batches=training_ancestry_batches(args.prior))
    (args.output/'linear_policy.json').write_text(json.dumps(linear,indent=2))
    report.update(linear_fit_seconds=time.perf_counter()-linear_started,total_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status='trained-not-evaluated',total_seconds=report['total_seconds'])),flush=True)


if __name__=='__main__':
    main()
