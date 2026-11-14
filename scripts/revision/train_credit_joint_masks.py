"""Paired on-policy joint-mask continuation; same mixed CreditCrew protocol."""
import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.revision.train_credit_mixed_updates import (
    sha,load,training_input,validate_inputs,OnPolicySampler,MixedManager,ShortUpdateHooks,
    configure_frozen_runtime,run_system,verify_history,checked_rows,verify_onpolicy,
    apply_paper_rewards,PaperRewardConfig,DiscretePPOTrainer,PPOConfig,training_rows,
    fit_linear,training_ancestry_batches,actor_from_payload,
)
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,convert_factorized_actor
from agent.runtime.state_collector import phase_aware_state_from_dict
from scripts.revision.joint_actor_exploration import initialize_joint_actor


def initial_actor(parent, *, seed, temperature, fresh=False):
    actor=initialize_joint_actor(parent,seed=seed,temperature=temperature)
    if fresh:
        # Retain only observation/action schema, never learned parent scores
        # or parent action-support restrictions. Uniform legal masks at start.
        actor.categorical_actor_weights=[[0.]*37 for _ in range(16)]
        actor.categorical_group_logits.clear()
        actor.frozen_support.clear()
        actor.observed_group_actions.clear()
    return actor


def audit_training(folder,prior):
    protocol=load(folder/'protocol.json')
    source,path=training_input(prior)
    if (protocol['joint_mask_training'] is not True or protocol['joint_rollouts_per_arm'] not in (2,4)
            or sha(path)!=protocol['mixed_training_input_sha256']
            or source!=load(folder/'mixed_training_input.json')):
        raise ValueError('Joint input or training budget changed')
    for name,digest in protocol['runtime_source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Executed source changed: '+name)
    combined,proofs=[],[]
    for arm in ('full','nophase'):
        parent=load(prior/f'{arm}_policy.json')
        if sha(prior/f'{arm}_policy.json')!=protocol['sampling_model_sha256'][f'{arm}_policy.json']:
            raise ValueError('Parent model changed')
        actor=initial_actor(parent,seed=protocol['actor_seed'],
            temperature=protocol.get('initial_temperature',1.),
            fresh=protocol.get('fresh_initialization',False))
        untempered=convert_factorized_actor(parent,seed=protocol['actor_seed'])
        old=actor_from_payload(parent,seed=protocol['actor_seed'])
        record=load(folder/f'{arm}_joint_fit.json')
        if record['config']!=dataclasses.asdict(PPOConfig(min_group_samples=4,initialize_from_behavior=False)):
            raise ValueError('Predeclared PPO configuration changed')
        if JointMaskTablePolicy.compile(actor,generation=2026101701,
                remove_phase_index_proxy=arm=='nophase').to_dict()!=record['before_actor']:
            raise ValueError('Joint actor conversion changed')
        raw=[]
        for batch in range(protocol['joint_rollouts_per_arm']):
            rollout=load(folder/f'{arm}_rollout{batch}.json')
            attempts=[a for t in rollout['attempts'] for a in t['attempt_details']]
            short=rollout['short_updates']
            if (verify_history(**rollout['history'])!=rollout['oracle']
                    or rollout['oracle']['verdict']!='verified'
                    or len(rollout['attempts'])!=2 or len(short)!=1
                    or short[0]['scheduled_arrival_seconds']!=8.
                    or short[0]['observed_start_seconds']<8.-1e-6
                    or len(attempts)!=rollout['result']['attempts']
                    or len(rollout['history']['attempts'])!=len(attempts)+len(short[0]['attempts'])):
                raise ValueError('Joint training history, roles or counts changed')
            raw.extend(rollout['transitions'])
        rows=checked_rows(raw,'joint:'+arm)
        propensity=verify_onpolicy(actor,rows,arm=='nophase')
        maximum=max(max(abs(a-b) for a,b in zip(untempered.probabilities(t.state),old.probabilities(
            dataclasses.replace(t.state,completed_rounds=0) if arm=='nophase' else t.state))) for t in rows)
        if maximum>1e-10:
            raise ValueError('Joint conversion changes parent behavior probabilities')
        rewarded,reward=apply_paper_rewards(rows,PaperRewardConfig(
            positive_wait_event_scaling=protocol['positive_wait_event_scaling'],
            terminal_cost_scaling=protocol['terminal_cost_scaling']))
        fit=DiscretePPOTrainer(PPOConfig(**record['config'])).train(actor,training_rows(rewarded,arm=='nophase'))
        after=JointMaskTablePolicy.compile(actor,generation=2026101702,
            remove_phase_index_proxy=arm=='nophase').to_dict()
        if (propensity!=record['propensity'] or reward!=record['reward'] or fit!=record['fit_report']
                or after!=record['after_actor'] or actor.critic_weights!=record['critic_after']
                or after!=load(folder/f'{arm}_policy.json')):
            raise ValueError('Joint PPO numerical reproduction failed')
        before_policy=JointMaskTablePolicy.from_dict(record['before_actor'])
        after_policy=JointMaskTablePolicy.from_dict(after)
        proofs.append(dict(arm=arm,transitions=len(rows),gradient_reproduced=True,
            propensity=propensity,conversion_max_probability_error=maximum,
            residual_fit=fit.get('group_residual_training'),
            gradient_updates=sum(e['updates'] for e in fit['epochs']),
            changed_greedy_training_answers=sum(before_policy.select(phase_aware_state_from_dict(r['state'])).encoded
                !=after_policy.select(phase_aware_state_from_dict(r['state'])).encoded for r in raw)))
        combined.extend(raw)
    if load(folder/'training_transitions.json')['transitions']!=combined:
        raise ValueError('Joint Linear input changed')
    linear=fit_linear(folder/'training_transitions.json',folder/'training_report.json',
        prior_batches=() if protocol.get('fresh_initialization') else training_ancestry_batches(prior))
    if linear!=load(folder/'linear_policy.json'):
        raise ValueError('Joint Linear numerical reproduction failed')
    return dict(passed=True,proofs=proofs,linear_fit_reproduced=True,
        initial_temperature=protocol.get('initial_temperature',1.),
        fresh_initialization=protocol.get('fresh_initialization',False),
        rollouts_per_arm=protocol['joint_rollouts_per_arm'],
        policy_sha256={a:sha(folder/f'{a}_policy.json') for a in ('full','nophase','linear')},
        source_sha256=sha(Path(__file__)),scope='paired fixed-support joint-mask PPO; company221 only')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prior','capture','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--initial-temperature',type=float,default=1.)
    p.add_argument('--rollouts',type=int,choices=(2,4),default=4)
    p.add_argument('--fresh-initialization',action='store_true',
        help='Zero scores and empty fixed support; train without learned parent coefficients or history')
    args=p.parse_args()
    if args.audit_only:
        proof=audit_training(args.output,args.prior)
        target=args.output/'training_audit.json'
        if target.exists():
            raise ValueError('Preserve previous audit')
        target.write_text(json.dumps(proof,indent=2))
        print(json.dumps(proof),flush=True)
        return
    original,_=validate_inputs(args.capture,args.prior)
    # Validate initialization before creating an experiment directory.
    initial_actor(load(args.prior/'full_policy.json'),seed=original['actor_seed'],
        temperature=args.initial_temperature,fresh=args.fresh_initialization)
    source,source_path=training_input(args.prior)
    args.output.mkdir(parents=True,exist_ok=False)
    names=list(load(ROOT/'results/credit_llm/credit_mixed_training_20261016_v1/protocol.json')['runtime_source_sha256'])
    names.extend(('agent/runtime/joint_mask_policy.py','scripts/revision/joint_actor_exploration.py',
        str(Path(__file__).relative_to(ROOT))))
    protocol=dict(original,sampling_model_folder=str(args.prior.resolve()),
        sampling_model_sha256={f'{a}_policy.json':sha(args.prior/f'{a}_policy.json') for a in ('full','nophase')},
        mixed_training_input_sha256=sha(source_path),joint_mask_training=True,joint_rollouts_per_arm=args.rollouts,
        initial_temperature=args.initial_temperature,
        fresh_initialization=args.fresh_initialization,
        training_workers=2,training_tasks_per_arm=2*args.rollouts,training_iterations=1,
        mixed_training_company=221,mixed_training_arrivals_seconds=[8.],
        planned_evaluation_company=222,planned_evaluation_arrivals_seconds=[10.],
        planned_evaluation_arms=['full','nophase','linear','hot_always','item_threshold3','silo'],
        on_policy_collection=True,training_exploration='own stochastic actor',replay_gradients=False,
        training_scope=('zero-score fresh complete-mask actor; no parent learned coefficients; own rollouts then one PPO update'
            if args.fresh_initialization else 'factorized-to-joint conversion, declared temperature initialization, own rollouts then one PPO update'),
        runtime_source_sha256={name:sha(ROOT/name) for name in names},new_api_calls=0,
        evaluation_performed=False,experiment_role='exploratory architecture repair; previously inspected evaluation input',
        joint_support=('all legal monotone actions, no inherited support restrictions'
            if args.fresh_initialization else 'parent support frozen throughout collection/update/deployment; unseen groups use full monotone action space'),
        linear_training_information=('same newly collected Full/NoPhase rows only'
            if args.fresh_initialization else 'ancestry and new Full/NoPhase rows'))
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ('additional_train_capture.json','merged_training_source.json'):
        shutil.copyfile(args.prior/name,args.output/name)
    shutil.copyfile(source_path,args.output/'mixed_training_input.json')
    cfg=PPOConfig(min_group_samples=4,initialize_from_behavior=False)
    reward_cfg=PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
        terminal_cost_scaling=original['terminal_cost_scaling'])
    all_raw,summaries=[],[]
    started=time.perf_counter()
    for arm in ('full','nophase'):
        actor=initial_actor(load(args.prior/f'{arm}_policy.json'),seed=original['actor_seed'],
            temperature=args.initial_temperature,fresh=args.fresh_initialization)
        before=JointMaskTablePolicy.compile(actor,generation=2026101701,
            remove_phase_index_proxy=arm=='nophase').to_dict()
        sampler=OnPolicySampler(actor,arm=='nophase')
        pooled=[]
        for batch in range(args.rollouts):
            holder=[]
            def factory(**kwargs):
                manager=MixedManager(**kwargs,arrival_seconds=[8.],update_key='credit:company:221:limit')
                configure_frozen_runtime(manager,original)
                manager.interceptor.hooks=ShortUpdateHooks(manager.interceptor.hooks)
                holder.append(manager)
                return manager
            raw,attempts=[],[]
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
            history=holder[0].history()
            oracle=verify_history(**history)
            source_id=f'joint-credit:{arm}:batch{batch}'
            for row in raw:
                row.update(source_id=source_id,txn_id=source_id+':'+row['txn_id'])
            record=dict(arm=arm,batch=batch,result=result,attempts=attempts,transitions=raw,
                history=history,oracle=oracle,short_updates=holder[0].short_records)
            (args.output/f'{arm}_rollout{batch}.json').write_text(json.dumps(record,indent=2))
            if oracle['verdict']!='verified':
                raise ValueError('Joint training history rejected')
            pooled.extend(raw)
            print(json.dumps(dict(arm=arm,batch=batch,transitions=len(raw),commits=result['commits'])),flush=True)
        rows=checked_rows(pooled,'joint:'+arm)
        propensity=verify_onpolicy(actor,rows,arm=='nophase')
        rewarded,reward=apply_paper_rewards(rows,reward_cfg)
        fit=DiscretePPOTrainer(cfg).train(actor,training_rows(rewarded,arm=='nophase'))
        after=JointMaskTablePolicy.compile(actor,generation=2026101702,
            remove_phase_index_proxy=arm=='nophase').to_dict()
        record=dict(before_actor=before,config=dataclasses.asdict(cfg),propensity=propensity,reward=reward,
            fit_report=fit,after_actor=after,critic_after=list(actor.critic_weights))
        (args.output/f'{arm}_joint_fit.json').write_text(json.dumps(record,indent=2))
        (args.output/f'{arm}_policy.json').write_text(json.dumps(after,indent=2))
        summaries.append(dict(arm=arm,transitions=len(rows),residual_fit=fit.get('group_residual_training')))
        all_raw.extend(pooled)
        print(json.dumps(summaries[-1]),flush=True)
    (args.output/'training_transitions.json').write_text(json.dumps(dict(transitions=all_raw),indent=2))
    _,combined_reward=apply_paper_rewards(checked_rows(all_raw,'joint-combined'),reward_cfg)
    report=dict(arms=summaries,reward=combined_reward,collect_fit_compile_seconds=time.perf_counter()-started,
        actor_score_gradient='categorical softmax on fixed monotone supported complete masks',
        deployment='immutable joint coefficient scoring, not a medoid-only table')
    report_path=args.output/'training_report.json'
    report_path.write_text(json.dumps(report,indent=2))
    linear=fit_linear(args.output/'training_transitions.json',report_path,
        prior_batches=() if args.fresh_initialization else training_ancestry_batches(args.prior))
    (args.output/'linear_policy.json').write_text(json.dumps(linear,indent=2))
    report['total_seconds']=time.perf_counter()-started
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status='trained-not-evaluated',**report)),flush=True)


if __name__=='__main__':
    main()
