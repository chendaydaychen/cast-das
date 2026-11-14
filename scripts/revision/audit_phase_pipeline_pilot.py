"""Offline reconstruction of the constructed pipeline PPO pilot."""
import argparse
from collections import defaultdict,deque
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.revision.train_phase_pipeline_pilot import (inputs,workflow_reward_rows,parent_rewarded_rows,
    remove_pipeline_rounds,pipeline_payload,restore_training_actor)
from scripts.revision.train_heterogeneous_phase_updates import payload,actor_training_rows
from scripts.revision.train_ycsb_mixed_prefix import checked_rows
from scripts.revision.train_ycsb_onpolicy import verify_onpolicy
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_audited_credit_pressure import LinearScorePolicy,fit_linear_rewarded
from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig
from agent.runtime import PaperRewardConfig,apply_paper_rewards
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,actor_from_joint_payload
from agent.runtime.state_collector import phase_aware_state_from_dict
from agent.runtime.state_collector import PHASE_FEATURE_FIELDS
from agent.cc.atcc.ppo import state_features


def audit(folder):
    plan=read(folder/'protocol.json');rewarded_all=[];proof=[];models={}
    if plan.get('curriculum_parent'):
        parent=Path(plan['curriculum_parent'])
        for name,digest in plan['parent_artifact_sha256'].items():
            if sha(parent/name)!=digest:raise ValueError('Frozen curriculum parent differs')
        rewarded_all=parent_rewarded_rows(parent)
    if sha(folder/'runner_executed.py')!=plan['sources']['scripts/revision/train_phase_pipeline_pilot.py']:
        raise ValueError('Runner snapshot mismatch')
    ntest=len(plan['test_input'])
    if plan['training_input']!=inputs(plan['training_seed'],len(plan['training_input'])) or plan['test_input']!=inputs(plan['test_seed'],ntest):
        raise ValueError('Input generation differs')
    if plan.get('warmup_input') and plan['warmup_input']!=inputs(plan['warmup_seed'],len(plan['warmup_input'])):
        raise ValueError('Warm input generation differs')
    for arm in ('full','nophase'):
        previous=None;critic=None;previous_bits=None
        for iteration in range(plan['updates_per_arm']):
            f=read(folder/f'{arm}_update_{iteration}.json')
            if previous is not None and (f['before']!=previous or f['critic_before']!=critic):
                raise ValueError('Training chain mismatch')
            if iteration==0 and (any(v for row in f['before']['weights'] for v in row)
                    or f['before']['support'] or f['before']['groups'] or any(f['critic_before'])):
                raise ValueError('Not from-zero initialization')
            if plan.get('shared_bit_actor'):
                if 'bit_before' not in f or 'bit_after' not in f:
                    raise ValueError('Missing shared-bit training parameters')
                if previous_bits is not None and f['bit_before']!=previous_bits:
                    raise ValueError('Shared-bit parameter chain mismatch')
                if not plan['config']['conditioned_mask_gradient']:
                    raise ValueError('Shared-bit derivatives not enabled')
            elif 'bit_before' in f:
                raise ValueError('Undeclared shared-bit parameters')
            actor=restore_training_actor(f);actor.critic_weights=f['critic_before'][:]
            if pipeline_payload(actor,arm,2026102100+iteration,plan)!=f['before']:
                raise ValueError('Stored training actor differs from frozen joint law')
            if plan.get('phase_feature_group_ablation') and actor.remove_phase_features!=(arm=='nophase'):
                raise ValueError('Declared phase-feature observation mask differs')
            if plan.get('no_explicit_phase_label') and actor.phase_awareness!='disabled':
                raise ValueError('No-label experiment received a phase label')
            if plan.get('curriculum_parent'):
                parent_payload=read(folder/f'parent_{arm}_policy.json')
                if (parent_payload!=read(parent/f'{arm}_policy.json') or f['before']['prior_policy']!=parent_payload
                        or f['after']['prior_policy']!=parent_payload or f['before'].get('allowed_masks') is not None):
                    raise ValueError('Parent prior changed or full action space not restored')
            rows=checked_rows(f['transitions'],f'pipeline:{arm}:{iteration}')
            if plan.get('label_only_nophase') and arm=='nophase' and (
                    actor.remove_phase_index_proxy or actor.remove_phase_features or actor.phase_awareness!='disabled'):
                raise ValueError('Label-only observation removes progress or retains phase')
            if verify_onpolicy(actor,rows,remove_pipeline_rounds(arm,plan))!=f['propensity']:raise ValueError('Propensity mismatch')
            if plan.get('curriculum_parent'):
                from agent.cc.atcc.ppo import valid_actions
                expected_space=dict(all_legal_actions_positive=all(all(t.behavior_action_probabilities[a]>0 for a in valid_actions(t.state.current_action)) for t in rows),
                    sampled_masks=sorted({t.action for t in rows}),sampled_outside_parent_pair=sum(t.action not in (0,3) for t in rows))
                if expected_space!=f['expanded_space'] or not expected_space['all_legal_actions_positive']:
                    raise ValueError('Full-action sampling support differs')
            if plan.get('reward_objective')=='workflow-makespan':
                episodes=[read(folder/f'{arm}_train_{iteration}_{j}.json') for j in range(plan['episodes_per_update'])]
                rewarded,reward=workflow_reward_rows(rows,episodes)
                # Check the complete logical-episode objective separately from
                # the helper: costs must cover both workflows and all retries.
                for j,r in enumerate(episodes):
                    selected=[t for t in rewarded if t.txn_id==f'workflow:{j}']
                    if sum(t.done for t in selected)!=1 or not selected[-1].done:
                        raise ValueError('Retry terminal linkage differs')
                    if selected[-1].reward!=-10.*r['summary']['elapsed_seconds'] or any(t.reward for t in selected[:-1]):
                        raise ValueError('Workflow reward formula differs')
            else:rewarded,reward=apply_paper_rewards(rows,PaperRewardConfig(**plan['reward_config']))
            if plan.get('no_explicit_phase_label'):
                for t in actor_training_rows(actor,rewarded,remove_pipeline_rounds(arm,plan)):
                    if t.state.phase!='all' or any(state_features(t.state)[i] for i in (1,2,3)):
                        raise ValueError('Explicit label reached gradient input')
            if plan.get('phase_feature_group_ablation') and arm=='nophase':
                training=actor_training_rows(actor,rewarded,True)
                for t in training:
                    for s in (t.state,t.next_state):
                        if s is None:continue
                        if s.phase!='all' or any(getattr(s,name)!=0 for name in PHASE_FEATURE_FIELDS):
                            raise ValueError('Removed phase feature reached gradient input')
                        if any(state_features(s)[i]!=0 for i in (1,2,3,6,7,8,10,27,28,29)):
                            raise ValueError('Derived phase feature reached gradient input')
            rewarded_all.extend(rewarded)
            fit=DiscretePPOTrainer(PPOConfig(**plan['config'])).train(actor,actor_training_rows(actor,rewarded,remove_pipeline_rounds(arm,plan)))
            if reward!=f['reward'] or fit!=f['fit'] or pipeline_payload(actor,arm,2026102101+iteration,plan)!=f['after'] or actor.critic_weights!=f['critic_after']:
                raise ValueError('PPO replay mismatch')
            previous=f['after'];critic=f['critic_after']
            if plan.get('shared_bit_actor'):
                if actor.to_dict()!=f['bit_after']:
                    raise ValueError('Shared-bit PPO parameter replay mismatch')
                previous_bits=f['bit_after']
            proof.append(dict(arm=arm,update=iteration,rows=len(rows),ppo_and_critic_reproduced=True))
        if read(folder/f'{arm}_policy.json')!=previous:raise ValueError('Frozen model differs')
        models[arm]=JointMaskTablePolicy.from_dict(previous)
    linear=fit_linear_rewarded(rewarded_all)
    if linear!=read(folder/'linear_policy.json'):raise ValueError('Linear replay mismatch')
    models['linear']=LinearScorePolicy(linear['weights'])
    history_count=0;query_count=0;rule_coverage={};first_decisions=[]
    for path in sorted(folder.glob('*_train_*_*.json'))+sorted(folder.glob('*_test_*.json')):
        r=read(path)
        if verify_history(**r['history'])!=r['oracle'] or r['oracle']['verdict']!='verified':
            raise ValueError('Invalid history')
        history_count+=1
        warmup=r.get('warmup_records',[])
        if len(warmup)!=len(plan.get('warmup_input',[])):
            raise ValueError('Shared warmup absent or unexpected')
        if warmup and not r.get('shared_key_hot_after_warmup'):
            raise ValueError('Actual warm hotspot not established')
        for warm,item in zip(warmup,plan.get('warmup_input',[])):
            if warm['executed_config']!=item['config'] or warm['profile']!=item['profile']:
                raise ValueError('Warm execution input differs')
            if verify_history(**warm['history'])!=warm['oracle'] or warm['oracle']['verdict']!='verified':
                raise ValueError('Warm history failed')
            if not all(t['committed'] for t in warm['tasks']):raise ValueError('Warm tasks incomplete')
            if plan.get('pre_expensive_validation') and any(not a['prefix_validation_calls'] or
                    a['prefix_validation_calls']!=len(a['prefix_validation_events']) for a in warm['tasks'][0]['attempts']):
                raise ValueError('Warm actual prefix checks absent')
        if r.get('pre_expensive_validation_enabled',False)!=plan.get('pre_expensive_validation',False):
            raise ValueError('Shared prefix-validation switch differs')
        if r.get('checkpoint_min_budget_ms',0.)!=plan.get('checkpoint_min_budget_ms',0.):
            raise ValueError('Shared checkpoint budget differs')
        if plan.get('pre_expensive_validation'):
            for attempt in r['tasks'][0]['attempts']:
                if attempt['prefix_validation_calls']!=len(attempt['prefix_validation_events']) or not attempt['prefix_validation_calls']:
                    raise ValueError('Actual foreground prefix checks missing')
        if '_test_' not in path.name:continue
        arm=r['summary']['arm'];s=r['summary'];tasks=r['tasks']
        records=r['decisions'];attempts=[a for t in tasks for a in t['attempts']]
        if records:
            d=records[0];state=d['state']
            first_decisions.append(dict(arm=arm,episode=s['episode'],profile=r['profile'],
                phase=state['phase'],reads=state['read_set_size'],current=state['current_action'],
                action=d['action'],paid_ms=state['observed_restart_cost_ms']))
        expected=dict(commits=sum(t['committed'] for t in tasks),task_tps=2/s['elapsed_seconds'],
            mean_logical_ms=sum(t['latency_ms'] for t in tasks)/2,
            aborts=sum(not a['committed'] for a in attempts),
            aborted_gap_ms=sum(p['actual_ms'] for a in attempts if not a['committed'] for p in a['paid']),
            action_changes=sum(a['action_changes'] for a in attempts),policy_calls=len(records))
        if any(s[k]!=v for k,v in expected.items()):raise ValueError('Summary counter mismatch')
        for d in records:
            state=phase_aware_state_from_dict(d['state'])
            if models[arm].select(state).encoded!=d['action'] or d['action']&state.current_action!=state.current_action:
                raise ValueError('Frozen deployment action mismatch')
        query_count+=len(records)
        if arm.startswith('hot_'):
            threshold=1 if arm=='hot_always' else 3
            if records:raise ValueError('Rule unexpectedly queried learned policy')
            if any(d['protect']!=(d['recent_accesses']>=threshold) for d in r['item_decisions']):
                raise ValueError('Local rule predicate mismatch')
            coverage=rule_coverage.setdefault(arm,dict(access_checks=0,eligible_checks=0))
            coverage['access_checks']+=len(r['item_decisions'])
            coverage['eligible_checks']+=sum(d['protect'] for d in r['item_decisions'])
            if r['item_decisions'] and 'observed_ns' in r['item_decisions'][0]:
                recent=defaultdict(deque);last=-1
                for d in r['item_decisions']:
                    now=d['observed_ns']
                    if now<last:raise ValueError('Access observations not ordered')
                    last=now;queue=recent[d['key']]
                    while queue and queue[0]<now-1_000_000_000:queue.popleft()
                    queue.append(now)
                    if len(queue)!=d['recent_accesses']:raise ValueError('Access window replay differs')
                coverage.setdefault('foreground_eligible_checks',0)
                coverage['foreground_eligible_checks']+=sum(d['protect'] and d['decision_applied'] for d in r['item_decisions'])
                coverage.setdefault('competitor_accesses',0)
                coverage['competitor_accesses']+=sum(not d['decision_applied'] for d in r['item_decisions'])
    results=read(folder/'results.json');aggregate=[]
    for arm in ('full','nophase','linear','hot_always','hot_threshold'):
        rows=[r for r in results if r['arm']==arm]
        if sorted(r['episode'] for r in rows)!=list(range(ntest)):raise ValueError('Incomplete matrix')
        if any(r!=read(folder/f'{arm}_test_{r["episode"]}.json')['summary'] for r in rows):
            raise ValueError('Raw results differ')
        elapsed=sum(r['elapsed_seconds'] for r in rows)
        aggregate.append(dict(arm=arm,logical_tasks=2*ntest,elapsed_seconds=elapsed,task_tps=2*ntest/elapsed,
            mean_logical_ms=sum(r['mean_logical_ms'] for r in rows)/ntest,
            aborts=sum(r['aborts'] for r in rows),aborted_gap_ms=sum(r['aborted_gap_ms'] for r in rows),
            policy_calls=sum(r['policy_calls'] for r in rows),action_changes=sum(r['action_changes'] for r in rows)))
        if 'item_protection_changes' in read(folder/'aggregate.json')[0]:
            aggregate[-1]['item_protection_changes']=sum(r.get('item_protection_changes',0) for r in rows)
    if aggregate!=read(folder/'aggregate.json'):raise ValueError('Aggregate differs')
    with (folder/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    report=dict(passed=True,training_updates=proof,linear_reproduced=True,histories_verified=history_count,
        frozen_queries_verified=query_count,input_seeds_disjoint=plan['training_seed']!=plan['test_seed'],
        auditor_sha256=sha(Path(__file__)),
        rule_coverage=rule_coverage,first_decisions=first_decisions,
        phase_feature_group_inputs_verified=bool(plan.get('phase_feature_group_ablation')),
        label_only_nophase_inputs_verified=bool(plan.get('label_only_nophase')),
        shared_pre_expensive_validation_verified=bool(plan.get('pre_expensive_validation')),
        no_explicit_phase_label_inputs_verified=bool(plan.get('no_explicit_phase_label')),
        limitations='Small constructed two-workflow pilot; class changes exclude item lock changes; not table fidelity or stable P99')
    (folder/'independent_audit.json').write_text(json.dumps(report,indent=2))
    (folder/'auditor_executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='first_decisions'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
