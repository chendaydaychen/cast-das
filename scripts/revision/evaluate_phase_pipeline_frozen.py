"""Compile from training records only, then confirm frozen pipeline deployments."""
import argparse
from collections import defaultdict,deque
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.native import load_cast_core
from agent.runtime.joint_mask_policy import JointMaskTablePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict
from scripts.revision.train_phase_pipeline_pilot import inputs,run
from scripts.revision.run_audited_credit_pressure import LinearScorePolicy
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy
from scripts.revision.evaluate_target_skew_domain import compile_tables
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.audit_phase_pipeline_pilot import audit as audit_training
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_current_credit_ablations import read,sha

ARMS=('full','nophase','linear','hot_always','hot_threshold','full_table','nophase_table')


def load_models(output):
    view=output/'training_view'
    models={a:JointMaskTablePolicy.from_dict(read(view/f'{a}_policy.json')) for a in ('full','nophase')}
    models['linear']=LinearScorePolicy(read(view/'linear_policy.json')['weights'])
    if (view/'parent_linear_policy.json').exists():
        models['linear_parent']=LinearScorePolicy(read(view/'parent_linear_policy.json')['weights'])
    models['hot_always']=models['full'];models['hot_threshold']=models['full']
    for arm in ('full','nophase'):
        table=load_pure_table(output/'compilation/audit.json',view,arm,4)[0]
        table.prebucketed_lookup=True;table.packed_fallback_lookup=True
        table.prepare_lookup_indexes();models[arm+'_table']=table
    return models


def aggregate(rows,arms=ARMS):
    result=[]
    for arm in arms:
        selected=[r for r in rows if r['arm']==arm];elapsed=sum(r['elapsed_seconds'] for r in selected)
        result.append(dict(arm=arm,logical_tasks=2*len(selected),elapsed_seconds=elapsed,
            task_tps=2*len(selected)/elapsed,
            mean_logical_ms=sum(r['mean_logical_ms'] for r in selected)/len(selected),
            max_logical_ms=max(r['max_logical_ms'] for r in selected),
            aborts=sum(r['aborts'] for r in selected),aborted_gap_ms=sum(r['aborted_gap_ms'] for r in selected),
            policy_calls=sum(r['policy_calls'] for r in selected),action_changes=sum(r['action_changes'] for r in selected),
            item_protection_changes=sum(r['item_protection_changes'] for r in selected)))
    return result


def audit(output):
    plan=read(output/'protocol.json');models=load_models(output)
    if sha(output/'runner_executed.py')!=plan['source_sha256']:
        raise ValueError('Frozen evaluator snapshot differs')
    if plan['input']!=inputs(plan['seed'],plan['episodes']):raise ValueError('Frozen inputs differ')
    if plan['seed'] in (plan['training_seed'],plan['prior_test_seed']):raise ValueError('Reused training/prior test seed')
    for name,digest in plan['artifact_sha256'].items():
        if sha(output/name)!=digest:raise ValueError('Frozen artifact differs: '+name)
    results=read(output/'results.json');queries={};histories=0
    for arm in plan['arms']:
        selected=[r for r in results if r['arm']==arm]
        if sorted(r['episode'] for r in selected)!=list(range(plan['episodes'])):raise ValueError('Incomplete confirmation')
        for row in selected:
            r=read(output/f'{arm}_{row["episode"]}.json');s=r['summary'];tasks=r['tasks']
            if s!=row or verify_history(**r['history'])!=r['oracle'] or r['oracle']['verdict']!='verified':
                raise ValueError('Raw summary/history mismatch')
            histories+=1;attempts=[a for t in tasks for a in t['attempts']]
            item=plan['input'][row['episode']]
            if r['profile']!=item['profile']:raise ValueError('Executed input profile differs')
            order=['prefix','cheap','reason'] if item['profile']=='explore' else ['prefix','reason','cheap']
            for attempt in tasks[0]['attempts']:
                pauses=attempt['paid']
                if [p['name'] for p in pauses]!=order[:len(pauses)]:raise ValueError('Executed work ordering differs')
                if any(p['requested_ms']!=item['config'][p['name']+'_ms'] for p in pauses):
                    raise ValueError('Executed wait differs from fixed input')
            expected=dict(commits=sum(t['committed'] for t in tasks),task_tps=2/s['elapsed_seconds'],
                mean_logical_ms=sum(t['latency_ms'] for t in tasks)/2,
                max_logical_ms=max(t['latency_ms'] for t in tasks),
                aborts=sum(not a['committed'] for a in attempts),
                aborted_gap_ms=sum(p['actual_ms'] for a in attempts if not a['committed'] for p in a['paid']),
                policy_calls=len(r['decisions']),action_changes=sum(a['action_changes'] for a in attempts),
                item_protection_changes=sum(a.get('item_protection_changes',0) for a in attempts))
            if any(s[k]!=v for k,v in expected.items()):raise ValueError('Measured counter reconstruction differs')
            if arm.startswith('hot_'):
                threshold=1 if arm=='hot_always' else 3;recent=defaultdict(deque);last=-1
                for d in r['item_decisions']:
                    now=d['observed_ns']
                    if now<last:raise ValueError('Unordered item observations')
                    last=now;queue=recent[d['key']]
                    while queue and queue[0]<now-1_000_000_000:queue.popleft()
                    queue.append(now)
                    if len(queue)!=d['recent_accesses'] or d['protect']!=(len(queue)>=threshold):
                        raise ValueError('Item-frequency replay differs')
            for d in r['decisions']:
                state=phase_aware_state_from_dict(d['state'])
                if models[arm].select(state).encoded!=d['action'] or d['action']&state.current_action!=state.current_action:
                    raise ValueError('Deployment action differs')
                family=arm.removesuffix('_table')
                if family in ('full','nophase'):
                    q=queries.setdefault(arm,dict(queries=0,matches=0))
                    q['queries']+=1
                    q['matches']+=models[family].select(state).encoded==models[family+'_table'].select(state).encoded
    recomputed=aggregate(results,plan['arms'])
    if recomputed!=read(output/'aggregate.json'):raise ValueError('Aggregate reconstruction differs')
    report=dict(passed=True,histories_verified=histories,actor_table_same_question=queries,
        table_training_input_only=True,models_frozen_before_new_input_execution=True,
        caveat='Constructed two-workflow action-space/return ablation; no universal PPO-over-Linear claim or stable P99',
        auditor_sha256=sha(Path(__file__)))
    (output/'independent_audit.json').write_text(json.dumps(report,indent=2))
    (output/'auditor_executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026102204);p.add_argument('--episodes',type=int,choices=(8,16),default=16)
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.audit_only:audit(args.output);return
    audit_training(args.training)
    training=read(args.training/'protocol.json')
    args.output.mkdir(parents=True,exist_ok=False);view=args.output/'training_view';view.mkdir()
    for name in ('full_policy.json','nophase_policy.json','linear_policy.json'):
        (view/name).write_bytes((args.training/name).read_bytes())
    arms=ARMS
    if training.get('curriculum_parent'):
        (view/'parent_linear_policy.json').write_bytes((args.training/'parent_linear_policy.json').read_bytes())
        arms=ARMS+('linear_parent',)
    records=[]
    stage=args.training
    while stage is not None:
        stage_plan=read(stage/'protocol.json')
        for arm in ('full','nophase'):
            for iteration in range(stage_plan['updates_per_arm']):
                records.extend(read(stage/f'{arm}_update_{iteration}.json')['transitions'])
        stage=Path(stage_plan['curriculum_parent']) if stage_plan.get('curriculum_parent') else None
    (view/'training_transitions.json').write_text(json.dumps(dict(transitions=records),indent=2))
    tick=time.perf_counter();compile_tables(view,args.output/'compilation')
    compile_seconds=time.perf_counter()-tick
    tick=time.perf_counter();models=load_models(args.output);load_index_seconds=time.perf_counter()-tick
    (args.output/'timing.json').write_text(json.dumps(dict(two_table_compile_seconds=compile_seconds,
        model_load_and_two_table_index_seconds=load_index_seconds),indent=2))
    names=['training_view/'+n for n in ('full_policy.json','nophase_policy.json','linear_policy.json','training_transitions.json')]
    names+=['compilation/'+n for n in ('audit.json','full_table.json','nophase_table.json')]
    if 'linear_parent' in arms:names.append('training_view/parent_linear_policy.json')
    plan=dict(training=str(args.training.resolve()),training_protocol_sha256=sha(args.training/'protocol.json'),
        training_seed=training['training_seed'],prior_test_seed=training['test_seed'],
        input=inputs(args.seed,args.episodes),seed=args.seed,episodes=args.episodes,arms=arms,
        runtime=training['runtime'],observed_access_checkpoint=training['observed_access_checkpoint'],
        reward_objective=training['reward_objective'],allowed_masks=training['allowed_masks'],
        scope='New-input frozen confirmation of constructed pipeline timing ablation; not original YCSB',
        source_sha256=sha(Path(__file__)),artifact_sha256={n:sha(args.output/n) for n in names},new_api_calls=0)
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    core=load_cast_core();rows=[]
    for i,item in enumerate(plan['input']):
        offset=i%len(arms)
        for arm in arms[offset:]+arms[:offset]:
            rec=RecordingPolicy(models[arm]);threshold={'hot_always':1,'hot_threshold':3}.get(arm)
            r,_=run(core,plan['runtime'],rec,item,i,threshold=threshold,
                observed_access=plan['observed_access_checkpoint'])
            r['summary'].update(arm=arm,episode=i,max_logical_ms=max(t['latency_ms'] for t in r['tasks']))
            (args.output/f'{arm}_{i}.json').write_text(json.dumps(r,indent=2))
            rows.append(r['summary'])
        print(json.dumps(dict(stage='confirmed_episode',episode=i,completed=len(arms))),flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2))
    totals=aggregate(rows,arms);(args.output/'aggregate.json').write_text(json.dumps(totals,indent=2))
    for name,data in (('results.csv',rows),('aggregate.csv',totals)):
        with (args.output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(stage='complete',aggregate=totals)),flush=True)
    audit(args.output)


if __name__=='__main__':main()
