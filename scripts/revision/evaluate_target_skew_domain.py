"""Compile training-only pure tables, then compare five policies on a new seed."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc.atcc.mask_closure import compile_mask_closed_policy
from agent.native import load_cast_core
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,actor_from_joint_payload
from agent.runtime.state_collector import PhaseAwareState,phase_aware_state_from_dict
from agent.runtime.paper_policy import HotRulePolicy
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.train_joint_skew_domain import audit as audit_training,TEST_SEED
from scripts.revision.run_joint_skew_transfer import make_skew_cohorts,footprint
from scripts.revision.run_frozen_ycsb_shift import run_cohort,summarize
from scripts.revision.audit_frozen_ycsb_shift import checked_attempts
from scripts.revision.run_audited_credit_pressure import AuditedManager,LinearScorePolicy
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy,FixedProtectionPolicy
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.audit_serial_history import verify_history

ARMS=('full','nophase','linear','hot_always','static')


def policies(training,compilation):
    result={a:load_pure_table(compilation/'audit.json',training,a,4)[0] for a in ('full','nophase')}
    result.update(linear=LinearScorePolicy(read(training/'linear_policy.json')['weights']),
        hot_always=HotRulePolicy(),static=FixedProtectionPolicy())
    return result


def compile_tables(training,output,*,zero_cost_prefix_closure=False):
    output.mkdir()
    rows=read(training/'training_transitions.json')['transitions']
    proof=dict(training_sha256=sha(training/'training_transitions.json'),evaluation_inputs_used=False,
        actor_retrained=False,arms={})
    for arm in ('full','nophase'):
        payload=read(training/f'{arm}_policy.json')
        actor=actor_from_joint_payload(payload,seed=20261012)
        for row in rows:
            state=actor.observation(PhaseAwareState(**row['state']))
            actor.prototypes.setdefault(actor.prototype_key(state),state)
        table,report=compile_mask_closed_policy(actor,generation=payload['generation'],
            medoids_per_group=4,all_current_masks=True,context_partitioned_lookup=True,
            schema_context_closure=True,mixed_access_closure=True,remove_phase_index_proxy=arm=='nophase',
            zero_cost_prefix_closure=zero_cost_prefix_closure)
        if JointMaskTablePolicy.compile(actor,generation=payload['generation'],
                remove_phase_index_proxy=arm=='nophase').to_dict()!=payload:
            raise ValueError('Compiler changed learned actor')
        filename=f'{arm}_table.json'
        (output/filename).write_text(json.dumps(table.to_dict(),indent=2))
        proof['arms'][arm]=dict(policy_sha256=sha(training/f'{arm}_policy.json'),
            all_mask_closed_medoid4=dict(compiled_table_filename=filename,
                compiled_table_sha256=sha(output/filename),report=report))
    (output/'audit.json').write_text(json.dumps(proof,indent=2))


def audit(args):
    plan=read(args.output/'plan.json')
    training_plan=read(args.training/'protocol.json')
    if plan['seed']!=training_plan['heldout_seed'] or plan['seed']!=TEST_SEED or plan['arms']!=list(ARMS):
        raise ValueError('Declared evaluation split changed')
    if sha(args.output/'compilation/audit.json')!=plan['compilation_sha256']:
        raise ValueError('Tables changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Executed source changed')
    models=policies(args.training,args.output/'compilation')
    cohorts=read(args.output/'cohorts.json')
    if cohorts!=make_skew_cohorts(seed=TEST_SEED) or sha(args.output/'cohorts.json')!=plan['input_sha256']:
        raise ValueError('Held-out input changed')
    rows=read(args.output/'results.json')
    if [(r['arm'],r['epoch']) for r in rows]!=[(a,e) for a in ARMS for e in range(2)]:
        raise ValueError('Incomplete comparison')
    checks=[]
    common_states={0:[],1:[]}
    for arm in ARMS:
        previous=None
        for cohort in cohorts:
            record=read(args.output/arm/f'epoch_{cohort["epoch"]}.json')
            row=record['summary']
            if row!=next(r for r in rows if (r['arm'],r['epoch'])==(arm,cohort['epoch'])):
                raise ValueError('Aggregate differs')
            recomputed=summarize(record['tasks'],row['workload_seconds'])
            if any(row[k]!=v for k,v in recomputed.items()):
                raise ValueError('Task counters differ')
            checked_attempts(record,cohort,4,6,plan['carry_prior_restart_cost'])
            if verify_history(**record['history'])!=record['oracle'] or record['oracle']['verdict']!='verified':
                raise ValueError('Invalid history')
            if previous and (record['heat_before']!=previous['heat_after']
                    or row['prior_history_attempts']!=len(previous['history']['attempts'])):
                raise ValueError('Hotness or history reset')
            actor=JointMaskTablePolicy.from_dict(read(args.training/f'{arm}_policy.json')) if arm in ('full','nophase') else None
            agreement=0
            for d in record['decisions']:
                state=phase_aware_state_from_dict(d['state'])
                if models[arm].select(state).encoded!=d['action']:
                    raise ValueError('Executed decision differs')
                if actor:
                    agreement+=actor.select(state).encoded==d['action']
                    common_states[cohort['epoch']].append(state)
            if arm=='static' and row['action_changes']!=0:
                raise ValueError('Static changed')
            checks.append(dict(arm=arm,epoch=cohort['epoch'],queries=len(record['decisions']),
                actor_table_matches=agreement if actor else None,records_sha256=sha(args.output/arm/f'epoch_{cohort["epoch"]}.json')))
            previous=record
    answers=[dict(epoch=e,total=len(states),same=sum(models['full'].select(s).encoded==models['nophase'].select(s).encoded for s in states))
        for e,states in common_states.items()]
    report=dict(passed=True,records=checks,full_nophase_same_inputs=answers,
        footprints=[footprint(c) for c in cohorts],new_api_calls=0,
        caveat='One new-seed four-task cohort per regime, conditional synthetic application, not stable P99 or universal superiority')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    with (args.output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    archive=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as bundle:
        bundle.add(args.output,arcname='evaluation')
        bundle.add(args.training,arcname='training')
        for name in plan['source_sha256']:
            bundle.add(ROOT/name,arcname='source/'+name)
    print(json.dumps(dict(**report,archive=str(archive),sha256=sha(archive))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.audit_only:
        audit(args);return
    audit_training(args.training)
    original=read(args.training/'protocol.json')
    args.output.mkdir(parents=True,exist_ok=False)
    compile_tables(args.training,args.output/'compilation')
    cohorts=make_skew_cohorts(seed=TEST_SEED)
    (args.output/'cohorts.json').write_text(json.dumps(cohorts,indent=2))
    sources=dict(original['runtime_source_sha256'])
    for name in ('scripts/revision/evaluate_target_skew_domain.py','agent/cc/atcc/mask_closure.py',
                 'agent/runtime/paper_policy.py','agent/runtime/hot_access_hooks.py'):
        sources[name]=sha(ROOT/name)
    plan=dict(seed=TEST_SEED,arms=ARMS,input_sha256=sha(args.output/'cohorts.json'),
        compilation_sha256=sha(args.output/'compilation/audit.json'),source_sha256=sources,
        carry_prior_restart_cost=original.get('carry_prior_restart_cost',False),
        trained_on_test=False,coefficients_changed=False)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    selected=policies(args.training,args.output/'compilation')
    core=load_cast_core()
    rows=[]
    for arm in ARMS:
        (args.output/arm).mkdir()
        recorded=RecordingPolicy(selected[arm])
        manager=AuditedManager(storage_engines={'default':core.Dbx1000VersionedKVStore(),
            'audit':core.Dbx1000VersionedKVStore()},paper_policy=recorded,record_traces=False,
            collect_trajectories=False,performance_guards_enabled=False,low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True,priority_enabled=True,
            policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
        configure_frozen_runtime(manager,original)
        manager.expensive_operation_checkpoint_enabled=original['before_reasoning_checkpoint']
        if arm=='static':manager.paper_static_initial_mask=4
        if arm=='hot_always':
            manager.first_hot_access_events=[]
            manager.interceptor.hooks=FirstHotAccessHooks(manager)
        for key in range(128):
            manager.register_object(f'shift-key:{key}','0',kind='row',storage_id='audit' if key%2 else 'default')
        for cohort in cohorts:
            record=run_cohort(manager,recorded,cohort,system='paper-atcc',delays_ms=original['delays_ms'],
                max_attempts=6,carry_cost=plan['carry_prior_restart_cost'])
            record['summary'].update(arm=arm)
            (args.output/arm/f'epoch_{cohort["epoch"]}.json').write_text(json.dumps(record,indent=2))
            rows.append(record['summary'])
            (args.output/'results.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(record['summary']),flush=True)


if __name__=='__main__':main()
