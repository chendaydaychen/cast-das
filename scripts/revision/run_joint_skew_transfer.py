"""Predeclared two-cohort skew change with frozen CreditCrew-trained pure tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.native import load_cast_core
from agent.runtime.state_collector import phase_aware_state_from_dict
from agent.runtime.joint_mask_policy import JointMaskTablePolicy
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.run_frozen_ycsb_shift import run_cohort,summarize
from scripts.revision.audit_frozen_ycsb_shift import checked_attempts
from scripts.revision.run_audited_credit_pressure import AuditedManager
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy,FixedProtectionPolicy
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.audit_serial_history import verify_history

SEED=2026101702
WORKERS=4
REGION=128
EXPONENTS=(0.,1.6)
ARMS=('full','nophase','static')


def make_skew_cohorts(seed=SEED,workers=WORKERS,region=REGION):
    """Common random numbers: only sampling weights change between cohorts."""
    if workers<2 or region<10:
        raise ValueError('At least two workers and ten keys required')
    cohorts=[]
    for epoch,exponent in enumerate(EXPONENTS):
        rng=random.Random(seed)
        tasks=[]
        for worker in range(workers):
            remaining=list(range(region))
            operations=[]
            for i in range(10):
                key=rng.choices(remaining,weights=[(k+1)**-exponent for k in remaining])[0]
                remaining.remove(key)
                operations.append(dict(key=key,kind='write' if i in (3,7) else 'read'))
            tasks.append(operations)
        cohorts.append(dict(epoch=epoch,label='uniform_before' if epoch==0 else 'skewed_after',tasks=tasks))
    return cohorts


def footprint(cohort):
    ops=[op for task in cohort['tasks'] for op in task]
    writes={op['key'] for op in ops if op['kind']=='write'}
    return dict(operations=len(ops),distinct_keys=len({op['key'] for op in ops}),
        top8_access_share=sum(op['key']<8 for op in ops)/len(ops),
        workers_with_cross_worker_write_overlap=sum(any(
            a['key']==b['key'] and ('write' in (a['kind'],b['kind']))
            for other,task in enumerate(cohort['tasks']) if other!=worker
            for a in own for b in task) for worker,own in enumerate(cohort['tasks'])),
        unique_write_keys=len(writes))


def audit(args):
    plan=read(args.output/'plan.json')
    cohorts=read(args.output/'cohorts.json')
    if (cohorts!=make_skew_cohorts() or plan['exponents']!=list(EXPONENTS)
            or plan['arms']!=list(ARMS) or plan['input_sha256']!=sha(args.output/'cohorts.json')):
        raise ValueError('Declared skew experiment changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Runtime source changed: '+name)
    for name,digest in plan['training_sha256'].items():
        if sha(args.training/name)!=digest:
            raise ValueError('Frozen model changed')
    if sha(args.calibration)!=plan['calibration_sha256']:
        raise ValueError('Measured delay calibration changed')
    policies={a:load_pure_table(args.table_audit,args.training,a,4)[0] for a in ('full','nophase')}
    policies['static']=FixedProtectionPolicy()
    rows=read(args.output/'results.json')
    if [(r['arm'],r['epoch']) for r in rows]!=[(a,e) for a in ARMS for e in range(2)]:
        raise ValueError('Incomplete declared matrix')
    reports=[]
    for arm in ARMS:
        previous=None
        for cohort in cohorts:
            record=read(args.output/arm/f'epoch_{cohort["epoch"]}.json')
            row=record['summary']
            if row!=next(r for r in rows if (r['arm'],r['epoch'])==(arm,cohort['epoch'])):
                raise ValueError('Recorded summary differs')
            expected=summarize(record['tasks'],row['workload_seconds'])
            if any(row[k]!=v for k,v in expected.items()):
                raise ValueError('Summary does not reproduce')
            checked_attempts(record,cohort,WORKERS,6,plan['carry_prior_restart_cost'])
            oracle=verify_history(**record['history'])
            if oracle!=record['oracle'] or oracle['verdict']!='verified':
                raise ValueError('Committed history failed')
            if arm=='static' and row['action_changes']!=0:
                raise ValueError('Static switched categories')
            agreement=0
            actor=None if arm=='static' else JointMaskTablePolicy.from_dict(read(args.training/f'{arm}_policy.json'))
            for d in record['decisions']:
                state=phase_aware_state_from_dict(d['state'])
                if policies[arm].select(state).encoded!=d['action']:
                    raise ValueError('Actual action differs from frozen policy')
                if d['action']&state.current_action!=state.current_action:
                    raise ValueError('Released held protection')
                agreement+=actor is not None and actor.select(state).encoded==d['action']
            if previous:
                if (record['heat_before']!=previous['heat_after']
                        or row['prior_history_attempts']!=len(previous['history']['attempts'])):
                    raise ValueError('Runtime history/hotness reset between cohorts')
            elif any(v['accesses'] for v in record['heat_before'].values()):
                raise ValueError('Initial heat was primed')
            reports.append(dict(arm=arm,epoch=cohort['epoch'],history='verified',
                decisions=len(record['decisions']),actor_table_matches=agreement if actor else None,
                records_sha256=sha(args.output/arm/f'epoch_{cohort["epoch"]}.json')))
            previous=record
    report=dict(passed=True,records=reports,footprints=[footprint(c) for c in cohorts],
        caveat='One drained cohort each side; not continuous stream or statistical steady-state/P99 proof')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    with (args.output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    archive=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as bundle:
        bundle.add(args.output,arcname='execution')
        bundle.add(args.calibration,arcname='calibration.json')
        bundle.add(args.table_audit.parent,arcname='compilation')
        for name in ('full_policy.json','nophase_policy.json','protocol.json','training_audit.json'):
            bundle.add(args.training/name,arcname='training/'+name)
        for name in plan['source_sha256']:
            bundle.add(ROOT/name,arcname='source/'+name)
    print(json.dumps(dict(**report,archive=str(archive),sha256=sha(archive))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','table-audit','calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.audit_only:
        audit(args);return
    original=read(args.training/'protocol.json')
    calibration=read(args.calibration)['calibration']
    delays=[calibration['delay_ms'][phase] for phase in ('explore','refine','commit')]
    if any(type(v) is not int or v<=0 for v in delays):
        raise ValueError('Measured positive integer delays required')
    policies={a:load_pure_table(args.table_audit,args.training,a,4)[0] for a in ('full','nophase')}
    policies['static']=FixedProtectionPolicy()
    cohorts=make_skew_cohorts()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cohorts.json').write_text(json.dumps(cohorts,indent=2))
    dependencies=('scripts/revision/run_joint_skew_transfer.py','scripts/revision/run_frozen_ycsb_shift.py',
        'scripts/revision/run_audited_credit_pressure.py','scripts/revision/compare_frozen_credit_trace.py',
        'scripts/revision/run_agent_schedule_comparison_20261012.py','agent/runtime/paper_policy.py',
        'agent/runtime/paper_hooks.py','agent/runtime/transaction.py','agent/runtime/operation_interceptor.py',
        'agent/runtime/atcc_lock_manager.py','agent/runtime/priority.py','agent/runtime/joint_mask_policy.py')
    plan=dict(seed=SEED,workers=WORKERS,region=REGION,exponents=EXPONENTS,arms=ARMS,
        delays_ms=delays,calibration_sha256=sha(args.calibration),calibration=calibration,
        input_sha256=sha(args.output/'cohorts.json'),
        training_sha256={n:sha(args.training/n) for n in ('full_policy.json','nophase_policy.json','protocol.json')},
        table_audit_sha256=sha(args.table_audit),source_sha256={n:sha(ROOT/n) for n in dependencies},
        carry_prior_restart_cost=original.get('carry_prior_restart_cost',False),max_attempts=6,
        semantics='Drain uniform cohort, change only Zipf weights, retain native stores/history/hotness/policy; no refit',
        workload='Synthetic 10 unique point accesses, writes at4/8, delays after4/7/10; prescribed phases',
        storage='Two native Dbx stores; even keys default, odd keys audit',
        caveat='Frozen CreditCrew-trained policy transferred to calibrated synthetic input; not live agent execution',
        retrained=False,tuned=False,new_api_calls=0)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    results=[]
    core=load_cast_core()
    for arm in ARMS:
        (args.output/arm).mkdir()
        recorded=RecordingPolicy(policies[arm])
        manager=AuditedManager(storage_engines={'default':core.Dbx1000VersionedKVStore(),
            'audit':core.Dbx1000VersionedKVStore()},paper_policy=recorded,record_traces=False,
            collect_trajectories=False,performance_guards_enabled=False,low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True,priority_enabled=True,
            policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
        configure_frozen_runtime(manager,original)
        manager.expensive_operation_checkpoint_enabled=original['before_reasoning_checkpoint']
        if arm=='static':
            manager.paper_static_initial_mask=4
        for key in range(REGION):
            manager.register_object(f'shift-key:{key}','0',kind='row',storage_id='audit' if key%2 else 'default')
        for cohort in cohorts:
            record=run_cohort(manager,recorded,cohort,system='paper-atcc',delays_ms=delays,
                max_attempts=6,carry_cost=plan['carry_prior_restart_cost'])
            record['summary'].update(arm=arm,zipf_exponent=EXPONENTS[cohort['epoch']])
            (args.output/arm/f'epoch_{cohort["epoch"]}.json').write_text(json.dumps(record,indent=2))
            results.append(record['summary'])
            (args.output/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(record['summary']),flush=True)
    if sha(args.table_audit)!=plan['table_audit_sha256']:
        raise ValueError('Pure table compilation changed')


if __name__=='__main__':
    main()
