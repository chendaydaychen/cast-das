"""Controlled pipeline headroom diagnostic; handwritten probes are not RL."""
import argparse
import contextlib
import csv
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import gc
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc import LockConflict
from agent.native import load_cast_core
from agent.runtime.context import LockAction
from agent.runtime.paper_hooks import PaperATCCHooks
from agent.workloads.frozen_credit_agent import tracked_agent_wait
from scripts.revision.train_heterogeneous_phase_updates import make_manager
from scripts.revision.compare_frozen_credit_trace import RecordingPolicy
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.run_audited_credit_pressure import AuditedManager
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.audit_serial_history import verify_history
from scripts.unified_trace.run_credit_review_experiment import transaction_metadata

ARMS=('early_reads','refine_reads','paid_60_reads','hold_current','full_table','nophase_table')
CONFIG=dict(prefix_ms=40.,cheap_ms=20.,reason_ms=300.,writer_arrival_after_read_ms=4.,
    writer_work_ms=5.,dependent_tool_ms=200.,max_attempts=3,backoff_ms=10.)


class Probe:
    def __init__(self,kind):self.kind=kind
    def select(self,s):
        on=(self.kind=='early_reads' or self.kind=='refine_reads' and s.phase in ('refine','commit')
            or self.kind=='paid_60_reads' and s.observed_restart_cost_ms>=60.)
        return LockAction.from_encoded(int(s.current_action)|(3 if on else 0))


class FixedWriterHooks(PaperATCCHooks):
    """Only the foreground is controlled by the policy under investigation."""
    def on_begin(self,txn):
        super().on_begin(txn)
        if txn.metadata.get('pipeline_fixed_writer'):
            txn.context.action=LockAction.from_encoded(12)

    def _select_and_apply(self,txn,state):
        if not txn.metadata.get('pipeline_fixed_writer'):
            return super()._select_and_apply(txn,state)


def execute_pair(manager,recording,profile,index,config=CONFIG,*,audit_at_end=True,allow_incomplete=False):
    root_seen=threading.Event()
    started=time.perf_counter()
    arrival_timeline={}
    def foreground():
        task_start=time.perf_counter();attempts=[]
        clock=getattr(manager,'pipeline_profile_clock',None)
        root=clock.scope('other',root=True) if clock else contextlib.nullcontext()
        root.__enter__()
        for retry in range(config['max_attempts']):
            metadata=transaction_metadata('paper-atcc',retry_count=retry,policy_commit_batch=False)
            metadata.update(workload='controlled_phase_pipeline',task_type='ycsb')
            txn=manager.begin(f'pipeline:{index}',metadata,strategy='paper-atcc')
            paid=[];reason='';committed=False
            def pause(name,duration):
                txn.check_active()
                # Opt-in synthetic caller budget, not measured future latency.
                budget=math.ceil(duration) if getattr(manager,'policy_checkpoint_min_interval_ms',0.)>0 else 1
                manager.interceptor.expensive_operation_checkpoint(txn,budget)
                tick=time.perf_counter()
                with tracked_agent_wait(txn):
                    time.sleep(duration/1000.)
                paid.append(dict(name=name,actual_ms=(time.perf_counter()-tick)*1000.,requested_ms=duration))
                if not getattr(manager,'track_inflight_agent_cost',False):
                    manager.interceptor.account_agent_interval(txn)
                txn.check_active()
            try:
                txn.enter_phase('explore');pause('prefix',config['prefix_ms'])
                if profile=='refine':txn.enter_phase('refine')
                value=str(txn.read('shift-key:0').value)
                if not root_seen.is_set():
                    arrival_timeline['first_read_returned_ms']=(time.perf_counter()-started)*1000.
                root_seen.set()
                if profile=='explore':
                    pause('cheap',config['cheap_ms']);txn.enter_phase('refine')
                    pause('reason',config['reason_ms'])
                else:
                    pause('reason',config['reason_ms']);pause('cheap',config['cheap_ms'])
                txn.write('shift-key:31',value)
                result=txn.commit('paper-atcc');committed=bool(result.committed);reason=str(result.reason)
            except LockConflict as exc:
                if txn.result is None:txn.abort(exc.reason,strategy='paper-atcc',conflict_object_ids=exc.targets)
                reason=str(exc.reason)
            except BaseException:
                root_seen.set()
                if txn.result is None:txn.abort('pipeline diagnostic execution error',strategy='paper-atcc')
                raise
            attempts.append(dict(tid=txn.context.tid,committed=committed,reason=reason,paid=paid,
                action_changes=sum(e.kind=='action-change' for e in txn.events),
                prefix_validation_calls=txn.metadata.get('_prefix_validation_calls',0),
                prefix_validation_events=[dict(e.detail) for e in txn.events if e.kind=='prefix-validation'],
                policy_checkpoint_budget_events=[dict(e.detail) for e in txn.events if e.kind=='policy-checkpoint-budget'],
                item_protection_changes=sum(e.kind=='item-protection-change' for e in txn.events)))
            if committed:break
            if retry+1<config['max_attempts']:time.sleep(config['backoff_ms']/1000.)
        latency_ms=(time.perf_counter()-task_start)*1000.
        root.__exit__(None,None,None)
        return dict(role='foreground',committed=committed,attempts=attempts,latency_ms=latency_ms)
    def writer():
        if not root_seen.wait(10.):raise ValueError('Foreground did not observe root')
        time.sleep(config['writer_arrival_after_read_ms']/1000.)
        arrival_timeline['writer_begin_ms']=(time.perf_counter()-started)*1000.
        arrival_timeline['requested_ms']=config['writer_arrival_after_read_ms']
        arrival_timeline['observed_ms']=arrival_timeline['writer_begin_ms']-arrival_timeline['first_read_returned_ms']
        task_start=time.perf_counter()
        clock=getattr(manager,'pipeline_profile_clock',None)
        root=clock.scope('other',root=True) if clock else contextlib.nullcontext()
        root.__enter__()
        attempts=[];committed=False;published=None;dependent=0.
        writer_budget=config.get('writer_max_attempts',1)
        for retry in range(writer_budget):
            if config.get('agent_writer',False):
                metadata=transaction_metadata('paper-atcc',retry_count=retry,policy_commit_batch=False)
                metadata.update(pipeline_fixed_writer=True,workload='controlled_phase_pipeline',task_type='ycsb')
            else:
                metadata=dict(runtime_background=True,paper_atcc_backend=True)
            txn=manager.begin(f'pipeline-writer:{index}',metadata,strategy='paper-atcc');paid=[];reason=''
            try:
                old=int(txn.read('shift-key:0').value)
                tick=time.perf_counter();time.sleep(config['writer_work_ms']/1000.)
                duration=(time.perf_counter()-tick)*1000.;paid.append(dict(name='writer',actual_ms=duration))
                if config.get('agent_writer',False):manager.interceptor.account_agent_interval(txn)
                else:txn.context.agent_cost_ms+=duration
                manager.refresh_atcc_priority(txn)
                txn.write('shift-key:0',str(old+1))
                result=txn.commit('paper-atcc');committed=bool(result.committed);reason=str(result.reason)
            except LockConflict as exc:
                if txn.result is None:txn.abort(exc.reason,strategy='paper-atcc',conflict_object_ids=exc.targets)
                reason=str(exc.reason)
            attempts.append(dict(tid=txn.context.tid,committed=committed,reason=reason,paid=paid,action_changes=0))
            if committed:
                published=(time.perf_counter()-started)*1000.
                tick=time.perf_counter();time.sleep(config['dependent_tool_ms']/1000.)
                dependent=(time.perf_counter()-tick)*1000.
                break
            if retry+1<writer_budget:time.sleep(config['backoff_ms']/1000.)
        if not committed and not allow_incomplete:raise ValueError('Controlled writer did not commit: '+reason)
        latency_ms=(time.perf_counter()-task_start)*1000.
        root.__exit__(None,None,None)
        return dict(role='writer_pipeline',committed=committed,attempts=attempts,
            latency_ms=latency_ms,published_ms=published,dependent_tool_ms=dependent)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fg=pool.submit(foreground);bg=pool.submit(writer)
        tasks=[fg.result(timeout=config.get('future_timeout_seconds',15.)),bg.result(timeout=config.get('future_timeout_seconds',15.))]
    elapsed=time.perf_counter()-started
    history=manager.history() if audit_at_end else None
    oracle=verify_history(**history) if audit_at_end else dict(verdict='deferred-global-audit')
    if (audit_at_end and oracle['verdict']!='verified') or (not allow_incomplete and not all(t['committed'] for t in tasks)):raise ValueError('Pipeline failed')
    attempts=[a for t in tasks for a in t['attempts']]
    return dict(profile=profile,executed_config=dict(config),arrival_timeline=arrival_timeline,
        tasks=tasks,history=history,oracle=oracle,decisions=recording.records if recording else [],
        summary=dict(commits=sum(t['committed'] for t in tasks),elapsed_seconds=elapsed,task_tps=sum(t['committed'] for t in tasks)/elapsed,
            mean_logical_ms=sum(t['latency_ms'] for t in tasks)/2,
            foreground_ms=tasks[0]['latency_ms'],writer_pipeline_ms=tasks[1]['latency_ms'],
            writer_published_ms=tasks[1]['published_ms'],aborts=sum(not a['committed'] for a in attempts),
            aborted_gap_ms=sum(p['actual_ms'] for a in attempts if not a['committed'] for p in a['paid']),
            policy_calls=len(recording.records) if recording else 0,
            action_changes=sum(a['action_changes'] for a in attempts),
            item_protection_changes=sum(a.get('item_protection_changes',0) for a in attempts)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--agent-writer',action='store_true',
        help='Use a fixed-policy short agent writer, rather than a version-retaining backend publisher')
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.audit_only:
        audit(args.output);return
    args.output.mkdir(parents=True,exist_ok=False)
    training=read(args.training/'protocol.json')
    config=dict(CONFIG,agent_writer=args.agent_writer)
    plan=dict(config=config,arms=ARMS,profiles=['refine','explore'],
        source_sha256=sha(Path(__file__)),training=str(args.training.resolve()),
        scope='Controlled two-workflow pipeline diagnostic; old learned models transfer only, no retraining',
        arrival='writer begins a fixed 4ms after the first returned foreground read, shared by all policies',
        dependent_tool='writer-dependent 200ms work runs after its transaction commits, included in logical task latency/TPS',
        new_api_calls=0,not_original_ycsb_or_real_trace=True)
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    models={a:Probe(a) for a in ARMS[:4]}
    for arm in ('full','nophase'):
        table=load_pure_table(args.training/'compilation/audit.json',args.training,arm,4)[0]
        table.prebucketed_lookup=True;table.packed_fallback_lookup=True;table.prepare_lookup_indexes()
        models[arm+'_table']=table
    core=load_cast_core();rows=[]
    for i,profile in enumerate(plan['profiles']):
        order=ARMS[i:]+ARMS[:i]
        for arm in order:
            gc.collect();before=len(AuditedManager.instances)
            recording=RecordingPolicy(models[arm]);manager=make_manager(core,training['runtime'],recording)
            if args.agent_writer:manager.interceptor.hooks=FixedWriterHooks(manager)
            r=execute_pair(manager,recording,profile,i,config)
            r['summary'].update(arm=arm,profile=profile)
            r['lock_diagnostics']=manager.atcc_locks.snapshot_diagnostics()
            AuditedManager.instances.remove(manager)
            r['manager_registry']=dict(before=before,after=len(AuditedManager.instances))
            if r['manager_registry']!={'before':0,'after':0}:raise ValueError('Manager retained')
            (args.output/f'{profile}_{arm}.json').write_text(json.dumps(r,indent=2))
            rows.append(r['summary']);manager=None;recording=None
            print(json.dumps(rows[-1]),flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2))
    audit(args.output)


def audit(output):
    """Reconstruct saved diagnostic counters; do not rerun wall-clock work."""
    plan=read(output/'protocol.json')
    if sha(output/'runner_executed.py')!=plan['source_sha256']:
        raise ValueError('Executed diagnostic source snapshot differs')
    models={a:Probe(a) for a in ARMS[:4]}
    training=Path(plan['training'])
    for arm in ('full','nophase'):
        models[arm+'_table']=load_pure_table(training/'compilation/audit.json',training,arm,4)[0]
    rows=[];checks=[]
    for profile in plan['profiles']:
        for arm in plan['arms']:
            r=read(output/f'{profile}_{arm}.json');s=r['summary'];tasks=r['tasks']
            if verify_history(**r['history'])!=r['oracle'] or r['oracle']['verdict']!='verified':
                raise ValueError('Invalid committed serial history')
            if r['manager_registry']!={'before':0,'after':0}:raise ValueError('Manager leak')
            attempts=[a for task in tasks for a in task['attempts']]
            expected=dict(commits=sum(t['committed'] for t in tasks),task_tps=2/s['elapsed_seconds'],
                mean_logical_ms=sum(t['latency_ms'] for t in tasks)/2,
                foreground_ms=tasks[0]['latency_ms'],writer_pipeline_ms=tasks[1]['latency_ms'],
                writer_published_ms=tasks[1]['published_ms'],
                aborts=sum(not a['committed'] for a in attempts),
                aborted_gap_ms=sum(p['actual_ms'] for a in attempts if not a['committed'] for p in a['paid']),
                policy_calls=len(r['decisions']),action_changes=sum(a['action_changes'] for a in attempts))
            if any(s[k]!=v for k,v in expected.items()):raise ValueError('Counter differs')
            for a in tasks[0]['attempts']:
                ordered=['prefix','cheap','reason'] if profile=='explore' else ['prefix','reason','cheap']
                observed=[p['name'] for p in a['paid']]
                if observed!=ordered[:len(observed)]:raise ValueError('Wait ordering differs')
                for pause in a['paid']:
                    name=pause['name'];key='reason_ms' if name=='reason' else name+'_ms'
                    if pause['requested_ms']!=plan['config'][key]:raise ValueError('Requested wait differs')
            for d in r['decisions']:
                from agent.runtime.state_collector import phase_aware_state_from_dict
                state=phase_aware_state_from_dict(d['state'])
                if models[arm].select(state).encoded!=d['action']:raise ValueError('Action replay differs')
                if d['action']&state.current_action!=state.current_action:raise ValueError('Released protection')
            rows.append(s);checks.append(dict(arm=arm,profile=profile,oracle='verified',queries=len(r['decisions'])))
    if sorted(rows,key=lambda r:(r['profile'],r['arm']))!=sorted(read(output/'results.json'),key=lambda r:(r['profile'],r['arm'])):
        raise ValueError('Saved aggregate differs')
    with (output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregate=[]
    for arm in plan['arms']:
        selected=[r for r in rows if r['arm']==arm]
        elapsed=sum(r['elapsed_seconds'] for r in selected)
        aggregate.append(dict(arm=arm,logical_tasks=2*len(selected),elapsed_seconds=elapsed,
            task_tps=2*len(selected)/elapsed,mean_logical_ms=sum(r['mean_logical_ms'] for r in selected)/len(selected),
            aborts=sum(r['aborts'] for r in selected),aborted_gap_ms=sum(r['aborted_gap_ms'] for r in selected)))
    with (output/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    proof=dict(passed=True,checks=checks,executed_source_verified=True,
        auditor_sha256=sha(Path(__file__)),training_policy_sha256={a:sha(training/f'{a}_policy.json') for a in ('full','nophase')},
        scope='Offline history, action, input wait order, counter and manager lifecycle reconstruction; not a timing repeat')
    (output/'independent_audit.json').write_text(json.dumps(proof,indent=2))
    (output/'auditor_executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(audit_passed=True,episodes=len(rows),aggregate=aggregate)),flush=True)


if __name__=='__main__':main()
