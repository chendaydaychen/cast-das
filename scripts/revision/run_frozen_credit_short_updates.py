"""Exploratory real-agent replay plus a predeclared time-based short-update stream.

Short updates are action-0 OCC transactions in the same manager and common commit
path. They do not bypass Aegis publication locks or use native background reader
bypass. All outcomes and the combined committed history are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc import LockConflict
from agent.runtime.operation_interceptor import NoopTransactionHooks
from agent.runtime.residual_policy import ResidualTablePolicy
from agent.runtime.paper_policy import HotRulePolicy,CompiledPhasePolicy
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from scripts.revision.compare_frozen_credit_trace import run_system
from scripts.revision.run_audited_credit_pressure import AuditedManager,LinearScorePolicy
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.merge_credit_evaluation_captures import load_merged
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import validate_inputs
from scripts.revision.audit_serial_history import verify_history

ARMS=('full','nophase','linear','hot_always','item_threshold3','silo')


class ShortUpdateHooks:
    """Only external short updates skip the foreground policy; normal OCC remains."""
    def __init__(self,foreground):
        self.foreground=foreground
        self.short=NoopTransactionHooks()

    def __getattr__(self,name):
        foreground=getattr(self.foreground,name)
        short=getattr(self.short,name)
        def call(txn,*args,**kwargs):
            return (short if txn.metadata.get('external_short_update') else foreground)(txn,*args,**kwargs)
        return call


class MixedManager(AuditedManager):
    def __init__(self,*args,arrival_seconds,update_key,deadline_seconds=40.,
                 short_max_attempts=4001,**kwargs):
        if not math.isfinite(deadline_seconds) or deadline_seconds<=0:
            raise ValueError('Positive finite short-update retry horizon required')
        super().__init__(*args,**kwargs)
        self.short_deadline_seconds=float(deadline_seconds)
        if type(short_max_attempts) is not int or short_max_attempts < 1:
            raise ValueError('Positive integer short-update attempt budget required')
        self.short_max_attempts=short_max_attempts
        self.arrival_seconds=arrival_seconds
        self.update_key=update_key
        self.epoch=None
        self.stream=None
        self.stream_lock=threading.Lock()
        self.short_records=[]
        self.short_errors=[]
        self.carry_short_retry_wait=False

    def begin(self,*args,**kwargs):
        txn=super().begin(*args,**kwargs)
        if not txn.metadata.get('external_short_update'):
            with self.stream_lock:
                if self.stream is None:
                    self.epoch=time.perf_counter()
                    self.stream=threading.Thread(target=self._short_stream,name='fixed-short-updates')
                    self.stream.start()
        return txn

    def _short_stream(self):
        try:
            with ThreadPoolExecutor(max_workers=len(self.arrival_seconds)) as pool:
                futures=[pool.submit(self._one_update,i,at) for i,at in enumerate(self.arrival_seconds)]
                for future in futures:
                    self.short_records.append(future.result())
        except BaseException as exc:
            self.short_errors.append(repr(exc))

    def _one_update(self,index,arrival,*,update_key=None,strategy='paper-atcc'):
        # A per-call target supports independent rule updates without mutating
        # shared manager state while multiple update threads are active.
        key=self.update_key if update_key is None else str(update_key)
        remaining=self.epoch+arrival-time.perf_counter()
        if remaining>0:
            time.sleep(remaining)
        started=time.perf_counter()
        attempts=[]
        # A declared wall-time deadline and backoff prevent unbounded busy retry.
        deadline=started+self.short_deadline_seconds
        prior_blocked_ms=0.
        for retry in range(self.short_max_attempts):
            txn=self.begin(f'short-update:{index}',dict(strategy=strategy,
                external_short_update=True,retry_count=retry,workload='credit_short_update',
                prior_retry_blocked_ms=prior_blocked_ms if self.carry_short_retry_wait else 0.,
                access_set_visibility='online_observed'),strategy=strategy)
            try:
                current=txn.read(key).value
                # The KV fixture stores decimal integers as strings, as do
                # captured agent writes; preserve that representation.
                if not isinstance(current,str) or str(int(current))!=current:
                    raise ValueError('Short update requires the decimal credit-limit fixture')
                txn.write(key,str(int(current)+1))
                outcome=txn.commit(strategy)
                committed,reason=outcome.committed,outcome.reason
            except LockConflict as exc:
                if txn.result is None:
                    txn.abort(exc.reason,strategy=strategy,conflict_object_ids=exc.targets)
                committed,reason=False,exc.reason
            except BaseException:
                if txn.result is None:
                    txn.abort('short-update execution error',strategy=strategy)
                raise
            attempts.append(dict(committed=bool(committed),reason=reason,
                elapsed_since_arrival_s=time.perf_counter()-started,
                attempt_blocked_ms=txn.context.blocked_time_ms,
                prior_retry_blocked_ms=txn.context.prior_retry_blocked_ms,priority=txn.context.priority))
            prior_blocked_ms+=txn.context.blocked_time_ms
            if committed or time.perf_counter()>=deadline:
                break
            getattr(self, 'short_retry_sleep', time.sleep)(.01)
        return dict(index=index,key=key,scheduled_arrival_seconds=arrival,
            observed_start_seconds=started-self.epoch,
            completion_seconds=time.perf_counter()-self.epoch,
            elapsed_seconds=time.perf_counter()-started,committed=bool(committed),attempts=attempts)

    def join_short_updates(self):
        if self.stream is None:
            raise ValueError('No foreground execution started the declared stream')
        self.stream.join()
        if self.short_errors:
            raise RuntimeError('Short-update stream failed: '+str(self.short_errors))
        if len(self.short_records)!=len(self.arrival_seconds):
            raise ValueError('Missing scheduled short updates')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture-folder','training-folder','merged-evaluation-folder','frequency-calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--company',type=int,required=True)
    p.add_argument('--arrivals',type=float,nargs='+',required=True)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    args=p.parse_args()
    if (any(at<0 for at in args.arrivals) or sorted(set(args.arrivals))!=args.arrivals
            or len(set(args.arms))!=len(args.arms)):
        p.error('Unique increasing nonnegative arrivals and distinct arms required')
    original,_=validate_inputs(args.capture_folder,args.training_folder)
    merged,identity=load_merged(args.merged_evaluation_folder,args.training_folder,args.capture_folder)
    source=copy.deepcopy(merged)
    selected=[i for i,t in enumerate(source['tasks']) if t['task']['company_id']==args.company]
    source['tasks']=[source['tasks'][i] for i in selected]
    if len(source['tasks'])!=2:
        raise ValueError('This pilot declares exactly two real agent intents')
    key=f'credit:company:{args.company}:limit'
    calibration=json.loads(args.frequency_calibration.read_text())
    window=float(calibration['selected_window_ms'])
    args.output.mkdir(parents=True,exist_ok=False)
    sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__),ROOT/'scripts/revision/compare_frozen_credit_trace.py',
            ROOT/'scripts/revision/run_audited_credit_pressure.py',
            ROOT/'scripts/revision/run_agent_schedule_comparison_20261012.py',
            ROOT/'agent/runtime/transaction.py',ROOT/'agent/runtime/paper_hooks.py',
            ROOT/'agent/runtime/atcc_lock_manager.py',ROOT/'agent/runtime/item_frequency_hooks.py',
            ROOT/'agent/runtime/hot_access_hooks.py',ROOT/'agent/runtime/operation_interceptor.py',
            ROOT/'agent/runtime/residual_policy.py',ROOT/'agent/runtime/paper_policy.py',
            ROOT/'agent/runtime/priority.py',ROOT/'agent/workloads/frozen_credit_agent.py')}
    from agent.runtime.joint_mask_policy import load_scoring_policy
    policies={arm:load_scoring_policy(json.loads((args.training_folder/f'{arm}_policy.json').read_text()))
        for arm in ('full','nophase')}
    policies.update(linear=LinearScorePolicy(json.loads((args.training_folder/'linear_policy.json').read_text())['weights']),
        hot_always=HotRulePolicy(),item_threshold3=CompiledPhasePolicy([]),silo=None)
    plan=dict(experiment_role='exploratory mechanism stress, not fresh independent confirmation',
        company=args.company,selected_source_task_indices=selected,merged_input=identity,
        workers=2,repeat_factor=1,evaluation_scale=1.,retrained=False,arms=args.arms,
        short_update_key=key,arrival_seconds=args.arrivals,short_update_deadline_s=40.,
        short_update_max_attempts=4001,short_retry_backoff_s=.01,
        short_cc='action-0 OCC using common Aegis commit admission, no background reader bypass or policy consultation',
        source_protocol=original,source_sha256=sources,
        policy_sha256={arm:hashlib.sha256((args.training_folder/f'{arm}_policy.json').read_bytes()).hexdigest()
            for arm in ('full','nophase','linear')},
        frequency_window_ms=window,calibration_sha256=hashlib.sha256(args.frequency_calibration.read_bytes()).hexdigest(),
        caveat='Actual model-selected agent intents and durations, constructed external updates; no online replanning; no stable population P99')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    (args.output/'input.json').write_text(json.dumps(source,indent=2))
    shutil.copyfile(__file__,args.output/'runner_executed.py')
    results=[]
    for arm in args.arms:
        holder=[]
        def factory(**kwargs):
            manager=MixedManager(**kwargs,arrival_seconds=args.arrivals,update_key=key)
            configure_frozen_runtime(manager,original)
            if arm=='hot_always':
                manager.first_hot_access_events=[]
                manager.interceptor.hooks=FirstHotAccessHooks(manager)
            elif arm=='item_threshold3':
                manager.interceptor.hooks=ItemFrequencyHooks(manager,minimum_accesses=3,window_ms=window)
            manager.interceptor.hooks=ShortUpdateHooks(manager.interceptor.hooks)
            holder.append(manager)
            return manager
        attempts,decisions=[],[]
        system='silo' if arm=='silo' else 'paper-atcc'
        try:
            row=run_system(source,system,workers=2,max_attempts=original['max_attempts'],
                time_scale=1.,repeat_factor=1,policy=policies[arm],
                attempt_sink=attempts,decision_sink=decisions,frozen_agent_schedule=True,
                manager_factory=factory,policy_commit_batch=original['policy_commit_batch'],
                policy_selected_write_admission_only=original['policy_selected_write_admission_only'],
                before_reasoning_checkpoint=original['before_reasoning_checkpoint'],
                carry_prior_restart_cost=original.get('carry_prior_restart_cost',False),
                allow_complete_aborted_capture=True)
        finally:
            if holder:
                holder[0].join_short_updates()
        manager=holder[0]
        history=manager.history()
        oracle=verify_history(**history)
        if oracle['verdict']!='verified':
            raise ValueError('Combined foreground/short-update history failed')
        short=manager.short_records
        result=dict(arm=arm,**row,short_completed=sum(r['committed'] for r in short),
            short_tasks=len(short),short_aborts=sum(not a['committed'] for r in short for a in r['attempts']),
            max_short_elapsed_seconds=max(r['elapsed_seconds'] for r in short),
            total_short_elapsed_seconds=sum(r['elapsed_seconds'] for r in short),
            joint_makespan_seconds=max(row['wall_seconds'],max(r['completion_seconds'] for r in short)),
            oracle_verdict=oracle['verdict'])
        record=dict(result=result,attempts=attempts,decisions=decisions,short_updates=short,
            history=history,oracle=oracle)
        (args.output/f'{arm}_records.json').write_text(json.dumps(record,indent=2))
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2))
        with (args.output/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]))
            writer.writeheader();writer.writerows(results)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
