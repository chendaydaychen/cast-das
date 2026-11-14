"""Current frozen deployment transferred to captured natural CreditCrew.

Historical two-task source remains the default; an explicit capture selects
new independent tasks. No repeated synthetic copies and no new LLM calls.
This is a realism/low-contention overhead control, not scaling.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agent.runtime import AgentTransactionManager
from agent.runtime.paper_contract import configure_paper_observed_contract
from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
from agent.runtime.single_writer_grace_20261109 import install
from scripts.revision.progressive_agent_environment_20261109 import ObservedBoundaryHooks
from scripts.revision.run_revision_supplement_20261110 import FrozenStatic
from scripts.revision import compare_frozen_credit_trace as replay
from scripts.revision.bridge_profile import ExclusiveClock,instrument_manager
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_audited_credit_pressure import AuditedManager

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--time-scale',type=float,default=1.)
    p.add_argument('--capture',type=Path,default=ROOT/'revision_experiments/real_credit_live_small_20261102.json')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-retries',type=int,default=1)
    p.add_argument('--storage', choices=['native', 'durable'], default='native',
                   help='Native replay or SQLite business records plus atomic JSON audit records')
    p.add_argument('--audit-storage', action='store_true',
                   help='Verify committed history and cross-store writes on a multi-storage capture')
    p.add_argument('--model-dir', type=Path,
                   default=ROOT/'revision_experiments/progressive_agent_final_20261109/selected_models')
    p.add_argument('--arms',nargs='+',default=['full','nophase','static','silo','2pl']);args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);capture=args.capture;source=json.loads(capture.read_text())
    if not source['tasks'] or not all(t['committed'] for t in source['tasks']):raise ValueError('Completed captured tasks required')
    audit_storage = args.audit_storage or args.storage == 'durable'
    if audit_storage and not source.get('multi_storage'):
        p.error('Storage auditing requires a multi-storage capture')
    original=replay.RecordingPolicy
    class Recorder(original):
        def __getattr__(self,name):return getattr(self.policy,name)
    replay.RecordingPolicy=Recorder;rows=[]
    for arm in args.arms:
        model=args.model_dir/('nophase_int16_table.json' if arm=='nophase' else 'full_int16_table.json')
        policy=FrozenStatic() if arm=='static' else QuantizedSharedBitTable(json.loads(model.read_text())) if arm in ['full','nophase'] else None
        clock=ExclusiveClock();profile={};attempts=[];decisions=[];refs={};changes=[]
        system={'silo':'silo','2pl':'2pl-nowait'}.get(arm,'paper-atcc')
        if arm=='2pl':
            from scripts.unified_trace.run_corrected_tail_matrix_20261106 import OperationTwoPhaseManager
            Base=OperationTwoPhaseManager
        else:Base=AgentTransactionManager
        class AuditedBase(AuditedManager, Base):
            pass

        durable = None
        if args.storage == 'durable':
            from agent.runtime.durable_storage import DurableRoutedVersionedKVStore
            durable = DurableRoutedVersionedKVStore(args.output/('storage_'+arm))

        if audit_storage:
            Base = AuditedBase
        class Manager(Base):
            def __init__(self,**kwargs):
                if durable is not None:
                    kwargs.pop('storage_engines')
                    kwargs['store'] = durable
                super().__init__(**kwargs);refs['manager']=self
                configure_paper_observed_contract(self,explicit_phase_label_ablation=True)
                self.priority_manager.config=dataclasses.replace(self.priority_manager.config,blocked_quantum_ms=10.)
                self.interceptor.hooks._action_transition_timeout_s=lambda tx:120.
                for name in ['wlock','validate_and_rlock','validate_and_wlock','acquire_write_set']:
                    method=getattr(self.atcc_locks,name)
                    def budget(*v,_method=method,**kw):kw.setdefault('timeout_s',120.);return _method(*v,**kw)
                    setattr(self.atcc_locks,name,budget)
                self.interceptor.hooks=ObservedBoundaryHooks(ObservedWriteReservationHooks(self.interceptor.hooks))
                install(self,4500.,commit_grace_ms=50.);self.paper_deferred_replay_enabled=False;self.atcc_retry_cache_enabled=False
                transition=self.transition_atcc_action
                def measured_change(tx,action,**kw):
                    before=int(tx.context.action.protected);started=time.perf_counter_ns();cpu=time.thread_time_ns()
                    try:return transition(tx,action,**kw)
                    finally:changes.append(dict(task=tx.task_id,actual=before!=int(tx.context.action.protected),wall_ns=time.perf_counter_ns()-started,cpu_ns=time.thread_time_ns()-cpu))
                self.transition_atcc_action=measured_change
                instrument_manager(self,clock,system)
        storage_audit = None
        try:
            result=replay.run_system(source,system,workers=args.workers,max_attempts=args.max_retries+1,time_scale=args.time_scale,repeat_factor=1,policy=policy,
                frozen_agent_schedule=True,manager_factory=Manager,static_protection=arm=='static',profile_clock=clock,profile_sink=profile,
                decision_sink=decisions if policy else None,attempt_sink=attempts,logical_task_feedback=True)
            if audit_storage:
                manager = refs['manager']
                history = manager.history()
                verified = verify_history(history['initial'], history['attempts'], history['final'])
                cross_store_writes = sum(
                    len({manager.store.storage_id_for(key) for key in tx.write_set}) > 1
                    for tx in manager.audit_transactions if tx.result and tx.result.committed
                )
                if not cross_store_writes:
                    raise RuntimeError('Replay did not commit any cross-storage write')
                storage_audit = dict(backend=manager.backend_name,
                                     cross_store_commits=cross_store_writes, history=verified)
                if durable is not None:
                    persisted = {key: durable.get(key) for key in manager._catalog}
        finally:
            if durable is not None:
                durable.close()
        if durable is not None:
            with DurableRoutedVersionedKVStore(args.output/('storage_'+arm)) as reopened:
                if reopened.recovery_report['redone_committed'] or reopened.recovery_report['discarded_prepared']:
                    raise RuntimeError('Unexpected pending decision after orderly shutdown')
                if any(reopened.get(key) != snapshot for key, snapshot in persisted.items()):
                    raise RuntimeError('Stored value or version changed on reopen')
                storage_audit['reopened_objects'] = len(persisted)
        record=dict(arm=arm,result=result,profile=profile,attempts=attempts,decisions=decisions,changes=changes,
            storage_audit=storage_audit,
            policy_sha256=hashlib.sha256(model.read_bytes()).hexdigest() if arm in ['full','nophase'] else None)
        (args.output/f'{arm}.json').write_text(json.dumps(record,indent=2));rows.append(record)
        print(json.dumps(dict(arm=arm,tps=result['throughput_tps'],completed=result['commits'],tasks=result['tasks'])),flush=True)
        if result['commits']!=len(source['tasks']):raise RuntimeError('Captured task replay incomplete')
    (args.output/'completion.json').write_text(json.dumps(dict(complete=True,new_api_calls=0,independent_captured_tasks=len(source['tasks']),time_scale=args.time_scale,
        workers=args.workers,max_retries=args.max_retries,
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),arms=args.arms,
        scope='Natural captured tasks replayed with frozen current policy and original delays; no new agent paths, no contention scaling proof'),indent=2))
    replay.RecordingPolicy=original

if __name__=='__main__':main()
