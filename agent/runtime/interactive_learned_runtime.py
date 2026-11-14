"""Interactive learned-CC execution substrate, not a trained NeurCC baseline yet.

Clean point reads, per-operation none/critical/all detection, learned priority
and timeout, early validation for critical detection, final atomic validation.
No dirty reads, exposed writes, partial retry, graph optimizer or model importer
is silently emulated. Those unsupported choices fail explicitly. Python metadata
condition replaces the original per-tuple native structures; fixed volatile KV
catalog only. All participants must use one manager.

Protocol reference: https://arxiv.org/html/2503.10036v1, interactive mode §4.5.
Pinned native reference uses microseconds in txn_impl.h::do_wait, despite a
conflicting print label; timeout_us below is explicit and never rescaled.
"""
from collections import Counter
from dataclasses import dataclass, field, asdict
import math
import threading
import time

from agent.cc.locks import LockConflict
from .transaction import AgentTransactionManager
from .operation_interceptor import NoopTransactionHooks
from .types import SnapshotValue, TransactionState
from .context import TransactionStatus
from .baseline_storage_contract import validate_volatile_baseline_storage


@dataclass(frozen=True)
class InteractiveObservation:
    transaction_type: int
    operation_type: str
    completed_operations: int
    blocked_on: int
    blocking: int
    times_blocked: int


@dataclass(frozen=True)
class InteractiveAction:
    detection: str = 'none'
    priority: float = 0.
    timeout_us: int = 100000
    expose: bool = False

    def __post_init__(self):
        if self.detection not in ('none','critical','all'):
            raise ValueError('Interactive detection must be none/critical/all')
        if isinstance(self.priority,bool) or not math.isfinite(self.priority) or not 0<=self.priority<=1:
            raise ValueError('Priority must be finite in [0,1]')
        if type(self.timeout_us) is not int or not 0<=self.timeout_us<=4294967295:
            raise ValueError('Explicit nonnegative uint32 microseconds required')
        if type(self.expose) is not bool or self.expose:
            raise ValueError('Interactive exposed writes are unsupported; do not silently mask them')


@dataclass
class InteractiveAttempt:
    transaction_type: int
    state: str = 'active'
    waiting_for: set = field(default_factory=set)
    times_blocked: int = 0


class InteractiveHooks(NoopTransactionHooks):
    def __init__(self,manager):self.manager=manager
    def on_abort(self,txn,reason):
        with self.manager.learned_condition:self.manager._retire(txn.context.tid,'aborted')


class InteractiveLearnedRuntimeManager(AgentTransactionManager):
    """Explicit operation-policy protocol; no Aegis mask/priority decisions."""
    protocol_name='interactive-learned-reference'

    def __init__(self, *, interactive_policy, retry_backoff_policy=None, **kwargs):
        if not callable(getattr(interactive_policy,'select',None)):
            raise ValueError('Explicit operation policy required')
        if retry_backoff_policy is not None and not callable(retry_backoff_policy):
            raise ValueError('Retry backoff must be an explicit callable, or disabled')
        for option in ('transaction_hooks','undo_log_path'):
            if kwargs.get(option) is not None:raise ValueError('Unsupported interactive option: '+option)
        kwargs.update(transaction_hooks=NoopTransactionHooks(),priority_enabled=False,
            performance_guards_enabled=False,collect_trajectories=False)
        super().__init__(**kwargs)
        validate_volatile_baseline_storage(self.store)
        self.interactive_policy=interactive_policy
        self.retry_backoff_policy=retry_backoff_policy
        self.learned_condition=threading.Condition(threading.RLock())
        self.learned_attempts={};self.learned_accesses={};self.learned_versions={}
        self.learned_metrics=Counter();self.learned_events=[];self.learned_poisoned=False
        self.interceptor.hooks=InteractiveHooks(self)

    def _healthy(self):
        if self.learned_poisoned:raise RuntimeError('Interactive publication metadata poisoned')

    def _current(self,key):
        self._healthy();value=self.store.get(key)
        if int(value.version)!=self.learned_versions[key]:
            self.learned_poisoned=True
            raise RuntimeError('External write bypassed interactive metadata: '+key)
        return value

    def register_object(self,key,value,**kwargs):
        with self.learned_condition:
            self._healthy()
            if self.learned_attempts:raise ValueError('Fixed catalog must precede all attempts')
            super().register_object(key,value,**kwargs)
            self.learned_versions[str(key)]=int(self.store.get(str(key)).version)

    def register_empty_object(self,*args,**kwargs):
        raise ValueError('No insert/delete/range implementation')

    def _retire(self,tid,state):
        row=self.learned_attempts.get(tid)
        if row is None:return
        if row.state=='committed' and state!='committed':raise RuntimeError('Cannot abort committed metadata')
        row.state=state;row.waiting_for.clear()
        for key in tuple(self.learned_accesses):
            keep=[op for op in self.learned_accesses[key] if op['tid']!=tid]
            if keep:self.learned_accesses[key]=keep
            else:self.learned_accesses.pop(key)
        self.learned_events.append(dict(kind='terminal',tid=tid,state=state,at_ns=time.monotonic_ns()))
        self.learned_condition.notify_all()

    def _validate_reads(self,txn):
        return tuple(key for key,read in txn.read_set.items() if int(self._current(key).version)!=read.version)

    def _cycle(self,start):
        pending=list(self.learned_attempts[start].waiting_for);visited=set()
        while pending:
            tid=pending.pop()
            if tid==start:return True
            if tid in visited:continue
            visited.add(tid)
            if tid in self.learned_attempts:pending.extend(self.learned_attempts[tid].waiting_for)
        return False

    def _before(self,txn,key,kind):
        tid=txn.context.tid;row=self.learned_attempts[tid]
        state=InteractiveObservation(row.transaction_type,kind,txn.context.completed_operations,
            len(row.waiting_for),sum(tid in other.waiting_for for other in self.learned_attempts.values()),
            row.times_blocked)
        action=self.interactive_policy.select(state)
        if not isinstance(action,InteractiveAction):raise ValueError('Wrong learned operation action schema')
        self.learned_metrics['policy_calls']+=1
        self.learned_events.append(dict(kind='decision',tid=tid,key=key,operation=kind,
            observation=asdict(state),action=asdict(action),at_ns=time.monotonic_ns()))
        if action.detection=='critical':
            self.learned_metrics['early_validations']+=1
            stale=self._validate_reads(txn)
            if stale:raise LockConflict('Interactive early version conflict',stale,kind='version-conflict')
        deadline=time.monotonic()+action.timeout_us/1e6;started=None
        try:
            while action.detection!='none':
                txn._ensure_active();self._healthy()
                blockers={op['tid'] for op in self.learned_accesses.get(key,())
                    if op['tid']!=tid and (kind=='write' or op['kind']=='write')
                    and self.learned_attempts[op['tid']].state=='active'
                    and (action.detection=='all' or op['priority']>=action.priority)}
                row.waiting_for=blockers
                if not blockers:break
                if started is None:
                    started=time.monotonic();row.times_blocked+=1
                    self.learned_metrics['wait_calls']+=1
                self.learned_events.append(dict(kind='wait',tid=tid,key=key,blockers=sorted(blockers),
                    at_ns=time.monotonic_ns()))
                if self._cycle(tid):
                    self.learned_metrics['deadlocks']+=1
                    raise LockConflict('Interactive lock dependency cycle',(key,),kind='lock-conflict')
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    self.learned_metrics['timeouts']+=1
                    raise LockConflict('Interactive lock timeout',(key,),kind='lock-timeout')
                self.learned_condition.wait(remaining)
        finally:
            row.waiting_for.clear()
            if started is not None:
                milliseconds=(time.monotonic()-started)*1000
                txn.context.blocked_time_ms+=milliseconds
                self.learned_metrics['wait_ms']+=milliseconds
        return action

    def begin(self,task_id,metadata=None,*,snapshot_object_ids=None,strategy='silo'):
        if strategy not in ('silo',self.protocol_name):raise ValueError('Do not mix CC protocols')
        metadata=dict(metadata or {})
        if any(metadata.get(k) for k in ('paper_atcc','paper_atcc_backend','retry_protection_mask')):
            raise ValueError('Interactive protocol cannot inherit Aegis protection')
        txn_type=metadata.get('learned_transaction_type',1)
        if type(txn_type) is not int or txn_type<1:raise ValueError('Positive transaction type required')
        retry=metadata.get('retry_count',0)
        if type(retry) is not int or retry<0:raise ValueError('Nonnegative retry count required')
        backoff=0
        if retry and self.retry_backoff_policy is not None:
            backoff=self.retry_backoff_policy(retry_count=retry,transaction_type=txn_type)
            if type(backoff) is not int or not 0<=backoff<=4294967295:
                raise ValueError('Retry backoff must be explicit uint32 microseconds')
        # Caller task latency and owner journal begin include this cost. Never
        # hold a protocol latch over sleep or silently add a tuned default.
        if backoff:
            started=time.monotonic_ns();time.sleep(backoff/1e6)
            metadata['interactive_retry_backoff']=dict(requested_us=backoff,
                start_ns=started,end_ns=time.monotonic_ns())
        with self.learned_condition:
            self._healthy()
            txn=super().begin(task_id,metadata,snapshot_object_ids=None,strategy='silo')
            self.learned_attempts[txn.context.tid]=InteractiveAttempt(txn_type)
            if backoff:self.learned_metrics['retry_backoff_calls']+=1
        original_read,original_write,original_abort=txn.read,txn.write,txn.abort
        def access(kind,key,value=None,**options):
            key=str(key)
            with txn._lifecycle_lock,self.learned_condition:
                txn._ensure_active();self._healthy()
                if key not in self.learned_versions:raise KeyError(key)
                action=self._before(txn,key,kind)
                current=self._current(key)
                if key not in txn.read_set and key not in txn.write_set:
                    txn.snapshot[key]=SnapshotValue(str(current.value),int(current.version),bool(current.exists))
                result=original_read(key,**options) if kind=='read' else original_write(key,value,**options)
                self.learned_accesses.setdefault(key,[]).append(dict(tid=txn.context.tid,kind=kind,
                    priority=action.priority,index=txn.context.completed_operations))
                self.learned_condition.notify_all()
                return result
        def abort(reason,*,strategy='',conflict_object_ids=()):
            if txn.result is not None:
                if txn.result.committed:raise RuntimeError('Cannot abort committed interactive transaction')
                return txn.result
            if txn.metadata.get('interactive_publication_fault',{}).get('outcome')=='indeterminate':
                raise RuntimeError('Interactive publication outcome indeterminate; manager poisoned')
            return original_abort(reason,strategy=self.protocol_name,conflict_object_ids=conflict_object_ids)
        txn.read=lambda key,**options:access('read',key,**options)
        txn.write=lambda key,value,**options:access('write',key,value,**options)
        txn.abort=abort
        return txn

    def commit(self,txn,*,strategy='silo'):
        if txn.manager is not self or strategy not in ('silo',self.protocol_name):raise ValueError('Wrong interactive CC')
        with self.learned_condition:
            self._healthy();txn._ensure_active();stale=self._validate_reads(txn)
            if stale:
                self.learned_metrics['validation_aborts']+=1
                return self._finish_abort(txn,strategy=self.protocol_name,reason='Interactive final version conflict',
                    conflict_object_ids=stale,lock_wait_s=0.)
            latest={key:int(self._current(key).version) for key in txn.write_set}
            txn.context.transition(TransactionStatus.COMMITTING)
            try:
                installed=self.store.batch_put_if_version(list(latest.items()),
                    [(key,w.value) for key,w in txn.write_set.items()])
                if not installed:raise RuntimeError('External publisher bypassed interactive validation')
                fresh={key:int(self.store.get_version(key)) for key in txn.write_set}
                if any(fresh[key]!=latest[key]+1 for key in fresh):raise RuntimeError('Unexpected version increment')
            except BaseException as exc:
                self.learned_poisoned=True
                try:unchanged=all(int(self.store.get_version(key))==version for key,version in latest.items())
                except BaseException:unchanged=False
                txn.metadata['interactive_publication_fault']=dict(error_type=type(exc).__name__,
                    outcome='not-published' if unchanged else 'indeterminate')
                if unchanged:
                    self._finish_abort(txn,strategy=self.protocol_name,reason='Interactive publication rejected before write',
                        conflict_object_ids=tuple(latest),lock_wait_s=0.)
                    raise
                # A backend may throw after installation. Do not invent an
                # abort or rollback: stop the manager and retain unresolved
                # outcome evidence for diagnosis/recovery outside this scope.
                self._retire(txn.context.tid,'indeterminate')
                with self._lock:self._live_transactions.pop(txn.context.tid,None)
                raise RuntimeError('Interactive publication outcome indeterminate; manager poisoned') from exc
            self.learned_versions.update(fresh);txn.metadata['runtime_published_versions']=fresh
            txn.state=TransactionState.COMMITTED;txn.context.transition(TransactionStatus.COMMITTED)
            txn._event('finish',dict(state='committed',protocol=self.protocol_name))
            txn.result=txn._result(strategy=self.protocol_name,committed=True,action='commit')
            self._retire(txn.context.tid,'committed');self.learned_metrics['commits']+=1
            self._note_runtime_outcome(txn,True);self._record(txn);self.interceptor.finish(txn)
            with self._lock:
                self._native_dirty_objects.update(txn.write_set);self._live_transactions.pop(txn.context.tid,None)
            self._finalize_commit_timing(txn)
            return txn.result
