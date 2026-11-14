"""Shared-runtime exposed-write/dirty-read substrate for learned CC adapters.

This is NOT yet Polyjuice or NeurCC: native guards, encoders, optimizers and
partial-piece rollback are not wired. Decisions are explicit; there is no Aegis
mask fallback. Volatile fixed point catalog, Python metadata latch, single
manager, same atomic KV publication as other runtime adapters.

Exposed values remain in an algorithm-owned access structure. Opaque negative
version IDs stay unchanged when published; physical backend version mappings
are exported separately. Dirty-source completion plus final identity validation
is mandatory regardless of selected policy. No opacity or external-effect claim.
"""
from dataclasses import dataclass,field,asdict
import math
import time
from agent.cc.locks import LockConflict
from .transaction import AgentTransactionManager
from .interactive_learned_runtime import InteractiveAction,InteractiveAttempt,InteractiveLearnedRuntimeManager
from .types import SnapshotValue


@dataclass(frozen=True)
class SpeculativeDecision:
    read_dirty: bool = False
    expose: bool = False
    early_validate: bool = False
    def __post_init__(self):
        if any(type(x) is not bool for x in (self.read_dirty,self.expose,self.early_validate)):
            raise ValueError('Explicit boolean speculative decisions required')


@dataclass
class ValueIdentity:
    key: str
    uid: int
    value: str
    owner: str | None
    published_version: int | None = None
    exposed: bool = False


@dataclass
class SpeculativeAttempt:
    reads: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    dependencies: dict = field(default_factory=dict)  # predecessor -> dirty-source flag
    abort_now: bool = False
    access_index: int = 0


class _NeverQueried:
    def select(self,state):raise AssertionError('No fallback operation policy may be queried')


class SpeculativePolicyRuntimeManager(InteractiveLearnedRuntimeManager):
    protocol_name='speculative-policy-reference'

    def __init__(self,*,speculative_policy,dependency_timeout_s=30.,**kwargs):
        if not callable(getattr(speculative_policy,'select',None)):raise ValueError('Explicit speculative policy required')
        if not math.isfinite(dependency_timeout_s) or dependency_timeout_s<=0:raise ValueError('Positive dependency timeout required')
        if kwargs.get('retry_backoff_policy') is not None:raise ValueError('Use the explicit owner lifecycle for retry control')
        super().__init__(interactive_policy=_NeverQueried(),**kwargs)
        self.speculative_policy=speculative_policy;self.dependency_timeout_s=dependency_timeout_s
        self.spec_attempts={};self.committed_values={};self.exposed_values={};self.value_identities=[];self.next_uid=-1

    def register_object(self,key,value,**kwargs):
        super().register_object(key,value,**kwargs)
        key=str(key);stored=self.store.get(key)
        token=ValueIdentity(key,int(stored.version),str(stored.value),None,int(stored.version))
        self.committed_values[key]=token;self.value_identities.append(token)

    def _live(self,tid):
        return self.learned_attempts[tid].state=='active' and not self.spec_attempts[tid].abort_now

    def _active_spec(self,txn):
        self._healthy();txn._ensure_active();row=self.spec_attempts[txn.context.tid]
        if row.abort_now:raise LockConflict('Speculative dirty dependency aborted',tuple(row.reads),kind='version-conflict')
        return row

    def _abort_now(self,tid):
        pending=[tid];seen=set()
        while pending:
            failed=pending.pop()
            if failed in seen:continue
            seen.add(failed)
            if self.learned_attempts[failed].state=='committed':
                self.learned_poisoned=True;raise RuntimeError('Committed transaction depended on unresolved dirty source')
            self.spec_attempts[failed].abort_now=True
            for child,other in self.spec_attempts.items():
                if other.dependencies.get(failed,False) and self.learned_attempts[child].state=='active':
                    pending.append(child)
                    self.learned_events.append(dict(kind='cascade',source=failed,target=child,at_ns=time.monotonic_ns()))
        self.learned_condition.notify_all()

    def _predecessors_cycle(self,start):
        pending=list(self.spec_attempts[start].dependencies);seen=set()
        while pending:
            tid=pending.pop()
            if tid==start:return True
            if tid in seen or not self._live(tid):continue
            seen.add(tid);pending.extend(self.spec_attempts[tid].dependencies)
        return False

    def _edge(self,predecessor,successor,dirty,actor):
        if predecessor==successor:return
        if not self._live(predecessor):
            if dirty and self.learned_attempts[predecessor].state!='committed':
                self._abort_now(actor)
                raise LockConflict('Speculative dirty dependency aborted',(),kind='version-conflict')
            return
        if not self._live(successor):return
        deps=self.spec_attempts[successor].dependencies
        deps[predecessor]=deps.get(predecessor,False) or dirty
        self.learned_events.append(dict(kind='dependency',source=predecessor,target=successor,
            dirty=dirty,at_ns=time.monotonic_ns()))
        if self._predecessors_cycle(successor):
            self._abort_now(actor)
            raise LockConflict('Speculative lock dependency cycle',(),kind='lock-conflict')

    def _visible(self,key):
        return [v for v in self.exposed_values.get(key,()) if self._live(v.owner)]

    def _validate_reads(self,txn):
        """Override parent's final validator: identity, not speculative integer CAS."""
        self._active_spec(txn);stale=[]
        for key,token in self.spec_attempts[txn.context.tid].reads.items():
            self._current(key)
            if token.published_version is None or self.committed_values[key].uid!=token.uid:stale.append(key)
        return tuple(stale)

    def _early_validate(self,txn):
        row=self._active_spec(txn);stale=[]
        for key,token in row.reads.items():
            self._current(key)
            if token.published_version is not None:
                if self.committed_values[key].uid!=token.uid:stale.append(key)
            elif token.owner is None or not self._live(token.owner) or self.spec_attempts[token.owner].writes.get(key) is not token:
                stale.append(key)
        self.learned_metrics['early_validations']+=1
        if stale:
            self._abort_now(txn.context.tid)
            raise LockConflict('Speculative early version conflict',stale,kind='version-conflict')

    def _expose(self,txn):
        row=self._active_spec(txn);self._early_validate(txn);tid=txn.context.tid
        for key,token in row.writes.items():
            if token.exposed:continue
            # Only already visible values/accesses, never future footprint.
            for prior in self._visible(key):self._edge(prior.owner,tid,False,tid)
            for access in self.learned_accesses.get(key,()):
                if access['kind']=='read':self._edge(access['tid'],tid,False,tid)
            self.exposed_values.setdefault(key,[]).append(token);token.exposed=True
            self.learned_metrics['exposed_writes']+=1
            self.learned_events.append(dict(kind='expose',tid=tid,key=key,uid=token.uid,at_ns=time.monotonic_ns()))

    def _retire(self,tid,state):
        super()._retire(tid,state)
        row=self.spec_attempts.get(tid)
        if row is None:return
        if state=='committed':
            for key,token in row.writes.items():
                token.published_version=self.learned_versions[key];self.committed_values[key]=token
            txn=self._live_transactions[tid]
            txn.metadata['runtime_published_identities']={key:token.uid for key,token in row.writes.items()}
        if state=='aborted':
            self._abort_now(tid)
        for key in tuple(self.exposed_values):
            self.exposed_values[key]=[v for v in self.exposed_values[key] if self._live(v.owner)]
            if not self.exposed_values[key]:self.exposed_values.pop(key)
        self.learned_condition.notify_all()

    def begin(self,task_id,metadata=None,*,snapshot_object_ids=None,strategy='silo'):
        if strategy not in ('silo',self.protocol_name):raise ValueError('Do not mix speculative CC protocols')
        metadata=dict(metadata or {})
        if any(metadata.get(k) for k in ('paper_atcc','paper_atcc_backend','retry_protection_mask')):
            raise ValueError('Cannot inherit Aegis protection')
        txn_type=metadata.get('learned_transaction_type',1);retry=metadata.get('retry_count',0)
        if type(txn_type) is not int or txn_type!=1:raise ValueError('Reference policy currently has one transaction type')
        if type(retry) is not int or retry<0:raise ValueError('Explicit nonnegative retry count required')
        with self.learned_condition:
            self._healthy()
            txn=AgentTransactionManager.begin(self,task_id,metadata,snapshot_object_ids=None,strategy='silo')
            self.learned_attempts[txn.context.tid]=InteractiveAttempt(1);self.spec_attempts[txn.context.tid]=SpeculativeAttempt()
        original_read,original_write,original_abort,original_check=txn.read,txn.write,txn.abort,txn.check_active
        def check_active():
            with txn._lifecycle_lock,self.learned_condition:original_check();self._active_spec(txn)
        def access(kind,key,value=None,**options):
            key=str(key)
            with txn._lifecycle_lock,self.learned_condition:
                row=self._active_spec(txn);tid=txn.context.tid
                if key not in self.committed_values:raise KeyError(key)
                decision=self.speculative_policy.select(dict(operation_type=kind,
                    access_index=row.access_index,
                    completed_operations=txn.context.completed_operations,dependencies=len(row.dependencies),
                    dirty_sources=sum(row.dependencies.values()),staged_writes=len(row.writes)))
                if not isinstance(decision,SpeculativeDecision):raise ValueError('Wrong speculative decision schema')
                self.learned_metrics['policy_calls']+=1
                self.learned_events.append(dict(kind='speculative-decision',tid=tid,key=key,
                    operation=kind,decision=asdict(decision),at_ns=time.monotonic_ns()))
                if decision.early_validate:self._early_validate(txn)
                self._current(key)
                if kind=='read':
                    if key not in txn.write_set and key not in row.reads:
                        visible=self._visible(key)
                        token=visible[-1] if decision.read_dirty and visible else self.committed_values[key]
                        if token.owner is not None and self._live(token.owner):
                            self._edge(token.owner,tid,True,tid);self.learned_metrics['dirty_reads']+=1
                        if not decision.read_dirty:
                            for later in visible:self._edge(tid,later.owner,False,tid)
                        row.reads[key]=token;txn.snapshot[key]=SnapshotValue(token.value,token.uid,True)
                    result=original_read(key,**options)
                    if key not in row.writes:
                        self.learned_accesses.setdefault(key,[]).append(dict(tid=tid,kind='read'))
                else:
                    result=original_write(key,value,**options)
                    token=ValueIdentity(key,self.next_uid,str(value),tid);self.next_uid-=1
                    row.writes[key]=token;self.value_identities.append(token)
                    txn.metadata.setdefault('runtime_write_identities',{})[key]=token.uid
                if decision.expose:self._expose(txn)
                row.access_index+=1
                return result
        def abort(reason,*,strategy='',conflict_object_ids=()):
            if txn.result is not None:
                if txn.result.committed:raise RuntimeError('Cannot abort committed speculative transaction')
                return txn.result
            if txn.metadata.get('interactive_publication_fault',{}).get('outcome')=='indeterminate':
                raise RuntimeError('Speculative publication outcome indeterminate; manager poisoned')
            return original_abort(reason,strategy=self.protocol_name,conflict_object_ids=conflict_object_ids)
        txn.read=lambda key,**options:access('read',key,**options)
        txn.write=lambda key,value,**options:access('write',key,value,**options)
        txn.abort=abort;txn.check_active=check_active
        return txn

    def commit(self,txn,*,strategy='silo'):
        if strategy not in ('silo',self.protocol_name) or txn.manager is not self:raise ValueError('Wrong speculative CC')
        started=time.monotonic();deadline=started+self.dependency_timeout_s
        with self.learned_condition:
            try:
                while True:
                    row=self._active_spec(txn)
                    for source,dirty in row.dependencies.items():
                        if dirty and not self._live(source) and self.learned_attempts[source].state!='committed':
                            raise LockConflict('Speculative dirty dependency aborted',tuple(row.reads),kind='version-conflict')
                    pending={source for source in row.dependencies if self._live(source)}
                    if not pending:break
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise LockConflict('Speculative dependency lock timeout',tuple(row.reads),kind='lock-timeout')
                    self.learned_metrics['dependency_waits']+=1
                    self.learned_condition.notify_all()
                    wait_started=time.monotonic_ns();self.learned_condition.wait(remaining);wait_ended=time.monotonic_ns()
                    txn.context.blocked_time_ms+=(wait_ended-wait_started)/1e6
                    self.learned_metrics['dependency_wait_ms']+=(wait_ended-wait_started)/1e6
                    self.learned_events.append(dict(kind='dependency-wait',tid=txn.context.tid,
                        sources=sorted(pending),start_ns=wait_started,end_ns=wait_ended))
                return super().commit(txn,strategy='silo')
            except LockConflict as exc:
                return self._finish_abort(txn,strategy=self.protocol_name,reason=exc.reason,
                    conflict_object_ids=exc.targets,lock_wait_s=max(0.,time.monotonic()-started))

    def export_value_identities(self):
        return [dict(key=v.key,uid=v.uid,value=v.value,owner=v.owner,published_version=v.published_version,
            exposed=v.exposed,owner_state=self.learned_attempts[v.owner].state if v.owner else 'initial') for v in self.value_identities]
