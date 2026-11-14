"""Opt-in retry repair using previously issued, policy-protected writes.

The prior attempt must actually have issued the write API and selected its
write category by failure time. At retry entry, reacquire only that observed key
before returning a new snapshot value. Values are never refreshed after being
returned. All commit validation is preserved. This is an engineering variant,
not a new PPO decision and not future-footprint knowledge.
"""
import threading
from collections import OrderedDict
from .context import LockClass
from agent.cc.locks import LockConflict

class ObservedRetryIssuePlan:
    def __init__(self,delegate,capacity=4096):
        self.delegate=delegate;self.manager=delegate.manager
        self.capacity=capacity;self.lock=threading.RLock()
        self.plans=OrderedDict();self.events=[]
    def __getattr__(self,name):return getattr(self.delegate,name)
    def on_begin(self,txn):
        self.delegate.on_begin(txn)
        txn._issued_policy_writes=set()
        txn._issued_write_requests=set()
        with self.lock:known=tuple(sorted(self.plans.get(txn.task_id,())))
        if txn.context.retry_count<=0 or not known:return
        if txn.read_set or txn.write_set or txn.context.snapshot_epoch>=0:
            raise ValueError('Retry issue plan requires an empty, unpinned entry')
        try:
            for key in known:
                self.manager.refresh_atcc_priority(txn)
                self.manager.atcc_locks.wlock(key,txn.context)
                txn.context.policy_write_lock_targets.add(key)
                with self.lock:self.events.append(dict(task_id=txn.task_id,tid=txn.context.tid,
                    object_id=key,source='prior issued write with learned write bit selected'))
        except LockConflict as exc:
            if txn.result is None:
                txn.abort(str(exc),strategy='paper-atcc',conflict_object_ids=exc.targets)
            raise
    def before_write(self,txn,object_id):
        key=str(object_id)
        # Register before delegating: another transaction can wound this
        # attempt while it waits inside the learned hook, before finally runs.
        with self.lock:txn._issued_write_requests.add(key)
        try:return self.delegate.before_write(txn,key)
        finally:
            bit=LockClass.HOT_WRITE if self.manager.peek_hot(key) else LockClass.COLD_WRITE
            if txn.context.action.protected&bit:
                with self.lock:txn._issued_policy_writes.add(key)
    def on_abort(self,txn,reason):
        with self.lock:
            issued=set(getattr(txn,'_issued_policy_writes',set()))
            requests=tuple(getattr(txn,'_issued_write_requests',set()))
        for key in requests:
            bit=LockClass.HOT_WRITE if self.manager.peek_hot(key) else LockClass.COLD_WRITE
            if txn.context.action.protected&bit:issued.add(key)
        with self.lock:
            if issued:
                self.plans.setdefault(txn.task_id,set()).update(issued)
                self.plans.move_to_end(txn.task_id)
                while len(self.plans)>self.capacity:self.plans.popitem(last=False)
        return self.delegate.on_abort(txn,reason)
    def on_finish(self,txn):
        try:return self.delegate.on_finish(txn)
        finally:
            if txn.result is not None and txn.result.committed:
                with self.lock:self.plans.pop(txn.task_id,None)

def install(manager):
    # Avoid materializing a stale inherited retry snapshot before the retry
    # plan can acquire its known write key. Decisions remain policy controlled.
    manager.policy_controlled_retry_protection=True
    manager.phase_policy_on_transition=True
    manager.expensive_operation_checkpoint_enabled=True
    manager.expensive_operation_checkpoint_fraction=0.0
    manager.pre_expensive_validation_enabled=True
    manager.validate_before_reasoning_enabled=True
    hooks=ObservedRetryIssuePlan(manager.interceptor.hooks)
    manager.interceptor.hooks=hooks
    manager.observed_retry_issue_plan=hooks
    return hooks
