"""Opt-in cross-task write history for policy-selected first-read reservation.

Only an actual completed write API populates the bounded history. It contains
object identifiers, never future accesses or values. The learned write bit
still decides whether to reserve the currently issued read. Existing values
already returned to the task cannot be refreshed, and validation is unchanged.
This engineering variant is separate from the original retry-only hook.
"""
from collections import OrderedDict
import threading

from .context import LockClass


class ObservedWriteReservationHooks:
    def __init__(self, delegate, *, capacity=4096):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive integer history capacity required')
        self.delegate = delegate
        self.manager = delegate.manager
        self.capacity = capacity
        self.observed_writes = OrderedDict()
        self.history_lock = threading.RLock()
        self.reservation_events = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def after_write(self, txn, object_id):
        result = self.delegate.after_write(txn, object_id)
        key = str(object_id)
        if key in txn.write_set and self.delegate.enabled(txn) and self.delegate.online_observed(txn):
            with self.history_lock:
                self.observed_writes.pop(key, None)
                self.observed_writes[key] = True
                while len(self.observed_writes) > self.capacity:
                    self.observed_writes.popitem(last=False)
        return result

    def before_read(self, txn, object_id):
        key = str(object_id)
        with self.history_lock:
            known = key in self.observed_writes
        bit = LockClass.HOT_WRITE if self.manager.peek_hot(key) else LockClass.COLD_WRITE
        if (known and self.delegate.enabled(txn) and self.delegate.online_observed(txn)
                and int(txn.context.action.protected) & int(bit)
                and txn.context.snapshot_epoch < 0
                and key not in txn.read_set and key not in txn.write_set
                and key not in txn.context.read_versions
                and key not in txn.context.held_write_locks):
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.wlock(key, txn.context)
            txn.refresh_unobserved_locked_snapshot(key)
            txn.context.policy_write_lock_targets.add(key)
            event = dict(tid=txn.context.tid, task_id=txn.task_id, key=key,
                protection_bit=int(bit), source='actual completed write API in this runtime')
            txn._event('observed-write-first-read-reservation', event)
            with self.history_lock:
                self.reservation_events.append(event)
        return self.delegate.before_read(txn, key)
