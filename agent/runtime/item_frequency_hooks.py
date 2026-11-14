"""Opt-in handwritten rule: protect only the currently issued frequent key."""
from collections import defaultdict, deque
from threading import RLock
import math
import time

from .operation_interceptor import NoopTransactionHooks


class ItemFrequencyHooks(NoopTransactionHooks):
    def __init__(self, manager, *, minimum_accesses=3, window_ms=1000., now_ns=None):
        count=float(minimum_accesses)
        if (not math.isfinite(count) or not count.is_integer() or count<1
                or not math.isfinite(float(window_ms)) or float(window_ms)<=0):
            raise ValueError("positive frequency threshold and window required")
        self.manager = manager
        self.minimum_accesses = int(minimum_accesses)
        self.window_ns = int(float(window_ms) * 1_000_000)
        self.now_ns = now_ns or time.monotonic_ns
        self.recent = defaultdict(deque)
        self.lock = RLock()
        self.decisions = []
        self.physical_changes = []

    def on_begin(self, txn):
        if txn.metadata.get("strategy") != "paper-atcc":
            raise ValueError("item frequency hooks require a dedicated paper-atcc manager")

    def _eligible(self, txn, key, write, *, decision_applied=True):
        with self.lock:
            # Take the timestamp under the same lock as queue insertion so
            # concurrent access records remain ordered for window eviction.
            now = self.now_ns()
            queue = self.recent[str(key)]
            cutoff = now - self.window_ns
            while queue and queue[0] < cutoff:
                queue.popleft()
            queue.append(now)
            count = len(queue)
            selected = count >= self.minimum_accesses
            self.decisions.append(dict(txn_id=txn.context.tid, key=str(key), write=write,
                                       recent_accesses=count, protect=selected,
                                       observed_ns=now, decision_applied=decision_applied))
        return selected

    def observe_only(self, txn, object_id, *, write=False):
        """Count a competing access whose concurrency policy is held fixed."""
        return self._eligible(txn,str(object_id),write,decision_applied=False)

    def _protect(self, txn, object_id, *, write):
        key = str(object_id)
        return self.protect_selected(txn,key,write=write,selected=self._eligible(txn,key,write))

    def protect_selected(self,txn,object_id,*,write,selected):
        """Install one already-journaled rule decision without double counting."""
        key=str(object_id)
        if not selected:
            return
        held_write = key in txn.context.held_write_locks
        held_read = key in txn.context.held_read_locks
        if held_write or (held_read and not write):
            return
        self.manager.refresh_atcc_priority(txn)
        if write:
            if key in txn.read_set:
                self.manager.atcc_locks.validate_and_wlock(key, txn.context, txn.snapshot[key].version,
                    lambda: int(self.manager.version_manager.read_committed(key).version))
            else:
                self.manager.atcc_locks.wlock(key, txn.context)
                txn.refresh_unobserved_locked_snapshot(key)
            txn.context.policy_write_lock_targets.add(key)
        else:
            if (getattr(self.manager,'first_access_read_refresh',False)
                    and key not in txn.read_set and key not in txn.context.read_versions
                    and key not in txn.context.write_targets and txn.context.snapshot_epoch < 0):
                self.manager.atcc_locks.validate_and_rlock(key,txn.context,txn.snapshot[key].version,
                    lambda: int(self.manager.version_manager.read_committed(key).version),
                    refresh_unobserved_snapshot=lambda: txn.refresh_unobserved_locked_snapshot(key))
            else:
                self.manager.atcc_locks.validate_and_rlock(key, txn.context, txn.snapshot[key].version,
                    lambda: int(self.manager.version_manager.read_committed(key).version))
            txn.context.policy_read_lock_targets.add(key)
        txn._event("item-protection-change", dict(key=key, mode="write" if write else "read",
                                                  previous="read" if held_read else "optimistic"))
        with self.lock:
            self.physical_changes.append(dict(txn_id=txn.context.tid,key=key,
                mode='write' if write else 'read',previous='read' if held_read else 'optimistic'))

    def before_read(self, txn, object_id):
        self._protect(txn, object_id, write=False)

    def before_write(self, txn, object_id):
        self._protect(txn, object_id, write=True)

    def before_commit(self, txn):
        # Same final paid-work accounting as PaperATCCHooks, without a lookup.
        self.manager.interceptor.account_agent_interval(txn)
        self.manager.refresh_atcc_priority(txn)
