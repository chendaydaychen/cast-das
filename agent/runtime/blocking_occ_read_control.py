"""Explicit mechanism control for deferred-write-enabled nonblocking OCC reads.

Both variants KEEP private write sets and atomic publication. This is not a
physical eager-write/undo implementation or a complete Figure12 NoDelay arm.
"""
import math
import threading


class BlockingOCCReadControl:
    def __init__(self,manager,*,timeout_s=5.):
        if isinstance(timeout_s,bool) or not math.isfinite(float(timeout_s)) or float(timeout_s)<0:
            raise ValueError('Finite nonnegative blocking read timeout required')
        self.manager=manager;self.timeout_s=float(timeout_s)
        self.records=[];self._lock=threading.Lock()

    def before_read(self,txn,key):
        key=str(key)
        # Repeated/read-your-writes accesses read transaction-private cached
        # state, not the in-place object. Pessimistic reads already use Acquire.
        if (key in txn.read_set or key in txn.write_set or
                key in txn.context.held_read_locks or key in txn.context.held_write_locks):return
        if txn.context.snapshot_epoch>=0:
            raise ValueError('Blocking read control supports unpinned observed reads only')
        def refresh():
            txn.refresh_unobserved_locked_snapshot(key)
        before=txn.context.blocked_time_ms;returned=False
        try:
            self.manager.atcc_locks.read_after_foreign_writers(
                key,txn.context,refresh,timeout_s=self.timeout_s)
            returned=True
        finally:
            with self._lock:self.records.append(dict(txn_id=str(txn.context.tid),key=key,
                returned=returned,waiting_ms=max(0.,txn.context.blocked_time_ms-before)))
        txn._event('blocking-occ-read-copy',dict(object_id=key))


def install_blocking_occ_read_control(manager,*,timeout_s=5.):
    from .paper_contract import PaperObservedHooks
    if manager._live_transactions:raise ValueError('Install control before transactions start')
    if getattr(manager,'blocking_occ_read_control',None) is not None:
        raise ValueError('Blocking OCC read control already installed')
    hooks=manager.interceptor.hooks
    if not isinstance(hooks,PaperObservedHooks) or not getattr(manager,'paper_observed_contract',None):
        raise ValueError('Explicit observed paper hooks required')
    control=BlockingOCCReadControl(manager,timeout_s=timeout_s);original=hooks.before_read
    def before_read(txn,key):
        original(txn,key)
        if hooks.enabled(txn):control.before_read(txn,key)
    hooks.before_read=before_read
    manager.blocking_occ_read_control=control
    manager.paper_observed_contract.update(
        optimistic_read_visibility='wait-for-foreign-writer-or-publisher-before-first-unprotected-read',
        private_write_buffering_retained=True,physical_eager_write_ablation=False)
    return control
