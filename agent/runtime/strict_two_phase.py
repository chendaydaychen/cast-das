"""Opt-in operation-level rigorous S/X 2PL for a single-strategy manager.

Uses the existing wait-die/no-wait lock table. Unlike the legacy commit-only
adapter, locks are acquired before the first read/write and retained to finish.
"""
from contextlib import ExitStack
from threading import RLock, local
from types import SimpleNamespace
import time

from .operation_interceptor import NoopTransactionHooks


class StrictTwoPhaseHooks(NoopTransactionHooks):
    def __init__(self, manager):
        self.manager = manager
        self.births = {}
        self.birth_lock = RLock()
        self.local = local()
        condition = manager.two_phase_locks._condition
        original_wait = condition.wait

        def recorded_wait(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original_wait(*args, **kwargs)
            finally:
                txn = getattr(self.local, "txn", None)
                if txn is not None:
                    elapsed = (time.perf_counter() - started) * 1000
                    txn.context.blocked_time_ms += elapsed
                    txn.metadata["strict_2pl_wait_ms"] += elapsed
        condition.wait = recorded_wait

    def on_begin(self, txn):
        strategy = txn.metadata.get("strategy")
        if strategy not in {"2pl-wait-die", "2pl-nowait"}:
            raise ValueError("strict 2PL hooks require a dedicated 2PL manager")
        with self.birth_lock:
            birth = self.births.setdefault(txn.task_id, txn.started_at)
        txn._strict_2pl_owner = SimpleNamespace(started_at=birth)
        txn._strict_2pl_stack = ExitStack()
        txn._strict_2pl_modes = {}
        txn.metadata.update(prelocked_lock_table="2pl", prelocked_targets=[],
                            strict_2pl_wait_ms=0., strict_2pl_acquire_ms=0.)

    def _acquire(self, txn, key, mode):
        key = str(key)
        held = txn._strict_2pl_modes.get(key)
        if held == "x" or held == mode:
            return
        policy = "wait-die" if txn.metadata["strategy"] == "2pl-wait-die" else "nowait"
        started = time.perf_counter()
        self.local.txn = txn
        try:
            txn._strict_2pl_stack.enter_context(self.manager.two_phase_locks.acquire(
                (key,), owner=txn._strict_2pl_owner, mode=mode, policy=policy))
            txn._strict_2pl_modes[key] = mode
            txn.metadata["prelocked_targets"] = sorted(txn._strict_2pl_modes)
        finally:
            txn.metadata["strict_2pl_acquire_ms"] += (time.perf_counter() - started) * 1000
            self.local.txn = None

    def before_read(self, txn, object_id):
        self._acquire(txn, object_id, "s")
        # AgentTransaction prepares a snapshot before invoking this hook.
        # Refresh an unreturned first-read value only after holding the SLock.
        txn.refresh_unobserved_locked_snapshot(object_id)

    def before_write(self, txn, object_id):
        self._acquire(txn, object_id, "x")
        txn.refresh_unobserved_locked_snapshot(object_id)

    def on_abort(self, txn, reason):
        txn._strict_2pl_stack.close()

    def on_finish(self, txn):
        txn._strict_2pl_stack.close()
