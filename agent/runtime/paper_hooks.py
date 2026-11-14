"""Operation hooks that execute the paper ATCC action inside a transaction."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from .context import LockAction, LockClass, TransactionPhase
from .operation_interceptor import NoopTransactionHooks
from .object_protection_plan import ObjectProtectionPlan


class PaperATCCHooks(NoopTransactionHooks):
    def __init__(self, manager: Any):
        self.manager = manager

    @staticmethod
    def enabled(txn: Any) -> bool:
        return bool(txn.metadata.get("paper_atcc", False))

    @staticmethod
    def coordinated_backend(txn: Any) -> bool:
        return bool(txn.metadata.get("paper_atcc_backend", False))

    @staticmethod
    def online_observed(txn: Any) -> bool:
        return (
            str(txn.metadata.get("access_set_visibility", "")).strip().lower()
            == "online_observed"
        )

    @classmethod
    def online_ycsb_high_mixed(cls, txn: Any) -> bool:
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        return bool(
            cls.online_observed(txn)
            and str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and int(agentic.get("background_workers", 0) or 0) > 0
        )

    @classmethod
    def mark_online_bypass_read(cls, txn: Any, object_id: str) -> None:
        if cls.online_ycsb_high_mixed(txn):
            txn.metadata.setdefault("_online_bypass_read_targets", set()).add(
                str(object_id)
            )

    def on_begin(self, txn: Any) -> None:
        if not self.enabled(txn):
            return
        # Category write protection is a commit-admission decision.  Holding
        # every planned write through multi-round Agent reasoning creates a
        # background convoy; only an exact object learned from a failed retry
        # is promoted early.
        txn.metadata["_defer_policy_write_locks"] = bool(
            txn.metadata.get("commit_admission_write_protection", False)
            and self.manager.delayed_write_apply_enabled
        )
        # Do not inspect the declared access set or static contention label.
        # All classification below is based on operations already executed.
        txn.metadata["_version_risk_exact_mode"] = bool(
            self.online_observed(txn)
        )
        txn.metadata["_version_risk_read_targets"] = ()
        txn.metadata["_cold_occ_fast_task"] = False
        txn.metadata["_native_occ_stable_windows"] = 0
        txn.metadata["_native_occ_fast_ready"] = False
        # Optional deferred protection plan.  The plan is deliberately
        # transaction-local and starts empty: it can only contain objects
        # observed by this attempt, never a declared/future footprint.
        if bool(getattr(self.manager, "deferred_object_plan_enabled", False)):
            threshold = float(getattr(self.manager, "deferred_object_plan_min_redo_ms", 0.0) or 0.0)
            txn.metadata["_object_protection_plan"] = ObjectProtectionPlan(threshold)
        if (
            not self.manager.collect_trajectories
            and self.manager.native_occ_fast_enabled
            and txn.context.retry_count == 0
            and not txn.context.retry_conflict_mask
            and txn.context.action.protected == LockClass.NONE
        ):
            txn.metadata["_cold_occ_fast_task"] = True
            self.manager.note_native_occ_fast_start()
            return
        # The switching ablation's Static arm is a fixed protection policy,
        # not an OCC policy that happens to consult a threshold later. Install
        # its mask before the first access so an immediate-write baseline
        # really acquires WLocks at write time. Dynamic policies still start
        # in OCC and transition from observed state as described in the paper.
        static_mask = int(
            getattr(self.manager, "paper_static_initial_mask", 0) or 0
        ) & 0xF
        if static_mask:
            txn.context.action = LockAction(LockClass(static_mask))

    def before_read(self, txn: Any, object_id: str) -> None:
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        self._start_online_observed_prefix(txn)
        hot = self.manager.is_hot(object_id)
        if hot:
            txn.context.hot_read_targets.add(str(object_id))
        key = str(object_id)
        exact_mode = bool(txn.metadata.get("_version_risk_exact_mode", False))
        exact_version_risk = key in set(
            txn.metadata.get("_version_risk_read_targets", ())
        )
        exact_write = key in txn.context.retry_conflict_write_targets
        exact_read = key in txn.context.retry_conflict_read_targets
        if self.online_ycsb_high_mixed(txn) and exact_write:
            # A previous conflict may identify this exact future write, but it
            # must not turn into a task-wide or reasoning-long WLock. Treat its
            # read as a bypassable observed dependency and upgrade only when
            # the write operation is actually issued.
            exact_write = False
            exact_read = True
        planned_write = key in txn.context.planned_write_targets
        deferred_tpcc_root_write = bool(
            txn.metadata.get("_deferred_reasoning_replay", False)
            and str(txn.metadata.get("workload", "")).strip().lower() == "tpcc"
            and planned_write
            and key.startswith(
                (
                    "tpcc:warehouse:",
                    "tpcc:district:",
                    "tpcc:customer:",
                )
            )
        )
        if deferred_tpcc_root_write:
            # The reasoning interval has already elapsed, so an exact WLock
            # protects only the short replay/commit suffix.  This serializes
            # one-warehouse roots before they are observed without recreating
            # a long Agent-side 2PL convoy.
            exact_write = True
        exact_retry_only = self._tpcc_all_agent_exact_retry(txn)
        if exact_retry_only:
            # Keep the exact prior-conflict write target, but do not broaden
            # it into read or category protection for all-Agent TPC-C retries.
            exact_read = False
        if key in txn.context.held_write_locks:
            txn.context.policy_write_lock_targets.add(key)
            return
        if key in txn.context.held_read_locks and not (exact_write or planned_write):
            txn.context.policy_read_lock_targets.add(key)
            return
        if exact_version_risk:
            guard_bit = LockClass.HOT_READ if hot else LockClass.COLD_READ
            if not (txn.context.action.protected & guard_bit):
                self.manager.transition_atcc_action(
                    txn,
                    LockAction(txn.context.action.protected | guard_bit),
                    timeout_s=self._action_transition_timeout_s(txn),
                )
        write_class_protected = bool(
            txn.context.action.protects(hot=hot, write=True)
            and not exact_retry_only
        )
        if exact_write or (
            planned_write
            and write_class_protected
            and not txn.metadata.get("_defer_policy_write_locks", False)
        ):
            self.manager.refresh_atcc_priority(txn)
            if key in txn.read_set:
                snapshot = txn.snapshot[key]
                self.manager.atcc_locks.validate_and_wlock(
                    key,
                    txn.context,
                    snapshot.version,
                    lambda: int(
                        self.manager.version_manager.read_committed(key).version
                    ),
                )
            else:
                # _ensure_snapshot has prepared a value, but it has not yet
                # been returned to the Agent. Establish the WLock first and
                # read the current committed base inside that protection.
                self.manager.atcc_locks.wlock(key, txn.context)
                txn.refresh_unobserved_locked_snapshot(key)
            txn.context.policy_write_lock_targets.add(key)
        elif (
            exact_version_risk
            and exact_mode
            and self.online_observed(txn)
        ):
            self.manager.refresh_atcc_priority(txn)
            snapshot = txn.snapshot[key]
            self.manager.atcc_locks.validate_and_rlock(
                key,
                txn.context,
                snapshot.version,
                lambda: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_read_lock_targets.add(key)
            self.mark_online_bypass_read(txn, key)
            self.manager.version_manager.note_version_risk_read_lock()
        elif (
            exact_version_risk
            and exact_mode
            and not txn.context.planned_write_targets
            and not txn.write_set
        ):
            self.manager.refresh_atcc_priority(txn)
            # Keep the old committed version through the long reasoning
            # interval without holding an RLock. Commit installs a short read
            # guard, so Agent writers remain serialized while background
            # publishers can keep creating private versions.
            self.manager.ensure_snapshot_epoch(txn)
            snapshot = txn.snapshot[key]
            if self.manager.version_manager.can_lock_pinned_version(
                txn.context.snapshot_epoch,
                key,
                int(snapshot.version),
                tid=txn.context.tid,
            ):
                txn.metadata.setdefault(
                    "_version_risk_pinned_read_targets", set()
                ).add(key)
                self.manager.version_manager.note_version_risk_read_lock()
                return
            self.manager.atcc_locks.validate_and_rlock(
                key,
                txn.context,
                snapshot.version,
                lambda: (
                    int(snapshot.version)
                    if self.manager.version_manager.can_lock_pinned_version(
                        txn.context.snapshot_epoch,
                        key,
                        int(snapshot.version),
                        tid=txn.context.tid,
                    )
                    else int(
                        self.manager.version_manager.read_committed(
                            key
                        ).version
                    )
                ),
            )
            txn.context.policy_read_lock_targets.add(key)
            self.mark_online_bypass_read(txn, key)
            if exact_version_risk:
                self.manager.version_manager.note_version_risk_read_lock()

        elif (
            exact_read
            or (
                txn.context.action.protects(hot=hot, write=False)
                and not exact_retry_only
                and self.manager.category_read_lock_needed(txn, key)
            )
        ):
            self.manager.refresh_atcc_priority(txn)
            snapshot = txn.snapshot[key]
            fresh_first_read = bool(
                self.manager.first_access_read_refresh
                and self.online_observed(txn)
                and txn.context.snapshot_epoch < 0
                and key not in txn.read_set and key not in txn.write_set
            )
            self.manager.atcc_locks.validate_and_rlock(
                key,
                txn.context,
                snapshot.version,
                lambda: (
                    int(snapshot.version)
                    if self.manager.version_manager.can_lock_pinned_version(
                        txn.context.snapshot_epoch,
                        key,
                        int(snapshot.version),
                        tid=txn.context.tid,
                    )
                    else int(
                        self.manager.version_manager.read_committed(
                            key
                        ).version
                    )
                ),
                refresh_unobserved_snapshot=(
                    lambda: txn.refresh_unobserved_locked_snapshot(key)
                ) if fresh_first_read else None,
            )
            txn.context.policy_read_lock_targets.add(key)
            self.mark_online_bypass_read(txn, key)

            if exact_version_risk:
                self.manager.version_manager.note_version_risk_read_lock()

    def before_write(self, txn: Any, object_id: str) -> None:
        if self.coordinated_backend(txn):
            # Short-lived backend writes remain optimistic and buffered; the
            # unified commit protocol acquires their WLocks as one write set.
            return
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        self._start_online_observed_prefix(txn)
        hot = self.manager.is_hot(object_id)
        if hot:
            txn.context.hot_write_targets.add(str(object_id))
        key = str(object_id)
        exact_write = key in txn.context.retry_conflict_write_targets
        exact_retry_only = self._tpcc_all_agent_exact_retry(txn)
        if key in txn.context.held_write_locks:
            txn.context.policy_write_lock_targets.add(key)
            return
        profiled_shared = self.manager.hotness_tracker.is_profiled_shared(key)
        if (
            bool(getattr(self.manager, "performance_guards_enabled", True))
            and profiled_shared
            and not exact_retry_only
            and (
                txn.context.retry_count > 0
                or txn.context.retry_conflict_mask
            )
            and txn.context.phase in {
                TransactionPhase.REFINE,
                TransactionPhase.COMMIT,
            }
        ):
            guard_bit = LockClass.HOT_WRITE if hot else LockClass.COLD_WRITE
            if not (txn.context.action.protected & guard_bit):
                self.manager.transition_atcc_action(
                    txn,
                    LockAction(txn.context.action.protected | guard_bit),
                    timeout_s=self._action_transition_timeout_s(txn),
                )
        category_write_protected = bool(
            txn.context.action.protects(hot=hot, write=True)
            and not exact_retry_only
        )
        if (
            txn.metadata.get("_defer_policy_write_locks", False)
            and not exact_write
        ):
            # DWA operates on the transaction's observed write set. Once a
            # write is actually issued it is safe to remember its exact key,
            # even when the current dynamic category action is OCC. Admission
            # then protects only this key at commit; no future access is used.
            if (not self.manager.policy_selected_write_admission_only
                    or category_write_protected):
                txn.context.policy_write_lock_targets.add(key)
            return
        if exact_write or category_write_protected:
            self.manager.refresh_atcc_priority(txn)
            # Exact retry writes acquire their WLock immediately. Category
            # write protection is deferred to the short commit admission.
            if key in txn.read_set:
                snapshot = txn.snapshot[key]
                self.manager.atcc_locks.validate_and_wlock(
                    key,
                    txn.context,
                    snapshot.version,
                    lambda: int(
                        self.manager.version_manager.read_committed(key).version
                    ),
                )
            else:
                # _ensure_snapshot materializes the transaction snapshot
                # before this hook, but a blind write has not observed it.
                # Lock first, then rebase the private write on the latest
                # committed version while publication is excluded.
                self.manager.atcc_locks.wlock(key, txn.context)
                txn.refresh_unobserved_locked_snapshot(key)
            txn.context.policy_write_lock_targets.add(key)

    def before_commit(self, txn: Any) -> None:
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        if self.enabled(txn):
            self.manager.interceptor.account_agent_interval(txn)
            self._select_and_apply(txn, self._state(txn))
            self._protect_tpcc_read_before_write(txn, require_evidence=False)
        if self.enabled(txn) or (
            self.coordinated_backend(txn) and bool(txn.write_set)
        ):
            self.manager.refresh_atcc_priority(txn)

    def _protect_tpcc_read_before_write(
        self,
        txn: Any,
        *,
        require_evidence: bool,
    ) -> None:
        """Late-protect exact TPC-C warehouse/district read-before-writes."""
        if str(txn.metadata.get("workload", "")).strip().lower() != "tpcc":
            return
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        if (
            int(agentic.get("background_workers", 0) or 0) == 0
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and not txn.metadata.get("_deferred_reasoning_replay", False)
        ):
            # Start one-warehouse all-Agent TPC-C optimistically. A failed
            # attempt feeds its exact root object into retry protection.
            txn.metadata["_tpcc_first_attempt_exact_only"] = True
            return
        candidates = set(txn.read_set) & (
            set(txn.write_set) | set(txn.context.planned_write_targets)
        )
        if self._tpcc_all_agent_exact_retry(txn):
            candidates &= set(txn.context.retry_conflict_write_targets)
        targets = tuple(
            sorted(
                object_id
                for object_id in candidates
                if object_id in getattr(self.manager,'tpcc_root_descriptors',{})
                or object_id.startswith("tpcc:warehouse:")
                or object_id.startswith("tpcc:district:")
            )
        )
        newly_protected = []
        for object_id in targets:
            if object_id in txn.context.held_write_locks:
                continue
            parts = object_id.split(":")
            object_type = parts[1] if len(parts) > 1 else ""
            field = parts[-1] if parts else ""
            descriptor=getattr(self.manager,'tpcc_root_descriptors',{}).get(object_id)
            if descriptor is not None:
                object_type,field=descriptor
            exact_changes, family_changes, total_changes = (
                self.manager.version_manager.background_change_evidence(
                    object_id,
                    prefix=f"tpcc:{object_type}:",
                    suffix=f":{field}",
                )
            )
            sufficient = bool(
                total_changes >= 32
                and (exact_changes >= 2 or family_changes >= 2)
            )
            self.manager.atcc_locks.note_tpcc_exact_guard_evidence(
                exact_changes=exact_changes,
                family_changes=family_changes,
                total_changes=total_changes,
                sufficient=sufficient,
            )
            if require_evidence and not sufficient:
                continue
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.validate_and_wlock(
                object_id,
                txn.context,
                int(txn.read_set[object_id].version),
                lambda key=object_id: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_write_lock_targets.add(object_id)
            newly_protected.append(object_id)
            txn.metadata.setdefault("_tpcc_exact_risk_targets", set()).add(
                object_id
            )
            self.manager.atcc_locks.note_tpcc_exact_risk_wlock(
                family_fallback=exact_changes < 2 <= family_changes,
            )
        if newly_protected:
            txn.metadata.setdefault("_tpcc_late_protected_targets", set()).update(
                newly_protected
            )

    def on_phase_change(self, txn: Any, phase: TransactionPhase) -> None:
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            return
        if getattr(self.manager, "commit_read_checkpoints", False) and phase == TransactionPhase.COMMIT:
            txn.metadata["_commit_checkpoint_reads"] = set(txn.context.read_versions)
        initial_explore = (
            phase == TransactionPhase.EXPLORE
            and not txn.context.read_versions
            and not txn.context.write_targets
        )
        if initial_explore:
            # A retry already knows the exact objects that invalidated the
            # previous attempt. Protect those objects before another long
            # interaction round starts; waiting until the object is visited
            # again leaves it exposed between the atomic snapshot and access.
            self._protect_retry_targets(txn)
            # Paper section 4.2 starts every transaction in OCC. The first
            # policy lookup occurs only after the initial operation batch,
            # when its observed access and cost state is available.
            return
        if self._policy_invocation_ops(txn) > 0 and not getattr(
            self.manager, "phase_policy_on_transition", False
        ):
            if phase == TransactionPhase.COMMIT:
                self._protect_tpcc_read_before_write(txn, require_evidence=True)
            return
        state = self._state(txn)
        self._select_and_apply(txn, state)
        if phase == TransactionPhase.COMMIT:
            # Keep explore/refine optimistic, then protect only the already
            # observed high-frequency root dependency before the write suffix.
            self._protect_tpcc_read_before_write(txn, require_evidence=True)

    def after_read(self, txn: Any, object_id: str, version: int) -> None:
        self._observe_deferred_object(txn, object_id, version, write=False)
        if (not getattr(self.manager, "commit_read_checkpoints", False)
                or not self.enabled(txn) or txn.metadata.get("_cold_occ_fast_task", False)
                or txn.context.phase != TransactionPhase.COMMIT):
            return
        # The operation has already populated read_set/read_versions. Never
        # use commit_targets, object names or a future declared footprint.
        seen = txn.metadata.setdefault("_commit_checkpoint_reads", set())
        key = str(object_id)
        if key not in seen:
            seen.add(key)
            txn.metadata["_commit_read_checkpoint_pending"] = key

    def after_write(self, txn: Any, object_id: str) -> None:
        if not self.enabled(txn):
            return
        key = str(object_id)
        record = txn.write_set.get(key)
        if record is not None:
            self._observe_deferred_object(txn, key, int(record.base_version), write=True)

    def on_operation_finished(self, txn: Any) -> None:
        if not self.enabled(txn) or txn.metadata.get("_cold_occ_fast_task", False):
            return
        checkpoint_target = txn.metadata.pop("_commit_read_checkpoint_pending", None)
        if checkpoint_target is not None:
            # Run after SQL accounting. Keep the original phase checkpoints;
            # new decision times are common to Full, rules and ablations.
            txn.metadata["_last_policy_checkpoint_ops"] = int(txn.context.completed_operations)
            self.manager.refresh_atcc_priority(txn)
            txn.metadata["_commit_read_checkpoint_target"] = checkpoint_target
            try:
                self._select_and_apply(txn, self._state(txn))
            finally:
                txn.metadata.pop("_commit_read_checkpoint_target", None)
                self.manager.interceptor.reset_agent_interval(txn)
            return
        interval = self._policy_invocation_ops(txn)
        completed = int(txn.context.completed_operations)
        last = int(txn.metadata.get("_last_policy_checkpoint_ops", 0) or 0)
        if interval <= 0 or completed < interval or completed - last < interval:
            return
        txn.metadata["_last_policy_checkpoint_ops"] = completed
        self.manager.interceptor.account_agent_interval(txn)
        self.manager.refresh_atcc_priority(txn)
        self._select_and_apply(txn, self._state(txn))

    def _observe_deferred_object(self, txn: Any, object_id: str, version: int, *, write: bool) -> None:
        """Record an actually observed dependency for the optional late plan."""
        plan = txn.metadata.get("_object_protection_plan")
        if plan is None:
            return
        # Measured redo cost includes the current agent interval and the
        # previous failed attempt's observed cost.  No synthetic future
        # footprint is used; zero is a valid low-cost observation.
        cost = max(
            0.0,
            float(getattr(txn.context, "agent_cost_ms", 0.0) or 0.0),
            float(getattr(txn.context, "prior_retry_cost_ms", 0.0) or 0.0),
        )
        plan.observe(str(object_id), int(version), write=bool(write), observed_redo_cost_ms=cost)

    def _nominate_deferred_objects(self, txn: Any, action: Any) -> int:
        """Nominate observed high-cost objects compatible with an action."""
        plan = txn.metadata.get("_object_protection_plan")
        if plan is None:
            return 0
        nominated = 0
        current_cost = max(
            0.0,
            float(getattr(txn.context, "agent_cost_ms", 0.0) or 0.0),
            float(getattr(txn.context, "prior_retry_cost_ms", 0.0) or 0.0),
        )
        for key, observed in tuple(plan.observed.items()):
            # Refresh the measured cost at the checkpoint.  This is still
            # the same observed version/object, never a future footprint.
            plan.observe(key, observed.version, write=observed.write,
                         observed_redo_cost_ms=current_cost)
            observed = plan.observed[key]
            hot = key in txn.context.hot_read_targets or key in txn.context.hot_write_targets
            protects = action.protects(hot=hot, write=bool(observed.write))
            if protects and plan.nominate(key):
                nominated += 1
        txn.metadata["_deferred_plan_nominations"] = int(
            txn.metadata.get("_deferred_plan_nominations", 0) or 0
        ) + nominated
        return nominated

    def _materialize_deferred_objects(self, txn: Any) -> int:
        """Validate and lock only nominated observed objects at commit."""
        plan = txn.metadata.get("_object_protection_plan")
        if plan is None:
            return 0
        acquired = []
        def validate_and_lock(target):
            if (target.object_id in txn.context.held_write_locks
                    or (not target.write and target.object_id in txn.context.held_read_locks)):
                return
            self.manager.refresh_atcc_priority(txn)
            current = lambda key=target.object_id: int(self.manager.version_manager.read_committed(key).version)
            if target.write:
                self.manager.atcc_locks.validate_and_wlock(
                    target.object_id, txn.context, int(target.version), current)
                txn.context.policy_write_lock_targets.add(target.object_id)
            else:
                self.manager.atcc_locks.validate_and_rlock(
                    target.object_id, txn.context, int(target.version), current)
                txn.context.policy_read_lock_targets.add(target.object_id)
        acquired = plan.materialize(
            validate_and_lock,
            held_reads=txn.context.held_read_locks,
            held_writes=txn.context.held_write_locks,
        )
        txn.metadata["_deferred_plan_materialized"] = int(
            txn.metadata.get("_deferred_plan_materialized", 0) or 0
        ) + len(acquired)
        return len(acquired)

    def before_expensive_operation(self, txn: Any, delay_ms: int) -> None:
        """Full-only checkpoint after the observed prefix and before reasoning.

        This does not reveal a future object or conflict: the policy sees only
        the state accumulated by operations that have already completed.  The
        delay value is used solely to recognize the configured expensive-op
        boundary and is not injected into the policy state.
        """
        if not getattr(self.manager, "expensive_operation_checkpoint_enabled", False):
            return
        if not self.enabled(txn) or txn.metadata.get("_cold_occ_fast_task", False):
            return
        if txn.metadata.get("_expensive_operation_checkpoint_active", False):
            return
        txn.metadata["_expensive_operation_checkpoint_active"] = True
        try:
            self.manager.interceptor.account_agent_interval(txn)
            self.manager.refresh_atcc_priority(txn)
            self._select_and_apply(txn, self._state(txn))
            txn.metadata["_expensive_operation_checkpoints"] = int(
                txn.metadata.get("_expensive_operation_checkpoints", 0) or 0
            ) + 1
        finally:
            txn.metadata.pop("_expensive_operation_checkpoint_active", None)
            self.manager.interceptor.reset_agent_interval(txn)


    @staticmethod
    def _policy_invocation_ops(txn: Any) -> int:
        context = dict(txn.metadata.get("context", {}) or {})
        return max(0, int(context.get("policy_invocation_ops", 0) or 0))

    def _protect_retry_targets(self, txn: Any) -> None:
        if txn.context.retry_count <= 0:
            return
        if self._tpcc_all_agent_exact_retry(txn):
            # TPC-C high reasoning is deferred before begin().  A cached retry
            # therefore has no long Agent interval inside the transaction.  Lock
            # only the roots that actually conflicted on the preceding attempt
            # at the replay boundary; waiting until commit lets the same
            # warehouse/district version change again and creates retry storms.
            # No future/unobserved target is introduced here.
            txn.metadata["_tpcc_retry_root_prelock"] = True
        read_targets = set(txn.context.retry_conflict_read_targets)
        write_targets = set(txn.context.retry_conflict_write_targets)
        # An online-observed retry knows a prior conflict key, but acquiring
        # its WLock at begin would hold it across the whole reasoning loop.
        # Reacquire it when that key is actually accessed instead.
        defer_exact_writes = self.online_observed(txn)
        for object_id in sorted(() if defer_exact_writes else write_targets):
            if object_id in txn.context.held_write_locks:
                continue
            snapshot = txn.snapshot.get(object_id)
            if snapshot is None:
                continue
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.validate_and_wlock(
                object_id,
                txn.context,
                int(snapshot.version),
                lambda key=object_id: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_write_lock_targets.add(object_id)
        for object_id in sorted(read_targets - write_targets):
            if (
                object_id in txn.context.held_read_locks
                or object_id in txn.context.held_write_locks
            ):
                continue
            snapshot = txn.snapshot.get(object_id)
            if snapshot is None:
                continue
            self.manager.refresh_atcc_priority(txn)
            self.manager.atcc_locks.validate_and_rlock(
                object_id,
                txn.context,
                int(snapshot.version),
                lambda key=object_id: int(
                    self.manager.version_manager.read_committed(key).version
                ),
            )
            txn.context.policy_read_lock_targets.add(object_id)
            self.mark_online_bypass_read(txn, object_id)

    def _select_and_apply(self, txn: Any, state: Any) -> None:
        if (
            getattr(self.manager, "learned_checkpoint_only", False)
            and not txn.metadata.get("_expensive_operation_checkpoint_active", False)
        ):
            # A timed action selected at the shared expensive-operation
            # checkpoint is a plan, not an immediate lock acquisition.  It
            # must still be materialized at commit even though checkpoint-only
            # policies do not perform another lookup there.
            pending = LockClass(int(txn.metadata.get("_pending_policy_protection", 0)))
            is_commit = str(getattr(state, "phase", "")).strip().lower() == "commit"
            if is_commit and pending != LockClass.NONE:
                txn.metadata.pop("_pending_policy_protection", None)
                selected = LockAction(txn.context.action.protected | pending)
                txn.metadata["_deferred_policy_materializations"] = int(
                    txn.metadata.get("_deferred_policy_materializations", 0)
                ) + 1
                if selected.protected != txn.context.action.protected:
                    self.manager.transition_atcc_action(
                        txn,
                        selected,
                        timeout_s=self._action_transition_timeout_s(txn),
                    )
            return
        policy_started = time.perf_counter()
        policy = self.manager.paper_policy.snapshot()
        distribution_sampler = getattr(policy, "select_with_distribution", None)
        behavior_action_probabilities = ()
        if distribution_sampler is not None:
            selected, behavior_action_probabilities = distribution_sampler(state)
            behavior_probability = behavior_action_probabilities[
                int(getattr(selected, "encoded", int(selected.protected)))
            ]
        else:
            sampler = getattr(policy, "select_with_probability", None)
            if sampler is not None:
                selected, behavior_probability = sampler(state)
            else:
                selected = policy.select(state)
                behavior_probability = 1.0
        # PPO's likelihood belongs to the sampled proposal, not to the action
        # subsequently transformed by safety/performance guards. Recording a
        # guarded mask with the proposal's probability corrupts pi(a|s)/q(a|s).
        # Guards still execute unchanged; their actual effects remain in the
        # action-change events and the resulting rewards / next state.
        sampled_proposal = selected
        refresh=getattr(self.manager,'paper_policy_refresh',None)
        if refresh is not None:
            refresh.record_lookup(getattr(policy,'last_lookup_distance',None))
        self._update_native_occ_hysteresis(txn, state)
        if self._low_conflict_occ_guard(txn, state):
            selected = LockAction(LockClass.NONE)
        selected = LockAction(
            (selected.protected if getattr(self.manager,'allow_future_action_mask_reduction',False)
             else selected.protected | txn.context.action.protected),
            defer_to_commit=bool(getattr(selected, "defer_to_commit", False)),
        )
        selected = self._apply_protection_guard(txn, state, selected)
        # In the opt-in phase-scheduling experiment, an early decision is a
        # *plan* rather than an instruction to acquire locks immediately.  At
        # commit we materialize the union of all nominated protection and the
        # current decision.  The context action remains the actual held-lock
        # mask, therefore no lock can be removed by this mechanism.
        planned = selected
        is_commit = str(getattr(state, "phase", "")).strip().lower() == "commit"
        pending_before_commit = LockClass(
            int(txn.metadata.get("_pending_policy_protection", 0))
        )
        defer_until_commit = bool(
            getattr(selected, "defer_to_commit", False)
            or getattr(self.manager, "defer_policy_protection_until_commit", False)
            or (is_commit and pending_before_commit != LockClass.NONE)
        )
        object_plan_enabled = bool(
            getattr(self.manager, "deferred_object_plan_enabled", False)
            and txn.metadata.get("_object_protection_plan") is not None
        )
        if object_plan_enabled and defer_until_commit:
            # The learned category action is converted into nominations over
            # the exact objects already observed by this attempt.  It does
            # not acquire a class-wide lock during the reasoning interval.
            self._nominate_deferred_objects(txn, planned)
            if is_commit:
                self._materialize_deferred_objects(txn)
            selected = LockAction(txn.context.action.protected)
            txn.metadata["_deferred_policy_materializations"] = int(
                txn.metadata.get("_deferred_policy_materializations", 0) or 0
            ) + int(is_commit)
        elif defer_until_commit and not is_commit:
            pending = LockClass(
                int(txn.metadata.get("_pending_policy_protection", 0))
            )
            pending |= planned.protected
            txn.metadata["_pending_policy_protection"] = int(pending)
            txn.metadata["_deferred_policy_plan_count"] = int(
                txn.metadata.get("_deferred_policy_plan_count", 0)
            ) + 1
            txn._event(
                "action-plan-deferred",
                {"planned": int(pending), "phase": str(getattr(state, "phase", ""))},
            )
            selected = LockAction(txn.context.action.protected)
        elif defer_until_commit:
            pending = LockClass(
                int(txn.metadata.pop("_pending_policy_protection", 0))
            )
            selected = LockAction(
                txn.context.action.protected | pending | planned.protected
            )
            txn.metadata["_deferred_policy_materializations"] = int(
                txn.metadata.get("_deferred_policy_materializations", 0)
            ) + int(pending != LockClass.NONE)
        self.manager.add_commit_timing(
            txn,
            "policy",
            (time.perf_counter() - policy_started) * 1000.0,
        )
        if self.manager.collect_trajectories:
            recorded_proposal = sampled_proposal if behavior_action_probabilities else planned
            self.manager.trajectory_collector.decision(
                txn.context.tid,
                state,
                int(getattr(recorded_proposal, "encoded", int(recorded_proposal.protected))),
                behavior_probability,
                blocked_time_ms=txn.context.blocked_time_ms,
                lock_count=self._lock_count(txn),
                behavior_action_probabilities=behavior_action_probabilities,
                monotone_actions=getattr(policy,'monotone_actions',True),
                **self._externality_metrics(txn),
            )
        if selected.protected != txn.context.action.protected:
            self.manager.transition_atcc_action(
                txn,
                selected,
                timeout_s=self._action_transition_timeout_s(txn),
                allow_mask_reduction=getattr(self.manager,'allow_future_action_mask_reduction',False),
            )
            # Action-transition lock wait is represented by Blocked(T), not
            # the separate inter-round agent interval in the priority formula.
            self.manager.interceptor.reset_agent_interval(txn)
        if getattr(self.manager, 'reconcile_selected_reads_at_checkpoints', False):
            self.manager.reconcile_selected_read_protection(
                txn, timeout_s=self._action_transition_timeout_s(txn))
        if getattr(self.manager, 'cost_aware_read_elision', False):
            self.manager.reconcile_costly_read_protection(txn)
        self._finish_online_observed_prefix(txn)

    @staticmethod
    def _update_native_occ_hysteresis(txn: Any, state: Any) -> None:
        """Require two cold policy windows and leave fast mode on any risk."""
        risk = bool(
            txn.context.retry_count > 0
            or txn.context.retry_conflict_mask
            or txn.context.hot_read_targets
            or txn.context.hot_write_targets
            or float(getattr(state, "global_conflict_abort_rate", 0.0) or 0.0)
            > 0.001
            or int(getattr(state, "global_waiter_count", 0) or 0) > 0
        )
        if risk:
            txn.metadata["_native_occ_stable_windows"] = 0
            txn.metadata["_native_occ_fast_ready"] = False
            return
        windows = min(
            2, int(txn.metadata.get("_native_occ_stable_windows", 0) or 0) + 1
        )
        txn.metadata["_native_occ_stable_windows"] = windows
        txn.metadata["_native_occ_fast_ready"] = windows >= 2

    def _apply_protection_guard(self, txn: Any, state: Any, selected: LockAction) -> LockAction:
        """Protect observed hot reads before late phases under conflict pressure."""
        context = dict(txn.metadata.get("context", {}) or {})
        workload = str(txn.metadata.get("workload", "")).strip().lower()
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        engineering_guards = bool(
            getattr(self.manager, "performance_guards_enabled", True)
        )
        tpcc_high_all_agent = bool(
            engineering_guards
            and workload == "tpcc"
            and int(agentic.get("background_workers", 0) or 0) == 0
        )
        tpcc_high_deferred_replay = bool(
            engineering_guards
            and workload == "tpcc"
            and txn.metadata.get("_deferred_reasoning_replay", False)
        )
        ycsb_deferred_all_agent_first_attempt = bool(
            engineering_guards
            and workload == "ycsb"
            and txn.metadata.get("_deferred_reasoning_replay", False)
            and int(agentic.get("background_workers", 0) or 0) == 0
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        )
        tpcc_high_all_agent_first_attempt = bool(
            tpcc_high_all_agent
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        )
        tpcc_high_exact_first_attempt = bool(
            (tpcc_high_all_agent or tpcc_high_deferred_replay)
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        )
        if (
            self.manager.low_conflict_occ_guard
            and workload == "ycsb"
            and str(context.get("level", "")).strip().lower() in {"low", "medium"}
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
        ):
            return LockAction(txn.context.action.protected)
        if (
            self.manager.low_conflict_occ_guard
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and txn.context.action.protected == LockClass.NONE
            and not txn.context.hot_read_targets
            and not txn.context.hot_write_targets
            and float(
                getattr(state, "global_conflict_abort_rate", 0.0) or 0.0
            )
            < 0.20
            and int(getattr(state, "global_waiter_count", 0) or 0) == 0
        ):
            # Low/medium contention is sensitive to unnecessary first-attempt
            # locks. Let an observed retry conflict activate the learned
            # monotonic protection path on the same logical transaction.
            return LockAction(LockClass.NONE)
        online_mixed_pressure = bool(
            engineering_guards
            and self.online_ycsb_high_mixed(txn)
            and (
                int(agentic.get("background_workers", 0) or 0) >= 4
                or float(
                    getattr(state, "global_background_abort_rate", 0.0) or 0.0
                )
                >= 0.05
                or int(getattr(state, "global_waiter_count", 0) or 0) > 0
                or float(
                    getattr(state, "global_conflict_abort_rate", 0.0) or 0.0
                )
                >= 0.20
            )
        )
        ycsb_high_all_agent_commit = bool(
            engineering_guards
            and workload == "ycsb"
            and int(agentic.get("background_workers", 0) or 0) == 0
            and txn.context.phase == TransactionPhase.COMMIT
        )
        if txn.context.phase not in {TransactionPhase.REFINE, TransactionPhase.COMMIT}:
            if online_mixed_pressure:
                observed_reads = set(txn.context.read_versions)
                if int(agentic.get("background_workers", 0) or 0) == 6:
                    # At the intermediate pressure bucket, two Agent slots can
                    # otherwise race past the first decision on a twice-seen
                    # key. Promote only those already-observed candidates; the
                    # 8-background-worker bucket deliberately keeps the normal
                    # HotnessTracker threshold to avoid over-coverage.
                    for object_id in observed_reads:
                        heat = self.manager.hotness_tracker.object_snapshot(
                            object_id
                        )
                        if int(heat.get("accesses", 0) or 0) >= 2:
                            txn.context.hot_read_targets.add(object_id)
                observed_hot_reads = bool(
                    observed_reads & set(txn.context.hot_read_targets)
                )
                return LockAction(
                    txn.context.action.protected
                    | (selected.protected & LockClass.HOT_READ)
                    | (LockClass.HOT_READ if observed_hot_reads else LockClass.NONE),
                    defer_to_commit=bool(getattr(selected, "defer_to_commit", False)),
                )
            return selected
        read_targets = set(txn.context.read_versions)
        hot_reads = read_targets & set(txn.context.hot_read_targets)
        hot_read_ratio = len(hot_reads) / len(read_targets) if read_targets else 0.0
        validation_pressure = max(
            0.0,
            float(getattr(state, "global_conflict_abort_rate", 0.0) or 0.0),
        )
        guarded = selected.protected
        if (
            self.manager.low_conflict_occ_guard
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and float(
                getattr(state, "global_conflict_abort_rate", 0.0) or 0.0
            )
            < 0.20
            and int(getattr(state, "global_waiter_count", 0) or 0) == 0
        ):
            # A system-wide pressure sample is insufficient evidence that a
            # particular transaction needs category-wide cold protection.
            # Preserve already-held bits, but require a concrete retry or very
            # high pressure before adding a new cold class.
            cold_classes = LockClass.COLD_READ | LockClass.COLD_WRITE
            guarded &= ~cold_classes
            guarded |= txn.context.action.protected & cold_classes
        if (
            engineering_guards
            and txn.context.retry_count <= 0
            and bool(
                txn.metadata.get("_defer_policy_write_locks", False)
            )
        ):
            # This is an optimized-path pressure guard, not paper policy
            # semantics. Pure delayed-write evaluation must retain the
            # selected write class so its behavior propensity remains exact;
            # the WLock itself is still acquired only at commit admission.
            write_classes = LockClass.HOT_WRITE | LockClass.COLD_WRITE
            guarded &= ~write_classes
            guarded |= txn.context.action.protected & write_classes
            if ycsb_high_all_agent_commit and validation_pressure >= 0.15:
                # The compiled policy may request write protection after the
                # observed explore/refine batches. Preserve only its hot-write
                # bit at the commit boundary; no future target or cold class is
                # introduced, and the WLock is acquired when the write occurs.
                guarded |= selected.protected & LockClass.HOT_WRITE
        if online_mixed_pressure:
            # Under real mixed pressure, broad action-15 protection creates a
            # write convoy. Keep only observed hot reads; actual writes use the
            # unified short commit admission, while exact retry writes upgrade
            # only when their operation is reached.
            guarded = txn.context.action.protected | (guarded & LockClass.HOT_READ)
        if tpcc_high_all_agent and txn.context.retry_count > 0:
            # Retry state already carries the exact failed objects. Keep the
            # monotonic action at those classes instead of letting a sparse
            # nearest-neighbor row expand one conflict into action-15.
            exact_retry = LockClass(int(txn.context.retry_conflict_mask) & 0xF)
            allowed = txn.context.action.protected | exact_retry
            guarded &= allowed
        if tpcc_high_exact_first_attempt:
            # The exact warehouse/district read-before-write guard below is
            # sufficient for one-warehouse TPC-C, including deferred mixed
            # replay. Broad action-1 read locks create an Agent convoy without
            # protecting any additional write dependency.
            guarded &= ~LockClass.HOT_READ
            guarded |= txn.context.action.protected & LockClass.HOT_READ
            txn.metadata["_tpcc_first_attempt_exact_only"] = True
        if ycsb_deferred_all_agent_first_attempt:
            # The observed-replay gate already admits a single short suffix
            # after reasoning completes. Category-wide read locks duplicate
            # that protection and turn the serialized suffix into lock-manager
            # work. A real failed attempt still enters exact retry protection.
            guarded &= ~LockClass.HOT_READ
            guarded |= txn.context.action.protected & LockClass.HOT_READ
        if (
            engineering_guards
            and hot_read_ratio >= 0.25
            and not tpcc_high_exact_first_attempt
            and not ycsb_deferred_all_agent_first_attempt
        ):
            guarded |= LockClass.HOT_READ
        if (
            getattr(self.manager, "force_agent_phase_protection", False)
            and workload == "ycsb"
            and txn.context.phase == TransactionPhase.REFINE
            and bool(txn.context.read_versions)
        ):
            # Once a concrete dependency has been observed in refine, protect
            # that dependency before the long agent interval. The initial
            # explore read remains optimistic and no future footprint is used.
            guarded |= LockClass.HOT_READ | LockClass.COLD_READ
        if (
            engineering_guards
            and hot_reads
            and validation_pressure >= 0.30
            and not tpcc_high_exact_first_attempt
            and not ycsb_deferred_all_agent_first_attempt
        ):
            # Global pressure justifies protecting observed hotspots, but it
            # is not transaction-specific evidence for a category-wide cold
            # read lock. Cold bits are introduced only by retry evidence.
            guarded |= LockClass.HOT_READ
        if guarded != selected.protected:
            return LockAction(
                guarded, defer_to_commit=bool(getattr(selected, "defer_to_commit", False))
            )
        return selected

    @staticmethod
    def _action_transition_timeout_s(txn: Any) -> float:
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        mixed_ycsb_high = bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and int(agentic.get("background_workers", 0) or 0) > 0
        )
        return 0.025 if mixed_ycsb_high else 5.0

    @staticmethod
    def _tpcc_all_agent_exact_retry(txn: Any) -> bool:
        agentic = dict(txn.metadata.get("agentic", {}) or {})
        return bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "tpcc"
            and int(agentic.get("background_workers", 0) or 0) == 0
            and txn.context.retry_count > 0
        )

    def _low_conflict_occ_guard(self, txn: Any, state: Any) -> bool:
        """Keep the zero-protection action while every observed risk signal is cold."""
        return bool(
            self.manager.low_conflict_occ_guard
            and txn.context.action.protected == LockClass.NONE
            and txn.context.retry_count <= 0
            and not txn.context.retry_conflict_mask
            and (self.manager.conflict_driven_fast_path or not txn.context.hot_read_targets)
            and (self.manager.conflict_driven_fast_path or not txn.context.hot_write_targets)
            and float(getattr(state, "global_conflict_abort_rate", 0.0) or 0.0)
            <= 0.001
            and int(getattr(state, "global_waiter_count", 0) or 0) == 0
        )

    def on_finish(self, txn: Any) -> None:
        self._finish_online_observed_prefix(txn)
        if not self.enabled(txn):
            return
        if txn.metadata.get("_cold_occ_fast_task", False):
            self.manager.observe_native_occ_window(
                risk=self._native_occ_outcome_risk(txn)
            )
            return
        if not self.manager.collect_trajectories:
            self.manager.observe_native_occ_window(
                risk=self._native_occ_outcome_risk(txn)
            )
        if not self.manager.collect_trajectories:
            return
        state = self._state(txn)
        committed = bool(getattr(getattr(txn, "result", None), "committed", False))
        self.manager.trajectory_collector.finish(
            txn.context.tid,
            state,
            committed=committed,
            operation_cost_ms=txn.context.operation_cost_ms,
            agent_cost_ms=txn.context.agent_cost_ms,
            retry_cost_ms=txn.context.prior_retry_cost_ms,
            blocked_time_ms=txn.context.blocked_time_ms,
            lock_count=self._lock_count(txn),
            **self._externality_metrics(txn),
        )

    def _native_occ_outcome_risk(self, txn: Any) -> bool:
        result = getattr(txn, "result", None)
        # Commit results may use explanatory text (e.g. "atomic version check
        # failed"), while the context stores its normalized conflict kind.
        # Any failed task must revoke a cold fast-path admission; matching only
        # a short list of result strings can silently miss native OCC failures.
        reason = str(getattr(txn.context, "recent_conflict_kind", "") or "").strip().lower()
        if self.manager.conflict_driven_fast_path:
            metrics = self.manager.paper_runtime_metrics()
            # Frequency alone does not imply conflicting writers. Retain the
            # original validated OCC implementation, but admit its fast path
            # from observed outcomes rather than static/shared-read frequency.
            # Never admit retries, failed attempts, protection or lock pressure.
            return bool(
                not bool(getattr(result, "committed", False))
                or txn.context.retry_count > 0
                or txn.context.retry_conflict_mask
                or txn.context.action.protected != LockClass.NONE
                or txn.context.blocked_time_ms > 0
                or float(metrics["conflict_abort_rate"]) > .001
                or int(metrics["waiter_count"]) > 0
            )
        return bool(
            not bool(getattr(result, "committed", False))
            or txn.context.action.protected != LockClass.NONE
            or txn.context.hot_read_targets
            or txn.context.hot_write_targets
            or reason in {
                "version-conflict",
                "lock-preempted",
                "lock-timeout",
                "lock-conflict",
            }
        )

    def _start_online_observed_prefix(self, txn: Any) -> None:
        if not self.online_ycsb_high_mixed(txn) or txn.metadata.get(
            "_online_prefix_active", False
        ):
            return
        txn.metadata["_online_prefix_active"] = True
        self.manager.enter_online_observed_prefix()

    def _finish_online_observed_prefix(self, txn: Any) -> None:
        if not txn.metadata.pop("_online_prefix_active", False):
            return
        self.manager.leave_online_observed_prefix()

    def _state(self, txn: Any) -> Any:
        collect_reviewer=getattr(self.manager,'collect_reviewer_contention_statistics',True)
        if not collect_reviewer and getattr(self.manager.paper_policy.snapshot(),
                'requires_reviewer_contention_statistics',True):
            raise ValueError('Installed policy requires omitted reviewer contention summaries')
        hotness_started = time.perf_counter()
        self.manager.refresh_hot_targets(txn)
        self.manager.add_commit_timing(
            txn,
            "hotness",
            (time.perf_counter() - hotness_started) * 1000.0,
        )
        if getattr(self.manager, "protection_relevant_hotness", False):
            targets = set(txn.context.write_targets)
            targets.update(key for key in txn.context.read_versions
                           if self.manager.category_read_lock_needed(txn, key))
            state = self.manager.state_collector.snapshot(
                txn.context, protection_relevant_targets=targets, include_observed_writes=True)
        else:
            state = self.manager.state_collector.snapshot(txn.context)
        # A rule baseline may use this summary to make its single threshold
        # local to an already-observed object, rather than using a global
        # abort rate.  It sees only past access/conflict counters maintained
        # by HotnessTracker; no planned or future object is exposed.
        observed_targets = set(txn.context.read_versions) | set(txn.context.write_targets)
        local_accesses = 0
        local_rate = 0.0
        local_recent_accesses = 0
        for key in observed_targets:
            if collect_reviewer:
                heat = self.manager.hotness_tracker.object_snapshot(key)
                accesses = int(heat.get("accesses", 0) or 0)
                conflicts = int(heat.get("conflicts", 0) or 0)
                local_accesses = max(local_accesses, accesses)
                if accesses:
                    local_rate = max(local_rate, conflicts / accesses)
            local_recent_accesses = max(
                local_recent_accesses,
                self.manager.hotness_tracker.recent_access_count(key, window_ms=1000.0),
            )
        metrics = self.manager.paper_runtime_metrics()
        return dataclasses.replace(
            state,
            global_active_transactions=int(metrics["active_transactions"]),
            global_waiter_count=int(metrics["waiter_count"]),
            global_abort_rate=float(metrics["abort_rate"]),
            global_throughput=float(metrics["throughput"]),
            global_avg_latency_ms=float(metrics["average_latency_ms"]),
            global_tail_latency_ms=float(metrics["tail_latency_ms"]),
            global_agent_task_throughput=float(metrics["agent_task_throughput"]),
            global_agent_task_avg_latency_ms=float(metrics["agent_task_average_latency_ms"]),
            global_agent_task_tail_latency_ms=float(metrics["agent_task_tail_latency_ms"]),
            global_conflict_abort_rate=float(metrics["conflict_abort_rate"]),
            global_background_throughput=float(metrics["background_throughput"]),
            global_background_abort_rate=float(metrics["background_abort_rate"]),
            max_observed_hot_accesses=local_accesses,
            max_observed_hot_contention_rate=local_rate,
            max_observed_recent_accesses=local_recent_accesses,
        )

    @staticmethod
    def _externality_metrics(txn: Any) -> dict[str, float | int]:
        context = txn.context
        helper = dict(agent_blocked_ms_caused=0., background_blocked_ms_caused=0.)
        coordinator = getattr(getattr(txn, "manager", None), "_coordinated_admission", None)
        if coordinator is not None and coordinator.record_externalities:
            helper = coordinator.externalities_for(context.tid)
            txn.manager.trajectory_collector.note_helper_externalities(context.tid, helper)
        return {
            "background_blocked_ms_caused": context.background_blocked_ms_caused + helper["background_blocked_ms_caused"],
            "background_aborts_caused": context.background_aborts_caused,
            "agent_blocked_ms_caused": context.agent_blocked_ms_caused + helper["agent_blocked_ms_caused"],
            "agent_aborts_caused": context.agent_aborts_caused,
        }

    @staticmethod
    def _lock_count(txn: Any) -> int:
        return len(
            txn.context.policy_read_lock_targets
            | txn.context.policy_write_lock_targets
        )
