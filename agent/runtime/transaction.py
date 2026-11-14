"""Single-plan agent transaction runtime over a versioned KV store."""

from __future__ import annotations

import dataclasses
import collections
import math
import threading
import time
from typing import Any, Dict, Iterable, Optional

from agent.cc import ConcurrencyControlRegistry, ExclusiveLockTable, LockConflict, ReservationTable, TwoPhaseLockTable
from agent.cc.locks import normalize_conflict_kind
from agent.cc.traditional import lock_conflict_result
from agent.native import load_cast_core
from agent.runtime.types import (
    ConflictDetail,
    ReadRecord,
    SnapshotValue,
    TransactionEvent,
    TransactionResult,
    TransactionState,
    WriteRecord,
)
from agent.runtime.context import (
    LockAction,
    LockClass,
    TransactionContext,
    TransactionPhase,
    TransactionStatus,
)
from agent.runtime.operation_interceptor import OperationInterceptor, TransactionHooks
from agent.runtime.atcc_lock_manager import PaperATCCLockManager
from agent.runtime.priority import PriorityConfig, PriorityManager
from agent.runtime.state_collector import StateCollector
from agent.runtime.undo_log import UndoLog
from agent.runtime.version_manager import VersionManager
from agent.runtime.paper_policy import AtomicPolicyManager, CompiledPhasePolicy
from agent.runtime.paper_hooks import PaperATCCHooks
from agent.runtime.trajectory import TrajectoryCollector
from agent.runtime.hotness import HotnessTracker

cc = load_cast_core()


@dataclasses.dataclass
class _RetryProtectionState:
    mask: LockClass = LockClass.NONE
    validation_conflicts: int = 0
    last_conflicts: tuple[ConflictDetail, ...] = ()
    protected_read_targets: frozenset[str] = frozenset()
    protected_write_targets: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class _NativeBackgroundBatchPayload:
    committed: bool
    native_admitted: bool


class AgentTransaction:
    """One logical agent task transaction with one concrete plan."""

    def __init__(
        self,
        manager: "AgentTransactionManager",
        task_id: str,
        snapshot: Dict[str, SnapshotValue],
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TransactionContext] = None,
    ):
        self.manager = manager
        self.task_id = str(task_id)
        self.snapshot = snapshot
        self.metadata = dict(metadata or {})
        self.context = context or TransactionContext(
            task_id=self.task_id,
            attempt_id=int(self.metadata.get("retry_count", 0) or 0),
            generation=int(self.metadata.get("generation", 0) or 0),
            snapshot_epoch=int(self.metadata.get("snapshot_epoch", -1)),
            retry_count=int(self.metadata.get("retry_count", 0) or 0),
            retry_validation_conflicts=int(
                self.metadata.get("retry_validation_conflicts", 0) or 0
            ),
            retry_conflict_mask=int(
                self.metadata.get("retry_conflict_mask", 0) or 0
            ),
            retry_conflict_read_targets={
                str(value)
                for value in self.metadata.get("retry_conflict_read_targets", ())
            },
            retry_conflict_write_targets={
                str(value)
                for value in self.metadata.get("retry_conflict_write_targets", ())
            },
            action=LockAction(
                LockClass(int(self.metadata.get("retry_protection_mask", 0) or 0) & 0xF)
            ),
            prior_retry_cost_ms=float(self.metadata.get("prior_retry_cost_ms", 0.0) or 0.0),
            prior_retry_blocked_ms=max(0., float(self.metadata.get("prior_retry_blocked_ms", 0.0) or 0.0)),
            recent_conflict_kind=normalize_conflict_kind(
                self.metadata.get("previous_failure_reason", "none")
            ),
            is_background=bool(self.metadata.get("paper_atcc_backend", False)),
            planned_write_targets={
                str(value)
                for value in self.metadata.get("planned_write_targets", ())
            },
        )
        self.state = TransactionState.ACTIVE
        self.started_at = time.perf_counter()
        self.events: list[TransactionEvent] = []
        self.read_set: Dict[str, ReadRecord] = {}
        self.write_set: Dict[str, WriteRecord] = {}
        self.result: Optional[TransactionResult] = None
        self._lifecycle_lock = threading.RLock()
        # Short metadata handshake only. Native wound callbacks must never
        # acquire a foreign lifecycle lock or hold this gate over lock/IO work.
        self._agent_work_gate = threading.RLock()
        self._event("begin", {"snapshot_objects": len(snapshot)})

    def _ensure_active(self) -> None:
        if self.state != TransactionState.ACTIVE or self.context.status != TransactionStatus.ACTIVE:
            if self.context.status in {TransactionStatus.ABORTING, TransactionStatus.ABORTED}:
                raise LockConflict(
                    "transaction was wounded",
                    tuple(self.context.held_read_locks | self.context.held_write_locks),
                    kind="lock-preempted",
                )
            raise RuntimeError(f"transaction is no longer active: {self.state.value}")

    def _event(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            TransactionEvent(time.perf_counter() - self.started_at, kind, dict(detail or {}))
        )
        if kind in {'read','write'} and getattr(self.manager,'paper_policy_refresh',None) is not None:
            key=str(detail['object_id'])
            value=self.snapshot[key].value if kind=='read' else self.write_set[key].value
            initial=self.snapshot[key].value if kind=='read' else self.write_set[key].base_value
            self.metadata.setdefault('_offline_replay_initial',{}).setdefault(key,str(initial))
            previous=float(self.metadata.get('_offline_replay_agent_cost_ms',0.))
            current=float(self.context.agent_cost_ms)
            self.metadata['_offline_replay_agent_cost_ms']=current
            route=self.manager.store.storage_id_for(key) if hasattr(self.manager.store,'storage_id_for') else 'default'
            self.metadata.setdefault('_offline_replay_operations',[]).append(dict(kind=kind,key=key,
                value=str(value),agent_gap_ms=max(0.,current-previous),storage_id=route))

    def check_active(self) -> None:
        """Check cancellation at an external-call boundary without holding a lock across it."""
        with self._lifecycle_lock:
            self._ensure_active()

    def validate_observed_prefix(self) -> None:
        """Conservatively reject stale observed reads before external paid work.

        This acquires no protection and makes no claim about future writes.
        Commit validation remains mandatory after this point-in-time check.
        """
        with self._lifecycle_lock:
            self._ensure_active()
            started = time.perf_counter()
            stale = self.manager._conflict_targets(
                (key, int(read.version)) for key, read in tuple(self.read_set.items()))
            self.metadata['_prefix_validation_calls'] = int(
                self.metadata.get('_prefix_validation_calls', 0)) + 1
            self.metadata['_prefix_validation_ms'] = float(
                self.metadata.get('_prefix_validation_ms', 0.)) + (time.perf_counter()-started)*1000
            self._event('prefix-validation', dict(reads=len(self.read_set), stale=list(stale)))
            if stale:
                raise LockConflict('pre-expensive observed prefix is stale', stale, kind='version-conflict')

    def read(self, object_id: str) -> SnapshotValue:
        with self._lifecycle_lock:
            operation_started = time.perf_counter()
            blocked_before_ms = self.context.blocked_time_ms
            self._ensure_active()
            key = str(object_id)
            if key not in self.manager._catalog:
                raise KeyError(f"object is not registered: {key}")
            self._ensure_snapshot(key)
            self.manager.interceptor.before_read(self, key)
            # A first-access protection hook may lock and refresh an
            # unobserved planned-write object to the current committed base.
            snapshot = self.snapshot[key]
            self.read_set.setdefault(key, ReadRecord(key, snapshot.version))
            self.manager.interceptor.after_read(self, key, snapshot.version)
            self._event("read", {"object_id": key, "version": snapshot.version})
            self.manager.interceptor.operation_finished(
                self,
                elapsed_ms=(time.perf_counter() - operation_started) * 1000.0,
                blocked_before_ms=blocked_before_ms,
            )
            # Validation still refers to the committed base version, but a
            # transaction must observe its own deferred/private writes. Never
            # publish this value or replace the committed snapshot with it.
            staged = self.write_set.get(key)
            return (SnapshotValue(value=staged.value,version=snapshot.version,exists=True)
                    if staged is not None else snapshot)

    def write(
        self,
        object_id: str,
        value: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        replace_staged: bool = False,
    ) -> "AgentTransaction":
        with self._lifecycle_lock:
            operation_started = time.perf_counter()
            blocked_before_ms = self.context.blocked_time_ms
            self._ensure_active()
            key = str(object_id)
            if key not in self.manager._catalog:
                raise KeyError(f"object is not registered: {key}")
            if key in self.write_set and not replace_staged:
                raise ValueError(f"transaction already writes object: {key}")
            self._ensure_snapshot(key)
            self.manager.interceptor.before_write(self, key)
            snapshot = self.snapshot[key]
            self.write_set[key] = WriteRecord(
                object_id=key,
                base_value=snapshot.value,
                base_version=snapshot.version,
                value=str(value),
                metadata=dict(metadata or {}),
            )
            self._event(
                "write",
                {
                    "object_id": key,
                    "base_version": snapshot.version,
                },
            )
            self.manager.interceptor.after_write(self, key)
            self.manager.interceptor.operation_finished(
                self,
                elapsed_ms=(time.perf_counter() - operation_started) * 1000.0,
                blocked_before_ms=blocked_before_ms,
            )
            return self

    def _ensure_snapshot(self, object_id: str) -> None:
        key = str(object_id)
        if key in self.snapshot:
            return
        if self.context.is_background:
            current = self.manager.store.get(key)
            self.snapshot[key] = SnapshotValue(
                value=current.value,
                version=int(current.version),
                exists=bool(current.exists),
            )
        elif (
            self.manager.paper_versioning_enabled
            and self.context.snapshot_epoch >= 0
        ):
            historical = self.manager.version_manager.read_at(
                self.context.snapshot_epoch,
                key,
            )
            if not historical.exists and self.manager.store.get(key).exists:
                detail = self.manager.version_manager.history_debug(
                    key,
                    tid=self.context.tid,
                )
                raise RuntimeError(
                    f"registered object has no historical version at epoch "
                    f"{self.context.snapshot_epoch}: {detail}"
                )
            self.snapshot[key] = historical
        else:
            current = self.manager.store.get(key)
            self.snapshot[key] = SnapshotValue(
                value=current.value,
                version=int(current.version),
                exists=bool(current.exists),
            )

    def refresh_unobserved_locked_snapshot(self, object_id: str) -> None:
        key = str(object_id)
        if key in self.read_set or key in self.write_set:
            return
        if self.manager.paper_versioning_enabled:
            self.snapshot[key] = self.manager.version_manager.read_committed(key)
        else:
            current = self.manager.store.get(key)
            self.snapshot[key] = SnapshotValue(
                value=current.value,
                version=int(current.version),
                exists=bool(current.exists),
            )

    def commit(self, strategy: str = "occ") -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            interceptor_started = time.perf_counter()
            self.manager.interceptor.before_commit(self)
            self.manager.add_commit_timing(
                self,
                "interceptor",
                (time.perf_counter() - interceptor_started) * 1000.0,
            )
            return self.manager.commit(self, strategy=strategy)

    def enter_phase(self, phase: str | TransactionPhase) -> None:
        if isinstance(phase, TransactionPhase):
            normalized = phase
        else:
            phase_name = str(phase).strip().lower()
            normalized = TransactionPhase.EXPLORE if phase_name == "plan" else TransactionPhase(phase_name)
        initial_explore = (
            normalized == TransactionPhase.EXPLORE
            and not self.context.read_versions
            and not self.context.write_targets
        )
        if (
            self.metadata.get("paper_atcc", False)
            and not self.metadata.get("_cold_occ_fast_task", False)
            and not initial_explore
        ):
            self.manager.interceptor.account_agent_interval(self)
            self.manager.refresh_atcc_priority(self)
        self.manager.interceptor.phase_change(self, normalized)
        self._event("phase", {"name": normalized.value, "source_name": str(phase)})

    def prepare_phase(self, operations: Iterable[Any]) -> None:
        # The paper policy observes completed execution only; it never inspects
        # the upcoming operation batch.
        _ = operations

    def abort(
        self,
        reason: str,
        *,
        strategy: str = "",
        conflict_object_ids: Iterable[str] = (),
    ) -> TransactionResult:
        with self._lifecycle_lock:
            self._ensure_active()
            conflict_ids = tuple(str(value) for value in conflict_object_ids)
            conflict_details, retry_mask = self.manager._prepare_retry_feedback(
                self,
                reason=reason,
                conflict_object_ids=conflict_ids,
            )
            self.context.transition(TransactionStatus.ABORTING)
            self.context.note_conflict(normalize_conflict_kind(reason))
            self.manager.interceptor.abort(self, str(reason))
            self.state = TransactionState.ABORTED
            self.context.transition(TransactionStatus.ABORTED)
            self._event("abort", {"reason": str(reason), "strategy": str(strategy)})
            self.result = self._result(
                strategy=strategy,
                committed=False,
                action="abort",
                reason=str(reason),
                conflict_object_ids=conflict_ids,
                conflict_details=conflict_details,
                retry_protection_mask=retry_mask,
            )
            if self.manager._note_runtime_outcome(self, False):
                with self.manager._lock:
                    self.manager._paper_aborts += 1
            self.manager._record(self)
            self.manager.interceptor.finish(self)
            self.manager.atcc_locks.release_all(self.context)
            self.manager.release_hotspot_admission(self)
            if self.manager.paper_versioning_enabled:
                self.manager.version_manager.finish(self.context.tid, committed=False)
            self.manager._live_transactions.pop(self.context.tid, None)
            self.manager._finalize_commit_timing(self)
            return self.result

    def abort_or_observe_wound(
        self, reason: str, *, strategy: str = "",
        conflict_object_ids: Iterable[str] = (),
    ) -> TransactionResult:
        """Finish an owner failure or await an already claimed native wound.

        Do not wait under the lifecycle lock or count a foreign abort twice.
        This is for exception handlers after an operation has unwound; ordinary
        abort() retains its existing active-transaction contract.
        """
        try:
            return self.abort(reason, strategy=strategy,
                              conflict_object_ids=conflict_object_ids)
        except LockConflict as exc:
            if exc.kind != "lock-preempted":
                raise
            self.manager.atcc_locks.synchronize_wound()
            if self.result is None or self.result.committed:
                raise RuntimeError('Native wound did not publish an aborted result') from exc
            return self.result

    def _result(
        self,
        *,
        strategy: str,
        committed: bool,
        action: str,
        reason: str = "",
        conflict_object_ids: Iterable[str] = (),
        conflict_details: Iterable[ConflictDetail] = (),
        retry_protection_mask: int = 0,
        lock_wait_s: float = 0.0,
    ) -> TransactionResult:
        return TransactionResult(
            task_id=self.task_id,
            state=self.state,
            strategy=str(strategy),
            committed=bool(committed),
            action=str(action),
            reason=str(reason),
            elapsed_s=time.perf_counter() - self.started_at,
            read_count=len(self.read_set),
            write_count=len(self.write_set),
            conflict_object_ids=tuple(str(value) for value in conflict_object_ids),
            conflict_details=tuple(conflict_details),
            retry_protection_mask=int(retry_protection_mask),
            lock_wait_s=float(lock_wait_s),
        )

    def to_trace(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
            "state": self.state.value,
            "snapshot_versions": {key: value.version for key, value in self.snapshot.items()},
            "read_set": {key: dataclasses.asdict(value) for key, value in self.read_set.items()},
            "write_set": {key: dataclasses.asdict(value) for key, value in self.write_set.items()},
            "events": [dataclasses.asdict(event) for event in self.events],
            "result": self.result.to_dict() if self.result else None,
        }


class AgentTransactionManager:
    """Thread-safe transaction manager backed by versioned KV primitives."""

    _NATIVE_BACKGROUND_YIELD_INTERVAL = 1
    _NATIVE_BACKGROUND_PRESSURE_YIELD_INTERVAL = 1
    _NATIVE_BACKGROUND_PRESSURE_WORKERS = 6
    _NATIVE_BACKGROUND_YIELD_SECONDS = 0.0005

    _KIND_MAP = {
        "generic": cc.ObjectType.kGeneric,
        "row": cc.ObjectType.kRow,
        "text": cc.ObjectType.kText,
        "counter": cc.ObjectType.kCounter,
    }
    _COMMIT_TIMING_PHASES = (
        "interceptor",
        "hotness",
        "policy",
        "lock",
        "validate",
        "install",
        "publish",
        "gc",
    )

    def __init__(
        self,
        *,
        store: Optional[Any] = None,
        storage_engines: Optional[Dict[str, Any]] = None,
        cc_registry: Optional[ConcurrencyControlRegistry] = None,
        atcc_policy: Optional[Any] = None,
        record_traces: bool = True,
        transaction_hooks: Optional[TransactionHooks] = None,
        undo_log_path: Optional[str] = None,
        paper_policy: Optional[CompiledPhasePolicy] = None,
        collect_trajectories: bool = True,
        low_conflict_occ_guard: bool = False,
        performance_guards_enabled: bool = True,
        commit_admission_priority_enabled: bool = False,
        delayed_write_apply_enabled: bool = True,
        policy_selected_write_admission_only: bool = False,
        priority_config: PriorityConfig | None = None,
        priority_enabled: bool = True,
        tracked_history_import: bool = False,
        conflict_driven_fast_path: bool = False,
        unified_commit_admission: bool = False,
        cycle_safe_hotspot_admission: bool = False,
        cycle_safe_gate_wait_ms: float = 0.0,
        coordinated_hotspot_admission: bool = False,
        round_interval_features: bool = False,
        commit_read_checkpoints: bool = False,
        complete_category_read_protection: bool = False,
        first_access_read_refresh: bool = False,
        consistent_hotness_classification: bool = False,
        exact_hotness_accounting: bool = False,
        helper_externality_accounting: bool = False,
        observed_write_read_lock_elision: bool = False,
        cost_aware_read_elision: bool = False,
        protection_relevant_hotness: bool = False,
        defer_policy_protection_until_commit: bool = False,
        deferred_object_plan_enabled: bool = False,
        deferred_object_plan_min_redo_ms: float = 0.0,
        unique_attempt_outcomes: bool = False,
    ):
        if store is not None and storage_engines is not None:
            raise ValueError("specify store or storage_engines, not both")
        if storage_engines is not None:
            from .storage_router import RoutedVersionedKVStore
            self.store = RoutedVersionedKVStore(storage_engines)
        else:
            self.store = store if store is not None else cc.Dbx1000VersionedKVStore()
        if undo_log_path and getattr(self.store, "owns_recovery", False):
            raise ValueError("durable coordinated store owns recovery; do not layer the undo log")
        self.version_manager = VersionManager(self.store,
            tracked_history_import=tracked_history_import)
        self.conflict_driven_fast_path = bool(conflict_driven_fast_path)
        self.unified_commit_admission = bool(unified_commit_admission)
        self.cycle_safe_hotspot_admission = bool(cycle_safe_hotspot_admission)
        self.coordinated_hotspot_admission = bool(coordinated_hotspot_admission)
        self.commit_read_checkpoints = bool(commit_read_checkpoints)
        self.complete_category_read_protection = bool(complete_category_read_protection)
        self.first_access_read_refresh = bool(first_access_read_refresh)
        self.consistent_hotness_classification = bool(consistent_hotness_classification)
        self.exact_hotness_accounting = bool(exact_hotness_accounting)
        self.helper_externality_accounting = bool(helper_externality_accounting)
        self.observed_write_read_lock_elision = bool(observed_write_read_lock_elision)
        self.cost_aware_read_elision = bool(cost_aware_read_elision)
        if self.cost_aware_read_elision and not self.observed_write_read_lock_elision:
            raise ValueError('cost-aware read elision requires observed-write read-lock elision')
        self.protection_relevant_hotness = bool(protection_relevant_hotness)
        # Experimental scheduling mode.  A policy may nominate additional
        # protection during explore/refine, but the nomination is materialized
        # only at the short commit checkpoint.  Existing locks are never
        # released, so the normal monotonic-lock invariant remains intact.
        # This is deliberately opt-in; the paper runtime keeps its original
        # immediate-install semantics.
        self.defer_policy_protection_until_commit = bool(
            defer_policy_protection_until_commit
        )
        if not math.isfinite(float(deferred_object_plan_min_redo_ms)) or float(deferred_object_plan_min_redo_ms) < 0:
            raise ValueError("deferred object-plan redo threshold must be finite and nonnegative")
        self.deferred_object_plan_enabled = bool(deferred_object_plan_enabled)
        self.deferred_object_plan_min_redo_ms = float(deferred_object_plan_min_redo_ms)
        self.unique_attempt_outcomes = bool(unique_attempt_outcomes)
        self._runtime_outcomes_recorded = 0
        self._runtime_duplicate_outcomes = 0
        if self.protection_relevant_hotness and not self.observed_write_read_lock_elision:
            raise ValueError("protection-relevant hotness requires observed-write read-lock elision")
        if (getattr(paper_policy, "protection_relevant_hotness", False)
                and not self.protection_relevant_hotness):
            raise ValueError("policy requires protection-relevant hotness encoding")
        self._observed_committed_write_objects: set[str] = set()
        if self.helper_externality_accounting and not self.coordinated_hotspot_admission:
            raise ValueError("helper externality accounting requires coordinated hotspot admission")
        self.cycle_safe_gate_wait_ms = float(cycle_safe_gate_wait_ms)
        if not math.isfinite(self.cycle_safe_gate_wait_ms) or self.cycle_safe_gate_wait_ms < 0:
            raise ValueError("cycle-safe gate wait must be finite and non-negative")
        if self.cycle_safe_gate_wait_ms and not self.cycle_safe_hotspot_admission:
            raise ValueError("cycle-safe gate wait requires cycle-safe hotspot admission")
        if sum((self.unified_commit_admission, self.cycle_safe_hotspot_admission,
                self.coordinated_hotspot_admission)) > 1:
            raise ValueError("Choose one hotspot admission mode, not both")
        self.cc_registry = cc_registry or ConcurrencyControlRegistry(atcc_policy=atcc_policy)
        self.exclusive_locks = ExclusiveLockTable()
        self.two_phase_locks = TwoPhaseLockTable()
        self.reservations = ReservationTable()
        self._lock = threading.RLock()
        self.tpcc_mixed_replay_gate = threading.Lock()
        # Saturated all-Agent YCSB can defer long reasoning before begin().
        # Once online conflict evidence exists, this gate serializes only the
        # short replay suffix and never receives a future access set.
        self.ycsb_observed_replay_gate = threading.Lock()
        self._online_prefix_condition = threading.Condition()
        self._online_prefix_agents = 0
        self._catalog: Dict[str, Any] = {}
        self._traces: list[Dict[str, Any]] = []
        self._record_traces = bool(record_traces)
        self.state_collector = StateCollector(round_interval_features=round_interval_features)
        self.priority_manager = PriorityManager(priority_config)
        self.priority_enabled = bool(priority_enabled)
        self.hotness_tracker = HotnessTracker()
        self.paper_policy = AtomicPolicyManager(paper_policy)
        self.trajectory_collector = TrajectoryCollector()
        self.collect_trajectories = bool(collect_trajectories)
        self.low_conflict_occ_guard = bool(low_conflict_occ_guard)
        self.performance_guards_enabled = bool(performance_guards_enabled)
        self.commit_admission_priority_enabled = bool(
            commit_admission_priority_enabled
        )
        # Orthogonal ablation switch: retain the policy-selected write class,
        # but buffer the private write and acquire its exact WLock only in the
        # short commit window. This remains an orthogonal ablation switch.
        self.delayed_write_apply_enabled = bool(delayed_write_apply_enabled)
        self.policy_selected_write_admission_only = bool(policy_selected_write_admission_only)
        hooks = transaction_hooks or PaperATCCHooks(self)
        self.interceptor = OperationInterceptor(hooks, state_collector=self.state_collector)
        self._live_transactions: Dict[str, AgentTransaction] = {}
        self.atcc_locks = PaperATCCLockManager(
            wound_callback=self._wound_context,
            priority_callback=(
                self.priority_manager.compute if self.priority_enabled else None
            ),
            contention_callback=self.hotness_tracker.observe_contention,
        )
        self.undo_log = UndoLog(undo_log_path)
        self._commit_fault_injector: Optional[Any] = None
        self._paper_attempts = 0
        self._paper_aborts = 0
        self._paper_background_aborts = 0
        self._runtime_started_at = time.monotonic()
        self.runtime_metrics_window_seconds = 1.0
        self._recent_outcomes = collections.deque()
        self._recent_agent_tasks = collections.deque()
        self._paper_metrics_cache_at = 0.0
        self._paper_metrics_cache: Dict[str, float | int] = {}
        self._retry_protection: Dict[str, _RetryProtectionState] = {}
        self.policy_controlled_retry_protection = False
        self._retry_protection_diagnostics: Dict[str, int] = collections.defaultdict(int)
        self._retry_conflict_objects: Dict[str, int] = collections.defaultdict(int)
        self._paper_versioning_enabled = False
        self._native_dirty_objects: set[str] = set()
        self._native_occ_stable_windows = 0
        self._native_occ_fast_enabled = False
        self._native_occ_fast_starts = 0
        self._native_occ_last_window_at = 0.0
        self._hotspot_admission_guard = threading.Lock()
        self._hotspot_admissions: Dict[str, threading.Lock] = {}
        self._cycle_safe_gate_diagnostics = {"admissions": 0, "bypasses": 0}
        self._coordinated_admission = None
        if self.coordinated_hotspot_admission:
            from .hotspot_admission import CoordinatedHotspotAdmission
            self._coordinated_admission = CoordinatedHotspotAdmission(
                record_externalities=self.helper_externality_accounting and self.collect_trajectories)
        self._background_sampling = threading.local()
        self._hotness_sampling = threading.local()
        self._native_background_plan_cache: Dict[
            int, tuple[tuple[str, ...], tuple[tuple[str, str], ...]] | None
        ] = {}
        # Bounded committed dependency footprints support conservative
        # serialization checks for online-observed reader bypass.  This is not
        # an access-set oracle: a footprint is appended only after operations
        # have actually executed and committed.
        self._commit_dependency_sequence = 0
        self._commit_dependency_history = collections.deque(maxlen=100_000)
        self._commit_timing_lock = threading.Lock()
        self._commit_timing_totals: Dict[str, float] = collections.defaultdict(float)
        self._commit_timing_counts: Dict[str, int] = collections.defaultdict(int)

    @property
    def paper_versioning_enabled(self) -> bool:
        with self._lock:
            return bool(self._paper_versioning_enabled)

    @property
    def native_occ_fast_enabled(self) -> bool:
        with self._lock:
            return bool(
                self.low_conflict_occ_guard and self._native_occ_fast_enabled
            )

    def note_native_occ_fast_start(self) -> None:
        with self._lock:
            self._native_occ_fast_starts += 1

    @property
    def native_occ_fast_starts(self) -> int:
        with self._lock:
            return self._native_occ_fast_starts

    def observe_native_occ_window(self, *, risk: bool) -> bool:
        """Global two-window hysteresis for the native action-0 path."""
        with self._lock:
            if risk:
                self._native_occ_stable_windows = 0
                self._native_occ_fast_enabled = False
                return False
            now = time.monotonic()
            if now - self._native_occ_last_window_at < 0.020:
                return bool(self._native_occ_fast_enabled)
            self._native_occ_last_window_at = now
            self._native_occ_stable_windows = min(
                2, self._native_occ_stable_windows + 1
            )
            if self._native_occ_stable_windows >= 2:
                self._native_occ_fast_enabled = True
            return bool(self._native_occ_fast_enabled)

    def acquire_hotspot_admission(
        self,
        txn: AgentTransaction,
        object_ids: Iterable[str],
        *,
        timeout_s: float = 5.0,
    ) -> bool:
        """Reserve already-observed hotspots in canonical order."""
        targets = tuple(sorted({str(value) for value in object_ids}))
        if self.coordinated_hotspot_admission:
            accounted = [False]
            def refresh_paid_work():
                self._refresh_hotspot_admission_priority(txn)
                accounted[0] = True
            try:
                return self._coordinated_admission.acquire(txn.context, targets,
                    timeout_s=timeout_s, native_dependencies=self.atcc_locks.wait_dependencies_snapshot,
                    priority_refresh=refresh_paid_work)
            finally:
                if accounted[0]:
                    # The helper has separately measured blocking. The next
                    # SQL must not charge this CC interval as Agent reasoning.
                    self.interceptor.reset_agent_interval(txn)
        if self.unified_commit_admission:
            if not targets:
                return False
            # Never nest a private admission gate with ATCC locks: a retry may
            # already hold an exact read lock that the gate holder needs to
            # upgrade. Admit the same revealed suffix using one lock order and
            # the existing priority/Wound-Wait scheduler for every policy.
            self.refresh_atcc_priority(txn)
            self.atcc_locks.acquire_write_set(targets, txn.context, timeout_s=timeout_s)
            observed = set(targets) & set(txn.read_set)
            stale = self._conflict_targets((key, int(txn.read_set[key].version)) for key in observed)
            if stale:
                raise LockConflict("retroactive admission validation failed", stale, kind="version-conflict")
            txn.metadata.setdefault("_unified_hotspot_targets", set()).update(targets)
            return True
        held = list(txn.metadata.get("_observed_hotspot_admissions", ()))
        held_targets = {str(target) for target, _gate in held}
        targets = tuple(target for target in targets if target not in held_targets)
        nonblocking = False
        if self.cycle_safe_hotspot_admission:
            # Private gates are a performance helper, not correctness locks.
            # Never allow an unbounded wait under an ATCC lock: the gate owner
            # may itself be waiting for that lock (even on another object).
            # Existing commit WriteSet admission and version validation remain
            # mandatory; unlike unified admission, do not acquire extra early
            # WLocks or introduce an additional retroactive validation point.
            locked = txn.context.held_read_locks or txn.context.held_write_locks
            backwards = bool(held_targets and targets and min(targets) < max(held_targets))
            if backwards or not targets:
                txn.metadata.setdefault("_cycle_safe_gate_bypass_targets", set()).update(targets)
                if backwards:
                    with self._hotspot_admission_guard:
                        self._cycle_safe_gate_diagnostics["bypasses"] += 1
                return True
            # Preserve useful admission on uncontended gates. Only the wait
            # creates a nested cycle; the default immediate try is safe under
            # existing ATCC locks. An explicit bounded wait budget allows a
            # benign short suffix to finish but expires a real cycle. Roll
            # back this call's partial gates on expiry; normal commit admission
            # and validation still arbitrate publication.
            nonblocking = bool(locked)
        if not targets:
            return False
        acquired: list[tuple[str, threading.Lock]] = []
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        if nonblocking:
            # One total bounded budget for this helper call, not per key.
            # Expiry rolls back partial gates and breaks any nested wait cycle.
            deadline = min(deadline,time.monotonic()+self.cycle_safe_gate_wait_ms/1000.)
        for target in targets:
            with self._hotspot_admission_guard:
                gate = self._hotspot_admissions.setdefault(target, threading.Lock())
            remaining = max(0.0, deadline - time.monotonic())
            obtained = (gate.acquire(blocking=False) if self.cycle_safe_gate_wait_ms <= 0
                        else gate.acquire(timeout=remaining)) if nonblocking else gate.acquire(timeout=remaining)
            if not obtained:
                for _key, held in reversed(acquired):
                    held.release()
                if nonblocking:
                    txn.metadata.setdefault("_cycle_safe_gate_bypass_targets", set()).update(targets)
                    with self._hotspot_admission_guard:
                        self._cycle_safe_gate_diagnostics["bypasses"] += 1
                    return True
                return False
            acquired.append((target, gate))
        txn.metadata["_observed_hotspot_admissions"] = held + acquired
        if self.cycle_safe_hotspot_admission:
            with self._hotspot_admission_guard:
                self._cycle_safe_gate_diagnostics["admissions"] += 1
        return True

    def cycle_safe_gate_diagnostics(self) -> Dict[str, int]:
        """Whole helper reservations/bypasses, not locks or logical commits."""
        with self._hotspot_admission_guard:
            return dict(self._cycle_safe_gate_diagnostics)

    def coordinated_gate_diagnostics(self) -> Dict[str, float]:
        if self._coordinated_admission is None:
            return {}
        return self._coordinated_admission.snapshot()["diagnostics"]

    def _refresh_hotspot_admission_priority(self, txn: AgentTransaction) -> None:
        # Called only on the contended coordinated-helper branch, outside its
        # condition. SQL hooks last accounted the interval at the previous SQL;
        # the caller may have since completed Agent reasoning/external work.
        # Account that observed elapsed interval BEFORE ranking queued paid
        # work. Do not change the formula, anticipate future work or charge
        # the forthcoming helper wait here. The uncontended fast path and the
        # default legacy helper never call this callback.
        self.interceptor.account_agent_interval(txn)
        self.refresh_atcc_priority(txn)

    @staticmethod
    def release_hotspot_admission(txn: AgentTransaction) -> None:
        coordinator = txn.manager._coordinated_admission
        if coordinator is not None:
            coordinator.release_all(txn.context)
        acquired = list(txn.metadata.pop("_observed_hotspot_admissions", ()))
        for _target, gate in reversed(acquired):
            gate.release()

    def _enable_paper_versioning(self) -> None:
        self.observe_native_occ_window(risk=True)
        with self._lock:
            if self._paper_versioning_enabled:
                return
            for object_id in sorted(self._native_dirty_objects):
                self.version_manager.synchronize_object(object_id)
            self._native_dirty_objects.clear()
            self._paper_versioning_enabled = True

    def _inject_commit_fault(self, stage: str, txn: AgentTransaction) -> None:
        injector = self._commit_fault_injector
        if injector is not None:
            injector(str(stage), txn)

    def is_hot(self, object_id: str) -> bool:
        """Sample access heat; contention events are always recorded separately.

        The first two accesses on a worker are sampled so a newly contended run
        can react quickly.  Steady-state accesses then update the shared heat
        table once every eight operations, avoiding metadata work on every
        native OCC operation.
        """
        if self.exact_hotness_accounting:
            # Actual issued operations are counted once by the interceptor.
            # Repeated policy/hook queries must not create artificial accesses.
            return self.hotness_tracker.is_hot(str(object_id))
        count = int(getattr(self._hotness_sampling, "count", 0) or 0) + 1
        self._hotness_sampling.count = count
        key = str(object_id)
        if count <= 2 or count % 8 == 0:
            return self.hotness_tracker.observe_access(key)
        if self.consistent_hotness_classification:
            # Sampling controls metadata updates, not the observed object's
            # lock class. An unsampled access must not demote a known hotspot
            # to cold while refresh_hot_targets() still classifies it as hot.
            return self.hotness_tracker.is_hot(key)
        return False

    def peek_hot(self, object_id: str, *, total_accesses: int | None = None) -> bool:
        _ = total_accesses
        return self.hotness_tracker.is_hot(str(object_id))

    def refresh_hot_targets(self, txn: AgentTransaction) -> None:
        read_targets = set(txn.read_set) | set(txn.context.read_versions)
        write_targets = set(txn.write_set) | set(txn.context.write_targets)
        if self.consistent_hotness_classification:
            # A key in both footprints must not straddle two expiry/ranking
            # instants. Observe the union once, then project the SAME snapshot.
            hot = self.hotness_tracker.hot_targets(read_targets | write_targets)
            txn.context.hot_read_targets = read_targets & hot
            txn.context.hot_write_targets = write_targets & hot
        else:
            # Preserve legacy experiments' separate observation points.
            txn.context.hot_read_targets = self.hotness_tracker.hot_targets(read_targets)
            txn.context.hot_write_targets = self.hotness_tracker.hot_targets(write_targets)

    def transition_atcc_action(self, txn: AgentTransaction, action: Any, *, timeout_s: float = 5.0,
                               reconcile_reads: bool = False,allow_mask_reduction: bool = False) -> None:
        """Retroactively validate and lock objects newly protected by an action."""
        if not self.paper_versioning_enabled:
            self._enable_paper_versioning()
        online_observed = (
            str(txn.metadata.get("access_set_visibility", "")).strip().lower()
            == "online_observed"
        )
        if not online_observed:
            self.ensure_snapshot_epoch(txn)
        previous = txn.context.action
        added = (action.protected & ~previous.protected if allow_mask_reduction
                 else action.added_since(previous))
        self.refresh_hot_targets(txn)
        if (reconcile_reads and action.protected==previous.protected
                and not (action.protected & (LockClass.HOT_READ | LockClass.COLD_READ))):
            # No selected read class can become eligible. Preserve the normal
            # versioning/snapshot/hotness side effects above, but avoid scanning
            # both footprints for a reconciliation that cannot acquire a lock.
            # Existing held locks (including revisable-mask leftovers) remain.
            return
        defer_write_locks = bool(
            txn.metadata.get("_defer_policy_write_locks", False)
        )
        allow_historical_read_lock = bool(
            not online_observed
            and not txn.context.planned_write_targets
            and not txn.write_set
        )
        read_targets = []
        write_targets: dict[str, int] = {}
        # Explicit engineering variant: category changes affect subsequently
        # issued reads, while old observations still undergo final validation.
        # Do not use this switch for declared/offline footprints, or release
        # locks already held by the attempt. Legacy category backfill remains
        # the default.
        prospective_reads = bool(online_observed and getattr(
            self, 'prospective_category_read_protection', False))
        for object_id, read in txn.read_set.items():
            hot = object_id in txn.context.hot_read_targets
            planned_write = object_id in txn.context.planned_write_targets
            if (
                planned_write
                and action.protects(hot=hot, write=True)
                and not previous.protects(hot=hot, write=True)
            ):
                if defer_write_locks:
                    txn.context.policy_write_lock_targets.add(str(object_id))
                else:
                    write_targets[object_id] = int(read.version)
            elif (not prospective_reads
                  and action.protects(hot=hot, write=False)
                  and (not previous.protects(hot=hot, write=False) or reconcile_reads)
                  and object_id not in txn.context.held_read_locks
                  and object_id not in txn.context.held_write_locks
                  and self.category_read_lock_needed(txn, object_id)):
                read_targets.append((object_id, int(read.version)))
        for object_id, write in txn.write_set.items():
            hot = object_id in txn.context.hot_write_targets
            if action.protects(hot=hot, write=True) and not previous.protects(hot=hot, write=True):
                if defer_write_locks:
                    txn.context.policy_write_lock_targets.add(str(object_id))
                else:
                    write_targets[object_id] = int(write.base_version)
        if (online_observed and len(read_targets) > 1
                and not getattr(self, "complete_category_read_protection", False)):
            # One exact observed root is sufficient to desynchronize the long
            # reasoning window. Holding several RLocks and later upgrading a
            # subset creates a multi-key upgrade convoy.
            checkpoint_target = txn.metadata.get("_commit_read_checkpoint_target") if self.commit_read_checkpoints else None
            checkpoint_reads = [row for row in read_targets if row[0] == checkpoint_target]
            # A post-read decision should protect the just-observed dependency,
            # not an unrelated older hot root. It must already be a matching
            # read_set entry; preserve the one-root bound and legacy fallback.
            read_targets = checkpoint_reads or [
                max(
                    read_targets,
                    key=lambda row: int(
                        self.hotness_tracker.object_snapshot(row[0]).get(
                            "accesses", 0
                        )
                        or 0
                    ),
                )
            ]
        try:
            for object_id, version in sorted(read_targets):
                self.atcc_locks.validate_and_rlock(
                    object_id,
                    txn.context,
                    version,
                    lambda key=object_id, observed=version: (
                        int(observed)
                        if allow_historical_read_lock
                        and self.version_manager.can_lock_pinned_version(
                            txn.context.snapshot_epoch,
                            key,
                            int(observed),
                            tid=txn.context.tid,
                        )
                        else int(self.version_manager.read_committed(key).version)
                    ),
                    timeout_s=timeout_s,
                )
                txn.context.policy_read_lock_targets.add(str(object_id))
                if online_observed:
                    txn.metadata.setdefault(
                        "_online_bypass_read_targets", set()
                    ).add(str(object_id))
            for object_id, version in sorted(write_targets.items()):
                self.atcc_locks.validate_and_wlock(
                    object_id,
                    txn.context,
                    version,
                    lambda key=object_id: int(
                        self.version_manager.read_committed(key).version
                    ),
                    timeout_s=timeout_s,
                )
                txn.context.policy_write_lock_targets.add(str(object_id))
            if action.protected != previous.protected or not reconcile_reads:
                self.interceptor.action_change(txn, action,allow_mask_reduction=allow_mask_reduction)
                detail={"added": int(added), "action": int(action.protected)}
                if not action.expands(previous):
                    detail['removed_future_classes']=int(previous.protected & ~action.protected)
                    detail['held_locks_released']=False
                txn._event("action-change", detail)
            elif read_targets:
                txn._event("selected-read-reconciliation", {"targets": len(read_targets),
                           "action": int(action.protected)})
        except LockConflict as exc:
            self.atcc_locks.release_all(txn.context)
            txn.abort(
                exc.reason,
                strategy="paper-atcc",
                conflict_object_ids=exc.targets,
            )
            raise

    def reconcile_selected_read_protection(self, txn: AgentTransaction, *, timeout_s: float = 5.0) -> None:
        """Materialize selected read protection for newly eligible observations.

        Used only by the opt-in checkpoint hook. This does not change the
        category action, predict future accesses, release locks or skip version
        validation. Existing one-root/category-elision settings still apply.
        """
        txn._ensure_active()
        self.transition_atcc_action(txn, txn.context.action, timeout_s=timeout_s,
                                    reconcile_reads=True)

    def refresh_atcc_priority(self, txn: AgentTransaction) -> int:
        if not self.priority_enabled:
            return 0
        return self.priority_manager.refresh(txn.context, self.atcc_locks)

    def paper_runtime_metrics(self) -> Dict[str, float | int]:
        now = time.monotonic()
        with self._lock:
            if now - self._paper_metrics_cache_at < 0.010 and self._paper_metrics_cache:
                cached = dict(self._paper_metrics_cache)
                cached["active_transactions"] = len(self._live_transactions)
                cached["waiter_count"] = self.atcc_locks.global_waiter_count()
                cached["background_aborts"] = self._paper_background_aborts
                return cached
            window = self.runtime_metrics_window_seconds
            while self._recent_outcomes and now - self._recent_outcomes[0][0] > window:
                self._recent_outcomes.popleft()
            while self._recent_agent_tasks and now - self._recent_agent_tasks[0][0] > window:
                self._recent_agent_tasks.popleft()
            outcomes = tuple(self._recent_outcomes)
            agent_tasks = tuple(self._recent_agent_tasks)
            active_transactions = len(self._live_transactions)
            background_aborts = self._paper_background_aborts
        commits = [row for row in outcomes if row[1]]
        agent_attempts = [row for row in outcomes if not row[3]]
        background_attempts = [row for row in outcomes if row[3]]
        committed_background = [row for row in background_attempts if row[1]]
        committed_agent_tasks = [row for row in agent_tasks if row[1]]
        elapsed = max(0.1, min(window, now - self._runtime_started_at))
        latencies = sorted(row[2] for row in commits)
        tail_index = max(0, min(len(latencies) - 1, int(0.99 * len(latencies))))
        agent_task_latencies = sorted(row[2] for row in committed_agent_tasks)
        agent_tail_index = max(
            0,
            min(len(agent_task_latencies) - 1, int(0.99 * len(agent_task_latencies))),
        )
        conflict_kinds = {"version-conflict", "lock-preempted", "lock-timeout", "lock-conflict"}
        metrics = {
            "active_transactions": active_transactions,
            "waiter_count": self.atcc_locks.global_waiter_count(),
            "abort_rate": (
                sum(row[5] for row in outcomes if not row[1])
                / sum(row[5] for row in outcomes)
                if outcomes else 0.0
            ),
            "throughput": sum(row[5] for row in commits) / elapsed,
            "average_latency_ms": (
                sum(row[2] * row[5] for row in commits)
                / sum(row[5] for row in commits)
                if commits else 0.0
            ),
            "tail_latency_ms": latencies[tail_index] if latencies else 0.0,
            "agent_task_throughput": len(committed_agent_tasks) / elapsed,
            "agent_task_average_latency_ms": (
                sum(agent_task_latencies) / len(agent_task_latencies)
                if agent_task_latencies else 0.0
            ),
            "agent_task_tail_latency_ms": (
                agent_task_latencies[agent_tail_index] if agent_task_latencies else 0.0
            ),
            "conflict_abort_rate": (
                sum(
                    row[5]
                    for row in agent_attempts
                    if not row[1] and row[4] in conflict_kinds
                )
                / sum(row[5] for row in agent_attempts)
                if agent_attempts else 0.0
            ),
            "background_throughput": (
                sum(row[5] for row in committed_background) / elapsed
            ),
            "background_abort_rate": (
                sum(row[5] for row in background_attempts if not row[1])
                / sum(row[5] for row in background_attempts)
                if background_attempts else 0.0
            ),
            "background_aborts": background_aborts,
        }
        with self._lock:
            self._paper_metrics_cache_at = now
            self._paper_metrics_cache = dict(metrics)
        return metrics

    def _note_runtime_outcome(self, txn: AgentTransaction, committed: bool) -> bool:
        """Claim attempt accounting once; do not skip late-owner cleanup/feedback.

        Opt-in keeps legacy accounting reproducible. The claim lives on the
        attempt object (not the logical task/tid) and uses the existing manager
        lock, never a foreign lifecycle lock from the wound callback.
        Unsampled background attempts still claim their abort counter once.
        """
        sample_rate = int(txn.metadata.get("_background_metric_sample_rate", 1) or 1)
        sampled = sample_rate <= 1 or txn.metadata.get("_background_metric_sampled", False)
        if not self.unique_attempt_outcomes and not sampled:
            return True  # Legacy: no metric row, but caller still counts abort.
        with self._lock:
            if self.unique_attempt_outcomes:
                if getattr(txn, "_runtime_outcome_recorded", False):
                    self._runtime_duplicate_outcomes += 1
                    return False
                txn._runtime_outcome_recorded = True
                self._runtime_outcomes_recorded += 1
            if sampled:
                self._recent_outcomes.append(
                    (
                        time.monotonic(),
                        bool(committed),
                        (time.perf_counter() - txn.started_at) * 1000.0,
                        bool(txn.context.is_background),
                        normalize_conflict_kind(txn.context.recent_conflict_kind) if not committed else "none",
                        sample_rate,
                    )
                )
        return True

    def runtime_outcome_diagnostics(self) -> Dict[str, int]:
        with self._lock:
            return {"recorded": self._runtime_outcomes_recorded,
                    "duplicates_suppressed": self._runtime_duplicate_outcomes}

    def _commit_timing_bucket(self, txn: AgentTransaction) -> dict[str, float] | None:
        if not txn.metadata.get("_commit_timing_sampled", False):
            return None
        return txn.metadata.setdefault("_commit_timing_ms", {})

    def add_commit_timing(
        self,
        txn: AgentTransaction,
        phase: str,
        elapsed_ms: float,
    ) -> None:
        bucket = self._commit_timing_bucket(txn)
        normalized = str(phase).strip().lower()
        if bucket is None or normalized not in self._COMMIT_TIMING_PHASES:
            return
        bucket[normalized] = bucket.get(normalized, 0.0) + max(
            0.0, float(elapsed_ms)
        )

    def _finalize_commit_timing(self, txn: AgentTransaction) -> None:
        if txn.metadata.get("_defer_commit_timing_finalize", False):
            return
        bucket = txn.metadata.pop("_commit_timing_ms", None)
        if not bucket or txn.metadata.get("_commit_timing_recorded", False):
            return
        txn.metadata["_commit_timing_recorded"] = True
        role = (
            "background"
            if txn.context.is_background or txn.metadata.get("runtime_background", False)
            else "agent"
        )
        weight = int(txn.metadata.get("_commit_timing_sample_rate", 1) or 1)
        with self._commit_timing_lock:
            self._commit_timing_counts["transactions"] += weight
            self._commit_timing_counts[f"{role}_transactions"] += weight
            self._commit_timing_counts["samples"] += 1
            self._commit_timing_counts[f"{role}_samples"] += 1
            for phase in self._COMMIT_TIMING_PHASES:
                if phase not in bucket:
                    continue
                value = max(0.0, float(bucket[phase]))
                self._commit_timing_totals[phase] += value * weight
                self._commit_timing_counts[phase] += weight
                key = f"{role}_{phase}"
                self._commit_timing_totals[key] += value * weight
                self._commit_timing_counts[key] += weight

    def commit_timing_diagnostics(self) -> Dict[str, float | int]:
        with self._commit_timing_lock:
            result: Dict[str, float | int] = dict(self._commit_timing_counts)
            for role in ("", "agent_", "background_"):
                for phase in self._COMMIT_TIMING_PHASES:
                    key = f"{role}{phase}"
                    count = int(self._commit_timing_counts.get(key, 0))
                    total = float(self._commit_timing_totals.get(key, 0.0))
                    result[f"{key}_ms_total"] = total
                    result[f"{key}_ms_mean"] = total / count if count else 0.0
            return result

    def reset_measurement_diagnostics(self) -> None:
        """Keep warmed runtime state but start measurement-only counters."""
        self.atcc_locks.reset_diagnostics()
        self.version_manager.reset_diagnostics()
        with self._lock:
            self._retry_protection_diagnostics.clear()
            self._retry_conflict_objects.clear()
        with self._commit_timing_lock:
            self._commit_timing_totals.clear()
            self._commit_timing_counts.clear()

    def next_background_metric_sample_rate(self, sample_rate: int = 16) -> int:
        """Return a weighted sampling rate for this background worker."""
        rate = max(1, int(sample_rate))
        sequence = int(getattr(self._background_sampling, "sequence", 0)) + 1
        self._background_sampling.sequence = sequence
        return rate if (sequence - 1) % rate == 0 else 0

    def _yield_after_native_background_commit(self, background_workers: int) -> None:
        sequence = int(
            getattr(self._background_sampling, "native_commit_sequence", 0)
        ) + 1
        self._background_sampling.native_commit_sequence = sequence
        interval = (
            self._NATIVE_BACKGROUND_PRESSURE_YIELD_INTERVAL
            if int(background_workers) >= self._NATIVE_BACKGROUND_PRESSURE_WORKERS
            else self._NATIVE_BACKGROUND_YIELD_INTERVAL
        )
        if sequence % interval == 0:
            time.sleep(self._NATIVE_BACKGROUND_YIELD_SECONDS)

    def try_native_background_batch(
        self,
        task_id: str,
        checks: Iterable[tuple[str, int]],
        writes: Iterable[tuple[str, Any]],
        *,
        sample_metrics: bool = False,
        background_workers: int = 0,
        allow_reader_bypass: bool = False,
    ) -> tuple[bool, bool | None]:
        """Execute a short background OCC batch without transaction metadata.

        ``handled`` is false when the ordinary transaction path must re-execute
        the stored procedure. ``committed=None`` is an admission-only deferral:
        no transactional work was exposed and the scheduler should move to a
        different row without counting an abort.
        """
        self.atcc_locks.note_background_native_batch("attempt")
        if self.undo_log.path is not None:
            self.atcc_locks.note_background_native_batch("unsupported_fallback")
            return False, False

        check_versions: dict[str, int] = {}
        for object_id, version in checks:
            key = str(object_id)
            observed = int(version)
            previous = check_versions.setdefault(key, observed)
            if previous != observed:
                self.atcc_locks.note_background_native_batch("validation_failure")
                return True, False
        write_values: dict[str, str] = {}
        for object_id, value in writes:
            key = str(object_id)
            if key in write_values:
                self.atcc_locks.note_background_native_batch("unsupported_fallback")
                return False, False
            write_values[key] = str(value)

        check_rows = tuple(check_versions.items())
        write_rows = tuple(write_values.items())
        if not write_rows:
            committed = bool(self.store.batch_put_if_version(check_rows, ()))
            if committed:
                self._record_commit_dependency(
                    (object_id for object_id, _version in check_rows),
                    (),
                )
            self.atcc_locks.note_background_native_batch(
                "read_only_commit" if committed else "validation_failure"
            )
            return True, committed

        context = TransactionContext(
            task_id=str(task_id),
            attempt_id=0,
            generation=0,
            is_background=True,
            planned_write_targets=set(write_values),
        )
        background_sample_rate = (
            self.next_background_metric_sample_rate()
            if sample_metrics
            else 1
        )
        def publish() -> _NativeBackgroundBatchPayload:
            native_admitted, committed = self.version_manager.try_native_publish(
                write_values,
                lambda: self.store.batch_put_if_version(check_rows, write_rows),
                background=True,
                # Weight sampled changes back to the original volume. Most
                # native commits then avoid the VersionManager metadata lock.
                background_sample_rate=background_sample_rate,
            )
            return _NativeBackgroundBatchPayload(
                committed=bool(native_admitted and committed),
                native_admitted=bool(native_admitted),
            )

        admitted, payload = self.atcc_locks.try_uncontended_background_publish(
            write_values,
            context,
            publish,
            allow_reader_bypass=bool(allow_reader_bypass),
        )
        if not admitted and allow_reader_bypass:
            # Online-observed Agent write protection is acquired only when the
            # write is actually reached, so its remaining critical section is
            # short.  Wait once on the lock-manager condition and retry the
            # native OCC publication instead of falling back to the much more
            # expensive Python transaction/backoff path.  Admission and store
            # validation remain atomic on the retry; no lock is bypassed here.
            ready = self.atcc_locks.wait_for_background_admission(
                write_values,
                timeout_s=0.005,
                allow_reader_bypass=True,
            )
            if ready:
                admitted, payload = self.atcc_locks.try_uncontended_background_publish(
                    write_values,
                    context,
                    publish,
                    allow_reader_bypass=True,
                )
        if not admitted:
            self.atcc_locks.note_background_native_batch("admission_fallback")
            if allow_reader_bypass:
                context.transition(TransactionStatus.ABORTING)
                context.transition(TransactionStatus.ABORTED)
                return True, None
            return False, False
        if not payload.native_admitted:
            self.atcc_locks.note_background_native_batch("pin_fallback")
            return False, False
        if payload.committed:
            self._record_commit_dependency(
                (object_id for object_id, _version in check_rows),
                write_values,
            )
            context.transition(TransactionStatus.COMMITTED)
            self.atcc_locks.note_background_native_batch("commit")
            # Native background batches are intentionally short and can form
            # a tight GIL loop. Periodic yielding preserves Agent progress at
            # full NUMA-node utilization without throttling every commit.
            self._yield_after_native_background_commit(background_workers)
            return True, True
        context.transition(TransactionStatus.ABORTING)
        context.transition(TransactionStatus.ABORTED)
        self.atcc_locks.note_background_native_batch("validation_failure")
        if allow_reader_bypass:
            # A native validation race has not exposed any update. Re-execute
            # the same stored-procedure operations once through the ordinary
            # path with a fresh observed snapshot, so it remains one logical
            # background attempt rather than an artificial fast-path abort.
            return False, False
        return True, False

    def note_agent_task_outcome(self, *, committed: bool, latency_ms: float) -> None:
        """Publish one logical task outcome for the paper Delta-Psys window."""
        with self._lock:
            self._recent_agent_tasks.append(
                (time.monotonic(), bool(committed), max(0.0, float(latency_ms)))
            )
            self._paper_metrics_cache_at = 0.0

    def note_background_abort(self) -> None:
        with self._lock:
            self._paper_background_aborts += 1

    def enter_online_observed_prefix(self) -> None:
        with self._online_prefix_condition:
            self._online_prefix_agents += 1

    def leave_online_observed_prefix(self) -> None:
        with self._online_prefix_condition:
            if self._online_prefix_agents <= 0:
                return
            self._online_prefix_agents -= 1
            if self._online_prefix_agents == 0:
                self._online_prefix_condition.notify_all()

    def wait_for_online_observed_prefix(self, *, timeout_s: float = 0.005) -> bool:
        """Give an observed initial batch a short admission-free interval."""

        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._online_prefix_condition:
            while self._online_prefix_agents > 0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return False
                self._online_prefix_condition.wait(timeout=remaining)
            return True

    def sample_online_prefix_admission(self, *, one_in: int = 6) -> bool:
        """Sample short prefix protection without globally pausing background."""

        rate = max(1, int(one_in))
        sequence = int(
            getattr(self._background_sampling, "online_prefix_sequence", 0)
        ) + 1
        self._background_sampling.online_prefix_sequence = sequence
        return sequence % rate == 0

    def _record_commit_dependency(
        self,
        read_targets: Iterable[str],
        write_targets: Iterable[str],
    ) -> None:
        """Record only the observed footprint of one completed transaction."""

        reads = frozenset(str(value) for value in read_targets)
        writes = frozenset(str(value) for value in write_targets)
        with self._lock:
            if self.observed_write_read_lock_elision:
                self._observed_committed_write_objects.update(writes)
            self._commit_dependency_sequence += 1
            self._commit_dependency_history.append(
                (self._commit_dependency_sequence, reads, writes)
            )

    def _costly_observed_write_suffix(self, txn: AgentTransaction) -> bool:
        """Default-off conservative fallback after paid work and actual writes."""
        return bool(getattr(self, 'cost_aware_read_elision', False)
            and txn.context.phase == TransactionPhase.COMMIT
            and txn.context.write_targets
            and txn.context.agent_cost_ms >= self.priority_manager.config.interval_quantum_ms)

    def reconcile_costly_read_protection(self, txn: AgentTransaction) -> None:
        """Install selected protection when an observed read stops being elidable.

        Eligibility can change without adding an action bit (e.g. HotAlways
        selected cold-read protection before any writes were observed). Keep
        that control fully effective too. Never protect an unobserved key or
        override the policy action; normal version validation remains required.
        """
        if not self._costly_observed_write_suffix(txn):
            return
        # A late lock can only be useful for an already-observed read that has
        # become an actual write dependency in this attempt.  Protecting the
        # whole read set here would turn private refinement reads into a commit
        # convoy under long reasoning.  ``write_set`` is populated only after
        # the write operation has been issued, so this uses no future plan.
        for key, read in sorted(txn.read_set.items()):
            if key not in txn.write_set:
                continue
            if key in txn.context.held_read_locks or key in txn.context.held_write_locks:
                continue
            hot = key in txn.context.hot_read_targets
            if not txn.context.action.protects(hot=hot, write=False):
                continue
            self.atcc_locks.validate_and_rlock(key, txn.context, int(read.version),
                lambda object_id=key: int(self.version_manager.read_committed(object_id).version))
            txn.context.policy_read_lock_targets.add(key)

    def category_read_lock_needed(self, txn: AgentTransaction, object_id: str) -> bool:
        """Default-off category elision from past writes/contention, not immutability.

        This does not authorize validation bypass or release an existing lock.
        Exact retry/version-risk protections are handled separately by hooks.
        A previously unseen writer can still invalidate an unprotected read;
        ordinary commit validation and retries remain mandatory.
        """
        if not self.observed_write_read_lock_elision:
            return True
        if self._costly_observed_write_suffix(txn):
            return True
        key = str(object_id)
        if (key in txn.context.retry_conflict_read_targets
                or key in txn.context.retry_conflict_write_targets
                or key in txn.metadata.get("_version_risk_read_targets", ())):
            return True
        if key in txn.write_set:
            return True
        with self._lock:
            if key in self._observed_committed_write_objects:
                return True
        heat = self.hotness_tracker.object_snapshot(key)
        return any(float(heat.get(field, 0) or 0) > 0 for field in
                   ("conflicts", "validation_failures", "lock_wait_events", "wounds"))

    def _safe_online_serialization_before_targets(
        self,
        txn: AgentTransaction,
        candidates: Iterable[str],
    ) -> set[str]:
        """Return changed reads safe to serialize before intervening commits.

        A bypassed read can remain valid in the serialization graph when all
        intervening writers create only Agent->committed rw-antidependencies.
        A reverse edge is safe only when it precedes every Agent->committed
        edge, leaving a valid insertion point in commit order. Otherwise a
        cycle is possible and ordinary version validation remains mandatory.
        A truncated history also falls back to validation.
        """

        if (
            str(txn.metadata.get("access_set_visibility", "")).strip().lower()
            != "online_observed"
        ):
            return set()
        candidate_set = {str(value) for value in candidates}
        if not candidate_set:
            return set()
        changed = {
            object_id
            for object_id in candidate_set
            if object_id in txn.read_set
            and int(self.store.get_version(object_id))
            != int(txn.read_set[object_id].version)
        }
        if not changed:
            return set()
        start_sequence = int(
            txn.metadata.get("_commit_dependency_start_seq", 0) or 0
        )
        with self._lock:
            current_sequence = int(self._commit_dependency_sequence)
            history = tuple(self._commit_dependency_history)
        if current_sequence <= start_sequence or not history:
            return set()
        if int(history[0][0]) > start_sequence + 1:
            return set()
        later_writes: set[str] = set()
        first_forward_sequence: int | None = None
        last_reverse_sequence: int | None = None
        agent_reads = set(txn.read_set)
        agent_writes = set(txn.write_set)
        for sequence, reads, writes in history:
            if int(sequence) <= start_sequence:
                continue
            later_writes.update(writes)
            if agent_reads & set(writes):
                first_forward_sequence = (
                    int(sequence)
                    if first_forward_sequence is None
                    else min(first_forward_sequence, int(sequence))
                )
            if agent_writes & set(reads):
                last_reverse_sequence = (
                    int(sequence)
                    if last_reverse_sequence is None
                    else max(last_reverse_sequence, int(sequence))
                )
        if (
            first_forward_sequence is not None
            and last_reverse_sequence is not None
            and last_reverse_sequence >= first_forward_sequence
        ):
            return set()
        if not changed.issubset(agent_reads & later_writes):
            return set()
        return changed

    @property
    def backend_name(self) -> str:
        return str(self.store.backend_name)

    def register_object(self, object_id: str, initial_value: Any, *, kind: str = "generic",
                        storage_id: str = "default") -> None:
        if kind not in self._KIND_MAP:
            raise ValueError(f"unsupported object kind: {kind}")
        key = str(object_id)
        with self._lock:
            if key in self._catalog:
                raise ValueError(f"object already registered: {key}")
            if hasattr(self.store, "register_route"):
                self.store.register_route(key, storage_id)
            elif storage_id != "default":
                raise ValueError("non-default storage_id requires storage_engines")
            try:
                self.store.put(key, str(initial_value))
            except BaseException:
                if hasattr(self.store, "unregister_route"):
                    self.store.unregister_route(key)
                raise
            self._catalog[key] = self._KIND_MAP[kind]
            self.version_manager.register_object(key)

    def attach_existing_object(self, object_id: str, *, kind: str = "generic",
                               storage_id: str = "default") -> None:
        """Rebuild a fresh manager catalog after a durable store has recovered.

        Never rewrites the persisted value/version. Use before admitting tasks.
        """
        if kind not in self._KIND_MAP:
            raise ValueError(f"unsupported object kind: {kind}")
        key = str(object_id)
        with self._lock:
            if self._live_transactions:
                raise RuntimeError("attach existing objects before admitting transactions")
            if key in self._catalog:
                raise ValueError(f"object already registered: {key}")
            if hasattr(self.store, "storage_id_for"):
                if self.store.storage_id_for(key) != storage_id:
                    raise ValueError("persisted storage route differs")
            elif storage_id != "default":
                raise ValueError("non-default storage_id requires a routed store")
            if not self.store.get(key).exists:
                raise ValueError(f"persisted object missing: {key}")
            self._catalog[key] = self._KIND_MAP[kind]
            self.version_manager.register_object(key)

    def register_empty_object(self, object_id: str, *, kind: str = "row",
                              storage_id: str = "default") -> None:
        """Register an insert target without publishing a placeholder value."""
        if kind not in self._KIND_MAP:
            raise ValueError(f"unsupported object kind: {kind}")
        key = str(object_id)
        with self._lock:
            if key in self._catalog:
                return
            if hasattr(self.store, "register_route"):
                self.store.register_route(key, storage_id)
            elif storage_id != "default":
                raise ValueError("non-default storage_id requires storage_engines")
            try:
                if self.store.get(key).exists:
                    raise ValueError(f"unregistered object already exists: {key}")
            except BaseException:
                if hasattr(self.store, "unregister_route"):
                    self.store.unregister_route(key)
                raise
            self._catalog[key] = self._KIND_MAP[kind]
            self.version_manager.register_object(key)

    def begin(
        self,
        task_id: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        snapshot_object_ids: Optional[Iterable[str]] = None,
        strategy: str | None = None,
    ) -> AgentTransaction:
        metadata = dict(metadata or {})
        selected_strategy = str(
            strategy or metadata.get("strategy") or metadata.get("cc") or ""
        ).strip()
        if selected_strategy:
            strategy_impl = self.cc_registry.resolve(selected_strategy)
            metadata.setdefault("strategy", strategy_impl.name)
            if strategy_impl.family == "paper-atcc":
                metadata["paper_atcc"] = True
                metadata["commit_admission_write_protection"] = True
                agentic = dict(metadata.get("agentic", {}) or {})
                if (
                    str(metadata.get("workload", "")).strip().lower() == "ycsb"
                    and (
                        metadata.get("runtime_background", False)
                        or int(agentic.get("background_workers", 0) or 0) > 0
                    )
                ):
                    metadata["paper_atcc_engineering_profile"] = "ycsb-high-mixed"
        task_key = str(task_id)
        runtime_background = bool(metadata.get("runtime_background", False))
        coordinated_backend = bool(metadata.get("paper_atcc_backend", False))
        if runtime_background:
            sample_rate = 16
            sampled_rate = self.next_background_metric_sample_rate(sample_rate)
            sampled = sampled_rate > 0
            metadata["_commit_timing_sample_rate"] = sample_rate
            metadata["_commit_timing_sampled"] = sampled
            if coordinated_backend:
                metadata["_background_metric_sample_rate"] = sample_rate
                metadata["_background_metric_sampled"] = sampled
        elif metadata.get("paper_atcc", False) or coordinated_backend:
            metadata["_background_metric_sample_rate"] = 1
            metadata["_background_metric_sampled"] = True
            metadata["_commit_timing_sample_rate"] = 1
            metadata["_commit_timing_sampled"] = True
        retry_count = int(metadata.get("retry_count", 0) or 0)
        if metadata.get("paper_atcc", False) and not metadata.get("paper_atcc_backend", False):
            explicit_mask = LockClass(
                int(metadata.get("retry_protection_mask", 0) or 0) & 0xF
            )
            with self._lock:
                if retry_count <= 0:
                    self._retry_protection.pop(task_key, None)
                inherited = self._retry_protection.get(task_key)
                if inherited is not None:
                    metadata["retry_validation_conflicts"] = int(
                        inherited.validation_conflicts
                    )
                    retry_conflict_mask = LockClass.NONE
                    for detail in inherited.last_conflicts:
                        retry_conflict_mask |= LockClass(int(detail.protection_bit))
                    metadata["retry_conflict_mask"] = int(retry_conflict_mask)
                    if self.policy_controlled_retry_protection:
                        self._retry_protection_diagnostics["policy_controlled_retry_observations"] += 1
                    else:
                        explicit_mask |= inherited.mask
                        metadata["retry_conflict_read_targets"] = sorted(
                            inherited.protected_read_targets
                        )
                        metadata["retry_conflict_write_targets"] = sorted(
                            inherited.protected_write_targets
                        )
                        self._retry_protection_diagnostics["inherited_attempts"] += 1
            metadata["retry_protection_mask"] = int(explicit_mask)
        if metadata.get("paper_atcc_backend", False):
            # Coordinated background publication uses private versions even
            # when no Agent has selected protection yet.
            self._enable_paper_versioning()
        # Paper Agent transactions start on the native OCC path. Version metadata
        # is materialized lazily by transition_atcc_action() after observed
        # risk selects a non-zero action.  This keeps action=0 identical to the
        # native Silo/OCC publication path in genuinely cold workloads.
        snapshot_ids = (
            tuple(dict.fromkeys(str(value) for value in snapshot_object_ids))
            if snapshot_object_ids is not None
            else ()
        )
        if (
            metadata.get("paper_atcc", False)
            and str(metadata.get("access_set_visibility", "")).strip().lower()
            == "online_observed"
        ):
            snapshot_ids = tuple(
                dict.fromkeys(
                    str(value)
                    for value in (
                        tuple(metadata.get("retry_conflict_read_targets", ()))
                        + tuple(metadata.get("retry_conflict_write_targets", ()))
                    )
                )
            )
        metadata["_planned_snapshot_object_ids"] = snapshot_ids
        with self._lock:
            metadata["_commit_dependency_start_seq"] = int(
                self._commit_dependency_sequence
            )
        initial_protection = LockClass(
            int(metadata.get("retry_protection_mask", 0) or 0) & 0xF
        )
        if (
            metadata.get("paper_atcc", False)
            and (
                initial_protection != LockClass.NONE
                or retry_count > 0
                or metadata.get("retry_conflict_read_targets")
                or metadata.get("retry_conflict_write_targets")
                or metadata.get("_deferred_reasoning_replay", False)
            )
        ):
            # Retries need version metadata for exact protection. Deferred
            # replay also uses it for the short, admitted native-publish
            # suffix while leaving the reasoning interval outside the txn.
            self._enable_paper_versioning()
        materialize_initial_snapshot = bool(
            snapshot_ids
            and self.paper_versioning_enabled
            and not coordinated_backend
            and initial_protection != LockClass.NONE
        )
        snapshot_epoch = -1
        snapshot: Dict[str, SnapshotValue] = {}
        metadata["snapshot_epoch"] = snapshot_epoch
        txn = AgentTransaction(self, str(task_id), snapshot, metadata)
        if materialize_initial_snapshot:
            online_observed_snapshot = (
                str(metadata.get("access_set_visibility", "")).strip().lower()
                == "online_observed"
            )
            snapshot_epoch, snapshot = self.version_manager.snapshot_and_pin(
                txn.context.tid,
                snapshot_ids,
                # The known conflict subset is preloaded, but future accesses
                # remain unknown and still require historical versions.
                materialized=bool(snapshot_ids) and not online_observed_snapshot,
                coverage_object_ids=(
                    self._catalog
                    if online_observed_snapshot
                    else snapshot_ids
                ),
            )
            txn.snapshot.update(snapshot)
            txn.context.snapshot_epoch = int(snapshot_epoch)
            txn.metadata["snapshot_epoch"] = int(snapshot_epoch)
            if txn.events:
                txn.events[0].detail["snapshot_objects"] = len(snapshot)
        read_only_background = bool(
            coordinated_backend
            and "planned_write_targets" in metadata
            and not metadata.get("planned_write_targets")
        )
        txn.metadata["_paper_background_read_only"] = read_only_background
        if read_only_background:
            txn.metadata.setdefault(
                "paper_background_aborts_at_begin", self._paper_background_aborts
            )
        else:
            with self._lock:
                self._live_transactions[txn.context.tid] = txn
                self._paper_attempts += 1
                txn.metadata.setdefault(
                    "paper_background_aborts_at_begin", self._paper_background_aborts
                )
        self.interceptor.begin(txn)
        return txn

    def ensure_snapshot_epoch(self, txn: AgentTransaction) -> int:
        if txn.context.snapshot_epoch >= 0:
            return int(txn.context.snapshot_epoch)
        planned_snapshot_ids = tuple(
            str(value)
            for value in txn.metadata.get("_planned_snapshot_object_ids", ())
        )
        epoch = self.version_manager.pin_lazy(
            txn.context.tid,
            planned_snapshot_ids or self._catalog,
        )
        txn.context.snapshot_epoch = epoch
        return epoch

    def _conflict_details(
        self,
        txn: AgentTransaction,
        conflict_object_ids: Iterable[str],
    ) -> tuple[ConflictDetail, ...]:
        details = []
        for object_id in sorted(set(str(value) for value in conflict_object_ids)):
            write = (
                object_id in txn.write_set
                or object_id in txn.context.write_targets
                or object_id in txn.context.planned_write_targets
            )
            hot = self.peek_hot(object_id)
            bit = (
                LockClass.HOT_WRITE if hot and write
                else LockClass.COLD_WRITE if write
                else LockClass.HOT_READ if hot
                else LockClass.COLD_READ
            )
            details.append(
                ConflictDetail(
                    object_id=object_id,
                    access_kind="write" if write else "read",
                    hot=hot,
                    protection_class=bit.name.lower().replace("_", "-"),
                    protection_bit=int(bit),
                )
            )
        return tuple(details)

    def _observed_protection_mask(self, txn: AgentTransaction) -> LockClass:
        mask = LockClass.NONE
        read_targets = set(txn.read_set) | set(txn.context.read_versions)
        write_targets = set(txn.write_set) | set(txn.context.write_targets)
        planned_observed_writes = read_targets & set(txn.context.planned_write_targets)
        write_targets.update(planned_observed_writes)
        for object_id in read_targets - planned_observed_writes:
            mask |= (
                LockClass.HOT_READ
                if self.peek_hot(object_id)
                else LockClass.COLD_READ
            )
        for object_id in write_targets:
            mask |= (
                LockClass.HOT_WRITE
                if self.peek_hot(object_id)
                else LockClass.COLD_WRITE
            )
        return mask
    def _prepare_retry_feedback(
        self,
        txn: AgentTransaction,
        *,
        reason: str,
        conflict_object_ids: Iterable[str],
    ) -> tuple[tuple[ConflictDetail, ...], int]:
        if (
            normalize_conflict_kind(reason) != "version-conflict"
            or not (
                txn.metadata.get("paper_atcc", False)
                or txn.metadata.get("paper_atcc_retry_feedback", False)
            )
            or txn.context.is_background
        ):
            return (), int(txn.context.action.protected)
        details = self._conflict_details(txn, conflict_object_ids)
        agentic_metadata = dict(txn.metadata.get("agentic", {}) or {})
        online_ycsb_high_mixed = bool(
            str(txn.metadata.get("access_set_visibility", "")).strip().lower()
            == "online_observed"
            and str(txn.metadata.get("workload", "")).strip().lower() == "ycsb"
            and int(agentic_metadata.get("background_workers", 0) or 0) > 0
        )
        tpcc_root_only = bool(
            str(txn.metadata.get("workload", "")).strip().lower() == "tpcc"
            and int(
                dict(txn.metadata.get("agentic", {}) or {}).get(
                    "background_workers", 0
                )
                or 0
            )
            == 0
        )
        conflict_mask = LockClass.NONE
        for detail in details:
            conflict_mask |= LockClass(int(detail.protection_bit))
        with self._lock:
            previous = self._retry_protection.get(txn.task_id, _RetryProtectionState())
            conflict_count = previous.validation_conflicts + 1
            mask = previous.mask | txn.context.action.protected
            if not online_ycsb_high_mixed:
                mask |= conflict_mask
            protected_reads = set(previous.protected_read_targets)
            protected_writes = set(previous.protected_write_targets)
            for detail in details:
                if tpcc_root_only and not detail.object_id.startswith(
                    ("tpcc:warehouse:", "tpcc:district:")
                ):
                    continue
                target_set = (
                    protected_writes
                    if detail.access_kind == "write"
                    else protected_reads
                )
                target_set.add(detail.object_id)
            if (
                conflict_count >= 2
                and not tpcc_root_only
                and not online_ycsb_high_mixed
            ):
                mask |= self._observed_protection_mask(txn)
                observed_reads = set(txn.read_set) | set(txn.context.read_versions)
                planned_observed_writes = (
                    observed_reads & set(txn.context.planned_write_targets)
                )
                protected_reads.update(observed_reads - planned_observed_writes)
                protected_writes.update(planned_observed_writes)
                protected_writes.update(txn.write_set)
                protected_writes.update(txn.context.write_targets)
                self._retry_protection_diagnostics["full_observed_escalations"] += 1
            if mask != previous.mask:
                self._retry_protection_diagnostics["mask_escalations"] += 1
            self._retry_protection[txn.task_id] = _RetryProtectionState(
                mask=mask,
                validation_conflicts=conflict_count,
                last_conflicts=details,
                protected_read_targets=frozenset(protected_reads),
                protected_write_targets=frozenset(protected_writes),
            )
            self._retry_protection_diagnostics["validation_conflicts"] += 1
            attempt_bucket = (
                "first_attempt"
                if int(txn.context.retry_count) <= 0
                else "retry_attempt"
            )
            self._retry_protection_diagnostics[
                f"validation_conflicts_{attempt_bucket}"
            ] += 1
            exact_tpcc_targets = {
                str(value)
                for value in txn.metadata.get("_tpcc_exact_risk_targets", ())
            }
            for detail in details:
                self._retry_conflict_objects[detail.object_id] += 1
                if detail.object_id in exact_tpcc_targets:
                    self._retry_protection_diagnostics[
                        "conflict_after_tpcc_exact_guard"
                    ] += 1
                conflict_class = detail.protection_class.replace("-", "_")
                self._retry_protection_diagnostics[
                    f"conflict_{conflict_class}"
                ] += 1
                if detail.access_kind == "write":
                    dependency = (
                        "read_before_write"
                        if detail.object_id in txn.read_set
                        else "blind_write"
                    )
                    self._retry_protection_diagnostics[
                        f"conflict_{dependency}"
                    ] += 1
                parts = detail.object_id.split(":")
                object_type = parts[1] if len(parts) > 1 else "other"
                if object_type not in {
                    "warehouse",
                    "district",
                    "stock",
                    "customer",
                }:
                    object_type = "other"
                self._retry_protection_diagnostics[
                    f"conflict_object_{object_type}"
                ] += 1
        txn.context.retry_validation_conflicts = conflict_count
        txn._event(
            "retry-protection",
            {
                "validation_conflicts": conflict_count,
                "mask": int(mask),
                "conflicts": [dataclasses.asdict(detail) for detail in details],
                "protected_read_targets": sorted(protected_reads),
                "protected_write_targets": sorted(protected_writes),
            },
        )
        return details, int(mask)

    def retry_protection_diagnostics(self) -> Dict[str, object]:
        with self._lock:
            return {
                **dict(self._retry_protection_diagnostics),
                "tracked_tasks": len(self._retry_protection),
                "conflict_objects": dict(sorted(self._retry_conflict_objects.items())),
            }

    def _clear_retry_protection(self, txn: AgentTransaction) -> None:
        if txn.context.is_background:
            return
        with self._lock:
            self._retry_protection.pop(txn.task_id, None)

    def _wound_context(self, context: TransactionContext, reason: str) -> None:
        txn = self._live_transactions.get(context.tid)
        if txn is None or txn.state != TransactionState.ACTIVE:
            return
        if not context.try_transition(
            TransactionStatus.ABORTED,
            from_statuses=(TransactionStatus.ABORTING,),
        ):
            return
        txn.state = TransactionState.ABORTED
        # Forced wound does not pass through AgentTransaction.abort(). Release
        # helper reservations here as well, before hooks or recording can fail.
        # The coordinator never enters the native lock table while holding its
        # condition, so this is safe from the native wound callback.
        self.release_hotspot_admission(txn)
        context.note_conflict("lock-preempted")
        txn._event("wound", {"reason": str(reason)})
        txn.result = txn._result(
            strategy="paper-atcc",
            committed=False,
            action="abort",
            reason=str(reason),
        )
        if self._note_runtime_outcome(txn, False):
            with self._lock:
                self._paper_aborts += 1
        self._record(txn)
        self.interceptor.abort(txn, str(reason))
        self.interceptor.finish(txn)
        if self.paper_versioning_enabled:
            self.version_manager.finish(context.tid, committed=False)
        self._live_transactions.pop(context.tid, None)
        self._finalize_commit_timing(txn)

    def commit(self, txn: AgentTransaction, *, strategy: str = "occ") -> TransactionResult:
        strategy_impl = self.cc_registry.resolve(strategy)
        plan = strategy_impl.plan(txn)
        txn._event(
            "validate",
            {
                "strategy": plan.strategy,
                "family": plan.family,
                "lock_targets": list(plan.lock_targets),
                "validate_reads": plan.validate_reads,
                "validate_writes": plan.validate_writes,
                "metadata": dict(plan.metadata),
            },
        )
        started_wait = time.perf_counter()
        try:
            lock_table = str(plan.metadata.get("lock_table", ""))
            if lock_table == "2pl":
                mode = "x" if txn.write_set else "s"
                if self._has_prelock(txn, plan, "2pl"):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
                with self.two_phase_locks.acquire(
                    plan.lock_targets,
                    owner=txn,
                    mode=mode,
                    policy=str(plan.metadata.get("policy", "nowait")),
                ):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
            if lock_table == "exclusive" and plan.lock_targets:
                if self._has_prelock(txn, plan, "exclusive"):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
                with self.exclusive_locks.acquire(
                    plan.lock_targets,
                    owner=txn,
                    wait=bool(plan.metadata.get("wait", True)),
                    priority=int(plan.metadata.get("priority", 0) or 0),
                ):
                    return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
            return self._commit_under_lock(txn, strategy_impl, plan, started_wait)
        except LockConflict as exc:
            validation = lock_conflict_result(exc)
            result = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason=validation.reason,
                conflict_object_ids=validation.conflict_object_ids,
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, result, txn)
            return result
        except BaseException:
            self._release_version_read_guard(txn)
            raise

    def _has_prelock(self, txn: AgentTransaction, plan: Any, lock_table: str) -> bool:
        metadata = dict(getattr(txn, "metadata", {}) or {})
        if str(metadata.get("prelocked_lock_table", "")) != str(lock_table):
            return False
        targets = {str(target) for target in metadata.get("prelocked_targets", ())}
        return set(str(target) for target in plan.lock_targets).issubset(targets)

    def _commit_under_lock(
        self,
        txn: AgentTransaction,
        strategy_impl: Any,
        plan: Any,
        started_wait: float,
    ) -> TransactionResult:
        coordinated_backend = bool(txn.metadata.get("paper_atcc_backend", False))
        exact_object_protection = bool(
            txn.metadata.get("_version_risk_exact_mode", False)
        )
        paper_atcc = bool(
            txn.context.action.protected
            or txn.context.held_read_locks
            or txn.context.held_write_locks
            or txn.metadata.get("paper_atcc", False)
            or coordinated_backend
            or getattr(plan, "family", "") == "paper-atcc"
        )

        if (
            paper_atcc
            and not coordinated_backend
            and not self.paper_versioning_enabled
            and txn.context.action.protected == LockClass.NONE
            and not txn.context.held_read_locks
            and not txn.context.held_write_locks
            and not txn.metadata.get("_deferred_reasoning_replay", False)
            and not self.atcc_locks.has_object_protection(
                txn.write_set, txn.context
            )
        ):
            # Genuine action-0: use exactly the native OCC validate/install
            # path. No ATCC lock-table probe, VersionManager publication, or
            # paper-specific validation is needed until risk is observed.
            txn.context.transition(TransactionStatus.COMMITTING)
            return self._commit_after_admission(
                txn,
                strategy_impl,
                plan,
                started_wait,
                paper_atcc=False,
                release_atcc_locks=False,
                allow_native_publish=False,
                native_publication_held=False,
            )

        if coordinated_backend and not txn.write_set:
            # A read-only backend transaction needs only native OCC read
            # validation. It owns no private writes and therefore must not
            # enter either lock-publication or VersionManager metadata paths.
            txn.context.transition(TransactionStatus.COMMITTING)
            return self._commit_after_admission(
                txn,
                strategy_impl,
                plan,
                started_wait,
                paper_atcc=True,
                release_atcc_locks=False,
                allow_native_publish=False,
                native_publication_held=False,
            )

        if coordinated_backend:
            prepare_started = time.perf_counter()
            self.version_manager.prepare(
                txn.context.tid,
                (
                    (write.object_id, write.value)
                    for write in txn.write_set.values()
                ),
            )
            txn.metadata["_private_prepared"] = True
            self.add_commit_timing(
                txn,
                "install",
                (time.perf_counter() - prepare_started) * 1000.0,
            )
            used_fast_path, result = self.atcc_locks.try_uncontended_background_publish(
                txn.write_set,
                txn.context,
                lambda: self._commit_after_admission(
                    txn,
                    strategy_impl,
                    plan,
                    started_wait,
                    paper_atcc=True,
                    release_atcc_locks=False,
                    allow_native_publish=False,
                    native_publication_held=False,
                ),
                allow_reader_bypass=True,
                timing_callback=lambda elapsed_ms: self.add_commit_timing(
                    txn, "lock", elapsed_ms
                ),
            )
            if used_fast_path:
                return result
            conflict_targets = tuple(
                getattr(result, "object_ids", ()) or tuple(txn.write_set)
            )
            aborted = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason="background-admission-busy",
                conflict_object_ids=conflict_targets,
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, aborted, txn)
            return aborted

        if (
            paper_atcc
            and not coordinated_backend
            and txn.write_set
            and (
                txn.context.action.protected == LockClass.NONE
                or exact_object_protection
            )
        ):
            last_conflicts: tuple[str, ...] = ()
            for _admission_round in range(len(txn.write_set) + 2):
                used_fast_path, admission_result = (
                    self.atcc_locks.try_uncontended_occ_publish(
                        txn.write_set,
                        txn.context,
                        lambda: self._commit_after_admission(
                            txn,
                            strategy_impl,
                            plan,
                            started_wait,
                            paper_atcc=True,
                            release_atcc_locks=bool(
                                txn.context.held_read_locks
                                or txn.context.held_write_locks
                            ),
                            allow_native_publish=True,
                            native_publication_held=False,
                        ),
                        timing_callback=lambda elapsed_ms: self.add_commit_timing(
                            txn, "lock", elapsed_ms
                        ),
                    )
                )
                if used_fast_path:
                    return admission_result
                last_conflicts = tuple(
                    str(value)
                    for value in getattr(admission_result, "object_ids", ())
                )
                new_conflicts = tuple(
                    sorted(set(last_conflicts) - txn.context.held_write_locks)
                )
                if not new_conflicts:
                    break
                lock_started = time.perf_counter()
                try:
                    self.atcc_locks.acquire_write_set(
                        new_conflicts,
                        txn.context,
                        timeout_s=5.0,
                    )
                except LockConflict as exc:
                    self.add_commit_timing(
                        txn,
                        "lock",
                        (time.perf_counter() - lock_started) * 1000.0,
                    )
                    txn.context.note_conflict(exc.kind)
                    aborted = self._finish_abort(
                        txn,
                        strategy=plan.strategy,
                        reason=exc.kind,
                        conflict_object_ids=exc.targets,
                        lock_wait_s=time.perf_counter() - started_wait,
                    )
                    self._observe_strategy(
                        strategy_impl, plan, aborted, txn
                    )
                    return aborted
                self.add_commit_timing(
                    txn,
                    "lock",
                    (time.perf_counter() - lock_started) * 1000.0,
                )
            aborted = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason="commit-admission-conflict",
                conflict_object_ids=last_conflicts,
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, aborted, txn)
            return aborted

        if paper_atcc:
            lock_started = time.perf_counter()
            previously_held_writes = set(txn.context.held_write_locks)
            try:
                # A low-priority backend must not occupy a worker for the
                # lifetime of a costly Agent transaction. If its private
                # publish collides with an Agent writer, briefly yield and
                # retry another trace transaction instead of joining a
                # multi-second lock queue.
                self.atcc_locks.acquire_write_set(
                    txn.write_set,
                    txn.context,
                    timeout_s=0.005 if coordinated_backend else 5.0,
                )
            except LockConflict as exc:
                self.add_commit_timing(
                    txn, "lock", (time.perf_counter() - lock_started) * 1000.0
                )
                txn.context.note_conflict(exc.kind)
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason=exc.kind,
                    conflict_object_ids=exc.targets,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
            self.add_commit_timing(
                txn, "lock", (time.perf_counter() - lock_started) * 1000.0
            )
            # Deferred write protection is admitted as one exact write set.
            # Once admitted, background publishers cannot advance these keys,
            # so read-before-write validation is atomic with the protection
            # upgrade.  Blind writes are rebased later and need no old-version
            # validation.
            newly_protected = set(txn.context.held_write_locks) - previously_held_writes
            observed_writes = newly_protected & set(txn.read_set)
            stale_observed_writes = self._conflict_targets(
                (
                    object_id,
                    int(txn.read_set[object_id].version),
                )
                for object_id in observed_writes
            )
            if stale_observed_writes:
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="atomic version check failed",
                    conflict_object_ids=stale_observed_writes,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
        if paper_atcc:
            committing_started = time.perf_counter()
            if not self.atcc_locks.begin_committing(txn.context):
                self.add_commit_timing(
                    txn,
                    "lock",
                    (time.perf_counter() - committing_started) * 1000.0,
                )
                if txn.result is not None:
                    return txn.result
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="transaction was wounded before commit became non-preemptible",
                    conflict_object_ids=(),
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
            self.add_commit_timing(
                txn,
                "lock",
                (time.perf_counter() - committing_started) * 1000.0,
            )
        else:
            txn.context.transition(TransactionStatus.COMMITTING)

        return self._commit_after_admission(
            txn,
            strategy_impl,
            plan,
            started_wait,
            paper_atcc=paper_atcc,
            release_atcc_locks=paper_atcc,
            allow_native_publish=False,
            native_publication_held=False,
        )

    def _commit_after_admission(
        self,
        txn: AgentTransaction,
        strategy_impl: Any,
        plan: Any,
        started_wait: float,
        *,
        paper_atcc: bool,
        release_atcc_locks: bool,
        allow_native_publish: bool,
        native_publication_held: bool,
    ) -> TransactionResult:
        validation_started = time.perf_counter()
        pinned_read_targets = set(
            txn.metadata.get("_version_risk_pinned_read_targets", ())
        )
        online_bypass_read_targets = set(
            txn.metadata.get("_online_bypass_read_targets", ())
        )
        online_serialization_before_targets = (
            self._safe_online_serialization_before_targets(
                txn, online_bypass_read_targets
            )
        )
        certified_read_only = False
        if (paper_atcc and not txn.write_set and not txn.context.is_background
                and getattr(self,'read_only_snapshot_certification_enabled',False)
                and '_read_certificate_begin_epoch' in txn.metadata):
            epoch=self.version_manager.certify_common_read_epoch(
                {key:int(read.version) for key,read in txn.read_set.items()},
                not_before_epoch=txn.metadata['_read_certificate_begin_epoch'])
            certified_read_only=epoch is not None
            self.read_only_snapshot_certificate_metrics['accepted' if certified_read_only else 'rejected']+=1
            if certified_read_only:
                txn.metadata['_certified_read_only_epoch']=epoch
                self.read_only_snapshot_certificates.append(dict(tid=txn.context.tid,epoch=epoch,
                    begin_epoch=txn.metadata['_read_certificate_begin_epoch'],
                    read_versions={key:int(read.version) for key,read in txn.read_set.items()}))
                online_serialization_before_targets.update(txn.read_set)
        if pinned_read_targets and txn.context.snapshot_epoch >= 0:
            pinned_read_targets = set(txn.read_set) - set(txn.write_set)
        if paper_atcc and not txn.context.is_background and pinned_read_targets:
            lock_started = time.perf_counter()
            guard, guard_conflicts = self.version_manager.enter_pinned_read_guard(
                txn.context.tid,
                txn.context.snapshot_epoch,
                {
                    object_id: int(txn.read_set[object_id].version)
                    for object_id in pinned_read_targets
                    if object_id in txn.read_set
                },
                txn.write_set,
            )
            self.add_commit_timing(
                txn,
                "lock",
                (time.perf_counter() - lock_started) * 1000.0,
            )
            if guard_conflicts:
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="atomic version check failed",
                    conflict_object_ids=guard_conflicts,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
            txn.metadata["_version_read_guard"] = int(guard)
            validation_started = time.perf_counter()
        if not paper_atcc:
            validation = strategy_impl.validate(txn, self.store)
            if not validation.ok:
                self.add_commit_timing(
                    txn,
                    "validate",
                    (time.perf_counter() - validation_started) * 1000.0,
                )
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason=validation.reason,
                    conflict_object_ids=validation.conflict_object_ids,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
        foreign_committing_writers: set[str] = set()
        batched_foreign_writer_snapshot = bool(
            paper_atcc
            and not txn.context.is_background
            and len(txn.read_set) <= 16
        )
        if batched_foreign_writer_snapshot:
            foreign_committing_writers.update(
                self.atcc_locks.foreign_committing_writer_targets(
                    txn.read_set,
                    txn.context,
                )
            )
        read_checks = []
        for read in txn.read_set.values():
            protected = (
                read.object_id in txn.context.held_read_locks
                or read.object_id in txn.context.held_write_locks
                or (
                    bool(txn.metadata.get("_version_read_guard", 0))
                    and read.object_id in pinned_read_targets
                )
            )
            if (
                read.object_id in online_bypass_read_targets
                and read.object_id not in txn.context.held_write_locks
            ):
                # ACTIVE online-observed readers may be bypassed by a short
                # background publisher. Once this transaction is COMMITTING,
                # new bypass is closed by the lock manager; validating the
                # observed version here conservatively detects the resulting
                # rw anti-dependency without any future access oracle.
                protected = False
            if (
                paper_atcc
                and not txn.context.is_background
                and not certified_read_only
                and not protected
                and (
                    read.object_id in foreign_committing_writers
                    if batched_foreign_writer_snapshot
                    else self.atcc_locks.has_foreign_committing_writer(
                        read.object_id,
                        txn.context,
                    )
                )
            ):
                self.hotness_tracker.observe_contention(
                    read.object_id, "validation-failure"
                )
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="optimistic read conflicts with committing writer",
                    conflict_object_ids=(read.object_id,),
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
            if (
                plan.validate_reads
                and not (paper_atcc and protected)
                and read.object_id not in online_serialization_before_targets
            ):
                read_checks.append((read.object_id, int(read.version)))
        checks = list(read_checks)
        checks.extend(
            (write.object_id, int(write.base_version))
            for write in txn.write_set.values()
            if not txn.context.is_background or write.object_id in txn.read_set
        )
        writes = [(write.object_id, write.value) for write in txn.write_set.values()]
        self.add_commit_timing(
            txn,
            "validate",
            (time.perf_counter() - validation_started) * 1000.0,
        )
        if paper_atcc and txn.context.is_background and checks:
            stale_targets = self._conflict_targets(checks)
            if stale_targets:
                self.atcc_locks.note_background_publish_fallback(
                    "version_mismatch",
                    count_total=False,
                )
                result = self._finish_abort(
                    txn,
                    strategy=plan.strategy,
                    reason="atomic version check failed",
                    conflict_object_ids=stale_targets,
                    lock_wait_s=time.perf_counter() - started_wait,
                )
                self._observe_strategy(strategy_impl, plan, result, txn)
                return result
        if (
            paper_atcc
            and not txn.context.is_background
        ):
            # A blind write has no semantic dependency on the value/version
            # captured when its private buffer was created. Commit admission
            # already excludes same-key publishers and protected writers, so
            # rebasing it here gives the batch a fresh serialization point
            # without turning every deferred write into a long-lived WLock.
            blind_writes = set(txn.write_set) - set(txn.read_set)
            if blind_writes:
                rebased_versions: dict[str, int] = {}
                for object_id in blind_writes:
                    write = txn.write_set[object_id]
                    current = self.store.get(object_id)
                    rebased_versions[object_id] = int(current.version)
                    txn.write_set[object_id] = dataclasses.replace(
                        write,
                        base_value=str(current.value),
                        base_version=int(current.version),
                    )
                checks = [
                    (object_id, rebased_versions.get(object_id, version))
                    for object_id, version in checks
                ]
                self.atcc_locks.note_agent_blind_write_rebases(
                    len(blind_writes)
                )
        durable_undo = self.undo_log.path is not None
        if durable_undo:
            self.undo_log.begin(txn.context.tid)
        if durable_undo and not txn.context.is_background:
            self.undo_log.update_batch(
                txn.context.tid,
                (
                    (write.object_id, write.base_value, write.base_version)
                    for write in txn.write_set.values()
                ),
            )
        if not txn.context.is_background:
            self._inject_commit_fault("after_undo_flush_before_install", txn)

        def install_private_versions() -> bool:
            nonlocal checks
            background_rebase = bool(
                txn.context.is_background
                and (
                    durable_undo
                    or self.version_manager.has_lazy_pins(txn.write_set)
                )
            )
            if background_rebase:
                for object_id, write in tuple(txn.write_set.items()):
                    if object_id in txn.read_set:
                        continue
                    current = self.store.get(object_id)
                    txn.write_set[object_id] = dataclasses.replace(
                        write,
                        base_value=str(current.value),
                        base_version=int(current.version),
                    )
                if durable_undo:
                    self.undo_log.update_batch(
                        txn.context.tid,
                        (
                            (write.object_id, write.base_value, write.base_version)
                            for write in txn.write_set.values()
                        ),
                    )
            if txn.context.is_background:
                self._inject_commit_fault("after_undo_flush_before_install", txn)
            install_started = time.perf_counter()
            installed = bool(self.store.batch_put_if_version(checks, writes))
            self.add_commit_timing(
                txn,
                "install",
                (time.perf_counter() - install_started) * 1000.0,
            )
            if not installed:
                return False
            self._inject_commit_fault("after_install_before_publish", txn)
            if durable_undo:
                self.undo_log.commit(txn.context.tid)
            return True

        timing_bucket = self._commit_timing_bucket(txn)
        background_sample_rate = 1
        if txn.context.is_background:
            background_sample_rate = (
                int(txn.metadata.get("_background_metric_sample_rate", 1) or 1)
                if txn.metadata.get("_background_metric_sampled", False)
                else 0
            )
        if not writes:
            ok = install_private_versions()
            if self.paper_versioning_enabled:
                self.version_manager.note_read_only_bypass()
        elif native_publication_held:
            ok = install_private_versions()
        elif self.paper_versioning_enabled:
            used_native = False
            ok = False
            version_read_guard_held = bool(
                txn.metadata.get("_version_read_guard", 0)
            )
            if (
                allow_native_publish
                and not txn.context.action.protected
                and not version_read_guard_held
            ):
                used_native, ok = self.version_manager.try_native_publish(
                    txn.write_set,
                    install_private_versions,
                    background=txn.context.is_background,
                    background_sample_rate=background_sample_rate,
                    timing_ms=timing_bucket,
                )
            if not used_native:
                ok = self.version_manager.atomic_publish(
                    txn.context.tid,
                    writes,
                    install_private_versions,
                    background=txn.context.is_background,
                    background_sample_rate=background_sample_rate,
                    publication_boundary_held=version_read_guard_held,
                    published_version=lambda object_id: int(
                        txn.write_set[str(object_id)].base_version
                    )
                    + 1,
                    timing_ms=timing_bucket,
                    private_prepared=bool(
                        txn.metadata.get("_private_prepared", False)
                    ),
                )
        else:
            ok = install_private_versions()
        self._release_version_read_guard(txn)
        if not ok:
            if durable_undo:
                self.undo_log.abort(txn.context.tid)
            if txn.context.is_background:
                self.atcc_locks.note_background_publish_fallback(
                    "version_mismatch",
                    count_total=False,
                )
            result = self._finish_abort(
                txn,
                strategy=plan.strategy,
                reason="atomic version check failed",
                conflict_object_ids=self._conflict_targets(checks),
                lock_wait_s=time.perf_counter() - started_wait,
            )
            self._observe_strategy(strategy_impl, plan, result, txn)
            return result
        self._inject_commit_fault("after_publish", txn)
        self._record_commit_dependency(txn.read_set, txn.write_set)
        txn.state = TransactionState.COMMITTED
        txn.context.transition(TransactionStatus.COMMITTED)
        txn._event("finish", {"state": txn.state.value, "action": "commit"})
        txn.result = txn._result(
            strategy=plan.strategy,
            committed=True,
            action="commit",
            lock_wait_s=time.perf_counter() - started_wait,
        )
        if not self.paper_versioning_enabled and txn.write_set:
            with self._lock:
                self._native_dirty_objects.update(txn.write_set)
        self._note_runtime_outcome(txn, True)
        self._clear_retry_protection(txn)
        self._record(txn)
        self.interceptor.finish(txn)
        if (
            release_atcc_locks
            or txn.metadata.get("paper_atcc", False)
            or txn.metadata.get("paper_atcc_backend", False)
            or txn.context.held_read_locks
            or txn.context.held_write_locks
        ):
            self.atcc_locks.release_all(txn.context)
        self.release_hotspot_admission(txn)
        if self.paper_versioning_enabled:
            self.version_manager.finish(txn.context.tid, committed=True)
        self._live_transactions.pop(txn.context.tid, None)
        self._finalize_commit_timing(txn)
        self._observe_strategy(strategy_impl, plan, txn.result, txn)
        return txn.result

    def recover(self) -> list[str]:
        return self.undo_log.recover(self.store)

    def _observe_strategy(
        self,
        strategy_impl: Any,
        plan: Any,
        result: TransactionResult,
        txn: AgentTransaction,
    ) -> None:
        observer = getattr(strategy_impl, "observe", None)
        if observer is None:
            return
        try:
            observer(plan, result, txn)
        except TypeError:
            observer(plan, result)

    def _finish_abort(
        self,
        txn: AgentTransaction,
        *,
        strategy: str,
        reason: str,
        conflict_object_ids: Iterable[str],
        lock_wait_s: float,
    ) -> TransactionResult:
        self._release_version_read_guard(txn)
        conflict_ids = tuple(str(value) for value in conflict_object_ids)
        conflict_details, retry_mask = self._prepare_retry_feedback(
            txn,
            reason=reason,
            conflict_object_ids=conflict_ids,
        )
        if txn.context.status not in {TransactionStatus.ABORTING, TransactionStatus.ABORTED}:
            txn.context.transition(TransactionStatus.ABORTING)
            self.interceptor.abort(txn, str(reason))
        txn.context.note_conflict(normalize_conflict_kind(reason))
        txn.state = TransactionState.ABORTED
        if txn.context.status != TransactionStatus.ABORTED:
            txn.context.transition(TransactionStatus.ABORTED)
        txn._event(
            "finish",
            {
                "state": txn.state.value,
                "action": "abort",
                "reason": str(reason),
                "conflict_object_ids": list(conflict_ids),
                "conflict_details": [
                    dataclasses.asdict(detail) for detail in conflict_details
                ],
                "retry_protection_mask": retry_mask,
            },
        )
        txn.result = txn._result(
            strategy=strategy,
            committed=False,
            action="abort",
            reason=str(reason),
            conflict_object_ids=conflict_ids,
            conflict_details=conflict_details,
            retry_protection_mask=retry_mask,
            lock_wait_s=lock_wait_s,
        )
        if self._note_runtime_outcome(txn, False):
            with self._lock:
                self._paper_aborts += 1
        self._record(txn)
        self.interceptor.finish(txn)
        self.atcc_locks.release_all(txn.context)
        self.release_hotspot_admission(txn)
        if self.paper_versioning_enabled:
            self.version_manager.finish(txn.context.tid, committed=False)
        self._live_transactions.pop(txn.context.tid, None)
        self._finalize_commit_timing(txn)
        return txn.result

    def _release_version_read_guard(self, txn: AgentTransaction) -> None:
        boundary = int(txn.metadata.pop("_version_read_guard", 0) or 0)
        if boundary:
            self.version_manager.exit_pinned_read_guard(boundary)

    def _snapshot_locked(
        self,
        object_ids: Optional[Iterable[str]] = None,
        *,
        snapshot_epoch: Optional[int] = None,
    ) -> Dict[str, SnapshotValue]:
        if object_ids is None:
            keys = ()
        else:
            keys = tuple(dict.fromkeys(str(object_id) for object_id in object_ids))
            missing = [key for key in keys if key not in self._catalog]
            if missing:
                raise KeyError(f"object is not registered: {missing[0]}")
        epoch = (
            self.version_manager.current_epoch()
            if snapshot_epoch is None
            else int(snapshot_epoch)
        )
        if not self.paper_versioning_enabled:
            return {
                object_id: SnapshotValue(
                    value=(value := self.store.get(object_id)).value,
                    version=int(value.version),
                    exists=bool(value.exists),
                )
                for object_id in keys
            }
        return self.version_manager.read_many_at(epoch, keys)

    def _conflict_targets(self, checks: Iterable[tuple[str, int]]) -> tuple[str, ...]:
        conflicts = []
        for object_id, expected_version in checks:
            if int(self.store.get_version(str(object_id))) != int(expected_version):
                key = str(object_id)
                conflicts.append(key)
                self.hotness_tracker.observe_contention(key, "validation-failure")
        return tuple(sorted(set(conflicts)))

    def value_of(self, object_id: str) -> str:
        with self._lock:
            return self.store.get(str(object_id)).value

    def values(self) -> Dict[str, str]:
        with self._lock:
            return {object_id: self.store.get(object_id).value for object_id in self._catalog}

    def _record(self, txn: AgentTransaction) -> None:
        if not self._record_traces:
            return
        with self._lock:
            self._traces.append(txn.to_trace())

    def traces(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._traces)
