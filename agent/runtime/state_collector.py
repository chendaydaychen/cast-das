"""Incremental paper state features independent of benchmark labels."""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable

from .context import TransactionContext,TransactionStatus


@dataclasses.dataclass(frozen=True)
class PhaseAwareState:
    phase: str
    inter_round_interval_ms: float
    read_set_size: int
    write_set_size: int
    read_set_growth: int
    write_set_growth: int
    access_overlap_ratio: float
    completed_rounds: int
    completed_operations: int
    recent_write_ratio: float
    hotspot_access_ratio: float
    blocked_time_ms: float
    retry_count: int
    current_action: int
    priority: int
    recent_conflict_kind: str = "none"
    global_active_transactions: int = 0
    global_waiter_count: int = 0
    global_abort_rate: float = 0.0
    global_throughput: float = 0.0
    global_avg_latency_ms: float = 0.0
    global_tail_latency_ms: float = 0.0
    global_agent_task_throughput: float = 0.0
    global_agent_task_avg_latency_ms: float = 0.0
    global_agent_task_tail_latency_ms: float = 0.0
    global_conflict_abort_rate: float = 0.0
    global_background_throughput: float = 0.0
    global_background_abort_rate: float = 0.0
    observed_restart_cost_ms: float = 0.0
    # Reviewer-rule-only online summary.  These values describe only objects
    # already accessed by the current attempt; they are not model features.
    max_observed_hot_accesses: int = 0
    max_observed_hot_contention_rate: float = 0.0
    max_observed_recent_accesses: int = 0
    # Opt-in held-footprint candidate. Legacy collectors/encoders leave these
    # at zero; actual ownership is sampled under the lock-manager condition.
    held_read_lock_count: int = 0
    held_write_lock_count: int = 0
    held_cold_read_lock_count: int = 0
    held_cold_write_lock_count: int = 0


PHASE_FEATURE_FIELDS = (
    'inter_round_interval_ms', 'read_set_growth', 'write_set_growth',
    'access_overlap_ratio', 'completed_rounds', 'completed_operations',
    'recent_write_ratio',
)


def policy_observation(state: PhaseAwareState, phase_awareness: str = "enabled", *,
                       remove_phase_index_proxy: bool = False,
                       remove_phase_features: bool = False) -> PhaseAwareState:
    """Mask policy inputs only; legacy label-only ablations remain unchanged.

    The opt-in group ablation removes the phase-related inputs in paper §4.1
    as well as the explicit label. Set sizes and measured restart cost remain
    available: these additional footprint/cost proxies are disclosed separately.
    Runtime priority, reward telemetry, locks and access history are untouched.
    """
    if phase_awareness not in {"enabled", "disabled"}:
        raise ValueError("phase_awareness must be enabled or disabled")
    if phase_awareness == "enabled" and not remove_phase_index_proxy and not remove_phase_features:
        return state
    changes = {}
    if (phase_awareness == "disabled" or remove_phase_features) and state.phase != "all":
        changes["phase"] = "all"
    if remove_phase_index_proxy and state.completed_rounds != 0:
        changes["completed_rounds"] = 0
    if remove_phase_features:
        changes.update({name: 0 for name in PHASE_FEATURE_FIELDS if getattr(state, name) != 0})
    return dataclasses.replace(state, **changes) if changes else state


class StateCollector:
    def __init__(self, *, round_interval_features: bool = False,
                 observed_restart_cost_features: bool = False,
                 observed_agent_rounds: bool = False):
        self.observed_agent_rounds = bool(observed_agent_rounds)
        self._agent_round_counts: dict[str, int] = {}
        self.round_interval_features = bool(round_interval_features)
        self.observed_restart_cost_features = bool(observed_restart_cost_features)
        self._round_agent_interval_ms: dict[str, float] = {}
        self._completed_agent_interval_ms: dict[str, float] = {}
        self._last_sizes: dict[str, tuple[int, int]] = {}
        self._completed_round: dict[str, set[str]] = {}
        self._current_round: dict[str, set[str]] = {}
        self._round_writes: dict[str, int] = {}
        self._round_operations: dict[str, int] = {}
        self._last_interval_ms: dict[str, float] = {}
        self._last_overlap_ratio: dict[str, float] = {}
        self._last_write_ratio: dict[str, float] = {}

    def _discard_if_ended(self,context):
        if getattr(context,'status',TransactionStatus.ACTIVE) in (
                TransactionStatus.ABORTING,TransactionStatus.ABORTED,TransactionStatus.COMMITTED):
            self.discard(context)
            return True
        return False

    def record_operation(
        self,
        context: TransactionContext,
        object_id: str,
        *,
        write: bool,
        interval_ms: float,
        hot: bool = False,
    ) -> None:
        if self._discard_if_ended(context):return
        tid = context.tid
        self._current_round.setdefault(tid, set()).add(str(object_id))
        self._round_operations[tid] = self._round_operations.get(tid, 0) + 1
        if write:
            self._round_writes[tid] = self._round_writes.get(tid, 0) + 1
            if hot:
                context.hot_write_targets.add(str(object_id))
        elif hot:
            context.hot_read_targets.add(str(object_id))
        self._last_interval_ms[tid] = max(0.0, float(interval_ms))
        self._discard_if_ended(context)

    def record_agent_interval(self, context: TransactionContext, interval_ms: float) -> None:
        if self._discard_if_ended(context):return
        self._last_interval_ms[context.tid] = max(0.0, float(interval_ms))
        if self.round_interval_features:
            tid = context.tid
            self._round_agent_interval_ms[tid] = self._round_agent_interval_ms.get(tid, 0.) + max(0., float(interval_ms))
        self._discard_if_ended(context)

    def finish_round(self, context: TransactionContext) -> None:
        if self._discard_if_ended(context):return
        tid = context.tid
        if self.observed_agent_rounds:
            self._agent_round_counts[tid] = self._agent_round_counts.get(tid, 0) + 1
        if self.round_interval_features:
            self._completed_agent_interval_ms[tid] = self._round_agent_interval_ms.pop(tid, 0.)
        previous = self._completed_round.get(tid, set())
        current = set(self._current_round.get(tid, set()))
        union = previous | current
        self._last_overlap_ratio[tid] = (
            len(previous & current) / len(union) if union else 0.0
        )
        operations = self._round_operations.get(tid, 0)
        writes = self._round_writes.get(tid, 0)
        self._last_write_ratio[tid] = writes / operations if operations else 0.0
        self._completed_round[tid] = current
        self._current_round[tid] = set()
        self._round_writes[tid] = 0
        self._round_operations[tid] = 0
        self._discard_if_ended(context)

    def snapshot(self, context: TransactionContext, *,
                 protection_relevant_targets: Iterable[str] | None = None,
                 include_observed_writes: bool = False,
                 footprint_witness: Callable | None = None) -> PhaseAwareState:
        tid = context.tid
        capture_footprint = include_observed_writes or footprint_witness is not None
        observed_reads = frozenset(context.read_versions) if capture_footprint else None
        observed_writes = frozenset(context.write_targets) if capture_footprint else None
        read_size = len(observed_reads) if capture_footprint else len(context.read_versions)
        write_size = len(observed_writes) if capture_footprint else len(context.write_targets)
        previous_sizes = self._last_sizes.get(tid, (0, 0))
        self._last_sizes[tid] = (read_size, write_size)
        operations = self._round_operations.get(tid, 0)
        writes = self._round_writes.get(tid, 0)
        previous_round = self._completed_round.get(tid, set())
        current_round = self._current_round.get(tid, set())
        overlap_union = previous_round | current_round
        overlap = (
            len(previous_round & current_round) / len(overlap_union)
            if current_round and overlap_union
            else self._last_overlap_ratio.get(tid, 0.0)
        )
        all_targets = (set(observed_reads) | set(observed_writes) if capture_footprint
                       else set(context.read_versions) | set(context.write_targets))
        hot_targets = context.hot_read_targets | context.hot_write_targets
        if capture_footprint:
            hot_targets = hot_targets & all_targets
        # Optional encoder only: do not change the actual access footprint,
        # hotness classification, lock installation, or version validation.
        # Intersect even caller-provided targets with already observed access.
        hotness_targets = (all_targets if protection_relevant_targets is None
                           else all_targets & set(protection_relevant_targets))
        if include_observed_writes:
            # The protection-relevant runtime encoder includes every write
            # actually observed at this snapshot, even if the caller built
            # its eligible-read target set before another write was recorded.
            # Never add planned/future targets. Default subset API is unchanged.
            hotness_targets.update(observed_writes)
        interval_ms = self._last_interval_ms.get(tid, 0.)
        if self.round_interval_features:
            # At phase entry use the completed round, not its bookkeeping
            # tail. Mid-round checkpoints use only elapsed observed gaps.
            interval_ms = (self._round_agent_interval_ms.get(tid, 0.) if operations
                           else self._completed_agent_interval_ms.get(tid, self._round_agent_interval_ms.get(tid, 0.)))
        state = PhaseAwareState(
            phase=context.phase.value,
            inter_round_interval_ms=interval_ms,
            read_set_size=read_size,
            write_set_size=write_size,
            read_set_growth=max(0, read_size - previous_sizes[0]),
            write_set_growth=max(0, write_size - previous_sizes[1]),
            access_overlap_ratio=overlap,
            completed_rounds=(self._agent_round_counts.get(tid, 0)
                              if self.observed_agent_rounds else context.round_index),
            completed_operations=context.completed_operations,
            recent_write_ratio=(
                writes / operations
                if operations
                else self._last_write_ratio.get(tid, 0.0)
            ),
            hotspot_access_ratio=(len(hotness_targets & hot_targets) / len(hotness_targets)
                                  if hotness_targets else 0.0),
            blocked_time_ms=context.blocked_time_ms,
            retry_count=context.retry_count,
            current_action=int(context.action.protected),
            priority=context.priority,
            recent_conflict_kind=context.recent_conflict_kind,
            observed_restart_cost_ms=(max(0.0, context.operation_cost_ms)
                + max(0.0, context.agent_cost_ms) + max(0.0, context.prior_retry_cost_ms)
                if self.observed_restart_cost_features else 0.0),
        )
        if footprint_witness is not None:
            # Immutable copies of the exact footprint used for this state,
            # not a later re-read of a concurrently finishing transaction.
            footprint_witness(state, dict(reads=observed_reads, writes=observed_writes,
                hot=frozenset(hot_targets), protection_relevant_targets=frozenset(hotness_targets)))
        return state

    def discard(self, context: TransactionContext) -> None:
        for mapping in (
            self._last_sizes,
            self._completed_round,
            self._current_round,
            self._round_writes,
            self._round_operations,
            self._last_interval_ms,
            self._last_overlap_ratio,
            self._last_write_ratio,
            self._round_agent_interval_ms,
            self._completed_agent_interval_ms,
            self._agent_round_counts,
        ):
            mapping.pop(context.tid, None)


def phase_aware_state_from_dict(data: dict[str, object]) -> PhaseAwareState:
    fields = {field.name for field in dataclasses.fields(PhaseAwareState)}
    return PhaseAwareState(
        **{key: value for key, value in dict(data).items() if key in fields}
    )
