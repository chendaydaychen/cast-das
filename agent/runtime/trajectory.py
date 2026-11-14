"""Transaction-internal trajectories for offline ATCC policy training."""

from __future__ import annotations

import dataclasses
import math
import threading
from typing import Dict, List, Sequence

from .state_collector import PhaseAwareState, phase_aware_state_from_dict


@dataclasses.dataclass(frozen=True)
class PolicyTransition:
    txn_id: str
    state: PhaseAwareState
    action: int
    reward: float
    next_state: PhaseAwareState
    done: bool
    behavior_probability: float = 1.0
    source_id: str = ""
    commit_indicator: int = 0
    abort_indicator: int = 0
    operation_cost_ms: float = 0.0
    agent_cost_ms: float = 0.0
    retry_cost_ms: float = 0.0
    lock_wait_ms: float = 0.0
    new_lock_count: int = 0
    system_delta: float = 0.0
    behavior_action_probabilities: tuple[float, ...] = ()
    background_blocked_ms_caused: float = 0.0
    background_aborts_caused: int = 0
    agent_blocked_ms_caused: float = 0.0
    agent_aborts_caused: int = 0
    monotone_actions: bool = True
    decision_origin: str = ''
    late_agent_cost_ms: float = 0.0


@dataclasses.dataclass(frozen=True)
class PaperRewardConfig:
    commit_reward: float = 100.0
    abort_weight: float = 80.0
    retry_weight: float = 1.0
    lock_weight: float = 5.0
    system_weight: float = 10.0
    # Opt-in training regularizer for the phase-scheduling experiment.  A
    # deferred action is a plan that is materialized at commit; unlike an
    # immediate mask, it can create a commit-side queue.  Keep the default at
    # zero so the paper's original reward is unchanged.
    deferred_action_weight: float = 0.0
    # Optional sensitivity experiment: sparse wait events are scaled using
    # their positive-event distribution, not a zero-dominated population.
    positive_wait_event_scaling: bool = False
    # Attempt costs are emitted at terminal decisions, unlike per-checkpoint
    # wait/lock deltas. This opt-in avoids dilution by zero-cost checkpoints.
    terminal_cost_scaling: bool = False


@dataclasses.dataclass(frozen=True)
class _PendingDecision:
    state: PhaseAwareState
    action: int
    behavior_probability: float
    blocked_time_ms: float
    lock_count: int
    behavior_action_probabilities: tuple[float, ...] = ()
    background_blocked_ms_caused: float = 0.0
    background_aborts_caused: int = 0
    agent_blocked_ms_caused: float = 0.0
    agent_aborts_caused: int = 0
    monotone_actions: bool = True
    decision_origin: str = ''


class TrajectoryCollector:
    def __init__(self):
        self._lock = threading.RLock()
        self._pending: Dict[str, _PendingDecision] = {}
        self._finished_txn_ids: set[str] = set()
        self._terminal_indices: Dict[str,int] = {}
        self._completed_agent_costs: Dict[str,float] = {}
        self._transitions: List[PolicyTransition] = []
        self._helper_externalities_seen: Dict[str, dict[str, float]] = {}
        self._helper_externalities_assigned: Dict[str, dict[str, float]] = {}
        self._helper_components: Dict[int, dict[str, float]] = {}
        self._helper_predecision: Dict[str, dict[str, float]] = {}

    def helper_predecision_snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {tid: dict(values) for tid, values in self._helper_predecision.items()}

    def _record_helper_component_locked(self, txn_id: str, index: int) -> None:
        seen = self._helper_externalities_seen.get(txn_id, {})
        assigned = self._helper_externalities_assigned.get(txn_id, {})
        delta = {field: max(0., float(seen.get(field, 0.)) - float(assigned.get(field, 0.)))
                 for field in ("agent_blocked_ms_caused", "background_blocked_ms_caused")}
        if any(delta.values()):
            self._helper_components[index] = delta
        self._helper_externalities_assigned[txn_id] = dict(seen)

    def helper_component_snapshot(self) -> dict[int, dict[str, float]]:
        with self._lock:
            return {index: dict(values) for index, values in self._helper_components.items()}

    def note_helper_externalities(self, txn_id: str, totals: dict[str, float]) -> None:
        """Owner-thread checkpoint totals already included in its metrics."""
        with self._lock:
            self._helper_externalities_seen[str(txn_id)] = dict(totals)

    def reconcile_helper_externalities(self, totals: dict[str, dict[str, float]]) -> dict[str, object]:
        """Add only costs after the owner's last checkpoint/finish.

        Call after workers have drained and reservations have been released.
        Never synthesize transitions for attempts with no policy decisions.
        Repeated reconciliation with the same totals is idempotent.
        """
        with self._lock:
            terminal = {row.txn_id: i for i, row in enumerate(self._transitions) if row.done}
            patched, uncovered, late, uncovered_ms = 0, 0, 0., 0.
            for tid, values in totals.items():
                seen = self._helper_externalities_seen.get(tid, {})
                assigned = self._helper_externalities_assigned.get(tid, {})
                assert all(math.isfinite(float(values.get(field, 0.))) and
                           float(values.get(field, 0.)) >= float(seen.get(field, 0.)) - 1e-9
                           for field in ("agent_blocked_ms_caused", "background_blocked_ms_caused"))
                # A finish callback may sample metrics again after the pending
                # decision was already consumed. Observed is not necessarily
                # assigned: reconcile against costs actually allocated to a
                # transition (or the pre-decision baseline), not the last read.
                assert all(float(values.get(field, 0.)) >= float(assigned.get(field, 0.)) - 1e-9
                           for field in ("agent_blocked_ms_caused", "background_blocked_ms_caused"))
                delta = {field: max(0., float(values.get(field, 0.)) - float(assigned.get(field, 0.)))
                         for field in ("agent_blocked_ms_caused", "background_blocked_ms_caused")}
                if not any(delta.values()):
                    continue
                if tid not in terminal:
                    uncovered += 1
                    uncovered_ms += sum(delta.values())
                    continue
                index = terminal[tid]
                old = self._transitions[index]
                updated = dataclasses.replace(old, **{field: getattr(old, field) + amount
                                                      for field, amount in delta.items()})
                self._transitions[index] = dataclasses.replace(updated,
                    reward=paper_reward(updated, reward_normalization((updated,))))
                components = self._helper_components.setdefault(index, {})
                for field, amount in delta.items():
                    components[field] = components.get(field, 0.) + amount
                self._helper_externalities_seen[tid] = dict(values)
                self._helper_externalities_assigned[tid] = dict(values)
                patched += 1
                late += sum(delta.values())
            return dict(patched_terminal_transitions=patched, uncovered_owner_attempts=uncovered,
                        late_externality_ms=late, uncovered_externality_ms=uncovered_ms,
                        synthetic_transitions_added=0)

    def decision(
        self,
        txn_id: str,
        state: PhaseAwareState,
        action: int,
        behavior_probability: float = 1.0,
        *,
        blocked_time_ms: float = 0.0,
        lock_count: int = 0,
        behavior_action_probabilities: Sequence[float] = (),
        background_blocked_ms_caused: float = 0.0,
        background_aborts_caused: int = 0,
        agent_blocked_ms_caused: float = 0.0,
        agent_aborts_caused: int = 0,
        monotone_actions: bool = True,
        decision_origin: str = '',
    ) -> None:
        with self._lock:
            # A wound callback can finish an attempt while its owner is still
            # returning from a policy lookup. Never reopen that terminal
            # trajectory or reset its pre-decision cost baseline.
            if str(txn_id) in self._finished_txn_ids:
                return
            previous = self._pending.get(str(txn_id))
            if previous is None:
                # Costs before the first policy lookup are not caused by its
                # selected action; keep them outside this action trajectory.
                self._helper_externalities_assigned[str(txn_id)] = dict(
                    self._helper_externalities_seen.get(str(txn_id), {}))
                if any(self._helper_externalities_assigned[str(txn_id)].values()):
                    self._helper_predecision[str(txn_id)] = dict(self._helper_externalities_assigned[str(txn_id)])
            if previous is not None:
                system_delta = system_performance_delta(previous.state, state)
                self._transitions.append(
                    PolicyTransition(
                        str(txn_id),
                        previous.state,
                        previous.action,
                        system_delta,
                        state,
                        False,
                        previous.behavior_probability,
                        lock_wait_ms=max(0.0, float(blocked_time_ms) - previous.blocked_time_ms),
                        new_lock_count=max(0, int(lock_count) - previous.lock_count),
                        system_delta=system_delta,
                        behavior_action_probabilities=previous.behavior_action_probabilities,
                        monotone_actions=previous.monotone_actions,
                        decision_origin=previous.decision_origin,
                        background_blocked_ms_caused=max(
                            0.0,
                            float(background_blocked_ms_caused)
                            - previous.background_blocked_ms_caused,
                        ),
                        background_aborts_caused=max(
                            0,
                            int(background_aborts_caused) - previous.background_aborts_caused,
                        ),
                        agent_blocked_ms_caused=max(
                            0.0,
                            float(agent_blocked_ms_caused) - previous.agent_blocked_ms_caused,
                        ),
                        agent_aborts_caused=max(
                            0,
                            int(agent_aborts_caused) - previous.agent_aborts_caused,
                        ),
                    )
                )
                self._record_helper_component_locked(str(txn_id), len(self._transitions) - 1)

            probability = float(behavior_probability)
            if not 0.0 < probability <= 1.0:
                raise ValueError("behavior probability must be in (0, 1]")
            self._pending[str(txn_id)] = _PendingDecision(
                state,
                int(action),
                probability,
                max(0.0, float(blocked_time_ms)),
                max(0, int(lock_count)),
                tuple(float(value) for value in behavior_action_probabilities),
                max(0.0, float(background_blocked_ms_caused)),
                max(0, int(background_aborts_caused)),
                max(0.0, float(agent_blocked_ms_caused)),
                max(0, int(agent_aborts_caused)),
                bool(monotone_actions),
                str(decision_origin),
            )

    def finish(
        self,
        txn_id: str,
        state: PhaseAwareState,
        *,
        committed: bool,
        operation_cost_ms: float,
        agent_cost_ms: float,
        retry_cost_ms: float = 0.0,
        blocked_time_ms: float = 0.0,
        lock_count: int = 0,
        background_blocked_ms_caused: float = 0.0,
        background_aborts_caused: int = 0,
        agent_blocked_ms_caused: float = 0.0,
        agent_aborts_caused: int = 0,
    ) -> None:
        with self._lock:
            if str(txn_id) in self._finished_txn_ids:
                return
            self._finished_txn_ids.add(str(txn_id))
            previous = self._pending.pop(str(txn_id), None)
            completed_cost=self._completed_agent_costs.pop(str(txn_id),0.)
            if previous is not None:
                system_delta = system_performance_delta(previous.state, state)
                transition = PolicyTransition(
                    str(txn_id),
                    previous.state,
                    previous.action,
                    0.0,
                    state,
                    True,
                    previous.behavior_probability,
                    commit_indicator=int(bool(committed)),
                    abort_indicator=int(not committed),
                    operation_cost_ms=max(0.0, float(operation_cost_ms)),
                    agent_cost_ms=max(0.0, float(agent_cost_ms),completed_cost),
                    late_agent_cost_ms=max(0.,completed_cost-float(agent_cost_ms)),
                    retry_cost_ms=max(0.0, float(retry_cost_ms)),
                    lock_wait_ms=max(0.0, float(blocked_time_ms) - previous.blocked_time_ms),
                    new_lock_count=max(0, int(lock_count) - previous.lock_count),
                    system_delta=system_delta,
                    behavior_action_probabilities=previous.behavior_action_probabilities,
                    monotone_actions=previous.monotone_actions,
                    decision_origin=previous.decision_origin,
                    background_blocked_ms_caused=max(
                        0.0,
                        float(background_blocked_ms_caused)
                        - previous.background_blocked_ms_caused,
                    ),
                    background_aborts_caused=max(
                        0,
                        int(background_aborts_caused) - previous.background_aborts_caused,
                    ),
                    agent_blocked_ms_caused=max(
                        0.0,
                        float(agent_blocked_ms_caused) - previous.agent_blocked_ms_caused,
                    ),
                    agent_aborts_caused=max(
                        0,
                        int(agent_aborts_caused) - previous.agent_aborts_caused,
                    ),
                )
                self._transitions.append(
                    dataclasses.replace(
                        transition,
                        reward=paper_reward(transition, reward_normalization((transition,))),
                    )
                )
                self._terminal_indices[str(txn_id)]=len(self._transitions)-1
                self._record_helper_component_locked(str(txn_id), len(self._transitions) - 1)

    def reconcile_terminal_agent_cost(self,txn_id: str,completed_agent_cost_ms: float) -> float:
        """Attach only owner-measured call work finalized after CC termination.

        A wound cannot cancel an in-flight external call. Its eventual paid
        cost belongs in the terminal reward, not in an earlier observation or
        a fabricated new decision. The late-accounted interval may also include
        time before the wound that had not yet been sampled. Idempotent.
        """
        observed=float(completed_agent_cost_ms)
        if not math.isfinite(observed) or observed<0:raise ValueError('Finite completed agent cost required')
        with self._lock:
            index=self._terminal_indices.get(str(txn_id))
            if index is None:
                # The owner may finish while a wound callback has copied its
                # arguments but has not yet appended the terminal row. Keep a
                # measured watermark only if an existing decision can finish.
                if str(txn_id) in self._pending:
                    self._completed_agent_costs[str(txn_id)]=max(observed,
                        self._completed_agent_costs.get(str(txn_id),0.))
                return 0.
            old=self._transitions[index]
            if observed<old.agent_cost_ms-1e-6:raise ValueError('Completed agent cost cannot decrease')
            delta=max(0.,observed-old.agent_cost_ms)
            if delta:
                updated=dataclasses.replace(old,agent_cost_ms=observed,late_agent_cost_ms=old.late_agent_cost_ms+delta)
                self._transitions[index]=dataclasses.replace(updated,
                    reward=paper_reward(updated,reward_normalization((updated,))))
            return delta

    def snapshot(self) -> tuple[PolicyTransition, ...]:
        with self._lock:
            return tuple(self._transitions)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._finished_txn_ids.clear()
            self._terminal_indices.clear()
            self._completed_agent_costs.clear()
            self._transitions.clear()
            self._helper_externalities_seen.clear()
            self._helper_externalities_assigned.clear()
            self._helper_components.clear()
            self._helper_predecision.clear()


def system_performance_delta(previous: PhaseAwareState, current: PhaseAwareState) -> float:
    """Paper Delta-Psys term over logical agent tasks and conflict aborts."""
    throughput_scale = max(
        1.0,
        previous.global_agent_task_throughput,
        current.global_agent_task_throughput,
    )
    throughput_delta = (
        current.global_agent_task_throughput - previous.global_agent_task_throughput
    ) / throughput_scale
    latency_scale = max(
        1.0,
        previous.global_agent_task_tail_latency_ms,
        current.global_agent_task_tail_latency_ms,
    )
    latency_delta = (
        previous.global_agent_task_tail_latency_ms
        - current.global_agent_task_tail_latency_ms
    ) / latency_scale
    abort_delta = previous.global_conflict_abort_rate - current.global_conflict_abort_rate
    background_throughput_scale = max(
        1.0,
        previous.global_background_throughput,
        current.global_background_throughput,
    )
    background_throughput_delta = (
        current.global_background_throughput - previous.global_background_throughput
    ) / background_throughput_scale
    background_abort_delta = (
        previous.global_background_abort_rate - current.global_background_abort_rate
    )
    combined = max(
        -3.0,
        min(
            3.0,
            throughput_delta
            + latency_delta
            + abort_delta
            + background_throughput_delta
            + background_abort_delta,
        ),
    )
    return combined


def reward_normalization(
    transitions: Sequence[PolicyTransition],
    *, positive_wait_event_scaling: bool = False, terminal_cost_scaling: bool = False,
) -> dict[str, float]:
    def wait_scale(values):
        return robust_scale(value for value in values if not positive_wait_event_scaling or value > 0)
    cost_rows = [row for row in transitions if row.done] if terminal_cost_scaling else transitions
    return {
        "operation_cost_ms": robust_scale(row.operation_cost_ms for row in cost_rows),
        "agent_cost_ms": robust_scale(row.agent_cost_ms for row in cost_rows),
        "retry_cost_ms": robust_scale(row.retry_cost_ms for row in cost_rows),
        "lock_wait_ms": wait_scale(row.lock_wait_ms for row in transitions),
        "new_lock_count": robust_scale(row.new_lock_count for row in transitions),
        "background_blocked_ms_caused": wait_scale(
            row.background_blocked_ms_caused for row in transitions
        ),
        "background_aborts_caused": robust_scale(
            row.background_aborts_caused for row in transitions
        ),
        "agent_blocked_ms_caused": wait_scale(
            row.agent_blocked_ms_caused for row in transitions
        ),
        "agent_aborts_caused": robust_scale(
            row.agent_aborts_caused for row in transitions
        ),
    }


def apply_paper_rewards(
    transitions: Sequence[PolicyTransition],
    config: PaperRewardConfig | None = None,
) -> tuple[list[PolicyTransition], dict[str, object]]:
    reward_config = config or PaperRewardConfig()
    normalization = reward_normalization(transitions,
        positive_wait_event_scaling=reward_config.positive_wait_event_scaling,
        terminal_cost_scaling=reward_config.terminal_cost_scaling)
    rows = [
        dataclasses.replace(
            transition,
            reward=paper_reward(transition, normalization, reward_config),
        )
        for transition in transitions
    ]
    return rows, {
        "formula": "beta1*commit-beta2*(1+C_restart+beta3*C_retry)*abort-lambda*C_lock+eta*delta_Psys",
        "config": dataclasses.asdict(reward_config),
        "normalization": normalization,
    }


def paper_reward(
    transition: PolicyTransition,
    normalization: dict[str, float],
    config: PaperRewardConfig | None = None,
) -> float:
    cfg = config or PaperRewardConfig()
    restart_cost = (
        transition.operation_cost_ms / normalization["operation_cost_ms"]
        + transition.agent_cost_ms / normalization["agent_cost_ms"]
    )
    retry_cost = transition.retry_cost_ms / normalization["retry_cost_ms"]
    lock_cost = (
        transition.lock_wait_ms / normalization["lock_wait_ms"]
        + transition.new_lock_count / normalization["new_lock_count"]
        + transition.background_blocked_ms_caused
        / normalization["background_blocked_ms_caused"]
        + transition.background_aborts_caused
        / normalization["background_aborts_caused"]
        + transition.agent_blocked_ms_caused
        / normalization["agent_blocked_ms_caused"]
        + transition.agent_aborts_caused
        / normalization["agent_aborts_caused"]
    )
    return (
        cfg.commit_reward * int(transition.commit_indicator)
        - cfg.abort_weight
        * (1.0 + restart_cost + cfg.retry_weight * retry_cost)
        * int(transition.abort_indicator)
        - cfg.lock_weight * lock_cost
        - cfg.deferred_action_weight * int(int(transition.action) >= 16)
        + cfg.system_weight * float(transition.system_delta)
    )


def robust_scale(values) -> float:
    rows = sorted(max(0.0, float(value)) for value in values if math.isfinite(float(value)))
    if not rows:
        return 1.0
    index = max(0, min(len(rows) - 1, math.ceil(0.95 * len(rows)) - 1))
    return max(1.0, rows[index])


def policy_transition_from_dict(
    data: dict[str, object],
    *,
    source_id: str | None = None,
) -> PolicyTransition:
    row = dict(data)
    fields = {field.name for field in dataclasses.fields(PolicyTransition)}
    payload = {
        key: value
        for key, value in row.items()
        if key in fields and key not in {"state", "next_state"}
    }
    if source_id is not None:
        payload["source_id"] = str(source_id)
    if "behavior_action_probabilities" in payload:
        payload["behavior_action_probabilities"] = tuple(
            float(value) for value in payload["behavior_action_probabilities"] or ()
        )
    return PolicyTransition(
        state=phase_aware_state_from_dict(dict(row["state"])),
        next_state=phase_aware_state_from_dict(dict(row["next_state"])),
        **payload,
    )
