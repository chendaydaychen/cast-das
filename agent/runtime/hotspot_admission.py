"""Opt-in priority-aware reservations for already-revealed hotspot suffixes.

These reservations are performance helpers, never correctness locks. Native
WriteSet admission and version validation remain mandatory after a bypass.
The coordinator never calls into ATCC while holding its condition: ATCC wound
cleanup may release a reservation while holding the ATCC condition.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections import defaultdict
from typing import Callable, Iterable

from agent.cc.locks import LockConflict
from .context import TransactionContext, TransactionStatus


@dataclasses.dataclass
class _Request:
    context: TransactionContext
    targets: frozenset[str]
    sequence: int
    bypasses: int = 0
    wake: threading.Event = dataclasses.field(default_factory=threading.Event)
    externality_started_ns: int = 0
    externality_ended_ns: int = 0
    externality_owners: tuple[TransactionContext, ...] = ()

    @property
    def order(self):
        # After one overlapping admission passes it, protect the older waiter.
        # This also prevents starvation of a large set whose keys never become
        # simultaneously free under an endless stream of runnable subsets.
        protected = bool(self.bypasses)
        return (-int(protected), 0 if protected else -int(self.context.priority), int(self.context.start_ts_ns),
                self.context.tid, self.sequence)


def contains_cycle(graph: dict[str, set[str]], start: str) -> bool:
    """Whether this waiter's dependency paths return to itself."""
    pending = list(graph.get(start, ()))
    visited = set()
    while pending:
        current = pending.pop()
        if current == start:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(graph.get(current, ()))
    return False


class CoordinatedHotspotAdmission:
    def __init__(self, *, record_externalities: bool = False):
        self._condition = threading.Condition(threading.Lock())
        self._owners: dict[str, TransactionContext] = {}
        self._owned_targets: dict[int, set[str]] = {}
        self._requests: list[_Request] = []
        self._sequence = 0
        self._diagnostics = defaultdict(float)
        self.record_externalities = bool(record_externalities)
        self._externalities: dict[str, dict[str, float]] = {}

    @staticmethod
    def wait_event(event,timeout):
        """Single profiling seam; the event owns the actual wake semantics."""
        return event.wait(timeout=timeout)

    def _wait_owners_locked(self, request):
        if not self._live(request.context):
            return ()
        owners = {owner.tid: owner for key in request.targets
                  if (owner := self._owners.get(key)) is not None and owner is not request.context}
        return tuple(owners[tid] for tid in sorted(owners))

    def _flush_wait_externality_locked(self, request, now_ns):
        if not request.externality_started_ns:
            return
        end_ns = min(now_ns, request.externality_ended_ns) if request.externality_ended_ns else now_ns
        elapsed_ms = max(0., (end_ns - request.externality_started_ns) / 1_000_000.)
        if request.externality_owners and elapsed_ms:
            field = "background_blocked_ms_caused" if request.context.is_background else "agent_blocked_ms_caused"
            share = elapsed_ms / len(request.externality_owners)
            for owner in request.externality_owners:
                totals = self._externalities.setdefault(owner.tid, dict(
                    agent_blocked_ms_caused=0., background_blocked_ms_caused=0.))
                totals[field] += share
            self._diagnostics["externality_wait_ms"] += elapsed_ms
        request.externality_started_ns = max(request.externality_started_ns, end_ns)

    def _flush_externalities_locked(self, now_ns):
        if self.record_externalities:
            for request in self._requests:
                self._flush_wait_externality_locked(request, now_ns)

    def _refresh_wait_owners_locked(self):
        if self.record_externalities:
            for request in self._requests:
                if request.externality_started_ns and not request.externality_ended_ns:
                    request.externality_owners = self._wait_owners_locked(request)

    def externalities_snapshot(self):
        """Only reservation-owner Event-wait costs; excludes barriers/native waits."""
        with self._condition:
            self._flush_externalities_locked(time.monotonic_ns())
            return {tid: dict(values) for tid, values in self._externalities.items()}

    def externalities_for(self, tid):
        return self.externalities_snapshot().get(str(tid), dict(
            agent_blocked_ms_caused=0., background_blocked_ms_caused=0.))

    @staticmethod
    def _live(context):
        return context.status in (TransactionStatus.ACTIVE, TransactionStatus.WAITING)

    def _admissible_locked(self):
        # Work-conserving: a high-priority request whose OTHER key is busy must
        # not block a lower-priority whole set that can run now.
        occupied = set()
        selected = []
        barriers = {}
        live = [request for request in self._requests if self._live(request.context)]
        for request in sorted(live, key=lambda item: item.order):
            if not request.targets & occupied:
                available = not any(key in self._owners for key in request.targets)
                if available:
                    selected.append(request)
                    occupied.update(request.targets)
                elif request.bypasses:
                    # An intent barrier is not ownership and cannot publish.
                    # Existing owners run to completion, while new overlapping
                    # subsets stop passing this protected waiting whole set.
                    occupied.update(request.targets)
                    for key in request.targets:
                        barriers[key] = request
        return selected, barriers

    def _signal_locked(self):
        if not self._requests:
            return
        selected, _barriers = self._admissible_locked()
        for request in self._requests:
            if request in selected or not self._live(request.context):
                request.wake.set()

    def _dependencies_locked(self, selected, barriers):
        graph = defaultdict(set)
        for request in self._requests:
            if not self._live(request.context):
                continue
            for target in request.targets:
                owner = self._owners.get(target)
                if owner is not None and owner.tid != request.context.tid:
                    graph[request.context.tid].add(owner.tid)
                barrier = barriers.get(target)
                if barrier is not None and barrier is not request:
                    graph[request.context.tid].add(barrier.context.tid)
            if request not in selected and not graph.get(request.context.tid):
                graph[request.context.tid].update(other.context.tid for other in selected
                    if other.targets & request.targets)
        return dict(graph)

    def acquire(self, context: TransactionContext, object_ids: Iterable[str], *,
                timeout_s: float, native_dependencies: Callable[[], dict[str, set[str]]],
                priority_refresh: Callable[[], object] | None = None) -> bool:
        started = time.monotonic()
        deadline = started + max(0., float(timeout_s))
        with self._condition:
            if not self._live(context):
                raise LockConflict("ended transaction cannot reserve hotspots", tuple(sorted(object_ids)),
                                   kind="lock-preempted")
            targets = frozenset(str(key) for key in object_ids
                if self._owners.get(str(key)) is not context)
            if not targets:
                return True
            if not self._requests and not any(key in self._owners for key in targets):
                self._grant_locked(context, targets)
                self._diagnostics["admissions"] += 1
                return True
        # No competing queue needs a fresh score on the fast path. Contended
        # requests refresh OUTSIDE the helper lock (native lock-order contract).
        if priority_refresh is not None:
            priority_refresh()
        request = None
        waited = False
        try:
            with self._condition:
                if not self._live(context):
                    raise LockConflict("ended transaction cannot reserve hotspots", tuple(sorted(object_ids)),
                                       kind="lock-preempted")
                request = _Request(context, targets, self._sequence)
                self._sequence += 1
                self._requests.append(request)
            while True:
                with self._condition:
                    if not self._live(context):
                        raise LockConflict("transaction ended while waiting for hotspot admission",
                                           tuple(sorted(targets)), kind="lock-preempted")
                    selected, barriers = self._admissible_locked()
                    if request in selected:
                        # Reserve the entire new revealed set, or none of it.
                        # No partial gate prefix is retained while waiting.
                        overtakes = 0
                        for other in self._requests:
                            if (other is not request and other.sequence < request.sequence
                                    and other.targets & targets and other.bypasses == 0
                                    and self._live(other.context)):
                                other.bypasses += 1
                                overtakes += int(other.context.priority) < int(context.priority)
                        self._grant_locked(context, targets)
                        self._diagnostics["admissions"] += 1
                        self._diagnostics["priority_overtakes"] += overtakes
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._diagnostics["timeouts"] += 1
                        return False
                    if not waited:
                        waited = True
                        self._diagnostics["wait_calls"] += 1
                    # A waiter without ATCC locks or older helper reservations
                    # cannot close a cross-resource ownership cycle itself.
                    probe = bool(context.held_read_locks or context.held_write_locks
                                 or self._owned_targets.get(id(context)))
                    helper_graph = self._dependencies_locked(selected, barriers) if probe else None
                    request.wake.clear()
                if probe:
                    # Lock order is crucial: do not hold the helper condition
                    # while obtaining an ATCC dependency snapshot.
                    native_graph = native_dependencies()
                    graph = defaultdict(set)
                    for source in (helper_graph, native_graph):
                        for tid, dependencies in source.items():
                            graph[tid].update(dependencies)
                    cyclic = contains_cycle(graph, context.tid)
                    with self._condition:
                        self._diagnostics["dependency_probes"] += 1
                        if not self._live(context):
                            raise LockConflict("transaction ended during hotspot cycle inspection",
                                               tuple(sorted(targets)), kind="lock-preempted")
                        if cyclic:
                            # The two snapshots need not be globally atomic.
                            # A stale positive can only skip a helper, not skip
                            # native admission or version validation.
                            self._diagnostics["cycle_bypasses"] += 1
                            return True
                # Wake only newly runnable requests. Unlocked waiters need no
                # periodic polling; lock-holding waiters still inspect cycles.
                blocked_start = time.monotonic_ns()
                if self.record_externalities:
                    with self._condition:
                        request.externality_started_ns = time.monotonic_ns()
                        request.externality_ended_ns = 0
                        request.externality_owners = self._wait_owners_locked(request)
                try:
                    self.wait_event(request.wake,
                        min(.005,max(0.,deadline-time.monotonic()))
                        if probe else max(0.,deadline-time.monotonic()))
                finally:
                    blocked_end = time.monotonic_ns()
                    context.blocked_time_ms += max(0., (blocked_end - blocked_start) / 1_000_000.)
                    if self.record_externalities:
                        # Publish the end before acquiring the condition so
                        # concurrent ownership flushes cannot bill scheduling
                        # time after Event.wait has actually returned.
                        request.externality_ended_ns = blocked_end
                        with self._condition:
                            self._flush_wait_externality_locked(request, blocked_end)
                            request.externality_started_ns = 0
                            request.externality_ended_ns = 0
                            request.externality_owners = ()
        finally:
            with self._condition:
                if request is not None and request in self._requests:
                    self._requests.remove(request)
                if waited:
                    self._diagnostics["wait_elapsed_ms"] += (time.monotonic() - started) * 1000.
                self._signal_locked()

    def _grant_locked(self, context, targets):
        self._flush_externalities_locked(time.monotonic_ns())
        for key in targets:
            self._owners[key] = context
        self._owned_targets.setdefault(id(context), set()).update(targets)
        self._refresh_wait_owners_locked()

    def release_all(self, context: TransactionContext):
        """Also safe when wound cleanup races with a just-granted reservation."""
        with self._condition:
            self._flush_externalities_locked(time.monotonic_ns())
            for key in self._owned_targets.pop(id(context), ()):
                if self._owners.get(key) is context:
                    del self._owners[key]
            self._refresh_wait_owners_locked()
            self._signal_locked()

    def snapshot(self):
        with self._condition:
            return dict(owners={key: context.tid for key, context in self._owners.items()},
                        pending=len(self._requests), diagnostics=dict(self._diagnostics))
