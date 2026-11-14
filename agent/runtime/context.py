"""Paper-aligned transaction context and monotonic ATCC lock action."""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
from typing import Dict, Iterable, Set


class TransactionStatus(str, enum.Enum):
    ACTIVE = "active"
    WAITING = "waiting"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"


class TransactionPhase(str, enum.Enum):
    EXPLORE = "explore"
    REFINE = "refine"
    COMMIT = "commit"


class LockClass(enum.IntFlag):
    NONE = 0
    HOT_READ = 1
    COLD_READ = 2
    HOT_WRITE = 4
    COLD_WRITE = 8


@dataclasses.dataclass(frozen=True)
class LockAction:
    protected: LockClass = LockClass.NONE
    # Experimental timing bit.  It is intentionally separate from the four
    # requested protection classes. Actual lock ownership is maintained by
    # held_read_locks / held_write_locks, independently of this request mask.
    defer_to_commit: bool = False

    @property
    def encoded(self) -> int:
        """Stable action id: [0,15] immediate; [16,31] commit-deferred."""
        return int(self.protected) | (16 if self.defer_to_commit else 0)

    @classmethod
    def from_encoded(cls, action: int) -> "LockAction":
        value = int(action)
        if value < 0 or value >= 32:
            raise ValueError("invalid timed lock action")
        return cls(LockClass(value & 0xF), defer_to_commit=bool(value & 16))

    def expands(self, previous: "LockAction") -> bool:
        return (self.protected | previous.protected) == self.protected

    def added_since(self, previous: "LockAction") -> LockClass:
        if not self.expands(previous):
            raise ValueError("ATCC lock actions may only expand protection")
        return self.protected & ~previous.protected

    def protects(self, *, hot: bool, write: bool) -> bool:
        target = (
            LockClass.HOT_WRITE if hot and write
            else LockClass.COLD_WRITE if write
            else LockClass.HOT_READ if hot
            else LockClass.COLD_READ
        )
        return bool(self.protected & target)


_ALLOWED_TRANSITIONS = {
    TransactionStatus.ACTIVE: {
        TransactionStatus.WAITING,
        TransactionStatus.COMMITTING,
        TransactionStatus.ABORTING,
    },
    TransactionStatus.WAITING: {TransactionStatus.ACTIVE, TransactionStatus.ABORTING},
    TransactionStatus.COMMITTING: {TransactionStatus.COMMITTED, TransactionStatus.ABORTING},
    TransactionStatus.ABORTING: {TransactionStatus.ABORTED},
    TransactionStatus.COMMITTED: set(),
    TransactionStatus.ABORTED: set(),
}


@dataclasses.dataclass
class TransactionContext:
    task_id: str
    attempt_id: int
    generation: int
    start_ts_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    snapshot_epoch: int = -1
    status: TransactionStatus = TransactionStatus.ACTIVE
    phase: TransactionPhase = TransactionPhase.EXPLORE
    action: LockAction = dataclasses.field(default_factory=LockAction)
    priority: int = 0
    priority_epoch: int = 0
    operation_cost_ms: float = 0.0
    agent_cost_ms: float = 0.0
    inflight_agent_started_ns: int = 0
    blocked_time_ms: float = 0.0
    retry_count: int = 0
    retry_validation_conflicts: int = 0
    retry_conflict_mask: int = 0
    retry_conflict_read_targets: Set[str] = dataclasses.field(default_factory=set)
    retry_conflict_write_targets: Set[str] = dataclasses.field(default_factory=set)
    prior_retry_cost_ms: float = 0.0
    # Actual lock wait from earlier attempts of this logical task; separate
    # from paid agent work and this attempt's measured blocked time.
    prior_retry_blocked_ms: float = 0.0
    recent_conflict_kind: str = "none"
    is_background: bool = False
    background_blocked_ms_caused: float = 0.0
    background_aborts_caused: int = 0
    agent_blocked_ms_caused: float = 0.0
    agent_aborts_caused: int = 0
    completed_operations: int = 0
    read_versions: Dict[str, int] = dataclasses.field(default_factory=dict)
    write_targets: Set[str] = dataclasses.field(default_factory=set)
    planned_write_targets: Set[str] = dataclasses.field(default_factory=set)
    hot_read_targets: Set[str] = dataclasses.field(default_factory=set)
    hot_write_targets: Set[str] = dataclasses.field(default_factory=set)
    held_read_locks: Set[str] = dataclasses.field(default_factory=set)
    held_write_locks: Set[str] = dataclasses.field(default_factory=set)
    policy_read_lock_targets: Set[str] = dataclasses.field(default_factory=set)
    policy_write_lock_targets: Set[str] = dataclasses.field(default_factory=set)
    lock_acquired_ns: int = 0
    lock_hold_time_ms: float = 0.0
    pending_request: str = ""
    undo_log_handle: str = ""
    checkpoint_id: str = ""
    last_operation_end_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    last_agent_accounted_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    round_index: int = 0
    _mutex: threading.RLock = dataclasses.field(default_factory=threading.RLock, repr=False)

    @property
    def tid(self) -> str:
        return f"{self.task_id}:{self.attempt_id}:{self.generation}"

    def transition(self, target: TransactionStatus) -> None:
        with self._mutex:
            if target not in _ALLOWED_TRANSITIONS[self.status]:
                raise RuntimeError(f"invalid transaction transition: {self.status.value}->{target.value}")
            self.status = target

    def try_transition(
        self,
        target: TransactionStatus,
        *,
        from_statuses: Iterable[TransactionStatus],
    ) -> bool:
        """Atomically transition only while the context is in an expected state."""
        expected = frozenset(from_statuses)
        with self._mutex:
            if self.status not in expected:
                return False
            if target not in _ALLOWED_TRANSITIONS[self.status]:
                raise RuntimeError(
                    f"invalid transaction transition: {self.status.value}->{target.value}"
                )
            self.status = target
            return True

    def change_phase(self, phase: TransactionPhase) -> None:
        with self._mutex:
            order = {TransactionPhase.EXPLORE: 0, TransactionPhase.REFINE: 1, TransactionPhase.COMMIT: 2}
            if order[phase] < order[self.phase]:
                raise ValueError("transaction phase cannot move backwards")
            if phase != self.phase:
                self.phase = phase
                self.round_index += 1

    def change_action(self, action: LockAction, *, allow_mask_reduction: bool = False) -> LockClass:
        with self._mutex:
            # Optional future-access mask revision is not an unlock. Held S/X
            # sets have an independent lifetime, still ending only at terminal.
            added = (action.protected & ~self.action.protected if allow_mask_reduction
                     else action.added_since(self.action))
            self.action = action
            return added

    def current_lock_hold_ms(self) -> float:
        with self._mutex:
            current = 0.0
            if self.lock_acquired_ns:
                current = (time.monotonic_ns() - self.lock_acquired_ns) / 1_000_000.0
            return self.lock_hold_time_ms + max(0.0, current)

    def note_conflict(self, kind: str) -> None:
        with self._mutex:
            normalized = str(kind).strip().lower()
            if normalized and normalized != "none":
                self.recent_conflict_kind = normalized
