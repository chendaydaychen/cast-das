"""Exact observed-object deferred plans; no class-wide or future-footprint locks."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObservedProtection:
    object_id: str
    version: int
    write: bool
    observed_redo_cost_ms: float


class ObjectProtectionPlan:
    """Owner-thread plan; caller supplies actual measured dependency cost.

    materialize invokes an atomic validate-and-lock callback. A stale version
    must raise; it must never be silently rebased or treated as protected.
    Existing locks remain owned by the transaction and are never released here.
    """
    def __init__(self, minimum_redo_cost_ms):
        if not math.isfinite(minimum_redo_cost_ms) or minimum_redo_cost_ms < 0:
            raise ValueError('Finite nonnegative cost threshold required')
        self.minimum_redo_cost_ms = minimum_redo_cost_ms
        self.observed = {}
        self.pending = {}

    def observe(self, object_id, version, *, write=False, observed_redo_cost_ms=0):
        cost = float(observed_redo_cost_ms)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('Measured redo cost must be finite and nonnegative')
        key = str(object_id)
        previous = self.observed.get(key)
        if previous and previous.version != int(version):
            raise ValueError('Observed version changed within one attempt')
        self.observed[key] = ObservedProtection(key, int(version), bool(write or (previous and previous.write)),
            max(cost, previous.observed_redo_cost_ms if previous else 0))
        if key in self.pending:
            self.pending[key] = self.observed[key]

    def nominate(self, object_id):
        target = self.observed.get(str(object_id))
        if target is None:
            raise ValueError('Cannot protect a future or unobserved object')
        if target.observed_redo_cost_ms < self.minimum_redo_cost_ms:
            return False
        self.pending[target.object_id] = target
        return True

    def materialize(self, validate_and_lock, *, held_reads=(), held_writes=()):
        reads, writes = set(held_reads), set(held_writes)
        acquired = []
        for key, target in sorted(tuple(self.pending.items())):
            if key not in writes and (target.write or key not in reads):
                validate_and_lock(target)
                acquired.append(target)
            # Remove only after successful acquisition or an existing lock.
            self.pending.pop(key)
        return acquired
