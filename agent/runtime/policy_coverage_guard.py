"""Count real training observations with the deployed policy's own encoder."""
from collections import Counter
import threading
from .context import LockAction, LockClass


class PolicyCoverageGuard:
    """Keep existing protection on sparse states; never release acquired locks.

    key_fn must be the same encoder/projection used by the deployed policy.
    Counts come from training only. Evaluation updates query counters only.
    """
    def __init__(self, base, key_fn, training_states, minimum_samples=4):
        if minimum_samples < 1:
            raise ValueError('minimum_samples must be positive')
        self.base = base
        self.key_fn = key_fn
        self.minimum_samples = minimum_samples
        self.training_counts = Counter(key_fn(s) for s in training_states)
        self.query_counts = Counter()
        self.fallback_count = 0
        self._counter_lock = threading.Lock()

    def __getattr__(self, name):
        # Sampling helpers from the wrapped policy would bypass this guard.
        if name.startswith('select_with_'):
            raise AttributeError(name)
        return getattr(self.base, name)

    def select(self, state):
        key = self.key_fn(state)
        sparse = self.training_counts[key] < self.minimum_samples
        with self._counter_lock:
            self.query_counts[key] += 1
            self.fallback_count += int(sparse)
        if sparse:
            return LockAction(LockClass(int(state.current_action) & 15))
        return self.base.select(state)

    def report(self):
        total = sum(self.query_counts.values())
        unseen = sum(n for k, n in self.query_counts.items() if not self.training_counts[k])
        return dict(training_groups=len(self.training_counts),
                    training_observations=sum(self.training_counts.values()),
                    minimum_samples=self.minimum_samples, evaluation_questions=total,
                    unseen_questions=unseen, sparse_fallbacks=self.fallback_count,
                    unseen_fraction=unseen/total if total else 0.0,
                    fallback_fraction=self.fallback_count/total if total else 0.0,
                    evaluation_updates_training_counts=False)
