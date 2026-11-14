"""Opt-in probability-preserving deployment, distinct from medoid argmax.

No actor is loaded. One uniform draw samples the monotone mask law encoded
by the four quantized coefficient rows. Retain argmax as a separate control.
"""
import math
import random
import threading
from .context import LockAction


class SampledQuantizedPolicy:
    supports_external_uniform=True
    requires_reviewer_contention_statistics=False

    def __init__(self,table,seed=0):
        self.table=table;self.metadata=table.metadata
        self.phase_awareness=self.metadata['phase_awareness']
        self._rng=random.Random(seed);self._lock=threading.Lock()

    def probabilities(self,state):
        scores=self.table.scores(state);maximum=max(scores.values())
        exp={a:math.exp(s-maximum) for a,s in scores.items()};total=sum(exp.values())
        return tuple(exp.get(a,0.)/total for a in range(16))

    def select_with_distribution(self,state,uniform=None):
        q=self.probabilities(state)
        if uniform is None:
            with self._lock:uniform=self._rng.random()
        if not 0<=uniform<1:raise ValueError('Uniform draw must be in [0,1)')
        cumulative=0.;last=None
        for action,p in enumerate(q):
            if not p:continue
            last=action;cumulative+=p
            if uniform<cumulative:return LockAction.from_encoded(action),q
        return LockAction.from_encoded(last),q

    def select(self,state):return self.select_with_distribution(state)[0]
