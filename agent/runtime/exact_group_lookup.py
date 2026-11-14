"""Opt-in equivalent compact indexing for an immutable compiled policy snapshot."""
import dataclasses
from .context import LockAction, LockClass
from .paper_policy import bucket_ratio, bucket_size, should_keep_occ


class ExactGroupLookup:
    """Replace only supported exact-group string construction; delegate fallback.

    The wrapped compiled snapshot and its configuration must not be mutated.
    Categorical refinement, guards, unknown groups and medoid ranking retain
    the original implementation. This is not a different learned policy.
    """
    def __init__(self, base):
        self.base = base
        self.protection_relevant_hotness = base.protection_relevant_hotness
        self._names = ('phase','action','waiters','abort','conflict_abort','hot',
                       'retry','conflict','recent_write','rounds','rs','ws')
        if base.local_context_groups:
            self._names = tuple(k for k in self._names if k not in ('waiters','abort','conflict_abort'))
        if base.growth_conditioned_groups:
            self._names += ('read_growth','write_growth')
        self._heads = {}
        self._texts = {}
        for text, action in base.calibrated_group_actions.items():
            items = dict(part.split('=',1) for part in text.split('|'))
            if set(items) != set(self._names):
                # Future/custom schema must use the original selector entirely.
                self._heads = {}
                break
            key=tuple(items[n] if n in ('phase','conflict') else int(items[n]) for n in self._names)
            self._heads[key] = action
            self._texts[key] = text
        self._categorical = base.refinement_actor.get('type') == 'locally-supported-categorical-lock-masks'

    def key(self, s):
        conflict = str(s.recent_conflict_kind).strip().lower() or 'none'
        common = (bucket_ratio(s.hotspot_access_ratio),min(3,s.retry_count),conflict,
                  bucket_ratio(s.recent_write_ratio),bucket_size(s.completed_rounds),
                  bucket_size(s.read_set_size),bucket_size(s.write_set_size))
        prefix = ('all' if self.base.phase_awareness=='disabled' else s.phase,int(s.current_action))
        if not self.base.local_context_groups:
            prefix += (bucket_size(s.global_waiter_count),bucket_ratio(s.global_abort_rate),
                       bucket_ratio(s.global_conflict_abort_rate))
        key = prefix + common
        if self.base.growth_conditioned_groups:
            key += (min(16,max(0,int(s.read_set_growth))),min(16,max(0,int(s.write_set_growth))))
        return key

    def select(self, state):
        if (self._categorical and not self.base.exact_categorical_lookup) or not self._heads:
            return self.base.select(state)
        if self.base.occ_cold_start_guard and should_keep_occ(state):
            return LockAction()
        key=self.key(state)
        head = self._heads.get(key)
        if head is not None:
            if self._categorical:
                observed = state if self.base.phase_awareness=='enabled' else dataclasses.replace(state,phase='all')
                refined=self.base._select_supported_categorical_exact(observed,self._texts[key])
                if refined is not None:return refined
            return LockAction(LockClass(head))
        return self.base.select(state)
