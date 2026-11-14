"""Explicit opt-in hooks for the reviewer's first-eligible-hot-access rule."""
from dataclasses import replace

from .paper_hooks import PaperATCCHooks


class FirstHotAccessHooks(PaperATCCHooks):
    """Consult hot-only policy before the observed access, not after its round.

    Ordinary validation, monotone transitions, deferred writes and read-lock
    eligibility are unchanged. Manager must use a HotRulePolicy-compatible
    selector; these hooks are not installed by default.
    """
    def _first(self, txn, object_id):
        if not self.enabled(txn) or not self.manager.is_hot(object_id):return
        if int(txn.context.action.protected)&5 == 5:return
        state=self._state(txn)
        # The issued key is observable now, even before it joins the read set.
        state=replace(state, hotspot_access_ratio=max(state.hotspot_access_ratio, 1e-12))
        previous=int(txn.context.action.protected)
        self._select_and_apply(txn,state)
        self.manager.first_hot_access_events.append(dict(txn_id=txn.context.tid,
            object_id=str(object_id),phase=state.phase,previous=previous,
            selected=int(txn.context.action.protected),
            read_previously_returned=str(object_id) in txn.read_set,
            risk=max(state.global_abort_rate,state.global_conflict_abort_rate)))

    def before_read(self, txn, object_id):
        self._first(txn,object_id)
        return super().before_read(txn,object_id)

    def before_write(self, txn, object_id):
        self._first(txn,object_id)
        return super().before_write(txn,object_id)
