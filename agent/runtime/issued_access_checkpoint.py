"""Opt-in policy checkpoint when an access is actually issued, before its read.

The policy receives only the existing observed state. No future access set or
new object feature is supplied. Native protection, monotonicity and final
validation are delegated unchanged. Training and serving must both install it.
"""
class IssuedAccessCheckpointHooks:
    def __init__(self, delegate):
        self.delegate = delegate
        self.events = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def _checkpoint(self, txn, key, kind):
        if not self.delegate.enabled(txn) or txn.metadata.get('_issued_checkpoint_active'):
            return
        txn.metadata['_issued_checkpoint_active'] = True
        try:
            state = self.delegate._state(txn)
            self.events.append(dict(tid=txn.context.tid, key=str(key), kind=kind,
                phase=state.phase, completed_operations=state.completed_operations))
            self.delegate._select_and_apply(txn, state)
        finally:
            txn.metadata.pop('_issued_checkpoint_active', None)

    def before_read(self, txn, key):
        self._checkpoint(txn, key, 'read')
        return self.delegate.before_read(txn, key)

    def before_write(self, txn, key):
        self._checkpoint(txn, key, 'write')
        return self.delegate.before_write(txn, key)
