"""Run a policy checkpoint before the first issued operation of each phase."""


class ObservedBoundaryHooks:
    """Use the current phase and observed state at the tool boundary."""

    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def checkpoint(self, tx):
        phase = tx.context.phase.value
        if self.delegate.enabled(tx) and tx.metadata.get('_progressive_issued_phase') != phase:
            tx.metadata['_progressive_issued_phase'] = phase
            self.delegate._decide(tx, 'current-issued-phase-tool-boundary')

    def before_read(self, tx, key):
        self.checkpoint(tx)
        return self.delegate.before_read(tx, key)

    def before_write(self, tx, key):
        self.checkpoint(tx)
        return self.delegate.before_write(tx, key)
