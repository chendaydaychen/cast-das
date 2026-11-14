"""Training-only frequency horizon and optional external-access observation."""
import math
from scripts.revision.run_frozen_credit_short_updates import ShortUpdateHooks


def training_window(source):
    durations=[]
    for task in source['tasks']:
        costs=[call['elapsed_ms'] for name in ('llm_calls','tool_calls') for call in task.get(name,[])]
        if not costs or any(isinstance(c,bool) or not isinstance(c,(int,float)) or not math.isfinite(c) or c<0 for c in costs):
            raise ValueError('Finite nonnegative captured training durations required')
        durations.append(sum(costs))
    if not durations:raise ValueError('Nonempty training capture required')
    return max(1000.,1000.*math.ceil(max(durations)/1000.)),durations


class CountedExternalHooks(ShortUpdateHooks):
    """External OCC is unchanged; its accesses also enter the local heat count."""
    def before_read(self,txn,key):
        if txn.metadata.get('external_short_update'):
            self.foreground.observe_only(txn,key,write=False)
            return self.short.before_read(txn,key)
        return self.foreground.before_read(txn,key)

    def before_write(self,txn,key):
        if txn.metadata.get('external_short_update'):
            self.foreground.observe_only(txn,key,write=True)
            return self.short.before_write(txn,key)
        return self.foreground.before_write(txn,key)
