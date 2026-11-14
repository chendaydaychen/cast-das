"""Opt-in fallback for a continued policy beyond its observed restart costs.

This is a limited negative-transfer guard, not a general OOD detector. It uses
only work already observed, never a future model-call duration or object set.
"""
import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class ContinuationCostGuard:
    child: object
    parent: object
    maximum_supported_cost_ms: float

    def __post_init__(self):
        if not math.isfinite(self.maximum_supported_cost_ms) or self.maximum_supported_cost_ms<0:
            raise ValueError('A finite nonnegative observed-cost bound is required')

    def uses_parent(self,state):
        try:cost=float(state.observed_restart_cost_ms)
        except (TypeError,ValueError):return True
        return not math.isfinite(cost) or cost<0 or cost>self.maximum_supported_cost_ms

    def select(self,state):
        policy=self.parent if self.uses_parent(state) else self.child
        action=policy.select(state)
        if action.encoded & int(state.current_action)!=int(state.current_action):
            raise ValueError('Fallback cannot remove protection already acquired')
        return action
