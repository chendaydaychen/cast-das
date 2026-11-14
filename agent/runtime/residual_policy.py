"""Lossless raw-factorized-actor deployment as shared coefficients and context rows.

This is a compiled scoring representation, not a nearest-medoid action table.
It deliberately excludes actor objects, prototype mutation and actor fallback.
"""
import dataclasses
import math

from .context import LockAction
from .state_collector import policy_observation


class ResidualTablePolicy:
    artifact_type = "cast-das-factorized-residual-table"

    def __init__(self, *, weights, groups, support, phase_awareness="enabled",
                 stay_probability=0., generation=0, growth_conditioned_groups=False,
                 local_context_groups=False, remove_phase_index_proxy=False):
        from agent.cc.atcc.ppo import FEATURE_COUNT
        if phase_awareness not in {"enabled", "disabled"}:
            raise ValueError("invalid phase awareness")
        if remove_phase_index_proxy and phase_awareness != "disabled":
            raise ValueError("phase proxy removal requires NoPhase")
        if len(weights) != 4 or any(len(row) != FEATURE_COUNT for row in weights):
            raise ValueError("expected four shared feature rows")
        if not all(math.isfinite(float(v)) for row in weights for v in row):
            raise ValueError("nonfinite shared coefficient")
        if any(len(row) != 4 or not all(math.isfinite(float(v)) for v in row) for row in groups.values()):
            raise ValueError("invalid context coefficient row")
        if not math.isfinite(stay_probability) or not 0 <= stay_probability <= 1:
            raise ValueError("invalid retention probability")
        if any(any(type(a) is not int or not 0 <= a <= 15 for a in actions) for actions in support.values()):
            raise ValueError("invalid supported mask")
        self.weights = tuple(tuple(float(v) for v in row) for row in weights)
        self.groups = {str(key): tuple(float(v) for v in row) for key, row in groups.items()}
        self.support = {str(key): frozenset(actions) for key, actions in support.items()}
        self.phase_awareness = phase_awareness
        self.stay_probability = float(stay_probability)
        self.generation = int(generation)
        self.growth_conditioned_groups = bool(growth_conditioned_groups)
        self.local_context_groups = bool(local_context_groups)
        self.remove_phase_index_proxy = bool(remove_phase_index_proxy)

    @classmethod
    def compile(cls, actor, *, generation, remove_phase_index_proxy=False):
        if actor.categorical_actor or actor.timing_actions:
            raise ValueError("residual compilation supports the original factorized 16-action actor only")
        with actor._lock:
            return cls(weights=actor.actor_weights, groups=actor.actor_group_logits,
                support=actor.observed_group_actions, phase_awareness=actor.phase_awareness,
                stay_probability=actor.stay_probability, generation=generation,
                growth_conditioned_groups=actor.growth_conditioned_groups,
                local_context_groups=actor.local_context_groups,
                remove_phase_index_proxy=remove_phase_index_proxy)

    def select(self, state):
        from agent.cc.atcc.ppo import dot, policy_group_key, sigmoid, state_features, valid_actions
        state = policy_observation(state, self.phase_awareness)
        if self.remove_phase_index_proxy:
            state = dataclasses.replace(state, completed_rounds=0)
        current = int(state.current_action)
        valid = valid_actions(current)
        key = policy_group_key(state, growth_conditioned=self.growth_conditioned_groups,
                               local_context=self.local_context_groups)
        features = state_features(state)
        residual = self.groups.get(key, (0., 0., 0., 0.))
        bits = (1, 2, 4, 8)
        bit_probabilities = [1. if current & bit else sigmoid(dot(self.weights[i], features) + residual[i])
                             for i, bit in enumerate(bits)]
        # Match the original actor's arithmetic and tie order, including its
        # saturation/zero-mass behavior and optional stay mixture.
        mass = {}
        for action in valid:
            probability = 1.
            for i, bit in enumerate(bits):
                if not current & bit:
                    probability *= bit_probabilities[i] if action & bit else 1. - bit_probabilities[i]
            mass[action] = probability
        supported = self.support.get(key)
        if supported:
            allowed = set(valid) & (supported | {current})
            total = sum(mass[a] for a in valid if a in allowed)
            mass = {a: (mass[a] / total if total > 0 else 1. / len(allowed)) if a in allowed else 0.
                    for a in valid}
        if self.stay_probability:
            retain = self.stay_probability
            mass = {a: (1. - retain) * probability + (retain if a == current else 0.)
                    for a, probability in mass.items()}
        return LockAction.from_encoded(max(valid, key=lambda a: mass[a]))

    def to_dict(self):
        return dict(artifact_type=self.artifact_type, version=1, generation=self.generation,
            phase_awareness=self.phase_awareness, weights=[list(row) for row in self.weights],
            groups={key: list(row) for key, row in sorted(self.groups.items())},
            support={key: sorted(actions) for key, actions in sorted(self.support.items())},
            stay_probability=self.stay_probability, growth_conditioned_groups=self.growth_conditioned_groups,
            local_context_groups=self.local_context_groups, remove_phase_index_proxy=self.remove_phase_index_proxy,
            representation="shared continuous coefficients plus learned discrete context table",
            actor_fallback=False, nearest_medoid=False, deployment_overrides=False)

    @classmethod
    def from_dict(cls, payload):
        if payload.get("artifact_type") != cls.artifact_type or payload.get("version") != 1:
            raise ValueError("unsupported residual policy artifact")
        fields = ("weights", "groups", "support", "phase_awareness", "stay_probability", "generation",
                  "growth_conditioned_groups", "local_context_groups", "remove_phase_index_proxy")
        return cls(**{key: payload[key] for key in fields})
