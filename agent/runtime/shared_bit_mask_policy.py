"""Opt-in four-bit parameter sharing with a complete, monotone 16-mask law.

Unlike the legacy factorized actor, rollout support is never restricted by
the trainer's observed-action bookkeeping. Frozen joint coefficients preserve
the scoring law used for sampling; there are no learned deployment overrides.
"""
import math

from agent.cc.atcc.ppo import DiscretePPOPolicy, FEATURE_COUNT, LOCK_BITS
from .joint_mask_policy import JointMaskTablePolicy, joint_context_key, validate_residual_context
from .state_collector import policy_observation
from .table_observation import TABLE_STATE_PROJECTION, TABLE_STATE_PROJECTIONS, project_table_state


class SharedBitMaskActor(DiscretePPOPolicy):
    artifact_type = 'cast-das-shared-bit-mask-actor'

    def __init__(self, *, remove_phase=False, seed=20261018031,
                 residual_context='observed-stage-mask-retry-v2', retain_explicit_phase=False,
                 state_projection=TABLE_STATE_PROJECTION):
        if remove_phase and retain_explicit_phase:
            raise ValueError('Removed phase inputs cannot retain an explicit phase label')
        super().__init__(phase_awareness='enabled' if retain_explicit_phase else 'disabled',
            local_context_groups=True, seed=seed)
        validate_residual_context(residual_context)
        self.remove_phase_index_proxy = bool(remove_phase)
        self.remove_phase_features = bool(remove_phase)
        if state_projection not in TABLE_STATE_PROJECTIONS:
            raise ValueError('Shared-bit actor requires a finite table projection')
        self.state_projection = state_projection
        self.residual_context = residual_context
        # Same 80% probability of mask0 as the categorical pilot, but a
        # factorized distribution among the other masks. Not identical priors.
        q = 1. - .8 ** .25
        bias = math.log(q / (1. - q))
        for row in self.actor_weights:
            row[0] = bias

    def observation(self, state):
        state = policy_observation(state, self.phase_awareness,
            remove_phase_index_proxy=self.remove_phase_index_proxy,
            remove_phase_features=self.remove_phase_features)
        return project_table_state(state, mode=self.state_projection)

    def group_key(self, state):
        return joint_context_key(self.observation(state), growth_conditioned=False,
            local_context=True, state_projection=self.state_projection,
            residual_context=self.residual_context)

    def prototype_key(self, state):
        return super().prototype_key(self.observation(state)) + '|' + self.group_key(state)

    def joint_payload(self, generation=0):
        # Sum coefficients first, then use exactly the same arithmetic as the
        # frozen joint scorer. The Bernoulli product and this softmax are
        # mathematically equal; this order also fixes floating-point tie order.
        with self._lock:
            return dict(artifact_type=JointMaskTablePolicy.artifact_type, version=1,
                weights=[[sum(self.actor_weights[i][j] for i, bit in enumerate(LOCK_BITS) if action & bit)
                    for j in range(FEATURE_COUNT)] for action in range(16)],
                groups={key: [sum(values[i] for i, bit in enumerate(LOCK_BITS) if action & bit)
                    for action in range(16)] for key, values in self.actor_group_logits.items()},
                support={}, phase_awareness=self.phase_awareness,
                remove_phase_index_proxy=self.remove_phase_index_proxy,
                remove_phase_features=self.remove_phase_features,
                growth_conditioned_groups=False, local_context_groups=True,
                generation=generation, state_projection=self.state_projection,
                residual_context=self.residual_context)

    def probabilities(self, state):
        state = self.observation(state)
        with self._lock:
            self.prototypes.setdefault(self.prototype_key(state), state)
            self.actor_group_logits[self.group_key(state)]
            return JointMaskTablePolicy(self.joint_payload()).probabilities(state)

    def value(self, state):
        return super().value(self.observation(state))

    def to_dict(self):
        with self._lock:
            payload = dict(artifact_type=self.artifact_type, version=1,
                weights=[list(row) for row in self.actor_weights],
                groups={key: list(value) for key, value in sorted(self.actor_group_logits.items())},
                remove_phase=self.remove_phase_features, residual_context=self.residual_context)
            if self.phase_awareness == 'enabled':
                payload.update(version=2, phase_awareness='enabled')
            if self.state_projection != TABLE_STATE_PROJECTION:
                # Old loaders reject v3 instead of silently applying their
                # legacy duration buckets to this opt-in artifact.
                payload.update(version=3, state_projection=self.state_projection,
                               phase_awareness=self.phase_awareness)
            return payload

    @classmethod
    def from_dict(cls, data, *, seed=20261018031):
        if data.get('artifact_type') != cls.artifact_type or data.get('version') not in (1, 2, 3):
            raise ValueError('Unsupported shared-bit actor')
        if data['version'] == 2 and data.get('phase_awareness') != 'enabled':
            raise ValueError('Version 2 requires explicit phase awareness')
        if data['version'] == 1 and 'phase_awareness' in data:
            raise ValueError('Legacy actor has no explicit phase metadata')
        if data['version'] in (1, 2) and 'state_projection' in data:
            raise ValueError('Legacy actor cannot declare a new projection')
        if data['version'] == 3 and (data.get('phase_awareness') not in ('enabled', 'disabled')
                or data.get('state_projection') not in TABLE_STATE_PROJECTIONS
                or data['state_projection'] == TABLE_STATE_PROJECTION):
            raise ValueError('Version 3 requires explicit phase and nonlegacy projection metadata')
        a = cls(remove_phase=data['remove_phase'], seed=seed, residual_context=data['residual_context'],
                retain_explicit_phase=data.get('phase_awareness') == 'enabled',
                state_projection=data.get('state_projection', TABLE_STATE_PROJECTION))
        if (len(data['weights']) != 4 or any(len(row) != FEATURE_COUNT for row in data['weights'])
                or any(len(row) != 4 for row in data['groups'].values())
                or any(not math.isfinite(v) for row in (*data['weights'], *data['groups'].values()) for v in row)):
            raise ValueError('Invalid shared-bit coefficients')
        a.actor_weights = [list(row) for row in data['weights']]
        a.actor_group_logits.update({key: list(row) for key, row in data['groups'].items()})
        return a
