"""Opt-in int16 coefficient-table deployment, distinct from medoid lookup.

Compiles frozen coefficients only. Runtime does not load a PPO actor, run
softmax, interpolate prototypes, or fall back to floating-point model weights.
Quantization can change close decisions; callers must measure fidelity.
"""
import base64
import math
import struct

from agent.cc.atcc.ppo import FEATURE_COUNT, state_features, valid_actions
from .context import LockAction
from .context_medoid_policy import OBSERVATION_FIELDS
from .joint_mask_policy import JointMaskTablePolicy, joint_context_key
from .state_collector import policy_observation
from .table_observation import project_table_state


class QuantizedMaskTable:
    artifact_type = 'cast-das-int16-mask-coefficient-table'
    coefficient_rows = 16

    def __init__(self, payload):
        expected = {'artifact_type', 'version', 'observation', 'generation',
                    'context_keys', 'scales', 'coefficients_le_i16'}
        if set(payload) != expected or payload['artifact_type'] != self.artifact_type or payload['version'] != 1:
            raise ValueError('Unsupported quantized mask table')
        self.metadata = dict(payload['observation'])
        if set(self.metadata) != set(OBSERVATION_FIELDS):
            raise ValueError('Incomplete observation contract')
        self.generation = int(payload['generation'])
        keys = payload['context_keys']
        if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys) or keys != sorted(set(keys)):
            raise ValueError('Invalid context keys')
        self.keys = tuple(keys)
        self.scales = tuple(payload['scales'])
        if len(self.scales) != self.coefficient_rows or any(type(s) not in (float, int) or not math.isfinite(s) or s < 0 for s in self.scales):
            raise ValueError('Invalid quantization scales')
        try:
            blob = base64.b64decode(payload['coefficients_le_i16'], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid coefficient encoding') from exc
        count = self.coefficient_rows * (FEATURE_COUNT + len(keys))
        if len(blob) != 2 * count:
            raise ValueError('Invalid coefficient length')
        values = struct.unpack('<' + str(count) + 'h', blob)
        rows = self.coefficient_rows
        self.weights = tuple(values[a * FEATURE_COUNT:(a + 1) * FEATURE_COUNT] for a in range(rows))
        offset = rows * FEATURE_COUNT
        self.groups = {key: values[offset + i * rows:offset + (i + 1) * rows] for i, key in enumerate(keys)}
        self.blob = blob
        # Validate metadata without retaining this zero-coefficient validator.
        JointMaskTablePolicy(dict(artifact_type=JointMaskTablePolicy.artifact_type,
            version=1, weights=[[0.] * FEATURE_COUNT for _ in range(16)],
            groups={}, support={}, generation=self.generation, **self.metadata))

    @classmethod
    def compile(cls, payload):
        model = JointMaskTablePolicy(payload)
        if model.support or model.allowed_masks is not None or model.prior_policy is not None:
            raise ValueError('Restricted supports or prior mixtures require a separate compiler')
        keys = sorted(model.groups)
        scales = [max([abs(v) for v in model.weights[a]] +
                      [abs(model.groups[k][a]) for k in keys]) / 32767. for a in range(16)]

        def quantize(value, action):
            return round(value / scales[action]) if scales[action] else 0

        values = [quantize(v, a) for a, row in enumerate(model.weights) for v in row]
        values += [quantize(model.groups[k][a], a) for k in keys for a in range(16)]
        blob = struct.pack('<' + str(len(values)) + 'h', *values)
        return cls(dict(artifact_type=cls.artifact_type, version=1,
            observation={k: getattr(model, k) for k in OBSERVATION_FIELDS},
            generation=model.generation, context_keys=keys, scales=scales,
            coefficients_le_i16=base64.b64encode(blob).decode('ascii')))

    def observation(self, state):
        m = self.metadata
        state = policy_observation(state, m['phase_awareness'],
            remove_phase_index_proxy=m['remove_phase_index_proxy'], remove_phase_features=m['remove_phase_features'])
        return project_table_state(state, preserve_growth=m['growth_conditioned_groups'], mode=m['state_projection']) if m['state_projection'] else state

    def _features_and_context(self, state):
        state = self.observation(state)
        m = self.metadata
        key = joint_context_key(state, growth_conditioned=m['growth_conditioned_groups'],
            local_context=m['local_context_groups'], state_projection=m['state_projection'],
            residual_context=m['residual_context'])
        return state_features(state), self.groups.get(key, (0,) * self.coefficient_rows)

    def scores(self, state):
        features, residual = self._features_and_context(state)
        return {a: self.scales[a] * (sum(v * f for v, f in zip(self.weights[a], features)) + residual[a])
                for a in valid_actions(state.current_action)}

    def score_error_bounds(self, state):
        """Conservative coefficient-rounding bounds, excluding float roundoff."""
        features, _ = self._features_and_context(state)
        size = .5 * (sum(abs(f) for f in features) + 1.)
        return {a: size * self.scales[a] for a in valid_actions(state.current_action)}

    def select(self, state):
        scores = self.scores(state)
        return LockAction.from_encoded(max(scores, key=scores.get))

    def to_dict(self):
        return dict(artifact_type=self.artifact_type, version=1,
            observation=dict(self.metadata), generation=self.generation,
            context_keys=list(self.keys), scales=list(self.scales),
            coefficients_le_i16=base64.b64encode(self.blob).decode('ascii'))


class QuantizedSharedBitTable(QuantizedMaskTable):
    """Quantize four shared rows directly, avoiding 16-mask expansion."""
    artifact_type = 'cast-das-int16-shared-bit-table'
    coefficient_rows = 4

    @classmethod
    def compile(cls, payload, generation=0):
        # Compile-time validation only; no actor is retained in the artifact.
        from .shared_bit_mask_policy import SharedBitMaskActor
        actor = SharedBitMaskActor.from_dict(payload)
        original = JointMaskTablePolicy(actor.joint_payload(generation))
        keys = sorted(payload['groups'])
        rows = payload['weights']
        groups = payload['groups']
        scales = [max([abs(v) for v in rows[a]] + [abs(groups[k][a]) for k in keys]) / 32767.
                  for a in range(4)]
        q = lambda v, a: round(v / scales[a]) if scales[a] else 0
        values = [q(v, a) for a, row in enumerate(rows) for v in row]
        values += [q(groups[k][a], a) for k in keys for a in range(4)]
        return cls(dict(artifact_type=cls.artifact_type, version=1,
            observation={k:getattr(original, k) for k in OBSERVATION_FIELDS},
            generation=generation, context_keys=keys, scales=scales,
            coefficients_le_i16=base64.b64encode(struct.pack('<' + str(len(values)) + 'h', *values)).decode('ascii')))

    def bit_scores(self, state):
        features, residual = self._features_and_context(state)
        return tuple(self.scales[i] * (sum(v * f for v, f in zip(row, features)) + residual[i])
                     for i, row in enumerate(self.weights))

    def scores(self, state):
        bits = self.bit_scores(state)
        return {a:sum(bits[i] for i in range(4) if a & (1 << i)) for a in valid_actions(state.current_action)}

    def score_error_bounds(self, state):
        features, _ = self._features_and_context(state)
        size = .5 * (sum(abs(f) for f in features) + 1.)
        return {a:size * sum(self.scales[i] for i in range(4) if a & (1 << i))
                for a in valid_actions(state.current_action)}

    def select(self, state):
        action = int(state.current_action)
        for i, score in enumerate(self.bit_scores(state)):
            if score > 0.:
                action |= 1 << i
        return LockAction.from_encoded(action)
