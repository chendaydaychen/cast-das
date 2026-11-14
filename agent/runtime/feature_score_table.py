"""Exact additive lookup deployment for the shared bucketed categorical actor.

This is an alternative representation, not the paper's nearest-medoid action
table. Each finite feature value indexes 16 precomputed score contributions;
ordered sums, context residuals and frozen prior probabilities retain the
original scoring arithmetic. No actor inference or test-state fitting occurs.
"""
import hashlib
import json
import math

from agent.cc.atcc.ppo import policy_group_key, state_features, valid_actions
from .context import LockAction
from .joint_mask_policy import JointMaskTablePolicy, _allowed_masks, _legal_choices, joint_context_key, validate_residual_context
from .state_collector import policy_observation
from .table_observation import TABLE_STATE_PROJECTION, TABLE_STATE_PROJECTIONS, LONG_TIME_STATE_PROJECTION, project_table_state, projected_context_key


def feature_domains(preserve_growth, state_projection=TABLE_STATE_PROJECTION):
    binary = (0., 1.)
    sizes = (0, 1, 4, 9)
    size4 = tuple(math.log1p(x)/4. for x in sizes)
    size5 = tuple(math.log1p(x)/5. for x in sizes)
    growth = tuple(math.log1p(x)/4. for x in (range(17) if preserve_growth else sizes))
    ratios = (0., .125, .5, 1.)
    log_values = (0.,) + tuple(float(2**i) for i in range(15))
    log8 = tuple(math.log1p(x)/8. for x in log_values)
    duration_values = ((0.,) + tuple(float(2**i) for i in range(31))
                       if state_projection == LONG_TIME_STATE_PROJECTION else log_values)
    log10 = tuple(math.log1p(x)/10. for x in duration_values)
    return ((1.,), binary, binary, binary, size5, size5, growth, growth,
            ratios, ratios, ratios, (0., 1/3, 2/3, 1.),
            binary, binary, binary, binary, binary, binary, binary, binary,
            size4, size4, ratios, log8, log10, log10, log10, log10,
            size4, size5, log10, log8, log10, log10, ratios, log8, ratios)


class FeatureScoreTable:
    artifact_type = 'cast-das-exact-feature-score-table'

    @classmethod
    def compile(cls, source):
        # Validate the source and require a finite, shared observation schema.
        JointMaskTablePolicy.from_dict(source)
        if source.get('state_projection') not in TABLE_STATE_PROJECTIONS:
            raise ValueError('Exact feature tables require shared bucketed observations')
        domains = feature_domains(source['growth_conditioned_groups'], source['state_projection'])
        payload = dict(artifact_type=cls.artifact_type, version=1,
            source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
            schema={key:source[key] for key in ('phase_awareness', 'remove_phase_index_proxy',
                'growth_conditioned_groups', 'local_context_groups', 'generation', 'state_projection')},
            allowed_masks=source.get('allowed_masks'), prior_uniform_mix=source.get('prior_uniform_mix', .2),
            groups=source['groups'], support=source['support'],
            # Feature order and floating-point reduction order match dot().
            features=[dict(values=list(values), contributions=[
                [float(source['weights'][action][index])*value for action in range(16)]
                for value in values]) for index, values in enumerate(domains)],
            prior=cls.compile(source['prior_policy']).to_dict() if source.get('prior_policy') else None,
            **({'remove_phase_features':True} if source.get('remove_phase_features') else {}),
            **({'residual_context':source['residual_context']} if source.get('residual_context') else {}))
        return cls(payload)

    def __init__(self, payload):
        if payload.get('artifact_type') != self.artifact_type or payload.get('version') != 1:
            raise ValueError('Unsupported feature score table')
        self.schema = dict(payload['schema'])
        self.remove_phase_features = bool(payload.get('remove_phase_features',False))
        self.residual_context=payload.get('residual_context','')
        validate_residual_context(self.residual_context)
        names = ('phase_awareness', 'remove_phase_index_proxy', 'growth_conditioned_groups',
                 'local_context_groups', 'generation', 'state_projection')
        if set(self.schema) != set(names) or self.schema['state_projection'] not in TABLE_STATE_PROJECTIONS:
            raise ValueError('Unsupported observation schema')
        if (self.schema['phase_awareness'] not in ('enabled', 'disabled') or
                self.schema['remove_phase_index_proxy'] and self.schema['phase_awareness'] != 'disabled'):
            raise ValueError('Invalid phase observation')
        for key, value in self.schema.items():
            setattr(self, key, value)
        expected = feature_domains(self.growth_conditioned_groups, self.state_projection)
        if len(payload['features']) != len(expected):
            raise ValueError('Feature count differs')
        self.lookups = []
        for feature, domain in zip(payload['features'], expected):
            if tuple(feature['values']) != domain or len(feature['contributions']) != len(domain):
                raise ValueError('Feature domain differs')
            contributions = [tuple(row) for row in feature['contributions']]
            if any(len(row) != 16 or any(not math.isfinite(v) for v in row) for row in contributions):
                raise ValueError('Invalid score contributions')
            self.lookups.append(dict(zip(domain, contributions)))
        self.groups = {key:tuple(values) for key, values in payload['groups'].items()}
        self.support = {key:frozenset(values) for key, values in payload['support'].items()}
        if (any(len(values) != 16 or any(not math.isfinite(v) for v in values) for values in self.groups.values()) or
                any(type(a) is not int or a not in range(16) for values in self.support.values() for a in values)):
            raise ValueError('Invalid group scores or action support')
        self.allowed_masks = _allowed_masks(payload['allowed_masks'])
        self.prior_uniform_mix = float(payload['prior_uniform_mix'])
        self.prior_policy = FeatureScoreTable(payload['prior']) if payload['prior'] else None
        if self.prior_policy and (not 0 < self.prior_uniform_mix <= 1 or
                any(getattr(self, name) != getattr(self.prior_policy, name)
                    for name in (*names, 'remove_phase_features') if name != 'generation')):
            raise ValueError('Prior observation or mixture differs')
        # Serialized data contains precomputed contributions, never actor weights.
        self._payload = json.loads(json.dumps(payload))

    def probabilities(self, state):
        state = policy_observation(state, self.phase_awareness,
            remove_phase_index_proxy=self.remove_phase_index_proxy,
            remove_phase_features=self.remove_phase_features)
        state = project_table_state(state, preserve_growth=self.growth_conditioned_groups, mode=self.state_projection)
        key = self._context_key(state)
        return self._probabilities_prepared(state, state_features(state), key)

    def _context_key(self,state):
        return joint_context_key(state,growth_conditioned=self.growth_conditioned_groups,
            local_context=self.local_context_groups,state_projection=self.state_projection,
            residual_context=self.residual_context)

    def _probabilities_prepared(self, state, values, key):
        # All frozen prior levels share this projection/context schema. Reuse
        # preprocessing without changing any score or floating reduction order.
        valid = _legal_choices(state.current_action, self.allowed_masks)
        supported = self.support.get(key)
        if supported:
            valid = tuple(a for a in valid if a in supported or a == int(state.current_action))
        columns = [lookup[value] for lookup, value in zip(self.lookups, values)]
        residual = self.groups.get(key, (0.,)*16)
        prior = {}
        if self.prior_policy is not None:
            prior_key=self.prior_policy._context_key(state)
            probabilities = self.prior_policy._probabilities_prepared(state, values, prior_key)
            total = sum(probabilities[action] for action in valid)
            if total <= 0:
                raise ValueError('Parent prior has no legal mass')
            prior = {action:math.log((1.-self.prior_uniform_mix)*probabilities[action]/total+
                                    self.prior_uniform_mix/len(valid)) for action in valid}
        logits = {action:sum(column[action] for column in columns)+residual[action]+prior.get(action, 0.)
                  for action in valid}
        maximum = max(logits.values())
        mass = {action:math.exp(value-maximum) for action, value in logits.items()}
        total = sum(mass.values())
        return [mass.get(action, 0.)/total for action in range(16)]

    def select(self, state):
        probabilities = self.probabilities(state)
        return LockAction.from_encoded(max(valid_actions(state.current_action),
            key=lambda action:probabilities[action]))

    def to_dict(self):
        return json.loads(json.dumps(self._payload))
