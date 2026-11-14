"""Joint complete-mask actor and immutable coefficient deployment.

The opt-in actor preserves a factorized parent's probabilities at conversion,
including its support restriction. PPO subsequently learns separate action
rows rather than tying all combinations to four Bernoulli logits.
"""
import math

from agent.cc.atcc.ppo import DiscretePPOPolicy, FEATURE_COUNT, LOCK_BITS, dot, policy_group_key, state_features, valid_actions
from .context import LockAction
from .residual_policy import ResidualTablePolicy
from .state_collector import policy_observation
from .table_observation import project_table_state, projected_context_key, validate_projection
from .paper_policy import bucket_size


def joint_context_key(state, *, growth_conditioned, local_context, state_projection, residual_context=''):
    if residual_context == 'mask-retry-v1':
        # Opt-in pooled residual: continuous phase/progress observations stay
        # in the actor feature vector. Only the tabular bias shares samples.
        return f'mask={int(state.current_action)}|retry={int(state.retry_count>0)}'
    if residual_context == 'observed-stage-mask-retry-v2':
        return (f'mask={int(state.current_action)}|retry={int(state.retry_count>0)}'
                f'|rounds={bucket_size(state.completed_rounds)}'
                f'|writing={int(state.write_set_growth>0 or state.recent_write_ratio>0)}')
    if residual_context == 'observed-stage-mask-retry-v1':
        # Inputs have already passed the actor's observation mask. No phase
        # label, planned access, future LLM duration or workload ID is used.
        # Two binary summaries keep the context small enough to share data.
        return (f'mask={int(state.current_action)}|retry={int(state.retry_count>0)}'
                f'|reasoned={int(state.completed_rounds>0)}'
                f'|writing={int(state.write_set_growth>0 or state.recent_write_ratio>0)}')
    if residual_context == 'phase-mask-retry-v1':
        return f'phase={state.phase}|mask={int(state.current_action)}|retry={int(state.retry_count>0)}'
    if residual_context:
        raise ValueError('Unknown residual context')
    key=policy_group_key(state,growth_conditioned=growth_conditioned,local_context=local_context)
    return projected_context_key(key,state) if state_projection else key


def validate_residual_context(mode):
    if mode not in ('','phase-mask-retry-v1','observed-stage-mask-retry-v1','observed-stage-mask-retry-v2','mask-retry-v1'):
        raise ValueError('Unknown residual context')


def _allowed_masks(value):
    if value is None:return None
    if (not isinstance(value,(list,tuple)) or not value or 0 not in value
            or any(type(a) is not int or a not in range(16) for a in value)
            or len(set(value))!=len(value)):
        raise ValueError('Allowed masks must be distinct four-bit integers including optimistic 0')
    return tuple(sorted(value))


def _legal_choices(current,allowed):
    valid=valid_actions(current)
    return valid if allowed is None else tuple(a for a in valid if a in allowed or a==int(current))


def _prior_logits(prior,state,valid,epsilon):
    if prior is None:return {}
    probabilities=prior.probabilities(state)
    total=sum(probabilities[a] for a in valid)
    if total<=0:raise ValueError('Parent prior has no legal mass')
    return {a:math.log((1.-epsilon)*probabilities[a]/total+epsilon/len(valid)) for a in valid}


def _validate_prior(owner,payload,epsilon):
    if payload is None:return None
    if not math.isfinite(epsilon) or not 0.<epsilon<=1.:
        raise ValueError('Prior uniform mixture must be in (0,1]')
    prior=JointMaskTablePolicy.from_dict(payload)
    for name in ('phase_awareness','remove_phase_index_proxy','remove_phase_features','growth_conditioned_groups',
                 'local_context_groups','state_projection'):
        if getattr(prior,name)!=getattr(owner,name):
            raise ValueError('Parent prior observation differs: '+name)
    return prior


class JointMaskActor(DiscretePPOPolicy):
    # State-only baselines also apply to this explicit categorical softmax.
    # This does not enable the legacy supported-projection deployment path.
    categorical_group_baseline_supported = True

    def __init__(self, *, frozen_support, remove_phase_index_proxy=False, remove_phase_features=False, state_projection='', allowed_masks=None,
                 prior_policy=None, prior_uniform_mix=.2, residual_context='', **kwargs):
        super().__init__(categorical_actor=True, **kwargs)
        self.frozen_support = {k:frozenset(v) for k,v in frozen_support.items()}
        self.remove_phase_index_proxy = remove_phase_index_proxy
        self.remove_phase_features = bool(remove_phase_features)
        validate_projection(state_projection)
        self.state_projection = state_projection
        validate_residual_context(residual_context)
        self.residual_context=residual_context
        self.allowed_masks=_allowed_masks(allowed_masks)
        self.prior_uniform_mix=float(prior_uniform_mix)
        self.prior_policy=_validate_prior(self,prior_policy,self.prior_uniform_mix)

    def observation(self, state):
        state = policy_observation(state, self.phase_awareness,
            remove_phase_index_proxy=self.remove_phase_index_proxy,
            remove_phase_features=self.remove_phase_features)
        return project_table_state(state, preserve_growth=self.growth_conditioned_groups, mode=self.state_projection) if self.state_projection else state

    def group_key(self, state):
        state = self.observation(state)
        return joint_context_key(state,growth_conditioned=self.growth_conditioned_groups,
            local_context=self.local_context_groups,state_projection=self.state_projection,
            residual_context=self.residual_context)

    def prototype_key(self, state):
        key = super().prototype_key(self.observation(state))
        return key + '|' + self.group_key(state) if self.state_projection else key

    def probabilities(self, state):
        state = self.observation(state)
        with self._lock:
            self.prototypes.setdefault(self.prototype_key(state),state)
            key = self.group_key(state)
            valid = _legal_choices(state.current_action,self.allowed_masks)
            supported = self.frozen_support.get(key)
            if supported:
                valid = tuple(a for a in valid if a in supported or a == int(state.current_action))
            features = state_features(state)
            residual = self.categorical_group_logits[key]
            prior=_prior_logits(self.prior_policy,state,valid,self.prior_uniform_mix)
            logits = {a:dot(self.categorical_actor_weights[a],features)+residual[a]+prior.get(a,0.) for a in valid}
            largest = max(logits.values())
            mass = {a:math.exp(v-largest) for a,v in logits.items()}
            total = sum(mass.values())
            return [mass.get(a,0.)/total for a in range(16)]


class JointMaskTablePolicy:
    artifact_type = 'cast-das-joint-mask-coefficients'

    def __init__(self, payload):
        if payload.get('artifact_type') != self.artifact_type or payload.get('version') != 1:
            raise ValueError('Unsupported joint artifact')
        self.weights = tuple(tuple(float(v) for v in row) for row in payload['weights'])
        self.groups = {str(k):tuple(float(v) for v in row) for k,row in payload['groups'].items()}
        self.support = {str(k):frozenset(v) for k,v in payload['support'].items()}
        self.phase_awareness = payload['phase_awareness']
        self.remove_phase_index_proxy = bool(payload['remove_phase_index_proxy'])
        self.remove_phase_features = bool(payload.get('remove_phase_features', False))
        self.growth_conditioned_groups = bool(payload['growth_conditioned_groups'])
        self.local_context_groups = bool(payload['local_context_groups'])
        self.generation = int(payload['generation'])
        self.state_projection = payload.get('state_projection', '')
        self.residual_context=payload.get('residual_context','')
        validate_residual_context(self.residual_context)
        self.allowed_masks=_allowed_masks(payload.get('allowed_masks'))
        self.prior_uniform_mix=float(payload.get('prior_uniform_mix',.2))
        self.prior_policy=_validate_prior(self,payload.get('prior_policy'),self.prior_uniform_mix)
        validate_projection(self.state_projection)
        if (self.phase_awareness not in ('enabled','disabled')
                or self.remove_phase_index_proxy and self.phase_awareness != 'disabled'
                or len(self.weights) != 16 or any(len(r) != FEATURE_COUNT for r in self.weights)
                or any(len(r) != 16 for r in self.groups.values())
                or any(not math.isfinite(v) for r in (*self.weights,*self.groups.values()) for v in r)
                or any(type(a) is not int or a not in range(16) for s in self.support.values() for a in s)):
            raise ValueError('Invalid joint artifact coefficients or support')

    def probabilities(self, state):
        state = policy_observation(state,self.phase_awareness,
            remove_phase_index_proxy=self.remove_phase_index_proxy,
            remove_phase_features=self.remove_phase_features)
        if self.state_projection:
            state = project_table_state(state, preserve_growth=self.growth_conditioned_groups, mode=self.state_projection)
        key = joint_context_key(state,growth_conditioned=self.growth_conditioned_groups,
            local_context=self.local_context_groups,state_projection=self.state_projection,
            residual_context=self.residual_context)
        valid = _legal_choices(state.current_action,self.allowed_masks)
        supported = self.support.get(key)
        if supported:
            valid = tuple(a for a in valid if a in supported or a == int(state.current_action))
        features = state_features(state)
        residual = self.groups.get(key,(0.,)*16)
        prior=_prior_logits(self.prior_policy,state,valid,self.prior_uniform_mix)
        logits = {a:dot(self.weights[a],features)+residual[a]+prior.get(a,0.) for a in valid}
        largest = max(logits.values())
        mass = {a:math.exp(v-largest) for a,v in logits.items()}
        total = sum(mass.values())
        return [mass.get(a,0.)/total for a in range(16)]

    def select(self, state):
        probabilities = self.probabilities(state)
        return LockAction.from_encoded(max(valid_actions(state.current_action),key=lambda a:probabilities[a]))

    def to_dict(self):
        return dict(artifact_type=self.artifact_type,version=1,weights=[list(r) for r in self.weights],
            groups={k:list(r) for k,r in sorted(self.groups.items())},
            support={k:sorted(v) for k,v in sorted(self.support.items())},
            phase_awareness=self.phase_awareness,remove_phase_index_proxy=self.remove_phase_index_proxy,
            growth_conditioned_groups=self.growth_conditioned_groups,local_context_groups=self.local_context_groups,
            generation=self.generation,
            **({'remove_phase_features':True} if self.remove_phase_features else {}),
            **({'residual_context':self.residual_context} if self.residual_context else {}),
            **({'allowed_masks':list(self.allowed_masks)} if self.allowed_masks is not None else {}),
            **({'prior_policy':self.prior_policy.to_dict(),'prior_uniform_mix':self.prior_uniform_mix}
               if self.prior_policy is not None else {}),
            **({'state_projection':self.state_projection} if self.state_projection else {}))

    @classmethod
    def from_dict(cls,payload):
        return cls(payload)

    @classmethod
    def compile(cls,actor,*,generation,remove_phase_index_proxy=False):
        if not isinstance(actor,JointMaskActor) or actor.stay_probability != 0.:
            raise ValueError('Expected unmixed joint actor')
        return cls(dict(artifact_type=cls.artifact_type,version=1,
            weights=actor.categorical_actor_weights,groups=actor.categorical_group_logits,
            support=actor.frozen_support,phase_awareness=actor.phase_awareness,
            remove_phase_index_proxy=remove_phase_index_proxy,
            **({'remove_phase_features':True} if actor.remove_phase_features else {}),
            growth_conditioned_groups=actor.growth_conditioned_groups,
            local_context_groups=actor.local_context_groups,generation=generation,
            **({'residual_context':actor.residual_context} if actor.residual_context else {}),
            **({'allowed_masks':list(actor.allowed_masks)} if actor.allowed_masks is not None else {}),
            **({'prior_policy':actor.prior_policy.to_dict(),'prior_uniform_mix':actor.prior_uniform_mix}
               if actor.prior_policy is not None else {}),
            **({'state_projection':actor.state_projection} if actor.state_projection else {})))


def convert_factorized_actor(payload, *, seed):
    parent = ResidualTablePolicy.from_dict(payload)
    if parent.stay_probability != 0.:
        raise ValueError('Exact joint conversion requires unmixed factorized parent')
    actor = JointMaskActor(seed=seed,phase_awareness=parent.phase_awareness,
        growth_conditioned_groups=parent.growth_conditioned_groups,
        local_context_groups=parent.local_context_groups,frozen_support=parent.support,
        remove_phase_index_proxy=parent.remove_phase_index_proxy)
    actor.categorical_actor_weights = [[sum(parent.weights[i][j] for i,b in enumerate(LOCK_BITS) if a&b)
        for j in range(FEATURE_COUNT)] for a in range(16)]
    actor.categorical_group_logits.update({k:[sum(row[i] for i,b in enumerate(LOCK_BITS) if a&b)
        for a in range(16)] for k,row in parent.groups.items()})
    actor.observed_group_actions.update({k:set(v) for k,v in parent.support.items()})
    return actor


def actor_from_joint_payload(payload, *, seed):
    policy = JointMaskTablePolicy.from_dict(payload)
    actor = JointMaskActor(seed=seed,phase_awareness=policy.phase_awareness,
        growth_conditioned_groups=policy.growth_conditioned_groups,local_context_groups=policy.local_context_groups,
        frozen_support=policy.support,remove_phase_index_proxy=policy.remove_phase_index_proxy,
        remove_phase_features=policy.remove_phase_features,
        state_projection=policy.state_projection,allowed_masks=policy.allowed_masks,
        residual_context=policy.residual_context,
        prior_policy=policy.prior_policy.to_dict() if policy.prior_policy is not None else None,
        prior_uniform_mix=policy.prior_uniform_mix)
    actor.categorical_actor_weights = [list(r) for r in policy.weights]
    actor.categorical_group_logits.update({k:list(r) for k,r in policy.groups.items()})
    actor.observed_group_actions.update({k:set(v) for k,v in policy.support.items()})
    return actor


def load_scoring_policy(payload):
    return (JointMaskTablePolicy.from_dict(payload) if payload.get('artifact_type') == JointMaskTablePolicy.artifact_type
        else ResidualTablePolicy.from_dict(payload))
