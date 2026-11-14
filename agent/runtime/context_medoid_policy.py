"""Training-context-preserving medoids; no actor weights or inference fallback."""
from .joint_mask_policy import joint_context_key, validate_residual_context
from .state_collector import policy_observation
from .table_observation import project_table_state, validate_projection
from .paper_policy import CompiledPhasePolicy

OBSERVATION_FIELDS = ('phase_awareness','remove_phase_index_proxy','remove_phase_features',
    'growth_conditioned_groups','local_context_groups','state_projection','residual_context')


class ContextMedoidPolicy:
    artifact_type = 'cast-das-context-preserving-medoids'

    def __init__(self, payload):
        if payload.get('artifact_type') != self.artifact_type or payload.get('version') != 1:
            raise ValueError('Unsupported context medoid artifact')
        if set(payload) != {'artifact_type','version','observation','tables','fallback','compilation'}:
            raise ValueError('Unexpected context medoid fields')
        self.metadata = dict(payload['observation'])
        if set(self.metadata) != set(OBSERVATION_FIELDS):
            raise ValueError('Incomplete context observation contract')
        validate_residual_context(self.metadata['residual_context'])
        validate_projection(self.metadata['state_projection'])
        self.tables = {k:CompiledPhasePolicy.from_dict(v) for k,v in payload['tables'].items()}
        self.fallback = CompiledPhasePolicy.from_dict(payload['fallback'])
        self.compilation = dict(payload['compilation'])
        if not self.tables:
            raise ValueError('Training contexts required')
        if any(t.refinement_actor or t.calibrated_group_actions for t in (*self.tables.values(),self.fallback)):
            raise ValueError('Medoid deployment must not contain actor/action overrides')

    def observation(self, state):
        m = self.metadata
        state = policy_observation(state,m['phase_awareness'],
            remove_phase_index_proxy=m['remove_phase_index_proxy'],remove_phase_features=m['remove_phase_features'])
        return project_table_state(state,preserve_growth=m['growth_conditioned_groups'],mode=m['state_projection']) if m['state_projection'] else state

    def context_key(self, state):
        m = self.metadata
        return joint_context_key(self.observation(state),growth_conditioned=m['growth_conditioned_groups'],
            local_context=m['local_context_groups'],state_projection=m['state_projection'],residual_context=m['residual_context'])

    def select(self, state):
        state = self.observation(state)
        return self.tables.get(self.context_key(state),self.fallback).select(state)

    def to_dict(self):
        return dict(artifact_type=self.artifact_type,version=1,observation=dict(self.metadata),
            tables={k:t.to_dict() for k,t in sorted(self.tables.items())},fallback=self.fallback.to_dict(),
            compilation=dict(self.compilation))
