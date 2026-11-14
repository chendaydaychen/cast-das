"""Experimental context-partitioned medoids; serving contains no actor weights.

Unlike the original flat k-per-action table, the budget is per learned residual
context AND action. Unseen contexts use a training-only global-component table.
This is an explicitly different deployment representation, not lossless PPO.
"""
from types import SimpleNamespace
import threading
from .paper_policy import bucket_size, bucket_ratio, bucket_log
from .paper_observation_policy import PaperObservationPolicy, compile_fitted_paper_medoids


def observed_progress_key(state,mode='observed-progress'):
    conflict=str(state.recent_conflict_kind).strip().lower() or 'none'
    if mode=='observed-risk':
        return '|'.join(map(str,(mode,state.phase,int(state.current_action),min(3,state.retry_count),conflict)))
    if mode=='observed-progress-compact':
        return '|'.join(map(str,(mode,state.phase,int(state.current_action),min(3,state.retry_count),
            conflict,bucket_size(state.completed_rounds),bucket_size(state.completed_operations))))
    if mode!='observed-progress':raise ValueError('Unsupported residual routing')
    return '|'.join(map(str,('observed-progress',state.phase,int(state.current_action),
        min(3,state.retry_count),conflict,bucket_size(state.read_set_growth),
        bucket_size(state.write_set_growth),bucket_ratio(state.access_overlap_ratio),
        bucket_size(state.completed_rounds),bucket_size(state.completed_operations),
        bucket_ratio(state.recent_write_ratio),bucket_log(state.inter_round_interval_ms))))


class ProgressMedoidPolicy:
    artifact_type='cast-das-observed-progress-partitioned-medoids'
    phase_awareness='disabled'
    requires_reviewer_contention_statistics=False

    def __init__(self,payload):
        expected={'artifact_type','version','remove_phase_features','tables','fallback','compilation'}
        if payload.get('version') in (2,3):expected.add('group_mode')
        if payload.get('version')==3:expected.update(('phase_label_only','phase_awareness'))
        if set(payload)!=expected:
            raise ValueError('Unexpected progress medoid fields')
        if payload['artifact_type']!=self.artifact_type or payload['version'] not in (1,2,3):
            raise ValueError('Unsupported progress medoid artifact')
        if type(payload['remove_phase_features']) is not bool:raise ValueError('Boolean mask required')
        self.remove_phase_features=payload['remove_phase_features']
        self.phase_label_only=payload['version']==3
        self.phase_awareness=payload.get('phase_awareness','disabled')
        if self.phase_label_only:
            if (payload['phase_label_only'] is not True or self.remove_phase_features
                    or self.phase_awareness not in ('enabled','disabled')):
                raise ValueError('Label-only context table must retain every derived feature')
        self.group_mode=payload.get('group_mode','observed-progress')
        if self.group_mode not in ('observed-progress','observed-progress-compact','observed-risk'):
            raise ValueError('Unsupported residual routing')
        self.tables={k:PaperObservationPolicy.from_dict(v) for k,v in payload['tables'].items()}
        self.fallback=PaperObservationPolicy.from_dict(payload['fallback'])
        self.monotone_actions=self.fallback.monotone_actions
        self._lookup_context=threading.local()
        self.compilation=dict(payload['compilation'])
        if bool(self.compilation.get('exact_phase_lookup',False))!=self.fallback.exact_phase_lookup:
            raise ValueError('Phase lookup partition differs from its declaration')
        for table in [self.fallback,*self.tables.values()]:
            if table.monotone_actions!=self.monotone_actions:
                raise ValueError('Inconsistent action domains')
            if table.remove_phase_features!=self.remove_phase_features:
                raise ValueError('Inconsistent feature removal')
            if table.generation!=self.fallback.generation:
                raise ValueError('Inconsistent deployment generations')
            if table.phase_awareness!=self.phase_awareness:
                raise ValueError('Inconsistent explicit phase-input contracts')
            if table.exact_phase_lookup!=self.fallback.exact_phase_lookup:
                raise ValueError('Inconsistent phase lookup partitions')
            if table.base.refinement_actor or table.base.calibrated_group_actions:
                raise ValueError('Actor/override forbidden in deployment')
        self.base=SimpleNamespace(entries=[e for t in [self.fallback,*self.tables.values()] for e in t.base.entries])

    @property
    def generation(self):
        return self.fallback.generation

    @property
    def last_lookup_distance(self):
        return getattr(self._lookup_context,'distance',None)

    def select(self,state):
        observed=self.fallback.observation(state)
        table=self.tables.get(observed_progress_key(observed,self.group_mode),self.fallback)
        # __init__ checks identical observation contracts across all children.
        # Route and search the SAME projected state; never expose removed data.
        action=table._select_observed(observed)
        self._lookup_context.distance=table.last_lookup_distance
        return action

    def to_dict(self):
        return dict(artifact_type=self.artifact_type,version=3 if self.phase_label_only else 2,
            **(dict(phase_label_only=True,phase_awareness=self.phase_awareness) if self.phase_label_only else {}),
            group_mode=self.group_mode,remove_phase_features=self.remove_phase_features,
            tables={k:v.to_dict() for k,v in sorted(self.tables.items())},
            fallback=self.fallback.to_dict(),compilation=self.compilation)


def compile_progress_medoids(actor,states,*,remove_phase_features=False,budget=4,generation=0,
                            phase_label_only=False,candidate_sampling='frequency',exact_phase_lookup=False):
    from collections import defaultdict
    from agent.cc.atcc.actor_artifact import actor_snapshot,actor_from_snapshot
    from agent.cc.atcc.ppo import valid_actions
    if type(generation) is not int or generation<0:
        raise ValueError('Nonnegative integer generation required')
    if type(phase_label_only) is not bool or (phase_label_only and remove_phase_features):
        raise ValueError('Label-only compilation cannot remove derived features')
    if candidate_sampling not in ('frequency','coverage'):
        raise ValueError('Unsupported medoid candidate sampling')
    if actor.group_mode not in ('observed-progress','observed-progress-compact','observed-risk') or (actor.phase_awareness!='disabled' and not phase_label_only) or actor.categorical_actor:
        raise ValueError('Only explicit inferred-phase observed-context bit actors supported')
    states=list(states)
    if not states:raise ValueError('Training questions required')
    # Factoring a support-restricted policy would change its semantics. Refuse.
    for key,actions in actor.observed_group_actions.items():
        current=int(key.split('|')[2])
        if set(actions)!=set(valid_actions(current,monotone=actor.monotone_actions)):
            raise ValueError('Global-component factorization requires full legal action support')
    snapshot=actor_snapshot(actor)
    snapshot['actor_group_logits']={}
    snapshot['observed_group_actions']={}
    global_actor=actor_from_snapshot(snapshot);global_actor.record_prototypes=False
    explicit=phase_label_only and actor.phase_awareness=='enabled'
    observed=PaperObservationPolicy(actor,remove_phase_features=remove_phase_features,
        explicit_phase_input=explicit)
    original_question_count=len(states)
    if phase_label_only:
        import dataclasses
        # Model-labelled finite mask variants, not added behavior/reward samples.
        states=[dataclasses.replace(s,current_action=mask) for s in states for mask in range(16)]
    groups=defaultdict(list)
    for state in states:
        s=observed.observation(state);key=observed_progress_key(s,actor.group_mode)
        assert actor.group_key(s)==key
        if any(actor.actor_group_logits.get(key,())):groups[key].append(state)
    def compile_one(model,population,*,partition_current_mask=False):
        return compile_fitted_paper_medoids(model,population,remove_phase_features=remove_phase_features,
            budget=budget,generation=generation,bins=8,hamming_backend='bitset',
            partition_current_mask=partition_current_mask,explicit_phase_input=explicit,
            candidate_sampling=candidate_sampling,exact_phase_lookup=exact_phase_lookup).to_dict()
    return ProgressMedoidPolicy(dict(artifact_type=ProgressMedoidPolicy.artifact_type,
        version=3 if phase_label_only else 2,
        **(dict(phase_label_only=True,phase_awareness=actor.phase_awareness) if phase_label_only else {}),
        group_mode=actor.group_mode,
        remove_phase_features=remove_phase_features,
        # Current protection is an action-domain boundary, not a noisy metric
        # feature: an entry's already-held bits are not a learned recommendation
        # to acquire them in a query that starts without those locks.
        fallback=compile_one(global_actor,states,partition_current_mask=True),
        tables={k:compile_one(actor,v) for k,v in sorted(groups.items())},
        compilation=dict(training_questions=original_question_count,
            **(dict(candidate_sampling=candidate_sampling) if candidate_sampling!='frequency' else {}),
            **(dict(exact_phase_lookup=True) if exact_phase_lookup else {}),
            **(dict(model_labelled_mask_variants=len(states)) if phase_label_only else {}),
            context_questions=sum(map(len,groups.values())),
            learned_contexts=len(groups),budget_per_context_and_action=budget,
            global_budget_scope='per current mask AND action',
            global_lookup_partition='exact-current-mask-preferred',
            fallback='training-only zero-residual global actor component',
            test_states_used=False,actor_inference_at_runtime=False)))
