"""Explicit observation contract shared by actor training and medoid serving."""
import threading
from .context import LockAction
from .paper_policy import CompiledPhasePolicy,_entry_key,_state_key
from .state_collector import policy_observation, PHASE_FEATURE_FIELDS


PAPER_KEY_WEIGHTS=(1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,2.,1.,2.,4.,2.,1.,1.,2.,
    .25,.25,.25,1.,.5,1.,2.,.5,2.)


def paper_hamming_distance(entry,state):
    left,right=_entry_key(entry),_state_key(state)
    assert len(left)==len(right)==len(PAPER_KEY_WEIGHTS)
    return sum(w for x,y,w in zip(left,right,PAPER_KEY_WEIGHTS) if x!=y)/sum(PAPER_KEY_WEIGHTS)


class PaperObservationPolicy:
    artifact_type='cast-das-paper-observed-medoid-policy'

    @property
    def requires_reviewer_contention_statistics(self):
        return getattr(self.base,'requires_reviewer_contention_statistics',True)

    def __init__(self, base, *, remove_phase_features=False,distance_mode='bucket-l1',
                 compaction_partition='action',exact_current_mask_lookup=False,
                 hamming_backend='scan',medoid_candidate_sampling='frequency',monotone_actions=None,
                 fitted_encoder=None,fitted_entry_keys=None,explicit_phase_input=False,
                 exact_phase_lookup=False):
        if type(exact_phase_lookup) is not bool:raise ValueError('Boolean phase lookup partition required')
        if exact_phase_lookup and (fitted_encoder is None or distance_mode!='weighted-hamming'):
            raise ValueError('Exact phase lookup requires fitted Hamming keys')
        self.exact_phase_lookup=exact_phase_lookup
        if type(explicit_phase_input) is not bool:
            raise ValueError('Boolean explicit phase-input declaration required')
        if explicit_phase_input and (getattr(base,'phase_awareness',None)!='enabled' or remove_phase_features):
            raise ValueError('Label-aware table must retain all inputs and enable phase')
        if not explicit_phase_input and getattr(base,'phase_awareness','disabled')!='disabled':
            raise ValueError('Paper inferred-phase policy cannot consume explicit phase labels')
        self.explicit_phase_input=explicit_phase_input
        self.base=base
        self.fitted_encoder=fitted_encoder
        self.key_weights=PAPER_KEY_WEIGHTS+((2.,) if fitted_encoder is not None else ())
        if fitted_encoder is not None:
            from .fitted_paper_encoder import KEY_FIELDS
            if distance_mode!='weighted-hamming' or not isinstance(base,CompiledPhasePolicy):
                raise ValueError('Fitted keys require a compiled Hamming policy')
            if fitted_entry_keys is None or len(fitted_entry_keys)!=len(base.entries):
                raise ValueError('One fitted key per medoid required')
            self.fitted_entry_keys=tuple(fitted_encoder.validate_key(k) for k in fitted_entry_keys)
            for entry,key in zip(base.entries,self.fitted_entry_keys):
                phases={'explore','refine','commit'} if explicit_phase_input else {'all'}
                if (key[0] not in phases or key[0]!=entry.phase
                        or key[KEY_FIELDS.index('current_action')]!=entry.current_action):
                    raise ValueError('Fitted entry observation/domain mismatch')
                if remove_phase_features and any(key[KEY_FIELDS.index(f)]!=0
                        for f in PHASE_FEATURE_FIELDS if f in KEY_FIELDS):
                    raise ValueError('NoPhase fitted keys contain removed information')
            self._fitted_keys_by_id={id(e):k for e,k in zip(base.entries,self.fitted_entry_keys)}
        elif fitted_entry_keys is not None:
            raise ValueError('Fitted keys require their frozen encoder')
        self.monotone_actions=(getattr(base,'monotone_actions',True) if monotone_actions is None
                               else bool(monotone_actions))
        if not self.monotone_actions and isinstance(base,CompiledPhasePolicy) and distance_mode!='weighted-hamming':
            raise ValueError('Revisable compiled masks require the versioned Hamming policy')
        self.remove_phase_features=bool(remove_phase_features)
        if distance_mode not in ('bucket-l1','weighted-hamming'):
            raise ValueError('Unsupported paper table metric')
        self.distance_mode=distance_mode
        if compaction_partition not in ('action','current-mask-and-action'):
            raise ValueError('Unsupported compaction partition')
        self.compaction_partition=compaction_partition
        if medoid_candidate_sampling not in ('frequency','coverage'):
            raise ValueError('Unsupported medoid candidate sampling')
        self.medoid_candidate_sampling=medoid_candidate_sampling
        self.exact_current_mask_lookup=bool(exact_current_mask_lookup)
        if self.exact_current_mask_lookup and (distance_mode!='weighted-hamming'
                or compaction_partition!='current-mask-and-action'):
            raise ValueError('Exact current-mask lookup requires declared Hamming partitions')
        self._lookup_context=threading.local()
        self._hamming_views={}
        if distance_mode=='weighted-hamming':
            self._hamming_views={current:tuple((entry,self.entry_key(entry)) for entry in base.entries
                if not self.monotone_actions or (entry.action | current)==entry.action) for current in range(16)}
            if self.exact_current_mask_lookup:
                # Current locks define the decision domain, not a noisy system
                # feature. Prefer its exact partition; unseen masks retain the
                # explicit monotone nearest-table fallback (never actor fallback).
                self._hamming_views={current:(tuple(row for row in rows
                    if row[0].current_action==current) or rows)
                    for current,rows in self._hamming_views.items()}
        if hamming_backend not in ('scan','bitset'):
            raise ValueError('Unsupported Hamming lookup backend')
        if hamming_backend=='bitset' and distance_mode!='weighted-hamming':
            raise ValueError('Bitset lookup requires weighted Hamming')
        self.hamming_backend=hamming_backend
        if hamming_backend=='bitset':self._build_hamming_bitset_index()

    def _build_hamming_bitset_index(self):
        # All current weights are exact binary fractions. Integer score planes
        # preserve the scanner's distance and support/action tie-break exactly.
        weights=tuple(int(w*4) for w in self.key_weights)
        if any(w*4!=v or v<=0 or v&(v-1) for w,v in zip(self.key_weights,weights)):
            raise ValueError('Bitset index requires positive power-of-two quarter weights')
        self._bitset_weights=weights
        self._bitset_rows=tuple(sorted(self.base.entries,key=lambda e:(-e.support_weight,e.action)))
        self._bitset_phases={}
        for index,entry in enumerate(self._bitset_rows):
            self._bitset_phases[entry.phase]=self._bitset_phases.get(entry.phase,0)|(1<<index)
        self._bitset_columns=[{} for _ in weights]
        for index,entry in enumerate(self._bitset_rows):
            bit=1<<index
            for column,value in zip(self._bitset_columns,self.entry_key(entry)):
                column[value]=column.get(value,0)|bit
        self._bitset_allowed={}
        for current in range(16):
            rows=[(i,e) for i,e in enumerate(self._bitset_rows)
                  if not self.monotone_actions or e.action|current==e.action]
            if self.exact_current_mask_lookup:
                rows=[(i,e) for i,e in rows if e.current_action==current] or rows
            self._bitset_allowed[current]=sum(1<<i for i,_ in rows)

    def _select_hamming_bitset(self,state):
        candidates=self._bitset_allowed[int(state.current_action)]
        if self.exact_phase_lookup:candidates&=self._bitset_phases.get(state.phase,0)
        if not candidates:
            self._lookup_context.distance=None
            return LockAction.from_encoded(state.current_action)
        total=sum(self._bitset_weights)
        planes=[0]*total.bit_length()
        for column,value,weight in zip(self._bitset_columns,self.state_key(state),self._bitset_weights):
            carry=column.get(value,0)
            bit=weight.bit_length()-1
            while carry:
                previous=planes[bit]
                planes[bit]=previous^carry
                carry=previous&carry
                bit+=1
        score=0
        for bit in range(len(planes)-1,-1,-1):
            positive=candidates&planes[bit]
            if positive:
                candidates=positive;score|=1<<bit
        index=(candidates&-candidates).bit_length()-1
        self._lookup_context.distance=(total-score)/total
        return LockAction.from_encoded(self._bitset_rows[index].action)

    def __getattr__(self, name):
        return getattr(self.base,name)

    def observation(self, state):
        return policy_observation(state,'enabled' if self.explicit_phase_input else 'disabled',
            remove_phase_features=self.remove_phase_features)

    def state_key(self,state):
        return self.fitted_encoder.key(state) if self.fitted_encoder is not None else _state_key(state)

    def entry_key(self,entry):
        return self._fitted_keys_by_id[id(entry)] if self.fitted_encoder is not None else _entry_key(entry)

    def select(self, state):
        # In the monotone four-bit domain mask 15 has exactly one legal action.
        # Quantization and nearest-medoid search cannot change the answer. Keep
        # the invocation observable but avoid useless serving work; revisable
        # masks must still run the normal selection path.
        if self.monotone_actions and int(state.current_action)==15:
            self._lookup_context.distance=None
            return LockAction.from_encoded(15)
        state=self.observation(state)
        return self._select_observed(state)

    def _select_observed(self, state):
        """Internal entry for an identically projected context-table query.

        The caller must apply this table's observation contract first. Keeping
        projection outside the search avoids copying a frozen state twice in
        ProgressMedoidPolicy, without caching answers or changing distances.
        """
        if self.monotone_actions and int(state.current_action)==15:
            self._lookup_context.distance=None
            return LockAction.from_encoded(15)
        if self.distance_mode=='weighted-hamming':
            if self.hamming_backend=='bitset':return self._select_hamming_bitset(state)
            allowed=self._hamming_views[int(state.current_action)]
            if self.exact_phase_lookup:allowed=tuple(row for row in allowed if row[0].phase==state.phase)
            if not allowed:
                self._lookup_context.distance=None
                return LockAction.from_encoded(state.current_action)
            key=self.state_key(state)
            def score(row):
                entry,encoded=row
                return (sum(w for x,y,w in zip(encoded,key,self.key_weights) if x!=y),
                    -entry.support_weight,entry.action)
            nearest=min(allowed,key=score)
            self._lookup_context.distance=score(nearest)[0]/sum(self.key_weights)
            return LockAction.from_encoded(nearest[0].action)
        if hasattr(self.base,'select'):
            return self.base.select(state)
        from agent.cc.atcc.ppo import valid_actions
        probabilities=self.base.probabilities(state)
        return LockAction.from_encoded(max(valid_actions(state.current_action,monotone=self.monotone_actions),
            key=lambda action:probabilities[action]))

    @property
    def last_lookup_distance(self):
        return getattr(self._lookup_context,'distance',None)

    def select_with_distribution(self, state):
        state=self.observation(state)
        sampler=getattr(self.base,'select_with_distribution',None)
        if sampler is not None:
            return sampler(state)
        action=self.select(state)
        return action,tuple(float(i==action.encoded) for i in range(16))

    def to_dict(self):
        if not isinstance(self.base,CompiledPhasePolicy):
            raise ValueError('Serving artifacts must contain a compiled table, not an actor')
        if self.base.refinement_actor or self.base.calibrated_group_actions:
            raise ValueError('Paper compiled artifact forbids inference/action overrides')
        if self.explicit_phase_input and self.fitted_encoder is None:
            raise ValueError('Version6 label-aware serving requires fitted full-input keys')
        return dict(artifact_type=self.artifact_type,version=(7 if self.exact_phase_lookup else 6 if self.explicit_phase_input else 5 if self.fitted_encoder is not None else 4 if not self.monotone_actions else
            (3 if self.exact_current_mask_lookup else (2 if self.distance_mode=='weighted-hamming' else 1))),
            explicit_phase_input=self.explicit_phase_input,remove_phase_features=self.remove_phase_features,
            removed_features=list(PHASE_FEATURE_FIELDS) if self.remove_phase_features else [],
            **(dict(distance_mode=self.distance_mode) if self.distance_mode=='weighted-hamming' else {}),
            **(dict(compaction_partition=self.compaction_partition)
               if self.compaction_partition!='action' else {}),
            **(dict(lookup_partition='exact-current-mask-preferred') if self.exact_current_mask_lookup else {}),
            **(dict(phase_lookup_partition='exact-phase') if self.exact_phase_lookup else {}),
            **(dict(hamming_backend='bitset') if self.hamming_backend=='bitset' else {}),
            **(dict(medoid_candidate_sampling='coverage') if self.medoid_candidate_sampling=='coverage' else {}),
            **(dict(monotone_actions=False) if not self.monotone_actions else {}),
            **(dict(fitted_encoder=self.fitted_encoder.to_dict(),fitted_entry_keys=self.fitted_entry_keys)
               if self.fitted_encoder is not None else {}),
            model=self.base.to_dict())

    @classmethod
    def from_dict(cls, data):
        if data.get('artifact_type')!=cls.artifact_type or data.get('version') not in (1,2,3,4,5,6,7):
            raise ValueError('Unsupported paper observation artifact')
        if type(data.get('monotone_actions',True)) is not bool:
            raise ValueError('Boolean action domain required')
        if data.get('lookup_partition') not in (None,'exact-current-mask-preferred'):
            raise ValueError('Unsupported lookup partition')
        if data.get('version') not in (4,5,6,7) and (data.get('version')==3)!=(data.get('lookup_partition')=='exact-current-mask-preferred'):
            raise ValueError('Lookup partition requires an explicit version3 artifact')
        if data.get('version') not in (5,6,7) and (data.get('version')==4)!=(data.get('monotone_actions',True) is False):
            raise ValueError('Revisable masks require explicit version4 domain')
        from .fitted_paper_encoder import FittedPaperEncoder
        fitted=(FittedPaperEncoder.from_dict(data['fitted_encoder']) if data.get('version') in (5,6,7) else None)
        if data.get('version') not in (5,6,7) and ('fitted_encoder' in data or 'fitted_entry_keys' in data):
            raise ValueError('Fitted schema requires explicit version5 artifact')
        exact_phase=data.get('version')==7
        if exact_phase!=(data.get('phase_lookup_partition')=='exact-phase') or data.get('phase_lookup_partition') not in (None,'exact-phase'):
            raise ValueError('Phase partition requires an explicit version7 artifact')
        explicit=data.get('explicit_phase_input') if exact_phase else data.get('version')==6
        if type(explicit) is not bool:raise ValueError('Boolean explicit phase input required')
        if data.get('explicit_phase_input') is not explicit or type(data.get('remove_phase_features')) is not bool:
            raise ValueError('Explicit inferred-phase metadata required')
        removed=list(PHASE_FEATURE_FIELDS) if data['remove_phase_features'] else []
        if data.get('removed_features')!=removed:
            raise ValueError('NoPhase feature mask differs from its declaration')
        result=cls(CompiledPhasePolicy.from_dict(data['model']),
            remove_phase_features=data['remove_phase_features'],
            distance_mode=data.get('distance_mode','bucket-l1'),
            compaction_partition=data.get('compaction_partition','action'),
            exact_current_mask_lookup=data.get('lookup_partition')=='exact-current-mask-preferred',
            hamming_backend=data.get('hamming_backend','scan'),
            medoid_candidate_sampling=data.get('medoid_candidate_sampling','frequency'),
            monotone_actions=data.get('monotone_actions',True),fitted_encoder=fitted,
            fitted_entry_keys=data.get('fitted_entry_keys'),explicit_phase_input=explicit,
            exact_phase_lookup=exact_phase)
        if data['version'] in (2,3,4,5,6,7) and result.distance_mode!='weighted-hamming':
            raise ValueError('Version2 requires declared Hamming metric')
        result.to_dict()
        return result


def compile_paper_medoids(actor, *, remove_phase_features=False,generation=0,budget=4,
                          partition_current_mask=False,candidate_sampling='frequency'):
    """Default: k=4 per action. Opt-in repair partitions by current mask as well.

    The opt-in mode uses more space and must be reported as a different
    compaction budget/partition, never silently substituted for the paper k=4.
    It preserves current-lock context without adding actor inference/fallback.
    """
    from collections import defaultdict
    from agent.cc.atcc.ppo import compiled_entry,valid_actions,weighted_k_medoids
    observed=PaperObservationPolicy(actor,remove_phase_features=remove_phase_features)
    groups=defaultdict(list)
    # probabilities() can register an already-normalized prototype. Freeze
    # iteration keys so NoPhase normalization cannot resize this traversal.
    for key,state in sorted(list(actor.prototypes.items())):
        state=observed.observation(state)
        probabilities=actor.probabilities(state)
        action=max(valid_actions(state.current_action,monotone=actor.monotone_actions),key=lambda a:probabilities[a])
        group=(int(state.current_action),action) if partition_current_mask else action
        groups[group].append((compiled_entry(state,action),state,max(1,actor.prototype_counts.get(key,0))))
    entries=[entry for _,rows in sorted(groups.items())
        for entry in weighted_k_medoids(rows,budget,distance=paper_hamming_distance,
            candidate_sampling=candidate_sampling)]
    table=CompiledPhasePolicy(entries,generation=generation,medoids_per_group=budget,phase_awareness='disabled')
    return PaperObservationPolicy(table,remove_phase_features=remove_phase_features,distance_mode='weighted-hamming',
        compaction_partition='current-mask-and-action' if partition_current_mask else 'action',
        exact_current_mask_lookup=partition_current_mask,
        medoid_candidate_sampling=candidate_sampling,monotone_actions=actor.monotone_actions)


def compile_fitted_paper_medoids(actor,training_states,*,remove_phase_features=False,
                                 generation=0,budget=4,bins=8,hamming_backend='bitset',
                                 partition_current_mask=False,candidate_sampling='frequency',
                                exclude_forced_states=False,explicit_phase_input=False,
                                exact_phase_lookup=False):
    """Compile actual training questions, never a validation/test-state closure.

    Frozen actor labels are preserved as labels, not optimized on a test score.
    Quantization can alias differing labels; fidelity must be measured separately.
    The actor still consumes its original continuous features. This is not a
    claim of quantization-aware training or lossless policy distillation.

    Optional decision-only compaction still fits its encoder on ALL training
    questions, but spends medoid support/budget only on questions having more
    than one legal action. The experiment manifest must disclose that support
    weights then count nontrivial queries, not the whole encoder population.
    Forced states are answered by the exact action-domain fastpath, not an actor.
    """
    import dataclasses
    from collections import Counter,defaultdict
    from .fitted_paper_encoder import FittedPaperEncoder
    from agent.cc.atcc.ppo import compiled_entry,valid_actions,weighted_k_medoids

    @dataclasses.dataclass(frozen=True)
    class Candidate:
        key: tuple
        state: object
        action: int
        support_weight: int=1

    if type(budget) is not int or budget<1:raise ValueError('Positive integer medoid budget required')
    if candidate_sampling not in ('frequency','coverage'):
        raise ValueError('Unsupported medoid candidate sampling')
    if type(exclude_forced_states) is not bool:
        raise ValueError('Boolean forced-state compaction option required')
    observed=PaperObservationPolicy(actor,remove_phase_features=remove_phase_features,
        explicit_phase_input=explicit_phase_input)
    states=[observed.observation(s) for s in training_states]
    encoder=FittedPaperEncoder.fit(states,bins=bins)
    keys={id(s):encoder.key(s) for s in states}
    key_fn=lambda s:keys[id(s)]
    representatives={};counts=Counter()
    for state in states:
        if exclude_forced_states and len(valid_actions(state.current_action,
                monotone=actor.monotone_actions))==1:
            continue
        probabilities=actor.probabilities(state)
        action=max(valid_actions(state.current_action,monotone=actor.monotone_actions),
                   key=lambda a:probabilities[a])
        group=(key_fn(state),action)
        representatives.setdefault(group,state);counts[group]+=1
    if exclude_forced_states and not counts:
        raise ValueError('Decision-only compaction requires nontrivial training queries')
    groups=defaultdict(list)
    for (key,action),state in sorted(representatives.items()):
        group=(int(state.current_action),action) if partition_current_mask else action
        groups[group].append((Candidate(key,state,action),state,counts[(key,action)]))
    weights=PAPER_KEY_WEIGHTS+(2.,);total=sum(weights)
    def distance(entry,state):
        return sum(w for a,b,w in zip(entry.key,key_fn(state),weights) if a!=b)/total
    chosen=[entry for _,rows in sorted(groups.items()) for entry in weighted_k_medoids(
        rows,budget,distance=distance,key_fn=key_fn,candidate_sampling=candidate_sampling)]
    base=CompiledPhasePolicy([dataclasses.replace(compiled_entry(e.state,e.action),
        support_weight=e.support_weight) for e in chosen],generation=generation,
        medoids_per_group=budget,phase_awareness='enabled' if explicit_phase_input else 'disabled')
    return PaperObservationPolicy(base,remove_phase_features=remove_phase_features,
        distance_mode='weighted-hamming',monotone_actions=actor.monotone_actions,
        compaction_partition='current-mask-and-action' if partition_current_mask else 'action',
        exact_current_mask_lookup=partition_current_mask,
        hamming_backend=hamming_backend,medoid_candidate_sampling=candidate_sampling,fitted_encoder=encoder,
        fitted_entry_keys=[e.key for e in chosen],explicit_phase_input=explicit_phase_input,
        exact_phase_lookup=exact_phase_lookup)
