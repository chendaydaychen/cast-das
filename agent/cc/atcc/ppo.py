"""Dependency-free discrete PPO for paper ATCC trajectories."""

from __future__ import annotations

import dataclasses
import math
import random
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from agent.runtime.context import LockClass
from agent.runtime.paper_policy import (
    CompiledPhasePolicy,
    CompiledPolicyEntry,
    bucket_log,
    bucket_ratio,
    bucket_size,
)
from agent.runtime.state_collector import PhaseAwareState, policy_observation
from agent.runtime.trajectory import PolicyTransition


ALL_ACTIONS = tuple(range(16))
TIMED_ACTIONS = tuple(range(32))
LOCK_BITS = (1, 2, 4, 8)
PAPER_MEDOIDS_PER_GROUP = 4


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 0.003
    group_learning_rate: float = 0.03
    critic_learning_rate: float = 0.005
    clip_ratio: float = 0.20
    discount: float = 0.99
    entropy_weight: float = 0.01
    epochs: int = 8
    min_group_samples: int = 16
    min_group_actions: int = 2
    calibration_return_mode: str = "per-source-run"
    # Opt-in shared scale for multiple rollouts of one frozen behavior actor.
    # The default preserves legacy experiments and their numerical replay.
    actor_return_mode: str = "per-source-run"
    advantage_baseline: str = "linear-critic"
    categorical_initialization: str = "behavior"
    ppo_reference_mode: str = "behavior"
    # Opt-in corrected score/entropy derivatives for a four-bit actor whose
    # joint mask probabilities are conditioned on observed complete actions.
    conditioned_mask_gradient: bool = False
    # Opt-in: positive probability in an exact logged behavior distribution
    # establishes legal importance-sampling support, even if a small local
    # cohort did not happen to sample every action. Qualification/sample counts
    # below still use actual sampled actions, never the expanded support set.
    logged_action_support: bool = False
    initialize_from_behavior: bool = True


class DiscretePPOPolicy:
    requires_reviewer_contention_statistics=False
    def __init__(self, *, seed: int = 0, stay_probability: float = 0.0,
                 phase_awareness: str = "enabled", growth_conditioned_groups: bool = False,
                 strict_local_action_support: bool = False, pooled_sparse_action_support: bool = False,
                 ppo_sparse_fallback: bool = False, ppo_supported_projection: bool = False,
                 categorical_actor: bool = False, local_qualified_action_support: bool = False,
                 local_context_groups: bool = False, protection_relevant_hotness: bool = False,
                 timing_actions: bool = False,monotone_actions: bool = True,group_mode: str = 'legacy',
                 initial_bit_probability: float = .5):
        if phase_awareness not in {"enabled", "disabled"}:
            raise ValueError("invalid phase awareness")
        self.phase_awareness = phase_awareness
        if group_mode not in ('legacy','observed-progress','observed-progress-compact','observed-risk'):
            raise ValueError('Unsupported actor residual grouping')
        self.group_mode=group_mode
        self.monotone_actions=bool(monotone_actions)
        if not self.monotone_actions and timing_actions:
            raise ValueError('Revisable masks do not support experimental timing actions')
        self.growth_conditioned_groups = bool(growth_conditioned_groups)
        self.strict_local_action_support = bool(strict_local_action_support)
        self.pooled_sparse_action_support = bool(pooled_sparse_action_support)
        self.ppo_sparse_fallback = bool(ppo_sparse_fallback)
        self.ppo_supported_projection = bool(ppo_supported_projection)
        self.categorical_actor = bool(categorical_actor)
        initial=float(initial_bit_probability)
        if not math.isfinite(initial) or not 0.<initial<1.:
            raise ValueError('Initial bit probability must be finite and strictly between zero and one')
        if self.categorical_actor and initial!=.5:
            raise ValueError('Bit initialization is not a categorical-actor prior')
        self.initial_bit_probability=initial
        self.local_qualified_action_support = bool(local_qualified_action_support)
        self.local_context_groups = bool(local_context_groups)
        self.protection_relevant_hotness = bool(protection_relevant_hotness)
        # Opt-in 32-way categorical space: masks 0..15 acquire immediately;
        # masks 16..31 nominate identical protection for commit materialization.
        # Legacy policies retain the original 16 immediate actions unchanged.
        self.timing_actions = bool(timing_actions)
        if self.timing_actions and not categorical_actor:
            raise ValueError("timed actions require the categorical actor")
        self.all_actions = TIMED_ACTIONS if self.timing_actions else ALL_ACTIONS
        if self.local_context_groups and self.pooled_sparse_action_support:
            raise ValueError("local-context groups already pool volatile buckets; sparse pooling is redundant")
        if self.local_qualified_action_support and not (self.ppo_supported_projection and self.categorical_actor):
            raise ValueError("local qualified support requires categorical supported PPO projection")
        self.categorical_actor_weights = ([[0.0] * FEATURE_COUNT for _ in self.all_actions]
                                          if self.categorical_actor else None)
        self.categorical_group_logits = (defaultdict(lambda: [0.0] * len(self.all_actions))
                                         if self.categorical_actor else None)
        self._sparse_actor_allowed: Dict[str, set[int]] = {}
        self._calibration_action_support: Dict[str, Dict[int, int]] = {}
        self._calibration_min_action_support = 0
        if self.ppo_sparse_fallback and (self.strict_local_action_support or self.pooled_sparse_action_support):
            raise ValueError("PPO sparse fallback is exclusive with other sparse projection modes")
        if self.ppo_supported_projection and (self.ppo_sparse_fallback or self.strict_local_action_support or self.pooled_sparse_action_support):
            raise ValueError("Supported PPO projection is exclusive with other projection modes")
        if self.strict_local_action_support and self.pooled_sparse_action_support:
            raise ValueError("strict-local and pooled-sparse support are distinct, exclusive projection modes")
        self.prototypes: Dict[str, PhaseAwareState] = {}
        self.record_prototypes = True
        self.prototype_counts: Dict[str, int] = defaultdict(int)
        self.actor_weights = [[0.0] * FEATURE_COUNT for _ in LOCK_BITS]
        # A trainable initialization, never an action override or a future-state
        # oracle. Every legal mask retains positive probability. Defaults are
        # byte-for-byte compatible with the old zero-logit initialization.
        initial_logit=math.log(initial/(1.-initial))
        for weights in self.actor_weights:weights[0]=initial_logit
        self.actor_group_logits: Dict[str, List[float]] = defaultdict(
            lambda: [0.0] * len(LOCK_BITS)
        )
        self.observed_group_actions: Dict[str, set[int]] = defaultdict(set)
        # PPO learns shared bit logits, but deployment must choose a complete
        # lock-mask action that has direct behavior support.  The calibrated
        # head prevents independent bits from synthesizing an untested mask.
        self.calibrated_group_actions: Dict[str, int] = {}
        self.critic_weights = [0.0] * FEATURE_COUNT
        self.rng = random.Random(int(seed))
        self.stay_probability = max(0.0, min(1.0, float(stay_probability)))
        self._lock = threading.RLock()

    def group_key(self, state: PhaseAwareState) -> str:
        state = policy_observation(state, self.phase_awareness)
        if self.group_mode=='observed-risk':
            # Pool residual biases over progress states. All continuous actor
            # features (including phase-related features in Full) remain intact.
            return '|'.join(map(str,('observed-risk',state.phase,int(state.current_action),
                min(3,state.retry_count),normalize_conflict_kind(state.recent_conflict_kind))))
        if self.group_mode=='observed-progress-compact':
            # Keep all continuous actor features. Only pool sparse residual
            # updates across volatile growth/composition/interval sub-buckets.
            return '|'.join(map(str,('observed-progress-compact',state.phase,int(state.current_action),
                min(3,state.retry_count),normalize_conflict_kind(state.recent_conflict_kind),
                bucket_size(state.completed_rounds),bucket_size(state.completed_operations))))
        if self.group_mode=='observed-progress':
            # Share residual learning across changing global traffic counters.
            # All original global features remain in the linear actor. This
            # only reduces fragmentation of the nonlinear state-group bias.
            return '|'.join(map(str,('observed-progress',state.phase,int(state.current_action),
                min(3,state.retry_count),normalize_conflict_kind(state.recent_conflict_kind),
                bucket_size(state.read_set_growth),bucket_size(state.write_set_growth),
                bucket_ratio(state.access_overlap_ratio),bucket_size(state.completed_rounds),
                bucket_size(state.completed_operations),bucket_ratio(state.recent_write_ratio),
                bucket_log(state.inter_round_interval_ms))))
        return policy_group_key(state, growth_conditioned=self.growth_conditioned_groups,
                                local_context=self.local_context_groups)

    def prototype_key(self, state: PhaseAwareState) -> str:
        key = state_key(state)
        # The legacy prototype buckets merge growth=4 and growth=5. Preserve
        # both observed prototypes when the deployment head distinguishes them.
        return key + "|" + self.group_key(state) if self.growth_conditioned_groups else key

    def probabilities(self, state: PhaseAwareState) -> list[float]:
        state = policy_observation(state, self.phase_awareness)
        with self._lock:
            if self.record_prototypes:
                key = self.prototype_key(state)
                self.prototypes.setdefault(key, state)
            features = state_features(state)
            if self.categorical_actor:
                valid = valid_actions(state.current_action, timing_actions=self.timing_actions,monotone=self.monotone_actions)
                residual = (self.categorical_group_logits[self.group_key(state)] if self.record_prototypes else
                            self.categorical_group_logits.get(self.group_key(state),(0.,)*len(self.all_actions)))
                logits = {a: dot(self.categorical_actor_weights[a], features) + residual[a] for a in valid}
                largest = max(logits.values())
                mass = {a: math.exp(value - largest) for a, value in logits.items()}
                total = sum(mass.values())
                probabilities = [mass.get(a, 0.0) / total for a in self.all_actions]
                if self.stay_probability > 0.0:
                    probabilities = [(1.0 - self.stay_probability) * p for p in probabilities]
                    probabilities[int(state.current_action) & 0xF] += self.stay_probability
                # Train in the logged FULL valid space. Deployment still
                # projects onto original qualified/observed/trusted masks.
                return probabilities
            group_logits = (self.actor_group_logits[self.group_key(state)] if self.record_prototypes else
                            self.actor_group_logits.get(self.group_key(state),(0.,)*len(LOCK_BITS)))
            valid = valid_actions(state.current_action, timing_actions=self.timing_actions,monotone=self.monotone_actions)
            bit_probabilities = []
            for bit_index, bit in enumerate(LOCK_BITS):
                if self.monotone_actions and int(state.current_action) & bit:
                    bit_probabilities.append(1.0)
                    continue
                logit = dot(self.actor_weights[bit_index], features) + group_logits[bit_index]
                bit_probabilities.append(sigmoid(logit))
            probabilities = [0.0] * len(self.all_actions)
            for action in valid:
                probability = 1.0
                for bit_index, bit in enumerate(LOCK_BITS):
                    if self.monotone_actions and int(state.current_action) & bit:
                        continue
                    selected = bool(action & bit)
                    bit_probability = bit_probabilities[bit_index]
                    probability *= bit_probability if selected else 1.0 - bit_probability
                probabilities[action] = probability
            supported = set(self.observed_group_actions.get(self.group_key(state), ()))
            if supported:
                supported.add(int(state.current_action) & 0xF)
                allowed = set(valid) & supported
                probabilities = [
                    probability if action in allowed else 0.0
                    for action, probability in enumerate(probabilities)
                ]
                total = sum(probabilities)
                if total > 0.0:
                    probabilities = [probability / total for probability in probabilities]
                else:
                    probabilities = [
                        1.0 / len(allowed) if action in allowed else 0.0
                        for action in self.all_actions
                    ]
            if self.stay_probability > 0.0:
                retain = self.stay_probability
                probabilities = [
                    (1.0 - retain) * probability
                    for probability in probabilities
                ]
                probabilities[int(state.current_action) & 0xF] += retain
            return probabilities

    def value(self, state: PhaseAwareState) -> float:
        state = policy_observation(state, self.phase_awareness)
        with self._lock:
            return dot(self.critic_weights, state_features(state))

    def sample(self, state: PhaseAwareState) -> tuple[int, float]:
        with self._lock:
            probabilities = self.probabilities(state)
            action = sample_action(self.rng, probabilities, state.current_action)
            return action, probabilities[action]

    def select_with_probability(self, state: PhaseAwareState) -> tuple[object, float]:
        from agent.runtime.context import LockAction

        action, probability = self.sample(state)
        return LockAction.from_encoded(action), probability

    def select_with_distribution(
        self,
        state: PhaseAwareState,
    ) -> tuple[object, tuple[float, ...]]:
        from agent.runtime.context import LockAction

        with self._lock:
            probabilities = tuple(self.probabilities(state))
            action = sample_action(self.rng, probabilities, state.current_action)
            return LockAction.from_encoded(action), probabilities

    def compile(
        self,
        *,
        generation: int,
        medoids_per_group: int = PAPER_MEDOIDS_PER_GROUP,
        refinement_distance_threshold: float | None = None,
        occ_cold_start_guard: bool = False,
        preserve_calibrated_groups: bool = False,
        statewise_supported_actor: bool = False,
        use_calibrated_actions: bool = True,
    ) -> CompiledPhasePolicy:
        if not use_calibrated_actions and (preserve_calibrated_groups or statewise_supported_actor):
            raise ValueError('actor-argmax compilation cannot preserve calibrated overrides')
        if statewise_supported_actor and not (self.categorical_actor and self.ppo_supported_projection
                and self.local_qualified_action_support and self.local_context_groups and preserve_calibrated_groups):
            raise ValueError("statewise supported actor requires categorical locally qualified preserved heads")
        if self.local_context_groups and not preserve_calibrated_groups:
            raise ValueError("local-context groups require preserved deployment heads")
        if self.categorical_actor and refinement_distance_threshold is not None:
            raise ValueError("categorical actor requires compiled supported heads, not bit refinement")
        if not self.monotone_actions or self.group_mode!='legacy':
            raise ValueError('Use the versioned paper medoid compiler for revisable masks or nonlegacy groups')
        grouped: Dict[tuple[str, int, int], list[tuple[CompiledPolicyEntry, PhaseAwareState, int]]] = defaultdict(list)
        for key, state in sorted(self.prototypes.items()):
            probabilities = self.probabilities(state)
            valid = valid_actions(state.current_action, timing_actions=self.timing_actions)
            calibrated = (self.calibrated_group_actions.get(self.group_key(state))
                          if use_calibrated_actions else None)
            action = (
                int(calibrated)
                if calibrated is not None and int(calibrated) in valid
                else max(valid, key=lambda candidate: probabilities[candidate])
            )
            entry = compiled_entry(state, action)
            grouped[(state.phase, int(state.current_action), action)].append(
                (entry, state, max(1, int(self.prototype_counts.get(key, 0))))
            )
        medoids_per_group = max(1, int(medoids_per_group))
        entries = [
            entry
            for _group, rows in sorted(grouped.items())
            for entry in weighted_k_medoids(rows, medoids_per_group)
        ]
        refinement_actor = {}
        if statewise_supported_actor:
            minimum = self._calibration_min_action_support
            supported = {key: sorted(a for a, count in actions.items()
                                    if count >= minimum and a in self._sparse_actor_allowed.get(key, ()))
                         for key, actions in self._calibration_action_support.items()}
            supported = {key: actions for key, actions in supported.items() if len(actions) >= 2}
            if minimum < 2 or not supported:
                raise ValueError("statewise actor has no locally qualified multi-action groups")
            refinement_actor = dict(type="locally-supported-categorical-lock-masks",
                feature_count=FEATURE_COUNT, phase_awareness=self.phase_awareness,
                actor_weights=[list(row) for row in self.categorical_actor_weights],
                group_logits={key: list(self.categorical_group_logits[key]) for key in sorted(supported)},
                group_action_support=supported, minimum_action_support=minimum,
                group_action_support_counts={key: {str(a): self._calibration_action_support[key][a]
                                                   for a in actions} for key, actions in supported.items()},
                tie_break="existing_calibrated_head_then_lowest_mask",
                state_source="observed_execution_only", workload_labels=False)
        if refinement_distance_threshold is not None:
            refinement_actor = {
                "type": "factorized-bernoulli-lock-bits",
                "preserve_actor_distribution": True,
                "stay_probability": self.stay_probability,
                "distance_threshold": max(0.0, float(refinement_distance_threshold)),
                "actor_weights": [list(row) for row in self.actor_weights],
                "group_logits": {
                    key: list(values)
                    for key, values in sorted(self.actor_group_logits.items())
                },
                "group_action_support": {
                    key: sorted(actions)
                    for key, actions in sorted(self.observed_group_actions.items())
                },
                "state_source": "observed_execution_only",
                "workload_labels": False,
            }
        return CompiledPhasePolicy(
            entries,
            generation=generation,
            medoids_per_group=medoids_per_group,
            refinement_actor=refinement_actor,
            occ_cold_start_guard=occ_cold_start_guard,
            phase_awareness=self.phase_awareness,
            calibrated_group_actions=(dict(self.calibrated_group_actions) if preserve_calibrated_groups else None),
            growth_conditioned_groups=self.growth_conditioned_groups,
            local_context_groups=self.local_context_groups,
            protection_relevant_hotness=self.protection_relevant_hotness,
            timing_actions=self.timing_actions,
        )


class EpsilonGreedyPolicy:
    """Explore around a deployed policy with an exact categorical propensity."""

    def __init__(self, base_policy: object, *, seed: int, epsilon: float = 0.2,
                 exploration_masks: tuple[int, ...] | None = None):
        self.base_policy = base_policy
        self.rng = random.Random(int(seed))
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self._lock = threading.RLock()
        if exploration_masks is not None and (
            not exploration_masks or any(type(mask) is not int or not 0 <= mask <= 15
                                         for mask in exploration_masks)
        ):
            raise ValueError("exploration masks must be nonempty four-bit integers")
        self.exploration_masks = exploration_masks

    def probabilities(self, state: PhaseAwareState) -> tuple[float, ...]:
        timing_actions = bool(getattr(self.base_policy, "timing_actions", False))
        valid = valid_actions(state.current_action, timing_actions=timing_actions)
        chosen = self.base_policy.select(state)
        base_action = int(getattr(chosen, "encoded", int(chosen.protected)))
        if base_action not in valid:
            base_action = int(state.current_action) & 0xF
        proposed = (sorted({int(state.current_action) | mask for mask in self.exploration_masks})
                    if self.exploration_masks is not None else valid)
        exploration = self.epsilon / len(proposed)
        probabilities = [0.0] * (32 if timing_actions else len(ALL_ACTIONS))
        for action in proposed:
            probabilities[action] = exploration
        probabilities[base_action] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_with_distribution(
        self,
        state: PhaseAwareState,
    ) -> tuple[object, tuple[float, ...]]:
        from agent.runtime.context import LockAction

        with self._lock:
            probabilities = self.probabilities(state)
            action = sample_action(self.rng, probabilities, state.current_action)
            return LockAction.from_encoded(action), probabilities

    def select_with_probability(self, state: PhaseAwareState) -> tuple[object, float]:
        action, probabilities = self.select_with_distribution(state)
        return action, probabilities[int(getattr(action, "encoded", int(action.protected)))]


class DiscretePPOTrainer:
    def __init__(self, config: PPOConfig | None = None, *, advantage_scale_floor: float = 1e-6):
        self.config = config or PPOConfig()
        if not math.isfinite(advantage_scale_floor) or advantage_scale_floor <= 0:
            raise ValueError('advantage scale floor must be positive and finite')
        # Optional small-batch stabilization. Preserve legacy arithmetic and
        # reports unless explicitly selected by a training protocol.
        self.advantage_scale_floor = float(advantage_scale_floor)
        if self.config.actor_return_mode not in {'per-source-run','global'}:
            raise ValueError('invalid actor return mode')
        if self.config.advantage_baseline not in {"linear-critic", "deployment-group", "deployment-group-observed-cost"}:
            raise ValueError("invalid advantage baseline")
        if (self.config.advantage_baseline == "deployment-group-observed-cost"
                and self.config.calibration_return_mode != "global"):
            raise ValueError("observed-cost baseline requires global return scaling")
        if self.config.categorical_initialization not in {"behavior", "qualified-return"}:
            raise ValueError("invalid categorical initialization")
        if self.config.ppo_reference_mode not in {"behavior", "frozen-actor"}:
            raise ValueError("invalid PPO reference mode")
        if type(self.config.initialize_from_behavior) is not bool:
            raise ValueError("behavior initialization must be boolean")
        if not self.config.initialize_from_behavior and self.config.categorical_initialization == "qualified-return":
            raise ValueError("preserving an actor cannot reinitialize qualified-return logits")

    def train(self, policy: DiscretePPOPolicy, transitions: Sequence[PolicyTransition], *,
              calibration_transitions: Sequence[PolicyTransition] | None = None,
              return_targets: Sequence[float] | None = None) -> dict[str, object]:
        if self.config.logged_action_support and not self.config.conditioned_mask_gradient:
            raise ValueError("logged action support requires corrected four-bit derivatives")
        if self.config.conditioned_mask_gradient and (
            policy.categorical_actor or policy.stay_probability != 0.0
        ):
            raise ValueError("conditioned mask gradient requires unmixed four-bit actor")
        if self.config.ppo_reference_mode == "frozen-actor" and not (
            (policy.categorical_actor and (policy.ppo_supported_projection
                or getattr(policy, 'categorical_group_baseline_supported', False)))
            or (self.config.conditioned_mask_gradient and not policy.categorical_actor
                and policy.stay_probability == 0.0)
        ):
            raise ValueError("frozen actor replay reference requires explicit complete categorical or corrected four-bit PPO")
        if self.config.categorical_initialization == "qualified-return" and not (
            policy.categorical_actor and policy.ppo_supported_projection and policy.local_qualified_action_support
        ):
            raise ValueError("qualified-return initialization requires categorical locally qualified supported PPO")
        if self.config.advantage_baseline.startswith("deployment-group") and not (
            policy.categorical_actor and (policy.ppo_supported_projection
                or getattr(policy, 'categorical_group_baseline_supported', False))
            or (not policy.categorical_actor and self.config.conditioned_mask_gradient
                and policy.stay_probability==0.0)
        ):
            raise ValueError("deployment-group advantages require categorical supported PPO, an explicit joint actor or corrected unmixed bit PPO")
        if calibration_transitions is not None and not (policy.categorical_actor and policy.ppo_supported_projection):
            raise ValueError("separate support calibration requires categorical supported PPO")
        # A caller that resamples individual checkpoints must compute returns
        # on the original episode order, then resample (transition, return)
        # pairs together. Recomputing after phase balancing changes credit
        # assignment, even though each immediate reward remains unchanged.
        if return_targets is not None:
            if len(return_targets) != len(transitions):
                raise ValueError("return targets must match transition count")
            if any(not math.isfinite(float(value)) for value in return_targets):
                raise ValueError("return targets must be finite")
        transitions = policy_transitions(transitions, policy.phase_awareness)
        if any(t.monotone_actions!=policy.monotone_actions for t in transitions):
            raise ValueError('Trajectory action domain differs from actor')
        if not transitions:
            return {"transitions": 0.0, "mean_return": 0.0, "mean_entropy": 0.0}
        returns = (list(map(float, return_targets)) if return_targets is not None
                   else discounted_returns(transitions, self.config.discount))
        samples = [
            (transition, value)
            for transition, value in zip(transitions, returns)
            if not excluded_policy_transition(transition)
        ]
        exclusion_counts: Dict[str, int] = defaultdict(int)
        for transition in transitions:
            reason = policy_transition_exclusion_reason(transition)
            if reason:
                exclusion_counts[reason] += 1
        for transition, _value in samples:
            key = policy.prototype_key(transition.state)
            policy.prototypes.setdefault(key, transition.state)
            policy.prototype_counts[key] += 1
            support = {int(transition.action)}
            if self.config.logged_action_support:
                q = transition.behavior_action_probabilities
                allowed = set(valid_actions(transition.state.current_action, timing_actions=policy.timing_actions,monotone=policy.monotone_actions))
                if len(q) != len(policy.all_actions) or any(not math.isfinite(p) or p < 0.0 for p in q):
                    raise ValueError("logged action support requires finite normalized probabilities")
                if abs(sum(q) - 1.0) > 1e-10 or abs(q[transition.action] - transition.behavior_probability) > 1e-10:
                    raise ValueError("logged action support requires a normalized exact action propensity")
                support = {a for a, probability in enumerate(q) if probability > 0.0}
                if not support <= allowed:
                    raise ValueError("logged action support cannot introduce nonmonotone actions")
            policy.observed_group_actions[policy.group_key(transition.state)].update(support)
        if not samples:
            return {
                "input_transitions": float(len(transitions)),
                "transitions": 0.0,
                "mean_return": 0.0,
                "mean_entropy": 0.0,
                "excluded_transitions": dict(sorted(exclusion_counts.items())),
                "epochs": [],
            }
        behavior_initialization = (initialize_group_logits_from_behavior(
            policy, [transition for transition, _value in samples])
            if self.config.initialize_from_behavior else
            {"mode": "preserve-existing-actor", "reinitialized": False})
        group_sample_counts: Dict[str, int] = defaultdict(int)
        group_action_support: Dict[str, set[int]] = defaultdict(set)
        for transition, _value in samples:
            group_key = policy.group_key(transition.state)
            group_sample_counts[group_key] += 1
            group_action_support[group_key].add(int(transition.action))
        trainable_group_keys = {
            key
            for key, count in group_sample_counts.items()
            if count >= max(1, int(self.config.min_group_samples))
            and len(group_action_support[key]) >= max(1, int(self.config.min_group_actions))
        }
        raw_returns = [value for _transition, value in samples]
        normalized_returns, normalization = normalize_actor_returns(samples,self.config.actor_return_mode)
        calibration_samples = samples
        if calibration_transitions is not None:
            historical = policy_transitions(calibration_transitions, policy.phase_awareness)
            historical_returns = discounted_returns(historical, self.config.discount)
            calibration_samples = [(t, v) for t, v in zip(historical, historical_returns)
                                   if not excluded_policy_transition(t)]
            if not calibration_samples:
                raise ValueError("support calibration corpus has no valid choices")
            # Compaction weights represent real support observations, not an
            # inflated count of the subset used in gradient updates.
            historical_counts: Dict[str, int] = defaultdict(int)
            for t, _v in calibration_samples:
                key = policy.prototype_key(t.state)
                policy.prototypes.setdefault(key, t.state)
                historical_counts[key] += 1
            policy.prototype_counts = defaultdict(int, historical_counts)
        deployment_returns = calibration_returns(calibration_samples, self.config.calibration_return_mode)
        calibration = calibrate_supported_group_actions(
            policy,
            calibration_samples,
            deployment_returns,
            min_action_support=max(
                2,
                int(self.config.min_group_samples)
                // max(2, int(self.config.min_group_actions)),
            ),
        )
        qualified_initialization = None
        if self.config.categorical_initialization == "qualified-return":
            # Only gradient rows initialize the actor. Historical calibration
            # may establish eligibility, but its rewards never enter this prior.
            qualified_initialization = initialize_qualified_logits_from_returns(
                policy, samples, self.config.calibration_return_mode,
                minimum_action_support=calibration['min_action_support'])
        return_mean = sum(raw_returns) / len(raw_returns)
        return_scale = math.sqrt(
            sum((value - return_mean) ** 2 for value in raw_returns) / len(raw_returns)
        )
        critic_order = list(range(len(samples)))
        critic_rng = random.Random(0xC71C)
        for _epoch in range(4):
            critic_rng.shuffle(critic_order)
            for sample_index in critic_order:
                transition, _value = samples[sample_index]
                features = state_features(transition.state)
                group_logits = policy.actor_group_logits[policy.group_key(transition.state)]
                error = normalized_returns[sample_index] - dot(
                    policy.critic_weights, features
                )
                add_scaled(
                    policy.critic_weights,
                    features,
                    self.config.critic_learning_rate * error,
                )
        if self.config.advantage_baseline.startswith("deployment-group"):
            # Deployment chooses one complete mask per state-only group.
            # Use the SAME affine reward scale and an action-independent
            # group baseline for this actor objective. The critic remains
            # trained as before; no calibration-only row enters gradients.
            advantages, advantage_report = deployment_group_advantages(
                policy, samples, self.config.calibration_return_mode,
                observed_cost_baseline=self.config.advantage_baseline == "deployment-group-observed-cost",
            )
        else:
            advantages = [
                target - policy.value(transition.state)
                for (transition, _value), target in zip(samples, normalized_returns)
            ]
        advantage_mean = sum(advantages) / len(advantages)
        advantage_scale = max(
            self.advantage_scale_floor,
            math.sqrt(
                sum((value - advantage_mean) ** 2 for value in advantages) / len(advantages)
            ),
        )
        advantages = [
            (value - advantage_mean) / advantage_scale for value in advantages
        ]
        # Replay importance remains pi/q_actual. Only the proximal clipping
        # reference changes, to a frozen snapshot of this initialized actor.
        # This is NOT an on-policy likelihood or a correction of state coverage.
        reference_probabilities = None
        if self.config.ppo_reference_mode == "frozen-actor":
            reference_probabilities = [policy.probabilities(t.state)[int(t.action)] for t, _ in samples]
            if not all(math.isfinite(p) and p > 0.0 for p in reference_probabilities):
                raise ValueError("invalid frozen actor reference probability")
        entropy_total = 0.0
        updates = 0
        epoch_reports = []
        sample_order = list(range(len(samples)))
        order_rng = random.Random(0xA73C)
        for epoch in range(max(1, int(self.config.epochs))):
            order_rng.shuffle(sample_order)
            epoch_entropy = 0.0
            epoch_updates = 0
            epoch_clipped = 0
            epoch_ratio_total = 0.0
            epoch_proximal_ratio_total = 0.0
            epoch_behavior_correction_total = 0.0
            for sample_index in sample_order:
                transition, _value = samples[sample_index]
                advantage = advantages[sample_index]
                normalized_return = normalized_returns[sample_index]
                group_key = policy.group_key(transition.state)
                group_logits = (policy.categorical_group_logits[group_key] if policy.categorical_actor
                                else policy.actor_group_logits[group_key])
                probabilities = policy.probabilities(transition.state)
                action = int(transition.action)
                if action not in valid_actions(transition.state.current_action, timing_actions=policy.timing_actions,monotone=policy.monotone_actions):
                    continue
                behavior = max(1e-9, float(getattr(transition, "behavior_probability", 1.0)))
                ratio = probabilities[action] / behavior
                proximal_ratio = ratio
                behavior_correction = 1.0
                if reference_probabilities is not None:
                    reference = reference_probabilities[sample_index]
                    proximal_ratio = probabilities[action] / reference
                    behavior_correction = reference / behavior
                clipped_ratio = max(1.0 - self.config.clip_ratio, min(1.0 + self.config.clip_ratio, proximal_ratio))
                if clipped_ratio != proximal_ratio:
                    epoch_clipped += 1
                clipped_high = advantage >= 0.0 and proximal_ratio > 1.0 + self.config.clip_ratio
                clipped_low = advantage < 0.0 and proximal_ratio < 1.0 - self.config.clip_ratio
                # (reference/q_actual)*(pi/reference) == pi/q_actual.
                # Keep the original arithmetic for the non-clipped coefficient.
                coefficient = 0.0 if clipped_high or clipped_low else advantage * ratio
                features = state_features(transition.state)
                if policy.categorical_actor:
                    valid = valid_actions(transition.state.current_action, timing_actions=policy.timing_actions,monotone=policy.monotone_actions)
                    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
                    for candidate in valid:
                        probability = probabilities[candidate]
                        gradient = (1.0 if candidate == action else 0.0) - probability
                        entropy_gradient = (-probability * (math.log(probability) + entropy)
                                            if probability > 0.0 else 0.0)
                        direction = coefficient * gradient + self.config.entropy_weight * entropy_gradient
                        add_scaled(policy.categorical_actor_weights[candidate], features,
                                   self.config.learning_rate * direction)
                        if group_key in trainable_group_keys:
                            group_logits[candidate] += self.config.group_learning_rate * direction
                else:
                    bit_probabilities = actor_bit_probabilities(policy, transition.state)
                    conditioned = self.config.conditioned_mask_gradient
                    entropy = mask_distribution_entropy(probabilities) if conditioned else 0.0
                    for bit_index, bit in enumerate(LOCK_BITS):
                        if policy.monotone_actions and int(transition.state.current_action) & bit:
                            continue
                        probability = max(
                            1e-9,
                            min(1.0 - 1e-9, bit_probabilities[bit_index]),
                        )
                        if conditioned:
                            gradient, entropy_gradient = conditioned_mask_derivatives(
                                probabilities, action, bit, entropy=entropy
                            )
                        else:
                            entropy += -probability * math.log(probability) - (
                                1.0 - probability
                            ) * math.log(1.0 - probability)
                            gradient = (1.0 if action & bit else 0.0) - probability
                            entropy_gradient = probability * (1.0 - probability) * math.log(
                                (1.0 - probability) / probability
                            )
                        update = self.config.learning_rate * (
                            coefficient * gradient
                            + self.config.entropy_weight * entropy_gradient
                        )
                        add_scaled(policy.actor_weights[bit_index], features, update)
                        if group_key in trainable_group_keys:
                            group_logits[bit_index] += self.config.group_learning_rate * (
                                coefficient * gradient
                                + self.config.entropy_weight * entropy_gradient
                            )
                value_error = normalized_return - dot(policy.critic_weights, features)
                add_scaled(
                    policy.critic_weights,
                    features,
                    self.config.critic_learning_rate * value_error,
                )
                entropy_total += entropy
                epoch_entropy += entropy
                epoch_ratio_total += ratio
                epoch_proximal_ratio_total += proximal_ratio
                epoch_behavior_correction_total += behavior_correction
                epoch_updates += 1
                updates += 1
            epoch_reports.append(
                {
                    "epoch": epoch + 1,
                    "updates": epoch_updates,
                    "mean_entropy": epoch_entropy / epoch_updates if epoch_updates else 0.0,
                    "mean_importance_ratio": epoch_ratio_total / epoch_updates if epoch_updates else 0.0,
                    "clip_fraction": epoch_clipped / epoch_updates if epoch_updates else 0.0,
                    **({"mean_proximal_ratio": epoch_proximal_ratio_total / epoch_updates if epoch_updates else 0.0,
                        "mean_behavior_correction": epoch_behavior_correction_total / epoch_updates if epoch_updates else 0.0}
                       if reference_probabilities is not None else {}),
                }
            )
        if policy.ppo_sparse_fallback or policy.ppo_supported_projection:
            mode = "ppo_supported_projection" if policy.ppo_supported_projection else "ppo_sparse_fallback"
            calibration[mode] = project_sparse_actor_heads(policy, calibration_samples)
            selected_counts: Dict[int, int] = defaultdict(int)
            for action in policy.calibrated_group_actions.values():
                selected_counts[action] += 1
            calibration["selected_action_counts"] = {
                str(action): count for action, count in sorted(selected_counts.items())
            }
        return {
            "input_transitions": float(len(transitions)),
            **({"replay_reference": {
                "mode": "frozen-actor", "samples": len(reference_probabilities),
                "reference_is_fixed_across_epochs": True, "logged_behavior_probabilities_unchanged": True,
                "importance_correction": "reference/q_actual; unclipped coefficient remains advantage*pi/q_actual",
                "reference_probability_min": min(reference_probabilities),
                "reference_probability_max": max(reference_probabilities),
                "state_distribution_corrected": False, "on_policy_ppo_claimed": False}}
               if reference_probabilities is not None else {}),
            **({"calibration_input_transitions": len(calibration_transitions),
                "calibration_samples": len(calibration_samples),
                "calibration_scope": "historical support only; PPO gradients and initialization use recent transitions"}
               if calibration_transitions is not None else {}),
            "transitions": float(len(samples)),
            "mean_return": sum(value for _transition, value in samples) / len(samples),
            "return_normalization_mean": return_mean,
            "return_normalization_scale": return_scale,
            "return_normalization": self.config.actor_return_mode,
            **({'advantage_normalization': dict(scale_floor=self.advantage_scale_floor,
                actual_scale=advantage_scale)} if self.advantage_scale_floor != 1e-6 else {}),
            "calibration_return_mode": self.config.calibration_return_mode,
            "actor_score_gradient": (
                "normalized-supported-complete-masks" if self.config.conditioned_mask_gradient
                else "legacy"
            ),
            "source_run_count": len(normalization),
            **({"actor_advantage_baseline": advantage_report}
               if self.config.advantage_baseline.startswith("deployment-group") else {}),
            "excluded_transitions": dict(sorted(exclusion_counts.items())),
            "behavior_initialization": behavior_initialization,
            **({"qualified_return_initialization": qualified_initialization}
               if qualified_initialization is not None else {}),
            "supported_action_calibration": calibration,
            "group_residual_training": {
                "min_samples": max(1, int(self.config.min_group_samples)),
                "min_actions": max(1, int(self.config.min_group_actions)),
                "eligible_groups": len(trainable_group_keys),
                "total_groups": len(group_sample_counts),
            },
            "mean_entropy": entropy_total / updates if updates else 0.0,
            "epochs": epoch_reports,
        }


def mask_distribution_entropy(probabilities: Sequence[float]) -> float:
    """Entropy of the actual normalized joint complete-mask distribution."""
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def conditioned_mask_derivatives(
    probabilities: Sequence[float], action: int, bit: int, *, entropy: float | None = None
) -> tuple[float, float]:
    """d log(pi(action))/d bit-logit and d H(pi)/d bit-logit.

    Conditioning independent Bernoulli bits on an observed set of complete
    masks couples the bits. The score must subtract E_pi[bit], not the raw
    sigmoid. Constant/single-action supports therefore have zero derivatives.
    Assumes a normalized distribution and no extra stay-probability mixture.
    """
    expected = sum(p for mask, p in enumerate(probabilities) if mask & bit)
    score = float(bool(action & bit)) - expected
    h = mask_distribution_entropy(probabilities) if entropy is None else entropy
    entropy_gradient = -sum(
        p * (math.log(p) + h)
        for mask, p in enumerate(probabilities) if mask & bit and p > 0.0
    )
    return score, entropy_gradient


def policy_transitions(transitions: Sequence[PolicyTransition], phase_awareness: str):
    return [dataclasses.replace(t,
            state=policy_observation(t.state, phase_awareness),
            next_state=policy_observation(t.next_state, phase_awareness))
            for t in transitions]


def state_key(state: PhaseAwareState) -> str:
    return "|".join(
        (
            f"phase={state.phase}",
            f"interval={bucket_log(state.inter_round_interval_ms)}",
            f"rs={bucket_size(state.read_set_size)}",
            f"ws={bucket_size(state.write_set_size)}",
            f"rsg={bucket_size(state.read_set_growth)}",
            f"wsg={bucket_size(state.write_set_growth)}",
            f"overlap={bucket_ratio(state.access_overlap_ratio)}",
            f"rounds={bucket_size(state.completed_rounds)}",
            f"ops={bucket_size(state.completed_operations)}",
            f"recent_write={bucket_ratio(state.recent_write_ratio)}",
            f"hot={bucket_ratio(state.hotspot_access_ratio)}",
            f"blocked={bucket_log(state.blocked_time_ms)}",
            f"retry={min(3, state.retry_count)}",
            f"action={int(state.current_action)}",
            f"conflict={normalize_conflict_kind(state.recent_conflict_kind)}",
            f"active={bucket_size(state.global_active_transactions)}",
            f"waiters={bucket_size(state.global_waiter_count)}",
            f"abort={bucket_ratio(state.global_abort_rate)}",
            f"throughput={bucket_log(state.global_throughput)}",
            f"avg_latency={bucket_log(state.global_avg_latency_ms)}",
            f"tail={bucket_log(state.global_tail_latency_ms)}",
            f"agent_tps={bucket_log(state.global_agent_task_throughput)}",
            f"agent_avg_latency={bucket_log(state.global_agent_task_avg_latency_ms)}",
            f"agent_tail={bucket_log(state.global_agent_task_tail_latency_ms)}",
            f"conflict_abort={bucket_ratio(state.global_conflict_abort_rate)}",
            f"background_tps={bucket_log(state.global_background_throughput)}",
            f"background_abort={bucket_ratio(state.global_background_abort_rate)}",
        )
    )


def policy_group_key(state: PhaseAwareState, *, growth_conditioned: bool = False,
                     local_context: bool = False) -> str:
    """Pre-decision aggregation shared by coordinated behavior rollouts.

    Keep action-dependent system outcomes and TM priority out of this key.
    Including them splits otherwise comparable behavior paths into disjoint
    groups, leaving PPO without multi-action support for an ablation state.
    """
    key = "|".join(
        (
            f"phase={state.phase}",
            f"action={int(state.current_action)}",
            f"waiters={bucket_size(state.global_waiter_count)}",
            f"abort={bucket_ratio(state.global_abort_rate)}",
            f"conflict_abort={bucket_ratio(state.global_conflict_abort_rate)}",
            f"hot={bucket_ratio(state.hotspot_access_ratio)}",
            f"retry={min(3, state.retry_count)}",
            f"conflict={normalize_conflict_kind(state.recent_conflict_kind)}",
            f"recent_write={bucket_ratio(state.recent_write_ratio)}",
            f"rounds={bucket_size(state.completed_rounds)}",
            f"rs={bucket_size(state.read_set_size)}",
            f"ws={bucket_size(state.write_set_size)}",
        )
    )
    if growth_conditioned:
        # Use only already-observed footprint growth. Unit resolution below 16
        # retains small branch differences; saturation bounds table growth.
        key += f"|read_growth={min(16, max(0, int(state.read_set_growth)))}"
        key += f"|write_growth={min(16, max(0, int(state.write_set_growth)))}"
    return sparse_support_parent_key(key) if local_context else key


def state_features(state: PhaseAwareState) -> tuple[float, ...]:
    phase = str(state.phase)
    current = int(state.current_action)
    return (
        1.0,
        float(phase == "explore"),
        float(phase == "refine"),
        float(phase == "commit"),
        math.log1p(max(0, state.read_set_size)) / 5.0,
        math.log1p(max(0, state.write_set_size)) / 5.0,
        math.log1p(max(0, state.read_set_growth)) / 4.0,
        math.log1p(max(0, state.write_set_growth)) / 4.0,
        max(0.0, min(1.0, state.access_overlap_ratio)),
        max(0.0, min(1.0, state.hotspot_access_ratio)),
        max(0.0, min(1.0, state.recent_write_ratio)),
        min(1.0, max(0.0, state.retry_count / 3.0)),
        float(normalize_conflict_kind(state.recent_conflict_kind) == "none"),
        float(normalize_conflict_kind(state.recent_conflict_kind) == "version-conflict"),
        float(normalize_conflict_kind(state.recent_conflict_kind) == "lock-preempted"),
        float(normalize_conflict_kind(state.recent_conflict_kind) in {"lock-timeout", "lock-conflict"}),
        float(bool(current & 1)),
        float(bool(current & 2)),
        float(bool(current & 4)),
        float(bool(current & 8)),
        math.log1p(max(0, state.global_active_transactions)) / 4.0,
        math.log1p(max(0, state.global_waiter_count)) / 4.0,
        max(0.0, min(1.0, state.global_abort_rate)),
        math.log1p(max(0.0, state.global_throughput)) / 8.0,
        math.log1p(max(0.0, state.global_avg_latency_ms)) / 10.0,
        math.log1p(max(0.0, state.global_tail_latency_ms)) / 10.0,
        math.log1p(max(0.0, state.blocked_time_ms)) / 10.0,
        math.log1p(max(0.0, state.inter_round_interval_ms)) / 10.0,
        math.log1p(max(0, state.completed_rounds)) / 4.0,
        math.log1p(max(0, state.completed_operations)) / 5.0,
        # Optional already-paid work, NOT the TM priority or future work.
        # The collector defaults to zero, preserving legacy model features.
        math.log1p(max(0.0, state.observed_restart_cost_ms)) / 10.0,
        math.log1p(max(0.0, state.global_agent_task_throughput)) / 8.0,
        math.log1p(max(0.0, state.global_agent_task_avg_latency_ms)) / 10.0,
        math.log1p(max(0.0, state.global_agent_task_tail_latency_ms)) / 10.0,
        max(0.0, min(1.0, state.global_conflict_abort_rate)),
        math.log1p(max(0.0, state.global_background_throughput)) / 8.0,
        max(0.0, min(1.0, state.global_background_abort_rate)),
    )


FEATURE_COUNT = 37


def compiled_entry(state: PhaseAwareState, action: int) -> CompiledPolicyEntry:
    return CompiledPolicyEntry(
        phase=state.phase,
        action=int(action),
        inter_round_bucket=bucket_log(state.inter_round_interval_ms),
        read_set_bucket=bucket_size(state.read_set_size),
        write_set_bucket=bucket_size(state.write_set_size),
        read_growth_bucket=bucket_size(state.read_set_growth),
        write_growth_bucket=bucket_size(state.write_set_growth),
        overlap_bucket=bucket_ratio(state.access_overlap_ratio),
        completed_rounds_bucket=bucket_size(state.completed_rounds),
        completed_operations_bucket=bucket_size(state.completed_operations),
        recent_write_bucket=bucket_ratio(state.recent_write_ratio),
        hotspot_bucket=bucket_ratio(state.hotspot_access_ratio),
        blocked_bucket=bucket_log(state.blocked_time_ms),
        retry_bucket=min(3, state.retry_count),
        current_action=int(state.current_action),
        priority_bucket=0,
        recent_conflict_kind=normalize_conflict_kind(state.recent_conflict_kind),
        active_bucket=bucket_size(state.global_active_transactions),
        waiter_bucket=bucket_size(state.global_waiter_count),
        abort_bucket=bucket_ratio(state.global_abort_rate),
        throughput_bucket=bucket_log(state.global_throughput),
        average_latency_bucket=bucket_log(state.global_avg_latency_ms),
        tail_latency_bucket=bucket_log(state.global_tail_latency_ms),
        agent_task_throughput_bucket=bucket_log(state.global_agent_task_throughput),
        agent_task_average_latency_bucket=bucket_log(
            state.global_agent_task_avg_latency_ms
        ),
        agent_task_tail_latency_bucket=bucket_log(
            state.global_agent_task_tail_latency_ms
        ),
        conflict_abort_bucket=bucket_ratio(state.global_conflict_abort_rate),
        background_throughput_bucket=bucket_log(state.global_background_throughput),
        background_abort_bucket=bucket_ratio(state.global_background_abort_rate),
    )


def normalize_conflict_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized else "none"


def weighted_one_medoid(
    rows: Sequence[tuple[CompiledPolicyEntry, PhaseAwareState, int]],
    *, distance=None,key_fn=None,
) -> CompiledPolicyEntry:
    support_weight = sum(max(1, int(weight)) for _entry, _state, weight in rows)
    if len(rows) == 1:
        return dataclasses.replace(rows[0][0], support_weight=support_weight)
    key_fn=key_fn or state_key
    candidates = sorted(rows, key=lambda row: (-row[2], key_fn(row[1])))[:128]
    if distance is None:
        matrix=_default_medoid_distance_matrix(candidates,rows)
        choice=min(range(len(candidates)),key=lambda i:sum(
            max(1,int(row[2]))*d for row,d in zip(rows,matrix[i])))
        return dataclasses.replace(candidates[choice][0],support_weight=support_weight)
    selected = min(
        candidates,
        key=lambda candidate: sum(
            max(1, int(weight)) * (distance(candidate[0],state) if distance else candidate[0].distance(state))
            for _entry, state, weight in rows
        ),
    )[0]
    return dataclasses.replace(selected, support_weight=support_weight)


def _default_medoid_distance_matrix(candidates,rows):
    """Exactly the legacy bucket-L1 scores; prebucket once, optional vectorization.

    Distances use integer quarter-units, preserving all floats and tie-breaks.
    Selection, candidate limits, assignments and custom metrics are unchanged.
    """
    from agent.runtime.paper_policy import _entry_key,_state_key,bucket_key_distance
    if any(type(row[0]) is not CompiledPolicyEntry for row in candidates):
        return [[entry.distance(state) for _,state,_ in rows] for entry,_,_ in candidates]
    entry_keys=[_entry_key(row[0]) for row in candidates]
    state_keys=[_state_key(row[1]) for row in rows]
    numeric=[i for i in range(1,27) if i!=14]
    safe=all(type(key[i]) is int and abs(key[i])<=1000000
        for key in entry_keys+state_keys for i in numeric)
    try:
        import numpy as np
    except ImportError:
        np=None
    if np is None or not safe or not entry_keys or not state_keys:
        return [[bucket_key_distance(e,s) for s in state_keys] for e in entry_keys]
    weights=(1,1,1,1,1,1,1,1,1,2,1,2,4,2,1,1,2,.25,.25,.25,1,.5,1,2,.5,2)
    scaled=np.asarray([int(weights[i-1]*4) for i in numeric],dtype=np.int64)
    left=np.asarray([[key[i] for i in numeric] for key in entry_keys],dtype=np.int64)
    right=np.asarray([[key[i] for i in numeric] for key in state_keys],dtype=np.int64)
    matrix=[[] for _ in entry_keys]
    # Bound temporary candidate x row x coordinate arrays to a small chunk.
    for start in range(0,len(state_keys),256):
        chunk=right[start:start+256]
        scores=(np.abs(left[:,None,:]-chunk[None,:,:])*scaled).sum(axis=2)
        for index,e in enumerate(entry_keys):
            for offset,s in enumerate(state_keys[start:start+256]):
                score=int(scores[index,offset])+(4000 if e[0]!=s[0] else 0)+(8 if e[14]!=s[14] else 0)
                matrix[index].append(score/4.)
    return matrix


def medoid_candidate_pool(rows, *, distance=None, sampling='frequency',key_fn=None):
    """Bounded deterministic candidates, with optional geometric coverage.

    The legacy pool truncates by support. Coverage instead seeds the most
    supported state and adds the farthest remaining observed state. It never
    creates a state, changes an actor label or uses evaluation questions.
    """
    if sampling not in ('frequency','coverage'):
        raise ValueError('Unsupported medoid candidate sampling')
    key_fn=key_fn or state_key
    unique=[];seen=set()
    for index,row in sorted(enumerate(rows),key=lambda item:(-item[1][2],key_fn(item[1][1]))):
        key=key_fn(row[1])
        if key not in seen:unique.append((index,row));seen.add(key)
    def distances(candidate):
        return [(distance(candidate[0],s) if distance else candidate[0].distance(s))
                for _,s,_ in rows]
    if sampling=='frequency':
        candidates=[row for _,row in unique[:128]]
        return candidates,(_default_medoid_distance_matrix(candidates,rows) if distance is None
                           else [distances(row) for row in candidates])
    candidates=[];matrix=[];selected=set();nearest=[float('inf')]*len(rows)
    for _ in range(min(128,len(unique))):
        choice=(0 if not selected else min((i for i in range(len(unique)) if i not in selected),
            key=lambda i:(-nearest[unique[i][0]],-unique[i][1][2],key_fn(unique[i][1][1]))))
        selected.add(choice);row=unique[choice][1];scores=distances(row)
        candidates.append(row);matrix.append(scores)
        nearest=[min(a,b) for a,b in zip(nearest,scores)]
    return candidates,matrix


def weighted_k_medoids(
    rows: Sequence[tuple[CompiledPolicyEntry, PhaseAwareState, int]],
    count: int,
    *, distance=None,candidate_sampling='frequency',key_fn=None,
) -> tuple[CompiledPolicyEntry, ...]:
    """Greedily retain several representative observed states per action group."""
    key_fn=key_fn or state_key
    requested = max(1, int(count))
    if candidate_sampling not in ('frequency','coverage'):
        raise ValueError('Unsupported medoid candidate sampling')
    if (requested == 1 and candidate_sampling=='frequency') or len(rows) <= 1:
        return (weighted_one_medoid(rows,distance=distance,key_fn=key_fn),)
    candidates,distances=medoid_candidate_pool(rows,distance=distance,sampling=candidate_sampling,key_fn=key_fn)
    requested = min(requested, len(candidates))
    selected: list[int] = []
    best_distances = [float("inf")] * len(rows)
    while len(selected) < requested:
        choice = min(
            (index for index in range(len(candidates)) if index not in selected),
            key=lambda index: (
                sum(
                    max(1, int(rows[row_index][2]))
                    * min(best_distances[row_index], distances[index][row_index])
                    for row_index in range(len(rows))
                ),
                key_fn(candidates[index][1]),
            ),
        )
        selected.append(choice)
        best_distances = [
            min(current, distance)
            for current, distance in zip(best_distances, distances[choice])
        ]
    assigned_weights = [0] * len(selected)
    for row_index, (_entry, _state, weight) in enumerate(rows):
        owner = min(
            range(len(selected)),
            key=lambda position: (
                distances[selected[position]][row_index],
                key_fn(candidates[selected[position]][1]),
            ),
        )
        assigned_weights[owner] += max(1, int(weight))
    return tuple(
        dataclasses.replace(
            candidates[candidate_index][0],
            support_weight=max(1, assigned_weights[position]),
        )
        for position, candidate_index in enumerate(selected)
    )


def actor_bit_probabilities(
    policy: DiscretePPOPolicy,
    state: PhaseAwareState,
) -> tuple[float, ...]:
    features = state_features(state)
    group_logits = policy.actor_group_logits[policy.group_key(state)]
    values = []
    for bit_index, bit in enumerate(LOCK_BITS):
        if policy.monotone_actions and int(state.current_action) & bit:
            values.append(1.0)
        else:
            values.append(
                sigmoid(
                    dot(policy.actor_weights[bit_index], features)
                    + group_logits[bit_index]
                )
            )
    return tuple(values)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        scale = math.exp(-min(60.0, value))
        return 1.0 / (1.0 + scale)
    scale = math.exp(max(-60.0, value))
    return scale / (1.0 + scale)


def sample_action(
    rng: random.Random,
    probabilities: Sequence[float],
    current_action: int,
) -> int:
    draw = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(probabilities):
        cumulative += float(probability)
        if draw <= cumulative:
            return action
    return valid_actions(current_action, timing_actions=len(probabilities) == 32)[-1]


def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def add_scaled(weights: List[float], features: Sequence[float], scale: float) -> None:
    for index, feature in enumerate(features):
        weights[index] += float(scale) * feature


def valid_actions(current_action: int, *, timing_actions: bool = False,monotone: bool = True) -> tuple[int, ...]:
    current = int(current_action) & 0xF
    actions = TIMED_ACTIONS if timing_actions else ALL_ACTIONS
    if not monotone:return actions
    return tuple(action for action in actions if ((action & 0xF) | current) == (action & 0xF))


def discounted_returns(transitions: Sequence[PolicyTransition], discount: float) -> list[float]:
    values = [0.0] * len(transitions)
    running_by_txn: Dict[tuple[str, str], float] = defaultdict(float)
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        txn_key = (str(transition.source_id), transition.txn_id)
        if transition.done:
            running_by_txn[txn_key] = 0.0
        running = float(transition.reward) + float(discount) * running_by_txn[txn_key]
        running_by_txn[txn_key] = running
        values[index] = running
    return values


def calibration_returns(samples, mode):
    """Preserve between-run cost differences for optional deployment calibration.

    PPO advantages retain their original normalization. A single global affine
    transform preserves the ordering of the unchanged paper reward; separate
    source centering can erase costs when sources use different behavior heads.
    This is not a causal correction for unequal state/action support.
    """
    if mode == "per-source-run":
        return normalize_returns_by_source(samples)[0]
    if mode != "global":
        raise ValueError("invalid calibration return mode")
    values = [float(value) for _, value in samples]
    if not values:
        return []
    mean = sum(values) / len(values)
    scale = max(1., math.sqrt(sum((v - mean)**2 for v in values) / len(values)))
    return [(v - mean) / scale for v in values]


def deployment_group_advantages(policy, samples, return_mode, *, observed_cost_baseline=False):
    """State-group centered actor targets aligned with deployment calibration.

    Only the gradient corpus is used. Baseline keys never include the chosen
    action, source label, or outcome. This is an opt-in variance-reduction
    estimator, not a causal correction or an empirical-action argmax.
    Singleton groups have zero advantage (entropy regularization is retained).
    """
    targets = calibration_returns(samples, return_mode)
    offsets = []
    if observed_cost_baseline:
        if return_mode != 'global':
            raise ValueError('observed-cost baseline requires global return scaling')
        values = [float(value) for _, value in samples]
        mean = sum(values)/len(values) if values else 0.
        scale = max(1., math.sqrt(sum((v-mean)**2 for v in values)/len(values))) if values else 1.
        for transition, _ in samples:
            cost = float(transition.state.observed_restart_cost_ms)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError('observed-cost baseline requires finite nonnegative state cost')
            offsets.append(cost/1000./scale)
        # E[R + cost | group] - cost(state) is an action-independent baseline.
        # Costs are observed before the decision; rewards, critic targets and
        # PPO propensities stay unchanged. Never use selected-action costs or
        # terminal outcome metadata here. Center with the SAME state offsets
        # so a constant offset per group has exactly zero effect.
        targets = [value+offset for value,offset in zip(targets,offsets)]
    grouped = defaultdict(list)
    keys = [policy.group_key(t.state) for t, _value in samples]
    for key, target in zip(keys, targets):
        grouped[key].append(target)
    centers = {key: sum(values) / len(values) for key, values in grouped.items()}
    return [target - centers[key] for key, target in zip(keys, targets)], {
        "type": "deployment-group-observed-cost" if observed_cost_baseline else "deployment-group",
        "return_mode": return_mode,
        "baseline_key": "policy_group_key_of_predecision_state",
        "baseline_uses_selected_action": False,
        "calibration_only_rows_used": False,
        "groups": len(grouped),
        "singleton_groups": sum(len(values) == 1 for values in grouped.values()),
        "samples": len(samples),
        **(dict(state_only_cost_field='observed_restart_cost_ms',
            reward_targets_unchanged=True, return_scale=scale,
            mean_normalized_state_cost=sum(offsets)/len(offsets) if offsets else 0.,
            terminal_outcomes_used_as_baseline_inputs=False) if observed_cost_baseline else {}),
    }


def normalize_returns_by_source(
    samples: Sequence[tuple[PolicyTransition, float]],
) -> tuple[list[float], dict[str, tuple[float, float]]]:
    """Give each exploration run equal reward scale without exposing its workload label."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for transition, value in samples:
        grouped[str(transition.source_id)].append(float(value))
    statistics: dict[str, tuple[float, float]] = {}
    for source_id, values in grouped.items():
        mean = sum(values) / len(values)
        scale = max(
            1.0,
            math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        )
        statistics[source_id] = (mean, scale)
    return [
        (float(value) - statistics[str(transition.source_id)][0])
        / statistics[str(transition.source_id)][1]
        for transition, value in samples
    ], statistics


def normalize_actor_returns(samples, mode):
    """Keep between-rollout cost ordering when a batch uses one behavior actor.

    Global mode uses one action-independent affine scale across the actual
    gradient corpus. It neither changes rewards nor corrects state coverage.
    """
    if mode == 'per-source-run':
        return normalize_returns_by_source(samples)
    if mode != 'global':
        raise ValueError('invalid actor return mode')
    if not samples:
        return [],{}
    values=[float(value) for _,value in samples]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('nonfinite actor return')
    mean=sum(values)/len(values)
    scale=max(1.,math.sqrt(sum((v-mean)**2 for v in values)/len(values)))
    statistics={str(t.source_id):(mean,scale) for t,_ in samples}
    return [(value-mean)/scale for value in values],statistics


def initialize_qualified_logits_from_returns(policy, samples, return_mode, *, minimum_action_support):
    """Warm-start qualified residuals from real gradient-corpus action returns.

    Preserve the full valid action space, unsupported logits and the aggregate
    qualified mass of the behavior-initialized residual distribution. Only
    relative probabilities within >=2 directly qualified actions change.
    No extra temperature, pseudo-observations, pooled support or reward edit.
    This is an optional supervised warm-start, not an off-policy causal estimator.
    """
    minimum = max(1, int(minimum_action_support))
    targets = calibration_returns(samples, return_mode)
    grouped = defaultdict(lambda: defaultdict(list))
    for (transition, _value), target in zip(samples, targets):
        if not math.isfinite(target):
            raise ValueError('nonfinite qualified return initialization target')
        grouped[policy.group_key(transition.state)][int(transition.action)].append(target)
    initialized, actions = {}, 0
    for key, by_action in grouped.items():
        eligible = sorted(a for a, values in by_action.items() if len(values) >= minimum
                          and a in policy._sparse_actor_allowed.get(key, ()))
        if len(eligible) < 2:
            continue
        logits = policy.categorical_group_logits[key]
        largest = max(logits[a] for a in eligible)
        log_mass = largest + math.log(sum(math.exp(logits[a]-largest) for a in eligible))
        scores = {a: sum(by_action[a])/len(by_action[a]) for a in eligible}
        top = max(scores.values())
        normalizer = top + math.log(sum(math.exp(v-top) for v in scores.values()))
        for action in eligible:
            logits[action] = log_mass + scores[action] - normalizer
        initialized[key] = {str(a): len(by_action[a]) for a in eligible}
        actions += len(eligible)
    return dict(type='qualified-return', samples=len(samples), return_mode=return_mode,
        minimum_action_support=minimum, initialized_groups=len(initialized), initialized_actions=actions,
        direct_action_counts=initialized, calibration_only_rewards_used=False,
        unsupported_logits_unchanged=True, residual_qualified_mass_preserved=True,
        reward_formula_changed=False, extra_temperature=False)


def initialize_group_logits_from_behavior(
    policy: DiscretePPOPolicy,
    transitions: Sequence[PolicyTransition],
) -> dict[str, int]:
    """Initialize actor residuals from the full behavior support when recoverable."""
    grouped: Dict[str, List[PolicyTransition]] = defaultdict(list)
    prototypes: Dict[str, PhaseAwareState] = {}
    for transition in transitions:
        key = policy.group_key(transition.state)
        prototypes.setdefault(key, transition.state)
        grouped[key].append(transition)
    exact_groups = 0
    fallback_groups = 0
    for key, rows in grouped.items():
        valid = valid_actions(prototypes[key].current_action, timing_actions=policy.timing_actions,monotone=policy.monotone_actions)
        reconstructed = [reconstruct_behavior_distribution(row) for row in rows]
        has_explicit_distributions = all(
            len(
                tuple(
                    getattr(row, "behavior_action_probabilities", ()) or ()
                )
            )
            == len(policy.all_actions)
            for row in rows
        )
        if all(distribution is not None for distribution in reconstructed) and (
            has_explicit_distributions
            or behavior_distributions_consistent(reconstructed)
        ):
            exact_groups += 1
            raw = {
                # A timing-aware action is invalid once its underlying mask
                # has already been acquired.  Local state groups can still
                # contain such rows through legacy coarse buckets; treat the
                # absent probability as zero rather than indexing a mask that
                # was not legal for that individual observation.
                action: sum(distribution.get(action, 0.0) for distribution in reconstructed) / len(rows)
                for action in valid
            }
        else:
            fallback_groups += 1
            observed: Dict[int, List[float]] = defaultdict(list)
            for row in rows:
                observed[int(row.action)].append(float(row.behavior_probability))
            raw = {
                action: (
                    sum(observed[action]) / len(observed[action])
                    if observed.get(action)
                    else 1e-6
                )
                for action in valid
            }
        total = sum(raw.values())
        normalized = {action: raw[action] / total for action in valid}
        if policy.categorical_actor:
            logits = policy.categorical_group_logits[key]
            for action in valid:
                logits[action] = math.log(max(1e-12, normalized[action]))
            continue
        logits = policy.actor_group_logits[key]
        for bit_index, bit in enumerate(LOCK_BITS):
            if policy.monotone_actions and int(prototypes[key].current_action) & bit:
                logits[bit_index] = 0.0
                continue
            bit_probability = sum(
                probability
                for action, probability in normalized.items()
                if action & bit
            )
            bit_probability = max(1e-6, min(1.0 - 1e-6, bit_probability))
            logits[bit_index] = math.log(bit_probability / (1.0 - bit_probability))
    return {
        "state_groups": len(grouped),
        "full_support_groups": exact_groups,
        "fallback_groups": fallback_groups,
    }


def behavior_distributions_consistent(
    distributions: Sequence[dict[int, float] | None],
    *,
    tolerance: float = 1e-6,
) -> bool:
    rows = [distribution for distribution in distributions if distribution is not None]
    if not rows:
        return False
    reference = rows[0]
    return all(
        set(distribution) == set(reference)
        and all(
            abs(distribution[action] - reference[action]) <= tolerance
            for action in reference
        )
        for distribution in rows[1:]
    )


def reconstruct_behavior_distribution(
    transition: PolicyTransition,
) -> dict[int, float] | None:
    """Recover DiscretePPOPolicy's stay-mixture from one recorded propensity."""
    raw_distribution = tuple(
        float(value)
        for value in getattr(transition, "behavior_action_probabilities", ()) or ()
    )
    timing_actions = len(raw_distribution) == len(TIMED_ACTIONS)
    valid = valid_actions(transition.state.current_action, timing_actions=timing_actions,monotone=transition.monotone_actions)
    if not valid:
        return None
    if len(valid) == 1:
        return {valid[0]: 1.0}
    action = int(transition.action)
    if action not in valid:
        return None
    recorded = raw_distribution
    if recorded:
        universe = TIMED_ACTIONS if timing_actions else ALL_ACTIONS
        if len(recorded) != len(universe):
            return None
        if any(value < 0.0 or value > 1.0 for value in recorded):
            return None
        if any(recorded[candidate] > 1e-12 for candidate in universe if candidate not in valid):
            return None
        if abs(sum(recorded) - 1.0) > 1e-6:
            return None
        return {candidate: recorded[candidate] for candidate in valid}
    selected_probability = float(transition.behavior_probability)
    if not 0.0 < selected_probability <= 1.0:
        return None
    current = int(transition.state.current_action) & 0xF
    if action == current:
        other_probability = (1.0 - selected_probability) / (len(valid) - 1)
        distribution = {candidate: other_probability for candidate in valid}
        distribution[current] = selected_probability
    else:
        other_probability = selected_probability
        current_probability = 1.0 - (len(valid) - 1) * other_probability
        if current_probability < -1e-9 or current_probability > 1.0 + 1e-9:
            return None
        distribution = {candidate: other_probability for candidate in valid}
        distribution[current] = max(0.0, min(1.0, current_probability))
    if any(value < -1e-9 or value > 1.0 + 1e-9 for value in distribution.values()):
        return None
    if abs(sum(distribution.values()) - 1.0) > 1e-6:
        return None
    return distribution


def excluded_policy_transition(transition: PolicyTransition) -> bool:
    return bool(policy_transition_exclusion_reason(transition))


def calibrate_supported_group_actions(
    policy: DiscretePPOPolicy,
    samples: Sequence[tuple[PolicyTransition, float]],
    normalized_returns: Sequence[float],
    *,
    min_action_support: int = 2,
) -> dict[str, object]:
    """Project the deployment head onto empirically supported full actions.

    The factorized actor is useful for sharing signal between lock bits, but
    its independent Bernoulli heads can combine individually useful bits into
    a mask that was never beneficial as a whole.  Deployment therefore uses
    the highest-return complete action with direct behavior support in the
    same pre-decision state group.
    """

    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    current_actions: Dict[str, int] = {}
    for (transition, _raw_return), normalized_return in zip(
        samples, normalized_returns
    ):
        group_key = policy.group_key(transition.state)
        current_actions.setdefault(
            group_key, int(transition.state.current_action) & 0xF
        )
        grouped[group_key][int(transition.action)].append(
            float(normalized_return)
        )

    minimum = max(1, int(min_action_support))
    policy._calibration_action_support = {key: {a: len(values) for a, values in actions.items()}
                                          for key, actions in grouped.items()}
    policy._calibration_min_action_support = minimum
    if policy.ppo_sparse_fallback or policy.ppo_supported_projection:
        policy._sparse_actor_allowed.clear()
    pooled: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    if policy.pooled_sparse_action_support:
        for key, by_action in grouped.items():
            parent = sparse_support_parent_key(key)
            for action, values in by_action.items():
                pooled[parent][action].extend(values)
    audit_minimum = max(3, minimum)
    trusted_action_groups: Dict[int, int] = defaultdict(int)
    for by_action in grouped.values():
        audited = {
            action: values
            for action, values in by_action.items()
            if len(values) >= audit_minimum
        }
        if len(audited) < 2:
            continue
        selected = min(
            audited,
            key=lambda action: (-sum(audited[action]) / len(audited[action]), action),
        )
        trusted_action_groups[int(selected)] += 1
    trusted_actions = set(trusted_action_groups)

    calibrated: Dict[str, int] = {}
    supported_groups = 0
    multi_action_groups = 0
    action_counts: Dict[int, int] = defaultdict(int)
    retained_sparse_groups = 0
    pooled_sparse_groups = 0
    for key, by_action in grouped.items():
        eligible = {
            action: values
            for action, values in by_action.items()
            if len(values) >= minimum
        }
        trusted_eligible = {
            action: values
            for action, values in eligible.items()
            if action in trusted_actions
        }
        if (trusted_actions and not policy.strict_local_action_support
                and not policy.local_qualified_action_support):
            eligible = trusted_eligible
        if not eligible and policy.pooled_sparse_action_support:
            # Preserve the original trust filter and every locally qualified
            # head. Only its frequency-based sparse fallback may borrow
            # measured return evidence across volatile global buckets, with
            # unchanged minimum support and local pre-decision footprint.
            eligible = {action: values for action, values in pooled[sparse_support_parent_key(key)].items()
                        if len(values) >= minimum and (not trusted_actions or action in trusted_actions)}
            pooled_sparse_groups += bool(eligible)
        if eligible and policy.ppo_supported_projection:
            # Keep exactly the original local minimum and global trust filter.
            # Let trained actor probabilities, not the pre-training return
            # table, select among those qualified complete masks.
            policy._sparse_actor_allowed[key] = set(eligible)
        if not eligible:
            if policy.strict_local_action_support:
                # Do not expand a mask using evidence from another state.
                # Retaining the current protection is always monotonic; OCC
                # validation and retry safety remain mandatory at runtime.
                eligible = {int(current_actions.get(key, 0)) & 0xF: [0.0]}
                retained_sparse_groups += 1
            else:
                supported = {
                    action: values
                    for action, values in by_action.items()
                    if action in trusted_actions
                }
                if not supported:
                    retained = int(current_actions.get(key, 0)) & 0xF
                    supported = {retained: [0.0]}
                if policy.ppo_sparse_fallback or policy.ppo_supported_projection:
                    # Exactly the original fallback's observed/trusted masks;
                    # do not override any locally qualified return head.
                    policy._sparse_actor_allowed[key] = set(supported)
                largest = max(len(values) for values in supported.values())
                eligible = {
                    action: values
                    for action, values in supported.items()
                    if len(values) == largest
                }
        if len(eligible) >= 2:
            multi_action_groups += 1
        selected = min(
            eligible,
            key=lambda action: (-sum(eligible[action]) / len(eligible[action]), action),
        )
        calibrated[key] = int(selected)
        action_counts[int(selected)] += 1
        supported_groups += 1

    policy.calibrated_group_actions = calibrated
    return {
        "min_action_support": minimum,
        "calibrated_groups": supported_groups,
        "multi_action_groups": multi_action_groups,
        "audit_min_action_support": audit_minimum,
        "trusted_action_groups": {
            str(action): trusted_action_groups[action]
            for action in sorted(trusted_action_groups)
        },
        "trusted_actions": sorted(trusted_actions),
        "selected_action_counts": {
            str(action): action_counts[action] for action in sorted(action_counts)
        },
        "projection": "highest_normalized_return_supported_complete_action",
        **({"strict_local_action_support": True, "retained_sparse_groups": retained_sparse_groups,
            "sparse_fallback": "retain_current_mask_when_no_action_meets_local_minimum"}
           if policy.strict_local_action_support else {}),
        **({"pooled_sparse_action_support": True, "pooled_sparse_groups": pooled_sparse_groups,
            "pooled_fields": ["waiters", "abort", "conflict_abort"],
            "pooled_scope": "only original sparse fallback; local eligibility and global trust unchanged"}
           if policy.pooled_sparse_action_support else {}),
    }


def project_sparse_actor_heads(
    policy: DiscretePPOPolicy,
    samples: Sequence[tuple[PolicyTransition, float]],
) -> dict[str, object]:
    """After PPO, project heads onto the same original allowed masks.

    Average learned probabilities over actual observations in each group.
    No new masks, pooled evidence, reward edits or future observations enter.
    Sparse-only mode leaves qualified complete-action heads unchanged.
    Supported mode uses their original qualified mask sets without relaxing
    the local support minimum or the global trust filter.
    """
    totals: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)
    for transition, _value in samples:
        key = policy.group_key(transition.state)
        allowed = policy._sparse_actor_allowed.get(key)
        if not allowed:
            continue
        probabilities = policy.probabilities(transition.state)
        counts[key] += 1
        for action in allowed:
            if action in valid_actions(transition.state.current_action, timing_actions=policy.timing_actions,monotone=policy.monotone_actions):
                totals[key][action] += probabilities[action]
    changes = 0
    for key, scores in totals.items():
        previous = policy.calibrated_group_actions[key]
        chosen = min(scores, key=lambda a: (-scores[a], a != previous, a))
        changes += chosen != previous
        policy.calibrated_group_actions[key] = chosen
    return dict(scope=("post-PPO qualified and sparse heads; identical original eligibility and trust"
                       if policy.ppo_supported_projection else
                       "post-PPO only original sparse fallback; identical observed/trusted allowed masks"),
                groups=len(totals), changed_heads=changes, observed_samples=sum(counts.values()))


def sparse_support_parent_key(key: str) -> str:
    """Keep phase/current mask/heat/retry/conflict/footprint, pool global noise."""
    return "|".join(part for part in key.split("|")
                    if part.split("=", 1)[0] not in {"waiters", "abort", "conflict_abort"})


def policy_transition_exclusion_reason(transition: PolicyTransition) -> str:
    if getattr(transition,'decision_origin','')=='asynchronous-policy-refinement':
        return 'asynchronous-policy-refinement'
    if fixed_initial_occ_transition(transition):
        return "fixed-initial-occ"
    if len(valid_actions(transition.state.current_action,monotone=transition.monotone_actions)) <= 1:
        return "no-action-choice"
    if float(transition.behavior_probability) >= 1.0 - 1e-12:
        return "deterministic-without-support"
    return ""


def audit_policy(
    policy: DiscretePPOPolicy,
    compiled: CompiledPhasePolicy,
    transitions: Sequence[PolicyTransition],
    *,
    min_action_support: int = 3,
    discount: float = 0.99,
) -> dict[str, object]:
    """Compare actor and compiled lookup decisions with empirical group returns."""
    transitions = policy_transitions(transitions, policy.phase_awareness)
    returns = discounted_returns(transitions, float(discount))
    samples = [
        (transition, value)
        for transition, value in zip(transitions, returns)
        if not excluded_policy_transition(transition)
    ]
    if not samples:
        return {
            "audited_state_groups": 0,
            "actor_top_action_agreement": 0.0,
            "compiled_action_agreement": 0.0,
        }
    normalized_returns, _normalization = normalize_returns_by_source(samples)
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    representatives: Dict[str, PhaseAwareState] = {}
    for (transition, _raw_return), normalized_return in zip(samples, normalized_returns):
        key = policy.group_key(transition.state)
        representatives.setdefault(key, transition.state)
        grouped[key][int(transition.action)].append(float(normalized_return))

    actor_agreement = 0
    compiled_agreement = 0
    weighted_actor_agreement = 0
    weighted_compiled_agreement = 0
    weighted_actor_regret = 0.0
    weighted_compiled_regret = 0.0
    actor_regret_weight = 0
    compiled_regret_weight = 0
    total_weight = 0
    audited_groups = 0
    actor_unobserved = 0
    compiled_unobserved = 0
    empirical_actions: Dict[int, int] = defaultdict(int)
    actor_actions: Dict[int, int] = defaultdict(int)
    compiled_actions: Dict[int, int] = defaultdict(int)
    for key, by_action in grouped.items():
        eligible = {
            action: values
            for action, values in by_action.items()
            if len(values) >= max(1, int(min_action_support))
        }
        if len(eligible) < 2:
            continue
        means = {
            action: sum(values) / len(values)
            for action, values in eligible.items()
        }
        best_action = min(
            means,
            key=lambda action: (-means[action], action),
        )
        state = representatives[key]
        probabilities = policy.probabilities(state)
        actor_action = max(
            valid_actions(state.current_action, timing_actions=policy.timing_actions,monotone=policy.monotone_actions),
            key=lambda action: (probabilities[action], -action),
        )
        selected = compiled.select(state)
        compiled_action = int(getattr(selected, "encoded", int(selected.protected)))
        weight = sum(len(values) for values in eligible.values())
        audited_groups += 1
        total_weight += weight
        actor_agreement += int(actor_action == best_action)
        compiled_agreement += int(compiled_action == best_action)
        weighted_actor_agreement += int(actor_action == best_action) * weight
        weighted_compiled_agreement += int(compiled_action == best_action) * weight
        empirical_actions[best_action] += weight
        actor_actions[actor_action] += weight
        compiled_actions[compiled_action] += weight
        if actor_action in means:
            weighted_actor_regret += (means[best_action] - means[actor_action]) * weight
            actor_regret_weight += weight
        else:
            actor_unobserved += 1
        if compiled_action in means:
            weighted_compiled_regret += (
                means[best_action] - means[compiled_action]
            ) * weight
            compiled_regret_weight += weight
        else:
            compiled_unobserved += 1

    def action_counts(values: Dict[int, int]) -> dict[str, int]:
        return {str(action): values[action] for action in sorted(values)}

    return {
        "min_action_support": max(1, int(min_action_support)),
        "policy_control_point_transitions": len(samples),
        "audited_state_groups": audited_groups,
        "audited_transition_weight": total_weight,
        "actor_top_action_agreement": actor_agreement / audited_groups if audited_groups else 0.0,
        "compiled_action_agreement": compiled_agreement / audited_groups if audited_groups else 0.0,
        "actor_weighted_agreement": (
            weighted_actor_agreement / total_weight if total_weight else 0.0
        ),
        "compiled_weighted_agreement": (
            weighted_compiled_agreement / total_weight if total_weight else 0.0
        ),
        "actor_weighted_regret": (
            weighted_actor_regret / actor_regret_weight if actor_regret_weight else 0.0
        ),
        "compiled_weighted_regret": (
            weighted_compiled_regret / compiled_regret_weight
            if compiled_regret_weight else 0.0
        ),
        "actor_unobserved_action_groups": actor_unobserved,
        "compiled_unobserved_action_groups": compiled_unobserved,
        "empirical_best_action_counts": action_counts(empirical_actions),
        "actor_top_action_counts": action_counts(actor_actions),
        "compiled_action_counts": action_counts(compiled_actions),
    }


def fixed_initial_occ_transition(transition: PolicyTransition) -> bool:
    state = transition.state
    return bool(
        state.phase in {"explore", "all"}
        and state.completed_operations == 0
        and state.current_action == 0
        and int(transition.action) == 0
        and abs(float(transition.behavior_probability) - 1.0) < 1e-12
    )
