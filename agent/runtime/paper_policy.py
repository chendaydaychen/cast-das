"""Compiled phase-aware policy table for the paper ATCC runtime."""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .context import LockAction, LockClass
from .state_collector import PhaseAwareState, policy_observation


@dataclasses.dataclass(frozen=True)
class CompiledPolicyEntry:
    phase: str
    action: int
    inter_round_bucket: int = 0
    read_set_bucket: int = 0
    write_set_bucket: int = 0
    read_growth_bucket: int = 0
    write_growth_bucket: int = 0
    overlap_bucket: int = 0
    completed_rounds_bucket: int = 0
    completed_operations_bucket: int = 0
    recent_write_bucket: int = 0
    hotspot_bucket: int = 0
    blocked_bucket: int = 0
    retry_bucket: int = 0
    current_action: int = 0
    priority_bucket: int = 0
    recent_conflict_kind: str = "none"
    active_bucket: int = 0
    waiter_bucket: int = 0
    abort_bucket: int = 0
    throughput_bucket: int = 0
    average_latency_bucket: int = 0
    tail_latency_bucket: int = 0
    agent_task_throughput_bucket: int = 0
    agent_task_average_latency_bucket: int = 0
    agent_task_tail_latency_bucket: int = 0
    conflict_abort_bucket: int = 0
    background_throughput_bucket: int = 0
    background_abort_bucket: int = 0
    support_weight: int = 1
    context_key: str = ""

    def distance(self, state: PhaseAwareState) -> float:
        phase_penalty = 1000.0 if self.phase != state.phase else 0.0
        return phase_penalty + sum(
            weight * abs(left - right)
            for left, right, weight in (
                (self.inter_round_bucket, bucket_log(state.inter_round_interval_ms), 1.0),
                (self.read_set_bucket, bucket_size(state.read_set_size), 1.0),
                (self.write_set_bucket, bucket_size(state.write_set_size), 1.0),
                (self.read_growth_bucket, bucket_size(state.read_set_growth), 1.0),
                (self.write_growth_bucket, bucket_size(state.write_set_growth), 1.0),
                (self.overlap_bucket, bucket_ratio(state.access_overlap_ratio), 1.0),
                (self.completed_rounds_bucket, bucket_size(state.completed_rounds), 1.0),
                (self.completed_operations_bucket, bucket_size(state.completed_operations), 1.0),
                (self.recent_write_bucket, bucket_ratio(state.recent_write_ratio), 1.0),
                (self.hotspot_bucket, bucket_ratio(state.hotspot_access_ratio), 2.0),
                (self.blocked_bucket, bucket_log(state.blocked_time_ms), 1.0),
                (self.retry_bucket, min(3, state.retry_count), 2.0),
                (self.current_action, int(state.current_action), 4.0),
                (
                    0 if self.recent_conflict_kind == normalize_conflict_kind(state.recent_conflict_kind) else 1,
                    0,
                    2.0,
                ),
                (self.active_bucket, bucket_size(state.global_active_transactions), 1.0),
                (self.waiter_bucket, bucket_size(state.global_waiter_count), 1.0),
                (self.abort_bucket, bucket_ratio(state.global_abort_rate), 2.0),
                (self.throughput_bucket, bucket_log(state.global_throughput), 0.25),
                (self.average_latency_bucket, bucket_log(state.global_avg_latency_ms), 0.25),
                (self.tail_latency_bucket, bucket_log(state.global_tail_latency_ms), 0.25),
                (
                    self.agent_task_throughput_bucket,
                    bucket_log(state.global_agent_task_throughput),
                    1.0,
                ),
                (
                    self.agent_task_average_latency_bucket,
                    bucket_log(state.global_agent_task_avg_latency_ms),
                    0.5,
                ),
                (
                    self.agent_task_tail_latency_bucket,
                    bucket_log(state.global_agent_task_tail_latency_ms),
                    1.0,
                ),
                (
                    self.conflict_abort_bucket,
                    bucket_ratio(state.global_conflict_abort_rate),
                    2.0,
                ),
                (
                    self.background_throughput_bucket,
                    bucket_log(state.global_background_throughput),
                    0.5,
                ),
                (
                    self.background_abort_bucket,
                    bucket_ratio(state.global_background_abort_rate),
                    2.0,
                ),
            )
        )


class CompiledPhasePolicy:
    requires_reviewer_contention_statistics=False
    def __init__(
        self,
        entries: Iterable[CompiledPolicyEntry] = (),
        *,
        generation: int = 0,
        medoids_per_group: int = 1,
        refinement_actor: dict[str, object] | None = None,
        occ_cold_start_guard: bool = False,
        phase_awareness: str = "enabled",
        prebucketed_lookup: bool = False,
        calibrated_group_actions: dict[str, int] | None = None,
        growth_conditioned_groups: bool = False,
        local_context_groups: bool = False,
        packed_fallback_lookup: bool = False,
        unanimous_fallback_lookup: bool = False,
        protection_relevant_hotness: bool = False,
        exact_categorical_lookup: bool = False,
        timing_actions: bool = False,
        remove_phase_index_proxy: bool = False,
        context_partitioned_lookup: bool = False,
        state_projection: str = '',
    ):
        if phase_awareness not in {"enabled", "disabled"}:
            raise ValueError("invalid phase awareness")
        self.phase_awareness = phase_awareness
        if remove_phase_index_proxy and phase_awareness != "disabled":
            raise ValueError("phase-index proxy removal requires NoPhase")
        self.remove_phase_index_proxy = bool(remove_phase_index_proxy)
        self.growth_conditioned_groups = bool(growth_conditioned_groups)
        self.local_context_groups = bool(local_context_groups)
        self.prebucketed_lookup = bool(prebucketed_lookup)
        self.packed_fallback_lookup = bool(packed_fallback_lookup)
        self.unanimous_fallback_lookup = bool(unanimous_fallback_lookup)
        self.protection_relevant_hotness = bool(protection_relevant_hotness)
        self.timing_actions = bool(timing_actions)
        self.entries = tuple(entries)
        self.context_partitioned_lookup = bool(context_partitioned_lookup)
        self.state_projection = state_projection
        if state_projection:
            from .table_observation import validate_projection
            validate_projection(state_projection)
            if not self.context_partitioned_lookup:
                raise ValueError('Shared state projection requires context-partitioned pure table')
        if self.context_partitioned_lookup:
            if (refinement_actor or calibrated_group_actions
                    or unanimous_fallback_lookup or occ_cold_start_guard):
                raise ValueError("context-partitioned pure table excludes actor/override/shortcut modes")
            if any(not entry.context_key for entry in self.entries):
                raise ValueError("context-partitioned entries require context keys")
        elif any(entry.context_key for entry in self.entries):
            raise ValueError("entry context keys require context-partitioned metadata")
        if phase_awareness == "disabled" and any(e.phase != "all" for e in self.entries):
            raise ValueError("NoPhase entries must be compiled from canonical observations")
        self.generation = int(generation)
        self.medoids_per_group = max(1, int(medoids_per_group))
        self.refinement_actor = dict(refinement_actor or {})
        self.exact_categorical_lookup = bool(exact_categorical_lookup)
        if self.exact_categorical_lookup and self.refinement_actor.get("type") != "locally-supported-categorical-lock-masks":
            raise ValueError("exact categorical lookup requires a locally supported categorical actor")
        self.occ_cold_start_guard = bool(occ_cold_start_guard)
        self.calibrated_group_actions = dict(calibrated_group_actions or {})
        for key, action in self.calibrated_group_actions.items():
            fields = dict(part.split("=", 1) for part in str(key).split("|"))
            current = int(fields.get("action", "-1"))
            phase = fields.get("phase")
            if self.local_context_groups and (
                {"waiters", "abort", "conflict_abort"}.intersection(fields)
                or not {"phase", "action", "hot", "retry", "conflict", "recent_write", "rounds", "rs", "ws"}.issubset(fields)
            ):
                raise ValueError("invalid local-context group key")
            allowed_phases = {"all"} if phase_awareness == "disabled" else {"explore", "refine", "commit"}
            growth_fields = {"read_growth", "write_growth"}
            if self.growth_conditioned_groups:
                if not growth_fields.issubset(fields) or any(
                    not 0 <= int(fields[name]) <= 16 for name in growth_fields
                ):
                    raise ValueError("invalid growth-conditioned group key")
            elif growth_fields.intersection(fields):
                raise ValueError("growth-conditioned keys require explicit artifact metadata")
            action_mask = int(action) & 0xF
            max_action = 31 if self.timing_actions else 15
            if (type(action) is not int or not 0 <= action <= max_action or not 0 <= current <= 15
                    or (action_mask | current) != action_mask or phase not in allowed_phases):
                raise ValueError("invalid or noncanonical calibrated group action")
        if self.refinement_actor.get("type") == "locally-supported-categorical-lock-masks":
            self._validate_supported_categorical_actor()
        exact = {}
        indexed = defaultdict(list)
        by_phase = defaultdict(list)
        by_context = defaultdict(list)
        for entry in self.entries:
            key = _entry_key(entry)
            if self.context_partitioned_lookup:
                key += (entry.context_key,)
            incumbent = exact.get(key)
            if incumbent is None or entry.support_weight > incumbent.support_weight:
                exact[key] = entry
            indexed[(entry.phase, int(entry.current_action))].append(entry)
            by_phase[entry.phase].append(entry)
            by_context[(entry.phase, int(entry.current_action), entry.context_key)].append(entry)
        self._exact = exact
        self._entry_keys = {id(entry): _entry_key(entry) for entry in self.entries}
        self._indexed = {key: tuple(values) for key, values in indexed.items()}
        self._by_phase = {key: tuple(values) for key, values in by_phase.items()}
        self._by_context = {key: tuple(values) for key, values in by_context.items()}
        self._fallback_cache: dict[tuple[object, ...], CompiledPolicyEntry] = {}
        self._packed_fallback_views: dict[tuple[object, ...], _PackedFallbackView] = {}
        self._unanimous_fallback_actions: dict[tuple[str, int], int | None] = {}
        if self.packed_fallback_lookup and self.context_partitioned_lookup:
            self.prepare_lookup_indexes()

    def prepare_lookup_indexes(self):
        """Build exact context/unknown-context views without observing a workload."""
        if not self.packed_fallback_lookup or not self.context_partitioned_lookup:
            return
        for key, candidates in self._indexed.items():
            view_key = key + (None,)
            if view_key not in self._packed_fallback_views:
                self._packed_fallback_views[view_key] = _PackedFallbackView(candidates)
        for key, candidates in self._by_context.items():
            if key not in self._packed_fallback_views:
                self._packed_fallback_views[key] = _PackedFallbackView(candidates)

    def select(self, state: PhaseAwareState) -> LockAction:
        state = policy_observation(state, self.phase_awareness,
            remove_phase_index_proxy=self.remove_phase_index_proxy)
        if self.state_projection:
            from .table_observation import project_table_state
            state = project_table_state(state, preserve_growth=self.growth_conditioned_groups)
        if self.occ_cold_start_guard and should_keep_occ(state):
            return LockAction()
        if self.calibrated_group_actions:
            from agent.cc.atcc.ppo import policy_group_key
            group_key = policy_group_key(
                state, growth_conditioned=self.growth_conditioned_groups,
                local_context=self.local_context_groups)
            head = self.calibrated_group_actions.get(group_key)
            if head is not None:
                if self.refinement_actor.get("type") == "locally-supported-categorical-lock-masks":
                    refined = (self._select_supported_categorical_exact(state, group_key) if self.exact_categorical_lookup
                        else self._select_refinement(state))
                    if refined is not None:
                        return refined
                return LockAction.from_encoded(head)
        if self.unanimous_fallback_lookup and self.entries and (not self.refinement_actor or
                self.refinement_actor.get("type") == "locally-supported-categorical-lock-masks"):
            uniform = self._unanimous_fallback_action(state)
            if uniform is not None:
                return LockAction.from_encoded(uniform)
        if not self.entries:
            return self._select_refinement(state) or LockAction()
        key = _state_key(state)
        context_key = None
        if self.context_partitioned_lookup:
            from agent.cc.atcc.ppo import policy_group_key
            context_key = policy_group_key(state, growth_conditioned=self.growth_conditioned_groups,
                                          local_context=self.local_context_groups)
            if self.state_projection:
                from .table_observation import projected_context_key
                context_key = projected_context_key(context_key, state)
            key += (context_key,)
        entry = self._exact.get(key)
        if entry is None:
            entry = self._fallback_cache.get(key)
            if entry is None:
                candidates = self._indexed.get((state.phase, int(state.current_action)))
                if context_key is not None:
                    candidates = self._by_context.get(
                        (state.phase, int(state.current_action), context_key)) or candidates
                if not candidates:
                    candidates = tuple(
                        candidate
                        for candidate in self._by_phase.get(state.phase, ())
                        if ((int(candidate.action) & 0xF) | int(state.current_action))
                        == (int(candidate.action) & 0xF)
                    )
                if not candidates:
                    return self._select_refinement(state) or LockAction(
                        LockClass(int(state.current_action))
                    )
                entry = None
                if self.packed_fallback_lookup:
                    view_key = (state.phase, int(state.current_action))
                    if context_key is not None:
                        # Unknown contexts share the unchanged phase/mask pool.
                        known = (state.phase, int(state.current_action), context_key)
                        view_key += (context_key if known in self._by_context else None,)
                    view = self._packed_fallback_views.get(view_key)
                    if view is None:
                        view = self._packed_fallback_views.setdefault(
                            view_key, _PackedFallbackView(candidates))
                    entry = view.nearest(key[:27] if context_key is not None else key)
                if entry is None:
                    entry = min(
                        candidates,
                        key=lambda candidate: (
                            self._distance(candidate, state, key),
                            -max(1, int(candidate.support_weight)),
                            int(candidate.action),
                        ),
                    )
                self._fallback_cache.setdefault(key, entry)
        threshold = float(self.refinement_actor.get("distance_threshold", -1.0) or 0.0)
        if self.refinement_actor and self._distance(entry, state, key) > threshold:
            refined = self._select_refinement(state)
            if refined is not None:
                return refined
        return LockAction.from_encoded(int(entry.action))

    def _unanimous_fallback_action(self, state):
        """Skip ranking only when every eligible entry yields the same action.

        Known heads have precedence. A direct exact hit belongs to the same
        phase/current-mask indexed pool, so it also returns that sole action.
        Refinement is excluded: its full-distance gate can change the output.
        No candidate is removed and no nearest-neighbor approximation is made.
        """
        view = (state.phase, int(state.current_action))
        if view not in self._unanimous_fallback_actions:
            candidates = self._indexed.get(view)
            if not candidates:
                candidates = tuple(e for e in self._by_phase.get(state.phase, ())
                    if ((int(e.action) & 0xF) | int(state.current_action)) == (int(e.action) & 0xF))
            actions = {int(e.action) for e in candidates}
            self._unanimous_fallback_actions.setdefault(
                view, next(iter(actions)) if len(actions) == 1 else None)
        return self._unanimous_fallback_actions[view]

    def _distance(self, entry, state, key):
        if not self.prebucketed_lookup:
            return entry.distance(state)
        return bucket_key_distance(self._entry_keys[id(entry)], key)

    def _select_refinement(self, state: PhaseAwareState) -> LockAction | None:
        if not self.refinement_actor:
            return None
        from agent.cc.atcc.ppo import dot, policy_group_key, state_features, valid_actions

        actor_weights = self.refinement_actor.get("actor_weights", ())
        if not actor_weights:
            return None
        group_logits = dict(self.refinement_actor.get("group_logits", {}) or {}).get(
            policy_group_key(state, growth_conditioned=self.growth_conditioned_groups,
                             local_context=self.local_context_groups),
            (),
        )
        if self.refinement_actor.get("type") == "locally-supported-categorical-lock-masks":
            key = policy_group_key(state, growth_conditioned=self.growth_conditioned_groups,
                                   local_context=self.local_context_groups)
            allowed = self.refinement_actor["group_action_support"].get(key)
            if not allowed:
                return None  # Unknown/single-action groups retain original heads/fallback.
            features = state_features(state)
            head = self.calibrated_group_actions[key]
            action = min(allowed, key=lambda a: (
                -(dot(actor_weights[a], features) + group_logits[a]), a != head, a))
            return LockAction.from_encoded(action)
        features = state_features(state)
        if self.refinement_actor.get("type") == "factorized-bernoulli-lock-bits":
            if self.refinement_actor.get("preserve_actor_distribution", False):
                from agent.cc.atcc.ppo import sigmoid
                current = int(state.current_action) & 0xF
                valid = valid_actions(current)
                support = set(dict(self.refinement_actor.get("group_action_support", {})).get(
                    policy_group_key(state, growth_conditioned=self.growth_conditioned_groups,
                                     local_context=self.local_context_groups), ()))
                allowed = sorted(set(valid) & (support | {current})) if support else valid
                bit_probabilities = [
                    sigmoid(dot(row, features) +
                            (float(group_logits[i]) if len(group_logits) > i else 0.0))
                    for i, row in enumerate(actor_weights)
                ]
                probabilities = {}
                for candidate in allowed:
                    probability = 1.0
                    for i, bit in enumerate((1, 2, 4, 8)):
                        if not current & bit:
                            p = bit_probabilities[i]
                            probability *= p if candidate & bit else 1.0 - p
                    probabilities[candidate] = probability
                total = sum(probabilities.values())
                retain = float(self.refinement_actor.get("stay_probability", 0.0))
                for candidate in allowed:
                    p = probabilities[candidate] / total if total else 1.0 / len(allowed)
                    probabilities[candidate] = (1.0 - retain) * p + (
                        retain if candidate == current else 0.0)
                action = max(allowed, key=lambda candidate: probabilities[candidate])
                return LockAction.from_encoded(int(action)) if self.timing_actions else LockAction(LockClass(action))
            action = int(state.current_action) & 0xF
            for bit_index, bit in enumerate((1, 2, 4, 8)):
                if action & bit:
                    continue
                logit = dot(actor_weights[bit_index], features) + (
                    float(group_logits[bit_index])
                    if len(group_logits) > bit_index else 0.0
                )
                if logit > 0.0:
                    action |= bit
            return LockAction.from_encoded(int(action)) if self.timing_actions else LockAction(LockClass(action))
        action = max(
            valid_actions(state.current_action, timing_actions=self.timing_actions),
            key=lambda candidate: (
                dot(actor_weights[candidate], features)
                + (float(group_logits[candidate]) if len(group_logits) > candidate else 0.0),
                -candidate,
            ),
        )
        return LockAction.from_encoded(int(action))

    def _validate_supported_categorical_actor(self):
        actor = self.refinement_actor
        if (not self.local_context_groups or not self.calibrated_group_actions
                or actor.get("feature_count") != 37 or actor.get("phase_awareness") != self.phase_awareness
                or actor.get("tie_break") != "existing_calibrated_head_then_lowest_mask"
                or actor.get("state_source") != "observed_execution_only" or actor.get("workload_labels") is not False):
            raise ValueError("invalid supported categorical actor metadata")
        weights = actor.get("actor_weights", ())
        action_count = 32 if self.timing_actions else 16
        if len(weights) != action_count or any(len(row) != 37 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in row) for row in weights):
            raise ValueError("invalid supported categorical actor weights")
        support, logits, counts = (actor.get(name, {}) for name in (
            "group_action_support", "group_logits", "group_action_support_counts"))
        minimum = actor.get("minimum_action_support", 0)
        if (type(minimum) is not int or minimum < 2 or not support
                or set(support) != set(logits) or set(support) != set(counts)):
            raise ValueError("invalid supported categorical actor support")
        for key, actions in support.items():
            fields = dict(part.split("=", 1) for part in key.split("|"))
            current = int(fields.get("action", "-1"))
            if (key not in self.calibrated_group_actions or len(actions) < 2
                    or any(type(a) is not int or not 0 <= a < action_count
                           or (a & 0xF) | current != (a & 0xF) for a in actions)
                    or len(set(actions)) != len(actions) or self.calibrated_group_actions[key] not in actions
                    or set(counts[key]) != {str(a) for a in actions}
                    or any(type(n) is not int or n < minimum for n in counts[key].values())
                    or len(logits[key]) != action_count or any(type(v) not in (int, float) or not math.isfinite(v)
                                                      for v in logits[key])):
                raise ValueError("invalid supported categorical actor group")

    def _select_supported_categorical_exact(self, state, key):
        """Same supported argmax, avoiding repeated keys/copies and zero work.

        Zero shared weights need no feature transform on finite numeric states.
        Other actors skip exactly-zero feature products in original order.
        Unusual/nonfinite states take the unchanged reference evaluator.
        All payloads are read live, so no stale weight/logit snapshot is used.
        """
        actor = self.refinement_actor
        allowed = actor["group_action_support"].get(key)
        if not allowed:
            return None
        weights, logits = actor["actor_weights"], actor["group_logits"][key]
        head = self.calibrated_group_actions[key]
        if all(not any(weights[a]) for a in allowed):
            try:
                finite = all(type(v) in (int, float) and math.isfinite(v)
                             for k, v in vars(state).items() if k not in {"phase", "recent_conflict_kind"})
            except (OverflowError, TypeError):
                finite = False
            if finite:
                return LockAction.from_encoded(
                    min(allowed, key=lambda a: (-logits[a], a != head, a))
                )
            return self._select_refinement(state)
        from agent.cc.atcc.ppo import state_features
        features = state_features(state)
        if not all(math.isfinite(v) for v in features):
            return self._select_refinement(state)
        nonzero = tuple((i, v) for i, v in enumerate(features) if v != 0.)
        action = min(allowed, key=lambda a: (
            -(sum(weights[a][i] * v for i, v in nonzero) + logits[a]), a != head, a))
        return LockAction.from_encoded(int(action))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "cast-das-paper-atcc-policy",
            "version": 9 if self.refinement_actor.get("type") == "locally-supported-categorical-lock-masks" else (8 if self.protection_relevant_hotness else (7 if self.local_context_groups else (6 if self.growth_conditioned_groups else (5 if self.calibrated_group_actions else 4)))),
            **({"protection_relevant_hotness": True,
                "hotness_encoding": "eligible_observed_reads_plus_observed_writes"}
               if self.protection_relevant_hotness else {}),
            **({"local_context_groups": True,
                "group_key_encoding": "local_context_without_waiters_abort_conflict_abort"}
               if self.local_context_groups else {}),
            **({"growth_conditioned_groups": True,
                "growth_key_encoding": "observed_integer_growth_saturated_at_16"}
               if self.growth_conditioned_groups else {}),
            **({"calibrated_group_actions": self.calibrated_group_actions,
                "known_group_lookup": ("locally_supported_statewise_actor_before_group_average" if
                    self.refinement_actor.get("type") == "locally-supported-categorical-lock-masks" else
                    "exact_trained_complete_action_before_medoid_fallback")}
               if self.calibrated_group_actions else {}),
            "generation": self.generation,
            **({'state_projection': self.state_projection} if self.state_projection else {}),
            "phase_awareness": self.phase_awareness,
            **({"context_partitioned_lookup": True,
                "context_partition_encoding": "actor_policy_group_key",
                "unknown_context_lookup": "legacy_phase_current_mask_nearest"}
               if self.context_partitioned_lookup else {}),
            **({"remove_phase_index_proxy": True} if self.remove_phase_index_proxy else {}),
            "prebucketed_lookup": self.prebucketed_lookup,
            **({"packed_fallback_lookup": True} if self.packed_fallback_lookup else {}),
            **({"unanimous_fallback_lookup": True} if self.unanimous_fallback_lookup else {}),
            **({"exact_categorical_lookup": True} if self.exact_categorical_lookup else {}),
            "action_space": ("timed_lock_protection_mask_4bit" if self.timing_actions
                             else "lock_protection_mask_4bit"),
            "timing_actions": self.timing_actions,
            "action_dimensions": ["hot_read", "cold_read", "hot_write", "cold_write"],
            "priority_control": "transaction_manager_formula",
            "priority_is_policy_action": False,
            "state_source": "observed_execution_only",
            "future_access_plan_features": False,
            "compaction": "weighted_k_medoids_per_phase_current_action_action",
            "medoids_per_group": self.medoids_per_group,
            "selective_refinement": bool(self.refinement_actor),
            "refinement_actor": self.refinement_actor,
            "occ_cold_start_guard": self.occ_cold_start_guard,
            "entries": [{key: value for key, value in dataclasses.asdict(entry).items()
                         if key != "context_key" or self.context_partitioned_lookup}
                        for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CompiledPhasePolicy":
        if data.get('artifact_type') == 'cast-das-paper-observed-medoid-policy':
            raise ValueError('Observed medoid wrapper requires PaperObservationPolicy.from_dict; refusing an empty bare-table load')
        if data.get("context_partitioned_lookup", False) and (
                data.get("context_partition_encoding") != "actor_policy_group_key"
                or data.get("unknown_context_lookup") != "legacy_phase_current_mask_nearest"):
            raise ValueError("invalid context-partition metadata")
        statewise = dict(data.get("refinement_actor", {}) or {}).get("type") == "locally-supported-categorical-lock-masks"
        if statewise != (int(data.get("version", 0)) == 9):
            raise ValueError("version9 requires explicit locally supported categorical actor metadata")
        if data.get("protection_relevant_hotness", False):
            if (int(data.get("version", 0)) not in {8, 9} or data.get("hotness_encoding")
                    != "eligible_observed_reads_plus_observed_writes"):
                raise ValueError("invalid protection-relevant hotness metadata")
        elif int(data.get("version", 0)) == 8:
            raise ValueError("version8 requires protection-relevant hotness metadata")
        if int(data.get("version", 0)) == 7 and not data.get("local_context_groups", False):
            raise ValueError("version7 local-context artifact requires group metadata")
        fields = {field.name for field in dataclasses.fields(CompiledPolicyEntry)}
        return cls(
            (
                CompiledPolicyEntry(
                    **{key: value for key, value in dict(row).items() if key in fields}
                )
                for row in data.get("entries", [])
            ),
            generation=int(data.get("generation", 0) or 0),
            medoids_per_group=int(data.get("medoids_per_group", 1) or 1),
            refinement_actor=dict(data.get("refinement_actor", {}) or {}),
            occ_cold_start_guard=bool(data.get("occ_cold_start_guard", False)),
            phase_awareness=str(data.get("phase_awareness", "enabled")),
            remove_phase_index_proxy=bool(data.get("remove_phase_index_proxy", False)),
            prebucketed_lookup=bool(data.get("prebucketed_lookup", False)),
            packed_fallback_lookup=bool(data.get("packed_fallback_lookup", False)),
            unanimous_fallback_lookup=bool(data.get("unanimous_fallback_lookup", False)),
            protection_relevant_hotness=bool(data.get("protection_relevant_hotness", False)),
            exact_categorical_lookup=bool(data.get("exact_categorical_lookup", False)),
            timing_actions=bool(data.get("timing_actions", False)),
            calibrated_group_actions=dict(data.get("calibrated_group_actions", {}) or {}),
            growth_conditioned_groups=bool(data.get("growth_conditioned_groups", False)),
            local_context_groups=bool(data.get("local_context_groups", False)),
            context_partitioned_lookup=bool(data.get("context_partitioned_lookup", False)),
            state_projection=str(data.get('state_projection', '')),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompiledPhasePolicy":
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get('artifact_type')=='cast-das-observed-progress-partitioned-medoids':
            from .progress_medoid_policy import ProgressMedoidPolicy
            return ProgressMedoidPolicy(payload)
        if payload.get('artifact_type')=='cast-das-paper-async-refinement-bundle':
            from .selective_refinement import AsyncRefinementBundle
            return AsyncRefinementBundle.from_dict(payload)
        if payload.get('artifact_type')=='cast-das-paper-observed-medoid-policy':
            from .paper_observation_policy import PaperObservationPolicy
            return PaperObservationPolicy.from_dict(payload)
        return cls.from_dict(payload)


class StaticThresholdPhasePolicy:
    """Naive one-signal threshold baseline for the switching ablation.

    Unlike the learned policy, this baseline does not inspect transaction
    shape, reasoning cost, retry state, hotspot ratio, or queue state.  A
    fixed global conflict-abort threshold switches observed hot writes from
    OCC to pessimistic protection.  The action remains monotonic.
    """

    CONFLICT_ABORT_THRESHOLD = 0.20
    phase_awareness = 'enabled'
    monotone_actions = True
    requires_reviewer_contention_statistics = False

    def __init__(
        self,
        conflict_abort_threshold: float | None = None,
        protection_mask: int = int(LockClass.HOT_WRITE),
    ):
        threshold = (
            self.CONFLICT_ABORT_THRESHOLD
            if conflict_abort_threshold is None
            else float(conflict_abort_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("static conflict threshold must be in [0, 1]")
        self.conflict_abort_threshold = threshold
        self.protection_mask = LockClass(int(protection_mask) & 0xF)

    def select(self, state: PhaseAwareState) -> LockAction:
        current = LockClass(int(state.current_action) & 0xF)
        selected = current
        if (
            str(state.phase) in {"refine", "commit"}
            and float(state.global_conflict_abort_rate)
            >= self.conflict_abort_threshold
        ):
            selected |= self.protection_mask
        return LockAction(selected)


class HotRulePolicy:
    """Strong phase-blind baseline: protect hot reads/writes, leave cold accesses OCC.

    Risk is the maximum online abort/conflict rate. Threshold None always
    protects; threshold selection belongs to a separate calibration run.
    Existing protection is retained, matching the runtime's monotonic semantics.
    """

    def __init__(self, threshold: float | None = None):
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def select(self, state: PhaseAwareState) -> LockAction:
        # ``hotspot_access_ratio`` is zero exactly when the observed,
        # protection-relevant prefix has no hot access.  Keeping that prefix optimistic is
        # the defining part of this reviewer-requested baseline; without this
        # gate, the so-called hot rule degenerates into all-access protection.
        has_observed_hot_access = float(state.hotspot_access_ratio) > 0.0
        risk = max(state.global_abort_rate, state.global_conflict_abort_rate)
        mask = LockClass(int(state.current_action))
        if has_observed_hot_access and (self.threshold is None or risk >= self.threshold):
            mask |= LockClass.HOT_READ | LockClass.HOT_WRITE
        return LockAction(mask)


class LocalHotContentionPolicy:
    """Phase-blind reviewer baseline using a per-observed-hot-key threshold.

    The policy sees the largest *past* contention ratio among keys already
    accessed by the current attempt.  ``minimum_accesses`` prevents a single
    early conflict from classifying an object as persistently contentious.
    Protection remains monotonic, like every other paper policy.
    """

    def __init__(self, threshold: float = 0.20, minimum_accesses: int = 2):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if int(minimum_accesses) < 1:
            raise ValueError("minimum_accesses must be positive")
        self.threshold = float(threshold)
        self.minimum_accesses = int(minimum_accesses)

    def select(self, state: PhaseAwareState) -> LockAction:
        mask = LockClass(int(state.current_action))
        if (
            int(state.max_observed_recent_accesses) >= self.minimum_accesses
            and float(state.max_observed_hot_contention_rate) >= self.threshold
        ):
            mask |= LockClass.HOT_READ | LockClass.COLD_READ
            if int(state.write_set_size) > 0:
                mask |= LockClass.HOT_WRITE | LockClass.COLD_WRITE
        return LockAction(mask)


class LocalAccessFrequencyPolicy:
    """Phase-blind hand rule based only on per-key trailing-window frequency.

    The runtime supplies the largest one-second access count among objects
    already observed by this transaction.  A threshold of one implements the
    aggressive ``hot-always`` arm; a larger fixed threshold implements
    ``hot-threshold``.  The rule cannot inspect phase, progress, global abort
    rate, or a declared/future access footprint.
    """

    requires_reviewer_contention_statistics=False

    def __init__(self, minimum_accesses: int = 2):
        if int(minimum_accesses) < 1:
            raise ValueError("minimum_accesses must be positive")
        self.minimum_accesses = int(minimum_accesses)

    def select(self, state: PhaseAwareState) -> LockAction:
        mask = LockClass(int(state.current_action))
        if int(state.max_observed_recent_accesses) >= self.minimum_accesses:
            mask |= LockClass.HOT_READ | LockClass.COLD_READ
            if int(state.write_set_size) > 0:
                mask |= LockClass.HOT_WRITE | LockClass.COLD_WRITE
        return LockAction(mask)


class AtomicPolicyManager:
    def __init__(self, policy: object | None = None):
        self._lock = threading.RLock()
        self._policy = policy or CompiledPhasePolicy()

    def snapshot(self) -> object:
        with self._lock:
            return self._policy

    def install(self, policy: CompiledPhasePolicy, *, expected_generation: int | None = None) -> None:
        with self._lock:
            if expected_generation is not None and self._policy.generation!=expected_generation:
                raise ValueError('Stale offline training generation at atomic installation')
            if policy.generation <= self._policy.generation:
                raise ValueError("policy generation must increase")
            self._policy = policy


def bucket_size(value: int) -> int:
    value = max(0, int(value))
    if value == 0:
        return 0
    if value <= 2:
        return 1
    if value <= 8:
        return 2
    return 3


def bucket_ratio(value: float) -> int:
    value = max(0.0, min(1.0, float(value)))
    if value == 0:
        return 0
    if value < 0.25:
        return 1
    if value < 0.75:
        return 2
    return 3


def bucket_log(value: float) -> int:
    value = max(0.0, float(value))
    if value <= 0.0:
        return 0
    bucket = 1
    boundary = 1.0
    while value > boundary and bucket < 15:
        boundary *= 2.0
        bucket += 1
    return bucket


def _entry_key(entry: CompiledPolicyEntry) -> tuple[object, ...]:
    return (
        entry.phase,
        entry.inter_round_bucket,
        entry.read_set_bucket,
        entry.write_set_bucket,
        entry.read_growth_bucket,
        entry.write_growth_bucket,
        entry.overlap_bucket,
        entry.completed_rounds_bucket,
        entry.completed_operations_bucket,
        entry.recent_write_bucket,
        entry.hotspot_bucket,
        entry.blocked_bucket,
        entry.retry_bucket,
        entry.current_action,
        entry.recent_conflict_kind,
        entry.active_bucket,
        entry.waiter_bucket,
        entry.abort_bucket,
        entry.throughput_bucket,
        entry.average_latency_bucket,
        entry.tail_latency_bucket,
        entry.agent_task_throughput_bucket,
        entry.agent_task_average_latency_bucket,
        entry.agent_task_tail_latency_bucket,
        entry.conflict_abort_bucket,
        entry.background_throughput_bucket,
        entry.background_abort_bucket,
    )


def _state_key(state: PhaseAwareState) -> tuple[object, ...]:
    return (
        state.phase,
        bucket_log(state.inter_round_interval_ms),
        bucket_size(state.read_set_size),
        bucket_size(state.write_set_size),
        bucket_size(state.read_set_growth),
        bucket_size(state.write_set_growth),
        bucket_ratio(state.access_overlap_ratio),
        bucket_size(state.completed_rounds),
        bucket_size(state.completed_operations),
        bucket_ratio(state.recent_write_ratio),
        bucket_ratio(state.hotspot_access_ratio),
        bucket_log(state.blocked_time_ms),
        min(3, state.retry_count),
        int(state.current_action),
        normalize_conflict_kind(state.recent_conflict_kind),
        bucket_size(state.global_active_transactions),
        bucket_size(state.global_waiter_count),
        bucket_ratio(state.global_abort_rate),
        bucket_log(state.global_throughput),
        bucket_log(state.global_avg_latency_ms),
        bucket_log(state.global_tail_latency_ms),
        bucket_log(state.global_agent_task_throughput),
        bucket_log(state.global_agent_task_avg_latency_ms),
        bucket_log(state.global_agent_task_tail_latency_ms),
        bucket_ratio(state.global_conflict_abort_rate),
        bucket_log(state.global_background_throughput),
        bucket_ratio(state.global_background_abort_rate),
    )


def bucket_key_distance(entry_key, state_key) -> float:
    """Original distance exactly, with state buckets computed once per lookup.

    Conflict kind (index 14) is categorical. Priority remains excluded, as in
    the reference distance. This is not approximate nearest-neighbor lookup.
    """
    weights = (1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 4, 2, 1, 1, 2,
               .25, .25, .25, 1, .5, 1, 2, .5, 2)
    return (1000.0 if entry_key[0] != state_key[0] else 0.0) + sum(
        weight * (int(left != right) if index == 14 else abs(left - right))
        for index, (left, right, weight) in enumerate(zip(entry_key[1:], state_key[1:], weights), 1))


class _PackedFallbackView:
    """Exact small-bucket L1 scores in independent 16-bit integer lanes.

    Four times the reference distance is integral for compiled bucket keys.
    A posting is the sum of one-bit lane markers for candidates with a given
    coordinate. Multiplying it by that coordinate's cost and summing postings
    scores every candidate together using Python's arbitrary-precision ints.
    All numeric coordinates must be integral in [0, 15]. Thus the sum is below
    65536 and cannot carry into an adjacent lane. Nonstandard artifacts/states
    use the original scanner; no approximate nearest-neighbor search is used.

    Coordinates constant across candidates may be omitted from argmin only.
    The caller still computes the FULL original distance for refinement gates.
    Candidate order and support/action tie breaks are identical to min().
    """
    _WEIGHTS = (4000, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 8, 16, 8,
                4, 4, 8, 1, 1, 1, 4, 2, 4, 8, 2, 8)
    _CATEGORICAL = frozenset((0, 14))

    @classmethod
    def compatible(cls, key):
        return all(type(value) is int and 0 <= value <= 15
                   for i, value in enumerate(key) if i not in cls._CATEGORICAL)

    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        keys = tuple(_entry_key(e) for e in self.candidates)
        self.supported = bool(keys) and all(self.compatible(k) for k in keys)
        self.postings = ()
        if self.supported:
            columns = []
            for i, weight in enumerate(self._WEIGHTS):
                values = {}
                for lane, key in enumerate(keys):
                    values[key[i]] = values.get(key[i], 0) | (1 << (16 * lane))
                if len(values) > 1:
                    columns.append((i, weight, tuple(values.items()), i in self._CATEGORICAL))
            self.postings = tuple(columns)
        self.tie_breaks = tuple((-max(1, int(e.support_weight)), int(e.action)) for e in self.candidates)

    def nearest(self, state_key):
        if not self.supported or not self.compatible(state_key):
            return None
        scores = 0
        for index, weight, postings, categorical in self.postings:
            observed = state_key[index]
            for value, lanes in postings:
                delta = int(value != observed) if categorical else abs(value - observed)
                scores += weight * delta * lanes
        winner = None
        best = None
        for entry, tie in zip(self.candidates, self.tie_breaks):
            rank = (scores & 65535, *tie)
            if best is None or rank < best:
                best, winner = rank, entry
            scores >>= 16
        return winner


def should_keep_occ(state: PhaseAwareState) -> bool:
    """Avoid speculative locking until the runtime has observed conflict pressure."""
    low_conflict_rate = 0.05
    return (
        int(state.current_action) == 0
        and int(state.retry_count) == 0
        and normalize_conflict_kind(state.recent_conflict_kind) == "none"
        and int(state.global_waiter_count) == 0
        and float(state.global_abort_rate) <= low_conflict_rate
        and float(state.global_conflict_abort_rate) <= low_conflict_rate
        and float(state.global_background_abort_rate) <= low_conflict_rate
    )


def normalize_conflict_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized else "none"
