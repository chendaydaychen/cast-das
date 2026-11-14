"""Training-only compilation of legal current-mask variants for a raw actor."""
from collections import defaultdict
import dataclasses
import itertools

from agent.runtime.paper_policy import CompiledPhasePolicy
from .ppo import compiled_entry, valid_actions, weighted_k_medoids


def compile_mask_closed_policy(actor, *, generation, medoids_per_group=4,
                               all_current_masks=False, remove_phase_index_proxy=False,
                               context_partitioned_lookup=False, schema_context_closure=False,
                               mixed_access_closure=False, zero_cost_prefix_closure=False):
    """Label mask variants without modifying the actor or using evaluation states.

    This closes the finite current-mask dimension, not the continuous state space.
    Model-labelled variants are not additional measured behavior/reward samples.
    """
    from agent.runtime.joint_mask_policy import JointMaskActor
    if getattr(actor, 'remove_phase_features', False):
        raise ValueError('Phase-feature group ablation requires its serialized coefficient or exact feature table; legacy medoid schema does not carry this mask')
    if (actor.categorical_actor and not isinstance(actor, JointMaskActor)) or actor.timing_actions:
        raise ValueError("mask closure supports factorized or JointMaskActor 16-action policies only")
    if medoids_per_group < 1:
        raise ValueError("positive medoid budget required")
    if schema_context_closure and not context_partitioned_lookup:
        raise ValueError("schema context closure requires context-partitioned lookup")
    if mixed_access_closure and not schema_context_closure:
        raise ValueError("mixed access closure requires schema context closure")
    if zero_cost_prefix_closure and not mixed_access_closure:
        raise ValueError("zero-cost prefix closure requires mixed access closure")
    with actor._lock:
        prototypes = dict(actor.prototypes)
        logits = actor.actor_group_logits.copy()
        categorical_logits = (actor.categorical_group_logits.copy()
                              if actor.categorical_group_logits is not None else None)
        grouped = defaultdict(list)
        seen = set()
        try:
            for _, original in sorted(prototypes.items()):
                variants = [original]
                if schema_context_closure:
                    # Fixed schema representatives, not evaluation states or outcomes.
                    for hot, reads in itertools.product((0., .125, .5, 1.), (0, 1, 4, 9)):
                        if reads < original.read_set_growth or (reads == 0 and original.write_set_size == 0):
                            continue
                        variants.append(dataclasses.replace(original, hotspot_access_ratio=hot, read_set_size=reads))
                    if mixed_access_closure:
                        # Generic cheap prefixes, labelled by the frozen actor, not a test trace.
                        for hot, reads, writes in itertools.product((0., .125, .5, 1.), (1, 4, 9), (1, 4)):
                            if reads < original.read_set_growth or writes < original.write_set_growth:
                                continue
                            variants.append(dataclasses.replace(original,
                                hotspot_access_ratio=hot, read_set_size=reads, write_set_size=writes,
                                inter_round_interval_ms=1., observed_restart_cost_ms=1.,
                                blocked_time_ms=0., completed_operations=reads+writes,
                                recent_write_ratio=writes/(reads+writes)))
                if zero_cost_prefix_closure:
                    # Before any paid external interval, restart cost is exactly
                    # zero. Cover this schema boundary from training prototypes,
                    # rather than borrowing a positive-cost nearest context.
                    variants += [dataclasses.replace(variant, observed_restart_cost_ms=0.,
                        inter_round_interval_ms=interval, blocked_time_ms=0.)
                        for variant in tuple(variants) for interval in (0., 1.)]
                for variant in variants:
                    for current in range(16) if all_current_masks else valid_actions(variant.current_action):
                        state = dataclasses.replace(variant, current_action=current)
                        if getattr(actor, 'state_projection', ''):
                            state = actor.observation(state)
                        key = actor.prototype_key(state)
                        if key in seen:
                            continue
                        seen.add(key)
                        probabilities = actor.probabilities(state)
                        action = max(valid_actions(current), key=lambda a: probabilities[a])
                        entry = compiled_entry(state, action)
                        context = actor.group_key(state) if context_partitioned_lookup else ""
                        if context_partitioned_lookup:
                            entry = dataclasses.replace(entry, context_key=context)
                        grouped[(state.phase, current, action, context)].append((entry, state, 1))
        finally:
            actor.prototypes.clear()
            actor.prototypes.update(prototypes)
            actor.actor_group_logits.clear()
            actor.actor_group_logits.update(logits)
            if categorical_logits is not None:
                actor.categorical_group_logits.clear()
                actor.categorical_group_logits.update(categorical_logits)
        entries = [entry for _, rows in sorted(grouped.items())
                   for entry in weighted_k_medoids(rows, medoids_per_group)]
    return CompiledPhasePolicy(entries, generation=generation,
                               medoids_per_group=medoids_per_group,
                               phase_awareness=actor.phase_awareness,
                               growth_conditioned_groups=actor.growth_conditioned_groups if context_partitioned_lookup else False,
                               local_context_groups=actor.local_context_groups if context_partitioned_lookup else False,
                               context_partitioned_lookup=context_partitioned_lookup,
                               state_projection=getattr(actor, 'state_projection', ''),
                               remove_phase_index_proxy=remove_phase_index_proxy), dict(
        source_training_prototypes=len(prototypes), model_labelled_mask_variants=len(seen),
        compiled_entries=len(entries), medoids_per_group=medoids_per_group,
        all_current_masks=bool(all_current_masks),
        actor_fallback=False, deployment_overrides=False,
        context_partitioned_lookup=bool(context_partitioned_lookup),
        schema_context_closure=bool(schema_context_closure),
        mixed_access_closure=bool(mixed_access_closure),
        zero_cost_prefix_closure=bool(zero_cost_prefix_closure),
        schema_representatives={"hotspot_access_ratio": [0., .125, .5, 1.],
                                "read_set_size": [0, 1, 4, 9],
                                "cheap_mixed_prefix": {"write_set_size": [1, 4],
                                    "interval_ms": 1., "restart_cost_ms": 1.}
                                    if mixed_access_closure else {}} if schema_context_closure else {},
        labels="original raw actor greedy actions; synthetic legal current masks, not measured outcomes",
        coverage="training prototypes with fixed schema context/mask variants; no evaluation inputs or continuous-state guarantee"
            if schema_context_closure else "current-mask closure of training prototypes only; no evaluation inputs or continuous-state guarantee")
