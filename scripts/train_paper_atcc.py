#!/usr/bin/env python3
"""Train and compile the paper ATCC policy from runtime trajectories."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cc.atcc.ppo import (
    DiscretePPOPolicy,
    DiscretePPOTrainer,
    PAPER_MEDOIDS_PER_GROUP,
    PPOConfig,
    audit_policy,
    state_key,
    policy_transitions,
)
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=810104)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--group-learning-rate", type=float, default=0.03)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--min-group-samples", type=int, default=16)
    parser.add_argument("--min-group-actions", type=int, default=2)
    parser.add_argument("--calibration-return-mode", choices=("per-source-run", "global"),
                        default="per-source-run")
    parser.add_argument("--system-weight", type=float, default=10.0)
    parser.add_argument("--lock-weight", type=float, default=5.0)
    parser.add_argument("--abort-weight", type=float, default=80.0)
    parser.add_argument("--coordinated-reward-weight", type=float)
    parser.add_argument("--coordinated-original-weight", type=float, default=100.0)
    parser.add_argument("--refinement-distance-threshold", type=float)
    parser.add_argument(
        "--medoids-per-group",
        type=int,
        default=PAPER_MEDOIDS_PER_GROUP,
        help="Representative state keys retained per phase/action group (paper default: 4).",
    )
    parser.add_argument("--disable-occ-cold-start-guard", action="store_true")
    parser.add_argument("--phase-awareness", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--remove-phase-index-proxy", action="store_true",
                        help="Remove completed_rounds, which counts phase changes; retain genuine progress features.")
    parser.add_argument("--preserve-calibrated-groups", action="store_true",
        help="Preserve trained group decisions exactly before nearest-medoid fallback.")
    parser.add_argument("--growth-conditioned-groups", action="store_true",
        help="Condition complete-action groups on already-observed footprint growth (opt-in).")
    parser.add_argument("--strict-local-action-support", action="store_true",
        help="Require local action support before expansion; otherwise retain the current mask (opt-in).")
    parser.add_argument("--pooled-sparse-action-support", action="store_true",
        help="Pool only volatile global buckets for the original sparse deployment fallback (opt-in).")
    parser.add_argument("--ppo-sparse-fallback", action="store_true",
        help="Use post-PPO probabilities only for original sparse fallback allowed masks (opt-in).")
    parser.add_argument("--ppo-supported-projection", action="store_true",
        help="Use post-PPO probabilities with original qualified and sparse allowed masks (opt-in).")
    parser.add_argument("--categorical-actor", action="store_true",
        help="Represent correlated full-mask behavior with categorical PPO (opt-in).")
    parser.add_argument("--local-qualified-action-support", action="store_true",
        help="Let original minimum-qualified local masks override cross-state winner veto (opt-in).")
    parser.add_argument("--local-context-groups", action="store_true",
        help="Pool three volatile global buckets throughout group training and deployment (opt-in).")
    args = parser.parse_args()
    if args.remove_phase_index_proxy and args.phase_awareness != "disabled":
        parser.error("--remove-phase-index-proxy requires --phase-awareness disabled")
    if args.local_context_groups and (not args.preserve_calibrated_groups or args.pooled_sparse_action_support):
        raise SystemExit("local-context groups require preserved heads and exclude redundant sparse pooling")
    if args.growth_conditioned_groups and not args.preserve_calibrated_groups:
        raise SystemExit("--growth-conditioned-groups requires --preserve-calibrated-groups")
    if args.strict_local_action_support and not args.preserve_calibrated_groups:
        raise SystemExit("--strict-local-action-support requires --preserve-calibrated-groups")
    if args.pooled_sparse_action_support and not args.preserve_calibrated_groups:
        raise SystemExit("--pooled-sparse-action-support requires --preserve-calibrated-groups")
    if args.pooled_sparse_action_support and args.strict_local_action_support:
        raise SystemExit("pooled-sparse and strict-local are exclusive projection modes")
    if args.ppo_sparse_fallback and (not args.preserve_calibrated_groups or args.pooled_sparse_action_support or args.strict_local_action_support):
        raise SystemExit("--ppo-sparse-fallback requires preserved heads and is exclusive with other sparse modes")
    if args.ppo_supported_projection and (not args.preserve_calibrated_groups or args.ppo_sparse_fallback or args.pooled_sparse_action_support or args.strict_local_action_support):
        raise SystemExit("--ppo-supported-projection requires preserved heads and is exclusive with other modes")
    if args.categorical_actor and (not args.ppo_supported_projection or args.refinement_distance_threshold is not None):
        raise SystemExit("--categorical-actor requires --ppo-supported-projection and no bit refinement")
    if args.local_qualified_action_support and not (args.categorical_actor and args.ppo_supported_projection):
        raise SystemExit("--local-qualified-action-support requires categorical supported PPO")

    if args.coordinated_reward_weight is not None:
        raise SystemExit(
            "--coordinated-reward-weight is incompatible with the paper reward formula"
        )
    if args.medoids_per_group <= 0:
        raise SystemExit("--medoids-per-group must be positive")
    transitions = []
    for path in args.trajectory:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("transitions", payload) if isinstance(payload, dict) else payload
        for row in rows:
            source_id = str(row.get("source_id") or path.stem)
            transitions.append(policy_transition_from_dict(row, source_id=source_id))
    if args.remove_phase_index_proxy:
        transitions = [dataclasses.replace(t,
            state=dataclasses.replace(t.state, completed_rounds=0),
            next_state=(dataclasses.replace(t.next_state, completed_rounds=0)
                        if t.next_state is not None else None)) for t in transitions]
    transitions, reward_report = apply_paper_rewards(
        transitions,
        PaperRewardConfig(
            abort_weight=float(args.abort_weight),
            lock_weight=float(args.lock_weight),
            system_weight=float(args.system_weight),
        ),
    )
    started = time.perf_counter()
    policy = DiscretePPOPolicy(seed=args.seed, phase_awareness=args.phase_awareness,
        growth_conditioned_groups=args.growth_conditioned_groups,
        strict_local_action_support=args.strict_local_action_support,
        pooled_sparse_action_support=args.pooled_sparse_action_support,
        ppo_sparse_fallback=args.ppo_sparse_fallback,
        ppo_supported_projection=args.ppo_supported_projection,
        categorical_actor=args.categorical_actor,
        local_qualified_action_support=args.local_qualified_action_support,
        local_context_groups=args.local_context_groups)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        group_learning_rate=args.group_learning_rate,
        clip_ratio=args.clip_ratio,
        discount=args.discount,
        entropy_weight=args.entropy_weight,
        epochs=args.epochs,
        min_group_samples=args.min_group_samples,
        min_group_actions=args.min_group_actions,
        calibration_return_mode=args.calibration_return_mode,
    )
    training = DiscretePPOTrainer(config).train(policy, transitions)
    compiled = policy.compile(
        generation=args.generation,
        medoids_per_group=args.medoids_per_group,
        refinement_distance_threshold=args.refinement_distance_threshold,
        occ_cold_start_guard=not args.disable_occ_cold_start_guard,
        preserve_calibrated_groups=args.preserve_calibrated_groups,
    )
    compiled.remove_phase_index_proxy = args.remove_phase_index_proxy
    policy_audit = audit_policy(policy, compiled, transitions, discount=config.discount)
    coverage = coverage_report(policy_transitions(transitions, args.phase_awareness))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compiled.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(
        json.dumps(
            {
                "artifact_type": "cast-das-paper-atcc-train-report",
                "algorithm": "discrete-clipped-ppo",
                "config": dataclasses.asdict(config),
                "seed": args.seed,
                "phase_awareness": args.phase_awareness,
                **({"remove_phase_index_proxy": True,
                    "removed_state_fields": ["phase", "completed_rounds"]}
                   if args.remove_phase_index_proxy else {}),
                "preserve_calibrated_groups": args.preserve_calibrated_groups,
                **({"growth_conditioned_groups": True} if args.growth_conditioned_groups else {}),
                **({"strict_local_action_support": True} if args.strict_local_action_support else {}),
                **({"pooled_sparse_action_support": True} if args.pooled_sparse_action_support else {}),
                **({"ppo_sparse_fallback": True} if args.ppo_sparse_fallback else {}),
                **({"ppo_supported_projection": True} if args.ppo_supported_projection else {}),
                **({"categorical_actor": True} if args.categorical_actor else {}),
                **({"local_qualified_action_support": True} if args.local_qualified_action_support else {}),
                **({"local_context_groups": True,
                    "removed_group_fields": ["waiters", "abort", "conflict_abort"],
                    "continuous_state_features_retained": True,
                    "coverage_uses_original_group_encoding": True} if args.local_context_groups else {}),
                "training_compile_audit_seconds": time.perf_counter() - started,
                "trajectory_count": len({(t.source_id, t.txn_id) for t in transitions}),
                "generation": args.generation,
                "inputs": [str(path) for path in args.trajectory],
                "training": training,
                "policy_audit": policy_audit,
                "coverage": coverage,
                "compiled_entries": len(compiled.entries),
                "medoids_per_group": compiled.medoids_per_group,
                "selective_refinement": bool(compiled.refinement_actor),
                "refinement_distance_threshold": args.refinement_distance_threshold,
                "occ_cold_start_guard": compiled.occ_cold_start_guard,
                "reward": reward_report,
                "action_space": "lock_protection_mask_4bit",
                "priority_control": "transaction_manager_formula",
                "priority_is_policy_action": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def coordinated_shared_scores(payload) -> dict[tuple[str, str], float]:
    if not isinstance(payload, dict):
        return {}
    scores = {}
    for row in payload.get("source_runs", []):
        path = row.get("path", [])
        if len(path) != 2:
            continue
        path_id = f"p{int(path[0])}_{int(path[1])}"
        scores[(str(row.get("config_id", "")), path_id)] = float(
            row.get("shared_system_score", 0.0)
        )
    return scores


def coverage_report(transitions: list[PolicyTransition]) -> dict[str, object]:
    action_counts: dict[str, int] = {}
    phase_action_counts: dict[str, dict[str, int]] = {}
    probabilities = []
    states = set()
    state_actions = set()
    for transition in transitions:
        action = str(int(transition.action))
        phase = transition.state.phase
        action_counts[action] = action_counts.get(action, 0) + 1
        phase_counts = phase_action_counts.setdefault(phase, {})
        phase_counts[action] = phase_counts.get(action, 0) + 1
        probabilities.append(float(transition.behavior_probability))
        key = state_key(transition.state)
        states.add(key)
        state_actions.add((key, action))
    return {
        "transition_count": len(transitions),
        "unique_states": len(states),
        "unique_state_action_pairs": len(state_actions),
        "actions_observed": sorted(int(action) for action in action_counts),
        "action_counts": action_counts,
        "phase_action_counts": phase_action_counts,
        "behavior_probability": {
            "min": min(probabilities) if probabilities else 0.0,
            "max": max(probabilities) if probabilities else 0.0,
            "mean": sum(probabilities) / len(probabilities) if probabilities else 0.0,
            "all_valid": all(0.0 < value <= 1.0 for value in probabilities),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
