#!/usr/bin/env python3
"""Small target-concurrency shared collection after accounting repair."""
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import policy_group_key, state_features, state_key
from agent.runtime.state_collector import phase_aware_state_from_dict, policy_observation
from agent.runtime.paper_policy import HotRulePolicy
from agent.runtime.context import LockAction, LockClass
from scripts.revision import run_p0_read_exposure as study
from scripts.revision.run_p0_revision import ConstantProtectionPolicy, csv_write, digest, dump, summary
from scripts.analysis.audit_p0_claims import compare


class ObservedReadAnchor:
    """Collection only: sample a read mask after an actual new Commit read."""
    def __init__(self, mask):
        self.mask = mask

    def select(self, state):
        mask = self.mask if state.phase == "commit" and state.read_set_growth == 1 else 0
        return LockAction(LockClass(int(state.current_action) | mask))


class MixtureBehaviorPolicy:
    """Collection only: original eight heads mixed at each observed decision.

    Marginal categorical propensities include both mixture and unchanged .2
    uniform exploration. No random base head is hidden behind a deterministic
    epsilon-greedy propensity. Every run uses the same mixture distribution.
    """
    def __init__(self, seed, epsilon=.2):
        self.seed, self.epsilon = int(seed), float(epsilon)
        if not 0. <= self.epsilon <= 1.:
            raise ValueError("epsilon must be in [0,1]")
        self.rng = random.Random(self.seed)
        self._lock = threading.RLock()
        self.members = [HotRulePolicy(.05), HotRulePolicy()]
        self.members += [ConstantProtectionPolicy(mask) for mask in (0, 5, 10, 15)]
        self.members += [study.CommitAnchor(mask) for mask in (5, 15)]

    def probabilities(self, state):
        current = int(state.current_action) & 15
        valid = [a for a in range(16) if a | current == a]
        values = [self.epsilon / len(valid) if a in valid else 0. for a in range(16)]
        for member in self.members:
            action = int(member.select(state).protected) | current
            assert action in valid
            values[action] += (1. - self.epsilon) / len(self.members)
        return tuple(values)

    def select_with_distribution(self, state):
        with self._lock:
            probabilities = self.probabilities(state)
            # Use the same categorical sampler as the existing PPO collector.
            from agent.cc.atcc.ppo import sample_action
            action = sample_action(self.rng, probabilities, state.current_action)
            return LockAction(LockClass(action)), probabilities

    def select(self, state):
        return self.select_with_distribution(state)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-models", type=Path, required=True)
    parser.add_argument("--commit-read-checkpoints", action="store_true")
    parser.add_argument("--growth-conditioned-groups", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--read-mask-support", action="store_true")
    parser.add_argument("--hot-read-only-support", action="store_true")
    parser.add_argument("--consistent-hotness-classification", action="store_true")
    parser.add_argument("--helper-externality-accounting", action="store_true")
    parser.add_argument("--mixed-behavior-collection", action="store_true")
    parser.add_argument("--observed-write-read-lock-elision", action="store_true")
    args = parser.parse_args()
    assert not args.read_mask_support or (args.commit_read_checkpoints and args.growth_conditioned_groups)
    assert not args.hot_read_only_support or args.read_mask_support
    assert not args.helper_externality_accounting or args.consistent_hotness_classification
    assert not args.mixed_behavior_collection or (args.helper_externality_accounting and not args.read_mask_support)
    assert not args.observed_write_read_lock_elision or (
        args.consistent_hotness_classification and args.helper_externality_accounting and args.mixed_behavior_collection)
    assert args.seed_offset >= 0
    assert platform.node().split(".")[0] == "node1"
    assert not args.output.exists()
    study.EXPOSE_READ, study.COORDINATED_HELPER, study.ROUND_INTERVAL = True, True, False
    if args.commit_read_checkpoints or args.consistent_hotness_classification:
        original_run_system = study.run_system
        def run_system(**kwargs):
            kwargs["commit_read_checkpoints"] = args.commit_read_checkpoints
            kwargs["consistent_hotness_classification"] = args.consistent_hotness_classification
            kwargs["helper_externality_accounting"] = args.helper_externality_accounting
            kwargs["observed_write_read_lock_elision"] = args.observed_write_read_lock_elision
            if args.mixed_behavior_collection and kwargs.get("trajectory_output") is not None:
                # The mixture already records the exact .2-exploration marginal.
                kwargs["paper_exploration_seed"] = None
            return original_run_system(**kwargs)
        study.run_system = run_system
    offset = args.seed_offset
    binaries = list(ROOT.glob("cast_core*.so"))
    assert len(binaries) == 1
    specs = [HotRulePolicy(.05), HotRulePolicy()]
    specs += [ConstantProtectionPolicy(mask) for mask in (0, 5, 10, 15)]
    specs += [study.CommitAnchor(mask) for mask in (5, 15)]
    if args.read_mask_support:
        # Same eight short runs and original 16-action space. Supply direct
        # support for read-only and hot-write/read masks absent from prior
        # globally trusted {0,5,10,15}; do not synthesize unsupported actions.
        specs = [HotRulePolicy()] + [ConstantProtectionPolicy(mask) for mask in (0, 2, 3, 7, 15)]
        specs += [ObservedReadAnchor(mask) for mask in (2, 3)]
    if args.hot_read_only_support:
        # The measured Commit/new-read/current=0 subset had only TWO mask1
        # samples, so no local group can meet minimum support8 for the action
        # that protects hot observed reads without adding early write locks.
        # Keep eight runs: substitute two anchors, not additional compute.
        specs = [HotRulePolicy()] + [ConstantProtectionPolicy(mask) for mask in (0, 1, 2, 3, 15)]
        specs += [ObservedReadAnchor(mask) for mask in (1, 3)]
    if args.mixed_behavior_collection:
        specs = [MixtureBehaviorPolicy(8401000 + offset + i) for i in range(8)]
    protocol = dict(scope="matched paid-helper target-concurrency training; controlled long read; not paper-default",
        source_root=str(ROOT), source_sha256={str(p.relative_to(ROOT)): digest(p)
            for p in (ROOT / "agent").rglob("*.py")},
        script_sha256={name: digest(ROOT / name) for name in (
            "scripts/revision/run_p0_paid_training.py", "scripts/revision/run_p0_read_exposure.py",
            "scripts/revision/run_p0_revision.py", "scripts/unified_trace/run_credit_review_experiment.py",
            "scripts/train_paper_atcc.py")}, native_binary_sha256=digest(binaries[0]),
        references=str(args.reference_models.resolve()),
        reference_sha256={name: digest(args.reference_models / f"{name}.json") for name in ("full", "nophase")},
        collection_clients=32, collection_seconds=.75, collection_warmup=0., collection_seed_base=8400000 + offset,
        exploration_seed_base=8401000 + offset, exploration_epsilon=.2,
        specs=[dict(policy=type(p).__name__, mask=getattr(p, "mask", None), threshold=getattr(p, "threshold", None)) for p in specs],
        training_seed=8410000 + offset, epochs=4, calibration_return_mode="global", exact_trained_groups=True,
        commit_read_checkpoints=args.commit_read_checkpoints, growth_conditioned_groups=args.growth_conditioned_groups,
        read_mask_behavior_support=args.read_mask_support,
        hot_read_only_behavior_support=args.hot_read_only_support,
        consistent_hotness_classification=args.consistent_hotness_classification,
        helper_externality_accounting=args.helper_externality_accounting,
        observed_write_read_lock_elision=args.observed_write_read_lock_elision,
        **({"prevalidation_helper_ledger_required": True,
            "helper_ledger_audit_source_sha256": digest(ROOT / "scripts/analysis/audit_p0_helper_ledger.py")}
           if args.helper_externality_accounting else {}),
        mixed_behavior_collection=args.mixed_behavior_collection,
        **({"mixture_seeds": [p.seed for p in specs],
            "mixture_members": ["HotThreshold.05", "HotAlways", "Constant0", "Constant5", "Constant10", "Constant15", "Commit5", "Commit15"],
            "mixture_scope": "per observed decision, collection only; exact marginal with .2 uniform exploration"}
           if args.mixed_behavior_collection else {}),
        reward_config_changed=False, model_architecture_changed=False, action_space_changed=False,
        priority_formula_changed=False, sql_reasoning_token_retry_budget_changed=False,
        company_count=1000, max_attempts=6, read_exposure=True, coordinated_hotspot_admission=True,
        round_interval_features=False, performance_guards=False, tracked_history_import=True,
        policy_commit_batch=True, compiled_cold_guard=False,
        validation_seed_base=8420000 + offset, validation_clients=[16, 32], validation_repeats=2,
        validation_seconds=2., validation_warmup=.2, order_seed_base=8430000 + offset,
        selection_rule="primary geomean >=1.05, each >=.90, window geomean >=1, completion loss <=.01" +
            ("; strongest-rule main and window geomeans >=1.03" if args.commit_read_checkpoints or args.consistent_hotness_classification else ""),
        cpu_affinity=sorted(os.sched_getaffinity(0)), numa_binding="node0 CPU and memory", real_llm=False)
    dump(args.output / "protocol.json", protocol)
    collection, paths = [], []
    start = time.monotonic()
    for index, policy in enumerate(specs):
        path = args.output / f"training/behavior_{index}.json"
        row = study.measure(args.reference_models / "full.json", policy, 32, 8400000 + offset + index,
            f"behavior_{index}", trajectory=path, exploration=8401000 + offset + index, seconds=.75, warmup=0.)
        collection.append(row)
        paths.append(path)
        csv_write(args.output / "training/collection.csv", collection)
    collection_seconds = time.monotonic() - start
    models = {name: args.output / f"training/{name}.json" for name in ("full", "nophase")}
    training_started = time.monotonic()
    for name, awareness in (("full", "enabled"), ("nophase", "disabled")):
        command = [sys.executable, str(ROOT / "scripts/train_paper_atcc.py"),
            "--output", str(models[name]), "--report", str(args.output / f"training/{name}_report.json"),
            "--generation", "4", "--seed", str(8410000 + offset), "--epochs", "4", "--phase-awareness", awareness,
            "--calibration-return-mode", "global", "--disable-occ-cold-start-guard", "--preserve-calibrated-groups"]
        if args.growth_conditioned_groups:
            command += ["--growth-conditioned-groups"]
        for path in paths:
            command += ["--trajectory", str(path)]
        subprocess.run(command, cwd=ROOT, check=True)
    no_phase = study.load(models["nophase"])
    checks = 0
    for path in paths:
        for transition in json.loads(path.read_text())["transitions"]:
            state = phase_aware_state_from_dict(transition["state"])
            outputs = []
            for phase in ("explore", "refine", "commit"):
                observed = policy_observation(dataclasses.replace(state, phase=phase), "disabled")
                outputs.append((state_key(observed), policy_group_key(observed,
                    growth_conditioned=args.growth_conditioned_groups), state_features(observed),
                    int(no_phase.select(observed).protected)))
            assert outputs[0] == outputs[1] == outputs[2]
            checks += 1
    dump(args.output / "training/audit.json", dict(collection_seconds=collection_seconds,
        training_and_phase_audit_seconds=time.monotonic() - training_started,
        explicit_phase_checks=checks, common_input_sha256={p.name: digest(p) for p in paths},
        model_sha256={name: digest(path) for name, path in models.items()},
        independent_training_not_relabeling=True, implicit_progress_cues_retained=True))
    if args.helper_externality_accounting:
        # Never spend performance validation on a candidate with inconsistent
        # actual cost components. This only strengthens data acceptance; the
        # original performance / completion / strongest-rule gates remain.
        subprocess.run([sys.executable, str(ROOT / "scripts/analysis/audit_p0_helper_ledger.py"),
            "--training", str(args.output / "training"),
            "--output", str(args.output / "training/helper_ledger_gate.json")], check=True)
    rows = []
    for clients in (16, 32):
        for rep in range(2):
            order = ["candidate", "reference"] + (["hotrule_always"] if args.commit_read_checkpoints or args.consistent_hotness_classification else [])
            random.Random(8430000 + offset + clients * 100 + rep).shuffle(order)
            for variant in order:
                path = models["full"] if variant == "candidate" else args.reference_models / "full.json"
                policy = HotRulePolicy() if variant == "hotrule_always" else study.load(path)
                row = study.measure(path, policy, clients, 8420000 + offset + clients * 100 + rep,
                    variant, rep, seconds=2., warmup=.2)
                row.update(model_sha256=digest(path), exact_group_head_active=variant != "hotrule_always",
                    growth_conditioned_groups=args.growth_conditioned_groups and variant == "candidate")
                rows.append(row)
                csv_write(args.output / "validation/raw.csv", rows)
    comparisons = compare(rows, "clients", "candidate", "reference")
    ratios, window_ratios = [r["tps_ratio"] for r in comparisons], [r["window_tps_ratio"] for r in comparisons]
    geo = math.exp(statistics.fmean(math.log(x) for x in ratios)) if min(ratios) > 0 else 0.
    window_geo = math.exp(statistics.fmean(math.log(x) for x in window_ratios)) if min(window_ratios) > 0 else 0.
    completion = all(statistics.fmean(r["task_completion_rate"] for r in rows if r["clients"] == c and r["variant"] == "candidate")
        >= statistics.fmean(r["task_completion_rate"] for r in rows if r["clients"] == c and r["variant"] == "reference") - .01
        for c in (16, 32))
    selected = geo >= 1.05 and min(ratios) >= .90 and window_geo >= 1. and completion
    strong_comparisons, strong_geo, strong_window_geo = [], None, None
    if args.commit_read_checkpoints or args.consistent_hotness_classification:
        strong_comparisons = compare(rows, "clients", "candidate", "hotrule_always")
        strong_geo = math.exp(statistics.fmean(math.log(r["tps_ratio"]) for r in strong_comparisons))
        strong_window_geo = math.exp(statistics.fmean(math.log(r["window_tps_ratio"]) for r in strong_comparisons))
        strong_completion = all(statistics.fmean(r["task_completion_rate"] for r in rows if r["clients"] == c and r["variant"] == "candidate")
            >= statistics.fmean(r["task_completion_rate"] for r in rows if r["clients"] == c and r["variant"] == "hotrule_always") - .01
            for c in (16, 32))
        completion = completion and strong_completion
        selected = selected and strong_geo >= 1.03 and strong_window_geo >= 1.03 and strong_completion
    csv_write(args.output / "validation/summary.csv", summary(rows))
    dump(args.output / "validation/selection.json", dict(comparisons=comparisons, geomean=geo,
        window_geomean=window_geo, completion_ok=completion, selected=selected, default_deployment_accepted=False))
    if args.commit_read_checkpoints or args.consistent_hotness_classification:
        dump(args.output / "validation/strongest_rule.json", dict(comparisons=strong_comparisons,
            main_geomean=strong_geo, window_geomean=strong_window_geo, selected=selected))
    dump(args.output / "completed.json", dict(collection_rows=8, validation_rows=len(rows),
        candidate_selected=selected, favorable_mechanisms_proven=False, original_paper_protection_proven=False, goal_completed=False))
    print(json.dumps(dict(selected=selected, geomean=geo, window_geomean=window_geo)), flush=True)


if __name__ == "__main__":
    main()
