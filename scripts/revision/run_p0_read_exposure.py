#!/usr/bin/env python3
"""Small prospective matched-policy study for the CONTROLLED long-read window.

Never replace or pool with default CreditReview results. All variants see the
same moved read and unchanged SQL, reasoning/token and six-attempt budgets.
"""
import argparse
import dataclasses
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import policy_group_key, state_features, state_key
from agent.runtime.context import LockAction, LockClass
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy, StaticThresholdPhasePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict, policy_observation
from agent.workloads.credit_review import CreditReviewConfig
from scripts.analysis.audit_p0_claims import compare, matrix, validate_rows
from scripts.revision.run_p0_revision import ConstantProtectionPolicy, FrozenPolicy, MeteredPolicy, csv_write, digest, dump, summary
from scripts.unified_trace.run_credit_review_experiment import run_system

SEED_OFFSET = 0
EXPOSE_READ = True
ROUND_INTERVAL = False
COORDINATED_HELPER = True


class CommitAnchor:
    """Behavior collection only: optimistic before Commit, then fixed mask."""
    def __init__(self, mask):
        self.mask = mask

    def select(self, state):
        return LockAction(LockClass(int(state.current_action) | (self.mask if state.phase == "commit" else 0)))


class MeasurementMeter(MeteredPolicy):
    def reset(self):
        # The runner calls this after warmup. Do not count warmup decisions in
        # the measurement-attempt denominator or clear the warmed table cache.
        super().reset()


def load(path):
    policy = CompiledPhasePolicy.load(path)
    policy.occ_cold_start_guard = False
    policy.prebucketed_lookup = True
    return policy


def measure(path, policy, clients, seed, variant, rep=0, trajectory=None, exploration=None,
            seconds=2., warmup=.2):
    wrapped = MeasurementMeter(policy) if trajectory is None else policy
    row = run_system(config=CreditReviewConfig(company_count=1000, mutable_read_before_reasoning=EXPOSE_READ),
        clients=clients, seed=seed, repeat=rep, system="silo" if variant == "silo" else "paper-atcc",
        paper_policy=path, policy_override=wrapped, measure_seconds=seconds, warmup_seconds=warmup,
        max_attempts=6, performance_guards=False, policy_commit_batch=True, tracked_history_import=True,
        coordinated_hotspot_admission=COORDINATED_HELPER, round_interval_features=ROUND_INTERVAL,
        paper_priority="disabled" if variant == "no_priority" else "enabled",
        trajectory_output=trajectory, paper_exploration_seed=exploration, paper_exploration_epsilon=.2)
    row.update(variant=variant, compiled_cold_guard=False, workload_scope=(
        "controlled_long_read_window_not_paper_default" if EXPOSE_READ else "paper_default_read_window_timing_feature_pilot"))
    if isinstance(wrapped, MeasurementMeter):
        row.update(policy_calls=wrapped.calls, decisions_per_attempt=wrapped.calls / max(1, row["attempts"]),
            policy_lookup_mean_us=wrapped.ns / max(1, wrapped.calls) / 1000,
            phase_actions_json=json.dumps({f"{phase}:{action}": count
                for (phase, action), count in wrapped.phase_actions.items()}), warmup_policy_counts_excluded=True)
    assert not validate_rows([row])
    print(json.dumps({key: row[key] for key in ("variant", "clients", "seed", "repeat", "elapsed_task_tps",
        "agent_tps", "p99_latency_ms", "failed_tasks", "coordinated_gate_cycle_bypasses")}), flush=True)
    return row


def train_validate(args, references):
    assert not args.output.exists(), "fresh namespace required"
    paths, behavior = [], []
    specs = [(8, HotRulePolicy(.05)), (16, HotRulePolicy(.05))]
    specs += [(16, ConstantProtectionPolicy(mask)) for mask in (0, 5, 10, 15)]
    specs += [(16, CommitAnchor(mask)) for mask in (5, 15)]
    dump(args.output / "protocol.json", dict(created=time.time(), config=dataclasses.asdict(
        CreditReviewConfig(company_count=1000, mutable_read_before_reasoning=EXPOSE_READ)),
        scope=("controlled read-before-reasoning; separate from original default results" if EXPOSE_READ
               else "unchanged paper-default read window; optional timing-feature training pilot"),
        collection_specs=[dict(clients=c, policy=type(p).__name__, mask=getattr(p, "mask", None)) for c, p in specs],
        collection_seed_base=8050000 + SEED_OFFSET, exploration_seed_base=8051000 + SEED_OFFSET,
        training_seed=8060000 + SEED_OFFSET, epochs=4,
        reward_changed=False, ppo_advantage_normalization_changed=False, calibration_return_mode="global",
        validation_clients=[16, 32], validation_repeats=2, validation_seed_base=8070000 + SEED_OFFSET,
        holdout_clients=[32], holdout_repeats=3, holdout_seed_base=8090000 + SEED_OFFSET,
        validation_seconds=2., holdout_seconds=4., warmup_seconds=.2, max_attempts=6,
        performance_guards=False, policy_commit_batch=True, compiled_cold_guard=False,
        tracked_history_import=True, coordinated_hotspot_admission=COORDINATED_HELPER,
        round_interval_features=ROUND_INTERVAL, threshold=.05,
        threshold_source="prior independent calibration, unchanged; no controlled-test threshold tuning",
        cpu_affinity=sorted(os.sched_getaffinity(0)), numa_binding="node0 CPU and memory", real_llm=False,
        selection_rule="candidate/reference geomean >=1.05; each point >=.90; completion loss <=.01",
        reference_sha256={name: digest(path) for name, path in references.items()},
        reference_paths={name: str(path.resolve()) for name, path in references.items()},
        source_sha256={relative: digest(ROOT / relative) for relative in (
            "agent/workloads/credit_review.py", "agent/runtime/transaction.py", "agent/runtime/hotspot_admission.py",
            "agent/runtime/paper_policy.py", "agent/runtime/state_collector.py", "agent/cc/atcc/ppo.py",
            "scripts/unified_trace/run_credit_review_experiment.py", "scripts/revision/run_p0_read_exposure.py")}))
    started = time.monotonic()
    for index, (clients, policy) in enumerate(specs):
        path = args.output / f"training/behavior_{index}.json"
        row = measure(references["full"], policy, clients, 8050000 + SEED_OFFSET + index, f"behavior_{index}",
                      trajectory=path, exploration=8051000 + SEED_OFFSET + index, seconds=.75, warmup=0.)
        paths.append(path)
        behavior.append(row)
        csv_write(args.output / "training/collection.csv", behavior)
    collection_seconds = time.monotonic() - started
    candidates = {name: args.output / f"training/{name}.json" for name in ("full", "nophase")}
    for name, awareness in (("full", "enabled"), ("nophase", "disabled")):
        command = [sys.executable, str(ROOT / "scripts/train_paper_atcc.py"), "--output", str(candidates[name]),
            "--report", str(args.output / f"training/{name}_report.json"), "--generation", "3",
            "--seed", str(8060000 + SEED_OFFSET), "--epochs", "4", "--phase-awareness", awareness,
            "--calibration-return-mode", "global", "--disable-occ-cold-start-guard"]
        for path in paths:
            command += ["--trajectory", str(path)]
        subprocess.run(command, check=True)
    checks, policy = 0, load(candidates["nophase"])
    for path in paths:
        for transition in json.loads(path.read_text())["transitions"]:
            state = phase_aware_state_from_dict(transition["state"])
            outputs = []
            for phase in ("explore", "refine", "commit"):
                replaced = dataclasses.replace(state, phase=phase)
                observed = policy_observation(replaced, "disabled")
                outputs.append((state_key(observed), policy_group_key(observed), state_features(observed),
                                int(policy.select(replaced).protected)))
            assert outputs[0] == outputs[1] == outputs[2]
            checks += 1
    dump(args.output / "training/audit.json", dict(collection_seconds=collection_seconds,
        explicit_phase_checks=checks, common_input_sha256={path.name: digest(path) for path in paths},
        candidate_sha256={name: digest(path) for name, path in candidates.items()},
        implicit_progress_cues_remain=True, no_full_artifact_relabeling=True))
    validation = []
    for clients in (16, 32):
        for rep in range(2):
            order = ["reference", "candidate"]
            random.Random(8080000 + SEED_OFFSET + clients * 100 + rep).shuffle(order)
            for name in order:
                path = candidates["full"] if name == "candidate" else references["full"]
                row = measure(path, load(path), clients, 8070000 + SEED_OFFSET + clients * 100 + rep, name, rep)
                validation.append(row)
                csv_write(args.output / "validation/raw.csv", validation)
    csv_write(args.output / "validation/summary.csv", summary(validation))
    comparisons = compare(validation, "clients", "candidate", "reference")
    ratios = [row["tps_ratio"] for row in comparisons]
    geomean = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    completion_ok = all(statistics.fmean(row["task_completion_rate"] for row in validation
        if row["clients"] == c and row["variant"] == "candidate") >= statistics.fmean(
        row["task_completion_rate"] for row in validation if row["clients"] == c and row["variant"] == "reference") - .01
        for c in (16, 32))
    selected = geomean >= 1.05 and min(ratios) >= .90 and completion_ok
    dump(args.output / "validation/selection.json", dict(comparisons=comparisons, geomean=geomean,
        completion_ok=completion_ok, candidate_selected=selected, default_workload_deployment_accepted=False))
    dump(args.output / "train_validation_completed.json", dict(collection_rows=len(behavior), validation_rows=len(validation),
        candidate_selected=selected, controlled_scope=EXPOSE_READ, final_goal_accepted=False))


def holdout(args):
    assert not (args.output / "holdout").exists()
    selection = json.loads((args.output / "validation/selection.json").read_text())
    assert selection["candidate_selected"], "negative validation preserved; no automatic promotion"
    variants = ("silo", "static_threshold", "hotrule_always", "hotrule_threshold", "full", "nophase", "frozen_full", "no_priority")
    rows = []
    for rep in range(3):
        order = list(variants)
        random.Random(8103200 + SEED_OFFSET + rep).shuffle(order)
        for variant in order:
            path = args.output / f'training/{"nophase" if variant == "nophase" else "full"}.json'
            if variant == "static_threshold":
                policy = StaticThresholdPhasePolicy()
            elif variant.startswith("hotrule"):
                policy = HotRulePolicy(None if variant == "hotrule_always" else .05)
            else:
                policy = load(path)
                if variant == "frozen_full":
                    policy = FrozenPolicy(policy)
            row = measure(path, policy, 32, 8093200 + SEED_OFFSET + rep, variant, rep, seconds=4.)
            rows.append(row)
            csv_write(args.output / "holdout/raw.csv", rows)
            csv_write(args.output / "holdout/summary.csv", summary(rows))
    checked = matrix([{key: str(value) for key, value in row.items()} for row in rows], [32], variants, 3, "clients", 8090000 + SEED_OFFSET)
    assert checked["complete"], checked
    comparisons = [item for variant in variants if variant != "full"
                   for item in compare(rows, "clients", "full", variant)]
    dump(args.output / "holdout/comparisons.json", comparisons)
    dump(args.output / "holdout/completed.json", dict(matrix=checked, favorable_mechanism_claims_proven=False,
        default_paper_protection_proven=False))


def main():
    global SEED_OFFSET, EXPOSE_READ, ROUND_INTERVAL, COORDINATED_HELPER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--improvement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("train-validate", "holdout"), required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workload-scope", choices=("controlled-long-read", "paper-default"), default="controlled-long-read")
    parser.add_argument("--round-interval-features", action="store_true")
    parser.add_argument("--legacy-helper", action="store_true")
    args = parser.parse_args()
    assert args.seed_offset >= 0
    SEED_OFFSET, EXPOSE_READ = args.seed_offset, args.workload_scope == "controlled-long-read"
    ROUND_INTERVAL, COORDINATED_HELPER = args.round_interval_features, not args.legacy_helper
    assert platform.node().split(".")[0] == "node1"
    mode = json.loads((args.improvement / "validation/selection.json").read_text())["selected_mode"]
    references = {name: args.improvement / "training" / mode / f"{name}.json" for name in ("full", "nophase")}
    if args.stage == "train-validate":
        train_validate(args, references)
    else:
        protocol = json.loads((args.output / "protocol.json").read_text())
        assert protocol["validation_seed_base"] == 8070000 + SEED_OFFSET
        assert protocol["config"]["mutable_read_before_reasoning"] == EXPOSE_READ
        assert protocol.get("round_interval_features", False) == ROUND_INTERVAL
        assert protocol["coordinated_hotspot_admission"] == COORDINATED_HELPER
        holdout(args)


if __name__ == "__main__":
    main()
