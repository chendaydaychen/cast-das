#!/usr/bin/env python3
"""Recompute controlled validation and phase invariance from raw artifacts."""
import argparse
import dataclasses
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import policy_group_key, state_features, state_key
from agent.runtime.state_collector import phase_aware_state_from_dict, policy_observation
from scripts.analysis.audit_p0_claims import compare, digest, matrix, read, validate_rows
from scripts.analysis.audit_p0_long_confirmation import integral_counter
from scripts.revision.run_p0_read_exposure import load
from scripts.revision.run_p0_revision import dump


def check_row(row, seconds, metered, exposure=True, coordinated=True, round_interval=False):
    assert not validate_rows([row])
    assert row["status"] == "ok"
    for key in ("policy_commit_batch", "tracked_history_import"):
        assert row[key] == "True", key
    assert row["mutable_read_before_reasoning"] == str(exposure)
    assert row["coordinated_hotspot_admission"] == str(coordinated)
    assert row.get("round_interval_features", "False") == str(round_interval)
    for key in ("performance_guards", "compiled_cold_guard", "unified_commit_admission",
                "cycle_safe_hotspot_admission", "conflict_driven_fast_path", "observed_commit_admission"):
        assert row[key] == "False", key
    assert float(row["cycle_safe_gate_wait_ms"]) == 0
    assert float(row["measurement_seconds"]) == seconds
    assert float(row["reasoning_scale"]) == 1 and float(row["commit_apply_ms"]) == 24
    assert float(row["retry_delay_scale"]) == 1
    assert row["workload_scope"] == ("controlled_long_read_window_not_paper_default" if exposure
        else "paper_default_read_window_timing_feature_pilot")
    counts = {k: integral_counter(row[f"coordinated_gate_{k}"]) for k in (
        "admissions", "cycle_bypasses", "dependency_probes", "wait_calls", "timeouts", "priority_overtakes")}
    assert counts["cycle_bypasses"] <= min(counts["dependency_probes"], counts["wait_calls"])
    assert counts["admissions"] + counts["cycle_bypasses"] <= int(row["attempts"])
    if row["variant"] == "no_priority":
        assert counts["priority_overtakes"] == 0
    if metered:
        assert row["warmup_policy_counts_excluded"] == "True"
        calls = integral_counter(row["policy_calls"])
        assert sum(json.loads(row["phase_actions_json"]).values()) == calls
        assert math.isclose(float(row["decisions_per_attempt"]), calls / max(1, int(row["attempts"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    measured_root = args.input.resolve().parents[1]
    protocol = json.loads((args.input / "protocol.json").read_text())
    training = json.loads((args.input / "training/audit.json").read_text())
    recovery = {}
    if (args.input / "recovery.json").exists():
        recovery = json.loads((args.input / "recovery.json").read_text())
        assert recovery["protocol_sha256"] == digest(args.input / "protocol.json")
        assert recovery["training_audit_sha256"] == digest(args.input / "training/audit.json")
    for relative, expected in {**protocol["source_sha256"], **recovery.get("source_sha256", {})}.items():
        assert digest(measured_root / relative) == expected, relative
    for name, path in protocol.get("reference_paths", {}).items():
        assert digest(Path(path)) == protocol["reference_sha256"][name]
    exposure = protocol["config"]["mutable_read_before_reasoning"]
    coordinated = protocol["coordinated_hotspot_admission"]
    round_interval = protocol.get("round_interval_features", False)
    collection_seed = protocol["collection_seed_base"]
    validation_seed = protocol["validation_seed_base"]
    assert validation_seed == collection_seed + 20000
    assert protocol["training_seed"] == collection_seed + 10000
    assert protocol["holdout_seed_base"] == collection_seed + 40000
    binaries = list(measured_root.glob("cast_core*.so"))
    assert len(binaries) == 1
    binary_hash = digest(binaries[0])
    assert binary_hash == "7b515a4ea8bf6ed6baea9781c7ec8ee5a9b1e7161cf815ac4c0a256d1379b760"
    collection = read(args.input / "training/collection.csv")
    assert len(collection) == 8
    for index, row in enumerate(collection):
        assert row["variant"] == f"behavior_{index}" and int(row["seed"]) == collection_seed + index
        assert int(row["clients"]) == (8 if index == 0 else 16)
        check_row(row, .75, False, exposure, coordinated, round_interval)
    reports = {name: json.loads((args.input / f"training/{name}_report.json").read_text())
               for name in ("full", "nophase")}
    for name, awareness in (("full", "enabled"), ("nophase", "disabled")):
        report = reports[name]
        assert report["seed"] == protocol["training_seed"] and report["phase_awareness"] == awareness
        assert report["config"]["epochs"] == 4 and report["config"]["calibration_return_mode"] == "global"
        assert [Path(p).name for p in report["inputs"]] == [f"behavior_{i}.json" for i in range(8)]
        assert digest(args.input / f"training/{name}.json") == training["candidate_sha256"][name]
    assert reports["full"]["config"] == reports["nophase"]["config"]
    assert reports["full"]["inputs"] == reports["nophase"]["inputs"]
    no_phase = load(args.input / "training/nophase.json")
    checks = 0
    intervals = defaultdict(list)
    for name, expected in training["common_input_sha256"].items():
        path = args.input / "training" / name
        assert digest(path) == expected
        for transition in json.loads(path.read_text())["transitions"]:
            state = phase_aware_state_from_dict(transition["state"])
            intervals[state.phase].append(state.inter_round_interval_ms)
            outputs = []
            for phase in ("explore", "refine", "commit"):
                changed = dataclasses.replace(state, phase=phase)
                observed = policy_observation(changed, "disabled")
                outputs.append((state_key(observed), policy_group_key(observed), state_features(observed),
                                int(no_phase.select(changed).protected)))
            assert outputs[0] == outputs[1] == outputs[2]
            checks += 1
    assert checks == training["explicit_phase_checks"]
    rows = read(args.input / "validation/raw.csv")
    checked = matrix(rows, [16, 32], ["candidate", "reference"], 2, "clients", validation_seed)
    assert checked["complete"], checked
    for row in rows:
        check_row(row, 2., True, exposure, coordinated, round_interval)
    comparisons = compare(rows, "clients", "candidate", "reference")
    ratios = [row["tps_ratio"] for row in comparisons]
    geomean = math.exp(statistics.fmean(math.log(r) for r in ratios)) if min(ratios) > 0 else 0.
    completion_ok = all(statistics.fmean(float(r["task_completion_rate"]) for r in rows
        if r["clients"] == str(c) and r["variant"] == "candidate") >= statistics.fmean(
        float(r["task_completion_rate"]) for r in rows if r["clients"] == str(c) and r["variant"] == "reference") - .01
        for c in (16, 32))
    selected = geomean >= 1.05 and min(ratios) >= .90 and completion_ok
    selection = json.loads((args.input / "validation/selection.json").read_text())
    assert selection["candidate_selected"] == selected and selection["completion_ok"] == completion_ok
    assert math.isclose(selection["geomean"], geomean)
    result = dict(validation_matrix=checked, comparisons=comparisons, candidate_selected=selected,
        analysis_source_sha256=digest(Path(__file__)),
        geomean=geomean, completion_ok=completion_ok, explicit_phase_checks=checks,
        collection_seconds=training["collection_seconds"], native_binary_sha256=binary_hash,
        source_hashes_verified=True, training_inputs_verified=True,
        raw_sha256=digest(args.input / "validation/raw.csv"), controlled_scope_only=exposure,
        round_interval_features=round_interval, coordinated_hotspot_admission=coordinated,
        training_interval_ms_by_phase={phase: dict(samples=len(values), median=statistics.median(values),
            mean=statistics.fmean(values), minimum=min(values), maximum=max(values))
            for phase, values in intervals.items()},
        default_deployment_accepted=False, goal_completion_proven=False)
    if (args.input / "holdout/completed.json").exists():
        assert selected
        held = read(args.input / "holdout/raw.csv")
        variants = ["silo", "static_threshold", "hotrule_always", "hotrule_threshold",
                    "full", "nophase", "frozen_full", "no_priority"]
        held_matrix = matrix(held, [32], variants, 3, "clients", protocol["holdout_seed_base"])
        assert held_matrix["complete"]
        for row in held:
            check_row(row, 4., True, exposure, coordinated, round_interval)
        result.update(holdout_matrix=held_matrix, holdout_comparisons=[item for v in variants if v != "full"
            for item in compare(held, "clients", "full", v)])
    dump(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
