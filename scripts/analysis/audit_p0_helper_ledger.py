#!/usr/bin/env python3
"""Independently reconcile helper components with actual collected transitions."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.revision.run_p0_revision import digest, dump

FIELDS = ("agent_blocked_ms_caused", "background_blocked_ms_caused")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-files",type=int,default=8)
    args = parser.parse_args()
    assert not args.output.exists()
    audit = json.loads((args.training / "audit.json").read_text())
    files, results = sorted(args.training.glob("behavior_*.json")), []
    assert args.expected_files>0 and len(files) == args.expected_files
    assert {p.name:digest(p) for p in files}==audit["common_input_sha256"]
    for path in files:
        assert digest(path) == audit["common_input_sha256"][path.name]
        data = json.loads(path.read_text())
        sidecar = data["helper_externalities"]
        assert sidecar["recording_active"]
        assert sidecar["reconciliation"]["synthetic_transitions_added"] == 0
        rows = data["transitions"]
        costs = defaultdict(lambda: defaultdict(float))
        component_count = 0
        for index, values in sidecar["components_by_transition"].items():
            row = rows[int(index)]
            for field in FIELDS:
                value = float(values.get(field, 0.))
                assert math.isfinite(value) and value >= 0
                assert float(row[field]) >= value - 1e-8, (path, index, field)
                costs[row["txn_id"]][field] += value
            component_count += 1
        closed = {row["txn_id"] for row in rows if row["done"]}
        total, predecision, uncovered = 0., 0., 0.
        for tid, values in sidecar["totals_by_owner"].items():
            for field in FIELDS:
                value = float(values.get(field, 0.))
                assert math.isfinite(value) and value >= 0
                before = float(sidecar["predecision_costs_by_owner"].get(tid, {}).get(field, 0.))
                assert 0 <= before <= value + 1e-8
                total += value
                predecision += before
                if tid in closed:
                    assert math.isclose(costs[tid][field] + before, value, abs_tol=1e-7), (path, tid, field)
                else:
                    assert not costs[tid][field], (path, tid)
                    uncovered += value - before
        assert math.isclose(uncovered, sidecar["reconciliation"]["uncovered_externality_ms"], abs_tol=1e-7)
        assert math.isclose(sum(sum(v.values()) for v in costs.values()) + predecision + uncovered,
                            total, abs_tol=1e-7)
        results.append(dict(file=path.name, transitions=len(rows), component_transitions=component_count,
            owners=len(sidecar["totals_by_owner"]), recorded_externality_ms=total,
            predecision_externality_ms=predecision, uncovered_externality_ms=uncovered,
            late_terminal_externality_ms=sidecar["reconciliation"]["late_externality_ms"]))
    dump(args.output, dict(scope="collected helper cost-component coverage, not performance causality",
        files=results, common_input_sha256=audit["common_input_sha256"],
        source_sha256=digest(Path(__file__)), exact_coverage_and_no_duplicate_component_verified=True,
        sql_or_future_state_added=False, reward_formula_or_coefficients_changed=False,
        performance_benefit_proven=False, goal_completed=False))
    print(json.dumps(dict(files=len(results), recorded_externality_ms=sum(r["recorded_externality_ms"] for r in results),
        component_transitions=sum(r["component_transitions"] for r in results))), flush=True)


if __name__ == "__main__":
    main()
