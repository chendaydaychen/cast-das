#!/usr/bin/env python3
"""Re-derive complete long-confirmation comparisons and two-layer wait counters."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.audit_p0_claims import compare, digest, matrix, read
from scripts.revision.run_p0_revision import dump


def integral_counter(value):
    parsed = float(value)
    assert parsed >= 0 and parsed.is_integer(), value
    return int(parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protocol = json.loads((args.input / "protocol.json").read_text())
    rows = read(args.input / "raw.csv")
    checked = matrix(rows, [32], protocol["variants"], 3, "clients", 7930000)
    assert checked["complete"], checked
    assert protocol["config"]["company_count"] == 1000
    assert protocol["measure_seconds"] == 8 and protocol["warmup_seconds"] == 1
    assert protocol["performance_guards"] is False and protocol["policy_commit_batch"] is True
    assert protocol["threshold"] == .05 and protocol["cycle_safe_gate_wait_ms"] == 0
    assert protocol["unified_commit_admission"] is False and protocol["cycle_safe_hotspot_admission"] is False
    measurements = {}
    for variant in protocol["variants"]:
        selected = [row for row in rows if row["variant"] == variant]
        attempts = sum(int(row["attempts"]) for row in selected)
        gate_elapsed = sum(float(row["commit_admission_wait_ms_per_attempt"]) * int(row["attempts"])
                           for row in selected)
        blocked = sum(float(row["blocked_ms_total"]) for row in selected)
        measurements[variant] = dict(attempts=attempts,
            helper_callback_elapsed_ms_per_attempt=gate_elapsed / max(1, attempts),
            context_blocked_ms_per_attempt=blocked / max(1, attempts),
            atcc_lock_wait_events=sum(integral_counter(row["paper_lock_wait_events"]) for row in selected),
            failed_tasks=sum(int(row["failed_tasks"]) for row in selected))
        for row in selected:
            assert float(row["measurement_seconds"]) == 8
            if variant in ("full", "nophase", "frozen_full", "no_priority"):
                name = "nophase" if variant == "nophase" else "full"
                assert row["policy_sha256"] == protocol["policy_sha256"][name]
    measured_root = args.input.resolve().parents[1]
    for relative, expected in protocol["source_sha256"].items():
        assert digest(measured_root / relative) == expected, relative
    comparisons = [item for baseline in protocol["variants"] if baseline != "full"
                   for item in compare(rows, "clients", "full", baseline)]
    dump(args.output, dict(matrix=checked, comparisons=comparisons, waiting=measurements,
        source_hashes_verified=True, raw_sha256=digest(args.input / "raw.csv"),
        wait_scope="callback elapsed includes gate acquisition bookkeeping; not exclusive blocked time or task latency",
        universal_learning_or_priority_necessity_proven=False,
        full_paper_protection_proven=False))
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
