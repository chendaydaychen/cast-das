#!/usr/bin/env python3
"""Aggregate paired compiled-policy choices without exporting task contents."""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.runtime.paper_policy import CompiledPhasePolicy
from agent.runtime.residual_policy import ResidualTablePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict, policy_observation
from agent.cc.atcc.ppo import state_key, policy_group_key


def load_policy(path):
    payload = json.loads(path.read_text())
    if payload.get("artifact_type") == ResidualTablePolicy.artifact_type:
        return ResidualTablePolicy.from_dict(payload)
    return CompiledPhasePolicy.load(path)


def recorded_rows(payload):
    rows = payload.get("transitions") or payload.get("decisions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty recorded transitions or decisions required")
    return rows


def summarize(rows, full, no_phase):
    by_phase = defaultdict(lambda: {"pairs": Counter(), "logged": Counter()})
    for row in rows:
        state = phase_aware_state_from_dict(row["state"])
        if state.phase not in {"explore", "refine", "commit"}:
            raise ValueError("expected original labeled states")
        first = full.select(state).encoded
        second = no_phase.select(state).encoded
        by_phase[state.phase]["pairs"][(first, second)] += 1
        by_phase[state.phase]["logged"][int(row["action"])] += 1

    def report(group):
        pairs = group["pairs"]
        total = sum(pairs.values())
        agreed = sum(count for (first, second), count in pairs.items()
                     if first == second)
        return {
            "states": total,
            "agreement": agreed / total if total else None,
            "paired_actions": [
                {"full": first, "no_phase": second, "count": count}
                for (first, second), count in sorted(pairs.items())
            ],
            "logged_exploratory_actions": dict(sorted(group["logged"].items())),
        }

    combined = {"pairs": Counter(), "logged": Counter()}
    for group in by_phase.values():
        combined["pairs"].update(group["pairs"])
        combined["logged"].update(group["logged"])
    return {"overall": report(combined), "sampling_audit": audit_propensities(rows),
            "phase_overlap": audit_phase_overlap(rows,
                remove_phase_index_proxy=no_phase.remove_phase_index_proxy,
                growth_conditioned=getattr(no_phase, "growth_conditioned_groups", False),
                local_context=getattr(no_phase, "local_context_groups", False)), "phases": {
        phase: report(by_phase[phase]) for phase in sorted(by_phase)
    }}


def audit_propensities(rows):
    checked = mismatched = invalid = missing = 0
    for row in rows:
        probabilities = row.get("behavior_action_probabilities", ())
        if not probabilities:
            missing += 1
            continue
        checked += 1
        action = int(row["action"])
        if (not 0 <= action < len(probabilities)
                or any(not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)):
            invalid += 1
            continue
        if not math.isclose(probabilities[action], row["behavior_probability"],
                            rel_tol=1e-9, abs_tol=1e-12):
            mismatched += 1
    return {"checked": checked, "missing_distribution": missing,
            "invalid_distribution": invalid, "recorded_action_probability_mismatches": mismatched}


def audit_phase_overlap(rows, remove_phase_index_proxy=False, growth_conditioned=False, local_context=False):
    result = {}
    group = lambda state: policy_group_key(state, growth_conditioned=growth_conditioned,
                                          local_context=local_context)
    for label, encode in (("state_buckets", state_key), ("calibration_groups", group)):
        phases = defaultdict(Counter)
        for row in rows:
            state = phase_aware_state_from_dict(row["state"])
            observed = policy_observation(state, "disabled")
            if remove_phase_index_proxy:
                observed = dataclasses.replace(observed, completed_rounds=0)
            phases[encode(observed)][state.phase] += 1
        mixed = [counts for counts in phases.values() if len(counts) > 1]
        result[label] = {"no_phase_groups": len(phases), "multi_phase_groups": len(mixed),
                         "states_in_multi_phase_groups": sum(sum(c.values()) for c in mixed)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--no-phase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    full = load_policy(args.full)
    no_phase = load_policy(args.no_phase)
    if full.phase_awareness != "enabled" or no_phase.phase_awareness != "disabled":
        parser.error("expected Full (enabled) and NoPhase (disabled) policies")
    source = args.trajectory.read_bytes()
    payload = json.loads(source)
    result = summarize(recorded_rows(payload), full, no_phase)
    result["trajectory_sha256"] = hashlib.sha256(source).hexdigest()
    result["scope"] = ("Same previously collected states; paired policy "
                       "choices are diagnostics, not held-out performance or causal effects.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": result["overall"], "sampling_audit": result["sampling_audit"],
                      "phase_overlap": result["phase_overlap"], "trajectory_sha256":
                      result["trajectory_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
