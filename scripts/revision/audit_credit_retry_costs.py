#!/usr/bin/env python3
"""Audit sanitized matched replay task/attempt costs without causal attribution."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


def summarize(payload, arm):
    rows = []
    abort_reasons = Counter()
    for task in payload["tasks"]:
        attempts = task["attempt_details"]
        if not attempts or len(attempts) != task["attempts"]:
            raise ValueError("missing attempt details")
        sequence = attempts[0]["sequence"]
        if any(a["sequence"] != sequence or a["attempt"] != i for i, a in enumerate(attempts)):
            raise ValueError("inconsistent attempt sequence")
        failed = [a for a in attempts if not a["committed"]]
        if len(failed) != task["aborts"] or bool(attempts[-1]["committed"]) != task["committed"]:
            raise ValueError("attempt/task outcomes disagree")
        if any(a["committed"] for a in attempts[:-1]):
            raise ValueError("successful logical task was retried")
        elapsed = sum(a["elapsed_ms"] for a in attempts)
        if elapsed > task["elapsed_ms"] + 1:
            raise ValueError("attempt durations exceed logical-task duration")
        for attempt in failed:
            abort_reasons[attempt["reason"]] += 1
        rows.append(dict(arm=arm, sequence=sequence, committed=task["committed"],
            attempts=len(attempts), aborts=len(failed),
            retry_inclusive_seconds=task["elapsed_ms"]/1000,
            failed_attempt_seconds=sum(a["elapsed_ms"] for a in failed)/1000,
            failed_agent_seconds=sum(a["agent_cost_ms"] for a in failed)/1000,
            total_wait_seconds=sum(a["blocked_ms"] for a in attempts)/1000,
            changes_on_failed_attempts=sum(a["action_changes"] for a in failed),
            changes_on_successful_attempts=sum(a["action_changes"] for a in attempts if a["committed"]),
            wasted_input_tokens=task["wasted_input_tokens"], wasted_output_tokens=task["wasted_output_tokens"]))
    if len({row["sequence"] for row in rows}) != len(rows):
        raise ValueError("duplicate logical-task sequence")
    successful = sorted(r["retry_inclusive_seconds"] for r in rows if r["committed"])
    if not successful:
        raise ValueError("no completed tasks; success latency is undefined")
    attempt_count = sum(r["attempts"] for r in rows)
    abort_count = sum(r["aborts"] for r in rows)
    summary = dict(arm=arm, source_sha256=payload["source_sha256"], time_scale=payload["time_scale"],
        tasks=len(rows), commits=len(successful), attempts=attempt_count, aborts=abort_count,
        attempt_abort_fraction=abort_count/attempt_count,
        tasks_with_retries=sum(r["attempts"] > 1 for r in rows),
        retry_inclusive_mean_seconds=sum(successful)/len(successful),
        retry_inclusive_p99_seconds=successful[math.ceil(.99*len(successful))-1],
        total_failed_attempt_seconds=sum(r["failed_attempt_seconds"] for r in rows),
        total_failed_agent_seconds=sum(r["failed_agent_seconds"] for r in rows),
        total_wait_seconds=sum(r["total_wait_seconds"] for r in rows),
        changes_on_failed_attempts=sum(r["changes_on_failed_attempts"] for r in rows),
        changes_on_successful_attempts=sum(r["changes_on_successful_attempts"] for r in rows),
        abort_reasons=dict(abort_reasons))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    summaries, rows = [], []
    for arm, filename in (
        ("Full", "supported_full_eval_20261010_attempts.json"),
        ("NoPhase_explicit_label_only", "supported_nophase_eval_20261010_attempts.json"),
        ("NoPhase_without_phase_index", "proxyfree_nophase_eval_20261010_attempts.json"),
    ):
        summary, task_rows = summarize(json.loads((args.directory / filename).read_text()), arm)
        summaries.append(summary)
        rows.extend(task_rows)
    for field in ("source_sha256", "time_scale", "tasks"):
        if len({s[field] for s in summaries}) != 1:
            raise ValueError("unmatched replay inputs")
    report = dict(arms=summaries, scope="descriptive attempt cost audit, not causal phase attribution",
                  parallel_costs_are_not_wall_time=True,
                  latency_includes_failed_attempts=True,
                  task_csv_contains_no_access_keys_or_customer_data=True)
    (args.directory / "supported_retry_cost_audit_20261010.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.directory / "supported_retry_task_costs_20261010.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
