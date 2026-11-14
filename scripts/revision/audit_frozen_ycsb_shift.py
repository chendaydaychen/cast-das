"""Verify two-cohort input conservation, heat continuity and frozen-run summaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_frozen_ycsb_shift import make_cohorts, summarize
from scripts.revision.audit_frozen_table_inputs import compare_inputs
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.frozen_actor_collection import FrozenActorDistribution
from agent.runtime.residual_policy import ResidualTablePolicy


def checked_attempts(record, cohort, workers, max_attempts, carry):
    tasks = record["tasks"]
    if len(tasks) != workers or sorted(t["worker"] for t in tasks) != list(range(workers)):
        raise ValueError("Missing or duplicate workers")
    for task in tasks:
        expected_id = f"shift:{cohort['epoch']}:{task['worker']}"
        if task["logical_task_id"] != expected_id or not 1 <= len(task["attempts"]) <= max_attempts:
            raise ValueError("Wrong task identity or attempt budget")
        paid_prior = 0.
        planned = cohort["tasks"][task["worker"]]
        for index, attempt in enumerate(task["attempts"]):
            if (attempt["attempt"] != index or attempt["logical_task_id"] != expected_id
                    or (index < len(task["attempts"])-1 and attempt["committed"])
                    or attempt["observed"] != planned[:len(attempt["observed"])]
                    or len(attempt["observed"]) > len(planned)
                    or (attempt["committed"] and attempt["observed"] != planned)):
                raise ValueError("Invalid observed input prefix or retry order")
            paid, gap, prior = (attempt[k] for k in ("paid_work_ms", "paid_gap_ms", "carried_prior_restart_cost_ms"))
            if (any(not math.isfinite(v) or v < 0 for v in (paid, gap, prior))
                    or gap > paid or gap > task["latency_ms"]+1e-6
                    or not math.isclose(prior, paid_prior, abs_tol=1e-6)):
                raise ValueError("Invalid paid work or carry ledger")
            if carry and not attempt["committed"]:
                paid_prior += paid


def heat_continuity(previous, current, region_size):
    if current["heat_before"] != previous["heat_after"]:
        raise ValueError("Heat metadata reset between cohorts")
    for key, before in current["heat_before"].items():
        index = int(key.split(":")[-1])
        after = current["heat_after"][key]
        if index >= region_size and before["accesses"] != 0:
            raise ValueError("New region was preheated")
        if index < region_size and any(after[k] != before[k] for k in before if k != "hot"):
            raise ValueError("After-shift cohort accessed or conflicted with the old region")


def audit(folder, policy_folder=None, table_audit=None, table_budget=4):
    protocol = json.loads((folder / "protocol.json").read_text())
    if (policy_folder is None) != (table_audit is None):
        raise ValueError("Both frozen model folder and pure-table audit required")
    table_models = {}
    if policy_folder is not None:
        actor_seed = protocol["source_runtime_protocol"]["actor_seed"]
        for arm in ("full", "nophase"):
            model_path = policy_folder / f"{arm}_policy.json"
            payload = json.loads(model_path.read_text())
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != protocol["policy_sha256"][model_path.name]:
                raise ValueError("Actor differs from frozen workload protocol")
            table, table_path = load_pure_table(table_audit, policy_folder, arm, table_budget)
            if protocol.get('deployment') == 'pure medoid table':
                if (hashlib.sha256(table_path.read_bytes()).hexdigest() != protocol['pure_table_sha256'][arm]
                        or hashlib.sha256(table_audit.read_bytes()).hexdigest() != protocol['pure_table_audit_sha256']):
                    raise ValueError('Audit uses a different table than the deployed artifact')
            table_models[arm] = (FrozenActorDistribution(payload, seed=actor_seed),
                                 ResidualTablePolicy.from_dict(payload), table)
    raw_input = (folder / "cohorts.json").read_bytes()
    cohorts = json.loads(raw_input)
    if (hashlib.sha256(raw_input).hexdigest() != protocol["cohort_input_sha256"]
            or cohorts != make_cohorts(protocol["seed"], protocol["workers"], protocol["region_size"])):
        raise ValueError("Declared input differs from deterministic generator")
    if (protocol["retrained"] or protocol["tuned"] or protocol["new_api_calls"] != 0
            or len(set(protocol["selected_arms"])) != len(protocol["selected_arms"])):
        raise ValueError("Unexpected training/tuning or duplicate controls")
    runner_hash = hashlib.sha256((folder / "runner_executed.py").read_bytes()).hexdigest()
    if runner_hash != protocol["source_sha256"]["scripts/revision/run_frozen_ycsb_shift.py"]:
        raise ValueError("Executed source differs from protocol")
    if 'agent/runtime/paper_policy.py' in protocol['source_sha256']:
        if hashlib.sha256((folder/'paper_policy_executed.py').read_bytes()).hexdigest() != protocol['source_sha256']['agent/runtime/paper_policy.py']:
            raise ValueError('Executed lookup implementation differs from protocol')
    summaries = json.loads((folder / "results.json").read_text())
    wanted = [(arm, epoch) for arm in protocol["selected_arms"] for epoch in range(2)]
    if [(r["arm"], r["epoch"]) for r in summaries] != wanted:
        raise ValueError("Incomplete or reordered cohort matrix")
    results = []
    initial = None
    for arm in protocol["selected_arms"]:
        previous = None
        for cohort in cohorts:
            path = folder / arm / f"epoch_{cohort['epoch']}_records.json"
            record = json.loads(path.read_text())
            row = record["summary"]
            if row != summaries[wanted.index((arm, cohort["epoch"]))]:
                raise ValueError("Raw record and aggregate summary disagree")
            recomputed = summarize(record["tasks"], row["workload_seconds"])
            if any(row.get(k) != v for k, v in recomputed.items()):
                raise ValueError("Summary does not reproduce task records")
            checked_attempts(record, cohort, protocol["workers"], protocol["max_attempts"],
                             protocol["carry_prior_restart_cost"])
            oracle = verify_history(**record["history"])
            if oracle["verdict"] != "verified":
                raise ValueError("History oracle rejected cohort")
            if initial is not None and initial != record["history"]["initial"]:
                raise ValueError("Initial values differ across cohorts/controls")
            initial = record["history"]["initial"]
            if row["policy_calls"] != len(record["decisions"]):
                raise ValueError("Recorded policy calls disagree")
            for d in record["decisions"]:
                current, action = d["state"]["current_action"], d["action"]
                if not 0 <= current < 16 or not 0 <= action < 16 or current & action != current:
                    raise ValueError("Protection proposal releases held bits")
            if row["proposed_changes"] != sum(d["action"] != d["state"]["current_action"] for d in record["decisions"]):
                raise ValueError("Suggested-change count disagrees")
            if arm == "static" and row["action_changes"] != 0:
                raise ValueError("Static control changed its mask")
            expected_prior = len(previous["history"]["attempts"]) if previous else 0
            if (row["prior_history_attempts"] != expected_prior
                    or row["cumulative_history_attempts"] != expected_prior+row["attempts"]):
                raise ValueError("Cumulative runtime history reset or attempts missing")
            if previous:
                heat_continuity(previous, record, protocol["region_size"])
                for before, after in zip(previous["history"]["attempts"], record["history"]["attempts"]):
                    ignored = {"end_ns"} if not before["committed"] else set()
                    if {k:v for k, v in before.items() if k not in ignored} != {
                            k:v for k, v in after.items() if k not in ignored}:
                        raise ValueError("Previous history was replaced")
            elif any(h["accesses"] for h in record["heat_before"].values()):
                raise ValueError("First cohort started with primed heat")
            item = dict(arm=arm, epoch=cohort["epoch"], verdict="verified",
                commits=row["commits"], failed_tasks=row["failed_tasks"],
                record_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if arm in table_models:
                item["frozen_table_comparison"] = compare_inputs(record["decisions"], *table_models[arm])
            results.append(item)
            previous = record
    return dict(verdict="verified", records=results,
        scope="input/heat/retry/history preservation; not proof of policy superiority or continuous online adaptation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--policy-folder", type=Path)
    parser.add_argument("--table-audit", type=Path)
    parser.add_argument("--table-budget", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.folder, args.policy_folder, args.table_audit, args.table_budget)
    output = args.output or args.folder / "shift_audit.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
