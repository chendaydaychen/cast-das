"""Two-cohort hotspot migration with immutable credit-trained policies."""
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.cc import LockConflict
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy
from agent.runtime.residual_policy import ResidualTablePolicy
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.compare_frozen_credit_trace import FixedProtectionPolicy, RecordingPolicy
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.run_audited_credit_pressure import AuditedManager, LinearScorePolicy
from scripts.unified_trace.run_credit_review_experiment import transaction_metadata
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table


ARMS = ("full", "nophase", "linear", "hot_always", "item_threshold3", "static")


def make_cohorts(seed, workers, region_size=128):
    if not 2 <= workers <= 8 or region_size < 10:
        raise ValueError("2-8 workers and region size >= 10 required")
    rng = random.Random(seed)
    first = []
    for worker in range(workers):
        remaining = list(range(region_size))
        keys = []
        for _ in range(10):
            chosen = rng.choices(remaining, weights=[(k+1)**-.99 for k in remaining])[0]
            remaining.remove(chosen)
            keys.append(chosen)
        first.append([dict(key=k, kind="write" if i in (3, 7) else "read") for i, k in enumerate(keys)])
    second = [[dict(op, key=op["key"]+region_size) for op in task] for task in first]
    return [dict(epoch=0, label="before_shift", tasks=first),
            dict(epoch=1, label="after_shift", tasks=second)]


def summarize(tasks, elapsed):
    if not tasks or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Valid measured cohort required")
    attempts = [a for t in tasks for a in t["attempts"]]
    if any(not t["attempts"] or t["committed"] != t["attempts"][-1]["committed"] for t in tasks):
        raise ValueError("Incomplete cohort outcome")
    latencies = [t["latency_ms"] for t in tasks if t["committed"]]
    return dict(tasks=len(tasks), commits=len(latencies), failed_tasks=len(tasks)-len(latencies),
        workload_seconds=elapsed, throughput_tps=len(latencies)/elapsed,
        maximum_successful_task_ms=max(latencies) if latencies else None,
        maximum_submitted_task_ms=max(t["latency_ms"] for t in tasks),
        attempts=len(attempts), aborts=sum(not a["committed"] for a in attempts),
        action_changes=sum(a["action_changes"] for a in attempts),
        item_changes=sum(a["item_changes"] for a in attempts),
        paid_gap_ms=sum(a["paid_gap_ms"] for a in attempts),
        aborted_gap_ms=sum(a["paid_gap_ms"] for a in attempts if not a["committed"]))


def run_cohort(manager, recorded, cohort, *, system, delays_ms, max_attempts,
               carry_cost, static=False, sleep_fn=time.sleep, profile_clock=None,
               account_agent_work=False, phase_schedule='operation-index', record_lock_exposure=False):
    if phase_schedule not in ('operation-index', 'refine-before-first-call'):
        raise ValueError('Unknown phase schedule')
    workers = len(cohort["tasks"])
    barrier = threading.Barrier(workers)
    start_decision = len(recorded.records) if recorded else 0
    start_attempt = len(manager.audit_transactions)
    heat_before = {key:manager.hotness_tracker.object_snapshot(key) for key in manager._catalog}

    def execute(worker):
        barrier.wait(timeout=30)
        with (profile_clock.scope('other', root=True) if profile_clock else contextlib.nullcontext()):
            return execute_task(worker)

    def execute_task(worker):
        started = time.perf_counter()
        attempts, carried = [], 0.
        for index in range(max_attempts):
            metadata = transaction_metadata(system, retry_count=index, policy_commit_batch=False)
            metadata.update(workload="frozen_ycsb_shift", task_type="ycsb")
            if carry_cost:
                metadata["prior_retry_cost_ms"] = carried
            task_id = f"shift:{cohort['epoch']}:{worker}"
            txn = manager.begin(task_id, metadata, snapshot_object_ids=None, strategy=system)
            observed, paid_gap, paid_intervals = [], 0., []
            try:
                txn.enter_phase("explore")
                for operation_index, op in enumerate(cohort["tasks"][worker], 1):
                    txn.check_active()
                    if operation_index == 8 or (operation_index == 5 and phase_schedule == 'operation-index'):
                        txn.enter_phase("refine" if operation_index == 5 else "commit")
                    object_id = f"shift-key:{op['key']}"
                    if op["kind"] == "read":
                        txn.read(object_id)
                    else:
                        txn.write(object_id, str(((cohort["epoch"]+1) << 40) | ((worker+1) << 16) | operation_index))
                    observed.append(op)
                    if operation_index in (4, 7, 10):
                        txn.check_active()
                        if operation_index == 4 and phase_schedule == 'refine-before-first-call':
                            txn.enter_phase('refine')
                        if manager.expensive_operation_checkpoint_enabled:
                            manager.interceptor.expensive_operation_checkpoint(txn, 1)
                        lock_exposure=dict(held_reads_before_call=len(txn.context.held_read_locks),
                            selected_read_targets_before_call=len(txn.context.policy_read_lock_targets),
                            held_writes_before_call=len(txn.context.held_write_locks),
                            private_writes_before_call=len(txn.write_set),
                            selected_write_targets_before_call=len(txn.context.policy_write_lock_targets),
                            defer_policy_writes=bool(txn.metadata.get('_defer_policy_write_locks',False))) if record_lock_exposure else None
                        gap_started = time.perf_counter()
                        with (profile_clock.scope('reasoning') if profile_clock else contextlib.nullcontext()):
                            sleep_fn(delays_ms[(4, 7, 10).index(operation_index)]/1000.)
                        actual_ms = (time.perf_counter()-gap_started)*1000.
                        paid_gap += actual_ms
                        if account_agent_work:
                            manager.interceptor.account_agent_interval(txn)
                            paid_intervals.append(dict(after_operation=operation_index,
                                requested_ms=delays_ms[(4, 7, 10).index(operation_index)],
                                actual_ms=actual_ms,cumulative_agent_cost_ms=txn.context.agent_cost_ms))
                            if record_lock_exposure:paid_intervals[-1]['lock_exposure']=lock_exposure
                            if phase_schedule != 'operation-index':
                                paid_intervals[-1]['phase_before_call'] = txn.context.phase.value
                        txn.check_active()
                result = txn.commit(system)
                committed, reason = bool(result.committed), str(result.reason)
            except LockConflict as exc:
                if txn.result is None:
                    txn.abort(exc.reason, strategy=system, conflict_object_ids=exc.targets)
                committed, reason = False, str(exc.reason)
            except BaseException:
                if txn.result is None:
                    txn.abort("shift execution error", strategy=system)
                raise
            paid = paid_gap + max(0., txn.context.operation_cost_ms)
            attempts.append(dict(attempt=index, logical_task_id=task_id, committed=committed,
                reason=reason, observed=observed, paid_gap_ms=paid_gap, paid_work_ms=paid,
                carried_prior_restart_cost_ms=txn.context.prior_retry_cost_ms,
                action_changes=sum(e.kind == "action-change" for e in txn.events),
                item_changes=sum(e.kind == "item-protection-change" for e in txn.events),
                final_action=int(txn.context.action.protected)))
            if account_agent_work:
                attempts[-1]['paid_intervals'] = paid_intervals
            if committed:
                break
            if carry_cost:
                carried += paid
            if index+1 < max_attempts:
                with (profile_clock.scope('retry_backoff') if profile_clock else contextlib.nullcontext()):
                    sleep_fn(.010)
        return dict(worker=worker, logical_task_id=task_id, committed=committed,
            latency_ms=(time.perf_counter()-started)*1000., attempts=attempts)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = list(pool.map(execute, range(workers)))
    elapsed = time.perf_counter()-started
    history = manager.history()
    oracle = verify_history(**history)
    decisions = list(recorded.records[start_decision:]) if recorded else []
    summary = summarize(tasks, elapsed)
    summary.update(epoch=cohort["epoch"], label=cohort["label"], policy_calls=len(decisions),
        proposed_changes=sum(r["action"] != r["state"]["current_action"] for r in decisions),
        oracle_verdict=oracle["verdict"], manager_identity_preserved=True,
        prior_history_attempts=start_attempt, cumulative_history_attempts=len(history["attempts"]))
    return dict(summary=summary, tasks=tasks, decisions=decisions, history=history, oracle=oracle,
        heat_before=heat_before,
        heat_after={key:manager.hotness_tracker.object_snapshot(key) for key in manager._catalog})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-folder", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026101303)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--pure-table-audit", type=Path)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("Distinct arms required")
    cohorts = make_cohorts(args.seed, args.workers)
    original = json.loads((args.policy_folder / "protocol.json").read_text())
    calibration = json.loads(args.calibration.read_text())
    delays = [calibration["delay_ms"][p] for p in ("explore", "refine", "commit")]
    if any(type(v) is not int or not 1 <= v <= 10000 for v in delays):
        parser.error("Measured positive integer delays <= 10 seconds required")
    policies = {arm:ResidualTablePolicy.from_dict(json.loads(
        (args.policy_folder / f"{arm}_policy.json").read_text())) for arm in ("full", "nophase")}
    policies.update(linear=LinearScorePolicy(json.loads((args.policy_folder / "linear_policy.json").read_text())["weights"]),
        hot_always=HotRulePolicy(), item_threshold3=CompiledPhasePolicy([]), static=FixedProtectionPolicy())
    table_hashes, table_paths, table_initialization = {}, {}, {}
    if args.pure_table_audit:
        for arm in set(args.arms) & {'full', 'nophase'}:
            started = time.perf_counter()
            policies[arm], path = load_pure_table(args.pure_table_audit, args.policy_folder, arm, 4)
            table_initialization[arm] = time.perf_counter()-started
            table_hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
            table_paths[arm] = path
    hashes = {name:hashlib.sha256((args.policy_folder / name).read_bytes()).hexdigest()
              for name in ("full_policy.json", "nophase_policy.json", "linear_policy.json", "protocol.json")}
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(cohorts, indent=2)
    (args.output / "cohorts.json").write_text(payload)
    protocol = dict(seed=args.seed, workers=args.workers, selected_arms=args.arms,
        deployment='pure medoid table' if args.pure_table_audit else 'exact coefficient scoring',
        pure_table_sha256=table_hashes,
        pure_table_initialization_seconds=table_initialization,
        pure_table_audit_sha256=hashlib.sha256(args.pure_table_audit.read_bytes()).hexdigest()
            if args.pure_table_audit else None,
        cohort_input_sha256=hashlib.sha256(payload.encode()).hexdigest(), policy_sha256=hashes,
        source_runtime_protocol=original, calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        source_capture_sha256=calibration["capture_sha256"], delays_ms=delays,
        max_attempts=6, application_retry_ms=10, region_size=128,
        item_threshold=3, item_window_ms=5000., hot_prior=False,
        carry_prior_restart_cost=original.get("carry_prior_restart_cost", False),
        execution_phases="explore ops1-4; refine ops5-7; commit ops8-10; prescribed",
        workload_change="after first cohort drains, offset all request keys by 128; preserve manager/history",
        retrained=False, tuned=False, new_api_calls=0,
        scope="synthetic migration transfer; recorded per-call median gaps, not full online agent replay",
        static_control="fixed initial HOT_WRITE plus Delay/Priority; not Figure 12 threshold replication",
        tail="cohort sample maximum, not stable P99",
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'agent/runtime/paper_policy.py':hashlib.sha256((ROOT/'agent/runtime/paper_policy.py').read_bytes()).hexdigest()})
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.output / "runner_executed.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "paper_policy_executed.py").write_bytes((ROOT/'agent/runtime/paper_policy.py').read_bytes())
    results = []
    for arm in args.arms:
        folder = args.output / arm
        folder.mkdir()
        recorded = RecordingPolicy(policies[arm]) if arm != "item_threshold3" else None
        manager = AuditedManager(paper_policy=recorded or policies[arm], record_traces=False,
            collect_trajectories=False, performance_guards_enabled=False, low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True, priority_enabled=True,
            policy_selected_write_admission_only=original["policy_selected_write_admission_only"])
        configure_frozen_runtime(manager, original)
        manager.expensive_operation_checkpoint_enabled = original["before_reasoning_checkpoint"]
        if arm == "static":
            manager.paper_static_initial_mask = 4
        elif arm == "hot_always":
            manager.first_hot_access_events = []
            manager.interceptor.hooks = FirstHotAccessHooks(manager)
        elif arm == "item_threshold3":
            manager.interceptor.hooks = ItemFrequencyHooks(manager, minimum_accesses=3, window_ms=5000.)
        for key in range(256):
            manager.register_object(f"shift-key:{key}", "0", kind="row")
        for cohort in cohorts:
            record = run_cohort(manager, recorded, cohort, system="paper-atcc", delays_ms=delays,
                max_attempts=protocol["max_attempts"], carry_cost=protocol["carry_prior_restart_cost"],
                static=arm == "static")
            record["summary"].update(arm=arm, priority_config=dataclasses.asdict(manager.priority_manager.config))
            (folder / f"epoch_{cohort['epoch']}_records.json").write_text(json.dumps(record, indent=2))
            results.append(record["summary"])
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(record["summary"]), flush=True)
            if (record["oracle"]["verdict"] != "verified"
                    or (arm == "static" and record["summary"]["action_changes"] != 0)):
                raise RuntimeError("History or static invariant failed; raw evidence retained")
    if any(hashlib.sha256((args.policy_folder / n).read_bytes()).hexdigest() != h for n, h in hashes.items()):
        raise RuntimeError("Frozen policy artifacts changed")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != table_hashes[arm]
           for arm, path in table_paths.items()):
        raise RuntimeError("Frozen pure tables changed")


if __name__ == "__main__":
    main()
