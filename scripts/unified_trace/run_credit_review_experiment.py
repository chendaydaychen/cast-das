#!/usr/bin/env python3
"""Run the online-observed native CreditReview Figure 3 experiment."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import math
import statistics
import sys
import threading
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cc import ConcurrencyControlRegistry, LockConflict
from agent.cc.atcc.ppo import EpsilonGreedyPolicy
from agent.benchmarks.phases import ReasoningProfile, sleep_for_reasoning
from agent.runtime import AgentTransactionManager
from agent.runtime.priority import PriorityConfig
from agent.runtime.paper_policy import CompiledPhasePolicy, StaticThresholdPhasePolicy
from agent.workloads.credit_review import (
    CreditReviewConfig,
    CreditReviewExecution,
    CreditReviewTaskSpec,
    CreditReviewWorkload,
)


SYSTEMS = ("2pl-wait-die", "bamboo", "silo", "polaris", "paper-atcc")
SYSTEM_LABELS = {
    "2pl-wait-die": "2PL",
    "bamboo": "Bamboo",
    "silo": "Silo",
    "polaris": "Polaris",
    "paper-atcc": "Aegis",
}
RAW_FIELDS = (
    "run_id",
    "experiment",
    "parameter",
    "parameter_value",
    "workload",
    "clients",
    "worker_count",
    "seed",
    "repeat",
    "cc",
    "system",
    "status",
    "max_attempts",
    "retry_budget",
    "measurement_seconds",
    "zipf_theta",
    "reasoning_scale",
    "retry_delay_scale",
    "commit_apply_ms",
    "drain_seconds",
    "logical_tasks",
    "commits",
    "failed_tasks",
    "attempts",
    "aborts",
    "agent_tps",
    "commit_rate",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "wasted_reasoning_ms_per_commit",
    "wasted_tokens_per_commit",
    "useful_tokens_per_commit",
    "total_tokens_per_commit",
    "avg_operations_per_attempt",
    "commit_admission_wait_ms_per_attempt",
    "paper_read_lock_acquires",
    "paper_write_lock_acquires",
    "paper_lock_wait_events",
    "paper_lock_wait_ms",
    "paper_wounds",
    "paper_lock_timeouts",
    "paper_commit_admission_conflicts",
    "paper_retry_validation_conflicts",
    "branch_counts_json",
    "failure_reasons_json",
    "access_set_visibility",
    "admission_scope",
    "policy_commit_batch",
    "performance_guards",
    "tracked_history_import",
    "conflict_driven_fast_path",
    "unified_commit_admission",
    "cycle_safe_hotspot_admission",
    "cycle_safe_gate_wait_ms",
    "cycle_safe_gate_admissions",
    "cycle_safe_gate_bypasses",
    "coordinated_hotspot_admission",
    "commit_read_checkpoints",
    "first_access_read_refresh",
    "consistent_hotness_classification",
    "helper_externality_accounting",
    "helper_externality_recording_active",
    "mutable_read_before_reasoning",
    "coordinated_gate_admissions",
    "coordinated_gate_cycle_bypasses",
    "coordinated_gate_wait_calls",
    "coordinated_gate_priority_overtakes",
    "coordinated_gate_dependency_probes",
    "coordinated_gate_timeouts",
    "native_occ_fast_tasks",
    "observed_commit_admission",
    "planned_target_count",
    "error",
)
METRICS = (
    "agent_tps",
    "commit_rate",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "wasted_reasoning_ms_per_commit",
    "wasted_tokens_per_commit",
    "useful_tokens_per_commit",
    "total_tokens_per_commit",
    "avg_operations_per_attempt",
)


@dataclasses.dataclass
class WindowCounters:
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    logical_tasks: int = 0
    commits: int = 0
    failed_tasks: int = 0
    attempts: int = 0
    aborts: int = 0
    wasted_reasoning_ms: int = 0
    wasted_tokens: int = 0
    useful_tokens: int = 0
    total_tokens: int = 0
    operation_count: int = 0
    blocked_ms: float = 0.0
    lock_hold_ms: float = 0.0
    commit_admission_wait_ms: float = 0.0
    latencies_ms: list[float] = dataclasses.field(default_factory=list)
    branches: Counter[str] = dataclasses.field(default_factory=Counter)
    failures: Counter[str] = dataclasses.field(default_factory=Counter)
    observed_failed_restart_cost_ms: float = 0.0
    carried_prior_restart_cost_ms: float = 0.0
    restart_cost_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def record_attempt(
        self,
        *,
        committed: bool,
        execution: CreditReviewExecution,
        failure_reason: str,
        observed_restart_cost_ms: float = 0.,
        carried_prior_restart_cost_ms: float = 0.,
        restart_cost_record: dict[str, Any] | None = None,
    ) -> None:
        with self.lock:
            self.attempts += 1
            self.carried_prior_restart_cost_ms += float(carried_prior_restart_cost_ms)
            if restart_cost_record is not None:
                self.restart_cost_records.append(dict(restart_cost_record))
            self.total_tokens += int(execution.reasoning_tokens)
            self.operation_count += int(execution.operation_count)
            self.blocked_ms += execution.blocked_ms
            self.lock_hold_ms += execution.lock_hold_ms
            self.commit_admission_wait_ms += float(execution.commit_admission_wait_ms)
            if execution.branch:
                self.branches[execution.branch] += 1
            if committed:
                self.commits += 1
                self.useful_tokens += int(execution.reasoning_tokens)
            else:
                self.aborts += 1
                self.observed_failed_restart_cost_ms += float(observed_restart_cost_ms)
                self.wasted_reasoning_ms += int(execution.reasoning_ms)
                self.wasted_tokens += int(execution.reasoning_tokens)
                self.failures[str(failure_reason or "unknown")] += 1

    def record_task(self, *, committed: bool, latency_ms: float) -> None:
        with self.lock:
            self.logical_tasks += 1
            if committed:
                self.latencies_ms.append(float(latency_ms))
            else:
                self.failed_tasks += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-policy", type=Path, required=True)
    parser.add_argument("--clients", default="8,16,24,32,40")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=6,
        help="Total attempts including the initial attempt (paper default: 6).",
    )
    parser.add_argument("--seed-base", type=int, default=983_000)
    parser.add_argument("--company-count", type=int, default=256)
    parser.add_argument("--sector-count", type=int, default=8)
    parser.add_argument("--region-count", type=int, default=4)
    parser.add_argument("--zipf-theta", type=float, default=0.99)
    parser.add_argument("--reasoning-scale", type=float, default=1.0)
    parser.add_argument("--mutable-read-before-reasoning", action="store_true",
        help="Controlled long-read-window variant; preserves SQL count, reasoning and retry budgets.")
    parser.add_argument(
        "--retry-delay-scale",
        type=float,
        default=1.0,
        help="Scale the paper's deterministic 500-5000 ms retry replanning delay.",
    )
    parser.add_argument("--commit-apply-ms", type=int, default=24)
    parser.add_argument("--paper-switching", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--paper-priority", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--observed-commit-admission",
        action="store_true",
        help="Enable the workload-specific observed-target admission ablation.",
    )
    parser.add_argument(
        "--disable-policy-commit-batch",
        action="store_false",
        dest="policy_commit_batch",
        help="Disable applying the current ATCC write action to the materialized commit batch.",
    )
    parser.set_defaults(policy_commit_batch=True)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--paper-exploration-seed", type=int)
    parser.add_argument("--paper-exploration-epsilon", type=float, default=0.2)
    parser.add_argument("--priority-quantum-scale", type=float, default=1.0)
    parser.add_argument("--performance-guards", action="store_true")
    parser.add_argument("--tracked-history-import", action="store_true",
        help="Import only invalidated lazy baselines; requires all writes through the manager.")
    parser.add_argument("--conflict-driven-fast-path", action="store_true",
        help="With performance guards, qualify cold OCC from observed conflicts, not frequency.")
    parser.add_argument("--unified-commit-admission", action="store_true",
        help="Use the existing ATCC priority scheduler instead of private commit-suffix gates.")
    parser.add_argument("--coordinated-hotspot-admission", action="store_true",
        help="Priority-aware atomic hotspot helper reservations with cross-resource cycle inspection.")
    parser.add_argument("--round-interval-features", action="store_true",
        help="Use observed per-round Agent gaps instead of the final bookkeeping gap in policy state.")
    parser.add_argument("--commit-read-checkpoints", action="store_true",
        help="Add policy checkpoints after already-observed Commit reads; retain phase checkpoints.")
    parser.add_argument("--first-access-read-refresh", action="store_true",
        help="For unobserved unpinned protected reads, lock before reading the committed value.")
    parser.add_argument("--consistent-hotness-classification", action="store_true",
        help="On unsampled accesses use observed hotness instead of treating known hot objects as cold.")
    parser.add_argument("--helper-externality-accounting", action="store_true",
        help="Record reservation-owner waiting externalities in collected policy trajectories.")
    parser.add_argument("--cycle-safe-hotspot-admission", action="store_true",
        help="Break nested private-gate waits under ATCC locks; retain normal commit validation.")
    parser.add_argument("--cycle-safe-gate-wait-ms", type=float, default=0.,
        help="Optional total bounded helper wait under ATCC locks; default zero/nonblocking.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    clients = parse_positive_ints(args.clients)
    systems = tuple(value.strip() for value in str(args.systems).split(",") if value.strip())
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unsupported systems: {sorted(unknown)}")
    if args.repeats < 1 or args.max_attempts < 1:
        raise SystemExit("--repeats and --max-attempts must be positive")
    if args.warmup_seconds < 0.0 or args.measure_seconds <= 0.0:
        raise SystemExit("invalid warmup or measurement duration")
    if args.retry_delay_scale < 0.0:
        raise SystemExit("--retry-delay-scale must be non-negative")
    if args.priority_quantum_scale <= 0.0:
        raise SystemExit("--priority-quantum-scale must be positive")
    if args.conflict_driven_fast_path and not args.performance_guards:
        raise SystemExit("--conflict-driven-fast-path requires --performance-guards")
    if sum((args.unified_commit_admission, args.cycle_safe_hotspot_admission,
            args.coordinated_hotspot_admission)) > 1:
        raise SystemExit("Choose one hotspot admission mode, not both")
    if args.cycle_safe_gate_wait_ms < 0 or not math.isfinite(args.cycle_safe_gate_wait_ms):
        raise SystemExit("cycle-safe gate wait must be finite and non-negative")
    if args.cycle_safe_gate_wait_ms and not args.cycle_safe_hotspot_admission:
        raise SystemExit("cycle-safe gate wait requires cycle-safe hotspot admission")
    if not 0.0 <= args.paper_exploration_epsilon <= 1.0:
        raise SystemExit("--paper-exploration-epsilon must be in [0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "credit_review_figure3_raw.csv"
    existing = [] if args.force else read_csv(raw_path)
    prior_manifest=args.output_dir / "credit_review_manifest.json"
    if existing and prior_manifest.exists():
        previous=json.loads(prior_manifest.read_text(encoding="utf-8"))
        for name in ("tracked_history_import","conflict_driven_fast_path","unified_commit_admission","cycle_safe_hotspot_admission","coordinated_hotspot_admission","round_interval_features","commit_read_checkpoints","first_access_read_refresh","consistent_hotness_classification","helper_externality_accounting"):
            if bool(previous.get(name,False)) != bool(getattr(args,name)):
                raise SystemExit("Optimization flags differ from existing evidence; use a fresh output directory")
        if float(previous.get("cycle_safe_gate_wait_ms",0.)) != args.cycle_safe_gate_wait_ms:
            raise SystemExit("Admission wait budget differs from existing evidence; use a fresh output directory")
        if bool(previous.get("mutable_read_before_reasoning",False)) != args.mutable_read_before_reasoning:
            raise SystemExit("Read exposure differs from existing evidence; use a fresh output directory")
    completed = {
        (int(row["clients"]), int(row["repeat"]), row["cc"])
        for row in existing
        if row.get("status") == "ok"
    }
    rows = list(existing)
    config = CreditReviewConfig(
        company_count=args.company_count,
        sector_count=args.sector_count,
        region_count=args.region_count,
        zipf_theta=args.zipf_theta,
        reasoning_scale=args.reasoning_scale,
        commit_apply_ms=args.commit_apply_ms,
        mutable_read_before_reasoning=args.mutable_read_before_reasoning,
    ).normalized()
    for client_count in clients:
        for repeat in range(args.repeats):
            seed = case_seed(args.seed_base, client_count, repeat)
            for system in systems:
                key = (client_count, repeat, system)
                if key in completed:
                    continue
                print(
                    f"[credit-review] clients={client_count} repeat={repeat} system={system}",
                    flush=True,
                )
                try:
                    row = run_system(
                        config=config,
                        clients=client_count,
                        seed=seed,
                        repeat=repeat,
                        system=system,
                        paper_policy=args.paper_policy,
                        warmup_seconds=args.warmup_seconds,
                        measure_seconds=args.measure_seconds,
                        max_attempts=args.max_attempts,
                        retry_delay_scale=args.retry_delay_scale,
                        paper_switching=args.paper_switching,
                        paper_priority=args.paper_priority,
                        priority_quantum_scale=args.priority_quantum_scale,
                        performance_guards=args.performance_guards,
                        tracked_history_import=args.tracked_history_import,
                        conflict_driven_fast_path=args.conflict_driven_fast_path,
                        unified_commit_admission=args.unified_commit_admission,
                        cycle_safe_hotspot_admission=args.cycle_safe_hotspot_admission,
                        cycle_safe_gate_wait_ms=args.cycle_safe_gate_wait_ms,
                        coordinated_hotspot_admission=args.coordinated_hotspot_admission,
                        round_interval_features=args.round_interval_features,
                        commit_read_checkpoints=args.commit_read_checkpoints,
                        first_access_read_refresh=args.first_access_read_refresh,
                        consistent_hotness_classification=args.consistent_hotness_classification,
                        helper_externality_accounting=args.helper_externality_accounting,
                        observed_commit_admission=args.observed_commit_admission,
                        policy_commit_batch=args.policy_commit_batch,
                        trajectory_output=(
                            args.trajectory_dir
                            / f"credit_review_c{client_count}_r{repeat}_s{seed}.json"
                            if args.trajectory_dir is not None and system == "paper-atcc"
                            else None
                        ),
                        paper_exploration_seed=(
                            args.paper_exploration_seed + client_count * 17 + repeat * 1009
                            if args.paper_exploration_seed is not None
                            and system == "paper-atcc"
                            else None
                        ),
                        paper_exploration_epsilon=args.paper_exploration_epsilon,
                    )
                except BaseException as exc:
                    row = error_row(client_count, seed, repeat, system, exc)
                rows.append(row)
                write_csv(raw_path, rows, RAW_FIELDS)
                if row["status"] != "ok":
                    raise RuntimeError(row["error"])

    rows.sort(key=lambda row: (int(row["clients"]), int(row["repeat"]), SYSTEMS.index(row["cc"])))
    write_csv(raw_path, rows, RAW_FIELDS)
    summary = summarize(rows)
    summary_path = args.output_dir / "credit_review_figure3_summary.csv"
    plot_path = args.output_dir / "credit_review_figure3_plot_data_clean.csv"
    write_csv(summary_path, summary, tuple(summary[0]) if summary else ())
    write_csv(plot_path, summary, tuple(summary[0]) if summary else ())
    manifest_path = args.output_dir / "credit_review_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clients": clients,
                "repeats": args.repeats,
                "systems": systems,
                "warmup_seconds": args.warmup_seconds,
                "measure_seconds": args.measure_seconds,
                "max_attempts": args.max_attempts,
                "retry_budget": max(0, int(args.max_attempts) - 1),
                "retry_delay_scale": args.retry_delay_scale,
                "retry_delay_range_ms": [500, 5000],
                "seed_base": args.seed_base,
                "paper_policy": str(args.paper_policy.resolve()),
                "paper_switching": args.paper_switching,
                "paper_priority": args.paper_priority,
                "delayed_write_apply": True,
                "performance_guards": bool(args.performance_guards),
                "tracked_history_import": bool(args.tracked_history_import),
                "conflict_driven_fast_path": bool(args.conflict_driven_fast_path),
                "unified_commit_admission": bool(args.unified_commit_admission),
                "cycle_safe_hotspot_admission": bool(args.cycle_safe_hotspot_admission),
                "cycle_safe_gate_wait_ms": float(args.cycle_safe_gate_wait_ms),
                "coordinated_hotspot_admission": bool(args.coordinated_hotspot_admission),
                "round_interval_features": bool(args.round_interval_features),
                "commit_read_checkpoints": bool(args.commit_read_checkpoints),
                "first_access_read_refresh": bool(args.first_access_read_refresh),
                "consistent_hotness_classification": bool(args.consistent_hotness_classification),
                "helper_externality_accounting": bool(args.helper_externality_accounting),
                "mutable_read_before_reasoning": bool(args.mutable_read_before_reasoning),
                "observed_commit_admission": bool(args.observed_commit_admission),
                "policy_commit_batch": bool(args.policy_commit_batch),
                "trajectory_dir": str(args.trajectory_dir.resolve()) if args.trajectory_dir else "",
                "paper_exploration_seed": args.paper_exploration_seed,
                "paper_exploration_epsilon": args.paper_exploration_epsilon,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(raw_path)
    print(plot_path)
    print_acceptance(summary)
    return 0


def run_system(
    *,
    config: CreditReviewConfig,
    clients: int,
    seed: int,
    repeat: int,
    system: str,
    paper_policy: Path,
    warmup_seconds: float,
    measure_seconds: float,
    max_attempts: int,
    retry_delay_scale: float = 1.0,
    paper_switching: str = "dynamic",
    paper_priority: str = "enabled",
    priority_quantum_scale: float = 1.0,
    performance_guards: bool = False,
    observed_commit_admission: bool = False,
    policy_commit_batch: bool = True,
    trajectory_output: Path | None = None,
    paper_exploration_seed: int | None = None,
    paper_exploration_epsilon: float = 0.2,
    policy_override: object | None = None,
    tracked_history_import: bool = False,
    conflict_driven_fast_path: bool = False,
    unified_commit_admission: bool = False,
    cycle_safe_hotspot_admission: bool = False,
    cycle_safe_gate_wait_ms: float = 0.,
    coordinated_hotspot_admission: bool = False,
    round_interval_features: bool = False,
    commit_read_checkpoints: bool = False,
    first_access_read_refresh: bool = False,
    consistent_hotness_classification: bool = False,
    helper_externality_accounting: bool = False,
    observed_write_read_lock_elision: bool = False,
    protection_relevant_hotness: bool = False,
    unique_attempt_outcomes: bool = False,
    carry_prior_restart_cost: bool = False,
) -> dict[str, Any]:
    if conflict_driven_fast_path and not performance_guards:
        raise ValueError("conflict-driven fast path requires performance guards")
    if sum((unified_commit_admission, cycle_safe_hotspot_admission,
            coordinated_hotspot_admission)) > 1:
        raise ValueError("Choose one hotspot admission mode, not both")
    if cycle_safe_gate_wait_ms < 0 or not math.isfinite(cycle_safe_gate_wait_ms):
        raise ValueError("cycle-safe gate wait must be finite and non-negative")
    if cycle_safe_gate_wait_ms and not cycle_safe_hotspot_admission:
        raise ValueError("cycle-safe gate wait requires cycle-safe hotspot admission")
    workload = CreditReviewWorkload(config)
    compiled_policy = policy_override if policy_override is not None else CompiledPhasePolicy.load(paper_policy)
    encoder_policy = compiled_policy
    encoder_seen = set()
    while id(encoder_policy) not in encoder_seen:
        encoder_seen.add(id(encoder_policy))
        if (getattr(encoder_policy, "protection_relevant_hotness", False)
                and not protection_relevant_hotness):
            raise ValueError("policy requires protection-relevant hotness encoding")
        nested = getattr(encoder_policy, "base", None)
        if nested is None:
            break
        encoder_policy = nested
    if paper_exploration_seed is not None:
        compiled_policy = EpsilonGreedyPolicy(
            compiled_policy,
            seed=paper_exploration_seed,
            epsilon=paper_exploration_epsilon,
        )
    manager = AgentTransactionManager(
        cc_registry=ConcurrencyControlRegistry(),
        record_traces=False,
        paper_policy=(
            StaticThresholdPhasePolicy()
            if str(paper_switching).strip().lower() == "static"
            else compiled_policy
        ),
        collect_trajectories=trajectory_output is not None,
        low_conflict_occ_guard=bool(performance_guards),
        performance_guards_enabled=bool(performance_guards),
        commit_admission_priority_enabled=False,
        delayed_write_apply_enabled=True,
        tracked_history_import=tracked_history_import,
        conflict_driven_fast_path=conflict_driven_fast_path,
        unified_commit_admission=unified_commit_admission,
        cycle_safe_hotspot_admission=cycle_safe_hotspot_admission,
        cycle_safe_gate_wait_ms=cycle_safe_gate_wait_ms,
        coordinated_hotspot_admission=coordinated_hotspot_admission,
        round_interval_features=round_interval_features,
        commit_read_checkpoints=commit_read_checkpoints,
        first_access_read_refresh=first_access_read_refresh,
        consistent_hotness_classification=consistent_hotness_classification,
        helper_externality_accounting=helper_externality_accounting,
        observed_write_read_lock_elision=observed_write_read_lock_elision,
        protection_relevant_hotness=protection_relevant_hotness,
        unique_attempt_outcomes=unique_attempt_outcomes,
        priority_config=PriorityConfig(
            sql_quantum_ms=10.0 * float(priority_quantum_scale),
            interval_quantum_ms=10.0 * float(priority_quantum_scale),
            blocked_quantum_ms=100.0 * float(priority_quantum_scale),
        ),
        priority_enabled=str(paper_priority).strip().lower() == "enabled",
    )
    workload.register(manager)
    if warmup_seconds > 0.0:
        run_window(
            manager,
            workload,
            system=system,
            clients=clients,
            seed=seed,
            duration_s=warmup_seconds,
            max_attempts=max_attempts,
            sequence_offset=-1_000_000,
            retry_delay_scale=retry_delay_scale,
            carry_prior_restart_cost=carry_prior_restart_cost,
            observed_commit_admission=observed_commit_admission,
            policy_commit_batch=policy_commit_batch,
        )
        manager.trajectory_collector.clear()
        manager.reset_measurement_diagnostics()
        reset_policy = getattr(compiled_policy, "reset", None)
        if reset_policy is not None:
            reset_policy()
    fast_starts_before=manager.native_occ_fast_starts
    cycle_safe_before=manager.cycle_safe_gate_diagnostics()
    coordinated_before=manager.coordinated_gate_diagnostics()
    outcomes_before=manager.runtime_outcome_diagnostics()
    counters, drain_s = run_window(
        manager,
        workload,
        system=system,
        clients=clients,
        seed=seed,
        duration_s=measure_seconds,
        max_attempts=max_attempts,
        sequence_offset=0,
        retry_delay_scale=retry_delay_scale,
        carry_prior_restart_cost=carry_prior_restart_cost,
        record_restart_costs=carry_prior_restart_cost and trajectory_output is not None,
        observed_commit_admission=observed_commit_admission,
        policy_commit_batch=policy_commit_batch,
    )
    if trajectory_output is not None and system == "paper-atcc":
        helper_accounting = None
        if helper_externality_accounting:
            coordinator = manager._coordinated_admission
            final = coordinator.snapshot()
            assert not final["owners"] and not final["pending"], "finalize only after reservations drain"
            totals = coordinator.externalities_snapshot()
            helper_accounting = dict(recording_active=coordinator.record_externalities,
                totals_by_owner=totals,
                reconciliation=manager.trajectory_collector.reconcile_helper_externalities(totals),
                components_by_transition=manager.trajectory_collector.helper_component_snapshot(),
                predecision_costs_by_owner=manager.trajectory_collector.helper_predecision_snapshot(),
                scope="reservation-owner Event-wait segments, not barriers or native waits")
        transitions = [
            {
                **dataclasses.asdict(transition),
                "state": dataclasses.asdict(transition.state),
                "next_state": dataclasses.asdict(transition.next_state),
            }
            for transition in manager.trajectory_collector.snapshot()
        ]
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        trajectory_output.write_text(
            json.dumps({"transitions": transitions,
                **({"attempt_restart_costs": counters.restart_cost_records} if carry_prior_restart_cost else {}),
                **({"helper_externalities": helper_accounting}
                if helper_accounting is not None else {})}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    commits = int(counters.commits)
    attempts = int(counters.attempts)
    paper_diagnostics = manager.atcc_locks.snapshot_diagnostics()
    retry_diagnostics = manager.retry_protection_diagnostics()
    return {
        "run_id": f"credit_review_c{clients}_r{repeat}_s{seed}_{system}",
        "experiment": "agentic_native_credit_review",
        "parameter": "workers",
        "parameter_value": clients,
        "workload": "CreditReview",
        "clients": clients,
        "worker_count": clients,
        "seed": seed,
        "repeat": repeat,
        "cc": system,
        "system": SYSTEM_LABELS[system],
        "status": "ok",
        "max_attempts": int(max_attempts),
        "retry_budget": max(0, int(max_attempts) - 1),
        "measurement_seconds": measure_seconds,
        "zipf_theta": config.zipf_theta,
        "reasoning_scale": config.reasoning_scale,
        "retry_delay_scale": float(retry_delay_scale),
        "commit_apply_ms": config.commit_apply_ms,
        "drain_seconds": drain_s,
        "logical_tasks": counters.logical_tasks,
        "commits": commits,
        "failed_tasks": counters.failed_tasks,
        "attempts": attempts,
        "aborts": counters.aborts,
        "agent_tps": commits / max(0.001, measure_seconds),
        "elapsed_task_tps": commits / max(0.001, measure_seconds + drain_s),
        "abort_rate": counters.aborts / attempts if attempts else 0.0,
        "task_completion_rate": commits / counters.logical_tasks if counters.logical_tasks else 0.0,
        "blocked_ms_total": counters.blocked_ms,
        "lock_hold_ms_total": counters.lock_hold_ms,
        "commit_rate": commits / attempts if attempts else 0.0,
        "p50_latency_ms": percentile(counters.latencies_ms, 0.50),
        "p95_latency_ms": percentile(counters.latencies_ms, 0.95),
        "p99_latency_ms": percentile(counters.latencies_ms, 0.99),
        "wasted_reasoning_ms_per_commit": counters.wasted_reasoning_ms / commits if commits else 0.0,
        "wasted_tokens_per_commit": counters.wasted_tokens / commits if commits else 0.0,
        "useful_tokens_per_commit": counters.useful_tokens / commits if commits else 0.0,
        "total_tokens_per_commit": counters.total_tokens / commits if commits else 0.0,
        "avg_operations_per_attempt": counters.operation_count / attempts if attempts else 0.0,
        "commit_admission_wait_ms_per_attempt": (
            counters.commit_admission_wait_ms / attempts if attempts else 0.0
        ),
        "paper_read_lock_acquires": paper_diagnostics.get("read_lock_acquires", 0),
        "paper_write_lock_acquires": paper_diagnostics.get("write_lock_acquires", 0),
        "paper_lock_wait_events": paper_diagnostics.get("lock_wait_events", 0),
        "paper_lock_wait_ms": paper_diagnostics.get("lock_wait_ms", 0.0),
        "paper_wounds": paper_diagnostics.get("wounds", 0),
        "paper_lock_timeouts": paper_diagnostics.get("lock_timeouts", 0),
        "paper_commit_admission_conflicts": paper_diagnostics.get(
            "commit_admission_conflicts", 0
        ),
        "paper_retry_validation_conflicts": retry_diagnostics.get(
            "validation_conflicts", 0
        ),
        "branch_counts_json": json.dumps(dict(sorted(counters.branches.items())), sort_keys=True),
        "failure_reasons_json": json.dumps(dict(sorted(counters.failures.items())), sort_keys=True),
        "access_set_visibility": "online_observed",
        "admission_scope": (
            "observed_commit_suffix"
            if system == "paper-atcc" and observed_commit_admission
            else "policy_commit_batch"
            if system == "paper-atcc" and policy_commit_batch
            else "policy_action" if system == "paper-atcc" else "none"
        ),
        "policy_commit_batch": bool(policy_commit_batch),
        "tracked_history_import": bool(tracked_history_import),
        "conflict_driven_fast_path": bool(conflict_driven_fast_path),
        "unified_commit_admission": bool(unified_commit_admission),
        "cycle_safe_hotspot_admission": bool(cycle_safe_hotspot_admission),
        "cycle_safe_gate_wait_ms": float(cycle_safe_gate_wait_ms),
        "cycle_safe_gate_admissions": manager.cycle_safe_gate_diagnostics()["admissions"]-cycle_safe_before["admissions"],
        "cycle_safe_gate_bypasses": manager.cycle_safe_gate_diagnostics()["bypasses"]-cycle_safe_before["bypasses"],
        "coordinated_hotspot_admission": bool(coordinated_hotspot_admission),
        "round_interval_features": bool(round_interval_features),
        "commit_read_checkpoints": bool(commit_read_checkpoints),
        "first_access_read_refresh": bool(first_access_read_refresh),
        "consistent_hotness_classification": bool(consistent_hotness_classification),
        "helper_externality_accounting": bool(helper_externality_accounting),
        "observed_write_read_lock_elision": bool(observed_write_read_lock_elision),
        "protection_relevant_hotness": bool(protection_relevant_hotness),
        "unique_attempt_outcomes": bool(unique_attempt_outcomes),
        "carry_prior_restart_cost": bool(carry_prior_restart_cost),
        "observed_failed_restart_cost_ms_total": counters.observed_failed_restart_cost_ms,
        "carried_prior_restart_cost_ms_total": counters.carried_prior_restart_cost_ms,
        **{f"runtime_outcomes_{key}": value-outcomes_before[key]
           for key, value in manager.runtime_outcome_diagnostics().items()},
        "helper_externality_recording_active": bool(manager._coordinated_admission is not None
            and manager._coordinated_admission.record_externalities),
        "mutable_read_before_reasoning": bool(workload.config.mutable_read_before_reasoning),
        **{f"coordinated_gate_{key}": manager.coordinated_gate_diagnostics().get(key, 0.)
           -coordinated_before.get(key, 0.) for key in ("admissions", "cycle_bypasses", "wait_calls",
              "priority_overtakes", "dependency_probes", "timeouts")},
        "performance_guards": bool(performance_guards),
        "native_occ_fast_tasks": manager.native_occ_fast_starts-fast_starts_before,
        "lazy_baseline_store_reads": manager.version_manager.snapshot_diagnostics().get(
            "lazy_baseline_store_reads", 0),
        "observed_commit_admission": bool(observed_commit_admission),
        "planned_target_count": 0,
        "error": "",
    }


def run_window(
    manager: AgentTransactionManager,
    workload: CreditReviewWorkload,
    *,
    system: str,
    clients: int,
    seed: int,
    duration_s: float,
    max_attempts: int,
    sequence_offset: int,
    retry_delay_scale: float = 1.0,
    observed_commit_admission: bool = False,
    policy_commit_batch: bool = True,
    carry_prior_restart_cost: bool = False,
    record_restart_costs: bool = False,
) -> tuple[WindowCounters, float]:
    if record_restart_costs and not carry_prior_restart_cost:
        raise ValueError("restart cost records require actual prior-cost propagation")
    counters = WindowCounters()
    started_at = 0.0
    deadline = 0.0
    barrier = threading.Barrier(clients + 1)
    worker_errors: list[BaseException] = []
    worker_error_lock = threading.Lock()

    def worker(worker_id: int) -> None:
        nonlocal deadline
        try:
            barrier.wait()
            sequence = 0
            while time.perf_counter() < deadline:
                logical_sequence = sequence_offset + sequence
                task = workload.task_for(seed=seed, worker_id=worker_id, sequence=logical_sequence)
                task_id = f"credit-review:{seed}:{worker_id}:{logical_sequence}"
                task_started = time.perf_counter()
                committed = False
                prior_restart_cost = 0.
                previous_failure_reason = "none"
                for attempt in range(max_attempts):
                    attempt_diagnostics = {} if carry_prior_restart_cost else None
                    result, execution = execute_attempt(
                        manager,
                        workload,
                        task,
                        task_id=task_id,
                        system=system,
                        attempt=attempt,
                        retry_delay_scale=retry_delay_scale,
                        observed_commit_admission=observed_commit_admission,
                        policy_commit_batch=policy_commit_batch,
                        **({"prior_retry_cost_ms": prior_restart_cost,
                            "previous_failure_reason": previous_failure_reason,
                            "attempt_diagnostics": attempt_diagnostics} if carry_prior_restart_cost else {}),
                    )
                    counters.record_attempt(
                        committed=bool(result.committed),
                        execution=execution,
                        failure_reason=str(result.reason),
                        observed_restart_cost_ms=(attempt_diagnostics['restart_cost_ms']
                            if carry_prior_restart_cost else 0.),
                        carried_prior_restart_cost_ms=prior_restart_cost,
                        restart_cost_record=(dict(attempt_diagnostics, task_id=task_id, attempt=attempt,
                            prior_restart_cost_ms=prior_restart_cost, previous_failure_reason=previous_failure_reason,
                            committed=bool(result.committed), failure_reason=str(result.reason))
                            if record_restart_costs else None),
                    )
                    if result.committed:
                        committed = True
                        break
                    if carry_prior_restart_cost:
                        prior_restart_cost += attempt_diagnostics['restart_cost_ms']
                        previous_failure_reason = str(result.reason)
                latency_ms = (time.perf_counter() - task_started) * 1000.0
                counters.record_task(committed=committed, latency_ms=latency_ms)
                if system == "paper-atcc":
                    manager.note_agent_task_outcome(committed=committed, latency_ms=latency_ms)
                sequence += 1
        except BaseException as exc:
            with worker_error_lock:
                worker_errors.append(exc)

    threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(clients)]
    for thread in threads:
        thread.start()
    started_at = time.perf_counter()
    deadline = started_at + max(0.001, float(duration_s))
    barrier.wait()
    for thread in threads:
        thread.join()
    if worker_errors:
        first_error = worker_errors[0]
        raise RuntimeError(
            f"{len(worker_errors)} credit-review worker(s) failed; first="
            f"{type(first_error).__name__}: {first_error}"
        ) from first_error
    finished_at = time.perf_counter()
    return counters, max(0.0, finished_at - deadline)


def execute_attempt(
    manager: AgentTransactionManager,
    workload: CreditReviewWorkload,
    task: CreditReviewTaskSpec,
    *,
    task_id: str,
    system: str,
    attempt: int,
    retry_delay_scale: float = 1.0,
    observed_commit_admission: bool = False,
    policy_commit_batch: bool = True,
    prior_retry_cost_ms: float | None = None,
    previous_failure_reason: str | None = None,
    attempt_diagnostics: dict[str, Any] | None = None,
) -> tuple[Any, CreditReviewExecution]:
    if prior_retry_cost_ms is not None and (
            not math.isfinite(prior_retry_cost_ms) or prior_retry_cost_ms < 0.):
        raise ValueError("prior retry cost must be finite observed nonnegative work")
    sleep_for_reasoning(
        credit_retry_delay_ms(
            task_id=task_id,
            attempt=attempt,
            retry_delay_scale=retry_delay_scale,
        )
    )
    metadata = transaction_metadata(
        system,
        retry_count=attempt,
        observed_commit_admission=observed_commit_admission,
        policy_commit_batch=policy_commit_batch,
    )
    if prior_retry_cost_ms is not None:
        metadata['prior_retry_cost_ms'] = float(prior_retry_cost_ms)
    if previous_failure_reason is not None:
        metadata['previous_failure_reason'] = str(previous_failure_reason)
    txn = manager.begin(task_id, metadata, snapshot_object_ids=None, strategy=system)
    cursor = workload.cursor(task)
    try:
        execution = cursor.execute(
            txn,
            before_access=None,
            before_commit_batch=commit_batch_callback(
                manager,
                system=system,
                observed_commit_admission=observed_commit_admission,
                policy_commit_batch=policy_commit_batch,
            ),
        )
        result = txn.commit(system)
        return result, dataclasses.replace(execution, blocked_ms=txn.context.blocked_time_ms,
                                          lock_hold_ms=txn.context.lock_hold_time_ms)
    except LockConflict as exc:
        if txn.result is not None:
            return txn.result, dataclasses.replace(cursor.snapshot(), blocked_ms=txn.context.blocked_time_ms,
                                                  lock_hold_ms=txn.context.lock_hold_time_ms)
        result = txn.abort(
            exc.reason,
            strategy=system,
            conflict_object_ids=exc.targets,
        )
        return result, dataclasses.replace(cursor.snapshot(), blocked_ms=txn.context.blocked_time_ms,
                                          lock_hold_ms=txn.context.lock_hold_time_ms)
    except BaseException:
        if txn.result is None:
            with contextlib.suppress(BaseException):
                txn.abort("credit-review execution error", strategy=system)
        raise
    finally:
        if attempt_diagnostics is not None:
            # This is completed post-begin work from the real attempt, not its
            # requested budget, future retry sleep, lock wait or a fake cost.
            agent_cost = float(txn.context.agent_cost_ms)
            operation_cost = float(txn.context.operation_cost_ms)
            if not all(math.isfinite(value) and value >= 0. for value in (agent_cost, operation_cost)):
                raise ValueError("invalid observed attempt restart cost")
            attempt_diagnostics.update(agent_cost_ms=agent_cost, operation_cost_ms=operation_cost,
                                       restart_cost_ms=agent_cost+operation_cost, txn_id=txn.context.tid)


def credit_retry_delay_ms(
    *,
    task_id: str,
    attempt: int,
    retry_delay_scale: float = 1.0,
) -> int:
    """Return the deterministic paper retry delay used by trace workloads."""
    profile = ReasoningProfile(
        "agentic",
        scale=1.0,
        retry_scale=max(0.0, float(retry_delay_scale)),
    )
    return profile.retry_delay_ms(
        level="high",
        task_id=str(task_id),
        attempt=int(attempt),
    )


def acquire_observed_commit_admission(
    manager: AgentTransactionManager,
    txn: Any,
    targets: tuple[str, ...],
) -> float:
    protected = tuple(
        sorted(
            {
                str(target)
                for target in targets
                if str(target).startswith(("credit:portfolio:", "credit:committee:"))
                or str(target).startswith("credit:compliance:")
                or str(target).endswith((":exposure", ":review_queue"))
            }
        )
    )
    if not protected:
        return 0.0
    started = time.perf_counter()
    if not manager.acquire_hotspot_admission(txn, protected, timeout_s=5.0):
        raise LockConflict(
            "observed access admission timeout",
            protected,
            kind="lock-timeout",
        )
    return (time.perf_counter() - started) * 1000.0


def commit_batch_callback(
    manager: AgentTransactionManager,
    *,
    system: str,
    observed_commit_admission: bool,
    policy_commit_batch: bool,
):
    if system != "paper-atcc":
        return None
    if observed_commit_admission:
        return lambda txn, targets: acquire_observed_commit_admission(
            manager, txn, targets
        )
    if policy_commit_batch:
        return lambda txn, targets: acquire_policy_commit_batch(
            manager, txn, targets
        )
    return None


def acquire_policy_commit_batch(
    manager: AgentTransactionManager,
    txn: Any,
    targets: tuple[str, ...],
) -> float:
    protected = tuple(
        sorted(
            target
            for target in {str(value) for value in targets}
            if manager.hotness_tracker.commit_admission_needed(target)
        )
    )
    if not protected:
        return 0.0
    started = time.perf_counter()
    if not manager.acquire_hotspot_admission(txn, protected, timeout_s=5.0):
        raise LockConflict(
            "adaptive commit-batch admission timeout",
            protected,
            kind="lock-timeout",
        )
    txn.metadata.setdefault("_policy_commit_batch_targets", set()).update(protected)
    return (time.perf_counter() - started) * 1000.0


def transaction_metadata(
    system: str,
    *,
    retry_count: int,
    observed_commit_admission: bool = False,
    policy_commit_batch: bool = True,
) -> dict[str, Any]:
    return {
        "workload": "credit_review",
        "task_type": "credit_limit_review",
        "strategy": str(system),
        "retry_count": int(retry_count),
        "access_set_visibility": "online_observed",
        "admission_scope": (
            "observed_commit_suffix"
            if system == "paper-atcc" and observed_commit_admission
            else "policy_commit_batch"
            if system == "paper-atcc" and policy_commit_batch
            else "policy_action" if system == "paper-atcc" else "none"
        ),
        "policy_commit_batch": bool(policy_commit_batch),
        "observed_commit_admission": bool(observed_commit_admission),
        "planned_write_targets": [],
        "context": {
            "level": "high",
            "profile": "paper",
            "phase_shape": "tool_heavy",
            "agent_cost_class": "expensive",
        },
        "agentic": {"background_workers": 0},
    }


def summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault((int(row["clients"]), str(row["cc"])), []).append(row)
    output = []
    for (clients, system), group in sorted(
        groups.items(), key=lambda item: (item[0][0], SYSTEMS.index(item[0][1]))
    ):
        summary: dict[str, Any] = {
            "figure": "Figure 3",
            "experiment": "agentic_native_credit_review",
            "parameter": "workers",
            "parameter_value": clients,
            "workload": "CreditReview",
            "system": SYSTEM_LABELS[system],
            "n_seeds": len({int(row["seed"]) for row in group}),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def print_acceptance(summary: list[dict[str, Any]]) -> None:
    for clients in sorted({int(row["parameter_value"]) for row in summary}):
        rows = [row for row in summary if int(row["parameter_value"]) == clients]
        aegis = next((row for row in rows if row["system"] == "Aegis"), None)
        baselines = [row for row in rows if row["system"] != "Aegis"]
        if aegis is None or not baselines:
            continue
        best_tps = max(float(row["agent_tps_mean"]) for row in baselines)
        best_p99 = min(float(row["p99_latency_ms_mean"]) for row in baselines)
        best_tokens = min(float(row["wasted_tokens_per_commit_mean"]) for row in baselines)
        speedup = float(aegis["agent_tps_mean"]) / best_tps if best_tps else math.inf
        p99_reduction = 1.0 - float(aegis["p99_latency_ms_mean"]) / best_p99 if best_p99 else 0.0
        token_reduction = (
            best_tokens / float(aegis["wasted_tokens_per_commit_mean"])
            if float(aegis["wasted_tokens_per_commit_mean"]) > 0.0
            else math.inf
        )
        print(
            json.dumps(
                {
                    "clients": clients,
                    "speedup": speedup,
                    "commit_rate": float(aegis["commit_rate_mean"]),
                    "p99_reduction": p99_reduction,
                    "token_reduction": token_reduction,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(float(quantile) * len(ordered)) - 1))
    return ordered[index]


def parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise SystemExit("--clients must contain positive integers")
    return values


def case_seed(seed_base: int, clients: int, repeat: int) -> int:
    payload = f"credit-review:{clients}:{repeat}".encode("utf-8")
    return int(seed_base) + zlib.crc32(payload) % 10_000


def error_row(clients: int, seed: int, repeat: int, system: str, exc: BaseException) -> dict[str, Any]:
    row = {field: "" for field in RAW_FIELDS}
    row.update(
        {
            "run_id": f"credit_review_c{clients}_r{repeat}_s{seed}_{system}",
            "experiment": "agentic_native_credit_review",
            "parameter": "workers",
            "parameter_value": clients,
            "workload": "CreditReview",
            "clients": clients,
            "worker_count": clients,
            "seed": seed,
            "repeat": repeat,
            "cc": system,
            "system": SYSTEM_LABELS.get(system, system),
            "status": "error",
            "access_set_visibility": "online_observed",
            "admission_scope": "unknown",
            "policy_commit_batch": "",
            "observed_commit_admission": "",
            "planned_target_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    if not fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
