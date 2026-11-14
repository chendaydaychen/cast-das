#!/usr/bin/env python3
"""Fresh, fixed YCSB phase-window comparison for internal rebuttal study.

The workload is deliberately specified before execution: paired reader/writer
workers share one key.  The reader observes it in explore and reads it again
before a deterministic refine/reasoning interval; the writer reaches the same
key near commit.  This gives a phase-aware policy a real protection-timing
choice while hot-only rules see only their fixed online hotness/risk summary.
All five arms use the same trace, retry budget, runtime, and no warmup priming.

This is a mechanism stress test, not a replacement for the paper's Figure 12
benchmark.  Results are retained verbatim, including failures and retries.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.runtime import AgentTransactionManager, PaperRewardConfig, apply_paper_rewards
from agent.runtime.context import LockAction, LockClass
from agent.runtime.paper_policy import CompiledPhasePolicy
from agent.runtime.trajectory import policy_transition_from_dict
from agent.runtime.state_collector import phase_aware_state_from_dict
from agent.cc.atcc.ppo import (
    DiscretePPOPolicy,
    DiscretePPOTrainer,
    PPOConfig,
    discounted_returns,
    excluded_policy_transition,
    state_features,
)
from scripts.revision.run_p0_revision import dump, csv_write, digest
from scripts.unified_trace import run_castdas_trace_fair as runner


ARMS = (
    ("static_delay_priority", "static-delay-priority"),
    ("base_linear", "linear"),
    ("hot_threshold", "hot-threshold"),
    ("hot_always", "hot-always"),
    ("aegis_nophase", "nophase"),
    ("aegis_full", "full"),
)

PROJECT_NO_PHASE_PROGRESS = False
PROJECT_NO_PHASE_DIRECT_COUNTERS = False
PROJECT_NO_PHASE_OPERATION_COUNTER = False
PROJECT_NO_PHASE_ROUND_COUNTER = False
PHASE_ACTION_GATING = False
FORCE_AGENT_PHASE_PROTECTION = False
COST_BASED_PRIORITY = False
DEFER_POLICY_UNTIL_COMMIT = False
DEFERRED_OBJECT_PLAN = False
DEFERRED_OBJECT_PLAN_MIN_REDO_MS = 0.0
COVERAGE_GUARD = False
COVERAGE_MIN_SAMPLES = 4
MATCHED_LEARNED_RUNTIME = False
OBSERVED_TOOL_ROUNDS = False
EXPENSIVE_OPERATION_CHECKPOINT = False
EXPENSIVE_OPERATION_CHECKPOINT_FRACTION = 1.0
BINARY_PHASE_ACTIONS = False
COMMON_DELAY_MODE = "legacy"
COMMON_POLICY_CHECKPOINTS = False
ACCOUNT_PREBEGIN_RETRY_COST = False
COST_AWARE_READ_ELISION = False
INFLIGHT_PRIORITY = False


def delayed_write_mode(*, static=False):
    if COMMON_DELAY_MODE not in ("legacy", "enabled", "disabled"):
        raise ValueError("invalid common delay mode")
    return ("enabled" if static else "disabled") if COMMON_DELAY_MODE == "legacy" else COMMON_DELAY_MODE

REMOVED_PROGRESS = (
    "inter_round_interval_ms",
    "read_set_growth",
    "write_set_growth",
    "access_overlap_ratio",
    "completed_rounds",
    "completed_operations",
    "recent_write_ratio",
)


def without_phase_progress(state):
    return dataclasses.replace(
        state,
        phase="all",
        **{name: 0 for name in REMOVED_PROGRESS},
    )


def without_phase_direct_counters(state):
    """Remove the label and counters that directly encode phase position."""
    return dataclasses.replace(
        state,
        phase="all",
        completed_rounds=0,
        completed_operations=0,
    )


def without_phase_operation_counter(state):
    return dataclasses.replace(state, phase="all", completed_operations=0)


def without_phase_round_counter(state):
    return dataclasses.replace(state, phase="all", completed_rounds=0)


def no_phase_projector(*, progress=False, direct=False, operations=False, rounds=False):
    if progress:
        return without_phase_progress
    if direct:
        return without_phase_direct_counters
    if operations:
        return without_phase_operation_counter
    if rounds:
        return without_phase_round_counter
    return lambda state: dataclasses.replace(state, phase="all")


def _solve(matrix, rhs):
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    size = len(augmented)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) <= 1e-12:
            raise ValueError("singular linear baseline fit")
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [
                    left - scale * right
                    for left, right in zip(augmented[row], augmented[column])
                ]
    return [row[-1] for row in augmented]


def fit_linear(rows, ridge, return_targets=None):
    samples = {}
    targets = discounted_returns(rows, 0.99) if return_targets is None else return_targets
    if len(targets) != len(rows):
        raise ValueError("linear return targets must match transition count")
    for transition, target in zip(rows, targets):
        if not excluded_policy_transition(transition):
            samples.setdefault(transition.action, []).append(
                (state_features(transition.state), target)
            )
    weights = {}
    for action, data in samples.items():
        feature_count = len(data[0][0])
        matrix = [
            [float(i == j) * ridge for j in range(feature_count)]
            for i in range(feature_count)
        ]
        rhs = [0.0] * feature_count
        for features, target in data:
            for i, left in enumerate(features):
                rhs[i] += left * target / len(data)
                for j, right in enumerate(features):
                    matrix[i][j] += left * right / len(data)
        weights[str(action)] = _solve(matrix, rhs)
    return {
        "weights": weights,
        "ridge": ridge,
        "discount": 0.99,
        "features": len(next(iter(weights.values()), [])),
        "independently_fitted": True,
    }


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
        handle.seek(0)
        fields = next(csv.reader(handle))
    return rows, fields


def transform_trace(
    source: Path,
    output: Path,
    *,
    refine_delay_ms: int,
    role_mix: bool = False,
    writer_delay_ms: int = 0,
    prelock_delay_ms: int = 0,
    phase_alias: bool = False,
    phase_alias_group_size: int = 8,
    phase_alias_writers: int = 1,
    phase_alias_reader_pairs: int = 1,
    phase_alias_decoy_readers: bool = False,
    global_shared_key: bool = False,
) -> None:
    """Turn fresh generator rows into deterministic paired dynamic windows."""
    if output.exists():
        raise FileExistsError(output)
    rows, fields = _read_csv(source)
    if not rows:
        raise ValueError("source trace is empty")
    for ordinal, row in enumerate(rows):
        worker = int(float(row["worker_id"]))
        sequence = int(float(row["sequence"]))
        # A global shared key is useful for stress-testing a single agent
        # coordination object: all reader/writer roles observe the same
        # mutable dependency while retaining their phase-specific footprints.
        pair = 0 if global_shared_key else worker // 2
        alias_group = worker // phase_alias_group_size if phase_alias else worker // 4
        # Each pair/sequence key is touched by exactly two workers in a fixed
        # no-cycle run, so it does not satisfy the >=3 hotness rule.
        shared = f"ycsb:record:fresh-dynamic:pair-{pair}:seq-{sequence}:field:0"
        private_a = f"ycsb:record:fresh-dynamic:private-{worker}-{sequence}-a:field:0"
        private_b = f"ycsb:record:fresh-dynamic:private-{worker}-{sequence}-b:field:0"
        reader = bool(role_mix and (worker == 0 if global_shared_key else worker % 2 == 0))
        if phase_alias:
            role = worker % phase_alias_group_size
            pair_index = role // 2 if phase_alias_reader_pairs > 1 else 0
            shared_a = f"ycsb:record:phase-alias:a-{alias_group}-pair-{pair_index}:seq-{sequence}:field:0"
            shared_b = f"ycsb:record:phase-alias:b-{alias_group}:seq-{sequence}:field:0"
            alias_private = f"ycsb:record:phase-alias:private-{worker}-{sequence}:field:0"
            alias_suffix = f"ycsb:record:phase-alias:suffix-{worker}-{sequence}:field:0"
            paired_reader = (
                phase_alias_reader_pairs > 1
                and role < 2 * phase_alias_reader_pairs
                and role % 2 == 0
            )
            paired_writer = (
                phase_alias_reader_pairs > 1
                and role < 2 * phase_alias_reader_pairs
                and role % 2 == 1
            )
            if role == 0 or paired_reader:
                # Expensive mutable reader. At the refine boundary it has the
                # same |R|, |W|, overlap and completed-op counts as role 2/3,
                # but must protect the cold shared read before long reasoning.
                ops = [
                    {"kind": "read", "object_id": shared_a, "key": 0, "value": "", "phase": "explore", "delay_ms": 0},
                    {"kind": "read", "object_id": alias_private, "key": 1, "value": "", "phase": "explore", "delay_ms": 0},
                    {"kind": "write", "object_id": alias_private, "key": 2, "value": f"a-private-{worker}-{sequence}", "phase": "refine", "delay_ms": 0},
                    # Revisit the same mutable object after entering refine.
                    # This is the phase-sensitive point: a policy can defer
                    # protection during the cheap explore read, then protect
                    # the repeated read before the long reasoning interval.
                    {"kind": "read", "object_id": shared_a, "key": 3, "value": "", "phase": "refine", "delay_ms": int(refine_delay_ms)},
                ]
                alias_role = "refine_expensive_reader"
            elif paired_writer or (phase_alias_reader_pairs == 1 and 1 <= role <= phase_alias_writers):
                ops = [
                    {"kind": "read", "object_id": alias_private, "key": 0, "value": "", "phase": "explore", "delay_ms": int(writer_delay_ms)},
                    {"kind": "write", "object_id": shared_a, "key": 1, "value": f"a-writer-{worker}-{sequence}", "phase": "commit", "delay_ms": 0},
                ]
                alias_role = "refine_conflict_writer"
            elif not phase_alias_decoy_readers or role == 2 * phase_alias_reader_pairs:
                # The remaining optimistic blind-write updaters share B. Their decision
                # is made in explore after the same observed 2R/1W prefix.
                ops = [
                    {"kind": "read", "object_id": shared_b, "key": 0, "value": "", "phase": "explore", "delay_ms": 0},
                    {"kind": "read", "object_id": alias_private, "key": 1, "value": "", "phase": "explore", "delay_ms": 0},
                    {"kind": "write", "object_id": shared_b, "key": 2, "value": f"b-updater-{worker}-{sequence}", "phase": "explore", "delay_ms": 0},
                    {"kind": "read", "object_id": alias_suffix, "key": 3, "value": "", "phase": "explore", "delay_ms": int(refine_delay_ms)},
                ]
                alias_role = "explore_optimistic_updater"
            else:
                # Short read-only followers arrive while the single long B updater is
                # reasoning. They do not invalidate an optimistic B write, whereas an
                # early protected B write blocks them.
                ops = [
                    {"kind": "read", "object_id": alias_private, "key": 0, "value": "", "phase": "explore", "delay_ms": int(writer_delay_ms)},
                    {"kind": "read", "object_id": shared_b, "key": 1, "value": "", "phase": "explore", "delay_ms": 0},
                ]
                alias_role = "explore_read_only_follower"
        elif role_mix and not reader:
            # The odd worker is a writer that never reads the shared key.  The
            # even worker is the long reader.  A protection decision on the
            # reader's early shared read can therefore serialize the writer
            # without a read/write upgrade deadlock.
            ops = [
                {"kind": "read", "object_id": private_a, "key": 0, "value": "", "phase": "explore", "delay_ms": int(writer_delay_ms)},
                {"kind": "write", "object_id": shared, "key": 1, "value": f"fresh-{worker}-{sequence}-{ordinal}", "phase": "commit", "delay_ms": 0},
            ]
        elif role_mix and reader:
            ops = [
                # The same dependency is cheap in explore but expensive in
                # refine.  Put the reasoning delay on the refine access itself
                # so the phase-entry decision protects the actual conflict key
                # rather than an unrelated private read.
                {"kind": "read", "object_id": shared, "key": 0, "value": "", "phase": "explore", "delay_ms": int(prelock_delay_ms)},
                {"kind": "read", "object_id": shared, "key": 1, "value": "", "phase": "refine", "delay_ms": int(refine_delay_ms)},
            ]
        else:
            ops = [
                {"kind": "read", "object_id": shared, "key": 0, "value": "", "phase": "explore", "delay_ms": 2},
                {"kind": "read", "object_id": private_a, "key": 1, "value": "", "phase": "refine", "delay_ms": int(refine_delay_ms)},
                {"kind": "read", "object_id": private_b, "key": 2, "value": "", "phase": "refine", "delay_ms": int(refine_delay_ms)},
                {"kind": "write", "object_id": shared, "key": 3, "value": f"fresh-{worker}-{sequence}-{ordinal}", "phase": "commit", "delay_ms": 0},
            ]
        row["trace_id"] = f"fresh_ycsb_dynamic_pair_c{row['clients']}"
        row["workload_variant"] = "ycsb_fresh_dynamic_pair_window"
        row["operation_count"] = str(len(ops))
        row["read_count"] = str(sum(op["kind"] == "read" for op in ops))
        row["write_count"] = str(sum(op["kind"] == "write" for op in ops))
        row["ops_json"] = json.dumps(ops, separators=(",", ":"), sort_keys=True)
        row["object_keys_json"] = json.dumps([op["object_id"] for op in ops], separators=(",", ":"))
        row["explore_delay_ms"], row["refine_delay_ms"], row["commit_delay_ms"] = "0", "0", "0"
        row["total_reasoning_delay_ms"] = str(sum(int(op.get("delay_ms", 0)) for op in ops))
        context = json.loads(row.get("context_json") or "{}")
        context.update({
            "fresh_dynamic_pair": True,
            "role_mix": bool(role_mix),
            "role": ("reader" if reader else "writer") if role_mix else "symmetric",
            "policy_invocation_ops": 2 if phase_alias else 0,
            "pair_id": pair,
            "global_shared_key": bool(global_shared_key),
            "pair_sequence": sequence,
            "refine_delay_ms": int(refine_delay_ms),
            "prelock_delay_ms": int(prelock_delay_ms),
            "shared_key_observations_per_fixed_run": 2,
            "scenario": ("phase-aliased 2R/1W prefixes with refine reader versus explore updater"
                         if phase_alias else "shared explore read, long shared refine read, late shared write"),
            **({"phase_alias_role": alias_role, "phase_alias_group": alias_group}
               if phase_alias else {}),
        })
        row["context_json"] = json.dumps(context, separators=(",", ":"), sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dump(output.with_suffix(".meta.json"), {
        "scenario": ((f"{phase_alias_group_size}-worker phase-aliasing YCSB microbenchmark") if phase_alias else
                     ("paired dynamic YCSB reader/writer window" if role_mix else "paired dynamic YCSB window")),
        "shared_key_observations": 2,
        "operations": "read shared (explore), long read of the same shared key (refine), write shared (commit)",
        "refine_delay_ms": int(refine_delay_ms),
        "prelock_delay_ms": int(prelock_delay_ms),
        "writer_delay_ms": int(writer_delay_ms),
        "cycle_trace": False,
        "role_mix": bool(role_mix),
        "phase_alias": bool(phase_alias),
        "phase_alias_writers": int(phase_alias_writers),
        "phase_alias_reader_pairs": int(phase_alias_reader_pairs),
        "phase_alias_decoy_readers": bool(phase_alias_decoy_readers),
        "result_independent": True,
        "source_trace": str(source),
    })


class NoLowRiskGuardManager(AgentTransactionManager):
    """Use the same unguarded policy semantics for every comparison arm."""

    def __init__(self, *args, **kwargs):
        # The low-conflict guard is an engineering fast path, not a policy arm;
        # disabling it for all arms lets the predeclared action differences be
        # observed from the first fixed trace window.
        kwargs["low_conflict_occ_guard"] = False
        super().__init__(*args, **kwargs)
        # Agent-style deployment optimization: learned protection is applied
        # to concrete objects already observed by the attempt.  Avoid the
        # legacy category-wide read refresh, which adds lock-manager work even
        # when the selected policy action is unchanged.
        self.complete_category_read_protection = False
        self.first_access_read_refresh = False
        self.reconcile_selected_reads_at_checkpoints = False
        self.force_agent_phase_protection = bool(FORCE_AGENT_PHASE_PROTECTION)
        self.phase_policy_on_transition = bool(PHASE_ACTION_GATING)
        if COST_BASED_PRIORITY:
            self.priority_manager.config = dataclasses.replace(
                self.priority_manager.config, operation_cost_based=True
            )
        self.account_prebegin_retry_cost = bool(ACCOUNT_PREBEGIN_RETRY_COST)
        if OBSERVED_TOOL_ROUNDS:
            self.synthetic_tool_rounds = True
            self.state_collector.observed_agent_rounds = True
        base = self.paper_policy.snapshot()
        if BINARY_PHASE_ACTIONS:
            self.paper_policy = type(self.paper_policy)(BinaryPhaseActionPolicy(base))
            base = self.paper_policy.snapshot()
        if (PROJECT_NO_PHASE_PROGRESS or PROJECT_NO_PHASE_DIRECT_COUNTERS
                or PROJECT_NO_PHASE_OPERATION_COUNTER or PROJECT_NO_PHASE_ROUND_COUNTER):
            if getattr(base, "phase_awareness", "enabled") == "disabled":
                projector = no_phase_projector(
                    progress=PROJECT_NO_PHASE_PROGRESS,
                    direct=PROJECT_NO_PHASE_DIRECT_COUNTERS,
                    operations=PROJECT_NO_PHASE_OPERATION_COUNTER,
                    rounds=PROJECT_NO_PHASE_ROUND_COUNTER,
                )
                self.paper_policy = type(self.paper_policy)(PhaseProgressProjection(base, projector))
                base = self.paper_policy.snapshot()
        if PHASE_ACTION_GATING:
            base = self.paper_policy.snapshot()
            # This scheduling constraint is part of the Full compiled-policy
            # experiment.  Do not accidentally give the phase-blind rule or
            # raw-linear baselines a phase gate (or suppress their explore
            # decisions), which would change the baseline being evaluated.
            if (
                isinstance(base, CompiledPhasePolicy)
                and getattr(base, "phase_awareness", "enabled") == "enabled"
            ):
                self.paper_policy = type(self.paper_policy)(PhaseEligiblePolicy(base))
                base = self.paper_policy.snapshot()
        self.expensive_operation_checkpoint_enabled = bool(
            EXPENSIVE_OPERATION_CHECKPOINT
        )
        self.expensive_operation_checkpoint_fraction = float(
            EXPENSIVE_OPERATION_CHECKPOINT_FRACTION
        )
        checkpoint_base = base
        while hasattr(checkpoint_base, "base"):
            checkpoint_base = checkpoint_base.base
        self.learned_checkpoint_only = bool(
            EXPENSIVE_OPERATION_CHECKPOINT
            and isinstance(checkpoint_base, CompiledPhasePolicy)
            and not COMMON_POLICY_CHECKPOINTS
        )


class BinaryPhaseActionPolicy:
    """Project every arm onto the same OCC-versus-protect-all action pair."""

    def __init__(self, base):
        self.base = base
        self.phase_awareness = getattr(base, "phase_awareness", "enabled")

    @staticmethod
    def _project(selected):
        encoded = int(getattr(selected, "encoded", int(selected.protected)))
        return LockAction(LockClass.NONE if (encoded & 0xF) == 0 else LockClass(15))

    def select(self, state):
        return self._project(self.base.select(state))

    def select_with_probability(self, state):
        sampler = getattr(self.base, "select_with_probability", None)
        if sampler is None:
            return self.select(state), 1.0
        selected, probability = sampler(state)
        return self._project(selected), probability

    def select_with_distribution(self, state):
        sampler = getattr(self.base, "select_with_distribution", None)
        if sampler is None:
            selected = self.select(state)
            result = [0.0] * 16
            result[int(selected.protected)] = 1.0
            return selected, tuple(result)
        selected, distribution = sampler(state)
        projected = self._project(selected)
        result = [0.0] * len(distribution)
        result[0] = float(distribution[0])
        result[15] = max(0.0, 1.0 - result[0])
        return projected, tuple(result)


class PhaseProgressProjection:
    """Apply the repository's declared phase/progress ablation at inference."""

    def __init__(self, base, projector=without_phase_progress):
        self.base = base
        self.projector = projector

    def select(self, state):
        return self.base.select(self.projector(state))


class PhaseEligiblePolicy:
    """Install only protection classes eligible at the observed phase.

    Existing bits are never removed. Refine may add read protection and commit
    may add write protection; this prevents a combined learned action from
    activating a future write class before the write suffix is reached.
    """

    phase_awareness = "enabled"

    def __init__(self, base):
        self.base = base

    def select(self, state):
        current = int(state.current_action) & 0xF
        proposed = int(self.base.select(state).protected) & 0xF
        if str(state.phase) == "refine":
            eligible = int(LockClass.HOT_READ | LockClass.COLD_READ)
        elif str(state.phase) == "commit":
            eligible = int(LockClass.HOT_WRITE | LockClass.COLD_WRITE)
        else:
            eligible = 0
        return LockAction(LockClass(current | (proposed & eligible)))


def _run_with_manager(**kwargs):
    old = runner.AgentTransactionManager
    managers = []

    class AuditedManager(NoLowRiskGuardManager):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.applied_category_changes = 0
            managers.append(self)

        def transition_atcc_action(self, txn, action, **kw):
            previous = int(txn.context.action.protected)
            super().transition_atcc_action(txn, action, **kw)
            # Successful applied transitions only; proposals, pending plans,
            # failed acquisitions and same-mask reconciliation do not count.
            if int(txn.context.action.protected) != previous:
                with self._lock:
                    self.applied_category_changes += 1

    runner.AgentTransactionManager = AuditedManager
    try:
        result = runner.run_trace(**kwargs)
        result["applied_category_changes"] = sum(m.applied_category_changes for m in managers)
        result['account_prebegin_retry_cost'] = bool(ACCOUNT_PREBEGIN_RETRY_COST)
        return result
    finally:
        runner.AgentTransactionManager = old


def generate_fresh_raw(out: Path, *, workers: int, transactions: int, seed: int) -> None:
    if out.exists():
        raise FileExistsError(out)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/unified_trace/generate_castdas_trace.py"),
        "--output", str(out),
        "--variant", "ycsb_high_z099",
        "--clients", str(workers),
        "--agent-ratio", "1.0",
        "--seed", str(seed),
        "--transactions-per-worker", str(transactions),
        "--reasoning-profile", "none",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def simple_linear_risk_model() -> dict[str, object]:
    """One fixed linear risk score over hotness and conflict pressure."""
    occ = [0.0] * 37
    protect = [0.0] * 37
    # Fixed, deliberately simple comparator: protect only when the combined
    # hotness/conflict score reaches 0.50.  This keeps Base-linear a single
    # hand-written threshold rule rather than a fitted policy.
    protect[0] = -0.50
    protect[9] = 0.50
    protect[34] = 0.50
    return {
        "weights": {"0": occ, "15": protect},
        "features": 37,
        "formula": "protect iff 0.5*hotspot_ratio + 0.5*conflict_abort_rate >= 0.50",
        "trained": False,
    }


def collect_one(
    out: Path,
    trace: Path,
    index: int,
    *,
    workers: int,
    max_attempts: int,
    exploration_epsilon: float = 0.20,
    priority_quantum_scale: float = 1.0,
    timing_actions: bool = False,
) -> None:
    rows = runner.read_trace(trace)
    trajectory = out / f"trajectory_{index}.json"
    if trajectory.exists():
        raise FileExistsError(trajectory)
    result = _run_with_manager(
        rows=rows,
        warmup_rows=None,
        cc="paper-atcc",
        policy=None,
        paper_policy=None,
        trajectory_output=trajectory,
        paper_exploration_seed=240100 + index,
        paper_exploration_stay_probability=0.10,
        paper_exploration_epsilon=float(exploration_epsilon),
        paper_exploration_timing_actions=bool(timing_actions),
        priority_quantum_scale=float(priority_quantum_scale),
        policy_mode="eval",
        paper_switching="dynamic",
        paper_priority="enabled",
        paper_performance_guards="disabled",
        paper_cost_aware_observed_read_protection=False,
        paper_delayed_write_apply=delayed_write_mode(),
        paper_deferred_replay_enabled=False,
        paper_inflight_priority=bool(INFLIGHT_PRIORITY),
        paper_defer_policy_protection_until_commit=(MATCHED_LEARNED_RUNTIME and DEFER_POLICY_UNTIL_COMMIT),
        paper_deferred_object_plan=(MATCHED_LEARNED_RUNTIME and DEFERRED_OBJECT_PLAN),
        paper_deferred_object_plan_min_redo_ms=DEFERRED_OBJECT_PLAN_MIN_REDO_MS,
        atcc_retry_cache_enabled=False,
        max_attempts=max_attempts,
        tokens_per_operation=64,
        warmup_seconds=0.0,
        measure_seconds=0.0,
        cycle_trace=False,
    )
    dump(out / f"collection_{index}.json", result)
    print("COLLECT", index, json.dumps({k: result.get(k) for k in ("agent_tps", "agent_p99_latency_ms", "agent_commits", "agent_aborts")}), flush=True)


def attempt_time_rewards(transitions, failure_penalty_s=0.0):
    """Charge observed attempt costs once, accumulating checkpoint wait deltas.

    Terminal operation/agent costs are cumulative. In contrast lock_wait_ms
    is a delta since the preceding decision. prior_retry_cost_ms is context
    about attempts already charged in their own terminal transitions.
    """
    pending_wait = {}
    rows = []
    for transition in transitions:
        key = (transition.source_id, transition.txn_id)
        pending_wait[key] = pending_wait.get(key, 0.0) + max(0.0, float(transition.lock_wait_ms))
        reward = 0.0
        if transition.done:
            cost_s = (max(0.0, float(transition.operation_cost_ms))
                      + max(0.0, float(transition.agent_cost_ms))
                      + pending_wait.pop(key)) / 1000.0
            reward = -(cost_s + (float(failure_penalty_s) if transition.commit_indicator <= 0 else 0.0))
        rows.append(dataclasses.replace(transition, reward=reward))
    return rows, {
        "kind": "terminal_attempt_observed_time_v2",
        "penalty_scope": "each_aborted_attempt; not_final_logical_task_failure",
        "failure_penalty_s": float(failure_penalty_s),
        "terminal_transitions": sum(int(r.done) for r in rows),
        "total_operation_s": sum(max(0., float(r.operation_cost_ms)) for r in rows if r.done) / 1000.,
        "total_agent_s": sum(max(0., float(r.agent_cost_ms)) for r in rows if r.done) / 1000.,
        "total_lock_wait_s": sum(max(0., float(r.lock_wait_ms)) for r in rows) / 1000.,
        "prior_retry_context_s_excluded": sum(max(0., float(r.retry_cost_ms)) for r in rows if r.done) / 1000.,
        "unterminated_attempts": len(pending_wait),
        "scope": "observed attempt work and decision-window waits; not wall-clock system makespan",
    }


def fit_models(
    out: Path,
    *,
    lock_weight: float,
    strict_local: bool,
    training_runs: int = 2,
    local_context: bool = False,
    supported_projection: bool = False,
    factorized_deployment: bool = False,
    qualified_return: bool = False,
    no_phase_progress: bool = False,
    no_phase_direct_counters: bool = False,
    no_phase_operation_counter: bool = False,
    no_phase_round_counter: bool = False,
    timing_actions: bool = False,
    linear_no_phase: bool = False,
    simple_linear_risk: bool = False,
    phase_balanced_training: bool = False,
    medoids_per_group: int = 8,
    nophase_actor_seed: int = 240111,
    preserve_return_targets: bool = False,
    stable_training: bool = False,
    task_time_reward: bool = False,
    task_failure_penalty_s: float = 60.0,
    logical_retry_horizon: bool = False,
    final_logical_failure_penalty_s: float = 0.0,
    final_logical_failure_max_attempts: int = 6,
) -> None:
    if (out / "full.json").exists():
        raise FileExistsError(out / "full.json")
    transitions = []
    for index in range(max(1, int(training_runs))):
        payload = json.loads((out / f"trajectory_{index}.json").read_text(encoding="utf-8"))
        transitions.extend(
            policy_transition_from_dict(t, source_id=f"fresh_train_{index}")
            for t in payload["transitions"]
        )
    if task_time_reward:
        rows, reward = attempt_time_rewards(transitions, task_failure_penalty_s)
    else:
        rows, reward = apply_paper_rewards(
            transitions, PaperRewardConfig(lock_weight=float(lock_weight)))
    if logical_retry_horizon:
        if not task_time_reward:
            raise ValueError('Logical retry horizon requires observed attempt time rewards')
        from agent.runtime.retry_horizon import link_observed_retries
        rows, chains = link_observed_retries(rows, allow_unobserved_attempts=True)
        reward = dict(reward, logical_retry_horizon=True, logical_chains=chains,
            horizon_scope='Preserve each observed attempt cost; carry future retry costs across abort boundaries')
    if final_logical_failure_penalty_s:
        if not logical_retry_horizon:
            raise ValueError('Final logical failure cost requires linked retry episodes')
        from agent.runtime.logical_completion_reward import penalize_final_failure
        rows, completion_report = penalize_final_failure(rows, final_logical_failure_penalty_s,
            final_retry_index=final_logical_failure_max_attempts - 1)
        reward = dict(reward, logical_completion=completion_report)
    original_returns = discounted_returns(rows, PPOConfig().discount)
    return_targets = original_returns if preserve_return_targets else None
    phase_balance_counts = {}
    if phase_balanced_training:
        by_phase = {}
        for transition, target_return in zip(rows, original_returns):
            by_phase.setdefault(str(transition.state.phase), []).append((transition, target_return))
        target = max(map(len, by_phase.values()), default=0)
        balanced = []
        for phase, phase_rows in sorted(by_phase.items()):
            if not phase_rows:
                continue
            copies, remainder = divmod(target, len(phase_rows))
            balanced.extend(phase_rows * copies)
            balanced.extend(phase_rows[:remainder])
            phase_balance_counts[phase] = {
                "before": len(phase_rows),
                "after": target,
            }
        rows = [row for row, _value in balanced]
        # Balancing changes temporal order and duplicates transitions. Returns
        # must refer to the original trajectories, never the synthetic order.
        return_targets = [value for _row, value in balanced]
    cfg = PPOConfig(
        epochs=12,
        min_group_samples=2,
        min_group_actions=2,
        entropy_weight=0.002,
        learning_rate=0.005,
        group_learning_rate=0.05,
        categorical_initialization=("qualified-return" if qualified_return else "behavior"),
        advantage_baseline=("deployment-group" if qualified_return else "linear-critic"),
        ppo_reference_mode=("frozen-actor" if qualified_return else "behavior"),
        calibration_return_mode="global" if stable_training else "per-source-run",
    )
    reports = {}
    for name, awareness, seed in (("full", "enabled", 240110), ("nophase", "disabled", nophase_actor_seed)):
        model_rows = rows
        if name == "nophase" and (no_phase_progress or no_phase_direct_counters
                or no_phase_operation_counter or no_phase_round_counter):
            projector = no_phase_projector(
                progress=no_phase_progress,
                direct=no_phase_direct_counters,
                operations=no_phase_operation_counter,
                rounds=no_phase_round_counter,
            )
            model_rows = [
                dataclasses.replace(
                    transition,
                    state=projector(transition.state),
                    next_state=projector(transition.next_state),
                )
                for transition in rows
            ]
        started = time.perf_counter()
        actor = DiscretePPOPolicy(
            seed=seed,
            phase_awareness=awareness,
            categorical_actor=not bool(factorized_deployment),
            # The supported categorical projection makes the PPO actor part of
            # deployment (rather than merely logging a gradient and then
            # selecting the empirical head).  It is opt-in so older runs stay
            # reproducible, while the repaired experiment can actually test
            # phase-conditioned decisions.
            ppo_supported_projection=(bool(supported_projection) and not bool(factorized_deployment)),
            local_qualified_action_support=(bool(supported_projection) and not bool(factorized_deployment)),
            strict_local_action_support=(bool(strict_local) and not bool(supported_projection) and not bool(factorized_deployment)),
            local_context_groups=bool(local_context),
            timing_actions=bool(timing_actions),
        )
        report = DiscretePPOTrainer(cfg).train(actor, model_rows, return_targets=return_targets)
        fit_seconds = time.perf_counter() - started
        model = actor.compile(
            generation=2401,
            medoids_per_group=int(medoids_per_group),
            occ_cold_start_guard=False,
            preserve_calibrated_groups=True,
            refinement_distance_threshold=(0.0 if factorized_deployment else None),
            statewise_supported_actor=bool(supported_projection and not factorized_deployment),
        )
        dump(out / f"{name}.json", model.to_dict())
        reports[name] = {
            "actor_seed": seed,
            "phase_awareness": awareness,
            "fit_seconds": fit_seconds,
            "entries": len(model.entries),
            "report": report,
        }
    linear_started = time.perf_counter()
    linear_rows = [
        dataclasses.replace(
            transition,
            action=int(transition.action) & 0xF,
            state=(dataclasses.replace(transition.state, phase="all")
                   if linear_no_phase else transition.state),
            next_state=(dataclasses.replace(transition.next_state, phase="all")
                        if linear_no_phase else transition.next_state),
        )
        for transition in rows
    ]
    if simple_linear_risk:
        # Preserve the predeclared, non-learned comparator written at prepare
        # time.  Previously fit() silently replaced it with a return-fitted
        # model, making the "simple linear" label inaccurate.
        dump(out / "linear.json", simple_linear_risk_model())
    else:
        dump(out / "linear.json", fit_linear(linear_rows, 0.01, return_targets=return_targets))
    dump(out / "training.json", {
        "fresh_from_scratch": True,
        "transitions": len(rows),
        "models": reports,
        "linear_fit_seconds": time.perf_counter() - linear_started,
        "reward": reward,
        "ppo_config": dataclasses.asdict(cfg),
        "trace_count": max(1, int(training_runs)),
        "no_test_driven_selection": True,
        "lock_weight": float(lock_weight),
        "task_time_reward": bool(task_time_reward),
        "logical_retry_horizon": bool(logical_retry_horizon),
        "task_failure_penalty_s": float(task_failure_penalty_s),
        "strict_local_action_support": bool(strict_local),
        "local_context_groups": bool(local_context),
        "ppo_supported_projection": bool(supported_projection),
        "statewise_supported_actor": bool(supported_projection and not factorized_deployment),
        "factorized_deployment": bool(factorized_deployment),
        "qualified_return": bool(qualified_return),
        "no_phase_progress": bool(no_phase_progress),
        "no_phase_direct_counters": bool(no_phase_direct_counters),
        "no_phase_operation_counter": bool(no_phase_operation_counter),
        "no_phase_round_counter": bool(no_phase_round_counter),
        "timing_actions": bool(timing_actions),
        "linear_no_phase": bool(linear_no_phase),
        "simple_linear_risk": bool(simple_linear_risk),
        "phase_balanced_training": bool(phase_balanced_training),
        "preserve_return_targets": bool(preserve_return_targets or phase_balanced_training),
        "stable_training": bool(stable_training),
        "phase_balance_counts": phase_balance_counts,
        "medoids_per_group": int(medoids_per_group),
        "deployment_contract": {
            "policy_group_key": "agent.cc.atcc.ppo.policy_group_key",
            "phase_awareness": "full=enabled,nophase=disabled",
            "phase_action_gating": bool(PHASE_ACTION_GATING),
            "action_space": ("32_choices_immediate_or_deferred_4bit_mask" if timing_actions
                             else "lock_protection_mask_4bit"),
            "priority_semantics": "transaction_manager_formula",
            "deferred_object_plan": bool(DEFERRED_OBJECT_PLAN),
            "deferred_object_plan_min_redo_ms": float(DEFERRED_OBJECT_PLAN_MIN_REDO_MS),
        },
    })
    print("FIT", len(rows), flush=True)


def count_action_changes(path: Path) -> tuple[int, int]:
    """Count proposed mask changes, not applied runtime lock transitions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("transitions", [])
    changes = 0
    for transition in decisions:
        state = transition.get("state", {})
        action = int(transition.get("action", 0)) & 0xF
        if action != int(state.get("current_action", 0)):
            changes += 1
    return changes, len(decisions)


def load_coverage_training_states(out: Path):
    """Use every real collection run; do not count balancing duplicates."""
    paths = sorted(out.glob("trajectory_*.json"))
    if not paths:
        raise FileNotFoundError(f"No training trajectories for coverage guard: {out}")
    states = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        states.extend(phase_aware_state_from_dict(t["state"])
                      for t in payload.get("transitions", []))
    return states


def measure_all(out: Path, *, max_attempts: int, priority_quantum_scale: float = 1.0,
                hot_frequency_threshold: int = 2,
                runtime_arms: tuple[str, ...] = ()) -> None:
    rows = runner.read_trace(out / "trace.csv")
    training_states = None
    if COVERAGE_GUARD:
        training_states = load_coverage_training_states(out)
    result_rows = []
    selected_arms = tuple(item for item in ARMS if not runtime_arms or item[0] in runtime_arms)
    for name, label in selected_arms:
        trajectory = out / "runtime" / f"{name}.json"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            rows=rows,
            warmup_rows=None,
            cc="paper-atcc",
            policy=None,
            paper_policy=None,
            paper_linear_policy=None,
            paper_rule_policy="none",
            paper_hot_frequency_threshold=int(hot_frequency_threshold),
            trajectory_output=trajectory,
            paper_exploration_seed=None,
            policy_mode="eval",
            paper_switching="dynamic",
            paper_priority="enabled",
            paper_performance_guards="disabled",
            paper_cost_aware_observed_read_protection=bool(
                COST_AWARE_READ_ELISION and name == "aegis_full"
            ),
            paper_delayed_write_apply=delayed_write_mode(),
            paper_deferred_replay_enabled=False,
            paper_inflight_priority=bool(INFLIGHT_PRIORITY),
            paper_defer_policy_protection_until_commit=(
                bool(DEFER_POLICY_UNTIL_COMMIT and (name == "aegis_full" or
                     (MATCHED_LEARNED_RUNTIME and name == "aegis_nophase")))
            ),
            paper_deferred_object_plan=(
                bool(DEFERRED_OBJECT_PLAN and (name == "aegis_full" or
                     (MATCHED_LEARNED_RUNTIME and name == "aegis_nophase")))
            ),
            paper_deferred_object_plan_min_redo_ms=(
                float(DEFERRED_OBJECT_PLAN_MIN_REDO_MS)
            ),
            paper_training_states=(training_states if name in {"aegis_full", "aegis_nophase"} else None),
            paper_coverage_min_samples=int(COVERAGE_MIN_SAMPLES),
            atcc_retry_cache_enabled=False,
            max_attempts=max_attempts,
            priority_quantum_scale=float(priority_quantum_scale),
            tokens_per_operation=64,
            warmup_seconds=0.0,
            measure_seconds=0.0,
            cycle_trace=False,
        )
        if name == "static_delay_priority":
            # The paper's fixed baseline: one calibrated contention gate,
            # delay and priority enabled, and a fixed hot-write protection
            # mask.  Keep this arm in the same trace and accounting path as
            # the learned/rule arms.
            kwargs["paper_switching"] = "static"
            kwargs["paper_static_conflict_threshold"] = 0.20
            kwargs["paper_static_protection_mask"] = 4
            kwargs["paper_delayed_write_apply"] = delayed_write_mode(static=True)
        elif name == "base_linear":
            kwargs["paper_linear_policy"] = out / "linear.json"
        elif name == "hot_threshold":
            kwargs["paper_rule_policy"] = "hot-threshold"
        elif name == "hot_always":
            kwargs["paper_rule_policy"] = "hot-always"
        elif name == "aegis_nophase":
            kwargs["paper_policy"] = out / "nophase.json"
        else:
            kwargs["paper_policy"] = out / "full.json"
        result = _run_with_manager(**kwargs)
        actual_changes, policy_calls = count_action_changes(trajectory)
        result.update({
            "variant": label,
            "display_variant": name,
            "suggested_mask_changes": actual_changes,
            "actual_action_changes": result["applied_category_changes"],
            "action_change_count_basis": "successful_runtime_category_transitions",
            "policy_calls_from_trace": policy_calls,
            "max_attempts": max_attempts,
            "fresh_trace": True,
        })
        dump(out / "runtime" / f"{name}.result.json", result)
        result_rows.append(result)
        csv_write(out / "runtime_raw.csv", result_rows)
        print("RUN", json.dumps({k: result.get(k) for k in ("display_variant", "agent_tps", "agent_p99_latency_ms", "agent_commits", "agent_aborts", "actual_action_changes")}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("prepare", "collect", "fit", "runtime"), required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--transactions-per-worker", type=int, default=8)
    p.add_argument("--refine-delay-ms", type=int, default=180)
    p.add_argument("--max-attempts", type=int, default=6)
    p.add_argument("--seed", type=int, default=24010001)
    p.add_argument("--lock-weight", type=float, default=5.0)
    p.add_argument("--strict-local", action="store_true")
    p.add_argument("--role-mix", action="store_true")
    p.add_argument("--training-runs", type=int, default=2)
    p.add_argument("--local-context", action="store_true")
    p.add_argument("--explore-epsilon", type=float, default=0.20)
    p.add_argument("--writer-delay-ms", type=int, default=0)
    p.add_argument("--prelock-delay-ms", type=int, default=0)
    p.add_argument("--priority-quantum-scale", type=float, default=1.0)
    p.add_argument("--supported-projection", action="store_true",
                   help="deploy the locally-supported categorical PPO actor")
    p.add_argument("--factorized-deployment", action="store_true",
                   help="deploy the PPO factorized lock-bit actor with a zero-distance gate")
    p.add_argument("--qualified-return", action="store_true",
                   help="use qualified-return initialization and deployment-group advantages")
    p.add_argument("--no-phase-progress", action="store_true",
                   help="for NoPhase only, remove declared phase-derived progress fields too")
    p.add_argument("--no-phase-direct-counters", action="store_true",
                   help="for NoPhase, remove phase plus completed round/operation counters")
    p.add_argument("--no-phase-operation-counter", action="store_true",
                   help="for NoPhase, remove phase plus completed operation count")
    p.add_argument("--no-phase-round-counter", action="store_true",
                   help="for NoPhase, remove phase plus completed round count")
    p.add_argument("--phase-action-gating", action="store_true",
                   help="for Full only, add read classes at refine and write classes at commit")
    p.add_argument("--force-agent-phase-protection", action="store_true",
                   help="protect observed YCSB reads after entering refine")
    p.add_argument("--cost-based-priority", action="store_true",
                   help="use observed operation cost in the dynamic priority")
    p.add_argument("--defer-policy-until-commit", action="store_true",
                   help="for Full only, defer learned protection selected before commit")
    p.add_argument("--deferred-object-plan", action="store_true",
                   help="for Full only, materialize protection over observed high-cost objects")
    p.add_argument("--deferred-object-plan-min-redo-ms", type=float, default=0.0,
                   help="minimum measured redo cost for an observed object to be nominated")
    p.add_argument("--coverage-guard", action="store_true",
                   help="wrap deployment policy with training-state coverage guard")
    p.add_argument("--coverage-min-samples", type=int, default=4)
    p.add_argument("--expensive-op-checkpoint", action="store_true",
                   help="give every arm the same observed-state decision during a delayed operation")
    p.add_argument("--expensive-op-checkpoint-fraction", type=float, default=1.0,
                   help="fraction of delayed reasoning completed before the shared decision point")
    p.add_argument("--timing-actions", action="store_true",
                   help="let learned policies choose immediate versus commit-deferred protection")
    p.add_argument("--phase-alias", action="store_true",
                   help="use the four-worker phase-aliasing YCSB microbenchmark")
    p.add_argument("--phase-alias-group-size", type=int, default=8,
                   help="roles per independent hotspot group (two fixed roles plus update roles)")
    p.add_argument("--phase-alias-writers", type=int, default=1,
                   help="late conflict writers per phase-alias hotspot group")
    p.add_argument("--phase-alias-reader-pairs", type=int, default=1,
                   help="independent expensive-reader/late-writer pairs per hotspot group")
    p.add_argument("--phase-alias-decoy-readers", action="store_true",
                   help="use one long B updater plus short read-only followers")
    p.add_argument("--global-shared-key", action="store_true",
                   help="make all role-mix readers/writers share one mutable key")
    p.add_argument("--linear-no-phase", action="store_true",
                   help="fit the simple linear comparator without a phase one-hot")
    p.add_argument("--simple-linear-risk", action="store_true",
                   help="use one fixed hotness/conflict linear score instead of fitted action heads")
    p.add_argument("--hot-frequency-threshold", type=int, default=2,
                   help="per-object trailing-window access count for the Hot-threshold arm")
    p.add_argument("--binary-phase-actions", action="store_true",
                   help="restrict learned phase probe to OCC versus protect-all")
    p.add_argument("--runtime-arm", action="append", choices=tuple(name for name, _ in ARMS),
                   help="measure only this arm; repeat to select multiple arms")
    p.add_argument("--phase-balanced-training", action="store_true",
                   help="resample observed training transitions equally across phases")
    p.add_argument("--medoids-per-group", type=int, default=8,
                   help="compiled policy-table capacity per state group")
    p.add_argument("--task-time-reward", action="store_true",
                   help="train on terminal attempt work plus all checkpoint wait deltas; no duplicate prior retry cost")
    p.add_argument("--common-delay-mode", choices=("legacy", "enabled", "disabled"), default="legacy",
                   help="use the same delayed-write setting for every arm; legacy preserves old diagnostics")
    p.add_argument("--task-failure-penalty-s", type=float, default=60.0)
    p.add_argument("--matched-learned-runtime", action="store_true",
                   help="strict phase-label ablation: same learned runtime controls and actor seed")
    p.add_argument("--observed-tool-rounds", action="store_true",
                   help="synthetic point operations are tool rounds; phase changes do not increment rounds")
    p.add_argument("--cost-aware-read-elision", action="store_true",
                   help="enable runtime cost-aware read-lock elision for Full only")
    p.add_argument("--inflight-priority", action="store_true",
                   help="refresh wound priorities using already observed agent cost")
    a = p.parse_args()
    if a.matched_learned_runtime and (a.no_phase_progress
            or a.no_phase_direct_counters or a.no_phase_operation_counter or a.no_phase_round_counter):
        raise ValueError("matched learned runtime requires no Full-only phase gate or extra feature ablation")
    if platform.node().split(".")[0] != "node1":
        raise RuntimeError("this experiment must run on node1")
    if a.workers < 2 or a.workers % 2 or a.transactions_per_worker < 1:
        raise ValueError("workers must be a positive even number >= 2")
    if a.phase_alias_group_size < 4:
        raise ValueError("phase-alias group size must be at least four")
    if a.phase_alias_writers < 1 or a.phase_alias_writers >= a.phase_alias_group_size - 1:
        raise ValueError("phase-alias writers must leave a reader and at least one updater")
    if a.phase_alias_reader_pairs < 1 or 2 * a.phase_alias_reader_pairs >= a.phase_alias_group_size:
        raise ValueError("phase-alias reader pairs must leave at least one updater")
    if a.hot_frequency_threshold < 2:
        raise ValueError("Hot-threshold must exceed Hot-always's one-access trigger")
    if sum(map(int, (a.no_phase_progress, a.no_phase_direct_counters,
                    a.no_phase_operation_counter, a.no_phase_round_counter))) > 1:
        raise ValueError("choose one NoPhase projection scope")
    if a.phase_alias and a.workers % a.phase_alias_group_size:
        raise ValueError("phase-alias workers must be divisible by phase-alias group size")
    if a.refine_delay_ms < 1 or a.max_attempts < 1 or a.training_runs < 1 or a.writer_delay_ms < 0 or a.prelock_delay_ms < 0 or a.priority_quantum_scale <= 0:
        raise ValueError("delay and max attempts must be positive")
    if a.medoids_per_group < 1:
        raise ValueError("medoids per group must be positive")
    if not 0.0 <= a.explore_epsilon <= 1.0:
        raise ValueError("explore epsilon must be in [0, 1]")
    if not 0.0 <= a.expensive_op_checkpoint_fraction <= 1.0:
        raise ValueError("checkpoint fraction must be in [0, 1]")
    global PROJECT_NO_PHASE_PROGRESS, PROJECT_NO_PHASE_DIRECT_COUNTERS
    global PROJECT_NO_PHASE_OPERATION_COUNTER, PROJECT_NO_PHASE_ROUND_COUNTER, PHASE_ACTION_GATING, FORCE_AGENT_PHASE_PROTECTION, COST_BASED_PRIORITY
    global DEFER_POLICY_UNTIL_COMMIT, DEFERRED_OBJECT_PLAN, DEFERRED_OBJECT_PLAN_MIN_REDO_MS
    global COVERAGE_GUARD, COVERAGE_MIN_SAMPLES
    global MATCHED_LEARNED_RUNTIME
    global OBSERVED_TOOL_ROUNDS
    global COMMON_DELAY_MODE, COST_AWARE_READ_ELISION, INFLIGHT_PRIORITY
    COMMON_DELAY_MODE = a.common_delay_mode
    COST_AWARE_READ_ELISION = bool(a.cost_aware_read_elision)
    INFLIGHT_PRIORITY = bool(a.inflight_priority)
    OBSERVED_TOOL_ROUNDS = bool(a.observed_tool_rounds)
    MATCHED_LEARNED_RUNTIME = bool(a.matched_learned_runtime)
    global EXPENSIVE_OPERATION_CHECKPOINT, EXPENSIVE_OPERATION_CHECKPOINT_FRACTION
    global BINARY_PHASE_ACTIONS
    PROJECT_NO_PHASE_PROGRESS = bool(a.no_phase_progress)
    PROJECT_NO_PHASE_DIRECT_COUNTERS = bool(a.no_phase_direct_counters)
    PROJECT_NO_PHASE_OPERATION_COUNTER = bool(a.no_phase_operation_counter)
    PROJECT_NO_PHASE_ROUND_COUNTER = bool(a.no_phase_round_counter)
    PHASE_ACTION_GATING = bool(a.phase_action_gating)
    FORCE_AGENT_PHASE_PROTECTION = bool(a.force_agent_phase_protection)
    COST_BASED_PRIORITY = bool(a.cost_based_priority)
    DEFER_POLICY_UNTIL_COMMIT = bool(a.defer_policy_until_commit)
    DEFERRED_OBJECT_PLAN = bool(a.deferred_object_plan)
    DEFERRED_OBJECT_PLAN_MIN_REDO_MS = float(a.deferred_object_plan_min_redo_ms)
    COVERAGE_GUARD = bool(a.coverage_guard)
    COVERAGE_MIN_SAMPLES = int(a.coverage_min_samples)
    EXPENSIVE_OPERATION_CHECKPOINT = bool(a.expensive_op_checkpoint)
    EXPENSIVE_OPERATION_CHECKPOINT_FRACTION = float(a.expensive_op_checkpoint_fraction)
    BINARY_PHASE_ACTIONS = bool(a.binary_phase_actions)
    a.output.mkdir(parents=True, exist_ok=(a.stage != "prepare")) if a.stage == "prepare" else None
    if a.stage == "prepare":
        if a.output.exists() and any(a.output.iterdir()):
            raise FileExistsError(f"non-empty output: {a.output}")
        a.output.mkdir(parents=True, exist_ok=True)
        raw = a.output / "raw_base.csv"
        test = a.output / "trace.csv"
        generate_fresh_raw(raw, workers=a.workers, transactions=a.transactions_per_worker, seed=a.seed)
        transform_trace(
            raw,
            test,
            refine_delay_ms=a.refine_delay_ms,
            role_mix=a.role_mix,
            writer_delay_ms=a.writer_delay_ms,
            prelock_delay_ms=a.prelock_delay_ms,
            phase_alias=a.phase_alias,
            phase_alias_group_size=a.phase_alias_group_size,
            phase_alias_writers=a.phase_alias_writers,
            phase_alias_reader_pairs=a.phase_alias_reader_pairs,
            phase_alias_decoy_readers=a.phase_alias_decoy_readers,
            global_shared_key=a.global_shared_key,
        )
        if a.simple_linear_risk:
            dump(a.output / "linear.json", simple_linear_risk_model())
        dump(a.output / "protocol.json", {
            "scenario": "fresh paired dynamic YCSB phase window",
            "workers": a.workers,
            "transactions_per_worker": a.transactions_per_worker,
            "refine_delay_ms": a.refine_delay_ms,
            "writer_delay_ms": a.writer_delay_ms,
            "prelock_delay_ms": a.prelock_delay_ms,
            "priority_quantum_scale": a.priority_quantum_scale,
            "max_attempts": a.max_attempts,
            "shared_key_observations": 2,
            "global_shared_key": bool(a.global_shared_key),
            "arms": [name for name, _ in ARMS],
            "same_trace_and_runtime_for_all_arms": True,
            "training_runs": max(1, int(a.training_runs)),
            "common_delay_mode": COMMON_DELAY_MODE,
            "cost_aware_read_elision": bool(a.cost_aware_read_elision),
            "effective_delay_by_arm": {name: delayed_write_mode(static=name == 'static_delay_priority')
                                       for name, _ in ARMS},
            "local_context_groups": bool(a.local_context),
            "training_exploration_epsilon": float(a.explore_epsilon),
            "no_phase_progress": bool(a.no_phase_progress),
            "no_phase_direct_counters": bool(a.no_phase_direct_counters),
            "no_phase_operation_counter": bool(a.no_phase_operation_counter),
            "no_phase_round_counter": bool(a.no_phase_round_counter),
            "phase_action_gating": bool(a.phase_action_gating),
            "deferred_object_plan": bool(a.deferred_object_plan),
            "deferred_object_plan_min_redo_ms": float(a.deferred_object_plan_min_redo_ms),
            "expensive_operation_checkpoint": bool(a.expensive_op_checkpoint),
            "expensive_operation_checkpoint_fraction": float(a.expensive_op_checkpoint_fraction),
            "timing_actions": bool(a.timing_actions),
            "phase_alias": bool(a.phase_alias),
            "phase_alias_group_size": int(a.phase_alias_group_size),
            "phase_alias_writers": int(a.phase_alias_writers),
            "phase_alias_reader_pairs": int(a.phase_alias_reader_pairs),
            "phase_alias_decoy_readers": bool(a.phase_alias_decoy_readers),
            "linear_no_phase": bool(a.linear_no_phase),
            "simple_linear_risk": bool(a.simple_linear_risk),
            "hot_frequency_threshold": int(a.hot_frequency_threshold),
            "binary_phase_actions": bool(a.binary_phase_actions),
            "phase_balanced_training": bool(a.phase_balanced_training),
            "medoids_per_group": int(a.medoids_per_group),
            "no_warmup_priming": True,
            "cycle_trace": False,
            "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "agent").rglob("*.py")},
            "script_sha256": digest(Path(__file__)),
            "integrity_note": "mechanism stress test; all results retained; not a replacement for Figure 12",
        })
        return
    if a.stage == "collect":
        for index in range(max(1, int(a.training_runs))):
            trace = a.output / "trace.csv"
            collect_one(
                a.output,
                trace,
                index,
                workers=a.workers,
                max_attempts=a.max_attempts,
                exploration_epsilon=a.explore_epsilon,
                priority_quantum_scale=a.priority_quantum_scale,
                timing_actions=a.timing_actions,
            )
        return
    if a.stage == "fit":
        fit_models(
            a.output,
            lock_weight=a.lock_weight,
            strict_local=a.strict_local,
            training_runs=a.training_runs,
            local_context=a.local_context,
            supported_projection=a.supported_projection,
            factorized_deployment=a.factorized_deployment,
            qualified_return=a.qualified_return,
            no_phase_progress=a.no_phase_progress,
            no_phase_direct_counters=a.no_phase_direct_counters,
            no_phase_operation_counter=a.no_phase_operation_counter,
            no_phase_round_counter=a.no_phase_round_counter,
            timing_actions=a.timing_actions,
            linear_no_phase=a.linear_no_phase,
            simple_linear_risk=a.simple_linear_risk,
            phase_balanced_training=a.phase_balanced_training,
            medoids_per_group=a.medoids_per_group,
            nophase_actor_seed=(240110 if a.matched_learned_runtime else 240111),
            task_time_reward=a.task_time_reward,
            task_failure_penalty_s=a.task_failure_penalty_s,
        )
        return
    measure_all(
        a.output,
        max_attempts=a.max_attempts,
        priority_quantum_scale=a.priority_quantum_scale,
        hot_frequency_threshold=a.hot_frequency_threshold,
        runtime_arms=tuple(a.runtime_arm or ()),
    )


if __name__ == "__main__":
    main()
