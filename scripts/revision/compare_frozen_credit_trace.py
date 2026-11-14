#!/usr/bin/env python3
"""Paired CC integration runner on exactly one captured task/LLM/tool trace.

Tiny samples and accelerated replays are smoke tests, not rebuttal evidence.
Paper Aegis is accepted only with an explicitly supplied compiled policy.
"""

from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.cc import LockConflict
from agent.cc.atcc.ppo import EpsilonGreedyPolicy
from agent.native import load_cast_core
from agent.runtime import AgentTransactionManager
from agent.runtime.paper_policy import CompiledPhasePolicy, LocalAccessFrequencyPolicy
from agent.runtime.context import LockAction, LockClass
from agent.workloads.credit_reasoner import CapturingReasoner, ReplayReasoner
from agent.workloads.credit_tools import CapturingTool, ReplayTool
from agent.workloads.credit_review import CreditReviewConfig, CreditReviewTaskSpec, CreditReviewWorkload
from agent.workloads.frozen_credit_agent import FrozenCreditAgentCursor
from scripts.unified_trace.run_credit_review_experiment import (
    commit_batch_callback, transaction_metadata,
)

SYSTEMS = {"silo", "2pl-wait-die", "bamboo", "polaris", "paper-atcc"}


class FixedProtectionPolicy:
    """Static hot-write protection with no runtime action switching."""

    def select(self, state):
        return LockAction(LockClass.HOT_WRITE)


class FixedReadProtectionPolicy:
    """Fixed read protection, installed before access, no action switching."""

    def select(self, state):
        return LockAction(LockClass.HOT_READ | LockClass.COLD_READ)


class RecordingPolicy:
    """Evaluation-only input/action audit; excludes object IDs and model output."""

    def __init__(self, policy, clock=None):
        self.policy = policy
        self.clock = clock
        self.records = []
        self.lock = threading.Lock()

    def select(self, state):
        started = time.perf_counter()
        if self.clock:
            with self.clock.scope("lookup"):
                action = self.policy.select(state)
        else:
            action = self.policy.select(state)
        elapsed = (time.perf_counter() - started) * 1000
        record = dict(state=dataclasses.asdict(state), action=action.encoded,
                      lookup_ms=elapsed, observed_monotonic_s=time.perf_counter())
        with self.lock:
            self.records.append(record)
        return action


def run_system(source, system, *, workers, max_attempts, time_scale,
               repeat_factor=1, policy=None, trajectory_sink=None,
               exploration_epsilon=0.0, performance_guards=False,
               attempt_sink=None, static_protection=False, decision_sink=None,
               exploration_masks=None, frozen_agent_schedule=False,
               manager_factory=AgentTransactionManager, policy_commit_batch=True,
               profile_clock=None, profile_sink=None,
               delayed_write_apply_enabled=True, priority_enabled=True,
               policy_selected_write_admission_only=False,
               before_reasoning_checkpoint=False, carry_prior_restart_cost=False,
               allow_complete_aborted_capture=False, task_start_gate=None,
               attempt_lifecycle=None, logical_task_feedback=False):
    # Opt-in for newly trained/evaluated candidates. Historical replays did not
    # publish task-level outcomes; do not silently change their online state.
    if type(logical_task_feedback) is not bool:
        raise ValueError('logical_task_feedback must be an explicit boolean')
    if attempt_lifecycle is not None and not all(callable(getattr(attempt_lifecycle,name,None)) for name in ('on_begin','on_finish')):
        raise ValueError('Explicit on_begin/on_finish attempt lifecycle required')
    if profile_sink is not None and profile_clock is None:
        raise ValueError("profile output requires an exclusive clock")
    if source.get("agent_directed") and not frozen_agent_schedule:
        raise ValueError("model-directed captures require a matching agent-loop runner; fixed cursor replay is invalid")
    if frozen_agent_schedule and not source.get("agent_directed"):
        raise ValueError("frozen agent schedule requires model-directed capture")
    if exploration_epsilon and performance_guards:
        raise ValueError("exploration requires unmodified policy actions")
    if exploration_epsilon and decision_sink is not None:
        raise ValueError("decision recording wrapper is evaluation-only")
    if source.get('portfolio_credit'):
        if not frozen_agent_schedule:
            raise ValueError('portfolio credit requires model-selected semantic replay')
        from agent.workloads.credit_portfolio import PortfolioCreditWorkload, FrozenPortfolioCreditCursor
        workload = PortfolioCreditWorkload(CreditReviewConfig(**source['config']),
            available_capacity=source['portfolio_available_capacity'])
        frozen_cursor_type = FrozenPortfolioCreditCursor
    else:
        workload = CreditReviewWorkload(CreditReviewConfig(**source["config"]))
        frozen_cursor_type = FrozenCreditAgentCursor
    engines = None
    if source.get("multi_storage"):
        core = load_cast_core()
        if not hasattr(core, "batch_put_across_stores"):
            raise ValueError("native cross-store coordinator is required")
        engines = {"default": core.Dbx1000VersionedKVStore(),
                   "audit": core.Dbx1000VersionedKVStore()}
    if system == "paper-atcc" and policy is None:
        raise ValueError("paper-atcc needs a compiled policy trained for this workload")
    if exploration_epsilon:
        policy = EpsilonGreedyPolicy(policy, seed=20260930,
                                    epsilon=exploration_epsilon,
                                    exploration_masks=exploration_masks)
    recorded_policy = RecordingPolicy(policy, profile_clock) if policy and decision_sink is not None else None
    if recorded_policy:
        policy = recorded_policy
    manager = manager_factory(storage_engines=engines,
                                      paper_policy=policy,
                                      record_traces=False,
                                      collect_trajectories=trajectory_sink is not None,
                                      performance_guards_enabled=performance_guards,
                                      low_conflict_occ_guard=False,
                                      delayed_write_apply_enabled=delayed_write_apply_enabled,
                                      policy_selected_write_admission_only=policy_selected_write_admission_only,
                                      priority_enabled=priority_enabled)
    manager.expensive_operation_checkpoint_enabled = bool(before_reasoning_checkpoint)
    if static_protection:
        manager.paper_static_initial_mask = int(LockClass.HOT_WRITE)
    storage_for = (lambda spec: "audit" if ":audit" in spec.object_id or
                   ":compliance:" in spec.object_id else "default") if engines else None
    workload.register(manager, storage_for=storage_for)

    def logical_task(sequence):
        prior = source["tasks"][sequence % len(source["tasks"])]
        task = CreditReviewTaskSpec(**prior["task"])
        started = time.perf_counter()
        # Optional business dependency, common to every compared CC policy.
        # Keep waiting inside logical latency, before beginning any snapshot.
        if task_start_gate is not None:
            task_start_gate(sequence, manager)
        outcomes = []
        attempt_details = []
        llm_count = tool_count = 0
        input_tokens = output_tokens = wasted_input_tokens = wasted_output_tokens = 0
        tool_ms = wasted_tool_ms = 0.0
        prior_restart_cost = 0.
        for attempt in range(max_attempts):
            attempt_started = time.perf_counter()
            llm_records = [dict(record, elapsed_ms=record["elapsed_ms"] * time_scale)
                           for record in prior["llm_calls"]]
            tool_records = [dict(record, elapsed_ms=record["elapsed_ms"] * time_scale)
                            for record in prior.get("tool_calls", [])]
            reasoner = CapturingReasoner(ReplayReasoner(llm_records))
            tool = CapturingTool(ReplayTool(tool_records)) if tool_records else None
            cursor = frozen_cursor_type(prior, reasoner=reasoner, tool=tool,
                time_scale=time_scale, clock=profile_clock,
                allow_complete_aborted_capture=allow_complete_aborted_capture) if frozen_agent_schedule else workload.cursor(
                task, reasoner=reasoner, tool=tool,
                sleep_fn=lambda seconds: time.sleep(seconds * time_scale))
            metadata = transaction_metadata(system, retry_count=attempt,
                                            policy_commit_batch=policy_commit_batch)
            if carry_prior_restart_cost:
                metadata["prior_retry_cost_ms"] = prior_restart_cost
            metadata.update(_commit_timing_sampled=True,
                            _defer_commit_timing_finalize=True)
            txn = manager.begin(f"frozen:{system}:{sequence}", metadata,
                                snapshot_object_ids=None, strategy=system)
            try:
                if attempt_lifecycle is not None:
                    attempt_lifecycle.on_begin(txn,sequence=sequence,attempt=attempt,max_attempts=max_attempts)
                execution = cursor.execute(txn, before_commit_batch=commit_batch_callback(
                    manager, system=system, observed_commit_admission=False,
                    policy_commit_batch=policy_commit_batch))
                result = txn.commit(system)
                if list(execution.revealed_targets) != prior["access_targets"] or \
                        execution.branch != prior["branch"]:
                    paired = list(zip(execution.revealed_targets, prior["access_targets"]))
                    first = next((i for i, (got, expected) in enumerate(paired)
                                  if got != expected), min(len(execution.revealed_targets),
                                                            len(prior["access_targets"])))
                    raise ValueError(
                        f"frozen access path diverged in {system}: first={first}, "
                        f"actual={len(execution.revealed_targets)}, "
                        f"expected={len(prior['access_targets'])}, "
                        f"branch={execution.branch}/{prior['branch']}")
                committed = bool(result.committed)
                reason = str(result.reason)
            except LockConflict as exc:
                txn.abort_or_observe_wound(exc.reason, strategy=system,
                                          conflict_object_ids=exc.targets)
                committed, reason = False, str(exc.reason)
            except BaseException:
                if txn.result is None:
                    txn.abort_or_observe_wound("frozen execution error", strategy=system)
                raise
            llm_count += len(reasoner.records)
            tool_count += len(tool.records) if tool else 0
            spent_input = sum(int(r["result"]["input_tokens"]) for r in reasoner.records)
            spent_output = sum(int(r["result"]["output_tokens"]) for r in reasoner.records)
            spent_tool_ms = sum(float(r["elapsed_ms"]) for r in tool.records) if tool else 0.0
            input_tokens += spent_input
            output_tokens += spent_output
            tool_ms += spent_tool_ms
            if not committed:
                wasted_input_tokens += spent_input
                wasted_output_tokens += spent_output
                wasted_tool_ms += spent_tool_ms
            outcomes.append((committed, reason))
            action_events = [event for event in txn.events if event.kind == "action-change"]
            paid_replay_work = (sum(float(r["elapsed_ms"]) for r in reasoner.records)
                               + spent_tool_ms + max(0., txn.context.operation_cost_ms))
            lifecycle_evidence = None
            if attempt_lifecycle is not None:
                # Terminal CC outcome and paid calls are complete. No CC lock
                # is held here; control cost remains in attempt/task latency.
                lifecycle_evidence = attempt_lifecycle.on_finish(txn,sequence=sequence,attempt=attempt,
                    will_retry=not committed and attempt+1<max_attempts)
                if not isinstance(lifecycle_evidence,dict):raise ValueError('Attempt lifecycle must return explicit evidence')
            attempt_details.append(dict(
                sequence=sequence, attempt=attempt, committed=committed,
                reason=reason, elapsed_ms=(time.perf_counter()-attempt_started)*1000,
                blocked_ms=txn.context.blocked_time_ms,
                agent_cost_ms=txn.context.agent_cost_ms,
                operation_cost_ms=txn.context.operation_cost_ms,
                paid_replay_work_ms=paid_replay_work,
                carried_prior_restart_cost_ms=txn.context.prior_retry_cost_ms,
                runtime_timing_ms=dict(txn.metadata.get("_commit_timing_ms", {})),
                action_changes=len(action_events),
                final_action=int(txn.context.action.protected),
                logical_task_id=txn.task_id,
                completed_operations=int(txn.context.completed_operations),
                llm_calls=len(reasoner.records),
                policy_read_targets=len(txn.context.policy_read_lock_targets),
                policy_write_targets=len(txn.context.policy_write_lock_targets),
                action_timeline=[dict(at_s=e.at_s, action=e.detail["action"],
                                      added=e.detail["added"]) for e in action_events]))
            if attempt_lifecycle is not None:
                attempt_details[-1]['attempt_lifecycle_evidence']=lifecycle_evidence
            txn.metadata.pop("_defer_commit_timing_finalize", None)
            manager._finalize_commit_timing(txn)
            if committed:
                break
            if carry_prior_restart_cost:
                prior_restart_cost += paid_replay_work
        finished = time.perf_counter()
        if logical_task_feedback:
            # Exactly one observation after all attempts, including terminal
            # failures. Use the same submission origin as the reported batch
            # latency; only already-finished tasks become visible to policies.
            manager.note_agent_task_outcome(committed=outcomes[-1][0],
                latency_ms=(finished-batch_started)*1000)
        return dict(committed=outcomes[-1][0], attempts=len(outcomes),
                    aborts=sum(not committed for committed, _ in outcomes),
                    elapsed_ms=(finished-started)*1000,
                    # All tasks in this closed batch are submitted together.
                    # Preserve legacy worker occupancy for exclusive profiles,
                    # but expose queue-inclusive latency for batch tail metrics.
                    sequence=sequence,
                    queue_wait_ms=(started-batch_started)*1000,
                    submission_elapsed_ms=(finished-batch_started)*1000,
                    llm_calls=llm_count, tool_calls=tool_count,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    wasted_input_tokens=wasted_input_tokens,
                    wasted_output_tokens=wasted_output_tokens,
                    tool_ms=tool_ms, wasted_tool_ms=wasted_tool_ms,
                    attempt_details=attempt_details,
                    reasons=[reason for _, reason in outcomes])

    batch_started = started = time.perf_counter()
    logical_count = len(source["tasks"]) * repeat_factor
    rooted_task = profile_clock.wrap(logical_task, "other", root=True) if profile_clock else logical_task
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = list(pool.map(rooted_task, range(logical_count)))
    elapsed = time.perf_counter()-started
    if attempt_sink is not None:
        attempt_sink.extend(dict(system=system, **task) for task in tasks)
    if recorded_policy:
        decision_sink.extend(recorded_policy.records)
    if trajectory_sink is not None:
        trajectory_sink.extend(dataclasses.asdict(t) for t in
                               manager.trajectory_collector.snapshot())
    committed = sum(task["committed"] for task in tasks)
    if profile_sink is not None:
        from scripts.revision.bridge_profile import summarize
        profile_sink.update(summarize(profile_clock, commits=committed, tasks=tasks))
    latencies = sorted(task["elapsed_ms"] for task in tasks if task["committed"])
    p99 = latencies[max(0, math.ceil(len(latencies)*.99)-1)] if latencies else ""
    submission_latencies = sorted(task["submission_elapsed_ms"] for task in tasks if task["committed"])
    submission_p99 = submission_latencies[max(0, math.ceil(len(submission_latencies)*.99)-1)] if submission_latencies else ""
    return dict(system=system, tasks=len(tasks), workers=workers,
                status="ok" if committed == len(tasks) else "incomplete",
                commits=committed, failed_tasks=len(tasks)-committed,
                attempts=sum(task["attempts"] for task in tasks),
                aborts=sum(task["aborts"] for task in tasks),
                action_changes=sum(a["action_changes"] for task in tasks
                                   for a in task["attempt_details"]),
                blocked_ms=sum(a["blocked_ms"] for task in tasks
                               for a in task["attempt_details"]),
                policy_calls=len(recorded_policy.records) if recorded_policy else "",
                lookup_ms=sum(r["lookup_ms"] for r in recorded_policy.records)
                          if recorded_policy else "",
                throughput_tps=committed/elapsed,
                retry_inclusive_p99_ms=p99,
                worker_retry_inclusive_p99_ms=p99,
                submission_retry_inclusive_p99_ms=submission_p99,
                latency_scope="legacy p99 starts at worker execution; submission_p99 includes closed-batch queue; both include retries, completed tasks only",
                llm_calls=sum(task["llm_calls"] for task in tasks),
                tool_calls=sum(task["tool_calls"] for task in tasks),
                input_tokens=sum(task["input_tokens"] for task in tasks),
                output_tokens=sum(task["output_tokens"] for task in tasks),
                wasted_input_tokens=sum(task["wasted_input_tokens"] for task in tasks),
                wasted_output_tokens=sum(task["wasted_output_tokens"] for task in tasks),
                tool_ms=sum(task["tool_ms"] for task in tasks),
                wasted_tool_ms=sum(task["wasted_tool_ms"] for task in tasks),
                wall_seconds=elapsed, time_scale=time_scale,
                batch_started_monotonic_s=batch_started,
                multi_storage=bool(source.get("multi_storage")),
                performance_guards=bool(performance_guards),
                delayed_write_apply_enabled=bool(manager.delayed_write_apply_enabled),
                priority_enabled=bool(manager.priority_enabled),
                policy_selected_write_admission_only=bool(manager.policy_selected_write_admission_only),
                repeat_factor=repeat_factor,
                independent_recorded_tasks=len(source["tasks"]),
                **(dict(logical_task_feedback=dict(enabled=True,
                    observations=len(tasks), latency_origin='initial batch submission',
                    includes_retries=True, includes_terminal_failures=True))
                   if logical_task_feedback else {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--systems", default="silo,2pl-wait-die")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--repeat-factor", type=int, default=1,
                        help="Repeat captured logical tasks for a labeled replay-only pressure run")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--policy", type=Path, help="Compiled policy fit on independent traces")
    parser.add_argument("--policy-mode", choices=["compiled", "static-hot-write", "local-frequency"],
                        default="compiled")
    parser.add_argument("--frequency-threshold", type=int, default=2)
    parser.add_argument("--attempt-output", type=Path,
                        help="Sanitized per-task retry-inclusive and per-attempt timing/actions")
    parser.add_argument("--decision-output", type=Path,
                        help="Evaluation-only numeric policy input/action audit")
    parser.add_argument("--performance-guards", action="store_true",
                        help="Opt into legacy action-rewriting guards; forbidden during exploration")
    parser.add_argument("--trajectory-output", type=Path,
                        help="Write Aegis training transitions; only with one paper-atcc system")
    parser.add_argument("--exploration-epsilon", type=float, default=0.0,
                        help="Training collection only; held-out evaluation must use zero")
    parser.add_argument("--exploration-masks", default="",
                        help="Optional phase-blind training proposal masks, e.g. 0,4,5,15")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if any(path and path.exists() for path in
           (args.attempt_output, args.trajectory_output, args.decision_output)):
        parser.error("diagnostic output already exists")
    if args.max_attempts < 1 or args.time_scale <= 0 or args.repeat_factor < 1:
        parser.error("positive attempts, repeat factor and time scale required")
    source = json.loads(args.capture.read_text())
    if source.get("mode") != "live":
        parser.error("the input must be a real live capture")
    workers = args.workers or source.get("workers", 1)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems or any(s not in SYSTEMS for s in systems):
        parser.error(f"systems must be among {sorted(SYSTEMS)}")
    if "paper-atcc" in systems and args.policy is None and args.policy_mode == "compiled":
        parser.error("--policy is required for Aegis; no default policy fallback")
    if args.trajectory_output and systems != ["paper-atcc"]:
        parser.error("trajectory output requires only paper-atcc")
    if not 0 <= args.exploration_epsilon <= 1 or (args.exploration_epsilon and
                                                     not args.trajectory_output):
        parser.error("exploration requires trajectory output and epsilon in [0,1]")
    if args.exploration_epsilon and args.performance_guards:
        parser.error("performance guards rewrite sampled actions; disable them for training")
    if args.decision_output and (systems != ["paper-atcc"] or args.exploration_epsilon):
        parser.error("decision output requires one paper-atcc evaluation arm")
    try:
        masks = tuple(int(item) for item in args.exploration_masks.split(",")) if args.exploration_masks else None
    except ValueError:
        parser.error("exploration masks must be comma-separated integers")
    if masks and (not args.exploration_epsilon or any(not 0 <= mask <= 15 for mask in masks)):
        parser.error("four-bit proposal masks require training exploration")
    if args.policy_mode != "compiled" and (args.policy or args.exploration_epsilon):
        parser.error("fixed rules cannot be combined with a compiled policy or training exploration")
    if args.frequency_threshold < 1:
        parser.error("frequency threshold must be positive")
    policy = (FixedProtectionPolicy() if args.policy_mode == "static-hot-write" else
              LocalAccessFrequencyPolicy(args.frequency_threshold)
              if args.policy_mode == "local-frequency" else
              CompiledPhasePolicy.load(args.policy) if args.policy else None)
    transitions = [] if args.trajectory_output else None
    attempts = [] if args.attempt_output else None
    decisions = [] if args.decision_output else None
    rows = [run_system(source, system, workers=workers,
                       max_attempts=args.max_attempts, time_scale=args.time_scale,
                       repeat_factor=args.repeat_factor,
                       policy=policy if system == "paper-atcc" else None,
                       trajectory_sink=transitions if system == "paper-atcc" else None,
                       exploration_epsilon=args.exploration_epsilon,
                       exploration_masks=masks,
                       attempt_sink=attempts,
                       decision_sink=decisions,
                       static_protection=args.policy_mode == "static-hot-write",
                       performance_guards=args.performance_guards)
            for system in systems]
    digest = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        fields = ["capture_sha256", "policy", "policy_mode", "frequency_threshold", *rows[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"capture_sha256": digest,
                             "policy": str(args.policy or ""),
                             "policy_mode": args.policy_mode,
                             "frequency_threshold": args.frequency_threshold, **row})
    if args.attempt_output:
        args.attempt_output.parent.mkdir(parents=True, exist_ok=True)
        args.attempt_output.write_text(json.dumps({"source_sha256": digest,
            "policy_mode": args.policy_mode, "time_scale": args.time_scale,
            "tasks": attempts}, indent=2) + "\n")
    if args.decision_output:
        args.decision_output.parent.mkdir(parents=True, exist_ok=True)
        args.decision_output.write_text(json.dumps({"source_sha256": digest,
            "transitions": decisions}, indent=2) + "\n")
    if args.trajectory_output:
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory_output.write_text(json.dumps({"transitions": transitions,
            "source_sha256": digest, "time_scale": args.time_scale,
            "repeat_factor": args.repeat_factor,
            "performance_guards": args.performance_guards,
            "exploration_epsilon": args.exploration_epsilon,
            "exploration_masks": masks}, indent=2) + "\n")
    print(json.dumps({"source_sha256": digest, "results": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
