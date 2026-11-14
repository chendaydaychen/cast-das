"""Fresh paired training/evaluation on model-selected frozen access intents."""
import argparse
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig, valid_actions
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy
from agent.runtime.residual_policy import ResidualTablePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.runtime.strict_two_phase import StrictTwoPhaseHooks
from scripts.revision.compare_frozen_credit_trace import run_system, FixedProtectionPolicy
from scripts.revision.run_audited_credit_pressure import AuditedManager, LinearScorePolicy, fit_linear
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_p0_protection_hotness import StageBalancedProposal
from scripts.revision.run_p0_actor_sampling import ActorMixtureBehaviorPolicy
from scripts.revision.frozen_actor_collection import (
    actor_from_payload, load_distributions, LateActorMixtureBehavior, training_ancestry_batches,
)


def configure_cost_observations(manager, enabled):
    manager.state_collector.round_interval_features = bool(enabled)
    manager.state_collector.observed_restart_cost_features = bool(enabled)


def configure_frozen_runtime(manager, original):
    manager.track_inflight_agent_cost = original.get('track_inflight_agent_cost', False)
    manager.priority_manager.config = dataclasses.replace(manager.priority_manager.config,
        include_inflight_agent_cost=original.get('include_inflight_agent_cost', False))
    manager.atcc_locks.refresh_priorities_before_wound = original.get('refresh_priorities_before_wound', False)
    manager.exact_hotness_accounting = original.get('exact_hotness_accounting', False)
    manager.consistent_hotness_classification = original.get('consistent_hotness_classification',
        getattr(manager,'consistent_hotness_classification',False))
    manager.reconcile_selected_reads_at_checkpoints = original.get('reconcile_selected_reads_at_checkpoints',
        getattr(manager,'reconcile_selected_reads_at_checkpoints',False))
    configure_cost_observations(manager, original.get('faithful_cost_observations', False))
    manager.state_collector.observed_agent_rounds = original.get('observed_agent_rounds', False)
    manager.complete_category_read_protection = original['complete_category_read_protection']
    manager.stop_cancelled_external_work = original.get('stop_cancelled_external_work', False)
    manager.policy_controlled_retry_protection = original.get('policy_controlled_retries', False)
    if original.get('priority_current_attempt_only', False):
        manager.priority_manager.config = dataclasses.replace(
            manager.priority_manager.config, include_prior_retry_cost=False)


class ObservedReadFirstProposal(StageBalancedProposal):
    """Explore read timing before writes, leaving later write choices available."""
    def probabilities(self, state):
        if state.phase != "refine" or state.write_set_size > 0:
            return super().probabilities(state)
        current = int(state.current_action)
        if not 0 <= current < 16:
            raise ValueError("invalid current mask")
        canonical = {current | mask for mask in (0, 1, 2, 3)}
        read_types = {current | mask for mask in (1, 2)}
        return tuple((.5 / len(canonical) if action in canonical else 0.)
                     + (.5 / len(read_types) if action in read_types else 0.) for action in range(16))


class StageBudgetedBehavior(ActorMixtureBehaviorPolicy):
    """Collector only: preserve late choices with explicitly logged stage budgets."""
    def probabilities(self, state):
        current = int(state.current_action)
        if not 0 <= current < 16 or state.phase not in {"explore", "refine", "commit"}:
            raise ValueError("original phase and legal current mask required")
        valid = valid_actions(current)
        if state.phase == "commit":
            return tuple(1. / len(valid) if a in valid else 0. for a in range(16))
        epsilon = .1 * self.epsilon
        base = self.members[0].probabilities(state)
        return tuple((epsilon / len(valid) if a in valid else 0.)
                     + (1. - epsilon) * base[a] for a in range(16))


class SparseReadTimingProposal(ObservedReadFirstProposal):
    """Collector only: retain later read-timing choices instead of saturating masks."""
    def probabilities(self, state):
        current = int(state.current_action)
        if not 0 <= current < 16:
            raise ValueError("invalid current mask")
        if state.phase != "refine" or state.write_set_size > 0:
            return super().probabilities(state)
        expansions = {current | mask for mask in (1, 2, 3)}
        return tuple((.85 if action == current else 0.)
                     + (.15 / len(expansions) if action in expansions else 0.)
                     for action in range(16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-scale", type=float, default=.02)
    parser.add_argument("--training-scale", type=float, default=.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training-repeat", type=int, default=4)
    parser.add_argument("--evaluation-repeat", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--training-epsilon", type=float, default=1.,
        help="Uniform exploration probability; retain current protection otherwise")
    parser.add_argument("--training-exploration", choices=("uniform-retain", "stage-balanced", "read-first", "stage-budgeted", "sparse-read-timing", "learned-mixture", "late-learned-mixture"),
                        default="uniform-retain")
    parser.add_argument("--training-policy-folder", type=Path,
        help="Frozen Full/NoPhase probability mixture and parameter-preserving continuation update")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--policy-selected-write-admission-only", action="store_true")
    parser.add_argument("--local-context-groups", action="store_true",
        help="Pool volatile global buckets in residual groups; keep continuous feature inputs")
    parser.add_argument("--before-reasoning-checkpoint", action="store_true")
    parser.add_argument("--carry-prior-restart-cost", action="store_true")
    parser.add_argument("--stop-cancelled-external-work", action="store_true")
    parser.add_argument("--faithful-cost-observations", action="store_true")
    parser.add_argument("--policy-controlled-retries", action="store_true",
        help="Observe failed-attempt conflicts without inheriting forced protection")
    parser.add_argument("--priority-current-only", action="store_true")
    parser.add_argument("--positive-wait-event-scaling", action="store_true")
    parser.add_argument("--terminal-cost-scaling", action="store_true")
    parser.add_argument("--reuse-training-folder", type=Path,
        help="Refit frozen training rows; requires the same capture and collection settings")
    parser.add_argument("--logged-action-support", action="store_true",
        help="Use positive logged full-action probabilities for legal support, not extra measured samples")
    parser.add_argument("--ppo-reference-mode", choices=("behavior", "frozen-actor"), default="behavior",
        help="Clipping reference; frozen actor retains logged behavior importance correction")
    parser.add_argument("--fit-only", action="store_true",
        help="Save all fitted models and shared training reports without running evaluation")
    parser.add_argument("--item-window-ms", type=float, default=1000.)
    args = parser.parse_args()
    if (args.training_exploration in {"learned-mixture", "late-learned-mixture"}) != (args.training_policy_folder is not None):
        parser.error("learned-mixture requires --training-policy-folder, other collectors must not use it")
    if args.training_policy_folder and (not args.logged_action_support or args.ppo_reference_mode != "frozen-actor"):
        parser.error("continuation requires exact logged support and frozen-actor clipping reference")
    known_arms = {"full", "nophase", "linear", "static", "hot_always",
                  "hot_global_threshold", "item_threshold3", "item_threshold8", "silo", "strict_2pl"}
    if min(args.evaluation_scale, args.training_scale) <= 0 or min(
            args.workers, args.training_repeat, args.evaluation_repeat, args.max_attempts) < 1:
        parser.error("positive scales, workers, repeats and attempts required")
    if not 0 < args.training_epsilon <= 1:
        parser.error("training epsilon must be in (0, 1]")
    if not 0 < args.item_window_ms < float("inf"):
        parser.error("item window must be finite and positive")
    if args.arms and (set(args.arms) - known_arms or len(set(args.arms)) != len(args.arms)):
        parser.error("unknown or duplicate arm")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "runner_executed.py")
    sources = {split: json.loads((args.capture_folder / f"{split}_live.json").read_text())
               for split in ("train", "evaluation")}
    if any(not s.get("agent_directed") or not all(t["committed"] for t in s["tasks"])
           for s in sources.values()):
        raise ValueError("completed model-directed live captures required")
    if {t["task"]["company_id"] for t in sources["train"]["tasks"]} & {
        t["task"]["company_id"] for t in sources["evaluation"]["tasks"]}:
        raise ValueError("train/evaluation companies overlap")
    protocol = dict(source_sha256={split: hashlib.sha256(
        (args.capture_folder / f"{split}_live.json").read_bytes()).hexdigest() for split in sources},
        training_scale=args.training_scale, evaluation_scale=args.evaluation_scale,
        training_workers=args.workers, training_repeat_factor=args.training_repeat, evaluation_workers=args.workers,
        evaluation_repeat_factor=args.evaluation_repeat, max_attempts=args.max_attempts,
        selected_arms=args.arms or sorted(known_arms), stable_logical_task_identity=True,
        policy_commit_batch=False,
        policy_selected_write_admission_only=args.policy_selected_write_admission_only,
        local_context_groups=args.local_context_groups,
        before_reasoning_checkpoint=args.before_reasoning_checkpoint,
        carry_prior_restart_cost=args.carry_prior_restart_cost,
        stop_cancelled_external_work=args.stop_cancelled_external_work,
        faithful_cost_observations=args.faithful_cost_observations,
        policy_controlled_retries=args.policy_controlled_retries,
        priority_current_attempt_only=args.priority_current_only,
        positive_wait_event_scaling=args.positive_wait_event_scaling,
        terminal_cost_scaling=args.terminal_cost_scaling,
        training_reused=args.reuse_training_folder is not None,
        logged_action_support=args.logged_action_support, fit_only=args.fit_only,
        ppo_reference_mode=args.ppo_reference_mode,
        actor_seed=20261012,
        training_epsilon=args.training_epsilon,
        training_exploration=args.training_exploration,
        shared_behavior_data_for_all_learned_arms=True,
        complete_category_read_protection=True,
        scope="frozen model-selected query/write-intent schedule; not online model replanning",
        latency="successful logical task including all retries; failed tasks separately",
        reasoning="recorded per-call durations replayed; token counts are simulated, not billed",
        deployment="factorized actor compiled scoring; not original pure medoid action table",
        no_test_driven_selection=not (args.terminal_cost_scaling or args.reuse_training_folder
                                     or args.policy_controlled_retries or args.priority_current_only
                                     or args.faithful_cost_observations or args.positive_wait_event_scaling
                                     or args.training_exploration != "uniform-retain"),
        experiment_role=("exploratory observation/reward repair informed by prior evaluation"
                         if args.terminal_cost_scaling or args.reuse_training_folder
                         or args.policy_controlled_retries or args.priority_current_only
                         or args.faithful_cost_observations or args.positive_wait_event_scaling
                         or args.training_exploration != "uniform-retain"
                         else "fixed-source paired comparison"),
        no_new_api_calls=True,
        item_thresholds=[3, 8], item_window_ms_at_scale1=args.item_window_ms,
        global_threshold=.20, no_claim_of_optimized_threshold=True)
    merge_manifest = args.capture_folder / "capture_merge_manifest.json"
    if merge_manifest.exists():
        manifest = json.loads(merge_manifest.read_text())
        if (manifest["merged_training_sha256"] != protocol["source_sha256"]["train"]
                or manifest["evaluation_source"]["sha256"] != protocol["source_sha256"]["evaluation"]):
            raise ValueError("merged capture hashes disagree with provenance")
        protocol.update(capture_merge_manifest_sha256=hashlib.sha256(merge_manifest.read_bytes()).hexdigest(),
                        no_test_driven_selection=False,
                        experiment_role=manifest["experiment_role"])
        shutil.copyfile(merge_manifest, args.output / "capture_merge_manifest.json")
    protocol["executed_source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ROOT/'scripts/revision/compare_frozen_credit_trace.py',
                  ROOT/'agent/workloads/frozen_credit_agent.py', ROOT/'agent/runtime/transaction.py',
                  ROOT/'agent/runtime/paper_hooks.py', ROOT/'agent/cc/atcc/ppo.py',
                  ROOT/'scripts/revision/run_p0_protection_hotness.py',
                  ROOT/'scripts/revision/run_p0_actor_sampling.py',
                  ROOT/'scripts/revision/run_p0_paid_training.py',
                  ROOT/'scripts/revision/run_audited_credit_pressure.py',
                  ROOT/'agent/runtime/residual_policy.py',
                  ROOT/'agent/runtime/priority.py',
                  ROOT/'agent/runtime/atcc_lock_manager.py',
                  ROOT/'agent/runtime/state_collector.py', ROOT/'agent/runtime/trajectory.py')}
    if args.training_policy_folder:
        prior = json.loads((args.training_policy_folder / "protocol.json").read_text())
        if prior["source_sha256"] != protocol["source_sha256"]:
            raise ValueError("Frozen sampling models must use the same disjoint training/evaluation sources")
        runtime_keys = ("local_context_groups", "before_reasoning_checkpoint", "faithful_cost_observations",
                        "carry_prior_restart_cost", "stop_cancelled_external_work",
                        "policy_controlled_retries", "priority_current_attempt_only",
                        "policy_selected_write_admission_only", "complete_category_read_protection")
        if any(prior.get(k) != protocol[k] for k in runtime_keys):
            raise ValueError("Frozen sampling models have different runtime semantics")
        paths = [args.training_policy_folder/f"{arm}_policy.json" for arm in ("full", "nophase")]
        protocol["sampling_model_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        protocol["sampling_model_folder"] = str(args.training_policy_folder.resolve())
        protocol["continuation_training"] = dict(
            actor_parameters_retained=True, behavior_group_reinitialization=False,
            actor_mixture_weights={"full": .5, "nophase": .5},
            uniform_legal_probability=args.training_epsilon,
            on_policy_ppo_claimed=False, state_distribution_corrected=False,
            source="frozen model coefficients; no evaluation decisions or returns loaded for updates")
        if args.training_exploration == "late-learned-mixture":
            protocol["continuation_training"]["stage_mixture_weights"] = LateActorMixtureBehavior.stage_weights
            protocol["continuation_training"]["stage_mixture_complement"] = "retain current mask; collector only"
        helper = ROOT/'scripts/revision/frozen_actor_collection.py'
        protocol["executed_source_sha256"][str(helper.relative_to(ROOT))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    if args.training_exploration in {"stage-budgeted", "sparse-read-timing"}:
        protocol["training_stage_budget"] = dict(
            explore_uniform_probability=.1 * args.training_epsilon,
            refine_uniform_probability=.1 * args.training_epsilon,
            commit="uniform over all legal masks, including before the first write",
            collector_only=True, full_action_propensities_recorded=True,
            no_evaluation_override=True)
        if args.training_exploration == "sparse-read-timing":
            protocol["training_stage_budget"]["refine_proposal"] = dict(
                retain_probability=.85, read_expansion_probability=.15,
                expansion_masks=[1, 2, 3], only_before_observed_writes=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    def execute(label, split, policy=None, training=False, rule=None, system="paper-atcc", static=False):
        def factory(**kwargs):
            manager = AuditedManager(**kwargs)
            configure_frozen_runtime(manager, protocol)
            if rule == "item3" or rule == "item8":
                # Scale observation windows with replay time, not access counts.
                scale = args.training_scale if training else args.evaluation_scale
                manager.interceptor.hooks = ItemFrequencyHooks(manager,
                    minimum_accesses=int(rule[-1]), window_ms=args.item_window_ms * scale)
            elif rule == "hot":
                manager.first_hot_access_events = []
                manager.interceptor.hooks = FirstHotAccessHooks(manager)
            elif rule == "strict":
                manager.interceptor.hooks = StrictTwoPhaseHooks(manager)
            return manager
        attempts, transitions, decisions = [], [], []
        row = run_system(sources[split], system, workers=args.workers,
            max_attempts=args.max_attempts,
            repeat_factor=args.training_repeat if training else args.evaluation_repeat,
            time_scale=args.training_scale if training else args.evaluation_scale,
            policy=policy, exploration_epsilon=(args.training_epsilon if training
                and args.training_exploration == "uniform-retain" else 0.),
            exploration_masks=list(range(16)) if training else None,
            trajectory_sink=transitions if training else None, attempt_sink=attempts,
            decision_sink=None if training else decisions, static_protection=static,
            frozen_agent_schedule=True, manager_factory=factory, policy_commit_batch=False,
            policy_selected_write_admission_only=args.policy_selected_write_admission_only,
            carry_prior_restart_cost=args.carry_prior_restart_cost,
            before_reasoning_checkpoint=args.before_reasoning_checkpoint)
        manager = AuditedManager.instances[-1]
        history = manager.history()
        oracle = verify_history(**history)
        row.update(arm=label, oracle_verdict=oracle["verdict"],
            policy_controlled_retries=manager.policy_controlled_retry_protection,
            priority_config=dataclasses.asdict(manager.priority_manager.config),
            item_changes=sum(e.kind == "item-protection-change" for txn in manager.audit_transactions for e in txn.events))
        (args.output / f"{label}_records.json").write_text(json.dumps(dict(result=row,
            attempts=attempts, transitions=transitions, decisions=decisions,
            history=history, oracle=oracle,
            retry_protection_diagnostics=manager.retry_protection_diagnostics(),
            lock_diagnostics=manager.atcc_locks.snapshot_diagnostics()), indent=2))
        print(json.dumps(row), flush=True)
        if oracle["verdict"] != "verified":
            raise RuntimeError("History audit failed; partial evidence retained")
        return row, transitions, decisions

    started = time.perf_counter()
    behavior = CompiledPhasePolicy([])
    if args.training_exploration in {"learned-mixture", "late-learned-mixture"}:
        collector = LateActorMixtureBehavior if args.training_exploration == "late-learned-mixture" else ActorMixtureBehaviorPolicy
        behavior = collector(20260930,
            load_distributions(args.training_policy_folder, seed=20261012), epsilon=args.training_epsilon)
    elif args.training_exploration != "uniform-retain":
        proposal = (SparseReadTimingProposal() if args.training_exploration == "sparse-read-timing"
                    else ObservedReadFirstProposal() if args.training_exploration in {"read-first", "stage-budgeted"}
                    else StageBalancedProposal())
        collector = (StageBudgetedBehavior if args.training_exploration in {"stage-budgeted", "sparse-read-timing"}
                     else ActorMixtureBehaviorPolicy)
        behavior = collector(20260930, (proposal, proposal), epsilon=args.training_epsilon)
    if args.reuse_training_folder:
        previous = json.loads((args.reuse_training_folder / "protocol.json").read_text())
        matching = ("source_sha256", "training_scale", "training_workers", "training_repeat_factor",
                    "max_attempts", "training_epsilon", "training_exploration",
                    "policy_selected_write_admission_only", "local_context_groups",
                    "before_reasoning_checkpoint", "carry_prior_restart_cost",
                    "stop_cancelled_external_work", "faithful_cost_observations",
                    "policy_controlled_retries", "priority_current_attempt_only")
        if any(previous.get(key) != protocol.get(key) for key in matching):
            raise ValueError("reused training collection does not match requested protocol")
        if previous.get("sampling_model_sha256") != protocol.get("sampling_model_sha256"):
            raise ValueError("reused collection used different frozen sampling models")
        path = args.reuse_training_folder / "training_records.json"
        records = json.loads(path.read_text())
        training_row, transitions = records["result"], records["transitions"]
        if transitions != json.loads((args.reuse_training_folder / "training_transitions.json").read_text())["transitions"]:
            raise ValueError("reused transitions disagree with collection records")
        shutil.copyfile(path, args.output / "training_records.json")
        protocol["reused_training_records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    else:
        training_row, transitions, _ = execute("training", "train", behavior, training=True)
    training_path = args.output / "training_transitions.json"
    training_path.write_text(json.dumps(dict(transitions=transitions)))
    rewarded, reward_report = apply_paper_rewards(
        [policy_transition_from_dict(t, source_id="fresh-agent-schedule") for t in transitions],
        PaperRewardConfig(positive_wait_event_scaling=args.positive_wait_event_scaling,
                          terminal_cost_scaling=args.terminal_cost_scaling))
    actors, policies, reports = {}, {}, {}
    cfg = PPOConfig(conditioned_mask_gradient=True, min_group_samples=4,
                    logged_action_support=args.logged_action_support,
                    ppo_reference_mode=args.ppo_reference_mode,
                    initialize_from_behavior=args.training_policy_folder is None)
    for arm, awareness in (("full", "enabled"), ("nophase", "disabled")):
        rows = rewarded if arm == "full" else [dataclasses.replace(t,
            state=dataclasses.replace(t.state, completed_rounds=0),
            next_state=dataclasses.replace(t.next_state, completed_rounds=0) if t.next_state else None) for t in rewarded]
        actor = (actor_from_payload(json.loads((args.training_policy_folder/f"{arm}_policy.json").read_text()), seed=20261012)
                 if args.training_policy_folder else DiscretePPOPolicy(seed=20261012, phase_awareness=awareness,
                                                                       local_context_groups=args.local_context_groups))
        before = time.perf_counter()
        fit_report = DiscretePPOTrainer(cfg).train(actor, rows)
        compiled = ResidualTablePolicy.compile(actor, generation=2026101202,
                                               remove_phase_index_proxy=arm == "nophase")
        reports[arm] = dict(config=dataclasses.asdict(cfg), fit_report=fit_report,
            fit_compile_seconds=time.perf_counter()-before)
        actors[arm], policies[arm] = actor, compiled
        (args.output / f"{arm}_policy.json").write_text(json.dumps(compiled.to_dict(), indent=2))
    training_report = dict(reward=reward_report, arms=reports,
        training_collect_fit_compile_seconds=time.perf_counter()-started, training_result=training_row)
    if args.reuse_training_folder:
        training_report.update(timing_scope="refitting and compilation only; collection reused, not timed",
                              collection_timing_source=str(args.reuse_training_folder / "training_report.json"))
    report_path = args.output / "training_report.json"
    report_path.write_text(json.dumps(training_report, indent=2))
    linear_started = time.perf_counter()
    linear_prior = training_ancestry_batches(args.training_policy_folder) if args.training_policy_folder else ()
    linear = fit_linear(training_path, report_path, prior_batches=linear_prior)
    (args.output / "linear_policy.json").write_text(json.dumps(linear, indent=2))
    training_report.update(linear_fit_seconds=time.perf_counter()-linear_started,
        training_collect_fit_compile_linear_seconds=time.perf_counter()-started)
    if linear_prior:
        training_report["linear_training_batches"] = linear["training_batches"]
        protocol["linear_common_training_information"] = dict(
            prior_and_current_measured_batches=True, transitions=linear["training_transitions"],
            round_reward_normalization_preserved=True)
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    report_path.write_text(json.dumps(training_report, indent=2))
    if args.fit_only:
        print(json.dumps(dict(status="fit-only", training_report=str(report_path),
                              evaluation_performed=False)), flush=True)
        return
    arms = [("full", policies["full"], None, "paper-atcc", False),
            ("nophase", policies["nophase"], None, "paper-atcc", False),
            ("linear", LinearScorePolicy(linear["weights"]), None, "paper-atcc", False),
            ("static", FixedProtectionPolicy(), None, "paper-atcc", True),
            ("hot_always", HotRulePolicy(), "hot", "paper-atcc", False),
            ("hot_global_threshold", HotRulePolicy(.20), "hot", "paper-atcc", False),
            ("item_threshold3", CompiledPhasePolicy([]), "item3", "paper-atcc", False),
            ("item_threshold8", CompiledPhasePolicy([]), "item8", "paper-atcc", False),
            ("silo", None, None, "silo", False),
            ("strict_2pl", None, "strict", "2pl-wait-die", False)]
    results, fidelity = [], {}
    for arm, policy, rule, system, static in arms:
        if args.arms and arm not in args.arms:
            continue
        row, _, decisions = execute(arm, "evaluation", policy, rule=rule, system=system, static=static)
        results.append(row)
        if arm in actors:
            matches = 0
            for decision in decisions:
                state = phase_aware_state_from_dict(decision["state"])
                if arm == "nophase":
                    state = dataclasses.replace(state, completed_rounds=0)
                probabilities = actors[arm].probabilities(state)
                raw_action = max(valid_actions(state.current_action), key=lambda a: probabilities[a])
                matches += raw_action == decision["action"]
            fidelity[arm] = dict(inputs=len(decisions), matches=matches,
                                action_agreement=matches/len(decisions) if decisions else None)
        (args.output / "summary.json").write_text(json.dumps(dict(protocol=protocol,
            results=results, fidelity=fidelity), indent=2))


if __name__ == "__main__":
    main()
