"""Bounded-scale transfer evaluation; no retraining, new API calls or tuning."""
import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.runtime.residual_policy import ResidualTablePolicy
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from scripts.revision.compare_frozen_credit_trace import run_system, FixedProtectionPolicy
from scripts.revision.run_audited_credit_pressure import AuditedManager, LinearScorePolicy
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime


ARMS = ("full", "nophase", "linear", "static", "hot_always", "item_threshold3")


def load_pure_table(audit_path, policy_folder, arm, budget):
    audit = json.loads(audit_path.read_text())
    train_hash = hashlib.sha256((policy_folder / 'training_transitions.json').read_bytes()).hexdigest()
    policy_hash = hashlib.sha256((policy_folder / f'{arm}_policy.json').read_bytes()).hexdigest()
    if audit['training_sha256'] != train_hash or audit['arms'][arm]['policy_sha256'] != policy_hash:
        raise ValueError('Pure table audit does not match frozen training/actor')
    compiled = audit['arms'][arm][f'all_mask_closed_medoid{budget}']
    filename = compiled['compiled_table_filename']
    if Path(filename).name != filename:
        raise ValueError('Table filename must be relative to audit folder')
    path = audit_path.parent / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != compiled['compiled_table_sha256']:
        raise ValueError('Pure table artifact differs from audited table')
    table = CompiledPhasePolicy.load(path)
    if (table.refinement_actor or table.calibrated_group_actions or table.occ_cold_start_guard
            or table.medoids_per_group != budget or len(table.entries) != compiled['report']['compiled_entries']
            or table.phase_awareness != ('enabled' if arm == 'full' else 'disabled')
            or table.remove_phase_index_proxy != (arm == 'nophase')
            or table.state_projection != json.loads((policy_folder / f'{arm}_policy.json').read_text()).get('state_projection','')):
        raise ValueError('Expected plain phase-matched medoid table with no actor fallback')
    return table, path


def validate_inputs(capture_folder, policy_folder):
    protocol = json.loads((policy_folder / "protocol.json").read_text())
    sources = {}
    for split in ("train", "evaluation"):
        path = capture_folder / f"{split}_live.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["source_sha256"][split]:
            raise ValueError(f"{split} capture differs from frozen-policy protocol")
        source = json.loads(path.read_text())
        if not source.get("agent_directed") or not source.get("tasks") or not all(
                task["committed"] for task in source["tasks"]):
            raise ValueError("completed model-directed captures required")
        sources[split] = source
    companies = [{task["task"]["company_id"] for task in sources[split]["tasks"]}
                 for split in ("train", "evaluation")]
    if companies[0] & companies[1]:
        raise ValueError("training and evaluation companies overlap")
    return protocol, sources["evaluation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-folder", type=Path, required=True)
    parser.add_argument("--policy-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="Explicit evaluation load; keep training protocol frozen")
    parser.add_argument("--repeat-factor", type=int, help="Explicit evaluation batch; default preserves original")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--pure-table-audit", type=Path)
    parser.add_argument("--pure-table-budget", type=int, default=16)
    parser.add_argument("--item-window-ms", type=float, default=1000.,
                        help="Declared frequency-rule window; does not change learned policies")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms")
    if not math.isfinite(args.item_window_ms) or args.item_window_ms <= 0:
        parser.error("positive item observation window required")
    if args.pure_table_audit and (set(args.arms) - {'full', 'nophase'} or args.pure_table_budget < 1):
        parser.error('Pure-table evaluation requires only full/nophase and a positive budget')
    original, source = validate_inputs(args.capture_folder, args.policy_folder)
    workers = args.workers if args.workers is not None else original['evaluation_workers']
    repeat = args.repeat_factor if args.repeat_factor is not None else original['evaluation_repeat_factor']
    if min(workers, repeat) < 1:
        parser.error('Positive evaluation workers and repeat factor required')
    artifacts = {arm: args.policy_folder / f"{arm}_policy.json"
                 for arm in ("full", "nophase", "linear")}
    policies = {arm: ResidualTablePolicy.from_dict(json.loads(artifacts[arm].read_text()))
                for arm in ("full", "nophase")}
    policies["linear"] = LinearScorePolicy(json.loads(artifacts["linear"].read_text())["weights"])
    policies.update(static=FixedProtectionPolicy(), hot_always=HotRulePolicy(),
                    item_threshold3=CompiledPhasePolicy([]))
    table_artifacts = {}
    if args.pure_table_audit:
        for arm in args.arms:
            policies[arm], table_artifacts[arm] = load_pure_table(
                args.pure_table_audit, args.policy_folder, arm, args.pure_table_budget)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(source_protocol=original, evaluation_scale=1., retrained=False,
        selected_arms=args.arms,
        policy_sha256={arm: hashlib.sha256(path.read_bytes()).hexdigest()
                       for arm, path in artifacts.items()},
        scope="frozen-policy replay at captured provider delays; training scale retained in source protocol",
        claims="single exploratory replay; no universal RL/phase superiority",
        metric="successful logical-task latency including retries; report failures separately",
        tail_caveat="small-sample P99; repeated templates are not independent captures",
        deployment="compiled actor scoring, not approximate pure action table",
        new_api_calls=0, threshold=3, item_window_ms=args.item_window_ms,
        runtime_configuration_version=2, workers=workers, repeat_factor=repeat,
        logical_tasks=len(source['tasks'])*repeat,
        explicit_evaluation_load=args.workers is not None or args.repeat_factor is not None,
        policy_controlled_retries=original.get('policy_controlled_retries', False),
        priority_current_attempt_only=original.get('priority_current_attempt_only', False),
        carry_prior_restart_cost=original.get('carry_prior_restart_cost', False),
        stop_cancelled_external_work=original.get('stop_cancelled_external_work', False))
    protocol['faithful_cost_observations'] = original.get('faithful_cost_observations', False)
    if args.pure_table_audit:
        protocol.update(deployment='pure medoid action table; no actor fallback',
            pure_table_budget=args.pure_table_budget,
            pure_table_audit_sha256=hashlib.sha256(args.pure_table_audit.read_bytes()).hexdigest(),
            pure_table_sha256={arm: hashlib.sha256(path.read_bytes()).hexdigest()
                               for arm, path in table_artifacts.items()})
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = []
    for arm in args.arms:
        def factory(**kwargs):
            manager = AuditedManager(**kwargs)
            configure_frozen_runtime(manager, original)
            if arm == "item_threshold3":
                manager.interceptor.hooks = ItemFrequencyHooks(manager,
                    minimum_accesses=3, window_ms=args.item_window_ms)
            elif arm == "hot_always":
                manager.first_hot_access_events = []
                manager.interceptor.hooks = FirstHotAccessHooks(manager)
            return manager
        attempts, decisions = [], []
        row = run_system(source, "paper-atcc", workers=workers,
            max_attempts=original["max_attempts"],
            repeat_factor=repeat, time_scale=1.,
            policy=policies[arm], static_protection=arm == "static",
            frozen_agent_schedule=True, manager_factory=factory,
            policy_commit_batch=original["policy_commit_batch"],
            policy_selected_write_admission_only=original["policy_selected_write_admission_only"],
            before_reasoning_checkpoint=original["before_reasoning_checkpoint"],
            carry_prior_restart_cost=original.get('carry_prior_restart_cost', False),
            attempt_sink=attempts, decision_sink=decisions)
        manager = AuditedManager.instances[-1]
        history = manager.history()
        oracle = verify_history(**history)
        row.update(arm=arm, oracle_verdict=oracle["verdict"],
            priority_config=dataclasses.asdict(manager.priority_manager.config),
            policy_controlled_retries=manager.policy_controlled_retry_protection,
            item_changes=sum(event.kind == "item-protection-change"
                for txn in manager.audit_transactions for event in txn.events))
        (args.output / f"{arm}_records.json").write_text(json.dumps(dict(
            result=row, attempts=attempts, decisions=decisions, history=history, oracle=oracle)))
        results.append(row)
        (args.output / "summary.json").write_text(json.dumps(dict(
            protocol=protocol, results=results, completed_arms=len(results),
            expected_arms=len(args.arms)), indent=2))
        print(json.dumps(row), flush=True)
        if oracle["verdict"] != "verified":
            raise RuntimeError("history audit failed or inconclusive; evidence retained")


if __name__ == "__main__":
    main()
