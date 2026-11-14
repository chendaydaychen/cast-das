"""Shared training-only mixed-prefix rollouts and frozen-actor continuation."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOTrainer, PPOConfig, valid_actions
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from agent.runtime.residual_policy import ResidualTablePolicy
from scripts.revision.frozen_actor_collection import actor_from_payload, load_distributions, training_ancestry_batches
from scripts.revision.run_p0_actor_sampling import ActorMixtureBehaviorPolicy
from scripts.revision.run_audited_credit_pressure import AuditedManager, fit_linear
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.run_frozen_ycsb_shift import make_cohorts, run_cohort


def coverage(rows):
    early = [r for r in rows if r.state.phase == 'explore' and r.state.write_set_size > 0]
    return dict(transitions=len(rows), early_mixed_inputs=len(early),
        early_actions=sorted({r.action for r in early}),
        early_min_interval_ms=min((r.state.inter_round_interval_ms for r in early), default=None),
        early_action_counts={str(a):sum(r.action == a for r in early) for a in sorted({r.action for r in early})})


def checked_rows(rows, source_id):
    transitions = [policy_transition_from_dict(r, source_id=source_id) for r in rows]
    for row in transitions:
        q = row.behavior_action_probabilities
        if (len(q) != 16 or abs(sum(q)-1.) > 1e-9 or any(v < 0 for v in q)
                or row.action not in valid_actions(row.state.current_action)
                or abs(q[row.action]-row.behavior_probability) > 1e-9):
            raise ValueError('Missing or inconsistent measured behavior distribution')
    return transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.prior/'protocol.json').read_text())
    delays = [json.loads(args.calibration.read_text())['delay_ms'][p] for p in ('explore','refine','commit')]
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [2026101401, 2026101402]
    protocol = dict(original, sampling_model_folder=str(args.prior.resolve()),
        sampling_model_sha256={f'{arm}_policy.json':hashlib.sha256((args.prior/f'{arm}_policy.json').read_bytes()).hexdigest()
                              for arm in ('full','nophase')},
        training_seeds=seeds, training_workers=4, training_tasks=16, training_epsilon=.2,
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        shared_behavior_data_for_all_learned_arms=True,
        new_api_calls=0, evaluation_performed=False, on_policy_ppo_claimed=False,
        training_scope='synthetic YCSB mixed-prefix transfer; two shared measured actor-mixture rollouts',
        source_sha256_role='inherited credit training ancestry, not the new synthetic rollout inputs',
        training_input_generator_sha256=hashlib.sha256((ROOT/'scripts/revision/run_frozen_ycsb_shift.py').read_bytes()).hexdigest(),
        experiment_role='training-coverage repair informed by prior diagnosis; independent later inputs required')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    raw, batches = [], []
    for index, seed in enumerate(seeds):
        behavior = ActorMixtureBehaviorPolicy(seed,
            load_distributions(args.prior, seed=original['actor_seed']), epsilon=.2)
        manager = AuditedManager(paper_policy=behavior, record_traces=False, collect_trajectories=True,
            performance_guards_enabled=False, low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True, priority_enabled=True,
            policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
        configure_frozen_runtime(manager, original)
        manager.expensive_operation_checkpoint_enabled = original['before_reasoning_checkpoint']
        for key in range(256):
            manager.register_object(f'shift-key:{key}', '0', kind='row')
        records = []
        for cohort in make_cohorts(seed, 4):
            record = run_cohort(manager, None, cohort, system='paper-atcc', delays_ms=delays,
                max_attempts=6, carry_cost=original['carry_prior_restart_cost'])
            if record['oracle']['verdict'] != 'verified':
                raise RuntimeError('Training history invalid')
            records.append(record)
            print(json.dumps(dict(seed=seed, **record['summary'])), flush=True)
        rows = [dataclasses.asdict(t) for t in manager.trajectory_collector.snapshot()]
        for row in rows:
            row['source_id'] = f'mixed-ycsb:{seed}'
            row['txn_id'] = f'mixed-ycsb:{seed}:'+row['txn_id']
        transitions = checked_rows(rows, f'mixed-ycsb:{seed}')
        if not coverage(transitions)['early_mixed_inputs']:
            raise RuntimeError('Rollout did not collect mixed-prefix states')
        path = args.output/f'rollout_{index}.json'
        path.write_text(json.dumps(dict(seed=seed, records=records, transitions=rows), indent=2))
        batches.append(dict(seed=seed, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **coverage(transitions)))
        raw.extend(rows)
    collect_seconds = time.perf_counter()-started
    path = args.output/'training_transitions.json'
    path.write_text(json.dumps(dict(transitions=raw)))
    transitions = checked_rows(raw, 'mixed-ycsb')
    reward_cfg = PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
                                  terminal_cost_scaling=original['terminal_cost_scaling'])
    rewarded, reward_report = apply_paper_rewards(transitions, reward_cfg)
    cfg = PPOConfig(conditioned_mask_gradient=True, min_group_samples=4,
        logged_action_support=True, ppo_reference_mode='frozen-actor', initialize_from_behavior=False)
    reports = {}
    for arm in ('full','nophase'):
        payload = json.loads((args.prior/f'{arm}_policy.json').read_text())
        actor = actor_from_payload(payload, seed=original['actor_seed'])
        rows = rewarded if arm == 'full' else [dataclasses.replace(t,
            state=dataclasses.replace(t.state, completed_rounds=0),
            next_state=dataclasses.replace(t.next_state, completed_rounds=0) if t.next_state else None) for t in rewarded]
        before = time.perf_counter()
        fit = DiscretePPOTrainer(cfg).train(actor, rows)
        policy = ResidualTablePolicy.compile(actor, generation=2026101401, remove_phase_index_proxy=arm == 'nophase')
        result = policy.to_dict()
        if result['weights'] == payload['weights'] and result['groups'] == payload['groups']:
            raise RuntimeError('Continuation did not update learned coefficients')
        reports[arm] = dict(config=dataclasses.asdict(cfg), fit_report=fit,
                           fit_compile_seconds=time.perf_counter()-before)
        (args.output/f'{arm}_policy.json').write_text(json.dumps(result, indent=2))
    report = dict(reward=reward_report, arms=reports, batches=batches,
        collect_seconds=collect_seconds, coverage=coverage(transitions),
        training_collect_fit_compile_seconds=time.perf_counter()-started)
    report_path = args.output/'training_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    linear_start = time.perf_counter()
    linear = fit_linear(path, report_path, prior_batches=training_ancestry_batches(args.prior))
    (args.output/'linear_policy.json').write_text(json.dumps(linear, indent=2))
    report.update(linear_fit_seconds=time.perf_counter()-linear_start,
                  training_collect_fit_compile_linear_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(report, indent=2))
    for name, expected in protocol['sampling_model_sha256'].items():
        if hashlib.sha256((args.prior/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen sampling models changed')
    print(json.dumps(dict(status='trained-not-evaluated', **report['coverage'],
                         elapsed_seconds=report['training_collect_fit_compile_linear_seconds'])), flush=True)


if __name__ == '__main__':
    main()
