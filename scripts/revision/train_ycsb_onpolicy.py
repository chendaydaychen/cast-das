"""Measured per-actor rollouts, then PPO updates; no mixture/replay gradients."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOTrainer, PPOConfig, sample_action
from agent.runtime import PaperRewardConfig, apply_paper_rewards
from agent.runtime.context import LockAction
from agent.runtime.residual_policy import ResidualTablePolicy
from scripts.revision.frozen_actor_collection import actor_from_payload, training_ancestry_batches
from scripts.revision.train_ycsb_mixed_prefix import checked_rows, coverage
from scripts.revision.run_audited_credit_pressure import AuditedManager, fit_linear
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.run_frozen_ycsb_shift import make_cohorts, run_cohort


def observation(state, remove_phase_index):
    return dataclasses.replace(state, completed_rounds=0) if remove_phase_index else state


def training_rows(rows, remove_phase_index):
    return [dataclasses.replace(t, state=observation(t.state, remove_phase_index),
        next_state=observation(t.next_state, remove_phase_index) if t.next_state else None) for t in rows]


class OnPolicySampler:
    def __init__(self, actor, remove_phase_index=False):
        self.actor = actor
        self.remove_phase_index = remove_phase_index
        self.lock = threading.RLock()

    def select_with_distribution(self, state):
        with self.lock:
            state = observation(state, self.remove_phase_index)
            probabilities = tuple(self.actor.probabilities(state))
            action = sample_action(self.actor.rng, probabilities, state.current_action)
            return LockAction.from_encoded(action), probabilities


def verify_onpolicy(actor, rows, remove_phase_index):
    maximum = 0.
    for row in rows:
        expected = actor.probabilities(observation(row.state, remove_phase_index))
        actual = row.behavior_action_probabilities
        if len(expected) != len(actual):
            raise ValueError('Missing on-policy distribution')
        maximum = max(maximum, max(abs(a-b) for a,b in zip(actual, expected)))
    if maximum > 1e-10:
        raise ValueError('Rollout probabilities differ from the pre-update actor')
    return dict(transitions=len(rows), maximum_probability_error=maximum,
        behavior_equals_preupdate_actor=True, mixture=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.prior/'protocol.json').read_text())
    calibration = json.loads(args.calibration.read_text())
    delays = [calibration['delay_ms'][p] for p in ('explore','refine','commit')]
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [2026101501, 2026101502]
    cfg = PPOConfig(conditioned_mask_gradient=True, logged_action_support=True,
        min_group_samples=4, ppo_reference_mode='behavior', initialize_from_behavior=False)
    protocol = dict(original, sampling_model_folder=str(args.prior.resolve()),
        sampling_model_sha256={f'{a}_policy.json':hashlib.sha256((args.prior/f'{a}_policy.json').read_bytes()).hexdigest()
                              for a in ('full','nophase')},
        training_seeds=seeds, training_tasks_per_arm=16, training_workers=4,
        training_epsilon=0., training_exploration='own stochastic actor',
        shared_behavior_data_for_all_learned_arms=False, paired_input_seeds=True,
        ppo_reference_mode='behavior', on_policy_collection=True,
        actor_distribution_check_before_each_update=True, replay_gradients=False,
        training_scope='two per-actor on-policy continuation iterations, not from-scratch training',
        evaluation_performed=False, new_api_calls=0,
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        training_input_generator_sha256=hashlib.sha256((ROOT/'scripts/revision/run_frozen_ycsb_shift.py').read_bytes()).hexdigest(),
        no_test_driven_selection=True, predeclared_evaluation_seed=2026101503,
        experiment_role='training protocol diagnostic; fixed-layout phase proxy remains')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    actors = {a:actor_from_payload(json.loads((args.prior/f'{a}_policy.json').read_text()),
                                  seed=original['actor_seed']) for a in ('full','nophase')}
    reward_cfg = PaperRewardConfig(positive_wait_event_scaling=original['positive_wait_event_scaling'],
                                  terminal_cost_scaling=original['terminal_cost_scaling'])
    reports, raw = [], []
    started = time.perf_counter()
    for iteration, seed in enumerate(seeds):
        for arm, actor in actors.items():
            remove = arm == 'nophase'
            before_payload = ResidualTablePolicy.compile(actor, generation=seed,
                remove_phase_index_proxy=remove).to_dict()
            sampler = OnPolicySampler(actor, remove)
            manager = AuditedManager(paper_policy=sampler, record_traces=False, collect_trajectories=True,
                performance_guards_enabled=False, low_conflict_occ_guard=False,
                delayed_write_apply_enabled=True, priority_enabled=True,
                policy_selected_write_admission_only=original['policy_selected_write_admission_only'])
            configure_frozen_runtime(manager, original)
            manager.expensive_operation_checkpoint_enabled = original['before_reasoning_checkpoint']
            for key in range(256):
                manager.register_object(f'shift-key:{key}', '0', kind='row')
            collected = time.perf_counter()
            records = []
            for cohort in make_cohorts(seed, 4):
                record = run_cohort(manager, None, cohort, system='paper-atcc', delays_ms=delays,
                    max_attempts=6, carry_cost=original['carry_prior_restart_cost'])
                if record['oracle']['verdict'] != 'verified':
                    raise RuntimeError('Invalid training history; raw evidence retained')
                records.append(record)
                print(json.dumps(dict(iteration=iteration, arm=arm, **record['summary'])), flush=True)
            rows = [dataclasses.asdict(t) for t in manager.trajectory_collector.snapshot()]
            source = f'onpolicy:{arm}:{seed}'
            for row in rows:
                row.update(source_id=source, txn_id=source+':'+row['txn_id'])
            transitions = checked_rows(rows, source)
            propensity = verify_onpolicy(actor, transitions, remove)
            collect_seconds = time.perf_counter()-collected
            rewarded, reward_report = apply_paper_rewards(transitions, reward_cfg)
            fit_started = time.perf_counter()
            fit = DiscretePPOTrainer(cfg).train(actor, training_rows(rewarded, remove))
            fit_seconds = time.perf_counter()-fit_started
            after_payload = ResidualTablePolicy.compile(actor, generation=seed,
                remove_phase_index_proxy=remove).to_dict()
            batch = dict(arm=arm, iteration=iteration, seed=seed, before_actor=before_payload,
                after_actor=after_payload, transitions=rows, records=records, propensity=propensity,
                reward=reward_report, config=dataclasses.asdict(cfg), fit_report=fit,
                collect_seconds=collect_seconds, fit_seconds=fit_seconds, coverage=coverage(transitions))
            path = args.output/f'{arm}_iteration_{iteration}.json'
            path.write_text(json.dumps(batch, indent=2))
            reports.append(dict(arm=arm, iteration=iteration, seed=seed,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), propensity=propensity,
                collect_seconds=collect_seconds, fit_seconds=fit_seconds, coverage=coverage(transitions)))
            raw.extend(rows)
            (args.output/f'{arm}_policy.json').write_text(json.dumps(after_payload, indent=2))
    path = args.output/'training_transitions.json'
    path.write_text(json.dumps(dict(transitions=raw)))
    report_path = args.output/'training_report.json'
    _, combined_reward = apply_paper_rewards(checked_rows(raw, 'onpolicy'), reward_cfg)
    report = dict(batches=reports, reward=combined_reward,
                  collect_fit_compile_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(report, indent=2))
    linear_start = time.perf_counter()
    linear = fit_linear(path, report_path, prior_batches=training_ancestry_batches(args.prior))
    (args.output/'linear_policy.json').write_text(json.dumps(linear, indent=2))
    report.update(linear_fit_seconds=time.perf_counter()-linear_start,
                  total_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status='trained-not-evaluated', total_seconds=report['total_seconds'])), flush=True)


if __name__ == '__main__':
    main()
