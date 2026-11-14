#!/usr/bin/env python3
"""Two full-probability actor rollouts; shared support, recent-only PPO, gated SQL."""
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from agent.runtime.paper_policy import HotRulePolicy
from scripts.revision import run_p0_read_exposure as study
from scripts.revision.run_p0_paid_training import MixtureBehaviorPolicy
from scripts.revision.run_p0_revision import dump, digest, csv_write
from scripts.analysis.audit_p0_claims import compare, validate_rows


class ActorMixtureBehaviorPolicy(MixtureBehaviorPolicy):
    """Use all member probabilities, not each member's compiled/modal action.

    The inherited single categorical draw records q=.2*uniform+.4*pFull+
    .4*pNoPhase in the FULL valid space. No hidden member draw or outer
    epsilon wrapper. Only the data collector uses this class.
    """
    def __init__(self, seed, members, epsilon=.2):
        super().__init__(seed, epsilon)
        self.members = tuple(members)
        if len(self.members) != 2 or any(not callable(getattr(m, 'probabilities', None)) for m in self.members):
            raise ValueError('two probability-producing members required')

    def probabilities(self, state):
        current = int(state.current_action) & 15
        valid = tuple(a for a in range(16) if (a | current) == a)
        values = [self.epsilon / len(valid) if a in valid else 0. for a in range(16)]
        for member in self.members:
            probs = tuple(member.probabilities(state))
            if (len(probs) != 16 or any(not math.isfinite(p) or p < 0. for p in probs)
                    or not math.isclose(sum(probs), 1., rel_tol=1e-10, abs_tol=1e-10)
                    or any(probs[a] != 0. for a in range(16) if a not in valid)):
                raise ValueError('member must return a normalized, monotonic full distribution')
            for action, probability in enumerate(probs):
                values[action] += (1. - self.epsilon) * probability / len(self.members)
        return tuple(values)


def freeze_actor(actor):
    assert actor.categorical_actor and actor.stay_probability == 0.
    return dict(phase_awareness=actor.phase_awareness, categorical_actor_weights=actor.categorical_actor_weights,
                categorical_group_logits=dict(actor.categorical_group_logits),
                local_context_groups=actor.local_context_groups, growth_conditioned_groups=actor.growth_conditioned_groups,
                stay_probability=actor.stay_probability, feature_count=len(actor.categorical_actor_weights[0]))


def read_transitions(paths):
    return [policy_transition_from_dict(t, source_id=str(t.get('source_id') or p.stem))
            for p in paths for t in json.loads(p.read_text())['transitions']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--lookup-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    prior = json.loads((args.prior / 'protocol.json').read_text())
    prior_audit_path = args.prior.parent / 'recent_gradient_audit_1200.json'
    assert json.loads(prior_audit_path.read_text())['structural_checks_passed']
    lookup = json.loads(args.lookup_audit.read_text())
    assert lookup['passed']
    source = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'agent').rglob('*.py')}
    assert source == lookup['source_sha256']
    paths = sorted((args.prior / 'training').glob('behavior_*.json'))
    assert len(paths) == 10 and {p.name: digest(p) for p in paths} == prior['common_input_sha256']
    flags = prior['runtime_flags']
    reference = Path(prior['reference_models'])
    fresh_ids = ['behavior_10', 'behavior_11']
    dump(args.output / 'protocol.json', dict(scope=__doc__, prior=str(args.prior),
        prior_protocol_sha256=digest(args.prior / 'protocol.json'), prior_audit_sha256=digest(prior_audit_path),
        prior_inputs=prior['common_input_sha256'], lookup_audit=str(args.lookup_audit),
        lookup_audit_sha256=digest(args.lookup_audit), source_root=str(ROOT), source_sha256=source,
        native_sha256=digest(next(ROOT.glob('cast_core*.so'))), runtime_flags=flags,
        scripts_sha256={p: digest(ROOT / p) for p in ('scripts/revision/run_p0_actor_sampling.py',
            'scripts/analysis/audit_p0_actor_sampling_inputs.py', 'scripts/analysis/audit_p0_actor_deployment_influence.py',
            'scripts/revision/run_p0_read_exposure.py', 'scripts/unified_trace/run_credit_review_experiment.py')},
        sampler='actual full categorical actor distributions, not compiled teacher argmax',
        actor_mixture_weights={'full': .5, 'nophase': .5}, epsilon=.2, outer_exploration_wrapping=False,
        collection_runs=2, collection_clients=32, collection_seconds=.75, collection_warmup=0.,
        collection_seed_base=10010000, sampler_seed_base=10011000, sampler_rebuild_seed=9800000,
        training_seed=10030000, ppo_training_source_ids=fresh_ids, gradient_epochs=4, support_files=12,
        reference_models=str(reference), reference_sha256={n: digest(reference / f'{n}.json') for n in ('full', 'nophase')},
        validation_seed_base=10050000, order_seed_base=10060000, variants=['candidate', 'reference', 'hotrule_always'],
        clients=[16, 32], repeats=2, seconds=2., warmup=.2, max_attempts=6,
        selection_rule='reference main geomean>=1.05 each>=.90 window geomean>=1; strong-rule both TPS geomeans>=1.03; completion loss<=.01; zero task failures',
        performance_preflight='both historical AND recent observed deployment actions changed by actor updates',
        reward_formula_changed=False, priority_formula_changed=False, logged_behavior_probabilities_changed=False,
        sql_reasoning_token_retry_budget_changed=False, minimum_action_support=8, packed_lookup_enabled=False,
        cpu_affinity=sorted(os.sched_getaffinity(0)), numa_binding='node0 CPU and memory', real_llm=False))
    training = args.output / 'training'
    training.mkdir(parents=True)
    for p in paths:
        (training / p.name).symlink_to(p.resolve())
    raw = read_transitions(paths)
    actors = {}
    start = time.monotonic()
    for name, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        report = json.loads((args.prior / f'training/{name}_report.json').read_text())
        rows, reward = apply_paper_rewards(raw, PaperRewardConfig(**report['reward']['config']))
        assert reward == report['reward']
        recent = [t for t in rows if t.source_id in prior['ppo_training_source_ids']]
        actor = DiscretePPOPolicy(seed=9800000, phase_awareness=awareness, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True)
        trained = DiscretePPOTrainer(PPOConfig(**report['config'])).train(actor, recent, calibration_transitions=rows)
        assert trained['epochs'] == report['training']['epochs']
        model = actor.compile(generation=4, medoids_per_group=4, occ_cold_start_guard=False, preserve_calibrated_groups=True)
        dump(args.output / f'sampler/{name}_rebuilt_prior.json', model.to_dict())
        assert digest(args.output / f'sampler/{name}_rebuilt_prior.json') == digest(args.prior / f'training/{name}.json')
        # Persist decision parameters before online probabilities() adds any
        # previously unseen prototype/default-zero group. These additions do
        # not modify weights or existing logits; the audit uses default zeros.
        dump(args.output / f'sampler/{name}_parameters.json', freeze_actor(actor))
        actors[name] = actor
    dump(args.output / 'sampler/audit.json', dict(rebuild_seconds=time.monotonic()-start,
        actor_parameters_sha256={n: digest(args.output / f'sampler/{n}_parameters.json') for n in actors}))
    original = study.run_system
    def run_system(**kwargs):
        kwargs.update(flags)
        if kwargs.get('trajectory_output') is not None:
            kwargs['paper_exploration_seed'] = None
        return original(**kwargs)
    study.run_system = run_system
    collection, start = [], time.monotonic()
    for i in range(2):
        sampler = ActorMixtureBehaviorPolicy(10011000 + i, (actors['full'], actors['nophase']))
        row = study.measure(args.output / 'sampler/full_rebuilt_prior.json', sampler, 32, 10010000+i,
            fresh_ids[i], i, trajectory=training / f'{fresh_ids[i]}.json', seconds=.75, warmup=0.)
        assert not validate_rows([row]) and row['failed_tasks'] == 0
        collection.append(row)
        csv_write(args.output / 'collection.csv', collection)
    all_paths = sorted(training.glob('behavior_*.json'))
    dump(training / 'audit.json', dict(common_input_sha256={p.name: digest(p) for p in all_paths},
        collection_seconds=time.monotonic()-start))
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_helper_ledger.py'),
        '--training', str(training), '--expected-files', '12', '--output', str(args.output / 'helper_ledger_gate.json')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_actor_sampling_inputs.py'),
        '--input', str(args.output), '--output', str(args.output / 'inputs_audit.json')], check=True)
    all_raw = read_transitions(all_paths)
    start = time.monotonic()
    for name, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        report = json.loads((reference / f'{name}_report.json').read_text())
        config = PPOConfig(**report['config'])
        old_rows, old_reward = apply_paper_rewards([t for t in all_raw if t.source_id in {f'behavior_{i}' for i in range(8)}],
            PaperRewardConfig(**report['reward']['config']))
        assert old_reward == report['reward']
        default = DiscretePPOPolicy(seed=9610000, phase_awareness=awareness)
        DiscretePPOTrainer(config).train(default, old_rows)
        rebuilt = default.compile(generation=4, medoids_per_group=4, occ_cold_start_guard=False, preserve_calibrated_groups=True)
        dump(args.output / f'default_rebuild/{name}.json', rebuilt.to_dict())
        assert digest(args.output / f'default_rebuild/{name}.json') == digest(reference / f'{name}.json')
        rows, reward = apply_paper_rewards(all_raw, PaperRewardConfig(**report['reward']['config']))
        fresh = [t for t in rows if t.source_id in fresh_ids]
        assert fresh and {t.source_id for t in fresh} == set(fresh_ids)
        actor = DiscretePPOPolicy(seed=10030000, phase_awareness=awareness, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True)
        trained = DiscretePPOTrainer(config).train(actor, fresh, calibration_transitions=rows)
        model = actor.compile(generation=4, medoids_per_group=4, occ_cold_start_guard=False, preserve_calibrated_groups=True)
        dump(training / f'{name}.json', model.to_dict())
        dump(training / f'{name}_report.json', dict(algorithm='discrete-clipped-ppo', config=dataclasses.asdict(config),
            seed=10030000, phase_awareness=awareness, generation=4, medoids_per_group=4, occ_cold_start_guard=False,
            preserve_calibrated_groups=True, categorical_actor=True, ppo_supported_projection=True,
            local_qualified_action_support=True, local_context_groups=True, inputs=[str(p) for p in all_paths],
            ppo_training_source_ids=fresh_ids, gradient_input_transitions=len(fresh), support_input_transitions=len(rows),
            reward=reward, training=trained, implicit_progress_features_retained=True))
    no_phase = study.load(training / 'nophase.json')
    for t in all_raw:
        assert len({int(no_phase.select(dataclasses.replace(t.state, phase=p)).protected)
                    for p in ('explore', 'refine', 'commit')}) == 1
    audit = json.loads((training / 'audit.json').read_text())
    audit.update(model_sha256={n: digest(training / f'{n}.json') for n in ('full', 'nophase')},
                 phase_checks=len(all_raw), training_seconds=time.monotonic()-start, default_rebuilt_byte_identical=True)
    dump(training / 'audit.json', audit)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_actor_deployment_influence.py'),
        '--training', str(training), '--output', str(args.output / 'actor_influence')], check=True)
    influence = json.loads((args.output / 'actor_influence/diagnostic.json').read_text())['results']
    historic = all(r['original_rebuilt_byte_identical'] and not r['compiled_byte_identical']
                   and r['deployed_actions_changed'] > 0 for r in influence.values())
    fresh_pass = all(r['fresh_gradient_observations_changed'] > 0 for r in influence.values())
    dump(args.output / 'preflight_completed.json', dict(actor_influence_gate_passed=historic,
        fresh_actor_influence_gate_passed=fresh_pass, performance_preflight_passed=historic and fresh_pass,
        phase_checks=len(all_raw), model_promoted=False, default_adopted=False, goal_completed=False))
    if not (historic and fresh_pass):
        return
    rows = []
    for c in (16, 32):
        for rep in range(2):
            order = ['candidate', 'reference', 'hotrule_always']
            random.Random(10060000+c*100+rep).shuffle(order)
            for v in order:
                path = training / 'full.json' if v == 'candidate' else reference / 'full.json'
                policy = HotRulePolicy() if v == 'hotrule_always' else study.load(path)
                row = study.measure(path, policy, c, 10050000+c*100+rep, v, rep, seconds=2., warmup=.2)
                assert not row['helper_externality_recording_active']
                rows.append(row)
                csv_write(args.output / 'validation/raw.csv', rows)
    own = compare(rows, 'clients', 'candidate', 'reference')
    strong = compare(rows, 'clients', 'candidate', 'hotrule_always')
    geo = lambda rs, f: math.exp(statistics.fmean(math.log(r[f]) for r in rs))
    completion = all(statistics.fmean(r['task_completion_rate'] for r in rows if r['clients']==c and r['variant']=='candidate')
        >= statistics.fmean(r['task_completion_rate'] for r in rows if r['clients']==c and r['variant']==v)-.01
        for c in (16, 32) for v in ('reference', 'hotrule_always'))
    selected = geo(own, 'tps_ratio') >= 1.05 and min(r['tps_ratio'] for r in own) >= .90 and geo(own, 'window_tps_ratio') >= 1.
    selected &= geo(strong, 'tps_ratio') >= 1.03 and geo(strong, 'window_tps_ratio') >= 1.03 and completion and all(r['failed_tasks']==0 for r in rows)
    dump(args.output / 'completed.json', dict(rows=len(rows), selected=bool(selected), comparisons=own, strongest_rule=strong,
        main_geomean=geo(own, 'tps_ratio'), window_geomean=geo(own, 'window_tps_ratio'), completion_ok=completion,
        raw_sha256=digest(args.output / 'validation/raw.csv'), model_promoted=False, default_adopted=False, goal_completed=False))


if __name__ == '__main__':
    main()
