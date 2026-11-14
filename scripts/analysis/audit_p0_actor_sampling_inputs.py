#!/usr/bin/env python3
"""Independently rebuild sampler parameters and every actual full marginal q."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import (DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig, valid_actions,
                              state_features, policy_group_key, fixed_initial_occ_transition,
                              excluded_policy_transition)
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from agent.runtime.state_collector import policy_observation
from scripts.analysis.audit_p0_claims import read, digest
from scripts.analysis.audit_p0_read_exposure import check_row
from scripts.revision.run_p0_revision import dump


def payload_sha(data):
    return hashlib.sha256((json.dumps(data, indent=2, sort_keys=True)+'\n').encode()).hexdigest()


def parameter_probabilities(params, state):
    # Deliberately no call to sampler or policy.probabilities(). Reconstruct
    # full valid categorical softmax from frozen decision parameters.
    canonical = policy_observation(state, params['phase_awareness'])
    key = policy_group_key(canonical, growth_conditioned=params['growth_conditioned_groups'],
                           local_context=params['local_context_groups'])
    features = state_features(canonical)
    assert len(features) == params['feature_count'] == 37 and params['stay_probability'] == 0.
    residual = params['categorical_group_logits'].get(key, [0.]*16)
    valid = valid_actions(canonical.current_action)
    logits = {a: sum(w*f for w, f in zip(params['categorical_actor_weights'][a], features))+residual[a]
              for a in valid}
    largest = max(logits.values())
    mass = {a: math.exp(v-largest) for a, v in logits.items()}
    total = sum(mass.values())
    return tuple(mass.get(a, 0.)/total for a in range(16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    p = json.loads((args.input/'protocol.json').read_text())
    assert p['source_root'] == str(ROOT)
    assert p['support_files'] == 12 and p['collection_runs'] == 2 and p['gradient_epochs'] == 4
    assert p['epsilon'] == .2 and p['actor_mixture_weights'] == {'full': .5, 'nophase': .5}
    assert p['minimum_action_support'] == 8 and p['sampler_rebuild_seed'] == 9800000
    assert not any(p[k] for k in ('outer_exploration_wrapping', 'reward_formula_changed',
        'priority_formula_changed', 'logged_behavior_probabilities_changed',
        'sql_reasoning_token_retry_budget_changed', 'packed_lookup_enabled', 'real_llm'))
    for rel, sha in {**p['source_sha256'], **p['scripts_sha256']}.items():
        assert digest(ROOT/rel) == sha, rel
    assert digest(next(ROOT.glob('cast_core*.so'))) == p['native_sha256'] == '7b515a4ea8bf6ed6baea9781c7ec8ee5a9b1e7161cf815ac4c0a256d1379b760'
    prior = Path(p['prior'])
    assert digest(prior/'protocol.json') == p['prior_protocol_sha256']
    assert digest(prior.parent/'recent_gradient_audit_1200.json') == p['prior_audit_sha256']
    assert json.loads((prior.parent/'recent_gradient_audit_1200.json').read_text())['structural_checks_passed']
    assert digest(Path(p['lookup_audit'])) == p['lookup_audit_sha256']
    assert json.loads(Path(p['lookup_audit']).read_text())['source_sha256'] == p['source_sha256']
    paths = sorted((prior/'training').glob('behavior_*.json'))
    assert len(paths) == 10 and {f.name:digest(f) for f in paths} == p['prior_inputs']
    raw = [policy_transition_from_dict(t, source_id=str(t.get('source_id') or f.stem))
           for f in paths for t in json.loads(f.read_text())['transitions']]
    old_protocol = json.loads((prior/'protocol.json').read_text())
    parameters = {}
    for name, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        report = json.loads((prior/f'training/{name}_report.json').read_text())
        assert report['seed'] == 9800000 and report['config']['epochs'] == 4
        rows, reward = apply_paper_rewards(raw, PaperRewardConfig(**report['reward']['config']))
        assert reward == report['reward']
        fresh = [t for t in rows if t.source_id in old_protocol['ppo_training_source_ids']]
        actor = DiscretePPOPolicy(seed=9800000, phase_awareness=awareness, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True)
        train = DiscretePPOTrainer(PPOConfig(**report['config'])).train(actor, fresh, calibration_transitions=rows)
        assert train['epochs'] == report['training']['epochs']
        model = actor.compile(generation=4, medoids_per_group=4, occ_cold_start_guard=False, preserve_calibrated_groups=True)
        assert payload_sha(model.to_dict()) == digest(prior/f'training/{name}.json')
        assert payload_sha(model.to_dict()) == digest(args.input/f'sampler/{name}_rebuilt_prior.json')
        expected = dict(phase_awareness=awareness, categorical_actor_weights=actor.categorical_actor_weights,
            categorical_group_logits=dict(actor.categorical_group_logits), local_context_groups=True,
            growth_conditioned_groups=False, stay_probability=0., feature_count=37)
        path = args.input/f'sampler/{name}_parameters.json'
        assert payload_sha(expected) == digest(path)
        parameters[name] = json.loads(path.read_text())
    audit = json.loads((args.input/'training/audit.json').read_text())
    all_paths = sorted((args.input/'training').glob('behavior_*.json'))
    assert len(all_paths) == 12 and {f.name:digest(f) for f in all_paths} == audit['common_input_sha256']
    assert {f.name:digest(f) for f in all_paths if f.name in p['prior_inputs']} == p['prior_inputs']
    rows = read(args.input/'collection.csv')
    assert len(rows) == 2
    checks = forced = excluded = 0
    modal_probabilities = []
    for i, source_id in enumerate(p['ppo_training_source_ids']):
        row = rows[i]
        check_row(row, .75, False)
        assert int(row['seed']) == p['collection_seed_base']+i
        assert int(row['clients']) == 32 and int(row['repeat']) == i and row['variant'] == source_id
        assert row['helper_externality_recording_active'] == 'True' and int(row['failed_tasks']) == 0
        for flag, enabled in p['runtime_flags'].items():
            assert row[flag] == str(enabled), flag
        path = args.input/f'training/{source_id}.json'
        for data in json.loads(path.read_text())['transitions']:
            t = policy_transition_from_dict(data, source_id=source_id)
            assert t.source_id == source_id
            if fixed_initial_occ_transition(t):
                assert t.action == 0 and t.behavior_probability == 1.
                forced += 1
                continue
            valid = valid_actions(t.state.current_action)
            probs = [parameter_probabilities(params, t.state) for params in parameters.values()]
            expected = tuple((.2/len(valid) if a in valid else 0.) + .4*probs[0][a]+.4*probs[1][a]
                             for a in range(16))
            assert len(t.behavior_action_probabilities) == 16
            assert all(math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)
                       for x, y in zip(expected, t.behavior_action_probabilities))
            assert math.isclose(expected[t.action], t.behavior_probability, rel_tol=1e-10, abs_tol=1e-10)
            checks += 1
            if excluded_policy_transition(t):
                excluded += 1
            else:
                modal_probabilities.append(max(expected))
    ledger = json.loads((args.input/'helper_ledger_gate.json').read_text())
    assert ledger['exact_coverage_and_no_duplicate_component_verified']
    assert len(ledger['files']) == 12 and ledger['common_input_sha256'] == audit['common_input_sha256']
    dump(args.output, dict(passed=True, common_input_sha256=audit['common_input_sha256'],
        full_marginal_probability_checks=checks, fixed_initial_occ_excluded=forced,
        no_action_choice_or_deterministic_excluded=excluded, actor_eligible_recent_rows=len(modal_probabilities),
        choice_qmax_mean=statistics.fmean(modal_probabilities), choice_qmax_at_least_075=sum(q>=.75 for q in modal_probabilities),
        sampler_parameters_sha256={n:digest(args.input/f'sampler/{n}_parameters.json') for n in parameters},
        analysis_source_sha256=digest(Path(__file__)), collected_failed_tasks=0,
        model_promoted=False, default_adopted=False, goal_completed=False))
    print(json.dumps(dict(passed=True, probability_checks=checks, actor_eligible_recent_rows=len(modal_probabilities),
        choice_qmax_mean=statistics.fmean(modal_probabilities), choice_qmax_at_least_075=sum(q>=.75 for q in modal_probabilities))), flush=True)


if __name__ == '__main__':
    main()
