#!/usr/bin/env python3
"""Independent new feature-frame coverage, frozen proposals, actual q and helper ledger."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig, valid_actions, fixed_initial_occ_transition, excluded_policy_transition
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from agent.runtime.paper_policy import CompiledPhasePolicy
from scripts.revision.run_p0_revision import dump, digest
from scripts.analysis.audit_p0_actor_sampling_inputs import parameter_probabilities, payload_sha
from scripts.analysis.audit_p0_claims import read
from scripts.analysis.audit_p0_read_exposure import check_row


def local(data):
    return tuple(sorted((k, v) for k, v in data.items() if not k.startswith('global_')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    p = json.loads((args.input / 'protocol.json').read_text())
    assert p['runtime_flags']['protection_relevant_hotness'] and p['runtime_flags']['observed_write_read_lock_elision']
    assert not p['historical_rows_used_for_new_training'] and p['support_files'] == p['collection_runs'] == 6
    assert p['minimum_action_support'] == 8 and p['gradient_epochs'] == 4
    assert not any(p[k] for k in ('reward_formula_changed', 'priority_formula_changed', 'sql_reasoning_token_retry_budget_changed',
                                  'outer_exploration_wrapping', 'packed_lookup_enabled', 'default_adopted', 'real_llm'))
    for rel, sha in {**p['source_sha256'], **p['scripts_sha256']}.items():
        assert digest(ROOT / rel) == sha, rel
    assert digest(next(ROOT.glob('cast_core*.so'))) == p['native_sha256']
    assert digest(Path(p['baseline_audit'])) == p['baseline_audit_sha256']
    prior = Path(p['prior'])
    assert digest(prior / 'protocol.json') == p['prior_protocol_sha256']
    old_paths = sorted((prior / 'training').glob('behavior_*.json'))
    old_audit = json.loads((prior / 'training/audit.json').read_text())
    assert {f.name: digest(f) for f in old_paths} == old_audit['common_input_sha256']
    old_raw = [policy_transition_from_dict(t, source_id=str(t.get('source_id') or f.stem))
               for f in old_paths for t in json.loads(f.read_text())['transitions']]
    parameters = {}
    strategy = p.get('proposal_strategy', 'frozen_actor')
    canonical = strategy == 'canonical_masks'
    phase_expansion = strategy == 'phase_expansion_mixture'
    stage_balanced = strategy == 'stage_balanced'
    phase_continuation = strategy in ('phase_continuation', 'deployment_mixture', 'phase_expansion_mixture')
    deployment_mixture = strategy in ('deployment_mixture', 'phase_expansion_mixture')
    continuation = None
    if stage_balanced:
        proposal = json.loads((args.input/'proposal/parameters.json').read_text())
        assert proposal == dict(type='stage_balanced_observed_write_schedule',
            canonical_masks=[0, 5, 10, 15], hot_cold_masks=[5, 10], refine_canonical_weight=.5,
            before_observed_writes='retain_current', after_observed_writes='uniform_valid',
            all_valid_exploration_epsilon=.2, outcomes_or_future_targets_used=False,
            deployed_action_overridden=False)
        assert digest(Path(p['stage_balanced_plan'])) == p['stage_balanced_plan_sha256']
        assert p['collect_only'] and p['collect_and_fit_only']
        assert p['actor_advantage_baseline'] == 'deployment-group'
        assert p['continuation_policy'] is None and p['continuation_policy_sha256'] is None
    elif canonical:
        proposal = json.loads((args.input / 'proposal/parameters.json').read_text())
        assert proposal == dict(type='canonical_monotone_mask_proposal', masks=[0, 5, 10, 15],
            all_valid_exploration_epsilon=.2, outcomes_or_future_targets_used=False)
    elif phase_continuation:
        proposal = json.loads((args.input / 'proposal/parameters.json').read_text())
        assert proposal == dict(type=('refine_deployment_canonical_commit_deployment_uniform_valid' if phase_expansion
                                     else 'refine_half_deployment_half_canonical' if deployment_mixture
                                     else 'refine_exploration_greedy_continuation'), masks=[0, 5, 10, 15],
            exploration_phase='refine', all_valid_exploration_epsilon=.2,
            continuation_sha256=p['continuation_policy_sha256'], outcomes_or_future_targets_used=False)
        assert digest(args.input / 'proposal/continuation.json') == p['continuation_policy_sha256']
        assert digest(Path(p['continuation_policy'])) == p['continuation_policy_sha256']
        continuation = CompiledPhasePolicy.load(args.input / 'proposal/continuation.json')
        assert continuation.protection_relevant_hotness and continuation.phase_awareness == 'enabled'
        assert p['actual_path_preflight_required'] and p['actor_advantage_baseline'] == 'deployment-group'
        if phase_expansion:
            assert digest(Path(p['phase_expansion_plan'])) == p['phase_expansion_plan_sha256']
    for name, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        report = json.loads((prior / f'training/{name}_report.json').read_text())
        rows, reward = apply_paper_rewards(old_raw, PaperRewardConfig(**report['reward']['config']))
        assert reward == report['reward']
        actor = DiscretePPOPolicy(seed=report['seed'], phase_awareness=awareness, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True)
        fresh = [t for t in rows if t.source_id in report['ppo_training_source_ids']]
        trained = DiscretePPOTrainer(PPOConfig(**report['config'])).train(actor, fresh, calibration_transitions=rows)
        assert trained['epochs'] == report['training']['epochs']
        model = actor.compile(generation=4, medoids_per_group=4, preserve_calibrated_groups=True)
        assert payload_sha(model.to_dict()) == digest(prior / f'training/{name}.json')
        expected = dict(phase_awareness=awareness, categorical_actor_weights=actor.categorical_actor_weights,
            categorical_group_logits=dict(actor.categorical_group_logits), local_context_groups=True,
            growth_conditioned_groups=False, stay_probability=0., feature_count=37)
        params_path = args.input / f'sampler/{name}_parameters.json'
        assert payload_sha(expected) == digest(params_path)
        parameters[name] = json.loads(params_path.read_text())
    default = {n: digest(args.input / f'default_rebuild/{n}.json') for n in ('full', 'nophase')}
    assert default == dict(full='5b6b741bc07a59728d2e1324a0e193335914af2c4f6c7a1d7b8346ad2f361cc3',
                          nophase='96e0fc9c4acb807a7a074e9c72f9a5dafc0f729c677997ceef8cfd60c876a80f')
    audit = json.loads((args.input / 'training/audit.json').read_text())
    prior_cost_audit = None
    if p['runtime_flags'].get('carry_prior_restart_cost', False):
        prior_cost_audit = args.input/'restart_cost_inputs_audit.json'
        assert json.loads(prior_cost_audit.read_text())['passed']
    paths = sorted((args.input / 'training').glob('behavior_*.json'))
    assert len(paths) == 6 and {f.name: digest(f) for f in paths} == audit['common_input_sha256']
    counts, qmax = Counter(), []
    collection = read(args.input / 'collection.csv')
    assert len(collection) == 6
    for i, source in enumerate(p['ppo_training_source_ids']):
        row = collection[i]
        check_row(row, .75, False, round_interval=bool(p['runtime_flags'].get('round_interval_features', False)))
        assert int(row['seed']) == p['collection_seed_base'] + i and int(row['clients']) == p['collection_clients'][i]
        assert row['variant'] == source and int(row['failed_tasks']) == 0
        assert row['helper_externality_recording_active'] == 'True'
        for flag, value in p['runtime_flags'].items():
            assert row[flag] == str(value)
        frames = json.loads((args.input / f'feature_frames/{source}.json').read_text())
        frame_keys = set()
        for f in frames:
            if p.get('coherent_collection_footprint_witness', False):
                assert f['footprint_witness_from_collector'] is True
            reads, writes, hot, relevant = (set(f[k]) for k in ('reads', 'writes', 'hot', 'protection_relevant_targets'))
            assert relevant <= reads | writes and writes <= relevant and hot <= reads | writes
            assert len(reads) == f['state']['read_set_size'] and len(writes) == f['state']['write_set_size']
            expected_ratio = len(relevant & hot) / max(1, len(relevant))
            old_ratio = len((reads | writes) & hot) / max(1, len(reads | writes))
            assert math.isclose(expected_ratio, f['state']['hotspot_access_ratio'], abs_tol=1e-12)
            counts['feature_frames'] += 1
            counts['feature_ratio_changed'] += abs(old_ratio - expected_ratio) > 1e-12
            frame_keys.add((f['txn_id'], local(f['state'])))
        transition_data = json.loads((args.input / f'training/{source}.json').read_text())['transitions']
        terminal_counts = Counter(t['txn_id'] for t in transition_data if t['done'])
        assert max(terminal_counts.values(), default=0) <= 1, 'attempt reopened after terminal callback'
        for data in transition_data:
            t = policy_transition_from_dict(data, source_id=source)
            assert t.source_id == source
            assert (t.txn_id, local(data['state'])) in frame_keys
            assert (t.txn_id, local(data['next_state'])) in frame_keys
            counts['raw_transitions'] += 1
            if fixed_initial_occ_transition(t):
                assert t.action == 0 and t.behavior_probability == 1
                counts['fixed_occ'] += 1
                continue
            valid = valid_actions(t.state.current_action)
            if stage_balanced:
                current = int(t.state.current_action)
                if t.state.phase == 'refine':
                    anchors = {current | mask for mask in (0, 5, 10, 15)}
                    pairs = {current | mask for mask in (5, 10)}
                    q = tuple((.2 / len(valid) if action in valid else 0.)
                        + (.4 / len(anchors) if action in anchors else 0.)
                        + (.4 / len(pairs) if action in pairs else 0.) for action in range(16))
                elif t.state.phase == 'commit' and t.state.write_set_size > 0:
                    q = tuple(1. / len(valid) if action in valid else 0. for action in range(16))
                else:
                    q = tuple((.2 / len(valid) if action in valid else 0.)
                        + (.8 if action == current else 0.) for action in range(16))
            elif canonical or phase_continuation:
                anchors = ({mask | int(t.state.current_action) for mask in (0, 5, 10, 15)}
                           if canonical or t.state.phase == 'refine'
                           else {int(continuation.select(t.state).protected)})
                q = tuple((.2 / len(valid) if action in valid else 0) + (.8 / len(anchors) if action in anchors else 0)
                          for action in range(16))
                if deployment_mixture and t.state.phase == 'refine':
                    greedy = int(continuation.select(t.state).protected)
                    assert greedy in valid
                    q = tuple((.2 / len(valid) if action in valid else 0)
                        + (.4 / len(anchors) if action in anchors else 0)
                        + (.4 if action == greedy else 0) for action in range(16))
                if phase_expansion and t.state.phase == 'commit':
                    greedy = int(continuation.select(t.state).protected)
                    assert greedy in valid
                    # Independent marginal: outer uniform .2 plus .8 times
                    # half valid uniform/half frozen deployment proposal.
                    q = tuple((.6 / len(valid) if action in valid else 0)
                              + (.4 if action == greedy else 0) for action in range(16))
            else:
                a, b = [parameter_probabilities(params, t.state) for params in parameters.values()]
                q = tuple((.2 / len(valid) if action in valid else 0) + .4 * a[action] + .4 * b[action] for action in range(16))
            assert len(t.behavior_action_probabilities) == 16
            assert all(math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10) for x, y in zip(q, t.behavior_action_probabilities))
            assert math.isclose(q[t.action], t.behavior_probability, rel_tol=1e-10, abs_tol=1e-10)
            counts['actual_q_checks'] += 1
            if not excluded_policy_transition(t):
                qmax.append(max(q))
    ledger = json.loads((args.input / 'helper_ledger_gate.json').read_text())
    assert ledger['exact_coverage_and_no_duplicate_component_verified']
    assert ledger['common_input_sha256'] == audit['common_input_sha256'] and len(ledger['files']) == 6
    assert counts['feature_ratio_changed'] > 0 and qmax
    dump(args.output, dict(passed=True, counts=dict(counts), actor_eligible_rows=len(qmax),
        choice_qmax_mean=statistics.fmean(qmax), choice_qmax_at_least_075=sum(q >= .75 for q in qmax),
        common_input_sha256=audit['common_input_sha256'], default_model_sha256=default,
        feature_frames_sha256={s: digest(args.input / f'feature_frames/{s}.json') for s in p['ppo_training_source_ids']},
        **({'restart_cost_audit_sha256': digest(prior_cost_audit)} if prior_cost_audit is not None else {}),
        script_sha256=digest(Path(__file__)), old_probabilities_relabelled=False, default_adopted=False, goal_completed=False))
    print(json.dumps(dict(counts=dict(counts), actor_eligible=len(qmax)), indent=2), flush=True)


if __name__ == '__main__':
    main()
