#!/usr/bin/env python3
"""Independent replay of both frozen policies on actual instrumented Full states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.runtime.paper_policy import CompiledPhasePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict, policy_observation
from agent.cc.atcc.ppo import policy_group_key
from scripts.revision.run_p0_revision import dump, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    p = json.loads((args.input/'completed.json').read_text())
    path_protocol = args.models/'supported_expansion_path_protocol.json'
    if path_protocol.exists():
        frozen = json.loads(path_protocol.read_text())
        assert digest(Path(frozen['plan'])) == frozen['plan_sha256']
        compiler_qualified = Path(frozen['plan']).name == 'COMPILER_QUALIFIED_EXPANSION_PATH_1977_PLAN.md'
        assert compiler_qualified or Path(frozen['plan']).name == 'SUPPORTED_EXPANSION_PATH_1947_PLAN.md'
        assert digest(Path(frozen['fit_audit'])) == frozen['fit_audit_sha256']
        fit = json.loads(Path(frozen['fit_audit']).read_text())
        assert fit['passed'] and fit['supported_expansion_diagnostic_qualified']
        if compiler_qualified:
            assert fit['strict_supported_expansion_diagnostic_qualified']
        assert frozen['model_sha256'] == p['model_sha256']
        assert (frozen['seed'], frozen['clients'], frozen['measurement_seconds'], frozen['warmup_seconds']) == (
            12871600 if compiler_qualified else 12671600, 16, .5, .2)
        assert (p['requested_clients'], p['requested_measurement_seconds']) == (16, .5)
        for rel, sha in frozen['scripts_sha256'].items():
            assert digest(ROOT/rel) == sha
    assert p['instrumented'] and not p['performance_evidence'] and not p['counterfactual_deployed']
    for rel, sha in p['source_sha256'].items():
        assert digest(ROOT/rel) == sha
    assert digest(next(ROOT.glob('cast_core*.so'))) == p['native_sha256']
    assert digest(args.input/'records.json') == p['records_sha256']
    paths = {'full': args.models/'training/full.json',
             'no_actor': args.models/'actor_influence/full_no_actor_updates.json'}
    assert {name: digest(path) for name, path in paths.items()} == p['model_sha256']
    full, baseline = (CompiledPhasePolicy.load(paths[name]) for name in ('full', 'no_actor'))
    for model in (full, baseline):
        model.unanimous_fallback_lookup = True
        model.exact_categorical_lookup = bool(p.get('exact_categorical_lookup', False))
    records = json.loads((args.input/'records.json').read_text())
    assert len(records) == p['records']
    differences, known, groups, actions = 0, 0, Counter(), Counter()
    for r in records:
        state = phase_aware_state_from_dict(r['state'])
        a, b = int(full.select(state).protected), int(baseline.select(state).protected)
        current = int(state.current_action)
        assert a | current == a and b | current == b
        assert (a, b) == (r['actual_action'], r['counterfactual_action'])
        key = policy_group_key(policy_observation(state, full.phase_awareness),
            local_context=full.local_context_groups, growth_conditioned=full.growth_conditioned_groups)
        assert key == r['group']
        assert r['known_group'] == (key in full.calibrated_group_actions)
        known += r['known_group']
        differences += a != b
        groups[key] += 1
        actions[f'{state.phase}:{a}'] += 1
    changed = {k for k, a in full.calibrated_group_actions.items()
               if baseline.calibrated_group_actions.get(k) != a}
    visited = {k: groups[k] for k in sorted(changed) if groups[k]}
    assert differences == p['actual_side_by_side_actor_differences'] and known == p['known_group_calls']
    assert visited == p['changed_calibrated_groups_visited'] and dict(actions) == p['observed_phase_actions']
    measurement_report = {}
    if 'measurement_start_index' in p:
        start = p['measurement_start_index']
        assert isinstance(start, int) and 0 <= start <= len(records)
        assert all(r['is_measurement'] == (i >= start) for i, r in enumerate(records))
        measured = records[start:]
        measurement_differences = sum(r['actual_action'] != r['counterfactual_action'] for r in measured)
        assert len(measured) == p['measurement_records'] == p['row']['policy_calls']
        assert measurement_differences == p['measurement_actor_differences']
        measured_actions = Counter(f"{r['state']['phase']}:{r['actual_action']}" for r in measured)
        assert dict(measured_actions) == json.loads(p['row']['phase_actions_json'])
        measurement_report = dict(measurement_records=len(measured),
            measurement_actor_differences=measurement_differences, warmup_excluded_from_measurement_gate=True,
            measurement_strict_commit_expansions=sum(r['state']['phase'] == 'commit'
                and int(r['state']['current_action']) != 0
                and r['actual_action'] != int(r['state']['current_action']) for r in measured),
            measurement_phase_current_actions=dict(Counter(
                f"{r['state']['phase']}:{r['state']['current_action']}->{r['actual_action']}" for r in measured)))
    dump(args.output, dict(passed=True, actual_states=len(records), side_by_side_actor_differences=differences,
        changed_calibrated_groups=len(changed), changed_calibrated_groups_visited=visited,
        known_group_calls=known, fallback_calls=len(records)-known,
        actor_only_performance_attribution_supported_on_this_path=differences > 0,
        **measurement_report,
        probe_includes_warmup=True, diagnostic_scope_only=True, general_unreachability_proven=False,
        performance_evidence=False, records_sha256=p['records_sha256'],
        script_sha256=digest(Path(__file__)), model_changed=False))
    print(json.dumps(dict(passed=True, states=len(records), actor_differences=differences,
                         changed_groups_visited=visited), indent=2), flush=True)


if __name__ == '__main__':
    main()
