#!/usr/bin/env python3
"""Instrumented actual Full path, side-by-side actor counterfactual; NOT a performance experiment."""
import argparse
from collections import Counter
import dataclasses
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.revision import run_p0_read_exposure as study
from scripts.revision.run_p0_revision import dump, digest
from agent.cc.atcc.ppo import policy_group_key
from agent.runtime.state_collector import policy_observation


class PathProbe:
    def __init__(self, actual, counterfactual):
        self.base, self.counterfactual = actual, counterfactual
        self.records = []
        self.measurement_start = 0

    def select(self, state):
        selected = self.base.select(state)
        other = self.counterfactual.select(state)
        canonical = policy_observation(state, self.base.phase_awareness)
        key = policy_group_key(canonical, local_context=self.base.local_context_groups,
                               growth_conditioned=self.base.growth_conditioned_groups)
        self.records.append(dict(state=dataclasses.asdict(state), group=key,
            actual_action=int(selected.protected), counterfactual_action=int(other.protected),
            known_group=key in self.base.calibrated_group_actions))
        return selected


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def positive_seconds(value):
    import math
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=10901600)
    parser.add_argument('--clients', type=positive_int, default=16)
    parser.add_argument('--seconds', type=positive_seconds, default=.5)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    p = json.loads((args.input/'protocol.json').read_text())
    source = {str(f.relative_to(ROOT)): digest(f) for f in (ROOT/'agent').rglob('*.py')}
    assert source == p['source_sha256']
    actual = study.load(args.input/'training/full.json')
    other = study.load(args.input/'actor_influence/full_no_actor_updates.json')
    for policy in (actual, other):
        policy.unanimous_fallback_lookup = True
        policy.packed_fallback_lookup = False
        policy.exact_categorical_lookup = bool(p.get('exact_categorical_lookup', False))
    probe = PathProbe(actual, other)
    original_run = study.run_system
    original_meter = study.MeasurementMeter
    class ProbeMeter(original_meter):
        def reset(self):
            super().reset()
            # Called by the real runner after warmup has fully drained.
            self.base.measurement_start = len(self.base.records)
    def run_system(**kwargs):
        kwargs.update(p['runtime_flags'])
        return original_run(**kwargs)
    study.run_system = run_system
    study.MeasurementMeter = ProbeMeter
    try:
        row = study.measure(args.input/'training/full.json', probe, args.clients, args.seed,
                            'full_path_diagnostic', seconds=args.seconds, warmup=.2)
    finally:
        study.run_system = original_run
        study.MeasurementMeter = original_meter
    assert row['failed_tasks'] == 0
    for i, record in enumerate(probe.records):
        record['is_measurement'] = i >= probe.measurement_start
    measured = probe.records[probe.measurement_start:]
    assert len(measured) == row['policy_calls']
    dump(args.output/'records.json', probe.records)
    groups = Counter(r['group'] for r in probe.records)
    differences = {k for k, a in actual.calibrated_group_actions.items()
                   if other.calibrated_group_actions.get(k) != a}
    dump(args.output/'completed.json', dict(scope=__doc__, records=len(probe.records),
        includes_warmup_decisions=True, actual_side_by_side_actor_differences=sum(
            r['actual_action'] != r['counterfactual_action'] for r in probe.records),
        measurement_start_index=probe.measurement_start, measurement_records=len(measured),
        measurement_actor_differences=sum(r['actual_action'] != r['counterfactual_action'] for r in measured),
        known_group_calls=sum(r['known_group'] for r in probe.records),
        observed_phase_actions=dict(Counter(f"{r['state']['phase']}:{r['actual_action']}" for r in probe.records)),
        changed_calibrated_groups=sorted(differences),
        changed_calibrated_groups_visited={k: groups[k] for k in sorted(differences) if groups[k]},
        visited_groups=dict(groups), source_sha256=source, native_sha256=p['native_sha256'],
        model_sha256={'full': digest(args.input/'training/full.json'),
                      'no_actor': digest(args.input/'actor_influence/full_no_actor_updates.json')},
        records_sha256=digest(args.output/'records.json'), script_sha256=digest(Path(__file__)),
        exact_categorical_lookup=bool(p.get('exact_categorical_lookup', False)),
        requested_clients=args.clients, requested_measurement_seconds=args.seconds,
        instrumented=True, performance_evidence=False, counterfactual_deployed=False,
        row=row, general_unreachability_proven=False))
    print(json.dumps(dict(records=len(probe.records), differences=sum(
        r['actual_action'] != r['counterfactual_action'] for r in probe.records),
        measurement_records=len(measured), measurement_differences=sum(
            r['actual_action'] != r['counterfactual_action'] for r in measured),
        changed_groups_visited={k: groups[k] for k in sorted(differences) if groups[k]}), indent=2), flush=True)


if __name__ == '__main__':
    main()
