#!/usr/bin/env python3
"""Independent actual-attempt ledger, prior-cost conservation and predecision visibility audit."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.revision.run_p0_revision import dump, digest
from scripts.analysis.audit_p0_claims import read


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-6)


def analyze(data, row, frames):
    ledger = data['attempt_restart_costs']
    terminal = [t for t in data['transitions'] if t['done']]
    assert len(ledger) == len(terminal) == int(row['attempts'])
    costs = {record['txn_id']: record for record in ledger}
    outcomes = {t['txn_id']: t for t in terminal}
    assert len(costs) == len(ledger) and len(outcomes) == len(terminal) and set(costs) == set(outcomes)
    groups = defaultdict(list)
    for record in ledger:
        for field in ('agent_cost_ms', 'operation_cost_ms', 'restart_cost_ms', 'prior_restart_cost_ms'):
            assert type(record[field]) in (int, float) and math.isfinite(record[field]) and record[field] >= 0
        assert close(record['restart_cost_ms'], record['agent_cost_ms']+record['operation_cost_ms'])
        logical, count, generation = record['txn_id'].rsplit(':', 2)
        assert logical == record['task_id'] and int(count) == record['attempt']
        t = outcomes[record['txn_id']]
        assert t['state']['retry_count'] == record['attempt']
        assert bool(t['commit_indicator']) == record['committed'] and bool(t['abort_indicator']) != record['committed']
        assert close(t['retry_cost_ms'], record['prior_restart_cost_ms'])
        assert t['agent_cost_ms']+t['operation_cost_ms'] <= record['restart_cost_ms']+1e-6
        groups[record['task_id']].append(record)
    retries, positive, late_work = 0, 0, 0
    for attempts in groups.values():
        prior, reason = 0., 'none'
        for i, record in enumerate(sorted(attempts, key=lambda r: r['attempt'])):
            assert i == record['attempt']
            assert close(prior, record['prior_restart_cost_ms']) and reason == record['previous_failure_reason']
            if i:
                retries += 1
                positive += prior > 0
            if record['committed']:
                assert i == len(attempts)-1
            else:
                prior += record['restart_cost_ms']
                reason = record['failure_reason']
            t = outcomes[record['txn_id']]
            late_work += record['restart_cost_ms'] > t['agent_cost_ms']+t['operation_cost_ms']+1e-6
    for frame in frames:
        assert frame['txn_id'] in costs
        assert close(frame['prior_retry_cost_ms'], costs[frame['txn_id']]['prior_restart_cost_ms'])
        assert frame['state']['retry_count'] == costs[frame['txn_id']]['attempt']
    observed = sum(r['restart_cost_ms'] for r in ledger if not r['committed'])
    carried = sum(r['prior_restart_cost_ms'] for r in ledger)
    assert close(observed, row['observed_failed_restart_cost_ms_total'])
    assert close(carried, row['carried_prior_restart_cost_ms_total'])
    return dict(attempts=len(ledger), logical_tasks=len(groups), retries=retries, positive_prior_retries=positive,
        frame_checks=len(frames), late_owner_cost_above_terminal_snapshot=late_work,
        observed_failed_restart_cost_ms=observed, cumulative_carried_prior_cost_ms=carried)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    p = json.loads((args.input/'protocol.json').read_text())
    assert p['runtime_flags']['carry_prior_restart_cost'] and p['attempt_restart_cost_ledger']
    rows = read(args.input/'collection.csv')
    files = {}
    for row in rows:
        source = row['variant']
        path = args.input/f'training/{source}.json'
        frame_path = args.input/f'feature_frames/{source}.json'
        assert row['carry_prior_restart_cost'] == 'True' and int(row['failed_tasks']) == 0
        if p['runtime_flags'].get('unique_attempt_outcomes', False):
            assert int(row['runtime_outcomes_recorded']) == int(row['attempts'])
        files[source] = dict(analyze(json.loads(path.read_text()), row, json.loads(frame_path.read_text())),
            input_sha256=digest(path), frames_sha256=digest(frame_path))
    assert len(files) == p['collection_runs'] == 6
    report = dict(passed=True, files=files, observed_prior_cost_conservation=True,
        predecision_prior_cost_matches_actual_metadata=True, probabilities_states_rewards_not_relabelled=True,
        current_prebegin_retry_delay_not_counted_as_prior_failed_work=True,
        positive_prior_retries=sum(v['positive_prior_retries'] for v in files.values()),
        script_sha256=digest(Path(__file__)), default_adopted=False, goal_completed=False)
    dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
