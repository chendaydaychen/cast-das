"""Frozen-model mechanism ablations on the current real CreditCrew mixed input.

Only one declared runtime switch differs per ablation. Reuses the audited
foreground and independent short-update execution, including all retry costs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.runtime.joint_mask_policy import load_scoring_policy
from agent.runtime.state_collector import phase_aware_state_from_dict
from scripts.revision.compare_frozen_credit_trace import FixedProtectionPolicy, run_system
from scripts.revision.run_frozen_credit_short_updates import MixedManager, ShortUpdateHooks
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.prepare_credit_fresh_confirmation import verify_fresh_selection
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.train_credit_joint_masks import audit_training

CASES = {
    'full': dict(static_protection=False, delayed_write_apply_enabled=True, priority_enabled=True),
    'static': dict(static_protection=True, delayed_write_apply_enabled=True, priority_enabled=True),
    'delay_off': dict(static_protection=False, delayed_write_apply_enabled=False, priority_enabled=True),
    'priority_off': dict(static_protection=False, delayed_write_apply_enabled=True, priority_enabled=False),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(output):
    plan = read(output/'plan.json')
    if plan['cases'] != CASES:
        raise ValueError('Declared mechanism settings differ')
    for path,digest in plan['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Executed source changed: '+path)
    training = Path(plan['training'])
    reference = Path(plan['reference'])
    if (read(reference/'input.json') != read(output/'input.json')
            or sha(reference/'input.json') != plan['input_sha256']):
        raise ValueError('Executed input differs from reference')
    for arm,digest in plan['model_sha256'].items():
        if sha(training/f'{arm}_policy.json') != digest:
            raise ValueError('Frozen model changed')
    protocol = read(training/'protocol.json')
    proof = audit_training(training, Path(protocol['sampling_model_folder']))
    if proof != read(training/'training_audit.json'):
        raise ValueError('Training audit does not reproduce')
    fresh = verify_fresh_selection(Path(plan['capture']), training, Path(plan['canonical']),
                                   read(output/'input.json')['tasks'])
    if fresh != plan['fresh_input_proof']:
        raise ValueError('Capture/selection proof changed')
    reference_plan = read(reference/'plan.json')
    if reference_plan != plan['reference_plan']:
        raise ValueError('Reference runtime protocol changed')
    summary = read(output/'summary.json')
    if [r['arm'] for r in summary] != list(CASES):
        raise ValueError('Missing predeclared arm')
    actor = load_scoring_policy(read(training/'full_policy.json'))
    initial, verified, attempts_csv = None, {}, []
    for row in summary:
        arm = row['arm']
        record = read(output/f'{arm}_records.json')
        if row != record['result'] or record['settings'] != CASES[arm]:
            raise ValueError('Result or recorded switch differs')
        for flag in ('delayed_write_apply_enabled', 'priority_enabled'):
            if row[flag] != CASES[arm][flag]:
                raise ValueError('Actual runtime switch differs')
        oracle = verify_history(**record['history'])
        if oracle != record['oracle'] or oracle['verdict'] != 'verified':
            raise ValueError('Combined history failed')
        if initial is None:
            initial = record['history']['initial']
        elif initial != record['history']['initial']:
            raise ValueError('Different initial databases')
        details = [a for task in record['attempts'] for a in task['attempt_details']]
        short = record['short_updates']
        if (len(record['attempts']) != 2 or len(details) != row['attempts']
                or sum(not a['committed'] for a in details) != row['aborts']
                or sum(t['committed'] for t in record['attempts']) != row['commits']
                or len(short) != len(reference_plan['arrival_seconds'])
                or sum(t['committed'] for t in short) != row['short_completed']
                or len(record['history']['attempts']) != len(details)+sum(len(t['attempts']) for t in short)):
            raise ValueError('Attempt/task accounting differs')
        for update,arrival in zip(short, reference_plan['arrival_seconds']):
            if (update['scheduled_arrival_seconds'] != arrival
                    or update['observed_start_seconds']+1e-6 < arrival
                    or update['key'] != reference_plan['short_update_key']):
                raise ValueError('Independent update timing changed')
        policy = FixedProtectionPolicy() if arm == 'static' else actor
        for d in record['decisions']:
            if policy.select(phase_aware_state_from_dict(d['state'])).encoded != d['action']:
                raise ValueError('Recorded policy answer differs')
        if arm == 'static' and row['action_changes'] != 0:
            raise ValueError('Static arm changed categories')
        for a in details:
            attempts_csv.append(dict(arm=arm, sequence=a['sequence'], attempt=a['attempt'],
                committed=a['committed'], reason=a['reason'], llm_calls=a['llm_calls'],
                paid_replay_work_seconds=a['paid_replay_work_ms']/1000,
                final_action=a['final_action'], action_timeline=json.dumps(a['action_timeline'])))
        verified[arm] = dict(records_sha256=sha(output/f'{arm}_records.json'),
            reproduced_policy_answers=len(record['decisions']), history=oracle['verdict'],
            failed_paid_work_seconds=sum(a['paid_replay_work_ms']/1000 for a in details if not a['committed']))
    report = dict(passed=True, arms=verified, training_reproduced=True, new_api_calls=0,
        scope='Exploratory deployment ablation, frozen real inputs and original delays; two tasks not population P99')
    (output/'audit.json').write_text(json.dumps(report, indent=2))
    with (output/'attempts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(attempts_csv[0]))
        writer.writeheader(); writer.writerows(attempts_csv)
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(output, arcname='execution')
        for name in ('protocol.json','training_audit.json','training_report.json','full_policy.json'):
            bundle.add(training/name, arcname='training/'+name)
        for path in plan['source_sha256']:
            bundle.add(ROOT/path, arcname='source/'+path)
    print(json.dumps(dict(**report, archive=str(archive), sha256=sha(archive))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    for name in ('training','reference','capture','canonical'):
        p.add_argument('--'+name, type=Path)
    args = p.parse_args()
    if args.audit_only:
        audit(args.output)
        return
    if any(getattr(args,n) is None for n in ('training','reference','capture','canonical')):
        p.error('Input paths required for execution')
    reference_plan = read(args.reference/'plan.json')
    source = read(args.reference/'input.json')
    original = reference_plan['source_protocol']
    if original != read(args.training/'protocol.json') or reference_plan['workers'] != 2:
        raise ValueError('Frozen training/runtime mismatch')
    fresh = verify_fresh_selection(args.capture, args.training, args.canonical, source['tasks'])
    sources = dict(reference_plan['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    sources['agent/runtime/joint_mask_policy.py'] = sha(ROOT/'agent/runtime/joint_mask_policy.py')
    plan = dict(cases=CASES, training=str(args.training.resolve()), reference=str(args.reference.resolve()),
        capture=str(args.capture.resolve()), canonical=str(args.canonical.resolve()),
        model_sha256={a:sha(args.training/f'{a}_policy.json') for a in ('full','nophase','linear')},
        input_sha256=sha(args.reference/'input.json'), reference_plan=reference_plan,
        fresh_input_proof=fresh, source_sha256=sources,
        role='Predeclared one-switch deployment ablations on already inspected frozen input; no refit')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    (args.output/'input.json').write_bytes((args.reference/'input.json').read_bytes())
    actor = load_scoring_policy(read(args.training/'full_policy.json'))
    results = []
    for arm,settings in CASES.items():
        holder = []
        def factory(**kwargs):
            manager = MixedManager(**kwargs, arrival_seconds=reference_plan['arrival_seconds'],
                update_key=reference_plan['short_update_key'])
            configure_frozen_runtime(manager, original)
            manager.interceptor.hooks = ShortUpdateHooks(manager.interceptor.hooks)
            holder.append(manager)
            return manager
        attempts, decisions = [], []
        try:
            row = run_system(source, 'paper-atcc', workers=2, max_attempts=original['max_attempts'],
                time_scale=1., repeat_factor=1, policy=FixedProtectionPolicy() if arm=='static' else actor,
                attempt_sink=attempts, decision_sink=decisions, frozen_agent_schedule=True,
                manager_factory=factory, policy_commit_batch=original['policy_commit_batch'],
                policy_selected_write_admission_only=original['policy_selected_write_admission_only'],
                before_reasoning_checkpoint=original['before_reasoning_checkpoint'],
                carry_prior_restart_cost=original.get('carry_prior_restart_cost',False),
                allow_complete_aborted_capture=True, **settings)
        finally:
            if holder:
                holder[0].join_short_updates()
        manager = holder[0]
        history = manager.history()
        oracle = verify_history(**history)
        if oracle['verdict'] != 'verified':
            raise ValueError('Invalid combined history')
        short = manager.short_records
        result = dict(arm=arm, **row, short_completed=sum(t['committed'] for t in short),
            short_tasks=len(short), short_aborts=sum(not a['committed'] for t in short for a in t['attempts']),
            max_short_elapsed_seconds=max(t['elapsed_seconds'] for t in short),
            joint_makespan_seconds=max(row['wall_seconds'],max(t['completion_seconds'] for t in short)))
        (args.output/f'{arm}_records.json').write_text(json.dumps(dict(result=result, settings=settings,
            attempts=attempts, decisions=decisions, short_updates=short, history=history, oracle=oracle), indent=2))
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps(results, indent=2))
        with (args.output/'results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
