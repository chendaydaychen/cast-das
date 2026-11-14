"""Offline checks of frozen cross-family CreditCrew transfer evidence."""
import argparse
from collections import Counter
import csv
import json
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.runtime.joint_mask_policy import JointMaskTablePolicy, load_scoring_policy
from agent.runtime.paper_policy import HotRulePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict
from agent.runtime.state_collector import policy_observation
from agent.runtime.table_observation import project_table_state, projected_context_key
from agent.cc.atcc.ppo import policy_group_key
from scripts.revision.audit_credit_retry_costs import summarize
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.evaluate_phase_pipeline_frozen import load_models
from scripts.revision.run_audited_credit_pressure import digest
from scripts.revision.run_current_credit_ablations import read, sha
from scripts.revision.compare_frozen_credit_trace import FixedProtectionPolicy,FixedReadProtectionPolicy


def audit(folder):
    plan = read(folder/'protocol.json')
    ref = Path(plan['reference'])
    if sha(folder/'runner_executed.py') != plan['source_sha256']:
        raise ValueError('Runner snapshot changed')
    for name,digest_value in plan.get('shared_source_sha256',{}).items():
        if sha(folder/'shared_sources'/name) != digest_value:
            raise ValueError('Shared executed source snapshot differs')
    for name, checksum in plan['reference_artifact_sha256'].items():
        if sha(ref/name) != checksum:
            raise ValueError('Frozen artifact changed')
    source = read(folder/'input.json')
    if (sha(folder/'input.json') != plan['capture_sha256']
            or sha(Path(plan['capture_reference'])/'input.json') != plan['capture_sha256']
            or plan['retrained'] or plan['retuned'] or plan['new_api_calls']
            or plan['time_scale'] != 1.
            or plan['workers'] not in (2,4,8) or plan['repeat_factor'] not in (1,2,4)):
        raise ValueError('Input or declared transfer scope changed')
    if plan['item_rule'].get('calibration'):
        from scripts.revision.credit_frequency_window import training_window
        path=Path(plan['item_rule']['calibration']);calibration=read(path)
        training_path=path.parent/'training_input.json';train=read(training_path)
        if (sha(path)!=plan['item_rule']['calibration_sha256'] or
                sha(training_path)!=calibration['training_input_sha256'] or
                training_window(train)[0]!=plan['item_rule']['window_ms'] or
                {t['task']['company_id'] for t in train['tasks']} & {t['task']['company_id'] for t in source['tasks']}):
            raise ValueError('Training-only frequency calibration differs')
    if plan.get('mixed_design_sha256'):
        from scripts.revision.merge_credit_evaluation_captures import merge
        capture=Path(plan['capture_reference']);design=read(capture/'design.json')
        if sha(capture/'design.json')!=plan['mixed_design_sha256']:raise ValueError('Mix design changed')
        originals=[]
        for item in design['sources']:
            path=capture/item['copied_filename']
            if sha(path)!=item['sha256']:raise ValueError('Mixed source changed')
            originals.append(read(path))
        reconstructed,origins=merge(originals,set(design['forbidden_companies']))
        if reconstructed!=source or origins!=design['origins']:raise ValueError('Mixed task payload changed')
    update=plan.get('shared_update')
    if bool(update)!=bool(plan['short_updates_injected']):raise ValueError('Short-update declaration inconsistent')
    if update:
        import math
        arrivals=update['arrivals_seconds']
        if (not arrivals or sorted(set(arrivals))!=arrivals or
                any(not math.isfinite(v) or v<0 for v in arrivals) or
                not math.isfinite(update['deadline_seconds']) or update['deadline_seconds']<=0 or
                update['max_attempts']!=4001 or update['backoff_seconds']!=.01):
            raise ValueError('Undeclared short-update scheduling')
        common=set.intersection(*[{e['object_id'] for e in t['events'] if e['event']=='read'} for t in source['tasks']])
        written={e['object_id'] for t in source['tasks'] for e in t['events'] if e['event']=='write'}
        if update['key'] not in common or update['key'] in written:raise ValueError('Update is not shared read-only foreground evidence')
    credit = Path(plan['credit_training'])/'full_policy.json'
    if sha(credit) != plan['credit_full_sha256']:
        raise ValueError('Credit model changed')
    models = load_models(ref)
    policies = dict(full=models['full_table'], nophase=models['nophase_table'],
        linear_fitted=models['linear'],static=FixedProtectionPolicy(),static_read=FixedReadProtectionPolicy(),
        linear_parent=models['linear_parent'], hot_always=HotRulePolicy(),
        credit_full=load_scoring_policy(read(credit)), silo=None)
    from agent.runtime.paper_policy import CompiledPhasePolicy
    policies['item_threshold3'] = CompiledPhasePolicy([])
    if plan.get('exact_feature_tables'):
        from agent.runtime.feature_score_table import FeatureScoreTable
        for arm in ('full','nophase'):
            path = Path(plan['exact_feature_tables'])/f'{arm}_feature_table.json'
            if sha(path) != plan['feature_table_sha256'][str(path)]:
                raise ValueError('Feature table changed')
            artifact = read(path)
            if artifact != FeatureScoreTable.compile(read(ref/f'training_view/{arm}_policy.json')).to_dict():
                raise ValueError('Feature lookup is not the exact frozen actor')
            policies[arm] = FeatureScoreTable(artifact)
    guarded_actors={}
    if plan.get('continuation_cost_guard'):
        from scripts.revision.build_continuation_cost_guard import load_guarded
        folder_guard=Path(plan['continuation_cost_guard'])
        if sha(folder_guard/'guard.json')!=plan['guard_config_sha256']:raise ValueError('Guard config changed')
        guarded,guarded_actors,metadata=load_guarded(ref,folder_guard)
        if metadata!=plan['guard_metadata']:raise ValueError('Guard metadata changed')
        policies.update(guarded)
    if (not plan['arms'] or len(set(plan['arms'])) != len(plan['arms'])
            or set(plan['arms']) - set(policies)):
        raise ValueError('Arm matrix changed')
    policies = {arm:policies[arm] for arm in plan['arms']}
    rows = read(folder/'results.json')
    if [r['arm'] for r in rows] != list(policies):
        raise ValueError('Incomplete arm matrix')
    initial = None
    checked = []
    expected_footprints = Counter((
        tuple(sorted({digest(e['object_id']) for e in t['events'] if e['event']=='read'})),
        tuple(sorted({digest(e['object_id']) for e in t['events'] if e['event']=='write'})))
        for t in source['tasks'])
    expected_footprints = Counter({key:value*plan['repeat_factor'] for key,value in expected_footprints.items()})
    logical_count = len(source['tasks'])*plan['repeat_factor']
    for row in rows:
        arm = row['arm']; record = read(folder/f'{arm}.json')
        if record['summary'] != row:
            raise ValueError('Summary differs from raw record')
        settings=plan.get('mechanism_settings',dict(delayed_write_apply_enabled=True,priority_enabled=True))
        if set(settings)!={'delayed_write_apply_enabled','priority_enabled'} or any(type(v) is not bool for v in settings.values()):
            raise ValueError('Invalid declared mechanism switches')
        for flag,value in settings.items():
            if row[flag]!=value:raise ValueError('Actual runtime switch differs: '+flag)
        history = record['history']; oracle = verify_history(**history)
        if oracle != record['oracle'] or oracle['verdict'] != 'verified':
            raise ValueError('Serial history verification failed')
        background=record.get('background_history',[])
        background_ids={r['history_id'] for r in background}
        if len(background_ids)!=len(background):raise ValueError('Duplicate short-attempt identity')
        foreground_history=[a for a in history['attempts'] if a['id'] not in background_ids]
        short_history=[a for a in history['attempts'] if a['id'] in background_ids]
        short=record.get('short_updates',[])
        if update:
            from collections import Counter as Counts
            from agent.workloads.credit_review import CreditReviewWorkload,CreditReviewConfig
            if sorted(s['index'] for s in short)!=list(range(len(update['arrivals_seconds']))):
                raise ValueError('Missing scheduled update')
            key=digest(update['key'])
            fixture=CreditReviewWorkload(CreditReviewConfig(**source['config']))
            initial_value=next(spec.initial_value for spec in fixture.objects() if spec.object_id==update['key'])
            if (not isinstance(initial_value,str) or str(int(initial_value))!=initial_value or
                    digest(initial_value)!=history['initial'][key]['value_hash']):
                raise ValueError('Update initial value differs from common fixture')
            for s in short:
                carried=0.;previous_elapsed=0.
                for attempt in s['attempts']:
                    if plan.get('carry_short_retry_wait'):
                        blocked=attempt['attempt_blocked_ms']
                        if (blocked<0 or abs(attempt['prior_retry_blocked_ms']-carried)>1e-7
                                or blocked>(attempt['elapsed_since_arrival_s']-previous_elapsed)*1000.+1.):
                            raise ValueError('Retry-wait credit differs from prior actual wait')
                        carried+=blocked
                    previous_elapsed=attempt['elapsed_since_arrival_s']
                if (s['key']!=update['key'] or s['scheduled_arrival_seconds']!=update['arrivals_seconds'][s['index']] or
                        s['observed_start_seconds']+.01<s['scheduled_arrival_seconds'] or
                        not 0<len(s['attempts'])<=update['max_attempts'] or
                        any(a['committed'] for a in s['attempts'][:-1]) or
                        s['committed']!=s['attempts'][-1]['committed']):
                    raise ValueError('Short update violates declared execution')
            if len(short_history)!=sum(len(s['attempts']) for s in short):raise ValueError('Missing short history')
            if Counts(a['committed'] for a in short_history)!=Counts(a['committed'] for s in short for a in s['attempts']):
                raise ValueError('Short outcomes differ from history')
            for a in short_history:
                if set(a['reads'])-{key} or set(a['writes'])-{key}:raise ValueError('Short access footprint changed')
                if a['committed']:
                    if set(a['reads'])!={key} or set(a['writes'])!={key} or a['writes'][key]['version']!=a['reads'][key]['version']+1:
                        raise ValueError('Short successful version changed')
                    next_value=str(int(initial_value)+a['writes'][key]['version']-history['initial'][key]['version'])
                    if a['writes'][key]['value_hash']!=digest(next_value):raise ValueError('Short increment value changed')
            expected=dict(short_tasks=len(short),short_commits=sum(s['committed'] for s in short),
                short_attempts=sum(len(s['attempts']) for s in short),
                max_short_elapsed_seconds=max(s['elapsed_seconds'] for s in short),
                joint_wall_seconds=max(row['wall_seconds'],max(s['completion_seconds'] for s in short)))
            if any(row[k]!=v for k,v in expected.items()):raise ValueError('Short summary differs')
        elif background or short:
            raise ValueError('Unexpected injected updates')
        if plan.get('durable_storage'):
            root=folder/('durable_'+arm)
            if not record.get('durable_reopen',{}).get('passed'):raise ValueError('Durable reopen missing')
            with sqlite3.connect('file:'+str(root/'coordinator.sqlite')+'?mode=ro',uri=True) as db:
                routes=dict(db.execute('SELECT key,storage FROM routes').fetchall())
                if db.execute("SELECT count(*) FROM decisions WHERE status!='DONE'").fetchone()[0]:raise ValueError('Pending durable publication')
            with sqlite3.connect('file:'+str(root/'records.sqlite')+'?mode=ro',uri=True) as db:
                sql_values={k:json.loads(v) for k,v in db.execute('SELECT key,image FROM kv').fetchall()}
            file_values=read(root/'audit.json')['kv']
            actual={}
            for key,storage in routes.items():
                if storage not in ('default','audit'):raise ValueError('Unexpected engine')
                image=(sql_values if storage=='default' else file_values).get(key)
                if image is None or not image['exists']:raise ValueError('Durable value missing')
                actual[digest(key)]=dict(version=image['version'],value_hash=digest(image['value']))
            if actual!=history['final'] or record['durable_reopen']['objects']!=len(actual):raise ValueError('Persisted history differs')
        if initial is not None and initial != history['initial']:
            raise ValueError('Unequal initial data')
        initial = history['initial']
        costs, task_rows = summarize(dict(tasks=record['attempts'], time_scale=1.,
            source_sha256=plan['capture_sha256']), arm)
        if sorted(r['sequence'] for r in task_rows) != list(range(logical_count)):
            raise ValueError('Missing or duplicated logical task')
        attempts = [a for t in record['attempts'] for a in t['attempt_details']]
        for task in record['attempts']:
            original = source['tasks'][task['attempt_details'][0]['sequence'] % len(source['tasks'])]
            for a in task['attempt_details']:
                n = a['llm_calls']
                if not 0 <= n <= len(original['llm_calls']):
                    raise ValueError('Replay call count exceeds frozen schedule')
                # Saved attempt counts preserve a prefix of the frozen calls;
                # tool count is only task-level, so check its admissible range.
                base = sum(c['elapsed_ms'] for c in original['llm_calls'][:n])
                residual = a['paid_replay_work_ms'] - base - max(0., a['operation_cost_ms'])
                if residual < -1e-6 or residual > sum(c['elapsed_ms'] for c in original.get('tool_calls', []))+1e-6:
                    raise ValueError('Paid replay work inconsistent with captured delays')
                if a['committed'] and n != len(original['llm_calls']):
                    raise ValueError('Successful attempt skipped captured model calls')
        for key in ('tasks', 'commits', 'attempts', 'aborts'):
            if row[key] != costs[key]:
                raise ValueError('Attempt counters differ')
        if len(foreground_history) != row['attempts']:
            raise ValueError('History omits attempts')
        actual_footprints = Counter((tuple(sorted(a['reads'])), tuple(sorted(a['writes'])))
            for a in foreground_history if a['committed'])
        if row['commits'] == logical_count and actual_footprints != expected_footprints:
            raise ValueError('Committed access footprints differ from capture')
        if abs(row['throughput_tps'] - row['commits']/row['wall_seconds']) > 1e-12:
            raise ValueError('Throughput denominator differs')
        if row['retry_inclusive_p99_ms'] != max(t['elapsed_ms'] for t in record['attempts'] if t['committed']):
            raise ValueError('Small-sample maximum omits retries')
        changes = sum(a['action_changes'] for a in attempts)
        if row['action_changes'] != changes or row['actual_changes'] != changes:
            raise ValueError('Change counter differs')
        if arm in ('static','static_read') and changes:raise ValueError('Static changed protection mask')
        if row.get('item_protection_changes',0) != len(record.get('item_protection_events',[])):
            raise ValueError('Per-item protection changes differ')
        if arm=='item_threshold3':
            queues={}
            for decision in record['item_decisions']:
                times=queues.setdefault(decision['key'],[])
                cutoff=decision['observed_ns']-int(plan['item_rule']['window_ms']*1_000_000)
                times[:]=[value for value in times if value>=cutoff]
                times.append(decision['observed_ns'])
                if (decision['recent_accesses']!=len(times) or
                        decision['protect']!=(len(times)>=plan['item_rule']['minimum_accesses'])):
                    raise ValueError('Local frequency rule differs from declared window')
        for key in ('llm_calls', 'tool_calls', 'input_tokens', 'output_tokens',
                    'wasted_input_tokens', 'wasted_output_tokens', 'tool_ms', 'wasted_tool_ms'):
            if abs(row[key] - sum(t[key] for t in record['attempts'])) > 1e-6:
                raise ValueError('Task call/token/cost totals differ')
        failed_work = sum(a['paid_replay_work_ms'] for a in attempts if not a['committed'])
        if failed_work != row['failed_paid_work_ms'] or row['policy_calls'] != len(record['decisions']):
            raise ValueError('Decision or failure-cost totals differ')
        actor = JointMaskTablePolicy.from_dict(read(ref/f'training_view/{arm}_policy.json')) if arm in ('full','nophase') else None
        if arm in guarded_actors:
            actor=guarded_actors[arm]
            parent_calls=sum(policies[arm].uses_parent(phase_aware_state_from_dict(d['state'])) for d in record['decisions'])
            if row['guard_parent_calls']!=parent_calls or row['guard_child_calls']!=len(record['decisions'])-parent_calls:
                raise ValueError('Guard source selection counts differ')
        matches = 0; disagreements = []
        for i, d in enumerate(record['decisions']):
            state = phase_aware_state_from_dict(d['state'])
            if policies[arm].select(state).encoded != d['action']:
                raise ValueError('Executed action differs from frozen policy')
            if d['action'] & state.current_action != state.current_action:
                raise ValueError('Executed action removes existing protection')
            if actor:
                original = actor.select(state).encoded
                matches += original == d['action']
                if original != d['action']:
                    if arm in guarded_actors:raise ValueError('Guarded exact representation differs from selected actor')
                    projected = policy_observation(state, actor.phase_awareness,
                        remove_phase_index_proxy=actor.remove_phase_index_proxy)
                    if actor.state_projection:
                        projected = project_table_state(projected,
                            preserve_growth=actor.growth_conditioned_groups)
                    context = policy_group_key(projected,
                        growth_conditioned=actor.growth_conditioned_groups,
                        local_context=actor.local_context_groups)
                    if actor.state_projection:
                        context = projected_context_key(context, projected)
                    table = policies[arm]
                    disagreements.append(dict(index=i, actor=original, table=d['action'],
                        training_actor_context_known=context in actor.groups,
                        compiled_table_context_known=(projected.phase, int(projected.current_action), context)
                            in getattr(table,'_by_context',{}),
                        context=context, state=d['state']))
        if actor and (matches != row['actor_table_matches'] or len(record['decisions']) != row['actor_table_queries']):
            raise ValueError('Actor/table fidelity differs')
        for e in record['first_hot_access_events']:
            if e['selected'] != e['previous'] | 5:
                raise ValueError('First eligible hot read/write protection suppressed')
        checked.append(dict(arm=arm, commits=row['commits'], aborts=row['aborts'],
            tps=row['throughput_tps'], max_task_seconds=row['retry_inclusive_p99_ms']/1000.,
            failed_paid_work_ms=failed_work, actual_changes=changes, calls=row['policy_calls'],
            actor_table_queries=len(record['decisions']) if actor else None,
            actor_table_matches=matches if actor else None, disagreements=disagreements,
            first_hot_events=len(record['first_hot_access_events']), record_sha256=sha(folder/f'{arm}.json')))
        if update:
            checked[-1].update(short_commits=row['short_commits'],short_tasks=row['short_tasks'],
                short_attempts=row['short_attempts'],joint_wall_seconds=row['joint_wall_seconds'])
    with (folder/'results.csv').open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    if len(csv_rows) != len(rows) or any(str(r[k]) != c[k] for r,c in zip(rows,csv_rows) for k in r if r[k] is not None):
        raise ValueError('CSV differs from JSON')
    return dict(passed=True, arms=checked, logical_tasks=sum(r['tasks'] for r in rows),
        histories=len(rows), no_target_training_or_retuning=True,
        durable_storage=plan.get('durable_storage',False),
        raw_persistent_participants_verified=plan.get('durable_storage',False),
        constructed_shared_updates=bool(update),
        scope='Previously inspected captured intents; optional declared constructed updates; no new API calls; frozen replay, not live replanning or stable P99')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = audit(args.output)
    (args.output/'independent_audit.json').write_text(json.dumps(result, indent=2))
    (args.output/'auditor_executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
