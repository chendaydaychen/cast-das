"""Unretuned pipeline policies on previously captured real CreditCrew intents.

All measured provider/tool delays retained. No new capture, API request,
target training or target threshold selection. Optional explicitly declared
shared-item updates are constructed events, not captured model actions.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.runtime.paper_policy import HotRulePolicy
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
from agent.runtime.feature_score_table import FeatureScoreTable
from agent.runtime.paper_policy import CompiledPhasePolicy
from agent.runtime.joint_mask_policy import JointMaskTablePolicy,load_scoring_policy
from agent.runtime.state_collector import phase_aware_state_from_dict
from scripts.revision.compare_frozen_credit_trace import run_system,FixedProtectionPolicy,FixedReadProtectionPolicy
from scripts.revision.evaluate_phase_pipeline_frozen import load_models
from scripts.revision.train_phase_pipeline_pilot import ObservedAccessHooks
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.run_audited_credit_pressure import AuditedManager
from scripts.revision.audit_serial_history import verify_history
from scripts.revision.run_current_credit_ablations import read,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','capture-reference','credit-training','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arms', nargs='+', choices=('full','nophase','linear_parent','linear_fitted','static','static_read','hot_always','item_threshold3','credit_full','silo'))
    p.add_argument('--workers', type=int, default=2, choices=(2,4,8))
    p.add_argument('--repeat-factor', type=int, default=1, choices=(1,2,4))
    p.add_argument('--shared-update-key')
    p.add_argument('--shared-update-arrivals', type=float, nargs='+')
    p.add_argument('--shared-update-deadline-seconds',type=float,default=40.)
    p.add_argument('--item-window-ms',type=float,default=1000.)
    p.add_argument('--item-window-calibration',type=Path)
    p.add_argument('--count-external-frequency',action='store_true')
    p.add_argument('--exact-feature-tables', type=Path,
        help='Explicit alternative contribution lookup deployment; not the original medoid action table')
    p.add_argument('--continuation-cost-guard',type=Path,help='Opt-in separately labelled training-cost fallback to parent actors')
    p.add_argument('--durable-storage',action='store_true',help='Opt-in serialized SQLite + atomic-file coordinator; not native-memory performance')
    p.add_argument('--disable-delay',action='store_true',help='Frozen-policy mechanism ablation: acquire selected write protection without deferred apply')
    p.add_argument('--disable-priority',action='store_true',help='Frozen-policy mechanism ablation: disable cost priority')
    p.add_argument('--carry-short-retry-wait',action='store_true',help='Opt-in fairness follow-up: preserve actual short-update lock wait across retries')
    p.add_argument('--retry-wait-quantum-ms',type=float,default=100.,help='Declared prior-wait priority quantum; 10ms equals the existing agent-work quantum')
    args=p.parse_args();reference=read(args.reference/'protocol.json')
    if args.carry_short_retry_wait and (not args.shared_update_key or args.disable_priority):
        p.error('Wait carry requires shared updates and enabled priority')
    if args.continuation_cost_guard and args.exact_feature_tables:
        p.error('Cost guard already declares its own exact contribution tables')
    import math
    if not math.isfinite(args.retry_wait_quantum_ms) or args.retry_wait_quantum_ms<=0:
        p.error('Positive finite prior-wait priority quantum required')
    if not math.isfinite(args.shared_update_deadline_seconds) or args.shared_update_deadline_seconds<=0:
        p.error('Positive finite short-update retry horizon required')
    if not math.isfinite(args.item_window_ms) or args.item_window_ms<=0:
        p.error('Positive finite frequency window required')
    if bool(args.shared_update_key) != bool(args.shared_update_arrivals):
        p.error('Shared update requires both key and arrivals')
    if args.shared_update_arrivals and (args.durable_storage or
            any(not math.isfinite(v) or v<0 for v in args.shared_update_arrivals) or
            sorted(set(args.shared_update_arrivals))!=args.shared_update_arrivals):
        p.error('Unique increasing finite arrivals and native storage required')
    for name,digest in reference['artifact_sha256'].items():
        if sha(args.reference/name)!=digest:raise ValueError('Frozen model changed')
    source=read(args.capture_reference/'input.json')
    if args.item_window_calibration:
        from scripts.revision.credit_frequency_window import training_window
        calibration=read(args.item_window_calibration)
        training_path=args.item_window_calibration.parent/'training_input.json'
        train=read(training_path)
        if (sha(training_path)!=calibration['training_input_sha256'] or
                training_window(train)[0]!=args.item_window_ms or
                {t['task']['company_id'] for t in train['tasks']} & {t['task']['company_id'] for t in source['tasks']}):
            raise ValueError('Training-derived frequency calibration differs or overlaps evaluation')
    if not source.get('agent_directed') or not source.get('multi_storage') or len(source['tasks']) not in (2,4):
        raise ValueError('Expected two or four actual captured multi-storage agent intents')
    mix_path=args.capture_reference/'design.json'
    mix_proof=None
    if mix_path.exists():
        from scripts.revision.merge_credit_evaluation_captures import merge
        mix_proof=read(mix_path)
        originals=[]
        for item in mix_proof['sources']:
            path=args.capture_reference/item['copied_filename']
            if sha(path)!=item['sha256']:raise ValueError('Mixed source changed')
            originals.append(read(path))
        rebuilt,origins=merge(originals,set(mix_proof['forbidden_companies']))
        if rebuilt!=source or origins!=mix_proof['origins'] or sha(args.capture_reference/'input.json')!=mix_proof['input_sha256']:
            raise ValueError('Mixed input is not the verbatim source union')
    if args.shared_update_key:
        common=set.intersection(*[{e['object_id'] for e in t['events'] if e['event']=='read'} for t in source['tasks']])
        written={e['object_id'] for t in source['tasks'] for e in t['events'] if e['event']=='write'}
        if args.shared_update_key not in common or args.shared_update_key in written:
            raise ValueError('Shared evidence update requires a common read-only foreground item')
    runtime=reference['runtime'];models=load_models(args.reference)
    models=dict(full=models['full_table'],nophase=models['nophase_table'],linear_parent=models['linear_parent'],
        linear_fitted=models['linear'],static=FixedProtectionPolicy(),static_read=FixedReadProtectionPolicy(),
        hot_always=HotRulePolicy(), item_threshold3=CompiledPhasePolicy([]),
        credit_full=load_scoring_policy(read(args.credit_training/'full_policy.json')),silo=None)
    feature_sha = {}
    guarded_actors={};guard_metadata=None
    if args.continuation_cost_guard:
        from scripts.revision.build_continuation_cost_guard import load_guarded
        guarded,guarded_actors,guard_metadata=load_guarded(args.reference,args.continuation_cost_guard)
        models.update(guarded)
    if args.exact_feature_tables:
        for arm in ('full','nophase'):
            path = args.exact_feature_tables/f'{arm}_feature_table.json'
            artifact = read(path)
            if FeatureScoreTable.compile(read(args.reference/f'training_view/{arm}_policy.json')).to_dict() != artifact:
                raise ValueError('Feature table differs from frozen original actor')
            models[arm] = FeatureScoreTable(artifact); feature_sha[str(path)] = sha(path)
    if args.arms:
        if len(set(args.arms)) != len(args.arms):raise ValueError('Duplicate arm')
        models={arm:models[arm] for arm in args.arms}
    else:
        models.pop('item_threshold3')
        models.pop('linear_fitted')
        models.pop('static')
        models.pop('static_read')
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(scope='Exploratory frozen cross-task-family transfer to previously inspected real CreditCrew capture',
        reference=str(args.reference),reference_artifact_sha256=reference['artifact_sha256'],
        capture_reference=str(args.capture_reference),capture_sha256=sha(args.capture_reference/'input.json'),
        credit_training=str(args.credit_training),credit_full_sha256=sha(args.credit_training/'full_policy.json'),
        workers=args.workers,repeat_factor=args.repeat_factor,time_scale=1.,max_attempts=6,arms=list(models),runtime=runtime,
        mechanism_settings=dict(delayed_write_apply_enabled=not args.disable_delay,priority_enabled=not args.disable_priority),
        carry_short_retry_wait=args.carry_short_retry_wait,
        retry_wait_quantum_ms=args.retry_wait_quantum_ms,
        exact_feature_tables=str(args.exact_feature_tables) if args.exact_feature_tables else None,
        feature_table_sha256=feature_sha,
        continuation_cost_guard=str(args.continuation_cost_guard) if args.continuation_cost_guard else None,
        guard_config_sha256=sha(args.continuation_cost_guard/'guard.json') if args.continuation_cost_guard else None,
        guard_metadata=guard_metadata,
        item_rule=dict(minimum_accesses=3, window_ms=args.item_window_ms,
            count_external_frequency=args.count_external_frequency,
            calibration=str(args.item_window_calibration) if args.item_window_calibration else None,
            calibration_sha256=sha(args.item_window_calibration) if args.item_window_calibration else None,
            scope='Per-item local frequency; not the global contention-threshold rule'),
        replay_templates=len(source['tasks']), logical_tasks=len(source['tasks'])*args.repeat_factor,
        repeated_templates_are_not_new_independent_captures=True,
        retrained=False,retuned=False,new_api_calls=0,short_updates_injected=bool(args.shared_update_key),
        mixed_design_sha256=sha(mix_path) if mix_proof else None,
        shared_update=dict(key=args.shared_update_key,arrivals_seconds=args.shared_update_arrivals,
            deadline_seconds=args.shared_update_deadline_seconds,max_attempts=4001,backoff_seconds=.01,
            operation='Read decimal integer then increment by one',
            policy='Action-0 OCC, same commit admission; constructed event, not captured LLM action') if args.shared_update_key else None,
        shared_runtime='Pipeline runtime observations/admission and observed-access hook shared by learned arms; not old Credit deployment',
        caveat='Captured intents replayed, not live replanning or new independent capture; no stable P99',
        durable_storage=args.durable_storage,
        storage_scope='Serialized durable SQLite + atomic-file KV' if args.durable_storage else 'Native in-memory two-Dbx coordinator',
        source_sha256=sha(Path(__file__)))
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'input.json').write_bytes((args.capture_reference/'input.json').read_bytes())
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    snapshot = args.output/'shared_sources'; snapshot.mkdir()
    names = ('scripts/revision/compare_frozen_credit_trace.py',
             'agent/workloads/frozen_credit_agent.py', 'agent/runtime/feature_score_table.py',
             'agent/runtime/hot_access_hooks.py', 'agent/runtime/item_frequency_hooks.py')
    if args.carry_short_retry_wait:
        names+=('agent/runtime/context.py','agent/runtime/priority.py','agent/runtime/transaction.py')
    if args.shared_update_key:
        names+=('scripts/revision/run_frozen_credit_short_updates.py',)
    if args.count_external_frequency:
        names+=('scripts/revision/credit_frequency_window.py',)
    if args.continuation_cost_guard:
        names+=('agent/runtime/continuation_cost_guard.py','scripts/revision/build_continuation_cost_guard.py')
    plan['shared_source_sha256'] = {name:sha(ROOT/name) for name in names}
    for name in names:
        destination = snapshot/name; destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(stage='declared',companies=[t['task']['company_id'] for t in source['tasks']],
        arms=list(models),model_calls_in_input=sum(len(t['llm_calls']) for t in source['tasks']))),flush=True)
    rows=[];proof=[]
    for arm,policy in models.items():
        holder=[]
        def factory(**kwargs):
            if args.durable_storage:
                from agent.runtime.durable_storage import DurableRoutedVersionedKVStore
                kwargs.pop('storage_engines',None)
                kwargs['store']=DurableRoutedVersionedKVStore(args.output/('durable_'+arm))
            if args.shared_update_key:
                from scripts.revision.run_frozen_credit_short_updates import MixedManager
                manager=MixedManager(**kwargs,arrival_seconds=args.shared_update_arrivals,
                    update_key=args.shared_update_key,deadline_seconds=args.shared_update_deadline_seconds)
            else:manager=AuditedManager(**kwargs)
            configure_frozen_runtime(manager,runtime)
            if args.carry_short_retry_wait:
                from dataclasses import replace
                manager.carry_short_retry_wait=True
                manager.priority_manager.config=replace(manager.priority_manager.config,include_prior_retry_wait=True,
                    prior_retry_wait_quantum_ms=args.retry_wait_quantum_ms)
            if arm=='static':manager.paper_static_initial_mask=4
            if arm=='static_read':manager.paper_static_initial_mask=3
            if arm=='hot_always':
                manager.first_hot_access_events=[];manager.interceptor.hooks=FirstHotAccessHooks(manager)
            elif arm=='item_threshold3':
                manager.interceptor.hooks=ItemFrequencyHooks(manager,minimum_accesses=3,window_ms=args.item_window_ms)
            elif arm!='silo':manager.interceptor.hooks=ObservedAccessHooks(manager)
            manager.foreground_hooks=manager.interceptor.hooks
            if args.shared_update_key:
                from scripts.revision.run_frozen_credit_short_updates import ShortUpdateHooks
                if arm=='item_threshold3' and args.count_external_frequency:
                    from scripts.revision.credit_frequency_window import CountedExternalHooks
                    manager.interceptor.hooks=CountedExternalHooks(manager.interceptor.hooks)
                else:manager.interceptor.hooks=ShortUpdateHooks(manager.interceptor.hooks)
            holder.append(manager);return manager
        attempts=[];decisions=[]
        row=run_system(source,'silo' if arm=='silo' else 'paper-atcc',workers=args.workers,max_attempts=6,
            repeat_factor=args.repeat_factor,time_scale=1.,policy=policy,manager_factory=factory,
            performance_guards=False,policy_commit_batch=False,
            delayed_write_apply_enabled=not args.disable_delay,priority_enabled=not args.disable_priority,
            policy_selected_write_admission_only=runtime['policy_selected_write_admission_only'],
            before_reasoning_checkpoint=True,carry_prior_restart_cost=runtime.get('carry_prior_restart_cost',False),
            frozen_agent_schedule=True,allow_complete_aborted_capture=True,attempt_sink=attempts,decision_sink=decisions)
        manager=holder[0]
        if args.shared_update_key:manager.join_short_updates()
        history=manager.history();oracle=verify_history(**history)
        if oracle['verdict']!='verified':raise ValueError('History failed')
        matches=0
        actor=JointMaskTablePolicy.from_dict(read(args.reference/f'training_view/{arm}_policy.json')) if arm in ('full','nophase') else None
        if arm in guarded_actors:actor=guarded_actors[arm]
        for d in decisions:
            state=phase_aware_state_from_dict(d['state'])
            if policy.select(state).encoded!=d['action']:raise ValueError('Frozen action mismatch')
            if actor:matches+=actor.select(state).encoded==d['action']
        details=[a for t in attempts for a in t['attempt_details']]
        row.update(arm=arm,policy_calls=len(decisions),actual_changes=sum(a['action_changes'] for a in details),
            failed_paid_work_ms=sum(a['paid_replay_work_ms'] for a in details if not a['committed']),
            actor_table_queries=len(decisions) if actor else None,actor_table_matches=matches if actor else None)
        if arm in guarded_actors:
            parent_calls=sum(policy.uses_parent(phase_aware_state_from_dict(d['state'])) for d in decisions)
            row.update(guard_parent_calls=parent_calls,guard_child_calls=len(decisions)-parent_calls)
        item_events=[dict(txn_id=txn.context.tid,at_s=event.at_s,detail=event.detail)
            for txn in manager.audit_transactions for event in txn.events if event.kind=='item-protection-change']
        row['item_protection_changes'] = len(item_events)
        background=[dict(history_id=f'attempt-{i}',txn_id=txn.context.tid)
            for i,txn in enumerate(manager.audit_transactions) if txn.metadata.get('external_short_update')]
        short=getattr(manager,'short_records',[])
        row.update(short_tasks=len(short),short_commits=sum(s['committed'] for s in short),
            short_attempts=sum(len(s['attempts']) for s in short),
            max_short_elapsed_seconds=max((s['elapsed_seconds'] for s in short),default=0.),
            joint_wall_seconds=max(row['wall_seconds'],max((s['completion_seconds'] for s in short),default=0.)))
        record=dict(summary=row,attempts=attempts,decisions=decisions,history=history,oracle=oracle,
            first_hot_access_events=getattr(manager,'first_hot_access_events',[]),
            item_decisions=getattr(manager.foreground_hooks,'decisions',[]) if arm=='item_threshold3' else [],
            item_protection_events=item_events,short_updates=short,background_history=background)
        if args.durable_storage:
            from agent.runtime.durable_storage import DurableRoutedVersionedKVStore
            expected={key:dict(value=manager.store.get(key).value,version=manager.store.get_version(key),
                storage=manager.store.storage_id_for(key)) for key in manager._catalog}
            manager.store.close()
            with DurableRoutedVersionedKVStore(args.output/('durable_'+arm)) as reopened:
                for key,value in expected.items():
                    if (reopened.get(key).value,reopened.get_version(key),reopened.storage_id_for(key))!=(value['value'],value['version'],value['storage']):
                        raise ValueError('Persisted CreditCrew data changed across restart')
                record['durable_reopen']=dict(passed=True,objects=len(expected),recovery=reopened.recovery_report)
        (args.output/f'{arm}.json').write_text(json.dumps(record,indent=2));rows.append(row)
        (args.output/'results.json').write_text(json.dumps(rows,indent=2))
        with (args.output/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        proof.append(dict(arm=arm,history='verified',queries=len(decisions),actions_reproduced=True))
        AuditedManager.instances.remove(manager)
        if AuditedManager.instances:raise ValueError('Manager retained')
        print(json.dumps(row),flush=True)
    (args.output/'execution_audit.json').write_text(json.dumps(dict(passed=True,checks=proof),indent=2))


if __name__=='__main__':main()
