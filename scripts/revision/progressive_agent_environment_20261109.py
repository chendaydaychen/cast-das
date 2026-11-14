"""Frozen-model mechanism pilots, real runtime, synthetic reasoning, no API.

This is a separately named workload, not a rerun of the 132-point curves.
DWA means observed write-lock admission timing; publication stays atomic.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agent.runtime import AgentTransactionManager
from agent.runtime.context import LockAction,LockClass,TransactionPhase,TransactionStatus
from agent.runtime.observed_boundary_hooks import ObservedBoundaryHooks
from agent.runtime.quantized_mask_table import QuantizedSharedBitTable

CAPTURE=ROOT/'revision_experiments/real_credit_live_small_20261102.json'
OBSERVED_LLM={phase:[float(e['duration_ms']) for task in json.loads(CAPTURE.read_text())['tasks']
    for e in task['events'] if e['event']=='llm' and e.get('phase')==phase] for phase in ['explore','refine']}


def percentile(xs,q):
    if not xs:return None
    xs=sorted(xs);return xs[max(0,math.ceil(q*len(xs))-1)]


def plan_task(index,seed,ratio,zipf,early):
    rng=random.Random(seed+index)
    group=rng.choices(range(8),weights=[1/(i+1)**zipf for i in range(8)])[0]
    discovery=rng.sample([f'discovery:{i}' for i in range(64)],rng.randint(4,12))
    target=f'g{group}:k0'
    return dict(index=index,keys=discovery+[target],discovery=discovery,target=target,writes=[target],early=False,
        explore_rounds=rng.randint(1,3),explore_multiplier=rng.choice(OBSERVED_LLM['explore'])/3161.,
        refine_multiplier=rng.choice(OBSERVED_LLM['refine'])/2933.)


class MeterPolicy:
    def __init__(self,base,static=False,sample=False):
        self.base=base;self.static=static;self.lock=threading.Lock();self.calls=[];self.questions=[]
        self.phase_awareness='disabled' if static else getattr(base,'phase_awareness',getattr(base,'metadata',{}).get('phase_awareness','disabled'))
        self.sample=sample;self.local=threading.local()
    def select(self,state):
        start=time.perf_counter_ns();cpu=time.thread_time_ns()
        probabilities=()
        uniform=None
        if self.sample:
            if getattr(self.base,'supports_external_uniform',False):
                uniform=self.local.rng.random();action,probabilities=self.base.select_with_distribution(state,uniform=uniform)
            else:action,probabilities=self.base.select_with_distribution(state)
        else:action=LockAction(LockClass.HOT_WRITE) if self.static else self.base.select(state)
        with self.lock:
            self.calls.append(dict(wall_ns=time.perf_counter_ns()-start,cpu_ns=time.thread_time_ns()-cpu))
            self.questions.append(dict(state=dataclasses.asdict(state),action=action.encoded,observed_monotonic_s=time.monotonic(),
                task=getattr(self.local,'task',None),attempt=getattr(self.local,'attempt',None),
                uniform_draw=uniform,
                behavior_action_probabilities=probabilities,behavior_probability=probabilities[action.encoded] if probabilities else 1.))
        return action


def run_cell(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    model=args.rollout_actor or args.training/('nophase_int16_table.json' if args.arm=='nophase' else 'full_int16_table.json')
    if args.rollout_actor:
        from agent.runtime.shared_bit_mask_policy import SharedBitMaskActor
        base=SharedBitMaskActor.from_dict(json.loads(model.read_text()),seed=args.seed)
    else:
        payload=json.loads(model.read_text())
        if payload.get('artifact_type')=='cast-das-paper-observed-medoid-policy':
            from agent.runtime.paper_observation_policy import PaperObservationPolicy
            base=PaperObservationPolicy.from_dict(payload)
        else:base=QuantizedSharedBitTable(payload)
        if args.sample_table:
            from agent.runtime.sampled_quantized_policy_20261109 import SampledQuantizedPolicy
            base=SampledQuantizedPolicy(base,args.seed)
    if args.phase_diagnostic:
        if args.arm!='full' or args.rollout_actor or args.sample_table:raise ValueError('Diagnostic is a separate hand-coded control')
        class PhaseControl:
            metadata=base.metadata
            phase_awareness='enabled'
            def select(self,state):return LockAction.from_encoded(int(state.current_action)|(12 if state.phase=='refine' else 0))
        base=PhaseControl()
    policy=MeterPolicy(base,args.arm=='static',sample=bool(args.rollout_actor or args.sample_table))
    manager=AgentTransactionManager(paper_policy=policy,record_traces=False,collect_trajectories=True,
        performance_guards_enabled=False,low_conflict_occ_guard=False,
        delayed_write_apply_enabled=True,priority_enabled=args.arm!='no_priority')
    from agent.runtime.paper_contract import configure_paper_observed_contract
    from agent.runtime.eager_pcc_store import install_eager_pcc_control
    configure_paper_observed_contract(manager,explicit_phase_label_ablation=True)
    manager.track_inflight_agent_cost=not args.disable_inflight
    from agent.workloads.frozen_credit_agent import tracked_agent_wait
    from agent.runtime.single_writer_grace_20261109 import install as install_grace
    grace=install_grace(manager,args.wound_grace_ms)
    physical=install_eager_pcc_control(manager,eager=args.arm=='no_dwa',timeout_s=5.)
    write_acquired={};write_tenures=[]
    original_note=manager.atcc_locks._note_acquired;original_release=manager.atcc_locks.release_all
    def note(context):
        original_note(context)
        now=time.monotonic_ns()
        for key in context.held_write_locks:write_acquired.setdefault((context.tid,key),now)
    def release(context):
        with manager.atcc_locks._condition:
            now=time.monotonic_ns()
            for key in tuple(context.held_write_locks):
                began=write_acquired.pop((context.tid,key),None)
                if began is not None:write_tenures.append(dict(tid=context.tid,key=key,held_ms=(now-began)/1e6))
            return original_release(context)
    manager.atcc_locks._note_acquired=note;manager.atcc_locks.release_all=release
    if args.arm=='static':manager.paper_static_initial_mask=4
    for group in range(8):manager.register_object(f'g{group}:k0','0')
    for i in range(64):manager.register_object(f'discovery:{i}','0')
    # A common, recorded historical update for every registered object. No
    # future footprint, hot classification or favorable failure is installed.
    history=[]
    for key in [f'g{g}:k0' for g in range(8)]+[f'discovery:{i}' for i in range(64)]:
        warm=manager.begin('history:'+key,strategy='silo');warm.write(key,'0')
        if not warm.commit('silo').committed:raise ValueError('Common history write failed')
        history.append(dict(key=key,version=manager.store.get(key).version))
    from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
    from agent.runtime.issued_access_checkpoint import IssuedAccessCheckpointHooks
    reservation=ObservedWriteReservationHooks(manager.interceptor.hooks)
    for event in history:reservation.observed_writes[event['key']]=True
    manager.interceptor.hooks=ObservedBoundaryHooks(reservation)
    if write_acquired:raise ValueError('Historical writes leaked ownership')
    write_tenures.clear();physical.events.clear();policy.calls.clear();policy.questions.clear()
    lock=threading.Lock();changes=[];original=manager.transition_atcc_action
    def transition(txn,action,**kw):
        before=int(txn.context.action.protected);start=time.perf_counter_ns()
        try:return original(txn,action,**kw)
        finally:
            after=int(txn.context.action.protected)
            with lock:changes.append(dict(tid=txn.context.tid,before=before,after=after,wall_ns=time.perf_counter_ns()-start,actual=before!=after,observed_monotonic_s=time.monotonic()))
    manager.transition_atcc_action=transition
    strategy='silo' if args.arm=='silo' else 'paper-atcc'
    start=time.monotonic();rows=[];attempts=[];submitted=[];managers=[]
    def task(spec,arrival,cohort,delay):
        began=time.monotonic();ok=False;wasted=0.;detail=[]
        policy.local.rng=random.Random(args.seed+1009*spec['index'])
        for attempt in range(args.max_attempts):
            policy.local.task=spec['index'];policy.local.attempt=attempt
            tx=manager.begin(f'task:{spec["index"]}',dict(retry_count=attempt,access_set_visibility='online_observed'),strategy=strategy)
            if args.arm=='static' and int(tx.context.action.protected)!=4:raise AssertionError('Static mask missing')
            first_write=None;reason='';sleep_paid=0.;held_write_after_issue=0
            try:
                tx.enter_phase(TransactionPhase.EXPLORE)
                for round_index in range(spec['explore_rounds']):
                    for key in spec['discovery'][round_index::spec['explore_rounds']]:tx.read(key)
                    tick=time.monotonic()
                    with tracked_agent_wait(tx):time.sleep(delay[0]*spec['explore_multiplier']/spec['explore_rounds'])
                    sleep_paid+=time.monotonic()-tick
                    manager.interceptor.agent_round_finished(tx)
                tx.enter_phase(TransactionPhase.REFINE)
                value=int(tx.read(spec['target']).value)
                def writes():
                    nonlocal first_write,held_write_after_issue
                    for key in spec['writes']:
                        if first_write is None:first_write=time.monotonic()
                        tx.write(key,str(int(tx.read(key).value)+1))
                        held_write_after_issue=max(held_write_after_issue,len(tx.context.held_write_locks))
                tick=time.monotonic()
                with tracked_agent_wait(tx):time.sleep(delay[1]*spec['refine_multiplier'])
                sleep_paid+=time.monotonic()-tick
                manager.interceptor.agent_round_finished(tx)
                tx.enter_phase(TransactionPhase.COMMIT)
                first_write=time.monotonic();tx.write(spec['target'],str(value+1))
                held_write_after_issue=len(tx.context.held_write_locks)
                result=tx.commit(strategy);ok=bool(result.committed);reason=str(result.reason)
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc)
                if tx.context.status != TransactionStatus.COMMITTED:tx.abort_or_observe_wound(reason,strategy=strategy)
            ended=time.monotonic()
            detail.append(dict(task=spec['index'],cohort=cohort,attempt=attempt,committed=ok,reason=reason,
                reasoning_s=sleep_paid,blocked_ms=tx.context.blocked_time_ms,priority=tx.context.priority,
                lock_hold_ms=tx.context.lock_hold_time_ms,
                deferred_write_admission=tx.metadata.get('_defer_policy_write_locks',False),
                write_locks_after_issue=held_write_after_issue,write_to_finish_s=ended-first_write if first_write else 0,
                manager_identity=id(manager),first_write_offset_s=first_write-start if first_write else None))
            if ok:break
            wasted+=sleep_paid
        row=dict(task=spec['index'],cohort=cohort,arrival_s=arrival,started_s=began-start,finished_s=ended-start,
            committed=ok,retry_inclusive_s=ended-start-arrival,queue_s=began-start-arrival,
            attempts=len(detail),wasted_reasoning_s=wasted,writes=spec['writes'] if ok else [],manager_identity=id(manager))
        with lock:rows.append(row);attempts.extend(detail)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        count=args.tasks if args.mode!='continuous' else math.ceil(args.period*3*args.arrival_rate)
        for index in range(count):
            arrival=index/args.arrival_rate if args.mode=='continuous' else 0.
            cohort=min(2,int(arrival//args.period)) if args.mode=='continuous' else 0
            ratio=args.write_ratio;zipf=args.zipf;delay=(args.explore_ms/1000,args.refine_ms/1000)
            if args.mode=='continuous':
                if args.factor=='reasoning':delay=tuple(x*[1.,2.,1.][cohort] for x in delay)
                elif args.factor=='zipf':zipf=[.5,.99,.7][cohort]
                else:ratio=[.05,.5,.8][cohort]
                time.sleep(max(0,start+arrival-time.monotonic()))
            spec=plan_task(index,args.seed,ratio,zipf,args.position=='early')
            submitted.append(dict(task=index,arrival_s=arrival,cohort=cohort,ratio=ratio,zipf=zipf,delay_s=delay,**spec))
            managers.append(pool.submit(task,spec,arrival,cohort,delay))
        for future in managers:future.result()
    elapsed=time.monotonic()-start
    expected={f'g{g}:k0':0 for g in range(8)}
    expected.update({f'discovery:{i}':0 for i in range(64)})
    for row in rows:
        for key in row['writes']:expected[key]+=1
    wrong={k:dict(expected=v,actual=manager.store.get(k).value) for k,v in expected.items() if int(manager.store.get(k).value)!=v}
    protocol=dict(vars(args),monotonic_start_s=start,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_api_calls=0,retrained=False,deferred_replay=False,
        workload='Constructed progressive-access sensitivity scenario: variable Explore length/rounds, late mutable RMW read, short dependent Refine; not original YCSB/TPCC or new real-LLM capture',
        common_write_history=history,reservation_source='Actual common pre-measurement write APIs; all catalog keys, no future footprint',
        access_checkpoint='Before currently issued reads/writes; only existing state supplied to policy',
        deployment_mode='sampled-compiled-probability-law' if args.sample_table else 'deterministic-argmax',
        diagnostic_phase_policy=args.phase_diagnostic,active_mutable_rows=8,discovery_rows=64,
        workload_revision='Progressive agent: catalog exploration then mutable dependent read/reason/update; separate from standard YCSB/TPCC',
        timing_source_sha256=hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
        empirical_llm_observations={p:len(v) for p,v in OBSERVED_LLM.items()},
        policy_checkpoints='First currently issued operation in phase and completed observed rounds; shared schedule for Full/NoPhase',
        physical_dwa=True,physical_adapter_contract=dict(manager.paper_observed_contract),
        dwa_definition='Paired logical-version adapter in both arms; NoDWA eagerly installs protected point writes with undo and blocks foreign readers; Full retains private writes. Volatile existing-point-object scope only.',
        preserved_across_periods=['manager','store','policy','hotness_history'],unexpected_errors=[a['reason'] for a in attempts if a['reason'].startswith(('AttributeError','TypeError','NameError','AssertionError','ValueError','RuntimeError'))])
    commits=[r for r in rows if r['committed']]
    result=dict(arm=args.arm,mode=args.mode,position=args.position,write_ratio=args.write_ratio,factor=args.factor,
        submitted=len(rows),committed=len(commits),completion_rate=len(commits)/len(rows),tps=len(commits)/elapsed,
        p99_s=percentile([r['retry_inclusive_s'] for r in commits],.99),aborts=sum(not a['committed'] for a in attempts),
        policy_calls=len(policy.calls),actual_changes=sum(c['actual'] for c in changes),
        lookup_mean_us=sum(c['wall_ns'] for c in policy.calls)/max(1,len(policy.calls))/1000,
        blocked_ms=sum(a['blocked_ms'] for a in attempts),writes_locked_after_issue=sum(a['write_locks_after_issue']>0 for a in attempts),
        max_priority=max(a['priority'] for a in attempts),value_audit_passed=not wrong,value_errors=wrong,elapsed_s=elapsed,
        write_lock_key_ms=sum(w['held_ms'] for w in write_tenures),write_lock_tenures=len(write_tenures),
        mean_write_lock_tenure_ms=sum(w['held_ms'] for w in write_tenures)/max(1,len(write_tenures)),
        p99_write_lock_tenure_ms=percentile([w['held_ms'] for w in write_tenures],.99),unreleased_write_tenures=len(write_acquired),
        physical_early_installs=sum(e['kind']=='early-install' for e in physical.events),
        physical_undo_events=sum(e['kind']=='undo-before-unlock' for e in physical.events),
        physical_pending_at_end=len(physical._pending),physical_poisoned=physical.poisoned,
        wound_grace_ms=args.wound_grace_ms,wound_grace_deferrals=grace['deferrals'],wound_grace_leaked=len(grace['started']),
        track_inflight_agent_cost=manager.track_inflight_agent_cost,
        unique_manager_instances=len({r['manager_identity'] for r in rows}),cohorts=[])
    for cohort in range(3 if args.mode=='continuous' else 1):
        group=[r for r in rows if r['cohort']==cohort];success=[r for r in group if r['committed']]
        result['cohorts'].append(dict(cohort=cohort,submitted=len(group),completed=len(success),
            arrival_cohort_p99_s=percentile([r['retry_inclusive_s'] for r in success],.99)))
    for name,value in [('protocol',protocol),('result',result),('tasks',rows),('attempts',attempts),('decisions',dict(decisions=policy.questions)),('changes',changes),('requests',submitted),('write_tenures',write_tenures),('physical_events',physical.events),('reservations',reservation.reservation_events)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2,default=str))
    with (out/'time_windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['start_s','end_s','completed','commit_window_tps','policy_calls','actual_changes']);writer.writeheader()
        for tick in range(math.ceil(elapsed/5)):
            n=sum(r['committed'] and tick*5<=r['finished_s']<(tick+1)*5 for r in rows)
            writer.writerow(dict(start_s=tick*5,end_s=(tick+1)*5,completed=n,commit_window_tps=n/5,
                policy_calls=sum(tick*5<=q['observed_monotonic_s']-start<(tick+1)*5 for q in policy.questions),
                actual_changes=sum(c['actual'] and tick*5<=c['observed_monotonic_s']-start<(tick+1)*5 for c in changes)))
    print(json.dumps(result),flush=True)
    if wrong or protocol['unexpected_errors'] or write_acquired or physical._pending or physical.poisoned or grace['started']:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--arm',choices=['full','nophase','no_dwa','no_priority','static','silo'],default='full');p.add_argument('--rollout-actor',type=Path);p.add_argument('--sample-table',action='store_true')
    p.add_argument('--mode',choices=['ablation','dwa','continuous'],default='ablation')
    p.add_argument('--position',choices=['early','late'],default='early');p.add_argument('--write-ratio',type=float,default=.2)
    p.add_argument('--zipf',type=float,default=.99);p.add_argument('--tasks',type=int,default=32);p.add_argument('--workers',type=int,default=16);p.add_argument('--max-attempts',type=int,default=8)
    p.add_argument('--explore-ms',type=float,default=3161);p.add_argument('--refine-ms',type=float,default=2933);p.add_argument('--phase-diagnostic',action='store_true');p.add_argument('--wound-grace-ms',type=float,default=4500.);p.add_argument('--disable-inflight',action='store_true')
    p.add_argument('--period',type=float,default=60);p.add_argument('--arrival-rate',type=float,default=2)
    p.add_argument('--factor',choices=['reasoning','zipf','rw'],default='reasoning');p.add_argument('--seed',type=int,default=20261109)
    a=p.parse_args();run_cell(a)


if __name__=='__main__':main()
