"""Final-deployment DWA conditions and continuous ten-operation agent YCSB."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from agent.runtime.context import TransactionPhase,TransactionStatus
from agent.runtime.paper_contract import configure_paper_observed_contract
from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
from agent.runtime.eager_pcc_store import install_eager_pcc_control
from agent.runtime.single_writer_grace_20261109 import install as install_grace
from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
from agent.workloads.frozen_credit_agent import tracked_agent_wait
from scripts.revision.progressive_agent_environment_20261109 import ObservedBoundaryHooks,MeterPolicy,OBSERVED_LLM
from scripts.revision.run_revision_supplement_20261110 import FrozenStatic

def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2,default=str))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def p99(values):return sorted(values)[max(0,math.ceil(.99*len(values))-1)] if values else None

def make_plan(args):
    rng=random.Random(args.seed);count=args.tasks if args.mode=='dwa' else math.ceil(3*args.period*args.rate)
    kinds=list(range(count));rng.shuffle(kinds);writers=set(kinds[:round(count*args.writer_fraction)])
    result=[]
    for index in range(count):
        cohort=min(2,int((index/args.rate)//args.period)) if args.mode=='continuous' else 0
        exponent=[0.,.99,.5][cohort] if args.factor=='zipf' else 0.
        update_p=[.1,.5,.9][cohort] if args.factor=='rw' else .5
        multiplier=[1.,2.,1.][cohort] if args.factor=='reasoning' else 1.
        key=rng.choices(range(args.objects),weights=[1/(i+1)**exponent for i in range(args.objects)])[0]
        remaining=[i for i in range(args.objects) if i!=key]
        second_key=rng.choices(remaining,weights=[1/(i+1)**exponent for i in remaining])[0]
        explore=rng.choice(OBSERVED_LLM['explore'])*multiplier
        refine=rng.choice(OBSERVED_LLM['refine'])*multiplier
        ops=[dict(kind='read',key=f'catalog:{k}') for k in rng.sample(range(64),8)]
        ops += [dict(kind='write' if rng.random()<update_p else 'read',key=f'account:{k}') for k in [key,second_key]]
        result.append(dict(index=index,arrival_s=index/args.rate if args.mode=='continuous' else 0.,cohort=cohort,
            explore_ms=explore,refine_ms=refine,key=f'account:{key}',catalog=[o['key'] for o in ops[:8]],operations=ops,
            is_writer=(index in writers if args.mode=='dwa' else any(o['kind']=='write' for o in ops)),
            read_offset_s=rng.uniform(.1,1.8),exponent=exponent,eligible_write_probability=update_p))
        if args.mode=='dwa' and index in writers:
            result[-1]['key']=f'account:{(key//8)*8}'
            result[-1]['write_keys']=[f'account:{(key//8)*8+k}' for k in range(8)]
    return result

def run(args):
    args.output.mkdir(parents=True,exist_ok=False);requests=make_plan(args);base=FrozenStatic() if args.arm=='static' else QuantizedSharedBitTable(read(args.model))
    policy=MeterPolicy(base);manager=AgentTransactionManager(paper_policy=policy,record_traces=False,collect_trajectories=True,
        performance_guards_enabled=False,low_conflict_occ_guard=False,priority_enabled=True,delayed_write_apply_enabled=True)
    configure_paper_observed_contract(manager,explicit_phase_label_ablation=True)
    manager.priority_manager.config=dataclasses.replace(manager.priority_manager.config,blocked_quantum_ms=10.)
    manager.interceptor.hooks._action_transition_timeout_s=lambda tx:120.
    for name in ['wlock','validate_and_rlock','validate_and_wlock','acquire_write_set']:
        original=getattr(manager.atcc_locks,name)
        def lock_budget(*v,_original=original,**kw):kw.setdefault('timeout_s',120.);return _original(*v,**kw)
        setattr(manager.atcc_locks,name,lock_budget)
    physical=install_eager_pcc_control(manager,eager=args.mode=='dwa' and args.arm=='no_dwa',timeout_s=120.)
    grace=install_grace(manager,4500.,commit_grace_ms=50.)
    if args.mode!='dwa' and args.arm!='static':
        reservation=ObservedWriteReservationHooks(manager.interceptor.hooks)
        manager.interceptor.hooks=ObservedBoundaryHooks(reservation)
    else:reservation=None;manager.interceptor.hooks=ObservedBoundaryHooks(manager.interceptor.hooks)
    dependency_guard=None
    if getattr(args,'dependency_guard',False):
        if args.mode!='continuous':raise ValueError('DWA publication pair retains unchanged admission')
        from agent.runtime.observed_dependency_guard_20261110 import install
        dependency_guard=install(manager,reservation=reservation)
    manager.paper_deferred_replay_enabled=False;manager.atcc_retry_cache_enabled=False
    strategy='silo' if args.arm=='silo' else 'paper-atcc'
    initial={**{f'catalog:{k}':'0' for k in range(64)},**{f'account:{k}':'0' for k in range(args.objects)}}
    for key in sorted({o['key'] for r in requests for o in r['operations']}):initial.setdefault(key,'0')
    for key,value in initial.items():manager.register_object(key,value)
    for key in initial:
        tx=manager.begin('maintenance:'+key,strategy='silo');tx.write(key,'0');assert tx.commit('silo').committed
        if reservation is not None:reservation.observed_writes[key]=True
    initial_versions={key:manager.store.get(key).version for key in initial};physical.events.clear();policy.calls.clear();policy.questions.clear()
    commits=[];old_batch=physical.batch_put_if_version
    def audited_batch(checks,writes):
        with physical._mutex:
            ok=old_batch(checks,writes)
            if ok:commits.append(dict(tid=physical._local.context.tid,writes=list(writes),at_ns=time.monotonic_ns()))
            return ok
    physical.batch_put_if_version=audited_batch
    changes=[];transition=manager.transition_atcc_action
    def action(tx,value,**kw):
        before=int(tx.context.action.protected);tick=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:return transition(tx,value,**kw)
        finally:changes.append(dict(task=tx.task_id,actual=before!=int(tx.context.action.protected),at_s=time.monotonic(),wall_ns=time.perf_counter_ns()-tick,cpu_ns=time.thread_time_ns()-cpu))
    manager.transition_atcc_action=action
    rows=[];attempts=[];continuity=[];operations=[];guard=threading.Lock();inflight=peak=0;start=time.monotonic()
    def boundary(cohort):
        continuity.append(dict(cohort=cohort,at_s=time.monotonic()-start,manager=id(manager),store=id(manager.store),policy=id(manager.paper_policy.snapshot()),
            history=id(manager.hotness_tracker),dependency_history=id(dependency_guard['stats']) if dependency_guard else None,
            outcomes=manager.runtime_outcome_diagnostics()['recorded'],model_sha256=sha(args.model),
            values={f'account:{k}':manager.store.get(f'account:{k}').value for k in range(args.objects)}))
    def task(spec):
        nonlocal inflight,peak
        began=time.monotonic();ok=False;details=[]
        with guard:inflight+=1;peak=max(peak,inflight)
        for retry in range(args.attempts):
            policy.local.task=spec['index'];policy.local.attempt=retry
            if args.mode=='continuous':time.sleep(.2*spec['explore_ms']/1000)
            tx=manager.begin(str(spec['index']),dict(retry_count=retry,access_set_visibility='online_observed'),strategy=strategy)
            if args.arm=='static' and strategy=='paper-atcc':
                # The threshold is frozen from independent main-workload validation.
                if manager.paper_runtime_metrics()['conflict_abort_rate']>=args.static_threshold:
                    from agent.runtime.context import LockAction
                    tx.context.action=LockAction.from_encoded(4)
            paid=0.;reason='';issued=[];observed_values={}
            def wait(seconds):
                nonlocal paid
                tick=time.monotonic()
                if strategy=='paper-atcc':
                    with tracked_agent_wait(tx):time.sleep(seconds)
                else:time.sleep(seconds)
                paid+=time.monotonic()-tick
            def perform(op,index):
                if op['kind']=='read':
                    value=tx.read(op['key']).value;observed_values[op['key']]=value
                else:
                    value=str(spec['index']*10+index+1)
                    if index==9 and spec['key'] in observed_values:value=str(int(observed_values[spec['key']])+1)
                    tx.write(op['key'],value)
                issued.append(dict(task=spec['index'],attempt=retry,index=index,kind=op['kind'],key=op['key'],value=value,at_s=time.monotonic()-start))
            def dwa_read(key):
                value=tx.read(key).value
                issued.append(dict(task=spec['index'],attempt=retry,index=len(issued),kind='read',key=key,value=value,at_s=time.monotonic()-start))
                return value
            def dwa_write_group(value):
                for offset,key in enumerate(spec['write_keys']):
                    v=str(value+1 if offset==0 else (value+1)*8+offset);tx.write(key,v)
                    issued.append(dict(task=spec['index'],attempt=retry,index=len(issued),kind='write',key=key,value=v,at_s=time.monotonic()-start))
            try:
                tx.enter_phase(TransactionPhase.EXPLORE)
                if args.mode=='dwa':
                    if not spec['is_writer']:
                        wait(spec['read_offset_s']);dwa_read(spec['key']);wait(.05);manager.interceptor.agent_round_finished(tx)
                    else:
                        for key in spec['catalog'][:2]:dwa_read(key)
                        wait(.1);manager.interceptor.agent_round_finished(tx);tx.enter_phase(TransactionPhase.REFINE)
                        value=int(dwa_read(spec['key']))
                        if args.position=='early':dwa_write_group(value)
                        wait(spec['refine_ms']/1000);manager.interceptor.agent_round_finished(tx);tx.enter_phase(TransactionPhase.COMMIT)
                        if args.position=='late':dwa_write_group(value)
                else:
                    for index,op in enumerate(spec['operations']):
                        if index==8:tx.enter_phase(TransactionPhase.REFINE)
                        if index==9:tx.enter_phase(TransactionPhase.COMMIT)
                        perform(op,index)
                        if index==7:wait(.8*spec['explore_ms']/1000);manager.interceptor.agent_round_finished(tx)
                        if index==8:wait(spec['refine_ms']/1000);manager.interceptor.agent_round_finished(tx)
                tx.enter_phase(TransactionPhase.COMMIT);result=tx.commit(strategy);ok=bool(result.committed);reason=str(result.reason)
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc)
                if tx.context.status!=TransactionStatus.COMMITTED:tx.abort_or_observe_wound(reason,strategy=strategy)
            details.append(dict(task=spec['index'],attempt=retry,committed=ok,reason=reason,paid_s=paid,blocked_ms=tx.context.blocked_time_ms,
                operations=len(issued),reads=sum(o['kind']=='read' for o in issued),writes=sum(o['kind']=='write' for o in issued)))
            with guard:operations.extend(issued)
            if ok:break
        ended=time.monotonic()
        with guard:
            inflight-=1;attempts.extend(details);rows.append(dict(task=spec['index'],cohort=spec['cohort'],committed=ok,is_writer=spec['is_writer'],attempts=len(details),
                arrival_s=spec['arrival_s'],start_s=began-start,finish_s=ended-start,queue_s=began-start-spec['arrival_s'],
                service_s=ended-began,arrival_latency_s=ended-start-spec['arrival_s']))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[];previous=None
        for spec in requests:
            if args.mode=='continuous':time.sleep(max(0,start+spec['arrival_s']-time.monotonic()))
            if spec['cohort']!=previous:boundary(spec['cohort']);previous=spec['cohort']
            futures.append(pool.submit(task,spec))
        for f in futures:f.result()
    elapsed=time.monotonic()-start;boundary(3)
    expected=dict(initial);versions=dict(initial_versions)
    for batch in commits:
        for key,value in batch['writes']:expected[key]=value;versions[key]+=1
    wrong={key:dict(expected=value,actual=manager.store.get(key).value) for key,value in expected.items() if value!=manager.store.get(key).value or versions[key]!=manager.store.get(key).version}
    if args.mode=='dwa':
        increments={f'account:{k}':0 for k in range(0,args.objects,8)}
        for row in rows:
            if row['committed'] and row['is_writer']:increments[requests[row['task']]['key']]+=1
        assert all(int(manager.store.get(k).value)==v for k,v in increments.items())
    success=[r for r in rows if r['committed']];readers=[r for r in success if not r['is_writer']]
    result=dict(mode=args.mode,arm=args.arm,factor=args.factor,position=args.position,writer_fraction=args.writer_fraction,
        submitted=len(rows),completed=len(success),completion_rate=len(success)/len(rows),agent_tps=len(success)/elapsed,elapsed_s=elapsed,
        retry_service_p99_s=p99([r['service_s'] for r in success]),arrival_p99_s=p99([r['arrival_latency_s'] for r in success]),
        reader_service_mean_s=sum(r['service_s'] for r in readers)/max(1,len(readers)),reader_service_p99_s=p99([r['service_s'] for r in readers]),reader_count=len(readers),
        aborts=sum(not r['committed'] for r in attempts),policy_calls=len(policy.calls),actual_changes=sum(r['actual'] for r in changes),
        peak_inflight=peak,physical_early_installs=sum(e['kind']=='early-install' for e in physical.events),physical_pending=len(physical._pending),physical_poisoned=physical.poisoned,
        blocking_reader_wait_ms=sum(r['waiting_ms'] for r in getattr(getattr(manager,'blocking_occ_read_control',None),'records',[])),
        values_correct=not wrong,value_errors=wrong,unique_manager_ids=len({r['manager'] for r in continuity}),
        continuity_verified=all(len({r[k] for r in continuity})==1 for k in ['manager','store','policy','history','dependency_history','model_sha256']),
        dependency_guard_metrics=dependency_guard['metrics'] if dependency_guard else None,
        unexpected_errors=[r for r in attempts if r['reason'].startswith(('ValueError','RuntimeError','AttributeError','TypeError','NameError','AssertionError'))])
    windows=[]
    for tick in range(math.ceil(elapsed/5)):
        left=tick*5;right=left+5
        windows.append(dict(start_s=left,end_s=right,submitted=sum(left<=r['arrival_s']<right for r in requests),
            completed=sum(r['committed'] and left<=r['finish_s']<right for r in rows),commit_tps=sum(r['committed'] and left<=r['finish_s']<right for r in rows)/5,
            queued_at_end=sum(r['arrival_s']<right for r in requests)-sum(r['start_s']<right for r in rows),
            running_at_end=sum(r['start_s']<right<=r['finish_s'] for r in rows),
            actual_changes=sum(c['actual'] and left<=c['at_s']-start<right for c in changes)))
    cohorts=[]
    for cohort in range(3 if args.mode=='continuous' else 1):
        group=[r for r in rows if r['cohort']==cohort];done=[r for r in group if r['committed']];plan=[r for r in requests if r['cohort']==cohort]
        target_ops=[op for s in plan for op in s['operations']]
        planned_count=len(target_ops) if args.mode=='continuous' else sum(11 if s['is_writer'] else 1 for s in plan)
        planned_writes=sum(o['kind']=='write' for o in target_ops) if args.mode=='continuous' else 8*sum(s['is_writer'] for s in plan)
        cohorts.append(dict(cohort=cohort,submitted=len(group),completed=len(done),completion_rate=len(done)/len(group),
            queue_mean_s=sum(r['queue_s'] for r in group)/len(group),service_p99_s=p99([r['service_s'] for r in done]),
            arrival_p99_s=p99([r['arrival_latency_s'] for r in done]),planned_operation_count=planned_count,
            planned_write_fraction=planned_writes/planned_count))
    protocol=dict(vars(args),new_api_calls=0,retrained=False,deferred_replay=False,retry_cache=False,model_sha256=sha(args.model),source_sha256=sha(Path(__file__)),
        cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
        dwa_scope='128 mixed tasks;each updater issues3reads+8writes to one8-record group,reader issues1read;report actual operation write fraction separately;early/late writes;readers stayExplore/OCC;paired logical versions;no first-read reservation',
        continuous_scope='Exactly10 issued API operations per completed attempt:8 catalog reads +2 independently sampled read/write operations on distinct shared accounts;overall write probabilities2/10/18percent in RW sequence;atomic publish history audit')
    for name,value in [('protocol',protocol),('result',result),('requests',requests),('tasks',rows),('attempts',attempts),('decisions',policy.questions),('lookup_samples',policy.calls),
        ('changes',changes),('continuity',continuity),('physical_events',physical.events),('publication_history',commits),('issued_operations',operations),('time_windows',windows),('arrival_cohorts',cohorts)]:save(args.output/f'{name}.json',value)
    print(json.dumps(result),flush=True)
    if wrong or result['unexpected_errors'] or physical._pending or physical.poisoned or not result['continuity_verified'] or grace['started'] or grace['commit_started']:raise SystemExit(2)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--mode',choices=['dwa','continuous'],required=True);p.add_argument('--arm',choices=['full','nophase','no_dwa','static','silo'],default='full')
    p.add_argument('--factor',choices=['reasoning','zipf','rw'],default='reasoning');p.add_argument('--position',choices=['early','late'],default='early')
    p.add_argument('--writer-fraction',type=float,default=.2);p.add_argument('--workers',type=int,default=32);p.add_argument('--objects',type=int,default=8)
    p.add_argument('--tasks',type=int,default=128);p.add_argument('--attempts',type=int,default=32);p.add_argument('--seed',type=int,default=61011010)
    p.add_argument('--period',type=float,default=30);p.add_argument('--rate',type=float,default=6);p.add_argument('--static-threshold',type=float,default=.2)
    p.add_argument('--dependency-guard',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
