"""Single cell of the progressive Agent-YCSB / Agent-TPC-C matrix.

Synthetic agent-derived workloads; shared native KV, frozen request/latency
plans, actual numeric RMW and business invariants. No external LLM calls.
"""
import argparse
import dataclasses
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
from agent.runtime.paper_contract import configure_paper_observed_contract
from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
from agent.runtime.single_writer_grace_20261109 import install as install_grace
from agent.workloads.frozen_credit_agent import tracked_agent_wait
from scripts.revision.progressive_agent_environment_20261109 import ObservedBoundaryHooks,OBSERVED_LLM
from scripts.revision.grace_phase_environment_20261109 import MeterPolicy
from scripts.revision.compare_progressive_agent_baselines_20261109 import manager_for

CCS=['paper-atcc','occ','2pl-nowait','mvcc','silo','tictoc','bamboo','polaris','mocc','hekaton','polyjuice']
WORKERS=[8,16,24,32,40,100,200]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def percentile(xs):return sorted(xs)[max(0,math.ceil(.99*len(xs))-1)] if xs else None


def make_plan(workload,count,seed,*,shared_objects=8,catalog_objects=64,stock_objects=64):
    if any(type(n) is not int or n<1 for n in [shared_objects,catalog_objects,stock_objects]):raise ValueError('Positive object counts required')
    if catalog_objects<8 or stock_objects<5:raise ValueError('Insufficient catalog/stock objects')
    rng=random.Random(seed);tasks=[];balanced=[]
    for offset in range(0,count,shared_objects):
        block=list(range(shared_objects));rng.shuffle(block);balanced.extend(block)
    for i in range(count):
        spec=dict(index=i,explore_ms=rng.choice(OBSERVED_LLM['explore']),refine_ms=rng.choice(OBSERVED_LLM['refine']),rounds=rng.randint(1,3))
        if workload=='ycsb':
            spec.update(kind='read-modify-write',catalog=rng.sample(range(catalog_objects),8),account=balanced[i],delta=1)
        else:
            kind='new_order' if rng.random()<.5 else 'payment'
            spec.update(kind=kind,district=rng.randrange(shared_objects),customer=rng.randrange(shared_objects),
                items=rng.sample(range(stock_objects),rng.randint(3,5)),quantity=rng.randint(1,3),amount=rng.randint(1,20))
        tasks.append(spec)
    result=dict(workload=workload,count=count,seed=seed,tasks=tasks,
        semantics='Progressive agent-derived9R/1W YCSB;TPC-C-derived NewOrder/Payment numeric point workload,not standard SQL TPC-C',
        latencies='Empirical sampling from16Explore/4Refine historical CreditCrew HTTP calls; synthetic visits,not live LLM',
        input_prefix_fraction=.2)
    if (shared_objects,catalog_objects,stock_objects)!=(8,64,64):
        result.update(shared_objects=shared_objects,catalog_objects=catalog_objects,stock_objects=stock_objects)
    return result


def initial_objects(workload,count,*,shared_objects=8,catalog_objects=64,stock_objects=64):
    if workload=='ycsb':return {**{f'catalog:{i}':'0' for i in range(catalog_objects)},**{f'account:{i}':'0' for i in range(shared_objects)}}
    initial={'warehouse:tax':'0.05','warehouse:ytd':'0'}
    for d in range(shared_objects):
        initial.update({f'district:{d}:tax':'0.03',f'district:{d}:next':'1',f'district:{d}:ytd':'0'})
        for order in range(1,count+1):initial[f'order:{d}:{order}']='EMPTY'
    for c in range(shared_objects):initial.update({f'customer:{c}:discount':'0.01',f'customer:{c}:balance':'0',f'customer:{c}:ytd':'0',f'customer:{c}:payments':'0'})
    for item in range(stock_objects):initial.update({f'item:{item}:price':'1',f'stock:{item}:quantity':'100000',f'stock:{item}:ytd':'0',f'stock:{item}:orders':'0'})
    return initial


def expected_numeric(workload,initial,done,requests):
    if workload=='ycsb':
        expected={k:0 for k in initial if k.startswith('account:')}
        for row in done:expected[f"account:{requests[row['task']]['account']}"]+=1
        return expected
    expected={k:int(v) for k,v in initial.items() if k not in ['warehouse:tax'] and not k.endswith((':tax',':discount',':price')) and not k.startswith('order:')}
    for row in done:
        s=requests[row['task']];d=s['district'];c=s['customer']
        if s['kind']=='new_order':
            expected[f'district:{d}:next']+=1
            for item in s['items']:
                expected[f'stock:{item}:quantity']-=s['quantity'];expected[f'stock:{item}:ytd']+=s['quantity'];expected[f'stock:{item}:orders']+=1
        else:
            expected['warehouse:ytd']+=s['amount'];expected[f'district:{d}:ytd']+=s['amount']
            expected[f'customer:{c}:balance']-=s['amount'];expected[f'customer:{c}:ytd']+=s['amount'];expected[f'customer:{c}:payments']+=1
    return expected


def run(args):
    source=read(args.input);requests=source['tasks']
    initial=initial_objects(source['workload'],len(requests),**{key:source.get(key,default) for key,default in [('shared_objects',8),('catalog_objects',64),('stock_objects',64)]})
    args.output.mkdir(exist_ok=False,parents=True)
    lock_timeout_s=float(getattr(args,'lock_timeout_s',30.))
    if not math.isfinite(lock_timeout_s) or lock_timeout_s<=0:raise ValueError('Positive finite lock timeout required')
    full=args.cc=='paper-atcc';policy=None
    if full:
        policy=MeterPolicy(QuantizedSharedBitTable(read(args.model)))
        manager=AgentTransactionManager(paper_policy=policy,record_traces=False,collect_trajectories=True,performance_guards_enabled=False,low_conflict_occ_guard=False)
        configure_paper_observed_contract(manager,explicit_phase_label_ablation=True)
        # Account for a second spent queued on the same scale as a second of
        # paid reasoning; the old100ms-vs10ms quanta allowed costly newcomers
        # to overtake a cheap waiter until its30s deadline expired.
        manager.priority_manager.config=dataclasses.replace(manager.priority_manager.config,blocked_quantum_ms=10.)
        manager.interceptor.hooks._action_transition_timeout_s=lambda tx:lock_timeout_s
        for name in ['wlock','validate_and_rlock','validate_and_wlock','acquire_write_set']:
            original=getattr(manager.atcc_locks,name)
            def budget(*values,_original=original,**kw):kw.setdefault('timeout_s',lock_timeout_s);return _original(*values,**kw)
            setattr(manager.atcc_locks,name,budget)
        grace=install_grace(manager,4500.,commit_grace_ms=float(getattr(args,'commit_grace_ms',0.)))
        reservation=ObservedWriteReservationHooks(manager.interceptor.hooks,capacity=max(4096,len(initial)))
        manager.interceptor.hooks=ObservedBoundaryHooks(reservation)
    else:manager=manager_for(args.cc);grace={'started':{}};reservation=None
    if args.cc=='mocc':manager.mocc_locks.timeout_s=lock_timeout_s
    strategy='silo' if args.cc=='polyjuice' else args.cc
    for key,value in initial.items():manager.register_object(key,value)
    for key,value in initial.items():
        if value=='EMPTY':continue
        tx=manager.begin('maintenance:'+key,strategy='silo' if full else strategy);tx.write(key,value)
        if not tx.commit('silo' if full else strategy).committed:raise ValueError('Historical maintenance failed')
        if reservation is not None:reservation.observed_writes[key]=True
    if policy is not None:policy.calls.clear();policy.questions.clear()
    lock_tenures=[];lock_starts={}
    if full:
        original_note=manager.atcc_locks._note_acquired;original_release=manager.atcc_locks.release_all
        def note(context):
            original_note(context)
            for key in context.held_write_locks:lock_starts.setdefault((context.tid,key),time.monotonic())
        def release(context):
            with manager.atcc_locks._condition:
                now=time.monotonic()
                for key in tuple(context.held_write_locks):
                    began=lock_starts.pop((context.tid,key),None)
                    if began is not None:lock_tenures.append(dict(key=key,tid=context.tid,seconds=now-began))
                return original_release(context)
        manager.atcc_locks._note_acquired=note;manager.atcc_locks.release_all=release
    start=time.monotonic();rows=[];attempt_rows=[];changes=[];waits=[];lock=threading.Lock();inflight=peak=0
    transition=manager.transition_atcc_action
    def measure_transition(tx,action,**kw):
        before=int(tx.context.action.protected);tick=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:return transition(tx,action,**kw)
        finally:changes.append(dict(actual=before!=int(tx.context.action.protected),wall_ns=time.perf_counter_ns()-tick,cpu_ns=time.thread_time_ns()-cpu))
    manager.transition_atcc_action=measure_transition
    def execute(spec):
        nonlocal inflight,peak
        began=time.monotonic();ok=False;detail=[];queue=began-start;paid_total=0.
        with lock:inflight+=1;peak=max(peak,inflight)
        for attempt in range(args.attempts):
            if policy is not None:policy.local.task=spec['index'];policy.local.attempt=attempt
            # Shared instruction parsing is independent of database values.
            time.sleep(source['input_prefix_fraction']*spec['explore_ms']/1000)
            tx=manager.begin(str(spec['index']),dict(retry_count=attempt,access_set_visibility='online_observed'),strategy=strategy)
            reason='';paid=0.;order_id=None
            try:
                def read_value(key):return tx.read(key).value
                def write_value(key,value):tx.write(key,str(value))
                def add(key,delta):write_value(key,int(read_value(key))+delta)
                def wait(ms,phase):
                    nonlocal paid
                    tick=time.monotonic()
                    if full:
                        with tracked_agent_wait(tx):time.sleep(ms/1000)
                    else:time.sleep(ms/1000)
                    elapsed=time.monotonic()-tick;paid+=elapsed
                    waits.append(dict(task=spec['index'],attempt=attempt,phase=phase,seconds=elapsed,
                        observed_versions={k:int(v.version) for k,v in tx.read_set.items()},pre_begin=False))
                tx.enter_phase(TransactionPhase.EXPLORE)
                if source['workload']=='ycsb':catalog=[f'catalog:{i}' for i in spec['catalog']]
                else:catalog=['warehouse:tax',f"district:{spec['district']}:tax",f"customer:{spec['customer']}:discount"]+[f'item:{i}:price' for i in spec['items']]
                for rnd in range(spec['rounds']):
                    for key in catalog[rnd::spec['rounds']]:read_value(key)
                    wait(.8*spec['explore_ms']/spec['rounds'],'explore')
                    manager.interceptor.agent_round_finished(tx)
                tx.enter_phase(TransactionPhase.REFINE)
                if source['workload']=='ycsb':critical=f"account:{spec['account']}"
                elif spec['kind']=='new_order':critical=f"district:{spec['district']}:next"
                else:critical=f"customer:{spec['customer']}:balance"
                observed=int(read_value(critical));wait(spec['refine_ms'],'refine');manager.interceptor.agent_round_finished(tx)
                tx.enter_phase(TransactionPhase.COMMIT)
                if source['workload']=='ycsb':write_value(critical,observed+1)
                elif spec['kind']=='new_order':
                    order_id=observed;write_value(critical,observed+1)
                    # Stable order avoids protocol-independent lock-order cycles.
                    for item in sorted(spec['items']):
                        add(f'stock:{item}:quantity',-spec['quantity']);add(f'stock:{item}:ytd',spec['quantity']);add(f'stock:{item}:orders',1)
                    write_value(f"order:{spec['district']}:{observed}",json.dumps(dict(task=spec['index'],items=spec['items'],quantity=spec['quantity']),sort_keys=True))
                else:
                    write_value(critical,observed-spec['amount'])
                    for key in sorted(['warehouse:ytd',f"district:{spec['district']}:ytd",f"customer:{spec['customer']}:ytd"]):add(key,spec['amount'])
                    add(f"customer:{spec['customer']}:payments",1)
                result=tx.commit(strategy);ok=bool(result.committed);reason=str(result.reason)
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc)
                if tx.context.status!=TransactionStatus.COMMITTED:tx.abort_or_observe_wound(reason,strategy=strategy)
            paid_total+=paid;detail.append(dict(task=spec['index'],attempt=attempt,committed=ok,reason=reason,paid_s=paid,blocked_ms=tx.context.blocked_time_ms))
            if ok:break
        ended=time.monotonic()
        with lock:
            inflight-=1;attempt_rows.extend(detail);rows.append(dict(task=spec['index'],kind=spec['kind'],committed=ok,attempts=len(detail),
                queue_s=queue,start_s=began-start,finish_s=ended-start,retry_inclusive_s=ended-began,arrival_to_finish_s=ended-start,paid_s=paid_total,order_id=order_id))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(execute,requests))
    elapsed=time.monotonic()-start;done=[r for r in rows if r['committed']];expected=expected_numeric(source['workload'],initial,done,requests)
    wrong={k:dict(expected=v,actual=manager.store.get(k).value) for k,v in expected.items() if int(manager.store.get(k).value)!=v}
    if source['workload']=='tpcc':
        orders=[r for r in done if r['kind']=='new_order'];published=[]
        for key in initial:
            if key.startswith('order:') and manager.store.get(key).value!='EMPTY':published.append((key,json.loads(manager.store.get(key).value)['task']))
        if sorted(t for _,t in published)!=sorted(r['task'] for r in orders):wrong['order-publication']=dict(expected=len(orders),actual=published)
    errors=[a for a in attempt_rows if a['reason'].startswith(('AttributeError','TypeError','NameError','AssertionError','RuntimeError','ValueError','KeyError'))]
    result=dict(workload='Agent-'+source['workload'].upper()+'-Progressive',cc=args.cc,workers=args.workers,submitted=len(rows),completed=len(done),completion_rate=len(done)/len(rows),
        agent_tps=len(done)/elapsed,retry_inclusive_p99_s=percentile([r['retry_inclusive_s'] for r in done]),arrival_to_finish_p99_s=percentile([r['arrival_to_finish_s'] for r in done]),
        queue_mean_s=sum(r['queue_s'] for r in rows)/len(rows),aborts=sum(not a['committed'] for a in attempt_rows),elapsed_s=elapsed,peak_inflight=peak,
        max_attempts=args.attempts,lock_timeout_s=lock_timeout_s,values_correct=not wrong,value_errors=wrong,unexpected_errors=errors,
        grace_leaks=len(grace['started'])+len(grace.get('commit_started',{})),
        policy_calls=len(policy.calls) if policy else 0,actual_changes=sum(c['actual'] for c in changes),
        commit_grace_ms=float(getattr(args,'commit_grace_ms',0.)),commit_grace_deferrals=grace.get('commit_deferrals',0),
        lookup_mean_us=sum(c['wall_ns'] for c in policy.calls)/max(1,len(policy.calls))/1000 if policy else 0,
        input_sha256=sha(args.input),model_sha256=sha(args.model) if full else None,new_api_calls=0)
    for name,value in [('result',result),('tasks',rows),('attempts',attempt_rows),('waits',waits),('changes',changes),('decisions',policy.questions if policy else []),
        ('write_lock_tenures',lock_tenures),('lock_diagnostics',manager.atcc_locks.snapshot_diagnostics() if full else {})]:
        (args.output/f'{name}.json').write_text(json.dumps(value,indent=2))
    (args.output/'protocol.json').write_text(json.dumps(dict(vars(args),source_sha256=sha(Path(__file__)),cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
        p99='First execution attempt through final commit including retries; initial executor queue separately reported',
        prefix='All CC share independent prebegin instruction parsing; dependent waits remain after actual reads',
        workload_scope=source['semantics'],new_api_calls=0),indent=2,default=str))
    print(json.dumps(result),flush=True)
    if wrong or errors or grace['started']:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--cc',choices=CCS,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--attempts',type=int,default=32)
    p.add_argument('--lock-timeout-s',type=float,default=30.)
    p.add_argument('--commit-grace-ms',type=float,default=0.)
    run(p.parse_args())


if __name__=='__main__':main()
