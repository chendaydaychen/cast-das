"""Current frozen main-workload ablations and exclusive instrumentation.

Import the immutable main runner: task accesses and delays are identical to
the154-point main curves. All policy/runtime substitutions are explicit here.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.revision import run_agent_main_matrix_20261109 as mainrun
from agent.runtime.context import LockAction,LockClass

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class ExclusiveMeter:
    def __init__(self):self.local=threading.local();self.lock=threading.Lock();self.rows=[]
    @contextmanager
    def section(self,name):
        stack=getattr(self.local,'stack',None)
        if stack is None:
            if name!='task':yield;return
            self.local.stack=stack=[]
        if not stack and name!='task':yield;return
        frame=dict(name=name,start=time.perf_counter_ns(),cpu=time.thread_time_ns(),child_wall=0,child_cpu=0)
        stack.append(frame)
        try:yield
        finally:
            wall=time.perf_counter_ns()-frame['start'];cpu=time.thread_time_ns()-frame['cpu'];stack.pop()
            if stack:stack[-1]['child_wall']+=wall;stack[-1]['child_cpu']+=cpu
            self.local.totals[name][0]+=wall-frame['child_wall'];self.local.totals[name][1]+=cpu-frame['child_cpu']
            if name=='task':
                with self.lock:self.rows.append(dict(task=self.local.task,wall_ns=wall,cpu_ns=cpu,
                    exclusive={k:dict(wall_ns=v[0],cpu_ns=v[1]) for k,v in self.local.totals.items()}))
    def wrap(self,fn,name):
        def wrapped(*args,**kwargs):
            with self.section(name):return fn(*args,**kwargs)
        return wrapped

class FrozenStatic:
    phase_awareness='disabled';monotone_actions=True
    def select(self,state):return LockAction.from_encoded(int(state.current_action))

def execute(args):
    if args.arm=='no_dwa' and not args.physical_pair:raise ValueError('NoDWA requires paired physical publication control')
    count_only=bool(getattr(args,'count_only',False))
    meter=ExclusiveMeter();refs={};used_tids=set();commit_timings=[];priorities=[];static_begins=[]
    original_manager=mainrun.AgentTransactionManager;original_configure=mainrun.configure_paper_observed_contract
    original_meter=mainrun.MeterPolicy;original_table=mainrun.QuantizedSharedBitTable
    original_executor=mainrun.ThreadPoolExecutor;original_time=mainrun.time;original_manager_for=mainrun.manager_for
    original_reservation=mainrun.ObservedWriteReservationHooks
    class InstrumentedPolicy(original_meter):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs['policy']=self
        def select(self,state):
            if count_only:return super().select(state)
            with meter.section('lookup'):return super().select(state)
    class Executor(ThreadPoolExecutor):
        def map(self,fn,*iterables,**kw):
            if count_only:return super().map(fn,*iterables,**kw)
            def task(spec):
                meter.local.task=spec['index'];meter.local.totals=defaultdict(lambda:[0,0])
                with meter.section('task'):return fn(spec)
            return super().map(task,*iterables,**kw)
    class Clock:
        def __getattr__(self,name):return getattr(time,name)
        def sleep(self,seconds):
            with meter.section('reasoning'):return time.sleep(seconds)
    def install_instrumentation(manager):
        refs['manager']=manager
        if not count_only:
            manager.commit=meter.wrap(manager.commit,'commit')
            manager.transition_atcc_action=meter.wrap(manager.transition_atcc_action,'change')
        condition=manager.atcc_locks._condition
        if not count_only:condition.wait=meter.wrap(condition.wait,'lock_wait')
        old_timing=manager.add_commit_timing
        def timing(txn,category,elapsed_ms):
            if getattr(meter.local,'stack',[]):commit_timings.append(dict(task=meter.local.task,category=category,ms=elapsed_ms))
            return old_timing(txn,category,elapsed_ms)
        if not count_only:manager.add_commit_timing=timing
        old_begin=manager.begin
        def begin(*a,**kw):
            if args.arm=='static' and kw.get('strategy')=='paper-atcc':
                risk=manager.paper_runtime_metrics()['conflict_abort_rate']
                with manager._lock:
                    manager.paper_static_initial_mask=4 if risk>=args.static_threshold else 0
                    tx=old_begin(*a,**kw)
                    static_begins.append(dict(tid=tx.context.tid,risk=risk,mask=int(tx.context.action.protected)))
            else:tx=old_begin(*a,**kw)
            if (getattr(meter.local,'stack',[]) or count_only) and not str(tx.task_id).startswith('maintenance:'):used_tids.add(tx.context.tid)
            return tx
        manager.begin=begin
        original_finish=manager.atcc_locks.release_all
        def release(context):
            if context.tid in used_tids:priorities.append(int(context.priority))
            return original_finish(context)
        manager.atcc_locks.release_all=release
        return manager
    def manager_factory(**kw):
        kw['priority_enabled']=args.arm!='no_priority'
        return install_instrumentation(original_manager(**kw))
    def manager_for(cc):return install_instrumentation(original_manager_for(cc))
    def configure(manager,**kw):
        result=original_configure(manager,**kw)
        if args.physical_pair:
            from agent.runtime.eager_pcc_store import install_eager_pcc_control
            refs['physical']=install_eager_pcc_control(manager,eager=args.arm=='no_dwa',timeout_s=120.)
        manager.paper_deferred_replay_enabled=False;manager.atcc_retry_cache_enabled=False
        return result
    def loader(payload):
        if args.arm=='static':return FrozenStatic()
        if args.encoding=='actor':
            from agent.runtime.shared_bit_mask_policy import SharedBitMaskActor
            from agent.runtime.joint_mask_policy import JointMaskTablePolicy
            # Frozen actor law, cached joint coefficients; the training object
            # exposes sampling/probabilities, not the runtime select API.
            return JointMaskTablePolicy(SharedBitMaskActor.from_dict(payload).joint_payload())
        if args.encoding=='medoid':
            from agent.runtime.paper_observation_policy import PaperObservationPolicy
            return PaperObservationPolicy.from_dict(payload)
        return original_table(payload)
    class IssuedWriteStatic(original_reservation):
        def before_read(self,txn,key):return self.delegate.before_read(txn,key)
    mainrun.AgentTransactionManager=manager_factory;mainrun.configure_paper_observed_contract=configure
    mainrun.MeterPolicy=InstrumentedPolicy;mainrun.QuantizedSharedBitTable=loader
    mainrun.ThreadPoolExecutor=Executor;mainrun.time=original_time if count_only else Clock();mainrun.manager_for=manager_for
    if args.arm=='static':mainrun.ObservedWriteReservationHooks=IssuedWriteStatic
    try:
        runargs=argparse.Namespace(input=args.input,output=args.output,model=args.model,workers=args.workers,attempts=args.attempts,
            cc='silo' if args.arm=='silo' else '2pl-nowait' if args.arm=='2pl' else 'paper-atcc',lock_timeout_s=120.,commit_grace_ms=50.)
        mainrun.run(runargs)
    finally:
        mainrun.AgentTransactionManager=original_manager;mainrun.configure_paper_observed_contract=original_configure
        mainrun.MeterPolicy=original_meter;mainrun.QuantizedSharedBitTable=original_table
        mainrun.ThreadPoolExecutor=original_executor;mainrun.time=original_time;mainrun.manager_for=original_manager_for
        mainrun.ObservedWriteReservationHooks=original_reservation
    manager=refs['manager'];physical=refs.get('physical')
    result=read(args.output/'result.json');protocol=read(args.output/'protocol.json')
    per_component=defaultdict(lambda:dict(wall_ns=0,cpu_ns=0))
    for row in meter.rows:
        assert abs(sum(x['wall_ns'] for x in row['exclusive'].values())-row['wall_ns'])<=100
        for name,values in row['exclusive'].items():
            for k,v in values.items():per_component[name][k]+=v
    validation_ns=sum(t['ms']*1e6 for t in commit_timings if t['category']=='validate')
    commit_ns=per_component['commit']['wall_ns']
    if validation_ns>commit_ns+1e6:raise ValueError('Validation subtotal exceeds exclusive commit time')
    result.update(arm=args.arm,encoding=args.encoding,physical_pair=args.physical_pair,count_only=count_only,
        max_observed_priority=max(priorities,default=0),lookup_cpu_mean_us=sum(c['cpu_ns'] for c in refs.get('policy',type('P',(),{'calls':[]})()).calls)/max(1,result['policy_calls'])/1000,
        validation_reported_ms=validation_ns/1e6,early_physical_installs=sum(e['kind']=='early-install' and e.get('tid') in used_tids for e in physical.events) if physical else 0,
        physical_pending=len(physical._pending) if physical else 0,physical_poisoned=physical.poisoned if physical else False)
    if args.arm=='no_priority':assert result['max_observed_priority']==0
    if args.arm=='static':assert result['actual_changes']==0
    assert not result['physical_pending'] and not result['physical_poisoned']
    protocol.update(supplement=dict(source_sha256=sha(Path(__file__)),arm=args.arm,encoding=args.encoding,
        model_path=str(args.model),model_sha256=sha(args.model),physical_pair=args.physical_pair,
        count_only=count_only,actor_scoring='Exact frozen joint coefficient law; no fitting at deployment',
        static_threshold=args.static_threshold,static_scope='Per-attempt mask chosen at begin from historical conflict EWMA, fixed inside attempt; no phase input',
        static_first_read_write_reservation=False if args.arm=='static' else None,
        deferred_replay=False,retry_cache=False,primary_trace_unchanged=True))
    outputs={'result':result,'protocol':protocol,'exclusive_profile':dict(tasks=meter.rows,totals=per_component,
        validation_reported_ns=validation_ns,commit_nonvalidation_ns=max(0,commit_ns-validation_ns),
        scope='Summed task service time including retries, excludes initial executor queue; task category is residual; change excludes child waits'),
        'lookup_samples':refs['policy'].calls if 'policy' in refs else [],'commit_timings':commit_timings,'static_begin_choices':static_begins,
        'physical_events':[e for e in physical.events if e.get('tid') in used_tids] if physical else []}
    for name,value in outputs.items():(args.output/f'{name}.json').write_text(json.dumps(value,indent=2))
    print('SUPPLEMENT',json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--arm',choices=['full','nophase','no_priority','no_dwa','static','silo','2pl'],default='full')
    p.add_argument('--encoding',choices=['int16','actor','medoid'],default='int16');p.add_argument('--physical-pair',action='store_true')
    p.add_argument('--workers',type=int,default=32);p.add_argument('--attempts',type=int,default=32);p.add_argument('--static-threshold',type=float,default=.2)
    p.add_argument('--count-only',action='store_true')
    execute(p.parse_args())

if __name__=='__main__':main()
