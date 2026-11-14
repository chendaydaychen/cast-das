"""Larger fixed-trace runtime comparison with explicit adapters and task samples.

This never invokes native CC binaries or an LLM. Legacy approximate adapters
are retained with their implementation scope recorded, not silently upgraded.
"""
import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.unified_trace import run_same_runtime_all_cc as prior
from agent.benchmarks import mixed
from agent.runtime.transaction import AgentTransactionManager
from agent.runtime.strict_two_phase import StrictTwoPhaseHooks
from agent.runtime.tictoc_runtime import TicTocRuntimeManager
from agent.runtime.point_insert_baselines import HekatonPointInsertRuntimeManager
from agent.benchmarks.mixed import percentile

class OperationTwoPhaseManager(AgentTransactionManager):
    def __init__(self, **kwargs):
        from agent.runtime.operation_interceptor import NoopTransactionHooks
        kwargs.update(transaction_hooks=NoopTransactionHooks(), priority_enabled=False,
                      performance_guards_enabled=False)
        super().__init__(**kwargs)
        self.interceptor.hooks=StrictTwoPhaseHooks(self)

class TaskSamples:
    """Only records task/attempt outcomes, without changing CC decisions."""
    def __init__(self, manager, strategy):
        self.manager=manager; self.strategy=strategy
        self.lock=threading.Lock(); self.local=threading.local()
        self.clock=self
        self.reset()
        begin=manager.begin
        def record_begin(*args, **kwargs):
            txn=begin(*args, **kwargs)
            opened=time.perf_counter()
            self.local.attempt_started=opened
            self.local.txn=txn
            return txn
        manager.begin=record_begin
        for method in ['commit', '_finish_abort']:
            original=getattr(manager,method)
            def finished(txn,*args,_original=original,**kwargs):
                result=_original(txn,*args,**kwargs)
                identity=getattr(self.local,'identity',{})
                entry=dict(identity=identity,txn_id=txn.context.tid,
                    elapsed_ms=(time.perf_counter()-getattr(self.local,'attempt_started',time.perf_counter()))*1000,
                    committed=bool(result.committed),reason=str(result.reason),
                    retry_count=int(txn.context.retry_count),phase=str(txn.context.phase))
                with self.lock:
                    # A commit failure may already have passed through _finish_abort.
                    key=str(txn.context.tid)
                    self.attempts[key]=entry
                return result
            setattr(manager,method,finished)
    def scope(self,*args):return contextlib.nullcontext()
    def reset(self):self.tasks=[];self.attempts={}
    def begin_task(self,started_s):self.local.identity={}
    def begin_identified_task(self,started_s,identity):self.local.identity=dict(identity)
    def finish_task(self,ended_s,elapsed_ms,*,committed,attempts):
        with self.lock:self.tasks.append(dict(identity=self.local.identity,
            elapsed_ms=elapsed_ms,committed=committed,attempts=attempts))
    def report(self):return dict(tasks=self.tasks,attempts=list(self.attempts.values()))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(output, tasks, seed=24010602):
    from scripts.revision.run_ycsb_fresh_dynamic_2401 import transform_trace
    for workers in [8,16,32,40]:
        folder=output/'traces';folder.mkdir(exist_ok=True)
        for workload,variant in [('ycsb','ycsb_high_z099'),('tpcc','tpcc_high_w1')]:
            raw=folder/f'{workload}_{workers}_base.csv'
            if not raw.exists():
                subprocess.run([sys.executable,str(ROOT/'scripts/unified_trace/generate_castdas_trace.py'),
                    '--output',str(raw),'--variant',variant,'--clients',str(workers),
                    '--agent-ratio','1','--seed',str(seed),'--transactions-per-worker',str(tasks),
                    '--reasoning-profile','credit_real','--reasoning-scale','1'],check=True,stdout=subprocess.DEVNULL)
            trace=folder/f'{workload}_{workers}.csv'
            if not trace.exists():
                if workload=='ycsb':transform_trace(raw,trace,refine_delay_ms=3161,role_mix=True)
                else:trace.write_bytes(raw.read_bytes())

def run_job(spec):
    output,workload,workers,name=spec
    output=Path(output);folder=output/'runs'/f'{workload}_{workers}_{name}'
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/'result.json'
    trace=output/'traces'/f'{workload}_{workers}.csv'
    policy_file=output/'policies'/workload/'full.json'
    if not policy_file.exists():
        policy_file=ROOT/'revision_experiments/real_credit_role_mix_retrain_20261105/full.json'
    if done.exists():
        cached=json.loads(done.read_text())
        if cached['trace_sha256']!=digest(trace):
            raise ValueError(f'Refusing stale trace result: {done}')
        if name=='paper-atcc' and cached.get('policy_sha256')!=digest(policy_file):
            raise ValueError(f'Refusing stale/unverified policy result: {done}; use a fresh output directory')
        return cached
    rows=prior.fair.read_trace(trace)
    old_manager=prior.fair.AgentTransactionManager
    old_plan=mixed.mixed_traditional_begin_lock_plan
    old_admission=prior.fair.paper_agent_admission_cap
    # Compare the requested worker window, without the legacy hard-coded
    # >=24-client Aegis-only admission step (12/13 active transactions).
    prior.fair.paper_agent_admission_cap=lambda manager,cc,config,*,agent_worker_count:agent_worker_count
    notes='shared runtime version-validation adapter'
    if name=='tictoc':
        manager=TicTocRuntimeManager;notes='tuple timestamp extension runtime adapter'
    elif name=='2pl-nowait':
        manager=OperationTwoPhaseManager;notes='operation-level S/X no-wait 2PL'
        mixed.mixed_traditional_begin_lock_plan=lambda strategy,task:None
    elif name=='paper-atcc':
        from scripts.revision.run_ycsb_fresh_dynamic_2401 import NoLowRiskGuardManager
        manager=NoLowRiskGuardManager;notes='frozen phase policy; no reasoning cache or moved replay'
        if __import__('os').environ.get('AEGIS_RETRY_ISSUE_PLAN')=='1':
            from agent.runtime.observed_retry_issue_plan import install
            parent=manager
            class RepairManager(parent):
                def __init__(self,**kw):super().__init__(**kw);install(self)
            manager=RepairManager
            notes+='; pre-reasoning checkpoints and observed retry issue repair'
    elif name=='hekaton':manager=HekatonPointInsertRuntimeManager;notes='point-version Hekaton runtime adapter'
    elif name in prior.ADAPTERS:
        manager=prior.adapter_class(name);notes='runtime protocol adapter'
        if name in ['polyjuice','neurcc']:
            from agent.runtime.learned_policy_adapters import AdaptedLearnedPolicy
            text=(ROOT/f'revision_experiments/real_delay_all_cc_20261106/policies/{name}.txt').read_text()
            policy=getattr(AdaptedLearnedPolicy,name)(text)
            parent=manager
            class BoundManager(parent):
                def __init__(self,**kw):super().__init__(speculative_policy=policy,**kw)
            manager=BoundManager;notes='frozen learned-policy runtime adapter'
    else:
        manager=AgentTransactionManager
        if name=='bamboo':notes='simplified commit-lock Bamboo adapter; no speculative retire'
        if name=='mvcc':notes='serializable MVCC fallback with observed-version validation'
        if name=='polaris':notes='Silo-priority-style commit-lock runtime adapter'
    prior.fair.AgentTransactionManager=manager
    profile_holder=[]
    def factory(m,s):
        p=TaskSamples(m,s);profile_holder.append(p);return p
    kw=dict(max_attempts=6,tokens_per_operation=64,execution_workers=0,
        measure_seconds=0,cycle_trace=False,paper_priority='enabled' if name=='paper-atcc' else 'disabled',
        paper_performance_guards='disabled',paper_delayed_write_apply='disabled',
        paper_deferred_replay_enabled=False,atcc_retry_cache_enabled=False,trace_profile_factory=factory)
    if name=='paper-atcc':
        kw.update(paper_policy=policy_file,
            paper_inflight_priority=True,trajectory_output=folder/'policy_trajectory.json')
    cc='silo' if name in prior.ADAPTERS else name
    started=time.time()
    (folder/'started.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),started=started,spec=list(spec))))
    try:
        r=prior.fair.run_trace(rows,cc=cc,policy=None,**kw)
        samples=profile_holder[0].report()
        (folder/'samples.json').write_text(json.dumps(samples,indent=2,default=str))
        success=[t['elapsed_ms'] for t in samples['tasks'] if t['committed']]
        all_samples=[t['elapsed_ms'] for t in samples['tasks']]
        assert len(success)==int(r['agent_completed_tasks'])
        assert len(all_samples)==int(r['agent_completed_tasks'])+int(r['agent_failed_tasks'])
        result=dict(workload=workload,workers=workers,strategy=prior.LABELS[name],
            manager_class=f'{manager.__module__}.{manager.__qualname__}',adapter_scope=notes,
            tps=r['agent_tps'],p50_ms=percentile(success,50),p95_ms=percentile(success,95),
            p99_ms=percentile(success,99),max_success_ms=max(success,default=0),
            terminal_p99_ms=percentile(all_samples,99),completed=len(success),failed=len(all_samples)-len(success),
            completion_rate=r['agent_task_completion_rate'],aborts=r['agent_aborts'],
            elapsed_s=r['elapsed_s'],trace_sha256=digest(trace),sample_file=str(folder/'samples.json'))
        result['tasks_per_worker']=len(all_samples)/workers
        result['p99_population']='completed logical tasks; includes all their attempts/backoff, excludes between-task think'
        result['failed_task_time_population']='terminal_p99_ms includes completed and exhausted logical tasks'
        result['policy_source']=str(policy_file) if name=='paper-atcc' else ''
        result['policy_sha256']=digest(policy_file) if name=='paper-atcc' else ''
        result['engineering_variant']=('retry_issue_plan_v3' if name=='paper-atcc' and
            __import__('os').environ.get('AEGIS_RETRY_ISSUE_PLAN')=='1' else '')
        (folder/'runtime_raw.json').write_text(json.dumps(r,indent=2,default=str))
        done.write_text(json.dumps(result,indent=2))
        return result
    except Exception as e:
        import traceback
        (folder/'error.txt').write_text(traceback.format_exc())
        return dict(workload=workload,workers=workers,strategy=name,error=repr(e))
    finally:
        prior.fair.AgentTransactionManager=old_manager
        mixed.mixed_traditional_begin_lock_plan=old_plan
        prior.fair.paper_agent_admission_cap=old_admission

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tasks-per-worker',type=int,default=20)
    ap.add_argument('--parallel',type=int,default=8)
    ap.add_argument('--seed',type=int,default=24010602)
    ap.add_argument('--workers',nargs='+',type=int,default=[8,16,32,40])
    ap.add_argument('--workloads',nargs='+',default=['ycsb','tpcc'])
    ap.add_argument('--strategies',nargs='+',default=list(prior.CORE+prior.ADAPTERS))
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    prepare(args.output,args.tasks_per_worker,args.seed)
    specs=[(str(args.output),l,w,n) for l in args.workloads for w in args.workers for n in args.strategies]
    # One strategy per process; global adapter binding never overlaps within a process.
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures=[pool.submit(run_job,s) for s in specs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps(result),flush=True)
            (args.output/'progress.json').write_text(json.dumps(results,indent=2))
    fields=sorted({key for row in results for key in row})
    with (args.output/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(results)

if __name__=='__main__':main()
