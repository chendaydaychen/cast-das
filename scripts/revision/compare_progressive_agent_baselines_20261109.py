"""Traditional/adapted CC controls, same progressive plan and real KV APIs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agent.runtime import AgentTransactionManager
from agent.runtime.context import TransactionPhase,TransactionStatus
from scripts.revision.progressive_agent_v2_environment_20261109 import plan_task

OUT=ROOT/'revision_experiments/progressive_agent_cc_20261109'


def manager_for(cc):
    if cc=='2pl-nowait':
        from scripts.unified_trace.run_corrected_tail_matrix_20261106 import OperationTwoPhaseManager
        return OperationTwoPhaseManager(record_traces=False,collect_trajectories=False)
    if cc=='tictoc':
        from agent.runtime.tictoc_runtime import TicTocRuntimeManager
        return TicTocRuntimeManager(record_traces=False,collect_trajectories=False)
    if cc in ['mocc','hekaton']:
        from scripts.unified_trace.run_same_runtime_all_cc import adapter_class
        return adapter_class(cc)(record_traces=False,collect_trajectories=False)
    if cc in ['polyjuice','neurcc']:
        from agent.runtime.learned_policy_adapters import AdaptedLearnedPolicy
        from agent.runtime.audited_learned_adapter import AuditedLearnedAdapter
        text=(ROOT/f'revision_experiments/real_delay_all_cc_20261106/policies/{cc}.txt').read_text()
        return AuditedLearnedAdapter(speculative_policy=getattr(AdaptedLearnedPolicy,cc)(text),record_traces=False,collect_trajectories=False)
    return AgentTransactionManager(record_traces=False,collect_trajectories=False)


def run(cc,repair_output=False):
    target=OUT/(cc+'_api_fixed' if repair_output else cc);target.mkdir(exist_ok=False);manager=manager_for(cc)
    strategy='silo' if cc in ['polyjuice','neurcc'] else cc
    for key in [f'g{g}:k0' for g in range(8)]+[f'discovery:{i}' for i in range(64)]:manager.register_object(key,'0')
    # Same completed historical maintenance writes as the Aegis input.
    for key in [f'g{g}:k0' for g in range(8)]+[f'discovery:{i}' for i in range(64)]:
        tx=manager.begin('history:'+key,strategy=strategy);tx.write(key,'0')
        if not tx.commit(strategy).committed:raise ValueError('Common history failed')
    plans=[plan_task(i,50911109,.2,0,True) for i in range(64)];start=time.monotonic();rows=[];attempts=[]
    def task(spec):
        ok=False;detail=[]
        for attempt in range(8):
            tx=manager.begin(str(spec['index']),dict(retry_count=attempt,access_set_visibility='online_observed'),strategy=strategy);reason=''
            try:
                tx.enter_phase(TransactionPhase.EXPLORE)
                for rnd in range(spec['explore_rounds']):
                    for key in spec['discovery'][rnd::spec['explore_rounds']]:tx.read(key)
                    time.sleep(3.161*spec['explore_multiplier']/spec['explore_rounds'])
                tx.enter_phase(TransactionPhase.REFINE);value=int(tx.read(spec['target']).value)
                time.sleep(2.933*spec['refine_multiplier'])
                tx.enter_phase(TransactionPhase.COMMIT);tx.write(spec['target'],str(value+1))
                result=tx.commit(strategy);ok=result.committed;reason=result.reason
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc)
                if tx.context.status!=TransactionStatus.COMMITTED:tx.abort_or_observe_wound(reason,strategy=strategy)
            detail.append(dict(task=spec['index'],attempt=attempt,committed=ok,reason=reason))
            if ok:break
        row=dict(task=spec['index'],committed=ok,target=spec['target'],retry_inclusive_s=time.monotonic()-start)
        rows.append(row);attempts.extend(detail)
    with ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(task,plans))
    elapsed=time.monotonic()-start;expected={f'g{g}:k0':0 for g in range(8)}
    for row in rows:
        if row['committed']:expected[row['target']]+=1
    wrong={k:dict(expected=v,actual=manager.store.get(k).value) for k,v in expected.items() if int(manager.store.get(k).value)!=v}
    done=[r['retry_inclusive_s'] for r in rows if r['committed']];done.sort()
    errors=[a for a in attempts if a['reason'].startswith(('AttributeError','TypeError','NameError','AssertionError','RuntimeError'))]
    result=dict(cc=cc,tps=len(done)/elapsed,p99_s=done[math.ceil(.99*len(done))-1] if done else None,
        completed=len(done),submitted=64,completion_rate=len(done)/64,aborts=sum(not a['committed'] for a in attempts),wall_s=elapsed,
        value_audit_passed=not wrong,value_errors=wrong,unexpected_errors=errors,new_api_calls=0,
        scope='Common runtime CC adapters; fixed64 tasks,32 workers,8 attempts; not native product comparison')
    for name,data in [('result',result),('tasks',rows),('attempts',attempts),('plans',plans)]:
        (target/f'{name}.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(result),flush=True)
    if wrong or errors:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cc');p.add_argument('--repair-output',action='store_true');args=p.parse_args()
    if args.cc:return run(args.cc,args.repair_output)
    OUT.mkdir(exist_ok=False);pending=['occ','2pl-nowait','mvcc','silo','tictoc','bamboo','polaris','mocc','hekaton','polyjuice','neurcc'];active={};records=[]
    groups=[list(range(i,i+4)) for i in range(48,72,4)]
    while pending or active:
        for slot,cpus in enumerate(groups):
            if slot in active or not pending:continue
            cc=pending.pop(0);log=(OUT/f'{cc}.log').open('w')
            process=subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),sys.executable,str(Path(__file__).resolve()),'--cc',cc],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            r=dict(cc=cc,pid=process.pid,cpus=cpus);records.append(r);active[slot]=(process,log,r)
        for slot,(process,log,r) in list(active.items()):
            if process.poll() is None:continue
            log.close();r['exit_code']=process.returncode;del active[slot];print('END',r['cc'],r['exit_code'],flush=True)
        if pending or active:time.sleep(2)
    (OUT/'completion.json').write_text(json.dumps(dict(completed=all(r['exit_code']==0 for r in records),records=records),indent=2))


if __name__=='__main__':main()
