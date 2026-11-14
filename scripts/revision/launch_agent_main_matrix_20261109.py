"""Fresh154-cell seven-worker matrix; pilot gate before immutable full run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.revision.run_agent_main_matrix_20261109 import CCS,WORKERS,make_plan

OUT=ROOT/'revision_experiments/agent_main_154_v2_20261109'
MODEL=ROOT/'revision_experiments/progressive_agent_final_20261109/selected_models/full_int16_table.json'
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def cpu_groups():
    allowed=os.sched_getaffinity(0);sockets={}
    topology=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    for line in topology.splitlines():
        if line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(','))
        if cpu in allowed:sockets.setdefault(socket,{}).setdefault(core,[]).append(cpu)
    groups=[]
    for socket,cores in sorted(sockets.items()):
        pairs=[sorted(c) for _,c in sorted(cores.items()) if len(c)==2]
        for i in range(0,len(pairs)-1,2):groups.append(sorted(pairs[i]+pairs[i+1]))
    if not groups:raise RuntimeError('Need disjoint2physical/4SMT CPU groups')
    return groups,topology


def execute(specs,groups,stage):
    pending=list(specs);active={};records=[]
    while pending or active:
        for slot,cpus in enumerate(groups):
            if slot in active or not pending:continue
            # node1 user process/thread limit is4096; leave headroom for
            # interpreter/native/helper threads at the200-worker points.
            if sum(r['workers'] for _,_,r in active.values())+pending[0][2]>2800:continue
            name,w,n,cc,inputfile=pending.pop(0);directory=OUT/'runs'/name
            if (directory/'result.json').exists():
                r=read(directory/'result.json')
                if not r['values_correct'] or r['input_sha256']!=sha(inputfile):raise ValueError('Invalid existing cell')
                records.append(dict(case=name,status='complete',cached=True,exit_code=0));continue
            if directory.exists():raise FileExistsError('Incomplete output; keep it and repair separately: '+name)
            log=(OUT/'logs'/f'{name}.log').open('w')
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,str(ROOT/'scripts/revision/run_agent_main_matrix_20261109.py'),
                '--input',str(inputfile),'--output',str(directory),'--model',str(MODEL),'--cc',cc,'--workers',str(n),'--attempts','32']
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            record=dict(case=name,workload=w,workers=n,cc=cc,pid=process.pid,cpus=cpus,started=time.time(),status='running');records.append(record);active[slot]=(process,log,record)
            print('START',name,cpus,flush=True)
        for slot,(process,log,r) in list(active.items()):
            if process.poll() is None:continue
            log.close();r.update(exit_code=process.returncode,ended=time.time(),status='complete' if process.returncode==0 else 'failed');del active[slot]
            print('END',r['case'],r['exit_code'],flush=True)
        save(OUT/f'{stage}_status.json',dict(stage=stage,expected=len(specs),pending=len(pending),running=len(active),records=records))
        if pending or active:time.sleep(3)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-only',action='store_true');parser.add_argument('--run-full',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    filelock=(OUT/'matrix.lock').open('w');fcntl.flock(filelock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups,topology=cpu_groups()
    for w in ['ycsb','tpcc']:
        for stage,count,seed in [('pilot',64,80901109),('main',240,80911109)]:
            path=OUT/'inputs'/f'{w}_{stage}.json';data=make_plan(w,count,seed)
            if path.exists():assert read(path)==data
            else:save(path,data)
    protocol=dict(ccs=CCS,workers=WORKERS,expected_cells=154,main_tasks=240,pilot_tasks=64,
        max_attempts=32,model_sha256=sha(MODEL),groups=groups,topology=topology,new_api_calls=0,
        trace_contract='Same workload request/latency file across every CC and worker point; no old132-point splicing',
        workload_scope='New progressive agent-derived YCSB9R/1W and numericTPC-C NewOrder/Payment; not standard benchmark figures',
        primary_latency='First attempt to successful commit includes retries; initial queue separately reported',
        main_seed=80911109,pilot_seed=80901109)
    protocol.update(ycsb_hot_accounts='Stratified shuffled blocks of8 requests, one per shared account; all CC/worker points share the plan',
        aegis_queue_aging_quantum_ms=10,previous_pilot_preserved='agent_main_154_20261109')
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    pilots=[(f'pilot_{w}_32_{cc}',w,32,cc,OUT/'inputs'/f'{w}_pilot.json') for w in ['ycsb','tpcc'] for cc in ['paper-atcc','silo','mocc']]
    records=execute(pilots,groups[:6],'pilot')
    if any(r['exit_code'] for r in records):
        save(OUT/'pilot_completion.json',dict(completed=False,records=records));return
    gains={}
    for w in ['ycsb','tpcc']:
        full=read(OUT/f'runs/pilot_{w}_32_paper-atcc/result.json')
        controls=[read(OUT/f'runs/pilot_{w}_32_{cc}/result.json') for cc in ['silo','mocc']]
        best=max(controls,key=lambda r:r['agent_tps'])
        gains[w]=dict(full_completion=full['completion_rate'],best_control=best['cc'],tps_ratio=full['agent_tps']/best['agent_tps'],p99_ratio=full['retry_inclusive_p99_s']/best['retry_inclusive_p99_s'])
    save(OUT/'pilot_completion.json',dict(completed=True,gains=gains,records=records))
    print('PILOT',json.dumps(gains),flush=True)
    if args.pilot_only or not args.run_full:return
    if any(g['full_completion']<.99 or g['tps_ratio']<1 for g in gains.values()):
        print('Pilot requires diagnosis before full matrix; no output overwritten',flush=True);return
    specs=[(f'{w}_{n}_{cc}',w,n,cc,OUT/'inputs'/f'{w}_main.json') for n in WORKERS for w in ['ycsb','tpcc'] for cc in CCS]
    save(OUT/'matrix_resources.json',dict(worker_thread_budget=2800,per_cell_cpus=4,per_cell_physical_cores=2,
        host_user_thread_limit=4096,shared_budget_limits_only_simultaneous_cells=True))
    records=execute(specs,groups,'matrix')
    save(OUT/'completion.json',dict(completed=all(r['exit_code']==0 for r in records),expected=154,
        completed_cells=sum(r['exit_code']==0 for r in records),failures=[r for r in records if r['exit_code']],records=records,new_api_calls=0))


if __name__=='__main__':main()
