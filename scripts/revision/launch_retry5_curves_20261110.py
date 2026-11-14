"""Fresh 40/80/120/160/200-worker curves with unchanged final frozen inputs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.revision.launch_agent_main_matrix_20261109 import cpu_groups
from scripts.revision.run_agent_main_matrix_20261109 import CCS
OUT=ROOT/'revision_experiments/agent_retry5_curves_20261110'
WORKERS=[40,80,120,160,200]
MODEL=ROOT/'revision_experiments/progressive_agent_final_20261109/selected_models/full_int16_table.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def main():
    OUT.mkdir(exist_ok=True)
    for name in ['inputs','runs','logs']:(OUT/name).mkdir(exist_ok=True)
    handle=(OUT/'matrix.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups,topology=cpu_groups();groups=groups[:12]
    inputs={}
    for w,folder in [('ycsb','agent_fixed64_scaling_20261109'),('tpcc','agent_tpcc128_scaling_20261109')]:
        original=ROOT/'revision_experiments'/folder/'inputs'/f'{w}_main.json';target=OUT/'inputs'/f'{w}_main.json'
        if not target.exists():shutil.copy2(original,target)
        assert sha(original)==sha(target);inputs[w]=target
    protocol=dict(workers=WORKERS,ccs=CCS,expected_cells=110,main_tasks=240,model_sha256=sha(MODEL),
        input_sha256={w:sha(p) for w,p in inputs.items()},shared_objects=dict(ycsb=64,tpcc=128),
        max_attempts=6,max_retries=5,lock_timeout_s=120.,full_active_commit_grace_ms=50.,full_single_write_grace_ms=4500.,
        per_cell_physical_cores=2,per_cell_logical_cpus=4,worker_thread_budget=2500,groups=groups,topology=topology,
        scope='Fresh measurements for all110cells;only retry cap changed from32attempts to6attempts=5retries;no retraining,no input edits,no smoothing',
        source_sha256=sha(ROOT/'scripts/revision/run_agent_main_matrix_20261109.py'),new_api_calls=0)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    pending=[(w,n,c) for n in WORKERS for w in ['ycsb','tpcc'] for c in CCS];records=[];active={}
    while pending or active:
        for slot,cpus in enumerate(groups):
            if slot in active or not pending:continue
            if sum(r['workers'] for _,_,r in active.values())+pending[0][1]>2500:continue
            w,n,c=pending.pop(0);name=f'{w}_{n}_{c}';folder=OUT/'runs'/name
            if (folder/'result.json').exists():
                r=read(folder/'result.json');assert r['values_correct'] and not r['unexpected_errors'] and r['input_sha256']==sha(inputs[w]) and r['lock_timeout_s']==120. and r['max_attempts']==6
                records.append(dict(case=name,status='complete',cached=True,exit_code=0));continue
            if folder.exists():raise FileExistsError('Preserve incomplete output: '+str(folder))
            log=(OUT/'logs'/f'{name}.log').open('w')
            command=['taskset','-c',','.join(map(str,cpus)),sys.executable,str(ROOT/'scripts/revision/run_agent_main_matrix_20261109.py'),
                '--input',str(inputs[w]),'--output',str(folder),'--model',str(MODEL),'--cc',c,'--workers',str(n),
                '--attempts','6','--lock-timeout-s','120','--commit-grace-ms','50']
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            record=dict(case=name,workload=w,workers=n,cc=c,status='running',pid=child.pid,cpus=cpus,started=time.time())
            records.append(record);active[slot]=(child,log,record)
            print('START',name,flush=True)
        for slot,(child,log,r) in list(active.items()):
            if child.poll() is None:continue
            log.close();r.update(exit_code=child.returncode,ended=time.time(),status='complete' if child.returncode==0 else 'failed');del active[slot]
            print('END',r['case'],child.returncode,flush=True)
        save(OUT/'status.json',dict(expected=110,pending=len(pending),running=len(active),records=records))
        if pending or active:time.sleep(3)
    save(OUT/'completion.json',dict(complete=len(records)==110 and all(r['exit_code']==0 for r in records),expected=110,records=records))

if __name__=='__main__':main()
