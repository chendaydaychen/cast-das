"""Run all declared cells, keeping failures/results without result selection."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'revision_experiments/mentor_additions_20261114'

def main():
    OUT.mkdir(exist_ok=False)
    from scripts.revision.launch_agent_main_matrix_20261109 import cpu_groups
    groups,_=cpu_groups();slots=min(6,len(groups));jobs=[]
    for workers in [8,16,32]:
        for workload in ['ycsb','tpcc']:
            for conflict in ['high','low']:
                for arm in ['full','nophase','linear','hot_threshold','hot_always','static']:
                    name=f'ablation_{workload}_{conflict}_w{workers}_{arm}'
                    jobs.append((name,'scripts/revision/run_mentor_rule_ablation_20261114.py',
                        ['--arm',arm,'--workload',workload,'--conflict',conflict,'--workers',str(workers)]))
    for factor in ['reasoning','zipf','rw']:
        for arm in ['full','nophase']:
            jobs.append((f'continuous_{factor}_{arm}','scripts/revision/run_mentor_continuous_phase_20261114.py',
                         ['--arm',arm,'--factor',factor]))
    (OUT/'plan.json').write_text(json.dumps(dict(jobs=jobs,ablation_cells=72,continuous_cells=6,
        tasks=64,workers=[8,16,32],rules_frozen_before_eval=True,public_workloads='deferred by user',
        high_shared_pairs=64,low_shared_pairs=8,shared_pairs_balanced_across_phases=True),indent=2))
    def run_slot(slot):
        results=[]
        for name,script,args in jobs[slot::slots]:
            tick=time.monotonic()
            with (OUT/(name+'.log')).open('w') as f:
                try:
                    code=subprocess.call(['taskset','-c',','.join(map(str,groups[slot])),sys.executable,script,
                        *args,'--output',str(OUT/name)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                except Exception as exc:
                    f.write(repr(exc));code=-1
            row=dict(case=name,exit_code=code,elapsed_s=time.monotonic()-tick)
            (OUT/(name+'.status.json')).write_text(json.dumps(row));print(json.dumps(row),flush=True)
            results.append(row)
        return results
    with ThreadPoolExecutor(max_workers=slots) as pool:
        results=sum(pool.map(run_slot,range(slots)),[])
    (OUT/'completion.json').write_text(json.dumps(results,indent=2))
    assert all(r['exit_code']==0 for r in results),results

if __name__=='__main__':main()
