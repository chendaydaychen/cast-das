"""Frozen Full/NoPhase plus simple reviewer rules under paired contention."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['full','nophase','linear','hot_threshold','hot_always','static'],required=True)
    p.add_argument('--workload',choices=['ycsb','tpcc'],required=True)
    p.add_argument('--conflict',choices=['high','low'],required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--tasks',type=int,default=64)
    p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.seed=61113400;a.model=None
    from scripts.revision.run_phase_business_variant_20261113 import run
    from agent.runtime.context import LockAction
    from agent.runtime.paper_policy import LocalAccessFrequencyPolicy
    from agent.cc.atcc.ppo import state_features
    selected=ROOT/'revision_experiments/phase_branch_repair_20261113'
    if a.arm in ['full','nophase']:
        a.model=selected/f'train_{a.arm}/table.json'
    policy=None
    definition=None
    if a.arm in ['hot_always','hot_threshold']:
        cutoff=1 if a.arm=='hot_always' else 8
        policy=LocalAccessFrequencyPolicy(cutoff)
        definition=dict(kind='per-key one-second observed access frequency',minimum_accesses=cutoff,
                        mask='read protection;write protection when writes observed',phase_used=False)
    elif a.arm=='static':
        class Static:
            def select(self,state):return LockAction.from_encoded(int(state.current_action)|12)
        policy=Static();definition=dict(kind='fixed write protection mask',mask=12,phase_used=False)
    elif a.arm=='linear':
        # Fit a single ridge score per action from existing training executions.
        # No evaluation rows or hand-coded phase conditions enter the fit.
        from agent.runtime.state_collector import phase_aware_state_from_dict
        training=selected/'train_full';samples={};task_count=0
        for folder in sorted(training.glob('batch_*'),key=lambda f:int(f.name.split('_')[-1])):
            tasks=json.loads((folder/'tasks.json').read_text())
            costs={r['task']:r['pair_cycle_s']+60*(not r['committed']) for r in tasks}
            task_count+=len(tasks)
            for q in json.loads((folder/'decisions.json').read_text()):
                state=dataclasses.replace(phase_aware_state_from_dict(q['state']),phase='all')
                samples.setdefault(q['action'],[]).append((state_features(state),-costs[q['task']]))
        weights={}
        for action,rows in samples.items():
            dim=len(rows[0][0]);matrix=[[float(i==j) for j in range(dim)] for i in range(dim)];rhs=[0.]*dim
            for features,y in rows:
                for i,x in enumerate(features):
                    rhs[i]+=x*y
                    for j,v in enumerate(features):matrix[i][j]+=x*v
            augmented=[row+[y] for row,y in zip(matrix,rhs)]
            for i in range(dim):
                pivot=max(range(i,dim),key=lambda j:abs(augmented[j][i]))
                augmented[i],augmented[pivot]=augmented[pivot],augmented[i]
                divisor=augmented[i][i];assert abs(divisor)>1e-12
                augmented[i]=[x/divisor for x in augmented[i]]
                for j in range(dim):
                    if j!=i:
                        scale=augmented[j][i]
                        augmented[j]=[x-scale*y for x,y in zip(augmented[j],augmented[i])]
            weights[action]=[row[-1] for row in augmented]
        class Linear:
            def select(self,state):
                state=dataclasses.replace(state,phase='all');f=state_features(state);current=int(state.current_action)
                allowed=[k for k in weights if k|current==k]
                chosen=min(allowed,key=lambda k:(-sum(w*v for w,v in zip(weights[k],f)),k!=current,k)) if allowed else current
                return LockAction.from_encoded(chosen)
        policy=Linear();definition=dict(kind='fixed per-action linear score',ridge=1.,training_tasks=task_count,
            training_source=str(training.relative_to(ROOT)),phase_used=False,test_fit=False,
            weights={str(k):v for k,v in weights.items()})
    result,_,_=run(a,policy_override=policy)
    (a.output/'rule_protocol.json').write_text(json.dumps(dict(arm=a.arm,definition=definition,
        model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest() if a.model else None,
        no_retraining=True,no_test_selection=True,no_api_calls=True),indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
