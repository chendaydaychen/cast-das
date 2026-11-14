"""Train-only phase recoverability diagnosis using the actual NoPhase features."""
import argparse
from collections import Counter
import dataclasses
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc.atcc.ppo import state_features
from agent.runtime.state_collector import phase_aware_state_from_dict,policy_observation
from scripts.revision.run_current_credit_ablations import read,sha

FEATURE_NAMES=('bias','phase_explore','phase_refine','phase_commit','read_set_size',
    'write_set_size','read_set_growth','write_set_growth','access_overlap','hotspot_ratio',
    'write_ratio','retry_count','conflict_none','conflict_version','conflict_preempted',
    'conflict_lock','current_hot_read','current_cold_read','current_hot_write','current_cold_write',
    'active_transactions','waiters','abort_rate','throughput','average_latency','tail_latency',
    'blocked_time','inter_round_interval','completed_rounds','completed_operations','restart_cost',
    'agent_throughput','agent_average_latency','agent_tail_latency','conflict_abort_rate',
    'background_throughput','background_abort_rate')


def features(raw):
    observed=policy_observation(phase_aware_state_from_dict(raw),'disabled')
    return state_features(dataclasses.replace(observed,completed_rounds=0))


def impurity(rows):
    counts=Counter(label for _,label in rows)
    return len(rows)-sum(c*c for c in counts.values())/len(rows) if rows else 0.


def fit(rows,depth=3,min_leaf=5):
    counts=Counter(label for _,label in rows)
    prediction=min(counts,key=lambda label:(-counts[label],label))
    result=dict(prediction=prediction,samples=len(rows),labels=dict(counts))
    if not depth or len(counts)==1:return result
    best=None;best_cost=impurity(rows)
    for feature in range(len(rows[0][0])):
        values=sorted({x[feature] for x,_ in rows})
        for low,high in zip(values,values[1:]):
            threshold=(low+high)/2.
            left=[r for r in rows if r[0][feature]<=threshold]
            right=[r for r in rows if r[0][feature]>threshold]
            if min(len(left),len(right))<min_leaf:continue
            cost=impurity(left)+impurity(right)
            if cost<best_cost-1e-10:
                best=(feature,threshold,left,right);best_cost=cost
    if best:
        index,threshold,left,right=best
        result.update(feature=index,feature_name=FEATURE_NAMES[index] if index<len(FEATURE_NAMES) else str(index),
            threshold=threshold,left=fit(left,depth-1,min_leaf),right=fit(right,depth-1,min_leaf))
    return result


def predict(tree,x):
    while 'feature' in tree:
        tree=tree['left' if x[tree['feature']]<=tree['threshold'] else 'right']
    return tree['prediction']


def score(tree,rows,majority):
    correct=sum(predict(tree,x)==label for x,label in rows)
    matrix=Counter((label,predict(tree,x)) for x,label in rows)
    return dict(samples=len(rows),correct=correct,accuracy=correct/len(rows),
        majority_accuracy=sum(label==majority for _,label in rows)/len(rows),
        confusion=[dict(actual=a,predicted=p,count=c) for (a,p),c in sorted(matrix.items())])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,nargs='+',required=True)
    p.add_argument('--evaluation',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];sources={}
    for folder in args.training:
        path=folder/'nophase_rollout.json';sources[str(path)]=sha(path)
        rows.extend((features(t['state']),t['state']['phase']) for t in read(path)['transitions'])
    if any(x[i]!=0 for x,_ in rows for i in (1,2,3,28)):
        raise ValueError('Explicit phase or removed round proxy leaked')
    tree=fit(rows,depth=3,min_leaf=5)
    majority=min(Counter(y for _,y in rows),key=lambda y:(-sum(v==y for _,v in rows),y))
    result=dict(training=score(tree,rows,majority),tree=tree,tests=[],
        protocol=dict(depth=3,min_leaf=5,source_arm='nophase only',features='Exact actor 37-vector, explicit phase and completed_rounds zero',
            models_changed=False,benchmark_rerun=False,test_used_to_fit=False))
    for folder in args.evaluation:
        for epoch in range(2):
            path=folder/'nophase'/f'epoch_{epoch}.json';sources[str(path)]=sha(path)
            test=[(features(d['state']),d['state']['phase']) for d in read(path)['decisions']]
            result['tests'].append(dict(folder=str(folder),epoch=epoch,**score(tree,test,majority)))
    sources[str(Path(__file__))]=sha(Path(__file__))
    result['source_sha256']=sources
    result['caveat']='Diagnostic predictability, not proof the PPO actor internally infers phase or gains from it; query states are correlated'
    (args.output/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
