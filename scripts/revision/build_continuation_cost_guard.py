"""Build a training-only restart-cost bound and exact additive policy tables."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.runtime.feature_score_table import FeatureScoreTable
from agent.runtime.joint_mask_policy import JointMaskTablePolicy
from agent.runtime.continuation_cost_guard import ContinuationCostGuard
from scripts.revision.run_current_credit_ablations import read,sha


def training_bound(training):
    plan=read(training/'protocol.json');costs=[];sources={}
    for arm in ('full','nophase'):
        for update in range(plan['updates_per_arm']):
            path=training/f'{arm}_update_{update}.json';sources[str(path)]=sha(path)
            costs.extend(float(r['state']['observed_restart_cost_ms']) for r in read(path)['transitions'])
    if not costs or any(not math.isfinite(v) or v<0 for v in costs):raise ValueError('Invalid training restart costs')
    maximum=max(costs)
    return math.ceil(1.1*maximum/100)*100,dict(maximum_training_cost_ms=maximum,
        rule='Common both-arm maximum, add fixed 10 percent timing margin, round upward to 100ms',
        training_queries=len(costs),sources_sha256=sources)


def load_guarded(reference,folder):
    plan=read(folder/'guard.json')
    if sha(folder/'runner_executed.py')!=plan['source_sha256'] or sha(ROOT/'agent/runtime/continuation_cost_guard.py')!=plan['implementation_sha256']:
        raise ValueError('Guard source/implementation changed')
    if Path(plan['child_reference'])!=reference:raise ValueError('Child reference differs')
    parent=Path(plan['parent_reference']);limit,proof=training_bound(reference)
    if limit!=plan['maximum_supported_cost_ms'] or proof!=plan['derivation']:raise ValueError('Training-only bound changed')
    deployed={};actors={}
    for arm in ('full','nophase'):
        children=[];originals=[]
        for role,root in (('child',reference),('parent',parent)):
            source=root/f'training_view/{arm}_policy.json'
            if sha(source)!=plan['actor_sha256'][f'{role}_{arm}']:raise ValueError('Source actor changed')
            path=folder/f'{role}_{arm}_feature_table.json'
            if sha(path)!=plan['table_sha256'][f'{role}_{arm}']:raise ValueError('Feature table changed')
            artifact=read(path)
            if artifact!=FeatureScoreTable.compile(read(source)).to_dict():raise ValueError('Table differs from frozen actor')
            children.append(FeatureScoreTable(artifact));originals.append(JointMaskTablePolicy.from_dict(read(source)))
        deployed[arm]=ContinuationCostGuard(*children,limit);actors[arm]=ContinuationCostGuard(*originals,limit)
    return deployed,actors,plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    training=read(args.training/'protocol.json');parent=Path(training['reference'])
    if not read(args.training/'serialized_training_audit.json')['passed']:raise ValueError('Training audit required')
    bound,proof=training_bound(args.training)
    args.output.mkdir(parents=True,exist_ok=False);actors={};tables={}
    for arm in ('full','nophase'):
        for role,root in (('child',args.training),('parent',parent)):
            source=root/f'training_view/{arm}_policy.json';table=FeatureScoreTable.compile(read(source))
            path=args.output/f'{role}_{arm}_feature_table.json';path.write_text(json.dumps(table.to_dict(),indent=2))
            actors[f'{role}_{arm}']=sha(source);tables[f'{role}_{arm}']=sha(path)
    plan=dict(child_reference=str(args.training),parent_reference=str(parent),maximum_supported_cost_ms=bound,
        derivation=proof,actor_sha256=actors,table_sha256=tables,test_states_used_to_derive_bound=False,
        scope='Opt-in observed-cost fallback to the frozen pre-continuation actor; not original medoid deployment, not a universal OOD guarantee',
        feature_table_representation='Separate exact additive contribution lookup for both actors; no medoid or memory compression claim',
        source_sha256=sha(Path(__file__)),implementation_sha256=sha(ROOT/'agent/runtime/continuation_cost_guard.py'))
    (args.output/'guard.json').write_text(json.dumps(plan,indent=2));(args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    load_guarded(args.training,args.output)
    print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
