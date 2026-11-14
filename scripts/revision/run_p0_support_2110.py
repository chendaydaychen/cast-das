#!/usr/bin/env python3
"""One prospectively fixed controlled-long-read follow-up; preserve every result."""
import argparse,dataclasses,json,math,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agent.cc.atcc.ppo import (DiscretePPOPolicy,DiscretePPOTrainer,PPOConfig,state_features,
    discounted_returns,excluded_policy_transition,policy_transition_exclusion_reason,valid_actions)
from agent.runtime import apply_paper_rewards,PaperRewardConfig
from agent.runtime.context import LockAction,LockClass
from agent.runtime.paper_policy import CompiledPhasePolicy,HotRulePolicy,StaticThresholdPhasePolicy
from agent.runtime.exact_group_lookup import ExactGroupLookup
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.workloads.credit_review import CreditReviewConfig
from scripts.revision.run_p0_actor_sampling import read_transitions
from scripts.revision.run_p0_2356 import PhaseProgressPolicy,without_phase_progress
from scripts.revision.run_p0_revision import dump,digest,csv_write,MeteredPolicy
from scripts.analysis.audit_p0_claims import compare,validate_rows
from scripts.unified_trace import run_credit_review_experiment as engine

FLAGS=dict(unique_attempt_outcomes=True,carry_prior_restart_cost=True,
    helper_externality_accounting=True,round_interval_features=True,
    consistent_hotness_classification=True,observed_write_read_lock_elision=True,
    protection_relevant_hotness=True,coordinated_hotspot_admission=True)
VARIANTS=('reference_full','full','nophase_label','nophase_progress',
          'static_threshold','hotrule_always','hotrule_threshold','linear_return')


def solve(matrix,rhs):
    a=[list(row)+[b] for row,b in zip(matrix,rhs)];n=len(a)
    for i in range(n):
        p=max(range(i,n),key=lambda k:abs(a[k][i]));a[i],a[p]=a[p],a[i]
        assert abs(a[i][i])>1e-12
        scale=a[i][i];a[i]=[v/scale for v in a[i]]
        for k in range(n):
            if k==i:continue
            scale=a[k][i]
            if scale:a[k]=[x-scale*y for x,y in zip(a[k],a[i])]
    return [row[-1] for row in a]


class LinearReturn:
    protection_relevant_hotness=True
    def __init__(self,data):self.data=data
    def select(self,s):
        f=state_features(s);valid=set(valid_actions(s.current_action))
        candidates=[int(a) for a in self.data['weights'] if int(a) in valid]
        if not candidates:return LockAction(LockClass(int(s.current_action)))
        action=min(candidates,key=lambda a:(-sum(w*x for w,x in zip(self.data['weights'][str(a)],f)),a))
        return LockAction(LockClass(action))


def fit_linear(rows):
    groups=defaultdict(list)
    for t,y in zip(rows,discounted_returns(rows,.99)):
        if not excluded_policy_transition(t):groups[t.action].append((state_features(t.state),y))
    weights={};counts={}
    for action,samples in groups.items():
        if len(samples)<8:continue
        n=len(samples[0][0]);matrix=[[float(i==j) for j in range(n)] for i in range(n)];rhs=[0.]*n
        # Fixed ridge=1, per-action mean-square target, no test fitting or tuning.
        for x,y in samples:
            for i,xi in enumerate(x):
                rhs[i]+=xi*y/len(samples)
                for j,xj in enumerate(x):matrix[i][j]+=xi*xj/len(samples)
        weights[str(action)]=solve(matrix,rhs);counts[str(action)]=len(samples)
    return dict(type='independent linear ridge discounted-return scorer',ridge=1.,discount=.99,
        weights=weights,action_samples=counts,minimum_global_action_samples=8,
        group_bias=False,features=37,test_tuning=False,full_actor_reused=False,
        candidate_space='all observed qualified complete masks compatible with current mask')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--behavior',type=Path,required=True)
    parser.add_argument('--original-models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','measure'),required=True)
    args=parser.parse_args();assert platform.node().split('.')[0]=='node1'
    if args.stage=='train':
        assert not args.output.exists()
        paths=sorted(args.behavior.glob('behavior_*.json'));assert len(paths)==6
        old=json.loads((args.behavior/'full_report.json').read_text())
        dump(args.output/'protocol.json',dict(scope=__doc__,variants=VARIANTS,clients=16,repeats=2,
            seconds=1.5,warmup=.1,seeds=[21101600,21101601],order_seeds=[21111600,21111601],
            config=dataclasses.asdict(CreditReviewConfig(company_count=1000,mutable_read_before_reasoning=True)),
            flags=FLAGS,performance_guards=False,tracked_history_import=True,max_attempts=6,
            threshold=.05,threshold_source='previous independent validation; unchanged',
            hot_rule_invocation='before first observed hot access; same lock eligibility and deferred writes',
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'agent').rglob('*.py')},
            engine_sha256=digest(ROOT/'scripts/unified_trace/run_credit_review_experiment.py'),
            script_sha256=digest(Path(__file__)),native_sha256=digest(next(ROOT.glob('cast_core*.so'))),
            inputs_sha256={p.name:digest(p) for p in paths},seed=old['seed'],config_training=old['config'],
            targeted_scope_not_paper_default=True,no_default_model_replacement=True,real_llm=False,
            gate='mean main/window>=1.03,p99<=1.05,waste<=1,2/2 wins,fail0; publish all comparisons'))
        rows,reward=apply_paper_rewards(read_transitions(paths),PaperRewardConfig(**old['reward']['config']))
        audits=[]
        for name in ('full','nophase_label','nophase_progress'):
            projected=rows if name!='nophase_progress' else [dataclasses.replace(t,
                state=without_phase_progress(t.state),next_state=without_phase_progress(t.next_state)) for t in rows]
            assert all(bool(policy_transition_exclusion_reason(a))==bool(policy_transition_exclusion_reason(b))
                for a,b in zip(rows,projected))
            started=time.perf_counter()
            actor=DiscretePPOPolicy(seed=old['seed'],phase_awareness='enabled' if name=='full' else 'disabled',
                categorical_actor=True,ppo_supported_projection=True,local_qualified_action_support=True,
                local_context_groups=True,protection_relevant_hotness=True)
            training=DiscretePPOTrainer(PPOConfig(**old['config'])).train(actor,projected,calibration_transitions=projected)
            model=actor.compile(generation=4,medoids_per_group=4,preserve_calibrated_groups=True,statewise_supported_actor=True)
            target=args.output/f'training/{name}.json';dump(target,model.to_dict())
            if name!='nophase_progress':assert digest(target)==digest(args.behavior/('full.json' if name=='full' else 'nophase.json'))
            audits.append(dict(variant=name,fit_compile_seconds=time.perf_counter()-started,training=training,sha256=digest(target)))
        started=time.perf_counter();linear=fit_linear(rows);dump(args.output/'training/linear.json',linear)
        dump(args.output/'training/audit.json',dict(passed=True,original_full_and_label_byte_identical=True,
            transitions=len(rows),models=audits,linear_seconds=time.perf_counter()-started,
            reward=reward,original_negatives_retained=True))
        # Equivalent exact-index benchmark on original non-categorical 2051 heads.
        source_paths=sorted((args.original_models.parents[1]/'evidence_inputs/original_training').glob('behavior_*.json'))
        assert len(source_paths)==6
        states=[t.state for t in read_transitions(source_paths)];micro=[]
        for name in ('full','nophase_label','nophase_progress'):
            base=CompiledPhasePolicy.load(args.original_models/f'training/{name}.json')
            fast=ExactGroupLookup(base)
            observed=[without_phase_progress(s) for s in states] if name=='nophase_progress' else states
            assert all(base.select(s)==fast.select(s) for s in observed)
            costs={'reference':[],'compact':[]}
            for rep in range(5):
                order=list(costs);random.Random(21121000+rep).shuffle(order)
                for variant in order:
                    policy=base if variant=='reference' else fast;start=time.perf_counter_ns()
                    for s in observed:policy.select(s)
                    costs[variant].append((time.perf_counter_ns()-start)/len(observed)/1000)
            micro.append(dict(variant=name,states=len(states),action_disagreements=0,batches_us=costs,
                reference_us=statistics.fmean(costs['reference']),compact_us=statistics.fmean(costs['compact']),
                scope='exact supported group indexing; unknown fallback unchanged; not system TPS'))
        dump(args.output/'micro.json',micro);print(json.dumps(micro),flush=True);return
    plan=json.loads((args.output/'protocol.json').read_text());assert not (args.output/'measure').exists()
    assert plan['source_sha256']=={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'agent').rglob('*.py')}
    assert plan['script_sha256']==digest(Path(__file__))
    rows=[];original_manager=engine.AgentTransactionManager;managers=[];current=''
    def factory(*a,**kw):
        m=original_manager(*a,**kw);m.first_hot_access_events=[]
        if current.startswith('hotrule'):m.interceptor.hooks=FirstHotAccessHooks(m)
        managers.append(m);return m
    engine.AgentTransactionManager=factory
    try:
        for rep in range(2):
            order=list(VARIANTS);random.Random(21111600+rep).shuffle(order)
            for current in order:
                if current.startswith('hotrule'):policy=HotRulePolicy(None if current=='hotrule_always' else .05)
                elif current=='static_threshold':policy=StaticThresholdPhasePolicy(.05,5)
                elif current=='linear_return':policy=LinearReturn(json.loads((args.output/'training/linear.json').read_text()))
                else:
                    name='full' if current in ('full','reference_full') else current
                    base=CompiledPhasePolicy.load(args.output/f'training/{name}.json');base.exact_categorical_lookup=True
                    policy=base if current=='reference_full' else ExactGroupLookup(base)
                    if current=='nophase_progress':policy=PhaseProgressPolicy(policy)
                meter=MeteredPolicy(policy);managers.clear()
                row=engine.run_system(config=CreditReviewConfig(company_count=1000,mutable_read_before_reasoning=True),
                    clients=16,seed=21101600+rep,repeat=rep,system='paper-atcc',paper_policy=args.output/'training/full.json',
                    policy_override=meter,measure_seconds=1.5,warmup_seconds=.1,max_attempts=6,
                    performance_guards=False,tracked_history_import=True,policy_commit_batch=True,**FLAGS)
                row.update(variant=current,case='controlled_long_read',policy_calls=meter.calls,
                    cumulative_policy_lookup_ms=meter.ns/1e6,selected_changes=meter.changes,
                    phase_actions_json=json.dumps({f'{p}:{a}':n for (p,a),n in meter.phase_actions.items()}))
                assert not validate_rows([row]);rows.append(row);csv_write(args.output/'measure/raw.csv',rows)
                dump(args.output/f'measure/events/{rep}_{current}.json',[e for m in managers for e in m.first_hot_access_events])
                print(json.dumps({k:row[k] for k in ('variant','repeat','elapsed_task_tps','agent_tps','p99_latency_ms','failed_tasks')}),flush=True)
    finally:engine.AgentTransactionManager=original_manager
    comparisons=[c for v in VARIANTS if v!='full' for c in compare(rows,'case','full',v)]
    for c in comparisons:
        c['direction_screen']=c['tps_ratio']>=1.03 and c['window_tps_ratio']>=1.03 and c['p99_ratio']<=1.05 and c['paired_wins']==2 and not c['failed_tasks_full'] and not c['failed_tasks_baseline']
    dump(args.output/'measure/completed.json',dict(rows=len(rows),comparisons=comparisons,
        raw_sha256=digest(args.output/'measure/raw.csv'),goal_completed=False,significance_claimed=False))
    print(json.dumps(comparisons),flush=True)

if __name__=='__main__':main()
