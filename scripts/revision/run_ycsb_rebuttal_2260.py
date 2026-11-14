#!/usr/bin/env python3
"""Matched original YCSB-High trace; opt-in study, retains every result."""
import argparse, contextlib, dataclasses, functools, json, platform, random, shutil, sys, threading, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.revision.run_p0_revision import dump, digest, csv_write
from scripts.unified_trace import run_castdas_trace_fair as runner
from agent.benchmarks import mixed, phases
from agent.runtime import AgentTransactionManager, PaperRewardConfig, apply_paper_rewards
from agent.runtime.context import LockAction, LockClass
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy, LocalAccessFrequencyPolicy, LocalHotContentionPolicy, StaticThresholdPhasePolicy
from agent.runtime.paper_hooks import PaperATCCHooks
from agent.runtime.state_collector import policy_observation, phase_aware_state_from_dict
from agent.runtime.trajectory import policy_transition_from_dict
from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig, state_features, state_key, policy_group_key
from agent.cc.base import ConcurrencyControl
from scripts.revision.run_p0_contract_2200 import fit_linear

ORIGINAL=Path('/home/chenht/codex-runs/aegis-repro-20260722/results')
TRACE=ORIGINAL/'five_retry_ablation_steady/traces/ablation/ycsb_ablation_Dynamic___DWA___Priority_c32_r0_Dynamic___DWA___Priority.csv'
ARMS=[('static',False,False),('static_delay',True,False),('static_delay_priority',True,True),
      ('dynamic',False,False),('dynamic_delay',True,False),('full',True,True),('full_deferred',True,True),
      ('nophase',True,True),('nophase_deferred',True,True),('linear',True,True),('hot_always',True,True),('local_hot_threshold',True,True),('local_frequency',True,True),
      ('hot_threshold',True,True),('silo',False,False),('2pl_wound_wait',False,False),('paper_reference',True,True)]
LOCAL_HOT_THRESHOLD = 0.20
LOCAL_FREQUENCY_MIN_ACCESS = 8

class RawLinear:
    """Same raw feature vector and monotonic masks, no hand-coded hot anchor."""
    requires_reviewer_contention_statistics=False

    def __init__(self,data,*,monotone_actions=True):
        self.weights=data['weights'];self.monotone_actions=monotone_actions
    def select(self,s):
        f=state_features(s); current=int(s.current_action)
        allowed=[int(a) for a in self.weights if not self.monotone_actions or ((int(a)&0xF)|current)==(int(a)&0xF)]
        a=min(allowed,key=lambda a:(-sum(w*x for w,x in zip(self.weights[str(a)],f)),a!=(current),a)) if allowed else current
        return LockAction.from_encoded(a)

class AllProtection:
    def select(self,s): return LockAction(LockClass(15))

class ExclusiveClock:
    """Thread-local exclusive wall-time accounting (nested time never added twice)."""
    def __init__(self): self.local=threading.local(); self.rows=[]
    @contextlib.contextmanager
    def scope(self,category,root=False):
        st=getattr(self.local,'state',None)
        if st is None:
            if not root: yield; return
            st={'category':category,'last':time.perf_counter_ns(),'totals':defaultdict(int),'depth':0}
            self.local.state=st
        now=time.perf_counter_ns(); st['totals'][st['category']]+=now-st['last']
        previous=st['category']; st['category']=category; st['last']=now; st['depth']+=1
        try: yield
        finally:
            now=time.perf_counter_ns(); st['totals'][st['category']]+=now-st['last']; st['last']=now
            st['category']=previous; st['depth']-=1
            if root:
                self.rows.append(dict(st['totals'])); self.local.state=None
    def category(self):
        s=getattr(self.local,'state',None); return s['category'] if s else None
    def wrap(self,fn,category,root=False):
        @functools.wraps(fn)
        def wrapped(*a,**kw):
            with self.scope(category,root=root): return fn(*a,**kw)
        return wrapped

class TimedPolicy:
    def __init__(self,base,clock): self.base=base; self.clock=clock; self.reset()
    def reset(self): self.calls=self.wall_ns=self.cpu_ns=self.suggestions=0; self.states=[]; self.actions=Counter()
    def select(self,s):
        w=time.perf_counter_ns(); c=time.thread_time_ns()
        with self.clock.scope('lookup'): a=self.base.select(s)
        self.wall_ns+=time.perf_counter_ns()-w; self.cpu_ns+=time.thread_time_ns()-c; self.calls+=1
        action_code=int(getattr(a, 'encoded', int(a.protected)))
        self.suggestions+=action_code != int(s.current_action); self.actions[(s.phase,int(s.current_action),action_code)]+=1
        if len(self.states)<1000: self.states.append(dataclasses.asdict(s))
        return a

def train(out):
    assert not out.exists(); out.mkdir(parents=True)
    shutil.copy2(TRACE,out/'trace.csv'); shutil.copy2(TRACE.with_suffix('.meta.json'),out/'trace.meta.json')
    shutil.copy2(ORIGINAL/'paper_scale_train/paper_policy.json',out/'paper_reference.json')
    report=json.loads((ORIGINAL/'paper_scale_train/train_report.json').read_text())
    paths=sorted((ORIGINAL/'paper_scale_train/trajectories').glob('ycsb_high*.json'))
    raw=[]
    for f in paths:
        shutil.copy2(f,out/f'training_{f.name}')
        p=json.loads(f.read_text()); raw.extend(policy_transition_from_dict(t,source_id=f.stem) for t in p['transitions'])
    rows,reward=apply_paper_rewards(raw,PaperRewardConfig()); timings=[]; models={}
    cfg=PPOConfig(**report['config'])
    for name,awareness in [('full','enabled'),('nophase','disabled')]:
        start=time.perf_counter(); actor=DiscretePPOPolicy(seed=810104,phase_awareness=awareness)
        fit=DiscretePPOTrainer(cfg).train(actor,rows); fitted=time.perf_counter()
        model=actor.compile(generation=2260,medoids_per_group=4,occ_cold_start_guard=True)
        dump(out/f'{name}.json',model.to_dict()); models[name]=model
        timings.append(dict(variant=name,ppo_fit_seconds=fitted-start,compile_serialize_seconds=time.perf_counter()-fitted,
            total_train_fit_compile_seconds=time.perf_counter()-start,report=fit,entries=len(model.entries)))
    checks=0
    for t in rows:
        outputs=[]
        for label in ['explore','refine','commit']:
            s=dataclasses.replace(t.state,phase=label); obs=policy_observation(s,'disabled')
            outputs.append((state_key(obs),policy_group_key(obs),state_features(obs),int(models['nophase'].select(s).protected)))
        assert outputs[0]==outputs[1]==outputs[2]; checks+=1
    # Fixed ridge: no test-driven selection. Same behavior rows as learned models.
    start=time.perf_counter(); linear=fit_linear(rows,.01); dump(out/'linear.json',linear)
    dump(out/'training.json',dict(models=timings,linear_total_seconds=time.perf_counter()-start,
        reward=reward,inputs=[str(p) for p in paths],transitions=len(rows),phase_invariance_checks=checks,
        no_phase_scope='explicit label only; actual cost/access/progress remain',data_collection_reused=True,
        data_collection_not_in_fit_total=True,seed=810104,ppo_config=dataclasses.asdict(cfg)))
    dump(out/'protocol.json',dict(scope=__doc__,arms=ARMS,trace_sha256=digest(out/'trace.csv'),
        model_sha256={n:digest(out/f'{n}.json') for n in ['full','nophase','linear','paper_reference']},
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'agent').rglob('*.py')},
        runner_sha256=digest(Path(runner.__file__)),script_sha256=digest(Path(__file__)),
        clients=32,workers=32,seconds=6.,warmup_seconds=1.,max_attempts=6,repeats=1,
        default_changed=False,all_results_retained=True,real_llm=False,
        shared_fixes=['first unobserved read refresh to match paper online read semantics',
                     'stable hotness classification','refresh accumulated actual priority at existing checkpoints'],
        baseline_note='Silo uses shared runtime native versioned KV; 2PL WW uses all-access locks with timestamp-only scheduling, not external DBx1000 executables',
        static_note='original Static threshold 0.20, hot-write mask=4 at startup; not frozen Full',
        generalization_limit='label independence is not proof of arbitrary access-pattern or real-agent generalization'))
    print('TRAIN_COMPLETE',json.dumps([{k:v for k,v in t.items() if k!='report'} for t in timings]),flush=True)

class StoreProxy:
    def __init__(self,base,clock): self.base=base; self.clock=clock
    def __getattr__(self,name):
        fn=getattr(self.base,name)
        if name=='validate_versions': return self.clock.wrap(fn,'validation')
        if name=='batch_put_if_version': return self.clock.wrap(fn,'atomic_validation_publish')
        return fn

def measure(out,selected,profile=False,deferred=True,tag='main',cost_aware_observed_read_protection=False,max_attempts=6,matched_v4=False):
    clock=ExclusiveClock(); created=[]; holder={}; events=[]; changes=[]; priority_audit=[]
    class MeteredManager(AgentTransactionManager):
        def __init__(self,*a,**kw):
            if holder.get('override') is not None: kw['paper_policy']=holder['override']
            kw.update(first_access_read_refresh=True,consistent_hotness_classification=True)
            if holder['variant']=='2pl_wound_wait': kw.update(low_conflict_occ_guard=False,performance_guards_enabled=False)
            super().__init__(*a,**kw); self.study_start_ns=time.monotonic_ns(); self.store=StoreProxy(self.store,clock); self.version_manager.store=self.store
            if holder['variant']=='2pl_wound_wait': self.paper_static_initial_mask=15
            created.append(self)
        def reset_measurement_diagnostics(self):
            super().reset_measurement_diagnostics(); clock.rows.clear(); events.clear(); changes.clear(); priority_audit.clear()
            if holder.get('override'): holder['override'].reset()
        def refresh_atcc_priority(self,txn):
            if holder['variant']=='2pl_wound_wait':
                # Strict age-only ordering, independent of any completed cost.
                age_priority=1000000000000000000-(txn.context.start_ts_ns-self.study_start_ns)
                self.atcc_locks.update_priority(txn.context,age_priority)
            else: return super().refresh_atcc_priority(txn)
        def begin(self,*a,**kw):
            if holder['variant']=='2pl_wound_wait': self.paper_static_initial_mask=15
            txn=super().begin(*a,**kw)
            if holder['variant']=='2pl_wound_wait': assert int(txn.context.action.protected)==15
            return txn
        def transition_atcc_action(self,txn,action,**kw):
            before=int(txn.context.action.protected); start=time.perf_counter_ns(); cpu=time.thread_time_ns()
            try:
                with clock.scope('change'): return super().transition_atcc_action(txn,action,**kw)
            finally: changes.append(dict(tid=txn.context.tid,before=before,after=int(txn.context.action.protected),wall_ns=time.perf_counter_ns()-start,cpu_ns=time.thread_time_ns()-cpu))
    patches=[]
    def patch(obj,name,fn): patches.append((obj,name,getattr(obj,name))); setattr(obj,name,fn)
    patch(runner,'AgentTransactionManager',MeteredManager)
    original_select=PaperATCCHooks._select_and_apply
    def select(hook,txn,s):
        hook.manager.refresh_atcc_priority(txn)
        if holder['variant']=='2pl_wound_wait':
            # Fixed 2PL has no learning/table lookup at checkpoints.
            return hook._finish_online_observed_prefix(txn)
        if holder['variant']!='2pl_wound_wait':
            priority_audit.append(dict(tid=txn.context.tid,prior=txn.context.prior_retry_cost_ms,
                actual_agent_ms=txn.context.agent_cost_ms,priority=txn.context.priority))
        before=int(txn.context.action.protected)
        try: return original_select(hook,txn,s)
        finally:
            if len(events)<1500: events.append(dict(tid=txn.context.tid,state=dataclasses.asdict(s),before=before,
                after=int(txn.context.action.protected),priority=txn.context.priority,
                read_locks=sorted(txn.context.held_read_locks),write_locks=sorted(txn.context.held_write_locks)))
    patch(PaperATCCHooks,'_select_and_apply',select)
    if profile:
        patch(runner,'run_agent_row',clock.wrap(runner.run_agent_row,'other',root=True))
        original_sleep=time.sleep
        def sleep(seconds):
            category='reasoning' if clock.category()=='reasoning' else 'waiting'
            with clock.scope(category): return original_sleep(seconds)
        patch(time,'sleep',sleep)
        patch(threading.Condition,'wait',clock.wrap(threading.Condition.wait,'waiting'))
        for obj in [runner,phases]:
            if hasattr(obj,'sleep_for_reasoning'): patch(obj,'sleep_for_reasoning',clock.wrap(obj.sleep_for_reasoning,'reasoning'))
        raw_mixed_sleep=mixed.sleep_for_reasoning
        def mixed_sleep(delay_ms):
            left=getattr(clock.local,'attempt_prefix',0)
            if left: clock.local.attempt_prefix=left-1
            with clock.scope('waiting' if left else 'reasoning'): return raw_mixed_sleep(delay_ms)
        patch(mixed,'sleep_for_reasoning',mixed_sleep)
        raw_attempt=runner.run_agent_attempt
        def attempt(*a,**kw):
            clock.local.attempt_prefix=2
            try: return raw_attempt(*a,**kw)
            finally: clock.local.attempt_prefix=0
        patch(runner,'run_agent_attempt',attempt)
        patch(ConcurrencyControl,'validate',clock.wrap(ConcurrencyControl.validate,'validation'))
    rows=runner.read_trace(out/'trace.csv'); result=[json.loads(f.read_text())['row'] for f in sorted((out/tag).glob('*.json'))]
    try:
        order=[x for x in ARMS if x[0] in selected and not (out/f'{tag}/{x[0]}.json').exists()]; random.Random(22602000).shuffle(order)
        dump(out/f'{tag}_order_{len(result)}.json',order)
        for name,delay,priority in order:
            assert not (out/f'{tag}/{name}.json').exists()
            clock.rows.clear(); created.clear(); changes.clear(); events.clear(); priority_audit.clear(); holder['variant']=name
            if name in ['silo']: base=None
            elif name=='2pl_wound_wait': base=AllProtection()
            elif name.startswith('static'): base=StaticThresholdPhasePolicy()
            elif name=='linear': base=RawLinear(json.loads((out/'linear.json').read_text()))
            elif name=='hot_always': base=HotRulePolicy()
            elif name=='hot_threshold': base=HotRulePolicy(.20)
            elif name=='local_hot_threshold': base=LocalHotContentionPolicy(LOCAL_HOT_THRESHOLD,8)
            elif name=='local_frequency': base=LocalAccessFrequencyPolicy(LOCAL_FREQUENCY_MIN_ACCESS)
            else: base=CompiledPhasePolicy.load(out/('nophase.json' if name in {'nophase', 'nophase_deferred'} else 'paper_reference.json' if name=='paper_reference' else 'full.json'))
            holder['override']=TimedPolicy(base,clock) if base else None
            r=runner.run_trace(rows,warmup_rows=None if matched_v4 else rows,cc='silo' if name=='silo' else 'paper-atcc',policy=None,
                paper_policy=None,policy_mode='eval',paper_priority='enabled' if priority or name=='2pl_wound_wait' else 'disabled',
                paper_switching='static' if name.startswith('static') else 'dynamic',
                paper_delayed_write_apply='enabled' if delay else 'disabled',paper_performance_guards='disabled',
                paper_defer_policy_protection_until_commit=(name in {'full_deferred', 'nophase_deferred'}) and not matched_v4,
                paper_deferred_replay_enabled=(deferred and name!='2pl_wound_wait') and not matched_v4,atcc_retry_cache_enabled=False,
                paper_cost_aware_observed_read_protection=bool(cost_aware_observed_read_protection),
                warmup_seconds=0. if matched_v4 else 1.,measure_seconds=0. if matched_v4 else 6.,max_attempts=int(max_attempts),tokens_per_operation=64)
            m=holder['override']; n=max(1,int(r['agent_completed_tasks'])+int(r['agent_failed_tasks']))
            r.update(variant=name,tag=tag,profiled=profile,driver_sha256=digest(Path(__file__)),actual_changes=sum(c['after']!=c['before'] for c in changes),
                transition_calls=len(changes),actual_changes_per_task=sum(c['after']!=c['before'] for c in changes)/n,
                change_total_cpu_ms=sum(c.get('cpu_ns',0) for c in changes)/1e6,
                change_mean_cpu_us=sum(c.get('cpu_ns',0) for c in changes)/max(1,len(changes))/1000,
                lookup_calls=m.calls if m else 0,lookup_mean_cpu_us=m.cpu_ns/max(1,m.calls)/1000 if m else 0,
                lookup_mean_wall_us=m.wall_ns/max(1,m.calls)/1000 if m else 0,
                lookup_total_cpu_ms=m.cpu_ns/1e6 if m else 0,lookup_total_wall_ms=m.wall_ns/1e6 if m else 0,
                suggestion_changes=m.suggestions if m else 0)
            if profile:
                totals={k:sum(t.get(k,0) for t in clock.rows)/1e6 for k in ['reasoning','waiting','lookup','change','validation','atomic_validation_publish','other']}
                r.update({f'breakdown_{k}_ms_per_task':v/n for k,v in totals.items()})
                r['profile_tasks']=len(clock.rows); assert len(clock.rows)==n
            assert r['status']=='ok'; assert int(r['agent_attempts'])==int(r['agent_commits'])+int(r['agent_aborts'])
            dump(out/f'{tag}/{name}.json',dict(row=r,decisions=events,transitions=changes,priority_ledger=priority_audit,
                policy_actions={str(k):v for k,v in m.actions.items()} if m else {},profile_tasks=clock.rows))
            result.append(r); csv_write(out/f'{tag}_raw.csv',result)
            print(json.dumps({k:r.get(k) for k in ['variant','agent_tps','agent_drain_task_tps','agent_p99_latency_ms','agent_failed_tasks','lookup_calls','actual_changes','lookup_mean_cpu_us']}),flush=True)
    finally:
        for obj,name,fn in reversed(patches): setattr(obj,name,fn)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--stage',choices=['train','main','profile','runtime'],required=True)
    p.add_argument('--arms',default=','.join(x[0] for x in ARMS)); p.add_argument('--matched-v4',action='store_true',help='use the same no-warmup/no-drain accounting as the three-regime v4 panel'); a=p.parse_args()
    assert platform.node().split('.')[0]=='node1'
    if a.stage=='train': train(a.output)
    else: measure(a.output,a.arms.split(','),profile=a.stage=='profile',deferred=a.stage!='runtime',tag=a.stage,matched_v4=a.matched_v4)
if __name__=='__main__': main()
