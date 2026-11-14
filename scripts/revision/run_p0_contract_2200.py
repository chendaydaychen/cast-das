#!/usr/bin/env python3
"""Prospective tiny native phase/CC experiment. JSON retains every attempt.

Two tasks share a root lookup and one initially-cold dependency. A background
writer pays 20ms before acquiring a native WLock, then waits for a 70ms external
response (blocked, not paid work). The foreground has an uncancellable 80ms
tool call in Refine and a 40ms suffix in Commit. Task logic and releases are
identical across policies. No future target, role label or planned time enters
the PE state. This is a mechanism workload, not paper-scale generalization.
"""
import argparse, dataclasses, json, os, platform, random, sys, threading, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agent.runtime import AgentTransactionManager, PaperRewardConfig, apply_paper_rewards
from agent.runtime.context import LockAction,LockClass,TransactionStatus
from agent.runtime.paper_hooks import PaperATCCHooks
from agent.runtime.hot_access_hooks import FirstHotAccessHooks
from agent.runtime.paper_policy import CompiledPhasePolicy,HotRulePolicy
from agent.runtime.trajectory import TrajectoryCollector,policy_transition_from_dict
from agent.cc.atcc.ppo import (DiscretePPOPolicy,DiscretePPOTrainer,PPOConfig,
    policy_group_key,state_features,excluded_policy_transition,discounted_returns)
from agent.cc.locks import LockConflict
from scripts.revision.run_p0_2356 import without_phase_progress
from scripts.revision.run_p0_support_2110 import solve
from scripts.revision.run_p0_revision import digest,dump


class WarmSwitch:
    def __init__(self,base):self.base=base;self.warm=True
    def select(self,s):return LockAction() if self.warm else self.base.select(s)
    def select_with_distribution(self,s):
        if self.warm:return LockAction(),(1.,)+(0.,)*15
        method=getattr(self.base,'select_with_distribution',None)
        if method:return method(s)
        a=self.base.select(s);return a,tuple(float(i==int(a.protected)) for i in range(16))


class Sampler:
    def __init__(self,seed):self.rng=random.Random(seed);self.draws=[]
    def select_with_distribution(self,s):
        current=int(s.current_action)
        if not s.read_set_size and not s.write_set_size:
            return LockAction(LockClass(current)),tuple(float(i==current) for i in range(16))
        if current==0 and s.hotspot_access_ratio==1 and s.read_set_size==1:
            return LockAction(LockClass(5)),tuple(float(i==5) for i in range(16))
        if current in (0,5):
            choices=(current,15);u=self.rng.random();a=choices[int(u>=.5)]
            self.draws.append(dict(state=dataclasses.asdict(s),u=u,choices=list(choices),selected=a))
            return LockAction(LockClass(a)),tuple(.5 if i in choices else 0. for i in range(16))
        return LockAction(LockClass(current)),tuple(float(i==current) for i in range(16))


class SupportedHeads:
    """Opt-in exact known heads, conservative unknown state, observed hot anchor.

    No medoid extrapolation. The anchor uses hotness/observed footprint only,
    is shared with the linear arm, and does not inspect phase or task identity.
    """
    def __init__(self,model,progress=False):self.model=model;self.progress=progress
    def select(self,s):
        if int(s.current_action)==0 and s.hotspot_access_ratio==1 and s.read_set_size==1:
            return LockAction(LockClass(5))
        if self.progress:s=without_phase_progress(s)
        if self.model.phase_awareness=='disabled':s=dataclasses.replace(s,phase='all')
        key=policy_group_key(s,local_context=self.model.local_context_groups,
                             growth_conditioned=self.model.growth_conditioned_groups)
        return LockAction(LockClass(self.model.calibrated_group_actions.get(key,int(s.current_action))))


class Linear:
    def __init__(self,data):self.data=data
    def select(self,s):
        if int(s.current_action)==0 and s.hotspot_access_ratio==1 and s.read_set_size==1:
            return LockAction(LockClass(5))
        f=state_features(s);current=int(s.current_action)
        allowed=[int(a) for a in self.data['weights'] if int(a)|current==int(a)]
        if not allowed:return LockAction(LockClass(current))
        a=min(allowed,key=lambda a:(-sum(w*x for w,x in zip(self.data['weights'][str(a)],f)),a!=current,a))
        return LockAction(LockClass(a))


class DecisionHooks(FirstHotAccessHooks):
    def _select_and_apply(self,txn,state):
        before=int(txn.context.action.protected)
        try:return super()._select_and_apply(txn,state)
        finally:self.manager.contract_decisions.append(dict(txn_id=txn.context.tid,
            state=dataclasses.asdict(state),before=before,applied=int(txn.context.action.protected),
            read_locks=sorted(txn.context.held_read_locks),write_locks=sorted(txn.context.held_write_locks),
            priority=txn.context.priority,status=txn.context.status.value,at_ns=time.monotonic_ns()))


def run_episode(policy,index,*,collect=False,stage_ms=80,suffix_ms=40,writer_ms=20,wait_ms=70):
    switch=WarmSwitch(policy)
    manager=AgentTransactionManager(paper_policy=switch,priority_enabled=True,
        low_conflict_occ_guard=False,performance_guards_enabled=False,
        delayed_write_apply_enabled=False,observed_write_read_lock_elision=False,
        round_interval_features=True,consistent_hotness_classification=True,
        collect_trajectories=collect,record_traces=True)
    manager.register_object('root','0',kind='row');manager.register_object('shared','0',kind='row')
    # Actual read-only history, built without a policy and outside measured work.
    for j in range(12):
        warm=manager.begin(f'warm:{index}:{j}',strategy='paper-atcc');warm.read('root')
        assert warm.commit('paper-atcc').committed
    manager.hotness_tracker.prime_accesses(['root']*12)
    assert manager.hotness_tracker.is_hot('root') and not manager.hotness_tracker.is_hot('shared')
    switch.warm=False;manager.trajectory_collector=TrajectoryCollector()
    manager.contract_decisions=[];manager.first_hot_access_events=[];manager.interceptor.hooks=DecisionHooks(manager)
    events={k:threading.Event() for k in ('bg_begun','fg_early','fg_first_done','bg_done')}
    lock=threading.Lock();attempts=[];tasks=[];errors=[]
    def record_attempt(row,txn,paid):
        row.update(txn_id=txn.context.tid,paid_actual_ms=paid,
            accounted_agent_ms=txn.context.agent_cost_ms,priority=txn.context.priority,
            blocked_ms=txn.context.blocked_time_ms,status=txn.context.status.value)
        with lock:attempts.append(row)
    def abort_if_live(txn,reason):
        if txn.context.status not in (TransactionStatus.ABORTED,TransactionStatus.COMMITTED):
            txn.abort(reason,strategy='paper-atcc')
    def foreground():
        assert events['bg_begun'].wait(2.)
        start=time.monotonic();committed=False
        for attempt in range(6):
            if attempt:assert events['bg_done'].wait(2.)
            txn=manager.begin(f'foreground:{index}',metadata={'retry_count':attempt},strategy='paper-atcc')
            paid=0.;row=dict(task='foreground',attempt=attempt,committed=False)
            try:
                txn.read('root');txn.enter_phase('explore')
                observed=txn.read('shared');row.update(observed_value=str(observed.value),observed_version=int(observed.version))
                txn.enter_phase('refine');row['early_mask']=int(txn.context.action.protected)
                row['early_read_locks']=sorted(txn.context.held_read_locks)
                if attempt==0:events['fg_early'].set()
                started=time.monotonic_ns();time.sleep(stage_ms/1000.);paid+=(time.monotonic_ns()-started)/1e6
                manager.interceptor.account_agent_interval(txn)
                if txn.context.status==TransactionStatus.ABORTED:raise LockConflict('cancelled during uncancellable tool call',('shared',))
                txn.enter_phase('commit');row['late_mask']=int(txn.context.action.protected)
                row['late_read_locks']=sorted(txn.context.held_read_locks)
                started=time.monotonic_ns();time.sleep(suffix_ms/1000.);paid+=(time.monotonic_ns()-started)/1e6
                manager.interceptor.account_agent_interval(txn)
                result=txn.commit('paper-atcc');committed=bool(result.committed)
                row.update(committed=committed,reason=str(result.reason))
            except LockConflict as exc:
                row['reason']=str(exc);abort_if_live(txn,str(exc))
            finally:
                record_attempt(row,txn,paid)
                if attempt==0:events['fg_first_done'].set()
            if committed:break
        with lock:tasks.append(dict(task='foreground',committed=committed,latency_ms=(time.monotonic()-start)*1000))
    def background():
        start=time.monotonic();committed=False
        for attempt in range(6):
            if attempt:assert events['fg_first_done'].wait(2.)
            txn=manager.begin(f'background:{index}',metadata={'retry_count':attempt,
                'runtime_background':True,'paper_atcc_backend':True},strategy='paper-atcc')
            if attempt==0:events['bg_begun'].set();assert events['fg_early'].wait(2.)
            paid=0.;row=dict(task='background',attempt=attempt,committed=False)
            try:
                started=time.monotonic_ns();time.sleep(writer_ms/1000.);paid+=(time.monotonic_ns()-started)/1e6
                manager.interceptor.account_agent_interval(txn);manager.refresh_atcc_priority(txn)
                manager.atcc_locks.wlock('shared',txn.context,timeout_s=1.)
                txn.write('shared','1')
                # A fixed external-response wait, not background computation.
                started=time.monotonic_ns();time.sleep(wait_ms/1000.)
                txn.context.blocked_time_ms+=(time.monotonic_ns()-started)/1e6
                manager.interceptor.reset_agent_interval(txn)
                if txn.context.status==TransactionStatus.ABORTED:raise LockConflict('cancelled pending external response',('shared',))
                result=txn.commit('paper-atcc');committed=bool(result.committed)
                row.update(committed=committed,reason=str(result.reason))
            except LockConflict as exc:
                row['reason']=str(exc);abort_if_live(txn,str(exc))
            finally:record_attempt(row,txn,paid)
            if committed:break
        events['bg_done'].set()
        with lock:tasks.append(dict(task='background',committed=committed,latency_ms=(time.monotonic()-start)*1000))
    def guarded(fn):
        try:fn()
        except BaseException as exc:
            with lock:errors.append(repr(exc))
            for e in events.values():e.set()
    started=time.monotonic();threads=[threading.Thread(target=guarded,args=(fn,)) for fn in (background,foreground)]
    for t in threads:t.start()
    for t in threads:t.join(3.)
    elapsed=time.monotonic()-started
    assert not any(t.is_alive() for t in threads) and not errors,errors
    assert len(tasks)==2 and all(t['committed'] for t in tasks),tasks
    assert manager.store.get('shared').value=='1'
    assert not manager.atcc_locks.global_waiter_count()
    assert not any(c.held_read_locks or c.held_write_locks for c in manager.atcc_locks._contexts.values())
    transitions=[dataclasses.asdict(t) for t in manager.trajectory_collector.snapshot()] if collect else []
    patched=[];by_tid={a['txn_id']:a for a in attempts}
    # Terminal outcome can arrive while an uncancellable call is still running.
    # Reconcile only real owner-thread paid time after both workers drain. This
    # is a training cost-accounting correction, never an input or fake outcome.
    for t in transitions:
        t['source_id']='contract_collection_2200'
        if t['done'] and t['txn_id'] in by_tid:
            actual=by_tid[t['txn_id']]['paid_actual_ms']
            if actual>t['agent_cost_ms']:
                patched.append(dict(txn_id=t['txn_id'],before=t['agent_cost_ms'],after=actual))
                t['agent_cost_ms']=actual
    foreground_task=next(t for t in tasks if t['task']=='foreground')
    row=dict(index=index,commits=2,failed_tasks=0,attempts=len(attempts),aborts=sum(not a['committed'] for a in attempts),
        elapsed_seconds=elapsed,task_tps=2/elapsed,max_task_latency_ms=max(t['latency_ms'] for t in tasks),
        foreground_latency_ms=foreground_task['latency_ms'],
        wasted_actual_ms_per_commit=sum(a['paid_actual_ms'] for a in attempts if not a['committed'])/2,
        policy_calls=len(manager.contract_decisions))
    return row,dict(row=row,tasks=tasks,attempts=attempts,decisions=manager.contract_decisions,
        transitions=transitions,terminal_paid_cost_reconciliation=patched,
        initial_hotness=dict(root=True,shared=False),final_hotness=manager.hotness_tracker.snapshot(),
        final_value=manager.store.get('shared').value,native=manager.atcc_locks.snapshot_diagnostics(),
        traces=manager.traces(),hot_access_events=manager.first_hot_access_events,
        sampler_draws=getattr(policy,'draws',[]))


class FixedSchedulePolicy:
    """Diagnostic mechanism only; NEVER used as a trained policy."""
    def __init__(self,early,late):self.early=early;self.late=late
    def select(self,s):
        if s.read_set_size==1 and s.hotspot_access_ratio==1:return LockAction(LockClass(5))
        return LockAction(LockClass(int(s.current_action)|(self.late if s.phase=='commit' else self.early)))


def fit_linear(rows,ridge):
    samples={}
    for t,y in zip(rows,discounted_returns(rows,.99)):
        if not excluded_policy_transition(t):samples.setdefault(t.action,[]).append((state_features(t.state),y))
    weights={}
    for a,data in samples.items():
        n=37;matrix=[[float(i==j)*ridge for j in range(n)] for i in range(n)];rhs=[0.]*n
        for f,y in data:
            for i,x in enumerate(f):
                rhs[i]+=x*y/len(data)
                for j,z in enumerate(f):matrix[i][j]+=x*z/len(data)
        weights[str(a)]=solve(matrix,rhs)
    return dict(weights=weights,ridge=ridge,discount=.99,features=37,independently_fitted=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('diagnostic','train','measure'),required=True)
    args=p.parse_args();assert platform.node().split('.')[0]=='node1'
    if args.mode=='diagnostic':
        assert not args.output.exists();all_rows=[]
        for name,policy in [('late',FixedSchedulePolicy(5,15)),('early',FixedSchedulePolicy(15,15)),('optimistic',HotRulePolicy())]:
            row,record=run_episode(policy,2200000+len(all_rows));row['variant']=name
            dump(args.output/f'{name}.json',record);all_rows.append(row);print(json.dumps(row),flush=True)
        dump(args.output/'rows.json',all_rows);return
    if args.mode=='train':
        assert not args.output.exists()
        dump(args.output/'protocol.json',dict(scope=__doc__,training_episodes=32,training_seeds=[22001000+i for i in range(32)],
            actor='original factorized four-bit linear scores + local state-group bias',epochs=4,min_group_samples=4,
            linear_ridge_candidates=[.001,.01,.1,1.],linear_validation_episodes=list(range(24,32)),
            holdout_cases=[dict(stage_ms=80,suffix_ms=40,writer_ms=20,wait_ms=70),
                           dict(stage_ms=100,suffix_ms=40,writer_ms=25,wait_ms=90)],
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'agent').rglob('*.py')},
            script_sha256=digest(Path(__file__)),native_sha256=digest(next(ROOT.glob('cast_core*.so'))),
            common_hooks='same first-eligible-hot-access and phase-entry hooks for all arms',
            phase_ablation='phase plus seven predeclared progress features, retrained, same rewards/actions/q',
            reward_formula_changed=False,cost_correction='terminal pending-call actual paid time reconciled after drain',
            architecture_changed=False,default_model_replaced=False,real_llm=False,conditional_mechanism_scope=True,
            repeats_per_holdout_case=1,all_results_retained=True))
        raw=[];records=[];started=time.monotonic()
        for i in range(32):
            sampler=Sampler(22001000+i);row,record=run_episode(sampler,22001000+i,collect=True)
            dump(args.output/f'collection/{i:02d}.json',record);records.append(record)
            raw.extend(policy_transition_from_dict(t) for t in record['transitions'])
            print(json.dumps(dict(collection=i,**row)),flush=True)
        collection_seconds=time.monotonic()-started
        training_raw=[policy_transition_from_dict(t) for r in records[:24] for t in r['transitions']]
        rows,reward=apply_paper_rewards(training_raw,PaperRewardConfig())
        reports=[]
        for name in ('full','nophase_progress'):
            projected=rows if name=='full' else [dataclasses.replace(t,state=without_phase_progress(t.state),next_state=without_phase_progress(t.next_state)) for t in rows]
            assert all(excluded_policy_transition(a)==excluded_policy_transition(b) for a,b in zip(rows,projected))
            for a,b in zip(rows,projected):
                assert all(getattr(a,f.name)==getattr(b,f.name) for f in dataclasses.fields(a) if f.name not in {'state','next_state'})
            started=time.monotonic();actor=DiscretePPOPolicy(seed=22003000,phase_awareness='enabled' if name=='full' else 'disabled',local_context_groups=True)
            report=DiscretePPOTrainer(PPOConfig(epochs=4,min_group_samples=4)).train(actor,projected)
            model=actor.compile(generation=1,medoids_per_group=4,preserve_calibrated_groups=True)
            dump(args.output/f'models/{name}.json',model.to_dict())
            reports.append(dict(variant=name,fit_compile_seconds=time.monotonic()-started,training=report,
                model_sha256=digest(args.output/f'models/{name}.json')))
        train_raw=[policy_transition_from_dict(t) for r in records[:24] for t in r['transitions']]
        validation_raw=[policy_transition_from_dict(t) for r in records[24:] for t in r['transitions']]
        # One normalization from the training split; no evaluation outcomes used.
        train,training_reward=apply_paper_rewards(train_raw,PaperRewardConfig())
        from agent.runtime.trajectory import paper_reward
        validation=[dataclasses.replace(t,reward=paper_reward(t,training_reward['normalization'],PaperRewardConfig())) for t in validation_raw]
        scores=[]
        for ridge in (.001,.01,.1,1.):
            data=fit_linear(train,ridge);policy=Linear(data);num=den=0.
            for t,y in zip(validation,discounted_returns(validation,.99)):
                if excluded_policy_transition(t):continue
                den+=1
                if int(policy.select(t.state).protected)==t.action:num+=y/t.behavior_probability
            scores.append(dict(ridge=ridge,validation_ips=num/max(1,den),data=data))
        best=max(scores,key=lambda r:(r['validation_ips'],-r['ridge']))
        dump(args.output/'models/linear.json',best['data'])
        dump(args.output/'training.json',dict(collection_seconds=collection_seconds,transitions=len(rows),reward=reward,
            models=reports,linear_validation=[{k:v for k,v in r.items() if k!='data'} for r in scores],
            selected_linear_ridge=best['ridge'],linear_training_episodes=24,linear_validation_episodes=8,
            full_training_episodes=24,matched_training_budget=True,validation_reserved_episodes=8,
            collection_transition_count=len(raw)))
        return
    plan=json.loads((args.output/'protocol.json').read_text());assert not (args.output/'holdout').exists()
    assert plan['script_sha256']==digest(Path(__file__))
    all_rows=[]
    for case_index,case in enumerate(plan['holdout_cases']):
        variants=['full','nophase_progress','hot_always','hot_threshold','linear'];random.Random(22004000+case_index).shuffle(variants)
        for v in variants:
            if v.startswith('hot_'):policy=HotRulePolicy(None if v=='hot_always' else .05)
            elif v=='linear':policy=Linear(json.loads((args.output/'models/linear.json').read_text()))
            else:policy=SupportedHeads(CompiledPhasePolicy.load(args.output/f'models/{v}.json'),v=='nophase_progress')
            row,record=run_episode(policy,22005000+case_index,**case);row.update(variant=v,case=case_index)
            dump(args.output/f'holdout/{case_index}_{v}.json',record);all_rows.append(row)
            print(json.dumps(row),flush=True)
    dump(args.output/'holdout/rows.json',all_rows)

if __name__=='__main__':main()
