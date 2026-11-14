"""Phase-sensitive business variants; not replacement of old main curves.

Explore can finish as a certified read-only branch; Refine updates its observed
value. Both compete with the same update-then-compute agent. No cost table.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import math
from pathlib import Path
import random
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
def read(p):return json.loads(p.read_text())
def p99(xs):return sorted(xs)[max(0,math.ceil(.99*len(xs))-1)]

def run(a,actor=None,sample=False,policy_override=None):
    from agent.runtime import AgentTransactionManager
    from agent.runtime.context import LockAction,TransactionPhase,TransactionStatus
    from agent.runtime.paper_contract import configure_paper_observed_contract
    from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
    from agent.runtime.read_only_snapshot_certificate_20261110 import install as certificate
    from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
    from agent.runtime.state_collector import phase_aware_state_from_dict
    from agent.workloads.frozen_credit_agent import tracked_agent_wait
    from scripts.revision.progressive_agent_environment_20261109 import OBSERVED_LLM
    rng=random.Random(a.seed);plan=[]
    for i in range(a.tasks):
        plan.append(dict(index=i,phase='explore' if i%2==0 else 'refine',
            competitor_shared=(getattr(a,'conflict','high')=='high' or i%16 in (0,1)),
            delay_s=rng.choice(OBSERVED_LLM['refine'])/1000*a.scale))
    rng.shuffle(plan)
    questions=[];changes=[];local=threading.local();guard=threading.Lock()
    table=QuantizedSharedBitTable(read(a.model)) if actor is None and a.model else None
    class Policy:
        phase_awareness='enabled' if a.arm in ['full','oracle','swapped'] else 'disabled'
        monotone_actions=True
        def select(self,state):
            observed=dataclasses.replace(state,phase={'explore':'refine','refine':'explore'}.get(state.phase,state.phase)) if a.arm=='swapped' else state
            if policy_override is not None:
                probs=[];action=policy_override.select(observed).encoded
            elif actor is not None:
                probs=actor.probabilities(observed)
                action=local.rng.choices(range(16),weights=probs,k=1)[0] if sample else max(range(16),key=lambda k:probs[k])
            elif table is not None:
                probs=[];action=table.select(observed).encoded
            else:
                probs=[];action=12 if a.arm=='fixed12' or a.arm=='oracle' and state.phase=='refine' else 0
            action|=state.current_action
            with guard:questions.append(dict(task=local.task,attempt=local.attempt,state=dataclasses.asdict(observed),action=action,probabilities=probs))
            return LockAction.from_encoded(action)
    policy=Policy()
    m=AgentTransactionManager(paper_policy=policy,record_traces=False,collect_trajectories=True,
        performance_guards_enabled=False,low_conflict_occ_guard=False,priority_enabled=False,delayed_write_apply_enabled=True)
    configure_paper_observed_contract(m,explicit_phase_label_ablation=True)
    certificate(m)
    reservation=ObservedWriteReservationHooks(m.interceptor.hooks)
    m.interceptor.hooks=reservation
    for k in range(8):m.register_object(f'catalog:{k}','1')
    for i in range(a.tasks):
        key=f'account:{i}';m.register_object(key,'0')
        m.register_object(f'competitor-only:{i}','0')
        if a.workload=='tpcc':
            for k in range(3):m.register_object(f'stock:{i}:{k}','100')
        tx=m.begin('maintenance:'+str(i),dict(access_set_visibility='online_observed'),strategy='paper-atcc');tx.write(key,'0');assert tx.commit('paper-atcc').committed
    # Populate history through actual writes, rather than a future footprint.
    assert all(f'account:{i}' in reservation.observed_writes for i in range(a.tasks))
    transition=m.transition_atcc_action
    def measured(tx,action,**kw):
        before=tx.context.action.encoded;tick=time.perf_counter_ns()
        try:return transition(tx,action,**kw)
        finally:changes.append(dict(task=tx.task_id,before=before,after=tx.context.action.encoded,ns=time.perf_counter_ns()-tick))
    m.transition_atcc_action=measured
    rows=[];attempts=[];competitors=[];start=time.monotonic()
    def task(spec):
        index=spec['index'];key=f'account:{index}';began=time.monotonic();started=threading.Event()
        background_result=[]
        def competitor():
            started.wait();time.sleep(.1*spec['delay_s'])
            tx=m.begin(f'competitor:{index}',dict(access_set_visibility='online_observed'),strategy='paper-atcc')
            tx.context.action=LockAction.from_encoded(12)
            competitor_key=key if spec['competitor_shared'] else f'competitor-only:{index}'
            value=int(tx.read(competitor_key).value);tx.write(competitor_key,str(value+1))
            with tracked_agent_wait(tx):time.sleep(.5*spec['delay_s'])
            result=tx.commit('paper-atcc');assert result.committed
            background_result.append(dict(task=index,finish_s=time.monotonic()-start))
        thread=threading.Thread(target=competitor);thread.start();ok=False
        for retry in range(6):
            local.task=index;local.attempt=retry;local.rng=random.Random(a.seed+index*1009+retry*100003)
            tx=m.begin(f'foreground:{index}',dict(retry_count=retry,access_set_visibility='online_observed'),strategy='paper-atcc')
            reason='';paid=0
            try:
                # Same concrete Explore prefix in both branch types. Its
                # values drive the later numeric update; no future flag is
                # provided to the selector. Explicit phase can now indicate
                # further exploration versus transition to a final RMW.
                tx.enter_phase(TransactionPhase.EXPLORE)
                catalog_total=sum(int(tx.read(f'catalog:{k}').value) for k in range(8))
                tx.enter_phase(TransactionPhase.EXPLORE if spec['phase']=='explore' else TransactionPhase.REFINE)
                # Same first-issued-access checkpoint in every arm. No future
                # duration, key or writer/read-only flag enters the policy.
                reservation.delegate._decide(tx,'first-issued-access')
                value=int(tx.read(key).value);started.set()
                tick=time.monotonic()
                with tracked_agent_wait(tx):time.sleep(spec['delay_s'])
                paid=time.monotonic()-tick
                if spec['phase']=='refine':
                    tx.write(key,str(value+catalog_total//8))
                    if a.workload=='tpcc':
                        for k in range(3):
                            stock=f'stock:{index}:{k}'
                            tx.write(stock,str(int(tx.read(stock).value)-1))
                else:
                    tx.enter_phase(TransactionPhase.REFINE)
                tx.enter_phase(TransactionPhase.COMMIT)
                result=tx.commit('paper-atcc');ok=bool(result.committed);reason=str(result.reason)
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc)
                if tx.context.status!=TransactionStatus.COMMITTED:tx.abort_or_observe_wound(reason,strategy='paper-atcc')
                started.set()
            attempts.append(dict(task=index,attempt=retry,committed=ok,reason=reason,paid_s=paid))
            if ok:break
        foreground_end=time.monotonic();thread.join(timeout=120)
        assert not thread.is_alive() and len(background_result)==1
        row=dict(task=index,phase=spec['phase'],committed=ok,attempts=retry+1,
            foreground_service_s=foreground_end-began,pair_cycle_s=time.monotonic()-began)
        with guard:rows.append(row);competitors.extend(background_result)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(task,plan))
    elapsed=time.monotonic()-start
    expected={s['index']:int(s['competitor_shared'])+int(s['phase']=='refine' and next(r for r in rows if r['task']==s['index'])['committed']) for s in plan}
    wrong={i:dict(expected=v,actual=m.store.get(f'account:{i}').value) for i,v in expected.items() if int(m.store.get(f'account:{i}').value)!=v}
    for spec in plan:
        key=f"competitor-only:{spec['index']}";want=int(not spec['competitor_shared'])
        if int(m.store.get(key).value)!=want:wrong[key]=dict(expected=want,actual=m.store.get(key).value)
    if a.workload=='tpcc':
        for spec in plan:
            for k in range(3):
                key=f"stock:{spec['index']}:{k}";want=100-int(spec['phase']=='refine')
                if int(m.store.get(key).value)!=want:wrong[key]=dict(expected=want,actual=m.store.get(key).value)
    unexpected=[r for r in attempts if r['reason'].startswith(('ValueError','RuntimeError','TypeError','AttributeError','AssertionError'))]
    done=[r for r in rows if r['committed']]
    result=dict(arm=a.arm,tasks=a.tasks,workers=a.workers,completed=len(done),completion_rate=len(done)/a.tasks,
        agent_tps=len(done)/elapsed,elapsed_s=elapsed,foreground_p99_s=p99([r['foreground_service_s'] for r in done]),
        pair_cycle_p99_s=p99([r['pair_cycle_s'] for r in rows]),aborts=sum(not r['committed'] for r in attempts),
        competitor_commits=len(competitors),policy_calls=len(questions),actual_changes=sum(r['before']!=r['after'] for r in changes),
        certificate=dict(m.read_only_snapshot_certificate_metrics),values_correct=not wrong,value_errors=wrong,unexpected_errors=unexpected)
    result['workload']=a.workload
    result['conflict']=getattr(a,'conflict','high')
    result['shared_competitors']=sum(s['competitor_shared'] for s in plan)
    a.output.mkdir(parents=True,exist_ok=False)
    for name,x in [('result',result),('requests',plan),('tasks',rows),('attempts',attempts),('decisions',questions),('changes',changes),('competitors',competitors)]:save(a.output/f'{name}.json',x)
    save(a.output/'protocol.json',dict(arguments=vars(a),scope='New phase-sensitive YCSB/TPC-C business variants;not original main curves or standard benchmarks',
        roles='Explore read-only branch versus Refine RMW branch;identical update-then-compute competitors',
        checkpoint='before first issued read;all nonphase features retained;no future key/cost/read-only flag supplied',
        priority_enabled=False,common_read_only_certificate=True,replay=False,final_exclusive_recovery=False,
        conflict_definition='high:all paired competitors share account;low:one in eight shares account,others update private disjoint key',
        tps='foreground commits/batch wall including competitor drain;competitor commits excluded',
        reward='negative actual pair cycle including externality;60s terminal failure',new_api_calls=0))
    assert not wrong and not unexpected and result['competitor_commits']==a.tasks,result
    return result,questions,rows


def train(a):
    from agent.runtime.shared_bit_mask_policy import SharedBitMaskActor
    from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
    from agent.runtime.state_collector import phase_aware_state_from_dict
    from agent.runtime.trajectory import PolicyTransition
    from agent.cc.atcc.ppo import DiscretePPOTrainer,PPOConfig
    from scripts.revision.train_heterogeneous_phase_updates import actor_training_rows
    a.output.mkdir(parents=True,exist_ok=False)
    actor=SharedBitMaskActor(remove_phase=False,retain_explicit_phase=a.arm=='full',seed=61113021,residual_context='phase-mask-retry-v1')
    for row in actor.actor_weights:row[0]=math.log(.2/.8)
    config=PPOConfig(learning_rate=.05,group_learning_rate=.1,epochs=12,min_group_samples=4,min_group_actions=2,
        actor_return_mode='global',calibration_return_mode='global',conditioned_mask_gradient=True,logged_action_support=True,initialize_from_behavior=False)
    save(a.output/'training_protocol.json',dict(config=dataclasses.asdict(config),batches=a.batches,tasks=a.tasks,workers=a.workers,
        seed=a.seed,scale=a.scale,only_removed_feature='phase label',from_neutral_prior=True,
        selection='last checkpoint;no postfit',reward='negative actual pair completion including competitor;60s failure penalty'))
    start=time.monotonic();updates=[]
    for batch in range(a.batches):
        options=argparse.Namespace(**vars(a));options.seed=a.seed+batch;options.output=a.output/f'batch_{batch}'
        result,questions,rows=run(options,actor=actor,sample=True)
        costs={r['task']:r['pair_cycle_s']+60*(not r['committed']) for r in rows};grouped={}
        for q in questions:
            state=phase_aware_state_from_dict(q['state']);prob=actor.probabilities(state)
            assert max(abs(x-y) for x,y in zip(prob,q['probabilities']))<1e-10
            grouped.setdefault(q['task'],[]).append((q,state))
        transitions=[];returns=[]
        for task,sequence in grouped.items():
            for j,(q,state) in enumerate(sequence):
                terminal=j==len(sequence)-1;target=-costs[task]
                transitions.append(PolicyTransition(txn_id=str(task),state=state,action=q['action'],reward=target if terminal else 0,
                    next_state=sequence[j+1][1] if not terminal else state,done=terminal,behavior_probability=q['probabilities'][q['action']],
                    behavior_action_probabilities=tuple(q['probabilities']),source_id=f'b{batch}',decision_origin='actual-runtime'))
                returns.append(target)
        fit=DiscretePPOTrainer(config,advantage_scale_floor=.05).train(actor,actor_training_rows(actor,transitions,False),return_targets=returns)
        updates.append(dict(batch=batch,result=result,transitions=len(transitions),fit=fit))
        save(a.output/f'actor_{batch}.json',actor.to_dict());print('UPDATE',a.arm,batch,result['agent_tps'],flush=True)
    save(a.output/'actor.json',actor.to_dict());save(a.output/'table.json',QuantizedSharedBitTable.compile(actor.to_dict(),a.batches).to_dict())
    save(a.output/'training_summary.json',dict(elapsed_s=time.monotonic()-start,updates=updates,last_checkpoint_only=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','test'],default='test');p.add_argument('--arm',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--tasks',type=int,default=64)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=61113051);p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--batches',type=int,default=8)
    p.add_argument('--conflict',choices=['high','low'],default='high')
    p.add_argument('--workload',choices=['ycsb','tpcc'],required=True);a=p.parse_args()
    if a.mode=='train':train(a)
    else:print(json.dumps(run(a)[0]),flush=True)
if __name__=='__main__':main()
