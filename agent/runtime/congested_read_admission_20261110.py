"""Opt-in cooldown for new category protection on an already queued object.

Fallback means ordinary optimistic reads, NOT version-validation bypass.
Existing locks are never released; actual writes retain normal admission.
"""
import threading,time
from collections import deque,Counter

def install(manager,*,queue_threshold=2,cooldown_s=10.,clock=time.monotonic,min_read_share=0.,read_window=1024,min_read_samples=128):
    if queue_threshold<1 or cooldown_s<=0:raise ValueError('Positive queue threshold and cooldown required')
    if hasattr(manager,'congested_read_admission'):raise ValueError('Install once')
    delegate=manager.interceptor.hooks;category=manager.category_read_lock_needed
    if not 0<=min_read_share<=1 or read_window<min_read_samples or min_read_samples<1:raise ValueError('Invalid concentration window')
    deadlines={};lock=threading.RLock();recent=deque();counts=Counter()
    metrics=dict(activations=0,deferred_reads=0,deferred_transactions=0,existing_locks_retained=0,diffuse_queue_bypasses=0,replay_uncertain_reads=0)
    state=dict(metrics=metrics,deadlines=deadlines,queue_threshold=queue_threshold,cooldown_s=cooldown_s)
    state.update(min_read_share=min_read_share,read_window=read_window,min_read_samples=min_read_samples)
    manager.congested_read_admission=state
    def needs_read(tx,key):
        if str(key) in tx.metadata.get('_congested_occ_reads',()):return False
        return category(tx,key)
    manager.category_read_lock_needed=needs_read
    class Hooks:
        def __getattr__(self,name):return getattr(delegate,name)
        def before_read(self,tx,object_id):
            key=str(object_id)
            if not delegate.enabled(tx) or not delegate.online_observed(tx):return delegate.before_read(tx,key)
            with lock:
                recent.append(key);counts[key]+=1
                if len(recent)>read_window:
                    old=recent.popleft();counts[old]-=1
                    if not counts[old]:del counts[old]
                concentrated=(min_read_share==0 or (len(recent)>=min_read_samples and counts[key]/len(recent)>=min_read_share))
            if key in tx.context.held_write_locks or key in tx.context.held_read_locks:
                metrics['existing_locks_retained']+=1
                return delegate.before_read(tx,key)
            # Query only current execution state; do not inspect next operation.
            delegate.checkpoint(tx)
            action=tx.context.action;hot=manager.peek_hot(key)
            if not (action.protects(hot=hot,write=False) or action.protects(hot=hot,write=True)):
                return delegate.before_read(tx,key)
            with manager.atcc_locks._condition:
                meta=manager.atcc_locks._objects.get(key)
                queued=sum(manager.atcc_locks._request_live(r) and r.tid!=tx.context.tid for r in meta.waiters) if meta else 0
            now=clock()
            reads,both=getattr(manager,'observed_dependency_guard',{}).get('stats',{}).get(key,(0,0))
            replay_uncertain=bool(getattr(manager,'replay_aware_admission_enabled',False)
                and getattr(manager,'observed_prefix_replay_enabled',False)
                and reads>=3 and (both+1)/(reads+2)<.8)
            with lock:
                if queued>=queue_threshold and concentrated:
                    if deadlines.get(key,0)<=now:metrics['activations']+=1
                    deadlines[key]=now+cooldown_s
                elif queued>=queue_threshold:metrics['diffuse_queue_bypasses']+=1
                active=deadlines.get(key,0)>now or replay_uncertain
            if not active:return delegate.before_read(tx,key)
            if '_congested_occ_reads' not in tx.metadata:metrics['deferred_transactions']+=1
            tx.metadata.setdefault('_congested_occ_reads',set()).add(key)
            metrics['deferred_reads']+=1
            if replay_uncertain:metrics['replay_uncertain_reads']+=1
            # Read-set/version tracking is performed by the normal transaction
            # and interceptor after this hook. No already exposed value moves.
            return None
    manager.interceptor.hooks=Hooks()
    return state
