"""Optional paper §4.4 inference overlapping actual agent-side waits.

Workers ONLY infer. The owner polls at safe call/access/control boundaries and
applies an answer only before the next table lookup.
No synchronous actor fallback, no wait for unfinished inference, and no early
lock release. These answers do not make the underlying table lossless.
"""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import copy
import hashlib
import json
import math
import threading
import time
from types import MappingProxyType

from .context import LockAction
from .paper_observation_policy import PaperObservationPolicy


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class AsyncRefinementBundle:
    artifact_type='cast-das-paper-async-refinement-bundle'

    def __init__(self,table,actor_snapshot,*,distance_threshold=.25):
        from agent.cc.atcc.actor_artifact import actor_from_snapshot
        if not isinstance(table,PaperObservationPolicy):raise ValueError('Pure paper table required')
        table.to_dict() # Also rejects hidden synchronous actor overrides.
        if not math.isfinite(distance_threshold) or not 0<=distance_threshold<=1:
            raise ValueError('Normalized refinement threshold required')
        self.table=table
        self._actor_snapshot=copy.deepcopy(actor_snapshot)
        actor=actor_from_snapshot(self._actor_snapshot)
        if actor.phase_awareness!='disabled' or actor.monotone_actions!=table.monotone_actions:
            raise ValueError('Refinement actor/table observation/action domains differ')
        actor.record_prototypes=False
        # Pure query inference: no runtime prototypes/default-group mutations.
        actor.actor_weights=tuple(tuple(v) for v in actor.actor_weights)
        actor.critic_weights=tuple(actor.critic_weights)
        actor.actor_group_logits=MappingProxyType({k:tuple(v) for k,v in actor.actor_group_logits.items()})
        actor.observed_group_actions=MappingProxyType({k:frozenset(v) for k,v in actor.observed_group_actions.items()})
        if actor.categorical_actor:
            actor.categorical_actor_weights=tuple(tuple(v) for v in actor.categorical_actor_weights)
            actor.categorical_group_logits=MappingProxyType({k:tuple(v) for k,v in actor.categorical_group_logits.items()})
        self.actor=PaperObservationPolicy(actor,remove_phase_features=table.remove_phase_features)
        self.distance_threshold=float(distance_threshold)
        self.actor_sha256=digest(self._actor_snapshot)
        self.table_sha256=digest(table.to_dict())

    def __getattr__(self,name):return getattr(self.table,name)
    def select(self,state):return self.table.select(state)
    def select_with_distribution(self,state):return self.table.select_with_distribution(state)
    def to_dict(self):
        return dict(artifact_type=self.artifact_type,version=1,table=self.table.to_dict(),
            actor=copy.deepcopy(self._actor_snapshot),actor_sha256=self.actor_sha256,
            table_sha256=self.table_sha256,distance_threshold=self.distance_threshold,
            application='owner-boundaries-before-next-table-lookup',
            mask_correction='only-latest-table-added-bits; held-locks-retained',
            synchronous_actor_fallback=False)

    @classmethod
    def from_dict(cls,data):
        if (data.get('artifact_type')!=cls.artifact_type or data.get('version')!=1 or
                data.get('application')!='owner-boundaries-before-next-table-lookup' or
                data.get('mask_correction')!='only-latest-table-added-bits; held-locks-retained' or
                data.get('synchronous_actor_fallback') is not False):
            raise ValueError('Unsupported async refinement bundle')
        if digest(data['table'])!=data['table_sha256'] or digest(data['actor'])!=data['actor_sha256']:
            raise ValueError('Refinement bundle hashes differ')
        return cls(PaperObservationPolicy.from_dict(data['table']),data['actor'],
                   distance_threshold=data['distance_threshold'])


@dataclasses.dataclass
class Request:
    txn_id: str
    owner_thread_id: int
    epoch: int
    bundle: object
    state: object
    operations: int
    distance: float
    table_action: int
    future: object=None


class SelectiveRefinement:
    def __init__(self,manager,*,workers=2,max_pending=64,max_events=4096):
        if any(type(n) is not int or n<1 for n in (workers,max_pending,max_events)):
            raise ValueError('Positive integer refinement bounds required')
        self.manager=manager;self.lock=threading.RLock()
        self.executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='aegis-refine')
        self.slots=threading.BoundedSemaphore(max_pending)
        self.requests={};self.wait_depth={};self.futures=set();self.events=deque(maxlen=max_events)
        self.counts=Counter();self.query_ms=0.;self.closed=False;self.events_emitted=0

    def event(self,kind,txn,**details):
        with self.lock:
            self.counts[kind]+=1
            self.events_emitted+=1
            self.events.append(dict(event=kind,txn_id=str(txn.context.tid),**details))

    def before_lookup(self,txn):
        # A later table control point always invalidates the old answer, even
        # if an inference thread completes a moment after this invalidation.
        self.forget(txn,reason='next-table-lookup')
        txn.metadata['_async_refinement_epoch']=int(txn.metadata.get('_async_refinement_epoch',0))+1

    def after_lookup(self,txn,state,bundle):
        if self.closed or not isinstance(bundle,AsyncRefinementBundle):return
        if self.manager.paper_policy.snapshot() is not bundle:return
        distance=bundle.last_lookup_distance
        if distance is None or distance<=bundle.distance_threshold:return
        if bundle.monotone_actions and int(state.current_action)==15:
            self.event('forced-action-skipped',txn);return
        request=Request(str(txn.context.tid),threading.get_ident(),txn.metadata['_async_refinement_epoch'],bundle,
            bundle.table.observation(state),int(txn.context.completed_operations),float(distance),
            int(txn.context.action.protected))
        with self.lock:
            if txn.result is not None:return
            self.requests[str(txn.context.tid)]=request
        self.event('eligible',txn,epoch=request.epoch,generation=bundle.generation,distance=distance)

    def begin_interval(self,txn):
        tid=str(txn.context.tid)
        with self.lock:
            self.wait_depth[tid]=self.wait_depth.get(tid,0)+1
            if self.wait_depth[tid]!=1:return
            request=self.requests.get(tid)
            if self.closed or request is None or request.future is not None:return
            if not self.slots.acquire(blocking=False):
                self.requests.pop(tid,None);self.event('capacity-skipped',txn);return
            try:request.future=self.executor.submit(self._infer,request)
            except Exception:
                self.slots.release();self.requests.pop(tid,None);self.event('submit-failed',txn);return
            self.futures.add(request.future)
            request.future.add_done_callback(self._finished)
            self.event('submitted',txn,epoch=request.epoch)

    def _finished(self,future):
        self.slots.release()
        with self.lock:self.futures.discard(future)

    def _infer(self,request):
        started=time.perf_counter()
        with self.lock:
            if (self.requests.get(request.txn_id) is not request or
                    not self.wait_depth.get(request.txn_id,0)):
                self.counts['inference-without-overlap-skipped']+=1
                return None
        try:return request.bundle.actor.select(request.state).encoded
        finally:
            with self.lock:self.query_ms+=(time.perf_counter()-started)*1000.

    def end_interval(self,txn,*,successful=True):
        tid=str(txn.context.tid)
        with self.lock:
            depth=self.wait_depth.get(tid,0)
            if depth>1:self.wait_depth[tid]=depth-1;return
            self.wait_depth.pop(tid,None)
        if not successful or txn.result is not None:
            self.forget(txn,reason='external-call-or-attempt-ended');return
        self.poll(txn)

    def poll(self,txn):
        """Owner-only, nonblocking inference check before a new table deadline.

        Access progress since submission is allowed: this is a delayed answer
        to the prior control point. Applying it STILL validates the complete
        current observed prefix through the ordinary transition protocol.
        """
        tid=str(txn.context.tid)
        with self.lock:request=self.requests.get(tid)
        if request is None:return
        if threading.get_ident()!=request.owner_thread_id:
            self.event('non-owner-skipped',txn);return
        future=request.future
        if (self.closed or self.manager.paper_policy.snapshot() is not request.bundle or
                txn.metadata.get('_async_refinement_epoch')!=request.epoch or txn.result is not None):
            self.forget(txn,reason='stale-model-epoch-or-attempt');self.event('stale',txn);return
        if future is None:return # No real external interval: never infer here.
        if not future.done():
            self.event('not-ready',txn)
            return
        with self.lock:
            if self.requests.get(tid) is not request:return
            self.requests.pop(tid)
        try:proposed=future.result() # Nonblocking: done() already true.
        except Exception as exc:
            self.event('inference-failed',txn,error=type(exc).__name__);return
        if proposed is None:
            self.event('unavailable',txn);return
        current=int(txn.context.action.protected)
        source_mask=int(request.state.current_action)
        if request.bundle.monotone_actions:
            if proposed|source_mask!=proposed:raise ValueError('Refinement violates its original actor decision domain')
            # Correct only the approximation introduced by THIS table lookup.
            # Older source bits and any subsequently introduced bits survive.
            selected=proposed|(current&~request.table_action)
        else:selected=proposed
        correction_reduces_mask=selected|current!=selected
        answer=dict(epoch=request.epoch,generation=request.bundle.generation,
            actor_action=proposed,applied_action=selected,current_action=current,
            table_action=request.table_action,
            inference_state=dataclasses.asdict(request.state),owner_thread_id=request.owner_thread_id,
            application_thread_id=threading.get_ident())
        if selected==current:
            self.event('unchanged',txn,**answer);return
        txn.check_active()
        hooks=self.manager.interceptor.hooks
        live=request.bundle.table.observation(hooks._state(txn))
        if self.manager.collect_trajectories:
            # Deterministic deployed correction, not a fresh stochastic PPO
            # sample. Existing exclusion logic rejects probability-one rows.
            self.manager.trajectory_collector.decision(tid,live,selected,1.,
                lock_count=hooks._lock_count(txn),blocked_time_ms=txn.context.blocked_time_ms,
                behavior_action_probabilities=tuple(float(i==selected) for i in range(16)),
                monotone_actions=request.bundle.monotone_actions and not correction_reduces_mask,
                decision_origin='asynchronous-policy-refinement',**hooks._externality_metrics(txn))
        started=time.perf_counter()
        try:
            self.manager.refresh_atcc_priority(txn)
            self.manager.transition_atcc_action(txn,LockAction.from_encoded(selected),
                timeout_s=hooks._action_transition_timeout_s(txn),
                allow_mask_reduction=not request.bundle.monotone_actions or correction_reduces_mask)
            self.manager.reconcile_selected_read_protection(txn,timeout_s=hooks._action_transition_timeout_s(txn))
        except Exception as exc:
            self.event('installation-failed',txn,error=type(exc).__name__,**answer)
            raise
        finally:
            self.manager.add_commit_timing(txn,'policy-refinement-change',(time.perf_counter()-started)*1000.)
            self.manager.interceptor.reset_agent_interval(txn)
        self.event('applied',txn,application_state=dataclasses.asdict(live),**answer)

    def forget(self,txn,*,reason='attempt-finished'):
        with self.lock:
            request=self.requests.pop(str(txn.context.tid),None)
            # Do not reset depth while a wound callback races an active wait.
        if request is not None:
            if request.future is not None:request.future.cancel()
            self.event('discarded',txn,reason=reason)

    def snapshot(self):
        with self.lock:return dict(counts=dict(self.counts),inference_ms=self.query_ms,
            pending=len(self.requests),active_intervals=len(self.wait_depth),events=list(self.events),
            outstanding_inferences=len(self.futures),
            events_emitted=self.events_emitted,events_dropped=max(0,self.events_emitted-len(self.events)),
            worker_changes_transaction_locks=False,synchronous_actor_fallback=False)

    def reset_metrics(self):
        """Only at a fully drained warmup boundary, never during measurement."""
        with self.lock:
            if self.requests or self.wait_depth:raise ValueError('Refinement warmup has not drained')
            futures=tuple(self.futures)
        for future in futures:
            try:future.result(timeout=5)
            except TimeoutError:raise
            except Exception:pass
        with self.lock:self.events.clear();self.counts.clear();self.query_ms=0.;self.events_emitted=0

    def close(self):
        with self.lock:
            self.closed=True
            for request in self.requests.values():
                if request.future is not None:request.future.cancel()
            self.requests.clear()
        self.executor.shutdown(wait=True)


def attach_selective_refinement(manager,**bounds):
    if not getattr(manager,'paper_observed_contract',None):raise ValueError('Observed paper contract required')
    if not isinstance(manager.paper_policy.snapshot(),AsyncRefinementBundle):
        raise ValueError('Atomically linked actor/table bundle required')
    if getattr(manager,'selective_refinement',None) is not None:raise ValueError('Refinement already attached')
    controller=SelectiveRefinement(manager,**bounds);manager.selective_refinement=controller
    manager.paper_observed_contract['selective_refinement']=dict(enabled=True,
        application='owner-boundaries-before-next-table-lookup',
        mask_correction='only-latest-table-added-bits; held-locks-retained',
        synchronous_actor_fallback=False,worker_changes_transaction_locks=False)
    return controller
