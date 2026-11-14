"""Replay only completed Explore work after reissuing identical-version reads.

Designed for frozen synthetic reasoning budgets. Refine is always executed;
the module never reads a future operation or reuses value-dependent updates.
"""
from collections import OrderedDict
import threading

class ObservedPrefixReplayClock:
    def __init__(self,clock):
        self.clock=clock;self.local=threading.local();self.manager=None;self.cache=OrderedDict();self.tasks=set();self.lock=threading.RLock()
        self.metrics=dict(explore_computations=0,explore_replays=0,invalidations=0,instruction_replays=0,configured_budget_saved_s=0.,refine_replays=0)
        self.events=[]
    def __getattr__(self,name):return getattr(self.clock,name)
    def bind(self,manager):
        self.manager=manager;original=manager.begin
        manager.observed_prefix_replay_enabled=True
        def begin(*v,**kw):
            tx=original(*v,**kw);self.local.tx=tx;return tx
        manager.begin=begin
    def sleep(self,seconds):
        m=self.manager
        if m is None:return self.clock.sleep(seconds)
        policy=m.paper_policy.snapshot();local=getattr(policy,'local',None)
        task=getattr(local,'task',None);attempt=getattr(local,'attempt',None)
        if task is None:return self.clock.sleep(seconds)
        key=str(task);tx=getattr(self.local,'tx',None)
        if tx is not None and (str(tx.task_id)!=key or tx.metadata.get('retry_count',0)!=attempt):tx=None
        if tx is None:
            with self.lock:
                reuse=key in self.tasks and attempt is not None and attempt>0
                if reuse:
                    self.metrics['instruction_replays']+=1;self.metrics['configured_budget_saved_s']+=seconds
            return None if reuse else self.clock.sleep(seconds)
        if tx.context.phase.value!='explore':return self.clock.sleep(seconds)
        if getattr(tx,'write_set',{}):return self.clock.sleep(seconds)
        observed=tuple(sorted((str(k),int(v.version)) for k,v in tx.read_set.items()))
        if not observed:return self.clock.sleep(seconds)
        slot=(key,tuple(k for k,v in observed))
        with self.lock:
            previous=self.cache.get(slot);reuse=previous==observed
            if previous is not None and not reuse:self.metrics['invalidations']+=1
            if reuse:
                self.metrics['explore_replays']+=1;self.metrics['configured_budget_saved_s']+=seconds
                self.events.append(dict(task=key,attempt=attempt,versions=dict(observed),kind='reused-explore',budget_s=seconds))
        if reuse:return None
        result=self.clock.sleep(seconds)
        with self.lock:
            self.cache[slot]=observed;self.cache.move_to_end(slot);self.tasks.add(key)
            while len(self.cache)>8192:self.cache.popitem(last=False)
            self.metrics['explore_computations']+=1
        return result
