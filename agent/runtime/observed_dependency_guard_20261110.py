"""Opt-in observed-dependency admission repair, independent of policy weights.

Private blind writes wait for commit admission. No held lock is released and
all read validation remains. First-read W reservation uses completed-attempt
read-to-write evidence rather than treating any historical write as a promise.
"""
from collections import OrderedDict
from threading import RLock

def install(manager,*,reservation=None,capacity=4096,min_reservation_write_probability=.5,
            optimistic_mixed_reads=False,defer_staged_blind_writes=False):
    if not 0 <= min_reservation_write_probability <= 1:
        raise ValueError('Reservation probability threshold must be in [0, 1]')
    if getattr(manager,'observed_dependency_guard',None) is not None:raise ValueError('Install once')
    if manager.paper_observed_contract.get('physical_eager_write_ablation',False):
        raise ValueError('Blind admission guard is separate from eager physical publication ablation')
    delegate=manager.interceptor.hooks;stats=OrderedDict();lock=RLock()
    metrics=dict(blind_deferrals=0,suppressed_first_read_writes=0,late_read_validations=0,completed_evidence=0,
                 mixed_reads_kept_optimistic=0,staged_blind_transition_deferrals=0)
    manager.observed_dependency_guard=dict(stats=stats,metrics=metrics)
    old_commit=manager.commit;old_transition=manager.transition_atcc_action
    def transition(tx,action,**kw):
        previous=tx.metadata.get('_defer_policy_write_locks',False)
        if (defer_staged_blind_writes and tx.write_set and not set(tx.write_set).intersection(tx.read_set)
                and not set(tx.write_set).intersection(tx.context.read_versions)):
            # These values were privately staged before the new action. A
            # category expansion must not retroactively turn blind writes
            # into long-lived write ownership. Ordinary final admission and
            # version validation are unchanged, and held locks are retained.
            tx.metadata.setdefault('_observed_blind_deferred',set()).update(
                key for key in tx.write_set if key not in tx.context.held_write_locks)
            metrics['staged_blind_transition_deferrals']+=1
        if tx.metadata.get('_observed_blind_deferred'):tx.metadata['_defer_policy_write_locks']=True
        try:return old_transition(tx,action,**kw)
        finally:tx.metadata['_defer_policy_write_locks']=previous
    manager.transition_atcc_action=transition
    def commit(tx,**kw):
        result=old_commit(tx,**kw)
        if result.committed and not str(tx.task_id).startswith('maintenance:'):
            with lock:
                for key in tx.read_set:
                    reads,both=stats.pop(key,(0,0));stats[key]=(reads+1,both+int(key in tx.write_set))
                while len(stats)>capacity:stats.popitem(last=False)
                metrics['completed_evidence']+=1
        return result
    manager.commit=commit
    if optimistic_mixed_reads:
        category_needed=manager.category_read_lock_needed
        def needs_read_lock(tx,key):
            if str(key) in tx.metadata.get('_mixed_optimistic_reads',()):return False
            return category_needed(tx,key)
        manager.category_read_lock_needed=needs_read_lock
    class Hooks:
        def __getattr__(self,name):return getattr(delegate,name)
        def before_read(self,tx,key):
            delegate.checkpoint(tx)
            pending=tx.metadata.get('_observed_blind_deferred',set())
            if key in pending:
                # A later issued read closes speculative admission using the
                # old base version, never by silently refreshing an exposed read.
                record=tx.write_set[key]
                manager.atcc_locks.validate_and_wlock(key,tx.context,record.base_version,
                    lambda:manager.version_manager.read_committed(key).version,timeout_s=120.)
                pending.remove(key);metrics['late_read_validations']+=1
            with lock:reads,both=stats.get(str(key),(0,0))
            candidate=False
            if reservation is not None:
                with reservation.history_lock:known=str(key) in reservation.observed_writes
                candidate=(known and reservation.delegate.enabled(tx) and tx.context.action.protects(hot=manager.peek_hot(key),write=True)
                    and tx.context.snapshot_epoch<0 and key not in tx.read_set and key not in tx.write_set
                    and key not in tx.context.read_versions and key not in tx.context.held_write_locks)
            if (reservation is not None and reads>0 and (both+1)/(reads+2)<min_reservation_write_probability
                    and candidate):
                metrics['suppressed_first_read_writes']+=1
                if optimistic_mixed_reads:
                    tx.metadata.setdefault('_mixed_optimistic_reads',set()).add(str(key))
                    metrics['mixed_reads_kept_optimistic']+=1
                return reservation.delegate.before_read(tx,key)
            return delegate.before_read(tx,key)
        def before_write(self,tx,key):
            delegate.checkpoint(tx)
            hot=manager.peek_hot(key)
            blind=(delegate.enabled(tx) and key not in tx.read_set and key not in tx.context.read_versions
                and key not in tx.context.held_read_locks and key not in tx.context.held_write_locks
                and key not in tx.context.retry_conflict_write_targets
                and tx.context.action.protects(hot=hot,write=True))
            previous=tx.metadata.get('_defer_policy_write_locks',False)
            if blind:tx.metadata['_defer_policy_write_locks']=True
            try:result=delegate.before_write(tx,key)
            finally:tx.metadata['_defer_policy_write_locks']=previous
            if blind and key not in tx.context.held_write_locks:
                tx.metadata.setdefault('_observed_blind_deferred',set()).add(str(key));metrics['blind_deferrals']+=1
            return result
    manager.interceptor.hooks=Hooks()
    return manager.observed_dependency_guard
