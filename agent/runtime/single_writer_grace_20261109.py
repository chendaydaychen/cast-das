"""Opt-in bounded grace for a running owner of one authoritative write lock.

No phase or future duration is inspected. Waiting/multi-write owners remain
preemptible, and the original wound order resumes after the fixed bound.
"""
import math
import time
from .context import TransactionStatus


def install(manager,grace_ms=800.,*,commit_grace_ms=0.,clock_ns=time.monotonic_ns):
    if not math.isfinite(grace_ms) or grace_ms<0:raise ValueError('Finite nonnegative grace required')
    if not math.isfinite(commit_grace_ms) or commit_grace_ms<0:raise ValueError('Finite nonnegative commit grace required')
    if hasattr(manager,'single_writer_grace'):raise ValueError('Install once')
    locks=manager.atcc_locks;old_note=locks._note_acquired;old_release=locks.release_all;old_can=locks._can_wound
    state=dict(grace_ms=float(grace_ms),started={},deferrals=0,commit_grace_ms=float(commit_grace_ms),commit_started={},commit_deferrals=0)
    manager.single_writer_grace=state
    def note(context):
        old_note(context)
        if len(context.held_write_locks)==1:state['started'].setdefault(context.tid,clock_ns())
    def release(context):
        with locks._condition:
            try:return old_release(context)
            finally:
                state['started'].pop(context.tid,None)
                state['commit_started'].pop(context.tid,None)
    def can_wound(requester,holder,meta):
        result=old_can(requester,holder,meta)
        if not result:return False
        # A bounded active commit/tool block can finish instead of losing all
        # preceding reasoning to a near-tied contender. Pending owners remain
        # woundable, so a wait cycle does not become a non-preemptible cycle.
        if (commit_grace_ms>0 and getattr(getattr(holder,'phase',None),'value',None)=='commit'
                and holder.status==TransactionStatus.ACTIVE and not holder.pending_request
                and holder.held_write_locks and meta.writer==holder.tid):
            start_commit=state['commit_started'].setdefault(holder.tid,clock_ns())
            if (clock_ns()-start_commit)/1e6<commit_grace_ms:
                state['commit_deferrals']+=1;return False
        start=state['started'].get(holder.tid)
        if (start is not None and grace_ms>0 and holder.status==TransactionStatus.ACTIVE
                and not holder.pending_request and len(holder.held_write_locks)==1
                and meta.writer==holder.tid and (clock_ns()-start)/1e6<grace_ms):
            state['deferrals']+=1;return False
        return True
    locks._note_acquired=note;locks.release_all=release;locks._can_wound=can_wound
    return state
