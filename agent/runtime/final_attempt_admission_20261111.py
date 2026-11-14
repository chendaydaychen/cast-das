"""Opt-in starvation recovery shared by CC implementations.

Ordinary attempts remain concurrent. The last budgeted attempt waits for active
attempts to drain and executes exclusively. Its wait is inside task latency.
No read is refreshed, validation skipped, or retry added. Admission uses only
the current retry counter, never a future access set or duration.
"""
import threading
import time


def install(manager, final_attempt_index=5):
    if final_attempt_index < 1:raise ValueError('Need at least one normal attempt')
    if hasattr(manager,'final_attempt_admission'):raise ValueError('Install once')
    condition=threading.Condition();active=0;exclusive=False;waiting=0
    metrics=dict(normal_admissions=0,recovery_admissions=0,wait_ns=0,active=0,waiting_recoveries=0)
    manager.final_attempt_admission=metrics
    original=manager.begin
    def begin(task_id,*args,**kwargs):
        nonlocal active,exclusive,waiting
        if str(task_id).startswith('maintenance:'):return original(task_id,*args,**kwargs)
        metadata=kwargs.get('metadata',args[0] if args else None) or {}
        recovery=int(metadata.get('retry_count',0))>=final_attempt_index
        tick=time.perf_counter_ns()
        with condition:
            if recovery:
                waiting+=1;metrics['waiting_recoveries']=waiting
                try:
                    condition.wait_for(lambda:active==0 and not exclusive)
                    exclusive=True
                finally:
                    waiting-=1;metrics['waiting_recoveries']=waiting
            else:condition.wait_for(lambda:not exclusive and waiting==0)
            active+=1;metrics['active']=active
            metrics['recovery_admissions' if recovery else 'normal_admissions']+=1
            metrics['wait_ns']+=time.perf_counter_ns()-tick
        released=False
        def release():
            nonlocal active,exclusive,released
            with condition:
                if released:return
                released=True;active-=1;metrics['active']=active
                if recovery:exclusive=False
                condition.notify_all()
        try:tx=original(task_id,*args,**kwargs)
        except BaseException:
            release();raise
        for name in ['commit','abort','abort_or_observe_wound']:
            method=getattr(tx,name)
            def finish(*v,_method=method,**kw):
                try:
                    result=_method(*v,**kw)
                except BaseException:
                    # The caller still owns a possibly live attempt and must
                    # abort it. Do not admit a recovery before that cleanup.
                    raise
                else:
                    release();return result
            setattr(tx,name,finish)
        return tx
    manager.begin=begin
    return metrics
