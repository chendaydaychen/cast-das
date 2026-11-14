"""Opt-in reservations at application-declared, currently issued RMW reads.

The declaration is a business/API contract, not a future trace lookup. A
reservation is a performance helper: normal native locks and final version
validation are always retained, including after a cycle-detection bypass.
"""
from contextlib import nullcontext
import time

from agent.cc.locks import LockConflict


class DeclaredRMWReadHooks:
    def __init__(self,manager,delegate,*,declared,select,read_selected=None,clock=None,timeout_s=60.):
        if not manager.coordinated_hotspot_admission:
            raise ValueError('Declared RMW reads require coordinated hotspot admission')
        self.manager,self.delegate=manager,delegate
        self.declared,self.select=declared,select
        self.read_selected=read_selected
        self.clock,self.timeout_s=clock,timeout_s
        self.decisions=[]

    def __getattr__(self,name):
        return getattr(self.delegate,name)

    def before_read(self,txn,key):
        if self.declared(txn,str(key)):
            repeated=key in txn.read_set or key in txn.write_set
            pinned=txn.context.snapshot_epoch>=0
            # Never refresh already returned values or change a pinned view.
            eligible=not repeated and not pinned
            selected=bool(self.select(txn,str(key))) if eligible else False
            row=dict(tid=txn.context.tid,key=str(key),eligible=eligible,selected=selected,
                repeated=repeated,pinned=pinned,reservation_owned=False,
                observed_reads_before=sorted(txn.read_set),wait_and_overhead_ms=0.,
                version_before=int(txn.snapshot[key].version))
            try:
                if selected:
                    started=time.perf_counter()
                    with self.clock.scope('reservation') if self.clock else nullcontext():
                        admitted=self.manager.acquire_hotspot_admission(txn,[key],timeout_s=self.timeout_s)
                    row['wait_and_overhead_ms']=(time.perf_counter()-started)*1000.
                    if not admitted:
                        raise LockConflict('declared RMW reservation timed out',(key,),kind='lock-timeout')
                    txn.check_active()
                    row['reservation_owned']=self.manager._coordinated_admission.snapshot()['owners'].get(str(key))==txn.context.tid
                    # A cycle bypass owns no reservation. Refresh is still
                    # safe for this unreturned non-pinned read; commit validates
                    # the full dependency set and can abort normally.
                    txn.refresh_unobserved_locked_snapshot(key)
                row['version_after']=int(txn.snapshot[key].version)
            finally:
                self.decisions.append(row)
            if eligible and self.read_selected is not None:
                return self.read_selected(txn,key,selected)
        return self.delegate.before_read(txn,key)
