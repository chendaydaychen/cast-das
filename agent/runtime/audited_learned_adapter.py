"""Actual API journal bridge for the common YCSB execution harness."""
import time
from .speculative_policy_runtime import SpeculativePolicyRuntimeManager
from scripts.revision.natural_runtime_journal_20261027 import NaturalRuntimeJournal


class AuditedLearnedAdapter(NaturalRuntimeJournal,SpeculativePolicyRuntimeManager):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.comparison_contexts={};self.audit_transactions=[]

    def begin(self,*args,**kwargs):
        txn=super().begin(*args,**kwargs)
        self.comparison_contexts[txn.context.tid]=txn;self.audit_transactions.append(txn)
        return txn

    def payload_evidence(self):
        journal=self.export_natural_journal()
        return dict(attempts=[dict(task_id=r['task_id'],attempt=r['attempt'],start_ns=r['start_ns'],
            end_ns=r['end_ns'],committed=r['committed'],ops=r['ops']) for r in journal['attempts']],
            final={key:dict(value=str(self.store.get(key).value),version=int(self.store.get_version(key)))
                for key in self._catalog})
