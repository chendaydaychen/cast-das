"""Opt-in absent-key point inserts for the same-runtime protocol adapters.

No deletion or range predicates. An absent registered key has a real version-0
dependency, validated by the original timestamp/version protocol. Existing
point-only managers remain unchanged for running experiments.
"""
from .transaction import AgentTransactionManager
from .tictoc_runtime import TicTocRuntimeManager, TupleStamp
from .hekaton_runtime import HekatonRuntimeManager, MVVersion
from .types import SnapshotValue


class TicTocPointInsertRuntimeManager(TicTocRuntimeManager):
    def register_empty_object(self, object_id, **kwargs):
        key = str(object_id)
        with self.timestamp_latch:
            self._check_healthy()
            if key in self._catalog:
                self._latest(key)
                return
            AgentTransactionManager.register_empty_object(self, key, **kwargs)
            self.tuple_stamps[key] = TupleStamp(int(self.store.get_version(key)))


class HekatonPointInsertRuntimeManager(HekatonRuntimeManager):
    def register_empty_object(self, object_id, **kwargs):
        key = str(object_id)
        with self.mv_condition:
            self._healthy()
            if key in self._catalog:
                self._backend_version(key)
                return
            AgentTransactionManager.register_empty_object(self, key, **kwargs)
            stored = self.store.get(key)
            if stored.exists:
                raise ValueError('Expected an absent insert target')
            absent = MVVersion(key, str(stored.value), int(stored.version))
            absent.exists = False
            self.mv_versions[key] = [absent]

    def begin(self, *args, **kwargs):
        txn = super().begin(*args, **kwargs)
        original = txn.read
        def read(key, **options):
            if options:
                raise ValueError('Hekaton point reads do not support extra options')
            key = str(key)
            with txn._lifecycle_lock, self.mv_condition:
                result = original(key)
                row = self.mv_transactions[txn.context.tid]
                observed = row.reads.get(key)
                if key not in txn.write_set and observed is not None and not getattr(observed, 'exists', True):
                    # Preserve the same absent version identity for validation;
                    # only its returned existence flag needs an explicit form.
                    result = SnapshotValue(result.value, result.version, False)
                    txn.snapshot[key] = result
                return result
        txn.read = read
        return txn


class SpeculativePointInsertMixin:
    """Registered absent point keys for the speculative learned adapter.

    Absence retains a real backend version and value identity. Policy selection,
    dependency checks, exposure and atomic publication remain in the parent.
    No delete/range/dynamic-catalog support is implied.
    """
    def register_empty_object(self,object_id,**kwargs):
        from .speculative_policy_runtime import ValueIdentity
        key=str(object_id)
        with self.learned_condition:
            self._healthy()
            if self.learned_attempts:raise ValueError('Fixed catalog must precede attempts')
            if key in self._catalog:return
            AgentTransactionManager.register_empty_object(self,key,**kwargs)
            stored=self.store.get(key)
            if stored.exists:raise ValueError('Expected absent insert target')
            self.learned_versions[key]=int(stored.version)
            token=ValueIdentity(key,int(stored.version),str(stored.value),None,int(stored.version))
            token.exists=False
            self.committed_values[key]=token;self.value_identities.append(token)

    def begin(self,*args,**kwargs):
        txn=super().begin(*args,**kwargs);original=txn.read
        def read(key,**options):
            if options:raise ValueError('Point reads do not accept extra options')
            with txn._lifecycle_lock,self.learned_condition:
                result=original(key)
                token=self.spec_attempts[txn.context.tid].reads.get(str(key))
                if str(key) not in txn.write_set and token is not None and not getattr(token,'exists',True):
                    result=SnapshotValue(result.value,result.version,False);txn.snapshot[str(key)]=result
                return result
        txn.read=read
        return txn
