"""Serializable optimistic Hekaton-style point-version runtime adapter.

Fixed volatile catalog, shared AgentTransaction/DBx publication interface.
Version intervals, first-writer claims, Preparing visibility, commit dependencies
and end-time read validation are independent of Aegis. Python metadata condition
replaces latch-free structures; no logging, range scans or production GC claim.
Tentative versions live in a shared metadata index, not the committed KV image.
All participants must use one manager. Never label this native SQL Server.

Protocol source: Larson et al., PVLDB 2011, sections 2/3:
https://www.cs.cmu.edu/~15721-f24/papers/Hekaton.pdf
"""
from collections import Counter
from dataclasses import dataclass, field
import math
import threading
import time

from agent.cc.locks import LockConflict
from .transaction import AgentTransactionManager
from .operation_interceptor import NoopTransactionHooks
from .context import TransactionStatus
from .types import SnapshotValue, TransactionState
from .baseline_storage_contract import validate_volatile_baseline_storage


@dataclass(eq=False)
class MVVersion:
    key: str
    value: str
    storage_version: int
    begin: int | None = 0
    end: int | None = None  # None means infinity, unless ended_by is set.
    created_by: str | None = None
    ended_by: str | None = None


@dataclass
class MVTransaction:
    begin: int
    state: str = 'active'
    end: int | None = None
    reads: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    dependencies: set = field(default_factory=set)
    dependents: set = field(default_factory=set)
    abort_now: bool = False


class HekatonHooks(NoopTransactionHooks):
    def __init__(self, manager): self.manager = manager
    def on_abort(self, txn, reason):
        with self.manager.mv_condition: self.manager._abort_metadata(txn.context.tid)


class HekatonRuntimeManager(AgentTransactionManager):
    def __init__(self, *, dependency_timeout_s=30., **kwargs):
        if not math.isfinite(dependency_timeout_s) or dependency_timeout_s <= 0:
            raise ValueError('Positive finite dependency timeout required')
        for name in ('transaction_hooks', 'undo_log_path'):
            if kwargs.get(name) is not None: raise ValueError('Unsupported Hekaton adapter option: '+name)
        kwargs.update(transaction_hooks=NoopTransactionHooks(), priority_enabled=False,
            performance_guards_enabled=False, collect_trajectories=False)
        super().__init__(**kwargs)
        validate_volatile_baseline_storage(self.store)
        self.mv_condition = threading.Condition(threading.RLock())
        self.mv_clock = 0; self.mv_versions = {}; self.mv_transactions = {}
        self.mv_metrics = Counter(); self.mv_events = []; self.mv_poisoned = False
        self.dependency_timeout_s = dependency_timeout_s
        self.interceptor.hooks = HekatonHooks(self)

    def _tick(self):
        self.mv_clock += 1; return self.mv_clock

    def _healthy(self):
        if self.mv_poisoned: raise RuntimeError('Hekaton publication fault; manager poisoned')

    def _backend_version(self, key):
        expected = next(v.storage_version for v in reversed(self.mv_versions[key]) if v.created_by is None)
        actual = int(self.store.get_version(key))
        if actual != expected:
            self.mv_poisoned = True
            raise RuntimeError('External write bypassed Hekaton version metadata: '+key)
        return actual

    def register_object(self, object_id, initial_value, **kwargs):
        with self.mv_condition:
            self._healthy()
            if self.mv_transactions: raise ValueError('Fixed catalog must be registered before transactions')
            super().register_object(object_id, initial_value, **kwargs)
            key = str(object_id); stored = self.store.get(key)
            self.mv_versions[key] = [MVVersion(key, str(stored.value), int(stored.version))]

    def register_empty_object(self, *args, **kwargs):
        raise ValueError('Insertion/deletion/range predicates are not implemented')

    def _active(self, txn):
        self._healthy(); row = self.mv_transactions[txn.context.tid]
        if row.abort_now:
            raise LockConflict('Hekaton commit dependency aborted', tuple(row.reads), kind='version-conflict')
        if row.state != 'active': raise RuntimeError('Hekaton transaction is not active')
        return row

    def _dependency(self, reader, target):
        if reader == target: return
        row, source = self.mv_transactions[reader], self.mv_transactions[target]
        if source.state == 'committed': return
        if source.state == 'aborted' or source.abort_now:
            row.abort_now = True
            raise LockConflict('Hekaton commit dependency aborted', tuple(row.reads), kind='version-conflict')
        if source.state != 'preparing' or source.end is None:
            raise RuntimeError('Speculative dependency must reference Preparing transaction')
        if row.end is not None and source.end >= row.end:
            raise RuntimeError('Commit dependency violates timestamp direction')
        if target not in row.dependencies:
            row.dependencies.add(target); source.dependents.add(reader)
            self.mv_metrics['dependencies_created'] += 1
            self.mv_events.append({'kind': 'dependency', 'reader': reader, 'source': target,
                'source_end': source.end, 'at_ns': time.monotonic_ns()})

    def _visible(self, reader, key, timestamp, *, validation=False):
        for version in reversed(self.mv_versions[key]):
            creator = version.created_by
            if creator == reader:
                if validation: continue  # Validate original reads, not own new value.
                return version
            begin = version.begin
            if creator is not None:
                source = self.mv_transactions[creator]
                if source.state in ('active', 'aborted') or source.abort_now: continue
                begin = source.end
            if begin is None or begin > timestamp: continue
            end = version.end
            if version.ended_by is not None and version.ended_by != reader:
                source = self.mv_transactions[version.ended_by]
                if source.state in ('preparing', 'committed') and not source.abort_now:
                    end = source.end
                    if end <= timestamp:
                        if source.state == 'preparing': self._dependency(reader, version.ended_by)
                        continue
                else: end = None  # Active writer's future end is after this timestamp.
            elif version.ended_by == reader:
                end = None
            if end is not None and timestamp >= end: continue
            if creator is not None: self._dependency(reader, creator)
            return version
        raise RuntimeError('No version visible for registered point object')

    def begin(self, task_id, metadata=None, *, snapshot_object_ids=None, strategy='hekaton'):
        if strategy not in ('hekaton', 'silo'): raise ValueError('Do not mix other CC in Hekaton manager')
        metadata = dict(metadata or {})
        if any(metadata.get(k) for k in ('paper_atcc', 'paper_atcc_backend', 'retry_protection_mask')):
            raise ValueError('Hekaton cannot inherit Aegis protection')
        with self.mv_condition:
            self._healthy(); begin = self._tick()
            metadata.update(runtime_cc_protocol='hekaton', hekaton_begin_ts=begin)
            txn = super().begin(task_id, metadata, snapshot_object_ids=None, strategy='silo')
            self.mv_transactions[txn.context.tid] = MVTransaction(begin)
        original_read, original_write, original_abort = txn.read, txn.write, txn.abort
        original_check = txn.check_active
        def check_active():
            # External-call boundaries must also honor irreversible AbortNow,
            # not only the generic transaction status (which remains ACTIVE
            # until the owner runs its abort hooks).
            with txn._lifecycle_lock:
                original_check()
                with self.mv_condition: self._active(txn)
        def read(key):
            key = str(key)
            with txn._lifecycle_lock, self.mv_condition:
                row = self._active(txn)
                if key not in self.mv_versions: raise KeyError(key)
                current_version = self._backend_version(key)
                if key not in txn.write_set and key not in row.reads:
                    version = self._visible(txn.context.tid, key, row.begin)
                    row.reads[key] = version
                    txn.snapshot[key] = SnapshotValue(version.value, version.storage_version, True)
                    self.mv_metrics['historical_reads'] += version.storage_version < current_version
                    self.mv_metrics['speculative_reads'] += version.created_by is not None
                return original_read(key)
        def write(key, value, **options):
            key = str(key)
            with txn._lifecycle_lock, self.mv_condition:
                row = self._active(txn)
                if key not in self.mv_versions: raise KeyError(key)
                self._backend_version(key)
                if key not in row.writes:
                    old = row.reads.get(key) or self._visible(txn.context.tid, key, row.begin)
                    updater = self.mv_transactions.get(old.ended_by)
                    if (updater is not None and updater.state != 'aborted' and not updater.abort_now) or old.end is not None:
                        self.mv_metrics['write_conflicts'] += 1
                        raise LockConflict('Hekaton first-writer version conflict', (key,), kind='version-conflict')
                    txn.snapshot[key] = SnapshotValue(old.value, old.storage_version, True)
                    new = MVVersion(key, str(value), old.storage_version+1, begin=None, created_by=txn.context.tid)
                    result = original_write(key, value, **options)
                    old.ended_by = txn.context.tid; self.mv_versions[key].append(new)
                    row.writes[key] = (old, new)
                    return result
                result = original_write(key, value, **options)
                row.writes[key][1].value = str(value)
                return result
        def abort(reason, *, strategy='', conflict_object_ids=()):
            return original_abort(reason, strategy='hekaton', conflict_object_ids=conflict_object_ids)
        txn.read, txn.write, txn.abort, txn.check_active = read, write, abort, check_active
        return txn

    def _abort_metadata(self, tid):
        row = self.mv_transactions.get(tid)
        if row is None or row.state == 'aborted': return
        if row.state == 'committed': raise RuntimeError('Cannot abort committed Hekaton metadata')
        row.state = 'aborted'
        for old, _new in row.writes.values():
            if old.ended_by == tid: old.ended_by = None; old.end = None
        for target in row.dependencies: self.mv_transactions[target].dependents.discard(tid)
        row.dependencies.clear()
        for dependent in row.dependents:
            other = self.mv_transactions[dependent]
            other.abort_now = True
            self.mv_metrics['cascade_abort_signals'] += 1
        row.dependents.clear()
        self.mv_metrics['aborts'] += 1; self.mv_condition.notify_all()

    def prepare(self, txn):
        """Separately exposed for controlled dependency interleaving tests."""
        with txn._lifecycle_lock, self.mv_condition:
            row = self._active(txn)
            row.end = self._tick(); row.state = 'preparing'
            txn.context.transition(TransactionStatus.COMMITTING)
            txn.metadata['hekaton_end_ts'] = row.end
            conflicts = [key for key, observed in row.reads.items()
                if self._visible(txn.context.tid, key, row.end, validation=True) is not observed]
            if conflicts:
                self.mv_metrics['read_validation_aborts'] += 1
                self._finish_abort(txn, strategy='hekaton', reason='Hekaton read stability version conflict',
                    conflict_object_ids=conflicts, lock_wait_s=0.)
                return False
            return True

    def abort_prepared(self, txn, reason='client abort during preparation'):
        with txn._lifecycle_lock, self.mv_condition:
            if self.mv_transactions[txn.context.tid].state != 'preparing': raise ValueError('Not Preparing')
            return self._finish_abort(txn, strategy='hekaton', reason=reason,
                conflict_object_ids=(), lock_wait_s=0.)

    def complete_prepared(self, txn):
        with txn._lifecycle_lock, self.mv_condition:
            self._healthy(); row = self.mv_transactions[txn.context.tid]
            if row.state != 'preparing': raise ValueError('Not Preparing')
            started = time.perf_counter(); deadline = time.monotonic()+self.dependency_timeout_s
            while row.dependencies and not row.abort_now:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return self._finish_abort(txn, strategy='hekaton', reason='Hekaton dependency timeout',
                        conflict_object_ids=(), lock_wait_s=time.perf_counter()-started)
                self.mv_condition.wait(remaining)
            waited = time.perf_counter()-started
            self.mv_metrics['dependency_wait_ms'] += waited*1000
            if row.abort_now:
                return self._finish_abort(txn, strategy='hekaton', reason='Hekaton dependency version conflict',
                    conflict_object_ids=tuple(row.reads), lock_wait_s=waited)
            checks = [(key, old.storage_version) for key, (old, new) in row.writes.items()]
            for key in set(row.reads)|set(row.writes): self._backend_version(key)
            try:
                ok = self.store.batch_put_if_version(checks,
                    [(key, new.value) for key, (old, new) in row.writes.items()])
                if not ok: raise RuntimeError('External publisher bypassed Hekaton version claims')
            except BaseException:
                self.mv_poisoned = True
                raise
            for old, new in row.writes.values():
                old.end = row.end; old.ended_by = None
                new.begin = row.end; new.created_by = None
            txn.metadata['runtime_published_versions'] = {key: new.storage_version
                for key, (old, new) in row.writes.items()}
            row.state = 'committed'
            for dependent in row.dependents:
                self.mv_transactions[dependent].dependencies.discard(txn.context.tid)
                self.mv_metrics['dependencies_resolved'] += 1
            row.dependents.clear(); self.mv_condition.notify_all()
            txn.state = TransactionState.COMMITTED; txn.context.transition(TransactionStatus.COMMITTED)
            txn._event('finish', {'state': 'committed', 'action': 'commit', 'protocol': 'hekaton'})
            txn.result = txn._result(strategy='hekaton', committed=True, action='commit', lock_wait_s=waited)
            self._note_runtime_outcome(txn, True); self._record(txn); self.interceptor.finish(txn)
            with self._lock:
                self._native_dirty_objects.update(txn.write_set)
                self._live_transactions.pop(txn.context.tid, None)
            self.mv_metrics['commits'] += 1
            self._finalize_commit_timing(txn)
            return txn.result

    def commit(self, txn, *, strategy='hekaton'):
        if strategy not in ('hekaton', 'silo') or txn.manager is not self:
            raise ValueError('Wrong Hekaton transaction/strategy')
        if not self.prepare(txn): return txn.result
        return self.complete_prepared(txn)
