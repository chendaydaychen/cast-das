"""TicToc timestamp semantics on the common AgentTransaction/KV runtime.

Opt-in reference adapter, NOT native TicToc or its scalable implementation.
A short Python latch makes tuple timestamp checks/extensions and backend atomic
publication indivisible. It is never held across agent reasoning. No Aegis PE,
priority, early protection, global commit timestamp, or version-equality read
validation replaces the timestamp protocol. Single volatile manager, point
objects only; all writes must go through this adapter. The latch cost must be
reported before using this version for performance rankings.

Reference: Yu et al., SIGMOD 2016, Algorithms 2/3:
https://db.cs.cmu.edu/papers/2016/yu-sigmod2016.pdf
"""
from dataclasses import dataclass, replace
import threading
import time

from .transaction import AgentTransactionManager
from .operation_interceptor import NoopTransactionHooks
from .context import TransactionStatus
from .types import TransactionState
from .baseline_storage_contract import validate_volatile_baseline_storage


@dataclass(frozen=True)
class TupleStamp:
    version: int
    wts: int = 0
    rts: int = 0


def prepare_timestamp(reads, writes, current):
    """Return (timestamp, extensions, conflicts) without modifying any record."""
    timestamp = max([0] + [r.wts for key, r in reads.items() if key not in writes]
        + [current[key].rts + 1 for key in writes])
    extensions = {}; conflicts = []
    for key, observed in reads.items():
        if observed.rts >= timestamp:
            continue  # The observed version is already valid at this logical time.
        latest = current[key]
        if latest.version != observed.version or latest.wts != observed.wts:
            conflicts.append(key)
        else:
            extensions[key] = max(latest.rts, timestamp)
    return timestamp, extensions, tuple(sorted(conflicts))


class TicTocRuntimeManager(AgentTransactionManager):
    """Explicit separate manager; never silently replace legacy `tictoc` arm."""
    def __init__(self, **kwargs):
        for key in ('transaction_hooks', 'undo_log_path'):
            if kwargs.get(key) is not None:
                raise ValueError('TicToc reference adapter does not support '+key)
        kwargs.update(transaction_hooks=NoopTransactionHooks(), priority_enabled=False,
            performance_guards_enabled=False, collect_trajectories=False)
        super().__init__(**kwargs)
        validate_volatile_baseline_storage(self.store)
        self.timestamp_latch = threading.RLock()
        self.tuple_stamps = {}
        self.timestamp_poisoned = False
        self.timestamp_metrics = {'commits': 0, 'validation_aborts': 0,
            'extensions': 0, 'stale_reads_accepted': 0, 'latch_wait_ms': 0.0}

    def _check_healthy(self):
        if self.timestamp_poisoned:
            raise RuntimeError('TicToc metadata poisoned; do not continue after external mutation/publication fault')

    def register_object(self, object_id, initial_value, **kwargs):
        with self.timestamp_latch:
            self._check_healthy()
            super().register_object(object_id, initial_value, **kwargs)
            key = str(object_id)
            if key in self.tuple_stamps:
                self._latest(key)
            else:
                self.tuple_stamps[key] = TupleStamp(int(self.store.get_version(key)))

    def register_empty_object(self, *args, **kwargs):
        raise ValueError('Point objects must exist; insertion/version lifecycle is not implemented')

    def _latest(self, key):
        self._check_healthy()
        version = int(self.store.get_version(key))
        latest = self.tuple_stamps.setdefault(key, TupleStamp(version))
        if version != latest.version:
            self.timestamp_poisoned = True
            raise RuntimeError('External write bypassed TicToc timestamp metadata: '+key)
        return latest

    def begin(self, task_id, metadata=None, *, snapshot_object_ids=None, strategy='tictoc'):
        if strategy != 'tictoc':
            raise ValueError('Do not mix CC protocols in one TicToc manager')
        metadata = dict(metadata or {})
        if any(metadata.get(k) for k in ('paper_atcc', 'paper_atcc_backend', 'retry_protection_mask')):
            raise ValueError('Aegis protection cannot be inherited by TicToc')
        self._check_healthy()
        txn = super().begin(task_id, metadata, snapshot_object_ids=snapshot_object_ids, strategy='tictoc')
        txn.metadata['_tictoc_reads'] = {}
        original_read, original_write = txn.read, txn.write
        def access(kind, key, value=None, **options):
            key = str(key)
            if key not in self._catalog:
                raise KeyError('object is not registered: '+key)
            with txn._lifecycle_lock:
                started = time.perf_counter()
                with self.timestamp_latch:
                    self.timestamp_metrics['latch_wait_ms'] += (time.perf_counter()-started)*1000
                    latest = self._latest(key)
                    own = key in txn.write_set
                    first = key not in txn.read_set
                    result = (original_read(key, **options) if kind == 'read'
                        else original_write(key, value, **options))
                    if kind == 'read' and not own and first:
                        if int(result.version) != latest.version:
                            raise RuntimeError('Unexpected untracked historical read')
                        txn.metadata['_tictoc_reads'][key] = latest
                    return result
        txn.read = lambda key, **options: access('read', key, **options)
        txn.write = lambda key, value, **options: access('write', key, value, **options)
        return txn

    def commit(self, txn, *, strategy='tictoc'):
        if strategy != 'tictoc' or txn.manager is not self:
            raise ValueError('Wrong TicToc transaction/strategy')
        reads = txn.metadata['_tictoc_reads']; writes = txn.write_set
        started = time.perf_counter()
        with self.timestamp_latch:
            waited = time.perf_counter()-started
            self.timestamp_metrics['latch_wait_ms'] += waited*1000
            latest = {key: self._latest(key) for key in set(reads)|set(writes)}
            timestamp, extensions, conflicts = prepare_timestamp(reads, writes, latest)
            txn.metadata['tictoc_commit_ts'] = timestamp
            if conflicts:
                self.timestamp_metrics['validation_aborts'] += 1
                return self._finish_abort(txn, strategy='tictoc', reason='TicToc timestamp conflict',
                    conflict_object_ids=conflicts, lock_wait_s=waited)
            txn.context.transition(TransactionStatus.COMMITTING)
            # Read validation is timestamp-based, NOT full read-version equality.
            # Check latest write versions for atomic backend installation only.
            checks = [(key, latest[key].version) for key in writes]
            try:
                installed = self.store.batch_put_if_version(checks,
                    [(key, write.value) for key, write in writes.items()])
                if not installed:
                    self.timestamp_poisoned = True
                    raise RuntimeError('External publisher bypassed TicToc latch')
                fresh = {key: int(self.store.get_version(key)) for key in writes}
                if any(fresh[key] != latest[key].version+1 for key in writes):
                    self.timestamp_poisoned = True
                    raise RuntimeError('Unsupported backend version increment')
            except BaseException:
                self.timestamp_poisoned = True
                raise
            for key, rts in extensions.items():
                self.tuple_stamps[key] = replace(latest[key], rts=rts)
            for key in writes:
                self.tuple_stamps[key] = TupleStamp(fresh[key], timestamp, timestamp)
            txn.metadata['runtime_published_versions'] = dict(fresh)
            self.timestamp_metrics['extensions'] += len(extensions)
            self.timestamp_metrics['stale_reads_accepted'] += sum(
                reads[key].version != latest[key].version for key in reads)
            self.timestamp_metrics['commits'] += 1
            txn.metadata['tictoc_extensions'] = dict(extensions)
        txn.state = TransactionState.COMMITTED
        txn.context.transition(TransactionStatus.COMMITTED)
        txn._event('finish', {'state': 'committed', 'action': 'commit', 'tictoc_ts': timestamp})
        txn.result = txn._result(strategy='tictoc', committed=True, action='commit', lock_wait_s=waited)
        self._note_runtime_outcome(txn, True)
        self._record(txn)
        self.interceptor.finish(txn)
        with self._lock:
            self._native_dirty_objects.update(writes)
            self._live_transactions.pop(txn.context.tid, None)
        self._finalize_commit_timing(txn)
        return txn.result
