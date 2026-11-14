"""MOCC protocol reference adapter on the common agent runtime/KV.

Per-record approximate conflict temperature, retrospective retry locks,
canonical lock restoration (locks MAY be dropped), and atomic full read
validation. Python condition queues replace native MQL; not the original
many-core implementation or its scalability claim. Volatile point objects;
all participants in one manager use this protocol. No Aegis PE or priority.

Protocol: https://www.vldb.org/pvldb/vol10/p49-wang.pdf, section 3, Algorithm 1.
"""
import collections
import hashlib
import math
import random
import threading
import time

from agent.cc.locks import LockConflict
from .context import TransactionStatus
from .operation_interceptor import NoopTransactionHooks
from .transaction import AgentTransactionManager
from .types import TransactionState
from .baseline_storage_contract import validate_volatile_baseline_storage


class ConflictTemperature:
    """Exponent counter: increment with probability 2**(-temperature)."""
    def __init__(self, threshold=10, reset_seconds=10., seed=20261026, clock=time.monotonic):
        if type(threshold) is not int or not 1 <= threshold <= 31:
            raise ValueError('Temperature threshold must be an integer in [1,31]')
        if not math.isfinite(reset_seconds) or reset_seconds <= 0:
            raise ValueError('Positive finite reset interval required')
        self.threshold, self.reset_seconds, self.seed, self.clock = threshold, reset_seconds, seed, clock
        self.rows = {}; self.rngs = {}; self.lock = threading.RLock()

    def _row(self, key):
        now = self.clock(); epoch, value = self.rows.get(key, (now, 0))
        if now-epoch >= self.reset_seconds: epoch, value = now, 0
        self.rows[key] = (epoch, value)
        return epoch, value

    def value(self, key):
        with self.lock: return self._row(str(key))[1]

    def hotter(self, key):
        key = str(key)
        with self.lock:
            epoch, value = self._row(key)
            if key not in self.rngs:
                salt = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
                self.rngs[key] = random.Random(self.seed ^ salt)
            if value < 31 and self.rngs[key].getrandbits(value) == 0:
                value += 1
            self.rows[key] = (epoch, value)
            return value


class CanonicalReadWriteLocks:
    """Logical-key canonical ordering, fair/cancellable condition queues."""
    def __init__(self, timeout_s=30.):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Positive finite lock timeout required')
        self.timeout_s = timeout_s; self.condition = threading.Condition(threading.RLock())
        self.holders = {}; self.held = {}; self.queues = {}; self.sequence = 0
        self.events = []

    def _event(self, kind, owner, key, mode):
        self.events.append({'kind': kind, 'owner': owner, 'key': key, 'mode': mode,
            'at_ns': time.monotonic_ns()})

    def _release(self, owner, keys):
        for key in list(keys):
            mode = self.held.get(owner, {}).pop(key, None)
            if mode is None: continue
            row = self.holders[key]; row.pop(owner)
            if not row: self.holders.pop(key)
            self._event('release', owner, key, mode)
        if not self.held.get(owner): self.held.pop(owner, None)
        self.condition.notify_all()

    def release_all(self, owner):
        with self.condition: self._release(owner, list(self.held.get(owner, {})))

    def take(self, owner, key, mode, retrospective=None):
        if mode not in ('s', 'x'): raise ValueError('Unknown lock mode')
        key = str(key); retrospective = retrospective or {}
        wanted = {k: m for k, m in retrospective.items() if k < key}
        wanted[key] = 'x' if mode == 'x' or retrospective.get(key) == 'x' else 's'
        with self.condition:
            for target, target_mode in sorted(wanted.items()):
                held = self.held.get(owner, {})
                if held.get(target) == 'x' or held.get(target) == target_mode: continue
                # Restore order even for upgrades: release S on the same key
                # and every higher lock before waiting for X.
                self._release(owner, [k for k in held if k >= target])
                self._take_one(owner, target, target_mode)

    def _take_one(self, owner, key, mode):
        request = (self.sequence, owner, mode); self.sequence += 1
        queue = self.queues.setdefault(key, []); queue.append(request)
        deadline = time.monotonic()+self.timeout_s
        try:
            while True:
                holders = self.holders.get(key, {})
                compatible = not holders or mode == 's' and all(m == 's' for m in holders.values())
                preceding = queue[:queue.index(request)]
                fair = not preceding or mode == 's' and all(r[2] == 's' for r in preceding)
                if compatible and fair:
                    self.holders.setdefault(key, {})[owner] = mode
                    self.held.setdefault(owner, {})[key] = mode
                    self._event('acquire', owner, key, mode)
                    return
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise LockConflict('MOCC canonical lock timeout', (key,), kind='lock-timeout')
                self.condition.wait(remaining)
        finally:
            queue.remove(request)
            if not queue: self.queues.pop(key, None)
            self.condition.notify_all()


class MoccHooks(NoopTransactionHooks):
    def __init__(self, manager): self.manager = manager
    def before_read(self, txn, object_id):
        key = str(object_id); rll = txn.metadata['_mocc_rll']
        if self.manager.temperature.value(key) >= self.manager.temperature.threshold or key in rll:
            self.manager.mocc_locks.take(txn.context.tid, key, rll.get(key, 's'), rll)
            txn.refresh_unobserved_locked_snapshot(key)
    def after_read(self, txn, object_id, version):
        if object_id not in txn.write_set:
            txn.metadata['_mocc_read_dependencies'].setdefault(str(object_id), int(version))
    def before_write(self, txn, object_id):
        key = str(object_id); rll = txn.metadata['_mocc_rll']
        # Blind writes remain buffered OCC; no invented RMW dependency.
        if key in txn.metadata['_mocc_read_dependencies'] and (
            self.manager.temperature.value(key) >= self.manager.temperature.threshold or key in rll):
            self.manager.mocc_locks.take(txn.context.tid, key, 'x', rll)
    def on_abort(self, txn, reason): self.manager._remember_failure(txn)
    def on_finish(self, txn):
        self.manager.mocc_locks.release_all(txn.context.tid)
        if txn.result and txn.result.committed:
            with self.manager.mocc_lock: self.manager.retrospective.pop(txn.task_id, None)


class MoccRuntimeManager(AgentTransactionManager):
    """Runtime CC; `silo` accepted only as the legacy harness routing alias."""
    def __init__(self, *, temperature_threshold=10, temperature_reset_seconds=10.,
                 mocc_lock_timeout_s=30., **kwargs):
        for name in ('transaction_hooks', 'undo_log_path'):
            if kwargs.get(name) is not None: raise ValueError('Unsupported MOCC adapter option: '+name)
        kwargs.update(transaction_hooks=NoopTransactionHooks(), priority_enabled=False,
            performance_guards_enabled=False, collect_trajectories=False)
        super().__init__(**kwargs)
        validate_volatile_baseline_storage(self.store)
        self.mocc_locks = CanonicalReadWriteLocks(mocc_lock_timeout_s)
        self.temperature = ConflictTemperature(temperature_threshold, temperature_reset_seconds)
        self.mocc_lock = threading.RLock(); self.retrospective = {}
        self.mocc_poisoned = False
        self.mocc_metrics = collections.Counter()
        self.interceptor.hooks = MoccHooks(self)

    def begin(self, task_id, metadata=None, *, snapshot_object_ids=None, strategy='mocc'):
        if self.mocc_poisoned: raise RuntimeError('MOCC publication fault; manager is poisoned')
        if strategy not in ('mocc', 'silo'): raise ValueError('Do not mix other CC in MOCC manager')
        metadata = dict(metadata or {})
        if any(metadata.get(k) for k in ('paper_atcc', 'paper_atcc_backend', 'retry_protection_mask')):
            raise ValueError('MOCC cannot inherit Aegis protection')
        with self.mocc_lock:
            if int(metadata.get('retry_count', 0)) == 0: self.retrospective.pop(str(task_id), None)
            metadata['_mocc_rll'] = dict(self.retrospective.get(str(task_id), {}))
        metadata.update(_mocc_read_dependencies={}, runtime_cc_protocol='mocc')
        return super().begin(task_id, metadata, snapshot_object_ids=snapshot_object_ids, strategy='silo')

    def _remember_failure(self, txn):
        if txn.metadata.get('_mocc_failure_remembered'): return
        txn.metadata['_mocc_failure_remembered'] = True
        failed = set(txn.metadata.get('_mocc_failed_reads', ()))
        rll = {key: 'x' for key in txn.write_set}
        for key in txn.metadata['_mocc_read_dependencies']:
            if key in failed or self.temperature.value(key) >= self.temperature.threshold:
                rll.setdefault(key, 's')
        with self.mocc_lock:
            self.retrospective[txn.task_id] = dict(sorted(rll.items()))
            self.mocc_metrics['rll_builds'] += 1

    def commit(self, txn, *, strategy='mocc'):
        if self.mocc_poisoned: raise RuntimeError('MOCC publication fault; manager is poisoned')
        if strategy not in ('mocc', 'silo') or txn.manager is not self:
            raise ValueError('Wrong MOCC transaction/strategy')
        started = time.perf_counter()
        try:
            for key in sorted(txn.write_set):
                self.mocc_locks.take(txn.context.tid, key, 'x', txn.metadata['_mocc_rll'])
        except LockConflict as exc:
            return self._finish_abort(txn, strategy='mocc', reason=exc.reason,
                conflict_object_ids=exc.targets, lock_wait_s=time.perf_counter()-started)
        waited = time.perf_counter()-started
        txn.context.transition(TransactionStatus.COMMITTING)
        reads = txn.metadata['_mocc_read_dependencies']
        checks = list(reads.items())
        checks.extend((key, int(self.store.get_version(key))) for key in txn.write_set if key not in reads)
        try:
            ok = self.store.batch_put_if_version(checks,
                [(key, write.value) for key, write in txn.write_set.items()])
        except BaseException:
            self.mocc_poisoned = True
            self.mocc_locks.release_all(txn.context.tid)
            raise
        if not ok:
            failed = tuple(sorted(key for key, version in reads.items() if int(self.store.get_version(key)) != version))
            txn.metadata['_mocc_failed_reads'] = failed
            for key in failed: self.temperature.hotter(key)
            with self.mocc_lock: self.mocc_metrics['validation_aborts'] += 1
            return self._finish_abort(txn, strategy='mocc', reason='MOCC version conflict',
                conflict_object_ids=failed, lock_wait_s=waited)
        checked_versions = dict(checks)
        txn.metadata['runtime_published_versions'] = {key: checked_versions[key]+1 for key in txn.write_set}
        txn.state = TransactionState.COMMITTED; txn.context.transition(TransactionStatus.COMMITTED)
        txn._event('finish', {'state': 'committed', 'action': 'commit', 'protocol': 'mocc'})
        txn.result = txn._result(strategy='mocc', committed=True, action='commit', lock_wait_s=waited)
        self._note_runtime_outcome(txn, True); self._record(txn); self.interceptor.finish(txn)
        with self._lock:
            self._native_dirty_objects.update(txn.write_set)
            self._live_transactions.pop(txn.context.tid, None)
        with self.mocc_lock: self.mocc_metrics['commits'] += 1
        self._finalize_commit_timing(txn)
        return txn.result
