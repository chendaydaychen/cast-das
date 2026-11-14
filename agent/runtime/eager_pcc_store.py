"""Opt-in volatile PCC early-install control, not durable crash recovery.

Logical committed versions are independent of physical CAS versions (undo also
advances the latter). Both paired arms must use this adapter. Agent reads in the
eager arm additionally use BlockingOCCReadControl; direct raw readers are outside
the contract. Only registered, existing point objects and native DBx stores are
supported. Defaults elsewhere in the runtime are unchanged.
"""
from contextlib import contextmanager
from threading import RLock, local

from agent.cc.locks import LockConflict
from .context import TransactionStatus
from .types import SnapshotValue


class EagerPCCStore:
    def __init__(self, raw, locks):
        from agent.native import load_cast_core
        from .storage_router import RoutedVersionedKVStore
        native = load_cast_core().Dbx1000VersionedKVStore
        if not (isinstance(raw, native) or
                isinstance(raw, RoutedVersionedKVStore) and
                all(isinstance(s, native) for s in raw._engines.values())):
            raise ValueError('Volatile native DBx backends required')
        self.raw = raw
        self.locks = locks
        self._mutex = RLock()
        self._local = local()
        self._committed = {}
        self._physical = {}
        self._pending = {}  # key -> (owner tid, tentative value)
        self.poisoned = False
        self.events = []

    @property
    def backend_name(self):
        return 'logical-version-control:' + self.raw.backend_name

    def __getattr__(self, name):
        # Never forward native publication fast paths around logical versions.
        if name in ('register_route', 'unregister_route', 'storage_id_for'):
            return getattr(self.raw, name)
        raise AttributeError(name)

    def _gate(self):
        if self.poisoned:
            raise RuntimeError('Eager PCC control unavailable after invariant failure')

    def _load(self, key):
        self._gate()
        value = self.raw.get(key)
        if key not in self._committed:
            self._committed[key] = SnapshotValue(value.value, int(value.version), value.exists)
            self._physical[key] = int(value.version)
        if int(value.version) != self._physical[key]:
            self.poisoned = True
            raise RuntimeError('Unexpected physical mutation outside eager control')
        return self._committed[key]

    def get(self, key):
        with self._mutex:
            return self._load(str(key))

    def get_version(self, key):
        return self.get(key).version

    def put(self, key, value):
        key, value = str(key), str(value)
        with self._mutex:
            self._gate()
            if key in self._pending:
                raise ValueError('Cannot register over pending writes')
            # Registration only; runtime updates must use conditional batches.
            if self._load(key).exists:
                raise ValueError('Eager control supports registration, not unconditional updates')
            result = self.raw.put(key, value)
            row = self.raw.get(key)
            self._committed[key] = SnapshotValue(row.value, int(row.version), row.exists)
            self._physical[key] = int(row.version)
            return result

    @contextmanager
    def owner(self, context):
        if getattr(self._local, 'context', None) is not None:
            raise RuntimeError('Nested committing owner')
        self._local.context = context
        try:
            yield
        finally:
            self._local.context = None

    def batch_validate_versions(self, checks):
        with self._mutex:
            return all(self._load(str(k)).version == int(v) for k, v in checks)

    def _raw_batch(self, checks, writes):
        try:
            ok = self.raw.batch_put_if_version(checks, writes)
            if not ok:
                raise RuntimeError('Physical CAS changed outside eager control')
            for key, _ in writes:
                self._physical[key] = int(self.raw.get_version(key))
        except BaseException:
            self.poisoned = True
            raise

    def stage(self, context, key, base_version, value):
        key, value = str(key), str(value)
        # Always ATCC -> store; wound must not release ownership during install.
        with self.locks._condition, self._mutex:
            self._gate()
            if context.status in (TransactionStatus.ABORTING, TransactionStatus.ABORTED):
                raise LockConflict('Early-install owner was preempted', (key,), kind='lock-preempted')
            if context.status != TransactionStatus.ACTIVE:
                raise RuntimeError('Early install requires an active owner')
            meta = self.locks._objects.get(key)
            if key not in context.held_write_locks or meta is None or meta.writer != context.tid:
                raise RuntimeError('Early install requires authoritative WLock ownership')
            before = self._load(key)
            if not before.exists or before.version != int(base_version):
                raise ValueError('Early install needs an existing current point object')
            pending = self._pending.get(key)
            if pending and pending[0] != context.tid:
                self.poisoned = True
                raise RuntimeError('Foreign tentative write under owned lock')
            if pending == (context.tid, value):
                return
            self._raw_batch([(key, self._physical[key])], [(key, value)])
            self._pending[key] = (context.tid, value)
            self.events.append(dict(kind='early-install', tid=context.tid, key=key))

    def batch_put_if_version(self, checks, writes):
        checks = [(str(k), int(v)) for k, v in checks]
        writes = [(str(k), str(v)) for k, v in writes]
        with self._mutex:
            self._gate()
            if len({k for k, _ in writes}) != len(writes):
                raise ValueError('Duplicate batch writes')
            if not self.batch_validate_versions(checks):
                return False
            context = getattr(self._local, 'context', None)
            if context is None:
                raise RuntimeError('Commit must identify its transaction owner')
            keys = {k for k, _ in writes}
            owned = {k for k, (tid, _) in self._pending.items() if tid == context.tid}
            if not owned.issubset(keys):
                raise RuntimeError('Commit omitted a tentative write')
            for key in keys:
                self._load(key)
                if not self._committed[key].exists:
                    raise ValueError('Insert/delete not supported by eager control')
                if key in self._pending and self._pending[key][0] != context.tid:
                    return False
            physical_checks = [(k, self._physical[k]) for k in sorted(keys | {k for k, _ in checks})]
            installs = [(k, v) for k, v in writes if self._pending.get(k) != (context.tid, v)]
            self._raw_batch(physical_checks, installs)
            for key, value in writes:
                before = self._committed[key]
                self._committed[key] = SnapshotValue(value, before.version + 1, True)
                self._pending.pop(key, None)
            if writes:
                self.events.append(dict(kind='commit-install', tid=context.tid, keys=sorted(keys)))
            return True

    def abort(self, context):
        with self.locks._condition, self._mutex:
            self._gate()
            keys = sorted(k for k, (tid, _) in self._pending.items() if tid == context.tid)
            for key in keys:
                self._load(key)
                meta = self.locks._objects.get(key)
                if key not in context.held_write_locks or meta is None or meta.writer != context.tid:
                    self.poisoned = True
                    raise RuntimeError('Cannot undo after WLock ownership was lost')
            if not keys:
                return
            self._raw_batch([(k, self._physical[k]) for k in keys],
                            [(k, self._committed[k].value) for k in keys])
            for key in keys:
                del self._pending[key]
            self.events.append(dict(kind='undo-before-unlock', tid=context.tid, keys=keys))


def install_eager_pcc_control(manager, *, eager, timeout_s=5., blocking_eager_reads=True):
    """Install both-arm logical accounting, optionally gating eager OCC reads.

    ``blocking_eager_reads=False`` retains the same committed shadow read path
    in both arms, isolating physical install/undo cost. It models an early
    installer that retains a committed version, not an in-place single-version
    engine. The default preserves the original blocking control.

    Volatile correctness control only. Any unexpected commit exception poisons
    the store: this adapter does not promise recovery from mid-publish failures.
    """
    from .paper_contract import PaperObservedHooks
    from .blocking_occ_read_control import install_blocking_occ_read_control
    if manager._live_transactions or getattr(manager, 'eager_pcc_control', None) is not None:
        raise ValueError('Install once before transactions start')
    hooks = manager.interceptor.hooks
    if not isinstance(hooks, PaperObservedHooks) or not getattr(manager, 'paper_observed_contract', None):
        raise ValueError('Explicit observed paper hooks required')
    if getattr(manager, 'blocking_occ_read_control', None) is not None:
        raise ValueError('Install the paired control before any blocking read control')
    # Validate timeout before mutating manager, even in the deferred arm.
    from .blocking_occ_read_control import BlockingOCCReadControl
    BlockingOCCReadControl(manager, timeout_s=timeout_s)
    store = EagerPCCStore(manager.store, manager.atcc_locks)
    manager.store = manager.version_manager.store = store
    manager.eager_pcc_control = store
    original_commit = manager.commit
    def commit(txn, **kwargs):
        with store.owner(txn.context):
            try:
                return original_commit(txn, **kwargs)
            except BaseException:
                store.poisoned = True
                raise
    manager.commit = commit
    original_release = manager.atcc_locks.release_all
    def release(context):
        # Also intercepts native wound, which bypasses txn.abort().
        store.abort(context)
        return original_release(context)
    manager.atcc_locks.release_all = release
    if eager:
        original_after = hooks.after_write
        def after_write(txn, key):
            original_after(txn, key)
            if key in txn.context.held_write_locks:
                write = txn.write_set[key]
                store.stage(txn.context, key, write.base_version, write.value)
        hooks.after_write = after_write
        original_transition = manager.transition_atcc_action
        def transition(txn, action, **kwargs):
            result = original_transition(txn, action, **kwargs)
            for key, write in tuple(txn.write_set.items()):
                if key in txn.context.held_write_locks:
                    store.stage(txn.context, key, write.base_version, write.value)
            return result
        manager.transition_atcc_action = transition
        if blocking_eager_reads:
            install_blocking_occ_read_control(manager, timeout_s=timeout_s)
    manager.paper_observed_contract.update(
        physical_eager_write_ablation=bool(eager),
        write_publication_mode=('volatile-pcc-early-install-with-undo' if eager
                                else 'transaction-private-write-set-until-atomic-commit'),
        eager_control_scope='native-volatile-existing-point-objects-no-crash-recovery',
        logical_version_adapter=True)
    manager.paper_observed_contract.update(
        eager_read_control=bool(eager and blocking_eager_reads),
        read_path_matched=not blocking_eager_reads,
        committed_shadow_reads=not blocking_eager_reads)
    return store
