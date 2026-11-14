from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from agent.native import load_cast_core
from agent.cc.locks import LockConflict
from agent.runtime import AgentTransactionManager
from agent.runtime.mocc_runtime import MoccRuntimeManager
from agent.runtime.tictoc_runtime import TicTocRuntimeManager
from agent.runtime.hekaton_runtime import HekatonRuntimeManager

CLASSES = (AgentTransactionManager, TicTocRuntimeManager, MoccRuntimeManager, HekatonRuntimeManager)


def build(cls, *, small_second=False):
    core = load_cast_core()
    engines = {'default': core.Dbx1000VersionedKVStore(),
        'audit': core.Dbx1000VersionedKVStore(max_value_bytes=3) if small_second else core.Dbx1000VersionedKVStore()}
    m = cls(storage_engines=engines, record_traces=False, collect_trajectories=False, priority_enabled=False)
    m.register_object('left', '100', storage_id='default')
    m.register_object('right', '0', storage_id='audit')
    cc = 'tictoc' if cls is TicTocRuntimeManager else 'silo'
    return m, engines, cc


@pytest.mark.parametrize('cls', CLASSES)
def test_supported_native_routing_and_cross_store_commit(cls):
    m, engines, cc = build(cls)
    t = m.begin('transfer', strategy=cc)
    assert t.read('left').value == '100' and t.read('right').value == '0'
    t.write('left', '99'); t.write('right', '1')
    assert t.commit(cc).committed
    assert (engines['default'].get('left').value, engines['audit'].get('right').value) == ('99', '1')
    assert not engines['default'].get('right').exists and not engines['audit'].get('left').exists
    assert m.store.get_version('left') == m.store.get_version('right') == 2


@pytest.mark.parametrize('cls', CLASSES)
def test_cross_store_private_abort_never_publishes_either_side(cls):
    m, engines, cc = build(cls)
    t = m.begin('abort', strategy=cc); t.write('left', '99'); t.write('right', '1')
    t.abort('client cancelled', strategy=cc)
    assert (m.store.get('left').value, m.store.get('right').value) == ('100', '0')
    assert m.store.get_version('left') == m.store.get_version('right') == 1


@pytest.mark.parametrize('cls', CLASSES)
def test_second_store_stale_read_rejects_without_first_store_publication(cls):
    m, engines, cc = build(cls)
    t = m.begin('stale', strategy=cc); t.read('right')
    other = m.begin('other', strategy=cc); other.write('right', '1'); assert other.commit(cc).committed
    t.write('left', '99'); assert not t.commit(cc).committed
    assert (m.store.get('left').value, m.store.get('right').value) == ('100', '1')
    assert m.store.get_version('left') == 1 and m.store.get_version('right') == 2


@pytest.mark.parametrize('cls', CLASSES)
def test_second_store_invalid_value_has_no_partial_publish(cls):
    m, engines, cc = build(cls, small_second=True)
    t = m.begin('invalid', strategy=cc); t.write('left', '99'); t.write('right', 'too-long')
    with pytest.raises(ValueError): t.commit(cc)
    assert (m.store.get('left').value, m.store.get('right').value) == ('100', '0')
    assert m.store.get_version('left') == m.store.get_version('right') == 1


@pytest.mark.parametrize('cls', (TicTocRuntimeManager, MoccRuntimeManager, HekatonRuntimeManager))
def test_unsupported_mixed_engines_rejected_before_any_transaction(cls):
    core = load_cast_core(); first = core.Dbx1000VersionedKVStore()
    wrapped = core.Dbx1000VersionedKVStore()
    class Unsupported:
        def __getattr__(self, name): return getattr(wrapped, name)
    with pytest.raises(ValueError, match='native atomic'):
        cls(storage_engines={'default': first, 'audit': Unsupported()})
    assert not first.get('left').exists and not wrapped.get('right').exists


@pytest.mark.parametrize('cls', CLASSES)
def test_concurrent_actual_runtime_transfers_keep_values_and_versions(cls):
    m, engines, cc = build(cls); barrier = threading.Barrier(4)
    def worker(index):
        conflicts = 0
        for logical in range(4):
            for attempt in range(100):
                t = m.begin(f'worker:{index}:logical:{logical}', {'retry_count': attempt}, strategy=cc)
                try:
                    left = int(t.read('left').value); right = int(t.read('right').value)
                    if logical == attempt == 0: barrier.wait(3)
                    t.write('left', str(left-1)); t.write('right', str(right+1))
                    result = t.commit(cc)
                except LockConflict as exc:
                    result = t.abort(str(exc), strategy=cc, conflict_object_ids=exc.targets)
                if result.committed: break
                conflicts += 1
            else: raise AssertionError('Transfer retry cap exceeded')
        return conflicts
    with ThreadPoolExecutor(max_workers=4) as executor: conflicts = list(executor.map(worker, range(4)))
    assert sum(conflicts) >= 3
    assert (m.store.get('left').value, m.store.get('right').value) == ('84', '16')
    assert m.store.get_version('left') == m.store.get_version('right') == 17
    assert not m._live_transactions
