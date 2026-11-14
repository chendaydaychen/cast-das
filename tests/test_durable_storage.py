import pytest
from agent.runtime.durable_storage import DurableRoutedVersionedKVStore


def bootstrap(path):
    store=DurableRoutedVersionedKVStore(path)
    store.register_route('source');store.register_route('audit','audit')
    store.put('source','10');store.put('audit','20')
    return store


def test_cross_store_validation_failure_is_noop(tmp_path):
    with bootstrap(tmp_path) as store:
        assert not store.batch_put_if_version([('source',1),('audit',0)],[('source','11'),('audit','21')])
        assert (store.get('source').value,store.get('audit').value)==('10','20')


def test_decided_batch_recovers_once_and_keeps_versions(tmp_path):
    store=bootstrap(tmp_path)
    def stop(stage,_):
        if stage=='after_participant:audit':raise RuntimeError('simulated install interruption')
    store._fault_hook=stop
    with pytest.raises(RuntimeError):store.batch_put_if_version([('source',1),('audit',1)],[('source','11'),('audit','21')])
    with pytest.raises(RuntimeError,match='unavailable'):store.get('source')
    store.close()
    with DurableRoutedVersionedKVStore(tmp_path) as recovered:
        assert len(recovered.recovery_report['redone_committed'])==1
        assert recovered.get('source').value=='11' and recovered.get('audit').value=='21'
        assert recovered.get_version('source')==recovered.get_version('audit')==2
    with DurableRoutedVersionedKVStore(tmp_path) as twice:
        assert twice.recovery_report==dict(discarded_prepared=[],redone_committed=[])
        assert twice.get_version('source')==2


def test_prepared_without_decision_does_not_publish(tmp_path):
    store=bootstrap(tmp_path)
    def stop(stage,_):
        if stage=='after_prepare':raise RuntimeError('stopped before decision')
    store._fault_hook=stop
    with pytest.raises(RuntimeError):store.put('source','bad')
    store.close()
    with DurableRoutedVersionedKVStore(tmp_path) as recovered:
        assert len(recovered.recovery_report['discarded_prepared'])==1
        assert recovered.get('source').value=='10'


def test_second_owner_rejected_and_close_releases_lock(tmp_path):
    with bootstrap(tmp_path):
        with pytest.raises(RuntimeError,match='owner'):DurableRoutedVersionedKVStore(tmp_path)
    with DurableRoutedVersionedKVStore(tmp_path):pass


def test_participant_replacement_fails_closed(tmp_path):
    with bootstrap(tmp_path):pass
    (tmp_path/'audit.json').unlink()
    with pytest.raises(RuntimeError,match='identity'):DurableRoutedVersionedKVStore(tmp_path)


def test_delete_and_reinsert_keep_monotone_versions(tmp_path):
    with bootstrap(tmp_path) as store:
        assert store.delete_if_version('audit',1)
        assert not store.get('audit').exists and store.get_version('audit')==2
        assert store.put_if_version('audit',2,'again')
        assert store.get_version('audit')==3
    with DurableRoutedVersionedKVStore(tmp_path) as store:
        assert store.get('audit').value=='again' and store.get_version('audit')==3


def test_manager_attach_preserves_data_and_route(tmp_path):
    from agent.runtime import AgentTransactionManager
    with bootstrap(tmp_path) as store:
        manager=AgentTransactionManager(store=store,record_traces=False,collect_trajectories=False)
        manager.attach_existing_object('source',kind='row')
        manager.attach_existing_object('audit',kind='row',storage_id='audit')
        txn=manager.begin('cross-store',strategy='silo')
        assert txn.read('source').value=='10' and txn.read('audit').value=='20'
        txn.write('source','11');txn.write('audit','21')
        assert txn.commit('silo').committed
        with pytest.raises(ValueError,match='route'):
            another=AgentTransactionManager(store=store,record_traces=False,collect_trajectories=False)
            another.attach_existing_object('audit')
        with pytest.raises(ValueError,match='owns recovery'):
            AgentTransactionManager(store=store,undo_log_path=str(tmp_path/'undo'),record_traces=False)
    with DurableRoutedVersionedKVStore(tmp_path) as store:
        assert store.get_version('source')==store.get_version('audit')==2


def test_read_gate_hides_partially_installed_batch(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    installed=Event();release=Event();reader_started=Event()
    with bootstrap(tmp_path) as store:
        def pause(stage,_):
            if stage=='after_participant:audit':
                installed.set()
                if not release.wait(5):raise RuntimeError('test timed out')
        store._fault_hook=pause
        def read():
            reader_started.set()
            return store.get('source').value,store.get('audit').value
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer=pool.submit(store.batch_put_if_version,[('source',1),('audit',1)],[('source','11'),('audit','21')])
            assert installed.wait(5)
            reader=pool.submit(read)
            assert reader_started.wait(5) and not reader.done()
            release.set()
            assert writer.result(timeout=5)
            assert reader.result(timeout=5)==('11','21')


def test_concurrent_stale_updates_have_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    gate=Barrier(2)
    with bootstrap(tmp_path) as store:
        def write(value):
            gate.wait(timeout=5)
            return store.batch_put_if_version([('source',1),('audit',1)],[('source',value),('audit',value)])
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(write,['left','right']))==[False,True]
        assert store.get('source').value==store.get('audit').value


def test_manager_attach_rejects_missing_object_or_active_transaction(tmp_path):
    from agent.runtime import AgentTransactionManager
    with bootstrap(tmp_path) as store:
        manager=AgentTransactionManager(store=store,record_traces=False,collect_trajectories=False)
        store.register_route('missing')
        with pytest.raises(ValueError,match='missing'):manager.attach_existing_object('missing')
        txn=manager.begin('running',strategy='silo')
        with pytest.raises(RuntimeError,match='before admitting'):manager.attach_existing_object('source')
        txn.abort('test done',strategy='silo')
