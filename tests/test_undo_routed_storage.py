import json
import sqlite3
import pytest

from agent.runtime.durable_storage import DurableRoutedVersionedKVStore
from agent.runtime.undo_storage import UndoRoutedVersionedKVStore


def bootstrap(path):
    store=UndoRoutedVersionedKVStore(path)
    store.register_route('source');store.register_route('audit','audit')
    store.put('source','10');store.put('audit','20')
    return store


@pytest.mark.parametrize('stage',['after_undo_flush','after_participant:audit',
    'after_participant:default','after_all_installs_before_commit'])
def test_precommit_interruption_undoes_every_installed_participant_and_is_idempotent(tmp_path,stage):
    store=bootstrap(tmp_path)
    def stop(at,_):
        if at==stage:raise RuntimeError('test interruption')
    store._fault_hook=stop
    with pytest.raises(RuntimeError):
        store.batch_put_if_version([('source',1),('audit',1)],[('source','11'),('audit','21')])
    with pytest.raises(RuntimeError,match='unavailable'):store.get('source')
    store.close()
    with UndoRoutedVersionedKVStore(tmp_path) as recovered:
        assert recovered.get('source').value=='10' and recovered.get('audit').value=='20'
        assert recovered.get_version('source')==recovered.get_version('audit')==1
        assert len(recovered.recovery_report['undone_incomplete'])==1
        assert recovered.recovery_report['redo_installs']==0
    with UndoRoutedVersionedKVStore(tmp_path) as again:
        assert not again.recovery_report['undone_incomplete']


@pytest.mark.parametrize('stage',['after_commit_record','after_done'])
def test_commit_marker_retains_installed_updates_without_redo(tmp_path,stage):
    store=bootstrap(tmp_path)
    def stop(at,_):
        if at==stage:raise RuntimeError('test response lost')
    store._fault_hook=stop
    with pytest.raises(RuntimeError):store.batch_put_if_version([], [('source','11'),('audit','21')])
    store.close()
    with UndoRoutedVersionedKVStore(tmp_path) as recovered:
        assert recovered.get('source').value=='11' and recovered.get('audit').value=='21'
        assert recovered.get_version('source')==recovered.get_version('audit')==2
        assert recovered.recovery_report['redo_installs']==0
        with sqlite3.connect(tmp_path/'coordinator.sqlite') as db:
            for body, in db.execute('SELECT body FROM decisions'):
                assert all(set(row)=={'key','storage','before'} for row in json.loads(body))


def test_recovery_interruption_is_idempotent_and_restores_tombstones(tmp_path):
    store=bootstrap(tmp_path);store.register_route('new','audit')
    def stop(at,_):
        if at=='after_all_installs_before_commit':raise RuntimeError('cut install')
    store._fault_hook=stop
    with pytest.raises(RuntimeError):store.batch_put_if_version([], [('source','11'),('new','created')])
    store.close()
    def cut_recovery(at,_):
        if at=='after_undo_restore:audit':raise RuntimeError('cut recovery')
    with pytest.raises(RuntimeError):UndoRoutedVersionedKVStore(tmp_path,fault_hook=cut_recovery)
    with UndoRoutedVersionedKVStore(tmp_path) as recovered:
        assert recovered.get('source').value=='10' and recovered.get_version('source')==1
        assert not recovered.get('new').exists and recovered.get_version('new')==0


def test_redo_and_undo_journals_cannot_be_reinterpreted(tmp_path):
    undo=tmp_path/'undo';redo=tmp_path/'redo'
    with bootstrap(undo):pass
    with pytest.raises(RuntimeError,match='protocol differs'):DurableRoutedVersionedKVStore(undo)
    with DurableRoutedVersionedKVStore(redo) as store:
        store.register_route('x');store.put('x','before')
    with pytest.raises(RuntimeError,match='protocol differs'):UndoRoutedVersionedKVStore(redo)


def test_legacy_redo_without_format_marker_cannot_be_opened_as_undo(tmp_path):
    with DurableRoutedVersionedKVStore(tmp_path) as store:
        store.register_route('x');store.put('x','before')
    with sqlite3.connect(tmp_path/'coordinator.sqlite') as db:db.execute('DROP TABLE journal_format')
    with pytest.raises(RuntimeError,match='legacy decision-redo'):UndoRoutedVersionedKVStore(tmp_path)
    with DurableRoutedVersionedKVStore(tmp_path) as recovered:assert recovered.get('x').value=='before'


@pytest.mark.parametrize('strategy',['silo','paper-atcc'])
def test_manager_commit_routes_and_recovers_without_rewriting_versions(tmp_path,strategy):
    from agent.runtime import AgentTransactionManager
    from agent.runtime.context import LockAction
    from agent.runtime.paper_contract import configure_paper_observed_contract
    from agent.runtime.paper_observation_policy import PaperObservationPolicy
    from agent.runtime.paper_policy import CompiledPhasePolicy
    with bootstrap(tmp_path) as store:
        m=AgentTransactionManager(store=store,collect_trajectories=False,
            paper_policy=PaperObservationPolicy(CompiledPhasePolicy(phase_awareness='disabled')))
        if strategy=='paper-atcc':configure_paper_observed_contract(m)
        m.attach_existing_object('source');m.attach_existing_object('audit',storage_id='audit')
        txn=m.begin('routed-commit',{},strategy=strategy)
        assert txn.read('source').value=='10' and txn.read('audit').value=='20'
        if strategy=='paper-atcc':m.transition_atcc_action(txn,LockAction.from_encoded(15))
        txn.write('source','11');txn.write('audit','21');assert txn.commit(strategy).committed
    with UndoRoutedVersionedKVStore(tmp_path) as recovered:
        assert recovered.get('source').value=='11' and recovered.get('audit').value=='21'
        assert recovered.get_version('source')==recovered.get_version('audit')==2
