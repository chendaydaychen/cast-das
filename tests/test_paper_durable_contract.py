"""The durable fault driver must actually execute the new policy contract."""
import json
import pytest
from agent.cc.atcc.priority_learning import PriorityReturnOptimizer
from agent.runtime.durable_storage import DurableRoutedVersionedKVStore
from agent.runtime.paper_observation_policy import PaperObservationPolicy
from agent.runtime.paper_policy import CompiledPhasePolicy
from scripts.revision.check_durable_multistore_recovery import manager_for,update


@pytest.mark.parametrize('arm',['full','silo'])
def test_current_policy_contract_commits_private_cross_store_values(tmp_path,arm):
    ref=tmp_path/'model';ref.mkdir()
    table=PaperObservationPolicy(CompiledPhasePolicy(phase_awareness='disabled'))
    (ref/'full_table.json').write_text(json.dumps(table.to_dict()))
    optimizer=PriorityReturnOptimizer();optimizer.propose();optimizer.update(2.,1.)
    # A synthetic optimizer input belongs to this unit fixture, not benchmark data.
    (ref/'priority.json').write_text(json.dumps(optimizer.artifact({'unit_fixture':'0'*64})))
    path=tmp_path/'durable'
    with DurableRoutedVersionedKVStore(path) as store:
        manager=manager_for(store,ref,arm,paper_observed=True)
        row=update(manager,arm,('10','20'))
        assert row['committed'] and row['source']=='10' and row['ledger']=='20'
        assert row['observed_policy_boundaries']==(1 if arm=='full' else 0)
        assert store.get_version('source')==store.get_version('ledger')==2
    with DurableRoutedVersionedKVStore(path) as store:
        manager=manager_for(store,ref,arm,attach=True,paper_observed=True)
        row=update(manager,arm,('11','21'))
        assert row['observed_policy_boundaries']==(1 if arm=='full' else 0)
        assert store.get_version('source')==store.get_version('ledger')==3
