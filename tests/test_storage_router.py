"""Routing must never silently turn a multi-engine commit into partial writes."""

import pytest

from agent.runtime.storage_router import CrossStorageAtomicityError, RoutedVersionedKVStore


class FakeStore:
    backend_name = "fake"

    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def get_version(self, key):
        return 1 if key in self.values else 0

    def put(self, key, value):
        self.values[key] = value

    def batch_validate_versions(self, checks):
        return all(self.get_version(key) == version for key, version in checks)

    def batch_put_if_version(self, checks, writes):
        if not self.batch_validate_versions(checks):
            return False
        self.values.update(writes)
        return True


def test_routes_objects_to_distinct_engines():
    first, second = FakeStore(), FakeStore()
    routed = RoutedVersionedKVStore({"default": first, "audit": second})
    routed.register_route("company", "default")
    routed.register_route("audit", "audit")
    routed.put("company", "one")
    routed.put("audit", "two")
    assert first.values == {"company": "one"}
    assert second.values == {"audit": "two"}
    assert routed.batch_put_if_version([("audit", 1)], [("audit", "three")])
    assert second.values["audit"] == "three"


def test_cross_engine_batch_rejected_without_partial_write():
    first, second = FakeStore(), FakeStore()
    routed = RoutedVersionedKVStore({"default": first, "audit": second})
    routed.register_route("company")
    routed.register_route("audit", "audit")
    routed.put("company", "one")
    routed.put("audit", "two")
    with pytest.raises(CrossStorageAtomicityError):
        routed.batch_put_if_version(
            [("company", 1), ("audit", 1)],
            [("company", "changed"), ("audit", "changed")],
        )
    assert first.values == {"company": "one"}
    assert second.values == {"audit": "two"}


def test_unknown_storage_rejected():
    routed = RoutedVersionedKVStore({"default": FakeStore()})
    with pytest.raises(ValueError, match="unknown storage"):
        routed.register_route("audit", "missing")


def test_manager_registers_each_object_in_selected_native_engine():
    from agent.native import load_cast_core
    from agent.runtime import AgentTransactionManager

    core = load_cast_core()
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    manager = AgentTransactionManager(storage_engines={"default": first, "audit": second},
                                      record_traces=False, collect_trajectories=False)
    manager.register_object("company", "one", storage_id="default")
    manager.register_object("audit", "two", storage_id="audit")
    assert first.get("company").exists and not first.get("audit").exists
    assert second.get("audit").exists and not second.get("company").exists
    assert manager.store.storage_id_for("audit") == "audit"


def test_empty_registration_checks_target_store_and_rolls_back_route():
    from agent.native import load_cast_core
    from agent.runtime import AgentTransactionManager

    core = load_cast_core()
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    manager = AgentTransactionManager(storage_engines={"default": first, "audit": second},
                                      record_traces=False, collect_trajectories=False)
    second.put("occupied", "existing")
    with pytest.raises(ValueError, match="already exists"):
        manager.register_empty_object("occupied", storage_id="audit")
    assert manager.store.storage_id_for("occupied") == "default"
    assert second.get("occupied").value == "existing"
    first.put("only-default", "other-store")
    manager.register_empty_object("only-default", storage_id="audit")
    assert manager.store.storage_id_for("only-default") == "audit"
    assert not manager.store.get("only-default").exists
    assert first.get("only-default").value == "other-store"


def test_cross_storage_transaction_commits_atomically():
    from agent.native import load_cast_core
    from agent.runtime import AgentTransactionManager

    core = load_cast_core()
    if not hasattr(core, "batch_put_across_stores"):
        pytest.skip("native multi-store coordinator has not been built")
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    manager = AgentTransactionManager(storage_engines={"default": first, "audit": second},
                                      record_traces=False, collect_trajectories=False)
    manager.register_object("company", "one")
    manager.register_object("audit", "two", storage_id="audit")
    txn = manager.begin("cross-engine", strategy="silo", snapshot_object_ids=None)
    txn.read("company")
    txn.read("audit")
    txn.write("company", "changed")
    txn.write("audit", "changed")
    assert txn.commit("silo").committed
    assert first.get("company").value == "changed"
    assert second.get("audit").value == "changed"


def test_native_cross_store_conflict_never_partially_publishes():
    from agent.native import load_cast_core

    core = load_cast_core()
    if not hasattr(core, "batch_put_across_stores"):
        pytest.skip("native multi-store coordinator has not been built")
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    routed = RoutedVersionedKVStore({"default": first, "audit": second})
    routed.register_route("company")
    routed.register_route("audit", "audit")
    routed.put("company", "one")
    routed.put("audit", "two")
    second.put("audit", "someone-else-updated")
    assert not routed.batch_put_if_version(
        (("company", 1), ("audit", 1)),
        (("company", "changed"), ("audit", "changed")))
    assert first.get("company").value == "one"
    assert second.get("audit").value == "someone-else-updated"


def test_native_cross_store_invalid_write_fails_before_other_store_changes():
    from agent.native import load_cast_core

    core = load_cast_core()
    if not hasattr(core, "batch_put_across_stores"):
        pytest.skip("native multi-store coordinator has not been built")
    first = core.Dbx1000VersionedKVStore()
    second = core.Dbx1000VersionedKVStore(max_value_bytes=3)
    routed = RoutedVersionedKVStore({"default": first, "audit": second})
    routed.register_route("company")
    routed.register_route("audit", "audit")
    routed.put("company", "one")
    routed.put("audit", "two")
    with pytest.raises(ValueError):
        routed.batch_put_if_version(
            (("company", 1), ("audit", 1)),
            (("company", "changed"), ("audit", "too-long")))
    assert first.get("company").value == "one"
    assert second.get("audit").value == "two"


def test_native_concurrent_multi_store_conflict_has_one_winner():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from agent.native import load_cast_core

    core = load_cast_core()
    if not hasattr(core, "batch_put_across_stores"):
        pytest.skip("native multi-store coordinator has not been built")
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    routed = RoutedVersionedKVStore({"default": first, "audit": second})
    routed.register_route("company")
    routed.register_route("audit", "audit")
    routed.put("company", "old")
    routed.put("audit", "old")
    gate = Barrier(2)

    def attempt(value):
        gate.wait()
        return routed.batch_put_if_version(
            (("company", 1), ("audit", 1)),
            (("company", value), ("audit", value)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ("first", "second")))
    assert sorted(results) == [False, True]
    assert first.get("company").value == second.get("audit").value
