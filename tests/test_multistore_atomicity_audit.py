from threading import RLock
from types import SimpleNamespace

import pytest

from scripts.revision.check_multistore_atomicity import exercise, invalid_write_check


def fake_core():
    lock = RLock()
    class Store:
        backend_name = "test_double"
        def __init__(self, max_value_bytes=100):
            self.values = {}
            self.limit = max_value_bytes
        def get(self, key):
            with lock:
                value, version = self.values.get(key, ("", 0))
                return SimpleNamespace(value=value, version=version)
        def put(self, key, value):
            with lock:
                if len(value) > self.limit:
                    raise ValueError("oversize value")
                self.values[key] = value, self.get(key).version + 1
    def batch(batches):
        with lock:
            if any(len(value) > store.limit for store, _, writes in batches for _, value in writes):
                raise ValueError("oversize value")
            if any(store.get(key).version != version for store, checks, _ in batches for key, version in checks):
                return False
            for store, _, writes in batches:
                for key, value in writes:
                    store.put(key, value)
            return True
    return SimpleNamespace(Dbx1000VersionedKVStore=Store, batch_put_across_stores=batch)


def test_transfer_audit_checks_conflicts_and_final_versions(monkeypatch):
    core = fake_core()
    monkeypatch.setattr("agent.native.load_cast_core", lambda: core)
    for mode in ("routed", "direct_native"):
        result = exercise(core, workers=4, per_worker=3, pairs=2, mode=mode)
        assert result["commits"] == 12
        assert result["conflicts"] >= 2
        assert result["stale_second_store_checks_passed"] == 2
        assert all(pair["left"] + pair["right"] == 100000 for pair in result["final_accounts"])
    assert invalid_write_check(core) == "ok"


def test_invalid_write_audit_detects_partial_publication():
    core = fake_core()
    def broken_batch(batches):
        for store, _, writes in batches:
            for key, value in writes:
                store.put(key, value)
        return True
    core.batch_put_across_stores = broken_batch
    with pytest.raises(AssertionError, match="partial publication"):
        invalid_write_check(core)
