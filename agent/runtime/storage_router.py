"""Object-to-engine routing with native in-memory atomic multi-store batches."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any


class CrossStorageAtomicityError(RuntimeError):
    """A transaction spans engines without a distributed commit protocol."""


class RoutedVersionedKVStore:
    def __init__(self, engines: Mapping[str, Any], *, default_storage_id: str = "default"):
        if not engines or default_storage_id not in engines:
            raise ValueError("a default storage engine is required")
        if any(not key for key in engines):
            raise ValueError("storage identifiers must be nonempty")
        if len({id(engine) for engine in engines.values()}) != len(engines):
            raise ValueError("each storage identifier needs a distinct engine")
        self._engines = dict(engines)
        self._default = default_storage_id
        self._objects: dict[str, str] = {}
        self._lock = RLock()

    @property
    def backend_name(self) -> str:
        return "routed:" + ",".join(
            f"{key}={engine.backend_name}" for key, engine in sorted(self._engines.items())
        )

    def register_route(self, object_id: str, storage_id: str = "default") -> None:
        key, storage_id = str(object_id), str(storage_id)
        with self._lock:
            if storage_id not in self._engines:
                raise ValueError(f"unknown storage identifier: {storage_id}")
            if key in self._objects:
                raise ValueError(f"object already routed: {key}")
            self._objects[key] = storage_id

    def unregister_route(self, object_id: str) -> None:
        with self._lock:
            self._objects.pop(str(object_id), None)

    def storage_id_for(self, object_id: str) -> str:
        with self._lock:
            return self._objects.get(str(object_id), self._default)

    def _engine(self, key: str) -> Any:
        return self._engines[self.storage_id_for(key)]

    def _batch_engine(self, checks: Any, writes: Any = ()) -> Any:
        identifiers = {self.storage_id_for(key) for key, *_ in (*checks, *writes)}
        if len(identifiers) > 1:
            raise CrossStorageAtomicityError(
                "cross-storage validation/publication needs an atomic commit coordinator"
            )
        return self._engines[next(iter(identifiers), self._default)]

    def _native_batch(self, checks: tuple, writes: tuple) -> bool:
        """Dispatch a cross-store batch only when all engines support native atomicity."""
        from agent.native import load_cast_core

        core = load_cast_core()
        batch_fn = getattr(core, "batch_put_across_stores", None)
        if batch_fn is None or not all(
            isinstance(engine, core.Dbx1000VersionedKVStore)
            for engine in self._engines.values()
        ):
            raise CrossStorageAtomicityError(
                "cross-storage commit requires native Dbx1000 engines with atomic coordinator"
            )
        identifiers = {self._objects.get(str(key), self._default)
                       for key, *_ in (*checks, *writes)}
        batches = []
        for identifier in sorted(identifiers):
            local_checks = [(key, version) for key, version in checks
                            if self._objects.get(str(key), self._default) == identifier]
            local_writes = [(key, value) for key, value in writes
                            if self._objects.get(str(key), self._default) == identifier]
            batches.append((self._engines[identifier], local_checks, local_writes))
        return bool(batch_fn(batches))

    def get(self, key: str) -> Any:
        return self._engine(key).get(str(key))

    def get_version(self, key: str) -> int:
        return self._engine(key).get_version(str(key))

    def put(self, key: str, value: str) -> Any:
        return self._engine(key).put(str(key), str(value))

    def put_if_version(self, key: str, version: int, value: str) -> bool:
        return self._engine(key).put_if_version(str(key), version, str(value))

    def delete_if_version(self, key: str, version: int) -> bool:
        return self._engine(key).delete_if_version(str(key), version)

    def batch_validate_versions(self, checks: Any) -> bool:
        checks = tuple(checks)
        with self._lock:
            if len({self._objects.get(str(key), self._default) for key, *_ in checks}) > 1:
                return self._native_batch(checks, ())
            return self._batch_engine(checks).batch_validate_versions(checks)

    def batch_put_if_version(self, checks: Any, writes: Any) -> bool:
        checks, writes = tuple(checks), tuple(writes)
        with self._lock:
            if len({self._objects.get(str(key), self._default)
                    for key, *_ in (*checks, *writes)}) > 1:
                return self._native_batch(checks, writes)
            return self._batch_engine(checks, writes).batch_put_if_version(checks, writes)
