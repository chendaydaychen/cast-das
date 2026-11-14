"""Supported volatile storage scope for standalone CC runtime adapters."""
from .storage_router import RoutedVersionedKVStore


def validate_volatile_baseline_storage(store):
    if getattr(store, 'owns_recovery', False):
        raise ValueError('Volatile CC metadata has no durable recovery protocol')
    if not isinstance(store, RoutedVersionedKVStore):
        return
    from agent.native import load_cast_core
    core = load_cast_core()
    if not callable(getattr(core, 'batch_put_across_stores', None)) or not all(
        isinstance(engine, core.Dbx1000VersionedKVStore) for engine in store._engines.values()
    ):
        raise ValueError('Routed CC adapter requires native atomic Dbx1000 engines; no sequential fallback')
