"""Opt-in exclusive accounting using the existing YCSB profiling helpers."""
import contextlib
import functools

from scripts.revision.run_ycsb_rebuttal_2260 import ExclusiveClock, StoreProxy


def instrument_manager(manager, clock, system):
    manager.store = StoreProxy(manager.store, clock)
    manager.version_manager.store = manager.store
    manager.transition_atcc_action = clock.wrap(manager.transition_atcc_action, "change")
    strategy = manager.cc_registry.resolve(system)
    strategy.validate = clock.wrap(strategy.validate, "validation")
    for name in ("validate_and_rlock", "validate_and_wlock"):
        raw = getattr(manager.atcc_locks, name)

        def wrap(raw):
            @functools.wraps(raw)
            def called(key, context, observed_version, current_version, **kwargs):
                def checked():
                    with clock.scope("validation"):
                        return current_version()
                return raw(key, context, observed_version, checked, **kwargs)
            return called
        setattr(manager.atcc_locks, name, wrap(raw))
    for condition in (manager.atcc_locks._condition, manager.two_phase_locks._condition,
                      manager.exclusive_locks._condition, manager._online_prefix_condition):
        condition.wait = clock.wrap(condition.wait, "waiting")
    if manager._coordinated_admission is not None:
        manager._coordinated_admission.wait_event=clock.wrap(manager._coordinated_admission.wait_event,'waiting')
    item = manager.interceptor.hooks
    if type(item).__name__ == "ItemFrequencyHooks":
        item._eligible = clock.wrap(item._eligible, "rule_check")
        item._protect = clock.wrap(item._protect, "item_protection")
    manager.preserve_commit_timing = True


def scope(clock, category):
    return clock.scope(category) if clock else contextlib.nullcontext()


def summarize(clock, *, commits, tasks):
    totals = {}
    for row in clock.rows:
        for category, value in row.items():
            totals[category] = totals.get(category, 0) + value
    elapsed_ms = sum(task["elapsed_ms"] for task in tasks)
    accounted_ms = sum(totals.values()) / 1e6
    return dict(exclusive_ms={k: value / 1e6 for k, value in totals.items()},
        per_committed_task_ms={k: value / 1e6 / commits for k, value in totals.items()} if commits else {},
        total_task_elapsed_ms=elapsed_ms, exclusive_total_ms=accounted_ms,
        conservation_error_ms=accounted_ms - elapsed_ms,
        logical_task_roots=len(clock.rows), includes_aborted_attempts=True,
        denominator=commits, waiting_definition="instrumented Condition.wait intervals only; retry_backoff separate",
        validation_definition="explicit strategy/store version checks and retroactive lock version callbacks",
        fused_definition="conditional version validation and publication cannot be separated by the storage API",
        scope="exclusive wall-clock occupancy across worker tasks, not CPU cycles; uninstrumented costs in other")
