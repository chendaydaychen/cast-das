"""Opt-in common discretization for freshly trained actors and pure tables."""
import dataclasses

from .paper_policy import bucket_log, bucket_ratio, bucket_size, normalize_conflict_kind

TABLE_STATE_PROJECTION = 'shared-table-buckets-v1'
LONG_TIME_STATE_PROJECTION = 'shared-table-long-time-buckets-v2'
TABLE_STATE_PROJECTIONS = (TABLE_STATE_PROJECTION, LONG_TIME_STATE_PROJECTION)
DURATION_FIELDS = frozenset(('inter_round_interval_ms', 'blocked_time_ms',
    'global_avg_latency_ms', 'global_tail_latency_ms', 'observed_restart_cost_ms',
    'global_agent_task_avg_latency_ms', 'global_agent_task_tail_latency_ms'))


def validate_projection(mode):
    if mode not in ('', *TABLE_STATE_PROJECTIONS):
        raise ValueError('Unknown actor/table state projection')


def project_table_state(state, *, preserve_growth=False, mode=TABLE_STATE_PROJECTION):
    """Use representatives of existing buckets, never future access metadata.

    Restart cost is retained as its own log bucket in the context key. Raw
    telemetry and reward accounting are unchanged; only policy input is mapped.
    """
    validate_projection(mode)
    if not mode:
        return state

    def size(value):
        return (0, 1, 4, 9)[bucket_size(value)]

    def ratio(value):
        return (0., .125, .5, 1.)[bucket_ratio(value)]

    def log(value, *, duration=False):
        if duration and mode == LONG_TIME_STATE_PROJECTION:
            # Opt-in finite duration domain, up to 2**30 ms. Legacy buckets
            # and throughput domains remain unchanged. Train afresh with the
            # same declared projection used by the deployed artifact.
            value = max(0., float(value))
            if value <= 0.:
                return 0.
            boundary = 1.
            for _ in range(30):
                if value <= boundary:
                    break
                boundary *= 2.
            return boundary
        bucket = bucket_log(value)
        return float(2 ** (bucket - 1)) if bucket else 0.

    changes = {}
    for name in ('read_set_size', 'write_set_size', 'completed_rounds',
                 'completed_operations', 'global_active_transactions', 'global_waiter_count'):
        changes[name] = size(getattr(state, name))
    for name in ('read_set_growth', 'write_set_growth'):
        value = getattr(state, name)
        changes[name] = min(16, max(0, int(value))) if preserve_growth else size(value)
    for name in ('access_overlap_ratio', 'hotspot_access_ratio', 'recent_write_ratio',
                 'global_abort_rate', 'global_conflict_abort_rate', 'global_background_abort_rate'):
        changes[name] = ratio(getattr(state, name))
    for name in ('inter_round_interval_ms', 'blocked_time_ms', 'global_throughput',
                 'global_avg_latency_ms', 'global_tail_latency_ms', 'observed_restart_cost_ms',
                 'global_agent_task_throughput', 'global_agent_task_avg_latency_ms',
                 'global_agent_task_tail_latency_ms', 'global_background_throughput'):
        changes[name] = log(getattr(state, name), duration=name in DURATION_FIELDS)
    changes['retry_count'] = min(3, max(0, int(state.retry_count)))
    changes['recent_conflict_kind'] = normalize_conflict_kind(state.recent_conflict_kind)
    changes = {name:value for name,value in changes.items() if getattr(state, name) != value}
    return dataclasses.replace(state, **changes) if changes else state


def projected_context_key(base, state):
    return base + f'|restart_cost={bucket_log(state.observed_restart_cost_ms)}'
