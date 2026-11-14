"""Completion-aware adjustment after observed retry episodes are joined."""
from dataclasses import replace
import math


def penalize_final_failure(rows, penalty_s, *, final_retry_index=None):
    penalty_s = float(penalty_s)
    if not math.isfinite(penalty_s) or penalty_s < 0:
        raise ValueError('Nonnegative finite logical failure cost required')
    adjusted = []
    charged = 0
    for row in rows:
        failed = bool(row.done and row.abort_indicator and not row.commit_indicator)
        if final_retry_index is not None:
            # A missing later decision is not evidence that the task failed.
            failed = failed and row.state.retry_count == final_retry_index
        charged += int(failed)
        adjusted.append(replace(row, reward=row.reward - (penalty_s if failed else 0.)))
    return adjusted, dict(penalty_s=penalty_s, final_failed_tasks=charged,
        required_final_retry_index=final_retry_index,
        scope='once per observed exhausted-retry failure; missing final outcomes uncharged')
