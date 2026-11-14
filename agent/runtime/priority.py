"""Abort-cost-aware dynamic priority from the Aegis paper."""

from __future__ import annotations

import dataclasses
import math
import time

from .context import TransactionContext


@dataclasses.dataclass(frozen=True)
class PriorityConfig:
    # Kept under its original public name for CLI compatibility.  It is the
    # paper's Delta_o (completed-operation quantum), not a wall-clock unit.
    sql_weight: float = 1.0
    blocked_weight: float = 1.0
    retry_weight: float = 1.0
    interval_weight: float = 1.0
    sql_quantum_ms: float = 10.0
    blocked_quantum_ms: float = 100.0
    interval_quantum_ms: float = 10.0
    max_priority: int = 1_000_000
    include_prior_retry_cost: bool = True
    include_prior_retry_wait: bool = False
    prior_retry_wait_quantum_ms: float = 100.0
    include_inflight_agent_cost: bool = False
    operation_cost_based: bool = False


class PriorityManager:
    def __init__(self, config: PriorityConfig | None = None):
        self.config = config or PriorityConfig()

    def compute(self, context: TransactionContext) -> int:
        cfg = self.config
        pending_ms = 0.
        if cfg.include_inflight_agent_cost and context.inflight_agent_started_ns:
            pending_ms = max(0., (time.monotonic_ns() - max(
                context.inflight_agent_started_ns, context.last_agent_accounted_ns)) / 1_000_000.)
        score = (
            math.floor(
                cfg.sql_weight
                * (context.operation_cost_ms if cfg.operation_cost_based else context.completed_operations)
                / max(cfg.sql_quantum_ms, 1e-9)
            )
            + math.floor(cfg.blocked_weight * (context.blocked_time_ms / max(cfg.blocked_quantum_ms, 1e-9)
                + (max(0., context.prior_retry_blocked_ms) / max(cfg.prior_retry_wait_quantum_ms, 1e-9)
                   if cfg.include_prior_retry_wait else 0.)))
            + math.floor(cfg.retry_weight * context.retry_count)
            + math.floor(
                cfg.interval_weight
                * (context.agent_cost_ms + pending_ms + (context.prior_retry_cost_ms
                   if cfg.include_prior_retry_cost else 0.0))
                / max(cfg.interval_quantum_ms, 1e-9)
            )
        )
        if cfg.operation_cost_based:
            # Paper mode never lowers an existing transaction's priority.
            score=max(context.priority,score)
        return min(cfg.max_priority, max(0, int(score)))

    def refresh(self, context: TransactionContext, lock_manager: object | None = None) -> int:
        priority = self.compute(context)
        if lock_manager is not None:
            lock_manager.update_priority(context, priority)
        else:
            context.priority = priority
            context.priority_epoch += 1
        return priority
