"""Opt-in paid-work accounting limited to explicitly marked agent intervals.

SQL/runtime gaps and lock waits retain their own accounting. The callback never
credits a future LLM duration; only elapsed tracked_agent_wait time is paid.
"""
import time
import types


def install(manager):
    interceptor=manager.interceptor
    def account(self,txn):
        if self._bypass_background_state(txn):return 0.0
        now=time.monotonic_ns()
        start=int(txn.context.inflight_agent_started_ns or 0)
        previous=int(txn.context.last_agent_accounted_ns)
        elapsed=max(0.,(now-max(start,previous))/1e6) if start else 0.
        txn.context.last_agent_accounted_ns=now
        txn.context.agent_cost_ms+=elapsed
        if self.state_collector is not None and elapsed>0:
            self.state_collector.record_agent_interval(txn.context,elapsed)
        return elapsed
    interceptor._account_agent_interval=types.MethodType(account,interceptor)
    manager.explicit_agent_cost_accounting=True
