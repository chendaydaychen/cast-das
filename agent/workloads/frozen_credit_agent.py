"""Replay captured model-selected access/intent schedules, not live decisions.

Reads use the actual transaction snapshots; writes use frozen captured intents.
Changed observations do NOT regenerate the model's proposal. This deliberately
separates CC trace replay from counterfactual autonomous-agent execution.
"""
import time
from contextlib import nullcontext, contextmanager
from types import SimpleNamespace


@contextmanager
def tracked_agent_wait(txn):
    if getattr(txn.manager,'stop_cancelled_external_work',False):txn.check_active()
    enabled = bool(getattr(txn.manager, 'track_inflight_agent_cost', False))
    refinement=getattr(txn.manager,'selective_refinement',None)
    if refinement is not None:refinement.begin_interval(txn)
    if enabled:
        with txn._agent_work_gate:
            depth=int(txn.metadata.get('_tracked_agent_wait_depth',0))
            txn.metadata['_tracked_agent_wait_depth']=depth+1
            if depth==0:txn.context.inflight_agent_started_ns = time.monotonic_ns()
    successful=False
    try:
        yield
        successful=True
    finally:
        if enabled:
            txn.manager.interceptor.account_agent_interval(txn)
            if txn.result is not None and getattr(txn.manager,'paper_observed_contract',None):
                delta=txn.manager.trajectory_collector.reconcile_terminal_agent_cost(
                    txn.context.tid,txn.context.agent_cost_ms)
                if delta:txn.metadata['_late_terminal_agent_cost_ms']=float(
                    txn.metadata.get('_late_terminal_agent_cost_ms',0.))+delta
            # Reconciliation precedes clearing in-flight: a wound's refresh
            # publisher cannot copy a stale label between these two operations.
            with txn._agent_work_gate:
                depth=max(0,int(txn.metadata.get('_tracked_agent_wait_depth',1))-1)
                txn.metadata['_tracked_agent_wait_depth']=depth
                if depth==0:txn.context.inflight_agent_started_ns = 0
                publish_pending=depth==0 and txn.metadata.get('_offline_refresh_waiting_for_paid_call')
            if publish_pending:
                publish=getattr(txn.manager.interceptor.hooks,'publish_completed_trajectory',None)
                if publish is not None:publish(txn)
        if refinement is not None:refinement.end_interval(txn,successful=successful)


def is_complete_aborted_intent(record):
    execution = record.get('agent_execution') or {}
    events = record.get('events', [])
    terminal = events[-1] if events else {}
    return (record.get('committed') is False and record.get('reason') == 'version conflict'
            and not execution.get('failed') and bool(execution.get('proposal'))
            and execution.get('model_selected_tools') is True
            and terminal.get('event') == 'agent_tool'
            and (terminal.get('action') or {}).get('tool') == 'finish'
            and (terminal.get('result') or {}).get('finished') is True
            and sum(event.get('event') == 'agent_tool' for event in events) == len(record.get('llm_calls', []))
            and sum(event.get('event') == 'llm' for event in events) == len(record.get('llm_calls', []))
            and any(event.get('event') == 'write' for event in events))


class FrozenCreditAgentCursor:
    def _write_value(self,txn,event):
        if event.get('write_role')=='portfolio_capacity':
            raise ValueError('portfolio capacity requires semantic accounting replay')
        return event['value']

    def __init__(self, record, *, reasoner, tool, time_scale=1., sleep_fn=time.sleep, clock=None,
                 allow_complete_aborted_capture=False):
        eligible = record.get('committed') or (allow_complete_aborted_capture and is_complete_aborted_intent(record))
        if time_scale <= 0 or not eligible:
            raise ValueError("positive scale and a completed captured task required")
        self.record, self.reasoner, self.tool = record, reasoner, tool
        self.scale, self.sleep = time_scale, sleep_fn
        self.clock = clock

    def execute(self, txn, before_commit_batch=None):
        segments, current = [], []
        for event in self.record["events"]:
            current.append(event)
            if event["event"] == "agent_tool":
                segments.append(current)
                current = []
        if current or len(segments) != len(self.record["llm_calls"]):
            raise ValueError("incomplete or inconsistent captured action stream")
        accesses, llm_index, tool_index = [], 0, 0
        commit_prepared = False
        for segment in segments:
            command = segment[-1].get("action")
            entered = False
            for event in segment:
                kind = event["event"]
                if kind == "llm":
                    if getattr(txn.manager, "stop_cancelled_external_work", False):
                        txn.check_active()
                    record = self.record["llm_calls"][llm_index]
                    if txn.context.phase.value != record["phase"]:
                        raise ValueError("captured call phase diverged")
                    if (getattr(txn.manager, "expensive_operation_checkpoint_enabled", False)
                            and txn.context.completed_operations > 0):
                        # Mark an observed call boundary, not its future measured duration.
                        txn.manager.interceptor.expensive_operation_checkpoint(txn, 1)
                    exposure = bool(getattr(txn.manager, 'record_frozen_lock_exposure', False))
                    if exposure:
                        sample = dict(call_index=llm_index, phase=txn.context.phase.value,
                            staged_writes=sorted(txn.write_set),
                            held_write_locks=sorted(txn.context.held_write_locks),
                            declared_wait_ms=float(record['elapsed_ms']) * self.scale)
                        wait_started = time.perf_counter()
                    with tracked_agent_wait(txn), self.clock.scope("reasoning") if self.clock else nullcontext():
                        self.sleep(float(record["elapsed_ms"]) * self.scale / 1000.)
                        # The call is paid even if a timely refinement fails
                        # retroactive validation on leaving this interval.
                        if exposure:
                            sample['observed_wait_ms'] = (time.perf_counter()-wait_started)*1000.
                            txn.metadata.setdefault('_frozen_lock_exposure', []).append(sample)
                        self.reasoner.records.append(dict(record, elapsed_ms=record["elapsed_ms"] * self.scale))
                    llm_index += 1
                    if getattr(txn.manager, "stop_cancelled_external_work", False):
                        txn.check_active()
                    if command is not None:
                        txn.enter_phase(command["phase"])
                        entered = True
                elif kind == "external_tool":
                    if getattr(txn.manager, "stop_cancelled_external_work", False):
                        txn.check_active()
                    if self.tool is None:
                        raise ValueError("captured external tool without replay records")
                    record = self.record["tool_calls"][tool_index]
                    with tracked_agent_wait(txn), self.clock.scope("external_tool") if self.clock else nullcontext():
                        self.sleep(float(record["elapsed_ms"]) * self.scale / 1000.)
                        self.tool.records.append(dict(record, elapsed_ms=record["elapsed_ms"] * self.scale))
                    tool_index += 1
                elif kind in {"read", "write"}:
                    if not entered or txn.context.phase.value != event["phase"]:
                        raise ValueError("access phase or action boundary diverged")
                    key = event["object_id"]
                    storage = (txn.manager.store.storage_id_for(key)
                               if hasattr(txn.manager.store, "storage_id_for") else "default")
                    if storage != event["storage_id"]:
                        raise ValueError("captured storage routing diverged")
                    if kind == "read":
                        txn.read(key)
                    else:
                        if not commit_prepared and event.get('write_role')!='draft':
                            if before_commit_batch is not None:
                                before_commit_batch(txn, tuple(e["object_id"] for e in segment
                                                              if e["event"] == "write"))
                            commit_prepared = True
                        replacing=event.get('replace_staged',False)
                        if type(replacing) is not bool or replacing and key not in txn.write_set:
                            raise ValueError('Invalid captured staged-write replacement')
                        value=self._write_value(txn,event)
                        if replacing:txn.write(key,value,replace_staged=True)
                        else:txn.write(key,value)
                    accesses.append(key)
            interceptor = getattr(txn.manager, 'interceptor', None)
            boundary = getattr(interceptor, 'agent_round_finished', None)
            if boundary is not None:
                boundary(txn)
        if accesses != self.record["access_targets"] or tool_index != len(self.record["tool_calls"]):
            raise ValueError("frozen schedule or tool count diverged")
        return SimpleNamespace(branch="model-directed", revealed_targets=tuple(accesses),
                               operation_count=len(accesses))
