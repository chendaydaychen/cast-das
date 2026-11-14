import json
import io
from unittest import mock

import pytest

from agent.runtime import AgentTransactionManager
from agent.workloads.credit_agent_loop import AgentDirectedCreditCursor
from agent.workloads.credit_reasoner import CapturingReasoner, ReplayReasoner, ReasoningResult, OpenAIResponsesReasoner
from agent.workloads.credit_review import CreditReviewConfig, CreditReviewWorkload
from agent.workloads.credit_tools import CapturingTool, ReplayTool
from scripts.unified_trace.run_credit_review_experiment import transaction_metadata


def action(tool, args=None, phase="explore"):
    return json.dumps(dict(tool=tool, args=args or {}, phase=phase))


class Scripted:
    def __init__(self, commands):
        self.commands = iter(commands)

    def reason(self, **kwargs):
        return ReasoningResult(next(self.commands), 1, 1, "test")


class Verifier:
    def apply(self, payload):
        return {"verified": True, "receipt_id": "test-receipt"}


def run(reasoner, tool, max_steps=12):
    workload = CreditReviewWorkload(CreditReviewConfig(company_count=16))
    task = workload.task_for(seed=71, worker_id=0, sequence=0)
    manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
    workload.register(manager)
    txn = manager.begin("agent-test", transaction_metadata("silo", retry_count=0),
                        snapshot_object_ids=None, strategy="silo")
    cursor = AgentDirectedCreditCursor(task, reasoner=reasoner, tool=tool, max_steps=max_steps)
    try:
        result = cursor.execute(txn)
        assert txn.commit("silo").committed
        return result, cursor.trace
    finally:
        if txn.result is None:
            txn.abort("test cleanup", strategy="silo")


@pytest.mark.parametrize("fields", [
    ["profile", "payments", "financials", "limit"],
    ["limit", "financials", "payments", "profile"],
])
def test_model_order_revision_feedback_and_strict_replay(fields):
    proposal = action("propose", dict(decision="review", score=50, new_limit=100), "refine")
    commands = ["invalid json", action("read_company", {"fields": fields}),
                action("read_evidence", {"source": "risk_policy"}), proposal,
                action("read_evidence", {"source": "collateral"}, "refine"),
                action("verify", phase="refine"), proposal,
                action("verify", phase="refine"), action("finish", phase="commit")]
    reasoner, tool = CapturingReasoner(Scripted(commands)), CapturingTool(Verifier())
    result, events = run(reasoner, tool)
    assert result["error_feedback"] == 2
    assert len(tool.records) == 1
    replay_result, replay_events = run(ReplayReasoner(reasoner.records), ReplayTool(tool.records))
    assert result == replay_result
    assert [(e["event"], e.get("object_id"), e.get("value")) for e in events] == [
        (e["event"], e.get("object_id"), e.get("value")) for e in replay_events]
    reads = [e["object_id"].rsplit(":", 1)[-1] for e in events if e["event"] == "read"]
    assert reads[:4] == fields
    assert [e['event_id'] for e in events]==list(range(len(events)))
    for e in events:
        assert 0<=e['start_offset_ns']<=e['end_offset_ns']
        if e['event']=='read':assert isinstance(e['version'],int)
    observed={}
    for e in events:
        if e['event']=='read':observed[e['object_id']]=e['version']
        if e['event']=='llm':assert e['input_read_versions']==observed


def test_exhaustion_has_no_fixed_fallback():
    with pytest.raises(RuntimeError, match="no fixed-plan fallback"):
        run(Scripted([action("finish", phase="commit")]), Verifier(), max_steps=1)


def test_live_checkpoint_is_opt_in_and_requires_observed_operations():
    from types import SimpleNamespace
    for enabled, completed, expected in ((False, 1, 0), (True, 0, 0), (True, 1, 1)):
        checkpoint = mock.Mock()
        manager = SimpleNamespace(expensive_operation_checkpoint_enabled=enabled,
            interceptor=SimpleNamespace(expensive_operation_checkpoint=checkpoint))
        txn = SimpleNamespace(manager=manager, check_active=mock.Mock(), context=SimpleNamespace(
            completed_operations=completed, phase=SimpleNamespace(value="explore")))
        task = SimpleNamespace(company_id=1, task_seed=1)
        cursor = AgentDirectedCreditCursor(task,
            reasoner=Scripted(["invalid json"]), tool=Verifier(), max_steps=1)
        with pytest.raises(RuntimeError, match="budget exhausted"):
            cursor.execute(txn)
        assert checkpoint.call_count == expected
        if expected:
            checkpoint.assert_called_once_with(txn, 1)


def test_aborted_attempt_does_not_start_another_model_call():
    from agent.cc import LockConflict
    workload = CreditReviewWorkload(CreditReviewConfig(company_count=16))
    task = workload.task_for(seed=71, worker_id=0, sequence=0)
    manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
    workload.register(manager)
    txn = manager.begin("cancelled", strategy="silo")
    txn.abort("cancelled", strategy="silo")
    reasoner = mock.Mock()
    cursor = AgentDirectedCreditCursor(task, reasoner=reasoner, tool=Verifier())
    with pytest.raises(LockConflict):
        cursor.execute(txn)
    reasoner.reason.assert_not_called()


def test_abort_during_model_call_stops_before_external_tool():
    from agent.cc import LockConflict
    workload = CreditReviewWorkload(CreditReviewConfig(company_count=16))
    task = workload.task_for(seed=71, worker_id=0, sequence=0)
    manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
    workload.register(manager)
    txn = manager.begin("cancelled-during-call", strategy="silo")
    def response(**kwargs):
        txn.abort("cancelled", strategy="silo")
        return ReasoningResult(action("verify", phase="refine"), 1, 1, "test")
    reasoner, tool = mock.Mock(), mock.Mock()
    reasoner.reason.side_effect = response
    cursor = AgentDirectedCreditCursor(task, reasoner=reasoner, tool=tool)
    with pytest.raises(LockConflict):
        cursor.execute(txn)
    tool.apply.assert_not_called()


def test_invalid_fields_and_phase_regression_do_not_execute_tools():
    commands = [action("read_company", {"fields": ["profile", "unknown"]}),
                action("read_company", {"fields": ["profile"]}, "refine"),
                action("read_evidence", {"source": "risk_policy"}, "explore")]
    captured = CapturingReasoner(Scripted(commands))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        run(captured, Verifier(), max_steps=3)
    final_context = json.loads(captured.records[-1]["context"])
    assert list(final_context["observed_company"]) == ["profile"]
    assert final_context["phase"] == "refine"
    assert "error" in final_context["transcript"][0]["result"]


def test_fixed_cursor_rejects_agent_directed_capture():
    from scripts.revision.compare_frozen_credit_trace import run_system
    with pytest.raises(ValueError, match="matching agent-loop runner"):
        run_system({"agent_directed": True}, "silo", workers=1, max_attempts=1, time_scale=1)


def test_custom_instructions_sent_to_chat_proxy():
    response = dict(choices=[dict(message=dict(content="{}"))], usage={})
    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
        reasoner = OpenAIResponsesReasoner("test", api_key="dummy", api_style="chat",
                    endpoint="https://example.invalid/v1/chat/completions", instructions="custom planner")
        reasoner.reason(label="next", phase="explore", context="facts")
    assert json.loads(call.call_args.args[0].data)["messages"][0]["content"] == "custom planner"


def test_frozen_schedule_preserves_tools_accesses_and_commit_boundary():
    from agent.workloads.frozen_credit_agent import FrozenCreditAgentCursor
    commands = [action("read_company", {"fields": ["profile", "payments", "financials", "limit"]}),
                action("read_evidence", {"source": "risk_policy"}),
                action("propose", dict(decision="review", score=50, new_limit=100), "refine"),
                action("verify", phase="refine"), action("finish", phase="commit")]
    reasoner, tool = CapturingReasoner(Scripted(commands)), CapturingTool(Verifier())
    _, events = run(reasoner, tool)
    record = dict(committed=True, events=events, llm_calls=reasoner.records, tool_calls=tool.records,
        access_targets=[e["object_id"] for e in events if e["event"] in {"read", "write"}])
    workload = CreditReviewWorkload(CreditReviewConfig(company_count=16))
    manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
    workload.register(manager)
    txn = manager.begin("frozen-test", transaction_metadata("silo", retry_count=0),
                        snapshot_object_ids=None, strategy="silo")
    replay_reasoner, replay_tool = CapturingReasoner(Scripted([])), CapturingTool(Verifier())
    boundaries, delays = [], []
    cursor = FrozenCreditAgentCursor(record, reasoner=replay_reasoner, tool=replay_tool,
                                    time_scale=.1, sleep_fn=delays.append)
    result = cursor.execute(txn, before_commit_batch=lambda txn, targets:
                            boundaries.append((len(txn.write_set), len(targets))))
    assert txn.commit("silo").committed
    assert result.revealed_targets == tuple(record["access_targets"])
    assert boundaries == [(0, 4)]
    assert len(replay_reasoner.records) == 5
    assert len(replay_tool.records) == 1
    assert len(delays) == 6
