import dataclasses
import io
import json
import urllib.error
from unittest import mock
from threading import BoundedSemaphore, Thread
import pytest

from agent.runtime import AgentTransactionManager
from agent.workloads.credit_reasoner import (
    CapturingReasoner, LimitedReasoner, OpenAIResponsesReasoner, ReasoningResult,
    ReplayReasoner,
)
from agent.workloads.credit_review import CreditReviewConfig, CreditReviewWorkload
from agent.workloads.credit_tools import CapturingTool, HttpVerificationTool, ReplayTool
from scripts.revision.credit_tool_service import make_server
from scripts.unified_trace.run_credit_review_experiment import transaction_metadata


class FaithfulFakeReasoner:
    def reason(self, *, label, phase, context):
        if label == "evidence-route":
            text = context.rsplit("|", 1)[-1]
        else:
            text = f"considered {label}"
        return ReasoningResult(text, 10, 3, "test-only")


def run_task(workload, task, reasoner, tool=None):
    manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
    workload.register(manager)
    events = []
    cursor = workload.cursor(task, reasoner=reasoner, tool=tool, trace=events,
                             sleep_fn=lambda _x: None)
    txn = manager.begin("llm-test", transaction_metadata("silo", retry_count=0),
                        snapshot_object_ids=None, strategy="silo")
    execution = cursor.execute(txn)
    assert txn.commit("silo").committed
    return execution, events


def test_live_shape_and_replay_keep_access_path():
    workload = CreditReviewWorkload(CreditReviewConfig(company_count=16, commit_apply_ms=0))
    task = workload.task_for(seed=7, worker_id=0, sequence=0)
    captured = CapturingReasoner(FaithfulFakeReasoner())
    original, events = run_task(workload, task, captured)
    replayed, replay_events = run_task(workload, task, ReplayReasoner(captured.records))
    assert original.revealed_targets == replayed.revealed_targets
    assert original.branch == replayed.branch
    assert len(captured.records) >= 10
    assert sum(e["event"] == "llm" for e in events) == len(captured.records)
    assert sum(e["event"] == "read" for e in events) > 0
    assert sum(e["event"] == "write" for e in events) > 0
    assert sum(e["event"] == "external_tool" for e in replay_events) == 1


def test_openai_response_usage_and_text_items_are_captured():
    response = dict(id="resp-test", model="test-model", output=[
        dict(type="reasoning"),
        dict(type="message", content=[dict(type="output_text", text="high")]),
    ], usage=dict(input_tokens=15, output_tokens=2))
    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
        reasoner = OpenAIResponsesReasoner("test-model", api_key="test-key")
        result = reasoner.reason(label="evidence-route", phase="explore", context="profile=1|0|high")
    assert result.text == "high"
    assert (result.input_tokens, result.output_tokens) == (15, 2)
    payload = json.loads(call.call_args.args[0].data)
    assert payload["store"] is False
    assert "profile=1|0|high" in payload["input"]


def test_http_proxy_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="HTTPS endpoint"):
        OpenAIResponsesReasoner("gpt5.6luna", api_key="test-key",
                                endpoint="http://localhost:35200/v1/responses")
    OpenAIResponsesReasoner("gpt5.6luna", api_key="test-key",
                            endpoint="http://localhost:35200/v1/responses",
                            allow_insecure_http=True)


def test_chat_compatible_response_usage():
    response = dict(model="gpt5.6luna", id="chat-test",
                    choices=[dict(message=dict(content="medium"))],
                    usage=dict(prompt_tokens=11, completion_tokens=2))
    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
        reasoner = OpenAIResponsesReasoner("gpt5.6luna", api_key="test-key",
                                           endpoint="https://example.invalid/v1/chat/completions",
                                           api_style="chat")
        result = reasoner.reason(label="evidence-route", phase="explore", context="profile=1|0|medium")
    assert (result.text, result.input_tokens, result.output_tokens) == ("medium", 11, 2)
    payload = json.loads(call.call_args.args[0].data)
    assert payload["messages"][1]["role"] == "user"


def test_model_request_limit_and_bounded_429_retry():
    response = io.BytesIO(json.dumps(dict(output=[
        dict(type="message", content=[dict(type="output_text", text="low")])
    ], usage=dict(input_tokens=2, output_tokens=1))).encode())
    error = urllib.error.HTTPError("https://example.invalid/v1/responses", 429,
                                   "rate limit", {"Retry-After": "0"}, None)
    delegate = OpenAIResponsesReasoner("test-model", api_key="test-key")
    limited = CapturingReasoner(LimitedReasoner(delegate, BoundedSemaphore(1)))
    with mock.patch("urllib.request.urlopen", side_effect=[error, response]) as request, \
         mock.patch("agent.workloads.credit_reasoner.time.sleep") as sleep:
        result = limited.reason(label="evidence-route", phase="explore", context="profile=1|0|low")
    assert result.text == "low"
    assert request.call_count == 2
    sleep.assert_called_once_with(1.0)
    assert limited.records[0]["elapsed_ms"] >= limited.records[0]["model_service_elapsed_ms"]


def test_real_local_http_tool_capture_and_replay():
    service = make_server()
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        workload = CreditReviewWorkload(CreditReviewConfig(company_count=16))
        task = workload.task_for(seed=17, worker_id=0, sequence=0)
        tool = CapturingTool(HttpVerificationTool(
            f"http://127.0.0.1:{service.server_port}/v1/verify",
            allow_insecure_http=True))
        captured = CapturingReasoner(FaithfulFakeReasoner())
        original, events = run_task(workload, task, captured, tool)
        replayed, replay_events = run_task(
            workload, task, ReplayReasoner(captured.records),
            ReplayTool(tool.records))
        assert original.revealed_targets == replayed.revealed_targets
        assert len(tool.records) == 1
        assert tool.records[0]["response"]["service"] == "local-test-verifier"
        assert events[-1]["event"] == "write"
        assert any(e["event"] == "external_tool" and
                   e["transport"] == "http-or-replay" for e in replay_events)
        with pytest.raises(ValueError, match="diverged"):
            ReplayTool(tool.records).apply({"company_id": -1})
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=2)


def test_concurrent_credit_transactions_across_native_stores():
    from concurrent.futures import ThreadPoolExecutor
    from agent.native import load_cast_core

    core = load_cast_core()
    if not hasattr(core, "batch_put_across_stores"):
        pytest.skip("requires rebuilt native multi-store coordinator")
    workload = CreditReviewWorkload(CreditReviewConfig(company_count=16, commit_apply_ms=0))
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    manager = AgentTransactionManager(storage_engines={"default": first, "audit": second},
                                      record_traces=False, collect_trajectories=False)
    workload.register(manager, storage_for=lambda spec:
                      "audit" if ":audit" in spec.object_id or
                      ":compliance:" in spec.object_id else "default")

    def attempt(index):
        cursor = workload.cursor(workload.task_for(seed=19, worker_id=index, sequence=0),
                                 reasoner=FaithfulFakeReasoner(), sleep_fn=lambda _: None)
        txn = manager.begin(f"multi-{index}", transaction_metadata("silo", retry_count=0),
                            snapshot_object_ids=None, strategy="silo")
        execution = cursor.execute(txn)
        return execution, txn.commit("silo")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert all(execution.operation_count > 0 for execution, _ in results)
    assert sum(result.committed for _, result in results) >= 1
    assert second.get("credit:compliance:0:decision_sequence").exists
