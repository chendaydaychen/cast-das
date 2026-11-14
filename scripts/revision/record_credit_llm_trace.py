#!/usr/bin/env python3
"""Capture or replay an actual LLM-driven CreditReview task stream.

This is an input-trace collector, not a concurrency-control performance run.
The recorded access path and per-call latency can be inspected before choosing
YCSB/TPC-C synthetic timing parameters.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
from pathlib import Path
import sys
from threading import BoundedSemaphore, Thread
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.runtime import AgentTransactionManager
from agent.native import load_cast_core
from agent.workloads.credit_reasoner import (
    CapturingReasoner, LimitedReasoner, OpenAIResponsesReasoner, ReplayReasoner,
)
from agent.workloads.credit_tools import CapturingTool, HttpVerificationTool, ReplayTool
from agent.workloads.credit_review import CreditReviewConfig, CreditReviewTaskSpec, CreditReviewWorkload
from agent.workloads.credit_agent_loop import AgentDirectedCreditCursor, AGENT_INSTRUCTIONS
from scripts.unified_trace.run_credit_review_experiment import transaction_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("live", "replay"))
    parser.add_argument("--model", help="Explicit provider model ID; required in live mode")
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/responses",
                        help="Responses-compatible endpoint; HTTPS required by default")
    parser.add_argument("--api-style", choices=("responses", "chat"), default="responses")
    parser.add_argument("--allow-insecure-http", action="store_true",
                        help="Explicitly permit plaintext HTTP bearer-token transmission")
    parser.add_argument("--input", type=Path, help="Captured JSON for replay")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--company-count", type=int,
                        help="Optional declared contention variant; does not alter replay")
    parser.add_argument("--company-id", type=int,
                        help="Explicit live capture contention fixture; same company for all task seeds")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent tasks on the same transaction manager (default 1)")
    parser.add_argument("--max-inflight-llm", type=int,
                        help="Cap simultaneous model requests, independently of task workers")
    parser.add_argument("--multi-storage", action="store_true",
                        help="Place audit/compliance objects in a second native engine")
    parser.add_argument("--agent-directed", action="store_true",
                        help="Let the model choose queries through a bounded JSON tool loop")
    parser.add_argument('--stage-draft-on-propose',action='store_true',help='Opt-in Credit extension: stage private audit draft at propose, finalize at finish')
    tools = parser.add_mutually_exclusive_group()
    tools.add_argument("--tool-endpoint", help="Actual external verification HTTP endpoint")
    tools.add_argument("--local-test-tool", action="store_true",
                       help="Start an in-process local HTTP test service (NOT a production tool)")
    parser.add_argument("--allow-insecure-tool-http", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve previous captures")
    if args.mode == "live" and not args.model:
        parser.error("--model required for live mode")
    if args.mode == "replay" and not args.input:
        parser.error("--input required for replay mode")
    if args.mode == "replay" and (args.tool_endpoint or args.local_test_tool):
        parser.error("replay uses captured tool calls; never issue live tool requests")
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    if args.company_count is not None and args.company_count < 8:
        parser.error("--company-count must be at least 8")
    if args.mode == "replay" and args.company_count is not None:
        parser.error("replay must use the original captured company count")
    source = json.loads(args.input.read_text()) if args.input else None
    agent_directed = args.agent_directed or bool(source and source.get("agent_directed"))
    stage_draft=args.stage_draft_on_propose or bool(source and source.get('stage_draft_on_propose'))
    if stage_draft and not agent_directed:parser.error('Draft staging requires model-directed Credit')
    if agent_directed and args.mode == "live" and not (args.tool_endpoint or args.local_test_tool):
        parser.error("agent-directed capture requires a verification tool")
    workers = args.workers or (source.get("workers", 1) if source else 1)
    multi_storage = args.multi_storage or bool(source and source.get("multi_storage"))
    if workers < 1:
        parser.error("--workers must be positive")
    if args.max_inflight_llm is not None and (args.max_inflight_llm < 1 or args.mode != "live"):
        parser.error("--max-inflight-llm must be positive and is only valid in live mode")
    model_semaphore = (BoundedSemaphore(args.max_inflight_llm)
                       if args.max_inflight_llm is not None else None)
    config = (CreditReviewConfig(**source["config"]) if source else
              CreditReviewConfig(company_count=args.company_count
                                 if args.company_count is not None else 256))
    if args.company_id is not None and (args.mode != "live" or not 0 <= args.company_id < config.company_count):
        parser.error("company-id is a live-only registered fixture index")
    workload = CreditReviewWorkload(config)
    engines = None
    if multi_storage:
        core = load_cast_core()
        if not hasattr(core, "batch_put_across_stores"):
            parser.error("multi-storage mode requires a rebuilt native atomic coordinator")
        engines = {"default": core.Dbx1000VersionedKVStore(),
                   "audit": core.Dbx1000VersionedKVStore()}
    manager = AgentTransactionManager(storage_engines=engines,
                                      record_traces=False, collect_trajectories=False)
    storage_for = (lambda spec: "audit" if ":audit" in spec.object_id or
                   ":compliance:" in spec.object_id else "default") if multi_storage else None
    workload.register(manager, storage_for=storage_for)
    service = None
    tool_endpoint = args.tool_endpoint
    if args.mode == "live" and args.local_test_tool:
        from scripts.revision.credit_tool_service import make_server
        service = make_server()
        Thread(target=service.serve_forever, daemon=True).start()
        tool_endpoint = f"http://127.0.0.1:{service.server_port}/v1/verify"
    run_started = time.perf_counter()
    def run_task(sequence):
        previous = source["tasks"][sequence] if source else None
        task = (CreditReviewTaskSpec(**previous["task"]) if previous else
                workload.task_for(seed=args.seed, worker_id=sequence % workers,
                                  sequence=sequence // workers))
        if args.company_id is not None:
            task = dataclasses.replace(task, company_id=args.company_id)
        delegate = (ReplayReasoner(previous["llm_calls"]) if previous else
                    OpenAIResponsesReasoner(args.model, endpoint=args.endpoint,
                                            allow_insecure_http=args.allow_insecure_http,
                                            api_style=args.api_style,
                                            instructions=(AGENT_INSTRUCTIONS+('\nThe propose tool also stages a private audit draft in this transaction; verify and finish finalize it. The draft is not externally published before commit.\n' if stage_draft else '')) if agent_directed else None))
        if model_semaphore is not None:
            delegate = LimitedReasoner(delegate, model_semaphore)
        reasoner = CapturingReasoner(delegate)
        tool_records = previous.get("tool_calls", []) if previous else []
        tool_delegate = (ReplayTool(tool_records) if previous and tool_records else
                         HttpVerificationTool(tool_endpoint,
                             allow_insecure_http=(args.allow_insecure_tool_http or args.local_test_tool))
                         if tool_endpoint else None)
        tool = CapturingTool(tool_delegate) if tool_delegate is not None else None
        events = []
        cursor = (AgentDirectedCreditCursor(task, reasoner=reasoner, tool=tool, trace=events,stage_draft_on_propose=stage_draft)
                  if agent_directed else workload.cursor(task, reasoner=reasoner, tool=tool, trace=events))
        txn = manager.begin(f"credit-llm:{args.seed}:{sequence}",
                            transaction_metadata("silo", retry_count=0),
                            snapshot_object_ids=None, strategy="silo")
        started = time.perf_counter()
        started_offset_ms = (started-run_started)*1000
        try:
            execution = cursor.execute(txn)
            result = txn.commit("silo")
            status, reason = bool(result.committed), str(result.reason)
        except Exception as exc:
            if txn.result is None:
                txn.abort("credit-llm execution error", strategy="silo")
            if not agent_directed or previous:
                raise
            status, reason = False, f"{type(exc).__name__}: {exc}"
            execution = {"turns": len(reasoner.records), "failed": True}
        accesses = [e["object_id"] for e in events if e["event"] in {"read", "write"}]
        record = dict(task=dataclasses.asdict(task), committed=status, reason=reason,
                      started_offset_ms=started_offset_ms,
                      branch="model-directed" if agent_directed else execution.branch,
                      elapsed_ms=(time.perf_counter()-started)*1000,
                      operation_count=len(accesses) if agent_directed else execution.operation_count,
                      agent_execution=execution if agent_directed else None,
                      llm_call_count=len(reasoner.records),
                      input_tokens=sum(r["result"]["input_tokens"] for r in reasoner.records),
                      output_tokens=sum(r["result"]["output_tokens"] for r in reasoner.records),
                      llm_elapsed_ms=sum(r["elapsed_ms"] for r in reasoner.records),
                      llm_provider_elapsed_ms=sum(r.get("model_service_elapsed_ms",
                                                    r["elapsed_ms"]) for r in reasoner.records),
                      llm_admission_wait_ms=sum(r["result"].get("admission_wait_ms", 0.0)
                                                for r in reasoner.records),
                      access_targets=accesses if agent_directed else list(execution.revealed_targets), events=events,
                      llm_calls=reasoner.records, tool_calls=tool.records if tool else [])
        if previous and (record["access_targets"] != previous["access_targets"] or
                         record["branch"] != previous["branch"]):
            raise ValueError("replay produced a different access path")
        return record

    task_count = len(source["tasks"]) if source else args.tasks
    if workers == 1:
        records = [run_task(i) for i in range(task_count)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(run_task, range(task_count)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(mode=args.mode, model=args.model or source["model"],
                                           endpoint=args.endpoint if args.mode == "live" else source["endpoint"],
                                           api_style=args.api_style if args.mode == "live" else source["api_style"],
                                           config=dataclasses.asdict(config),
                                           workers=workers, multi_storage=multi_storage,
                                           agent_directed=agent_directed,
                                           stage_draft_on_propose=stage_draft,
                                           max_inflight_llm=args.max_inflight_llm if source is None else
                                               source.get("max_inflight_llm"),
                                           trace_schema_version=2,
                                           local_test_tool=bool(args.local_test_tool), tasks=records),
                                      indent=2, ensure_ascii=False) + "\n")
    if service is not None:
        service.shutdown()
        service.server_close()
    print(json.dumps({"tasks": len(records), "committed": sum(r["committed"] for r in records),
                      "llm_calls": sum(r["llm_call_count"] for r in records),
                      "tool_calls": sum(len(r["tool_calls"]) for r in records),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
