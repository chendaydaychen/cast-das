"""Check recorded call timing and ordered live/replay intent equivalence."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def duration(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {label}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}")
    return float(value)


def intent_signature(event):
    fields = ("event", "phase", "label", "object_id", "value", "storage_id",
              "tool", "transport", "action", "result")
    return {key: event[key] for key in fields if key in event}


def audit(capture, replay=None):
    if capture.get("mode") != "live" or not capture.get("tasks"):
        raise ValueError("Nonempty live capture required")
    if replay is not None and (replay.get("mode") != "replay"
                              or len(replay.get("tasks", [])) != len(capture["tasks"])):
        raise ValueError("Paired replay tasks required")
    totals = Counter()
    tasks = []
    for index, task in enumerate(capture["tasks"]):
        calls = task.get("llm_calls", [])
        tools = task.get("tool_calls", [])
        events = task.get("events", [])
        if not calls or not events:
            raise ValueError("Recorded calls and ordered events required")
        previous = 0.0
        for event in events:
            offset = duration(event.get("elapsed_ms"), "event offset")
            if offset < previous:
                raise ValueError("Event offsets must be nondecreasing")
            previous = offset
        llm_events = [event for event in events if event["event"] == "llm"]
        if [(event.get("label"), event.get("phase")) for event in llm_events] != [
                (call.get("label"), call.get("phase")) for call in calls]:
            raise ValueError("LLM event/call order differs")
        if len([event for event in events if event["event"] == "external_tool"]) != len(tools):
            raise ValueError("External tool event/call count differs")
        measured = Counter()
        for call in calls:
            elapsed = duration(call.get("elapsed_ms"), "LLM elapsed time")
            service = duration(call.get("model_service_elapsed_ms"), "LLM service time")
            admission = duration(call.get("result", {}).get("admission_wait_ms"),
                                 "LLM admission wait")
            if not math.isclose(elapsed, service + admission, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("LLM elapsed time does not equal service plus admission")
            measured.update(llm_elapsed_ms=elapsed, provider_service_ms=service,
                            captured_admission_ms=admission)
        measured["tool_elapsed_ms"] = sum(duration(call.get("elapsed_ms"), "tool elapsed time")
                                          for call in tools)
        task_elapsed = duration(task.get("elapsed_ms"), "task elapsed time")
        gap = task_elapsed - measured["llm_elapsed_ms"] - measured["tool_elapsed_ms"]
        if gap < -1e-6 or previous > task_elapsed + 1e-6:
            raise ValueError("Sequential call/event time exceeds task elapsed time")
        measured.update(other_task_time_ms=max(0.0, gap), task_elapsed_ms=task_elapsed)
        totals.update(measured)
        if replay is not None:
            other = replay["tasks"][index]
            if task["task"] != other["task"]:
                raise ValueError("Replay task identity differs")
            if [intent_signature(event) for event in events] != [
                    intent_signature(event) for event in other.get("events", [])]:
                raise ValueError("Ordered replay intents differ")
            if [(call["label"], call["phase"], call["context"], call["result"]) for call in calls] != [
                    (call["label"], call["phase"], call["context"], call["result"])
                    for call in other.get("llm_calls", [])]:
                raise ValueError("Replay model inputs/results differ")
            if [(call["request"], call["response"]) for call in tools] != [
                    (call["request"], call["response"]) for call in other.get("tool_calls", [])]:
                raise ValueError("Replay tool inputs/results differ")
        tasks.append(dict(index=index, committed=task.get("committed"),
                          llm_calls=len(calls), tool_calls=len(tools), **measured))
    return dict(verdict="verified", tasks=tasks, totals=dict(totals),
        ordered_replay_intents_identical=replay is not None,
        replay_timing_note="Replay measured sleeps have scheduling overhead; exact measured-time equality is not claimed",
        delay_scope="Frozen cursor replays whole recorded LLM elapsed time, including captured admission waits",
        admission_note="Recorded admission delay is not CC lock waiting or provider-only reasoning time",
        scope="Recorded sequential timing accounting and intent order; not online replanning or retry correctness")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous audits")
    report = audit(json.loads(args.capture.read_text()),
                   json.loads(args.replay.read_text()) if args.replay else None)
    report["capture_sha256"] = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    if args.replay:
        report["replay_sha256"] = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
