"""Merge complete training captures verbatim, preserving evaluation and provenance."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


COMPATIBILITY_FIELDS = (
    "mode", "model", "api_style", "config", "multi_storage", "agent_directed",
    "trace_schema_version", "local_test_tool",
)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_capture(capture):
    if capture.get("mode") != "live" or capture.get("agent_directed") is not True:
        raise ValueError("Completed model-directed live captures required")
    tasks = capture.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Nonempty capture required")
    for task in tasks:
        if task.get("committed") is not True:
            raise ValueError("Incomplete capture cannot silently enter training")
        company = task["task"]["company_id"]
        if type(company) is not int or company < 0:
            raise ValueError("Invalid company identity")
        events = task["events"]
        accesses = [e["object_id"] for e in events if e["event"] in {"read", "write"}]
        if accesses != task["access_targets"] or len(accesses) != task["operation_count"]:
            raise ValueError("Captured accesses are inconsistent")
        for event, field in (("llm", "llm_calls"), ("external_tool", "tool_calls")):
            calls = task[field]
            if sum(e["event"] == event for e in events) != len(calls):
                raise ValueError("Captured call stream is incomplete")
            for call in calls:
                elapsed = call["elapsed_ms"]
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not (
                        math.isfinite(elapsed) and elapsed >= 0):
                    raise ValueError("Captured duration must be finite and nonnegative")
        if sum(e["event"] == "agent_tool" for e in events) != len(task["llm_calls"]):
            raise ValueError("Model actions and calls are inconsistent")
        digest(task)


def merge_captures(training, evaluation):
    if not training:
        raise ValueError("At least one training source required")
    for capture in (*training, evaluation):
        validate_capture(capture)
        for key in COMPATIBILITY_FIELDS:
            if key not in capture or capture[key] != training[0].get(key):
                raise ValueError(f"Incompatible capture field: {key}")
    held_out = {t["task"]["company_id"] for t in evaluation["tasks"]}
    tasks, origins, seen = [], [], set()
    for source_index, capture in enumerate(training):
        for task_index, task in enumerate(capture["tasks"]):
            if task["task"]["company_id"] in held_out:
                raise ValueError("Training/evaluation company overlap")
            sha = digest(task)
            if sha in seen:
                raise ValueError("Duplicate training task; do not inflate path coverage")
            seen.add(sha)
            tasks.append(copy.deepcopy(task))
            origins.append(dict(source_index=source_index, task_index=task_index,
                                task_sha256=sha, company_id=task["task"]["company_id"]))
    merged = copy.deepcopy(training[0])
    merged["tasks"] = tasks
    merged["mode"] = "live"
    return merged, origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve() for p in args.training]
    evaluation = args.evaluation.resolve()
    if len(set(paths)) != len(paths) or evaluation in paths:
        parser.error("Unique training files distinct from evaluation required")
    if any(p.name != "train_live.json" for p in paths):
        parser.error("Training sources must be train_live.json, not evaluation captures")
    if evaluation.name != "evaluation_live.json":
        parser.error("Evaluation source must be evaluation_live.json")
    raw = [p.read_bytes() for p in paths]
    evaluation_bytes = evaluation.read_bytes()
    merged, origins = merge_captures([json.loads(b) for b in raw], json.loads(evaluation_bytes))
    payload = json.dumps(merged, indent=2, ensure_ascii=True, allow_nan=False).encode()
    manifest = dict(
        training_sources=[dict(path=str(p), sha256=hashlib.sha256(b).hexdigest())
                          for p, b in zip(paths, raw)],
        evaluation_source=dict(path=str(evaluation), sha256=hashlib.sha256(evaluation_bytes).hexdigest()),
        task_origins=origins,
        merged_training_sha256=hashlib.sha256(payload).hexdigest(),
        evaluation_preserved_byte_for_byte=True,
        task_payloads_preserved=True,
        training_templates=len(origins),
        training_companies=sorted({o["company_id"] for o in origins}),
        collection_metadata="capture headers inherited from first training source; not a new live run",
        experiment_role="exploratory training diversity expansion; previously observed evaluation reused",
        scope="existing captured paths combined without phase, access or duration edits; no new API calls",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "train_live.json").write_bytes(payload)
    (args.output / "evaluation_live.json").write_bytes(evaluation_bytes)
    (args.output / "capture_merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(training_templates=len(origins),
                         training_companies=manifest["training_companies"],
                         training_sha256=manifest["merged_training_sha256"],
                         evaluation_sha256=manifest["evaluation_source"]["sha256"])))


if __name__ == "__main__":
    main()
