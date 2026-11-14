#!/usr/bin/env python3
"""Small native two-store CAS/transfer integrity test, not a performance ranking."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from threading import Barrier
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.native import load_cast_core
from agent.runtime.storage_router import RoutedVersionedKVStore


def exercise(core, workers, per_worker, pairs, mode):
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore()
    routed = RoutedVersionedKVStore({"default": first, "audit": second})
    initial = 100000
    for pair in range(pairs):
        routed.register_route(f"left:{pair}")
        routed.register_route(f"right:{pair}", "audit")
        routed.put(f"left:{pair}", str(initial))
        routed.put(f"right:{pair}", "0")
    barrier = Barrier(workers, timeout=15)

    def publish(pair, left, right, writes=True, reverse=False):
        left_key, right_key = f"left:{pair}", f"right:{pair}"
        checks = ((left_key, left.version), (right_key, right.version))
        updates = ((left_key, str(int(left.value)-1)), (right_key, str(int(right.value)+1))) if writes else ()
        if mode == "routed":
            return routed.batch_put_if_version(checks, updates)
        batches = [(first, [checks[0]], [updates[0]] if writes else []),
                   (second, [checks[1]], [updates[1]] if writes else [])]
        return core.batch_put_across_stores(list(reversed(batches)) if reverse else batches)

    def worker(index):
        pair = index % pairs
        attempts = commits = conflicts = valid_observations = 0
        left, right = routed.get(f"left:{pair}"), routed.get(f"right:{pair}")
        barrier.wait()
        while commits < per_worker:
            attempts += 1
            if attempts > per_worker * 1000:
                raise RuntimeError("transfer retry cap exceeded")
            if publish(pair, left, right, reverse=bool(index % 2)):
                commits += 1
            else:
                conflicts += 1
                time.sleep(0)
            left, right = routed.get(f"left:{pair}"), routed.get(f"right:{pair}")
            if publish(pair, left, right, writes=False, reverse=bool(index % 2)):
                valid_observations += 1
                if int(left.value) + int(right.value) != initial:
                    raise AssertionError("validated snapshot contains a partial cross-store transfer")
        return dict(commits=commits, attempts=attempts, conflicts=conflicts,
                    validated_observations=valid_observations)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(worker, range(workers)))
    elapsed = time.perf_counter() - started
    final = []
    for pair in range(pairs):
        expected = sum(per_worker for worker_index in range(workers) if worker_index % pairs == pair)
        left, right = routed.get(f"left:{pair}"), routed.get(f"right:{pair}")
        if (int(left.value), int(right.value)) != (initial-expected, expected):
            raise AssertionError("lost or partial transfer in final account balances")
        if (left.version, right.version) != (expected+1, expected+1):
            raise AssertionError("cross-store versions disagree with committed transfers")
        # A stale version in the second store must leave both values untouched.
        before = (left.value, left.version, right.value, right.version)
        stale = type("Stale", (), dict(value=right.value, version=right.version-1))()
        if publish(pair, left, stale):
            raise AssertionError("stale second-store version unexpectedly committed")
        after_left, after_right = routed.get(f"left:{pair}"), routed.get(f"right:{pair}")
        if before != (after_left.value, after_left.version, after_right.value, after_right.version):
            raise AssertionError("failed version check partially published")
        final.append(dict(pair=pair, committed_transfers=expected, left=int(left.value),
                          right=int(right.value), version=left.version))
    return dict(mode=mode, workers=workers, transfers_per_worker=per_worker, pairs=pairs,
                commits=sum(r["commits"] for r in records), attempts=sum(r["attempts"] for r in records),
                conflicts=sum(r["conflicts"] for r in records),
                validated_observations=sum(r["validated_observations"] for r in records),
                elapsed_seconds=elapsed, final_accounts=final,
                stale_second_store_checks_passed=pairs, status="ok")


def invalid_write_check(core):
    first, second = core.Dbx1000VersionedKVStore(), core.Dbx1000VersionedKVStore(max_value_bytes=3)
    first.put("left", "one")
    second.put("right", "two")
    try:
        core.batch_put_across_stores([(first, [("left", 1)], [("left", "changed")]),
                                     (second, [("right", 1)], [("right", "too-long")])])
    except ValueError:
        pass
    else:
        raise AssertionError("oversize second-store value was not rejected")
    if (first.get("left").value, first.get("left").version,
        second.get("right").value, second.get("right").version) != ("one", 1, "two", 1):
        raise AssertionError("invalid value caused partial publication")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--transfers-per-worker", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    if min(args.workers, args.transfers_per_worker, args.pairs) < 1:
        parser.error("all sizes must be positive")
    core = load_cast_core()
    if not hasattr(core, "batch_put_across_stores"):
        raise RuntimeError("native multi-store coordinator unavailable")
    result = dict(scope="in-process two-DBx1000 storage integrity; not durability/distributed recovery",
                  cases=[exercise(core, args.workers, args.transfers_per_worker, args.pairs, mode)
                         for mode in ("routed", "direct_native")],
                  oversize_second_store_prepublication_check=invalid_write_check(core),
                  native_module_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
