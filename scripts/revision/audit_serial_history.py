"""Check serializability of a fixed-key, monotonically versioned KV history.

Input contains only opaque key/value hashes. This checks the recorded database
history, not external tool effects, crash recovery, or all possible schedules.
"""
from collections import defaultdict, deque


def verify_history(initial, attempts, final):
    if set(initial) != set(final):
        raise AssertionError("initial/final key domains differ")
    committed = [record for record in attempts if record["committed"]]
    identifiers = [record["id"] for record in attempts]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("duplicate attempt identifier")
    writers = defaultdict(dict)
    for index, record in enumerate(committed):
        if record["start_ns"] > record["end_ns"]:
            raise AssertionError("negative committed attempt interval")
        for key, write in record["writes"].items():
            if key not in initial:
                raise AssertionError("write outside registered domain")
            version = int(write["version"])
            if version <= int(initial[key]["version"]) or version in writers[key]:
                raise AssertionError("duplicate or non-increasing committed write version")
            writers[key][version] = (index, write["value_hash"])
    edges = set()
    reasons = defaultdict(int)

    def edge(left, right, reason):
        if left == right:
            return
        if (left, right) not in edges:
            edges.add((left, right))
            reasons[reason] += 1

    for key, base in initial.items():
        versions = sorted(writers[key])
        expected = list(range(int(base["version"]) + 1, int(base["version"]) + len(versions) + 1))
        if versions != expected:
            raise AssertionError("missing or leaked committed version")
        if int(final[key]["version"]) != int(base["version"]) + len(versions):
            raise AssertionError("final version inconsistent with committed writes")
        expected_hash = writers[key][versions[-1]][1] if versions else base["value_hash"]
        if final[key]["value_hash"] != expected_hash:
            raise AssertionError("final value inconsistent with last committed writer")
        for previous, following in zip(versions, versions[1:]):
            edge(writers[key][previous][0], writers[key][following][0], "write-write")
    read_count = 0
    for index, record in enumerate(committed):
        for key, read in record["reads"].items():
            read_count += 1
            if key not in initial:
                raise AssertionError("read outside registered domain")
            version = int(read["version"])
            if version == int(initial[key]["version"]):
                expected_hash = initial[key]["value_hash"]
            elif version in writers[key]:
                producer, expected_hash = writers[key][version]
                if producer == index:
                    raise AssertionError("history format does not represent read-your-writes")
                edge(producer, index, "write-read")
            else:
                raise AssertionError("read observes uncommitted or missing version")
            if read["value_hash"] != expected_hash:
                raise AssertionError("read value inconsistent with recorded version")
            next_versions = [v for v in writers[key] if v > version]
            if next_versions:
                edge(index, writers[key][min(next_versions)][0], "read-write")
        for other_index, other in enumerate(committed):
            if record["end_ns"] < other["start_ns"]:
                edge(index, other_index, "real-time")
    incoming = [0] * len(committed)
    outgoing = defaultdict(list)
    for left, right in edges:
        incoming[right] += 1
        outgoing[left].append(right)
    queue = deque(i for i, degree in enumerate(incoming) if not degree)
    order = []
    while queue:
        index = queue.popleft()
        order.append(index)
        for following in outgoing[index]:
            incoming[following] -= 1
            if not incoming[following]:
                queue.append(following)
    if len(order) != len(committed):
        raise AssertionError("serialization dependency cycle")
    return dict(verdict="verified", registered_keys=len(initial), attempts=len(attempts),
        commits=len(committed), aborted_attempts=len(attempts) - len(committed),
        committed_reads=read_count, committed_writes=sum(len(r["writes"]) for r in committed),
        dependency_edges=len(edges), first_edge_reasons=dict(reasons),
        serialization_order=[committed[i]["id"] for i in order],
        scope="recorded version/value history and real-time serializability, fixed key domain; not external effects or durability")
