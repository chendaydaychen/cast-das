"""Join observed attempt episodes without changing rewards or policy inputs."""
from collections import defaultdict
from dataclasses import replace


def link_observed_retries(rows, *, allow_unobserved_attempts=False):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        task, attempt, generation = row.txn_id.rsplit(':', 2)
        attempt = int(attempt)
        if attempt != row.state.retry_count or int(generation) != 0:
            raise ValueError('Unexpected retry/generation identity')
        grouped[(row.source_id, task)].append((index, attempt))
    result, proof = list(rows), []
    for (source, task), positions in grouped.items():
        attempts = defaultdict(list)
        for index, attempt in positions:
            attempts[attempt].append(index)
        observed = sorted(attempts)
        missing = sorted(set(range(max(attempts) + 1)) - set(observed))
        if missing and not allow_unobserved_attempts:
            raise ValueError('Missing logical task attempt')
        if [a for _, a in positions] != sorted(a for _, a in positions):
            raise ValueError('Logical task rows not chronological')
        for attempt, indices in attempts.items():
            if sum(bool(rows[i].done) for i in indices) != 1 or not rows[indices[-1]].done:
                raise ValueError('Attempt requires exactly one final terminal')
            for index in indices:
                result[index] = replace(rows[index], txn_id=task)
            if attempt < max(attempts):
                terminal = indices[-1]
                if not rows[terminal].abort_indicator or rows[terminal].commit_indicator:
                    raise ValueError('Intermediate attempt must abort')
                result[terminal] = replace(result[terminal], done=False,
                    next_state=rows[attempts[observed[observed.index(attempt) + 1]][0]].state)
        proof.append(dict(source_id=source, task=task, attempts=len(attempts),
            intermediate_aborts=len(attempts) - 1,
            missing_attempt_indices=missing,
            cost_scope='observed decisions only; absent attempts have no imputed cost',
            final_committed=bool(rows[positions[-1][0]].commit_indicator)))
    return result, proof
