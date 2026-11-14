"""Same frozen input/timing pressure comparison with a version-history oracle."""
import argparse
import csv
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import discounted_returns, excluded_policy_transition, state_features
from agent.runtime import AgentTransactionManager, PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from agent.runtime.context import LockAction
from agent.runtime.paper_policy import LocalAccessFrequencyPolicy
from agent.runtime.residual_policy import ResidualTablePolicy
from scripts.revision.audit_serial_history import verify_history
try:
    runner = importlib.import_module("scripts.revision.compare_frozen_credit_trace_supported_20261010")
except ModuleNotFoundError as exc:
    if exc.name != "scripts.revision.compare_frozen_credit_trace_supported_20261010":
        raise
    runner = importlib.import_module("scripts.revision.compare_frozen_credit_trace")


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class AuditedManager(AgentTransactionManager):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_initial = {}
        self.audit_transactions = []
        self.audit_lock = threading.Lock()
        self.instances.append(self)

    def register_object(self, object_id, initial_value, **kwargs):
        super().register_object(object_id, initial_value, **kwargs)
        stored = self.store.get(str(object_id))
        self.audit_initial[digest(object_id)] = dict(version=int(stored.version), value_hash=digest(stored.value))

    def begin(self, *args, **kwargs):
        started = time.monotonic_ns()
        txn = super().begin(*args, **kwargs)
        txn.metadata["_audit_start_ns"] = started
        with self.audit_lock:
            self.audit_transactions.append(txn)
        return txn

    def commit(self, txn, **kwargs):
        result = super().commit(txn, **kwargs)
        txn.metadata["_audit_end_ns"] = time.monotonic_ns()
        return result

    def _record(self, txn):
        # Direct execution aborts do not pass through manager.commit(). Record
        # their terminal instant rather than substituting history export time.
        if txn.result is not None:
            txn.metadata.setdefault("_audit_end_ns", time.monotonic_ns())
        return super()._record(txn)

    def add_commit_timing(self, txn, phase, elapsed_ms):
        super().add_commit_timing(txn, phase, elapsed_ms)
        if getattr(self, "preserve_commit_timing", False):
            retained = txn.metadata.setdefault("_preserved_commit_timing_ms", {})
            retained[phase] = retained.get(phase, 0.) + max(0., float(elapsed_ms))

    def history(self):
        attempts = []
        for index, txn in enumerate(self.audit_transactions):
            attempts.append(dict(id=f"attempt-{index}", committed=bool(txn.result and txn.result.committed),
                start_ns=txn.metadata["_audit_start_ns"], end_ns=txn.metadata.get("_audit_end_ns", time.monotonic_ns()),
                reads={digest(key): dict(version=int(read.version), value_hash=digest(txn.snapshot[key].value))
                       for key, read in txn.read_set.items()},
                writes={digest(key): dict(version=int(write.base_version) + 1, value_hash=digest(write.value))
                        for key, write in txn.write_set.items()}))
        final = {}
        for key in self._catalog:
            stored = self.store.get(key)
            final[digest(key)] = dict(version=int(stored.version), value_hash=digest(stored.value))
        return dict(initial=self.audit_initial, attempts=attempts, final=final)


class LinearScorePolicy:
    """Fixed per-action linear return predictor, fitted once on training only."""
    def __init__(self, weights):
        self.weights = weights

    def select(self, state):
        features = state_features(state)
        current = int(state.current_action)
        allowed = [int(a) for a in self.weights if int(a) | current == int(a)]
        if not allowed:
            return LockAction.from_encoded(current)
        selected = min(allowed, key=lambda a: (-sum(w * x for w, x in zip(self.weights[str(a)], features)),
                                               a != current, a))
        return LockAction.from_encoded(selected)


def fit_linear(path, report_path, *, prior_batches=()):
    rows, batch_metadata = [], []
    batches = list(prior_batches) + [(path, report_path)]
    for index, (batch_path, batch_report_path) in enumerate(batches):
        raw = [policy_transition_from_dict(row, source_id=(
            f"linear-batch-{index}:" if prior_batches else "") + str(row.get("source_id") or batch_path.stem))
            for row in json.loads(batch_path.read_text())["transitions"]]
        report = json.loads(batch_report_path.read_text())
        rewarded, reward_report = apply_paper_rewards(raw, PaperRewardConfig(**report["reward"].get("config", {})))
        if prior_batches and reward_report != report["reward"]:
            raise ValueError("Historical linear rewards do not reproduce the recorded training report")
        rows.extend(rewarded)
        batch_metadata.append(dict(path=str(batch_path), transitions=len(raw),
            sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest()))
    return fit_linear_rewarded(rows, batch_metadata if prior_batches else None)


def fit_linear_rewarded(rows, batch_metadata=None, *, discount=.99):
    """Fit the existing ridge model without recomputing supplied rewards."""
    returns = discounted_returns(rows, discount)
    samples = {}
    for row, value in zip(rows, returns):
        if not excluded_policy_transition(row):
            samples.setdefault(str(row.action), []).append((state_features(row.state), value))
    weights = {}
    for action, data in samples.items():
        dimension = len(data[0][0])
        matrix = [[.01 if i == j else 0. for j in range(dimension)] for i in range(dimension)]
        rhs = [0.] * dimension
        for features, target in data:
            for i, x in enumerate(features):
                rhs[i] += x * target / len(data)
                for j, y in enumerate(features):
                    matrix[i][j] += x * y / len(data)
        # Same small ridge system as the existing base-linear experiment,
        # solved with pivoting to avoid adding a node1 NumPy dependency.
        augmented = [row + [value] for row, value in zip(matrix, rhs)]
        for i in range(dimension):
            pivot = max(range(i, dimension), key=lambda k: abs(augmented[k][i]))
            augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
            divisor = augmented[i][i]
            if abs(divisor) <= 1e-12:
                raise ValueError("singular ridge system")
            augmented[i] = [v / divisor for v in augmented[i]]
            for k in range(dimension):
                if k != i:
                    scale = augmented[k][i]
                    augmented[k] = [x - scale * y for x, y in zip(augmented[k], augmented[i])]
        weights[action] = [row[-1] for row in augmented]
    return dict(weights=weights, ridge=.01, discount=discount, features=37, training_transitions=len(rows),
                learning="training-only ridge fit, no RL, no learned context table, fixed coefficients at evaluation",
                **({"training_batches": batch_metadata, "round_reward_normalization_preserved": True}
                   if batch_metadata is not None else {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat-factor", type=int, default=8)
    parser.add_argument("--time-scale", type=float, default=.02)
    parser.add_argument("--arms", default="full,nophase,linear,frequency_always,frequency_threshold,silo,2pl-wait-die")
    args = parser.parse_args()
    if min(args.workers, args.repeat_factor) < 1 or args.time_scale <= 0:
        parser.error("positive scale, workers and repeat factor required")
    arms = args.arms.split(",")
    valid = {"full", "nophase", "linear", "frequency_always", "frequency_threshold", "silo", "2pl-wait-die"}
    if len(set(arms)) != len(arms) or set(arms) - valid:
        parser.error("unknown or duplicated arm")
    args.output.mkdir(exist_ok=False, parents=True)
    capture_path = ROOT / "results/credit_llm/fresh_live_20261011_v1/evaluation_live.json"
    source = json.loads(capture_path.read_text())
    training = ROOT / "results/credit_llm/fresh_coverage_20261011"
    policy_folder = ROOT / "results/credit_llm/residual_20261011"
    linear = fit_linear(training / "training.json", training / "full_report.json")
    (args.output / "linear_policy.json").write_text(json.dumps(linear, indent=2))
    policies = {arm: ResidualTablePolicy.from_dict(json.loads((policy_folder / f"coverage_{arm}_policy.json").read_text()))
                for arm in ("full", "nophase")}
    policies.update(linear=LinearScorePolicy(linear["weights"]),
                    frequency_always=LocalAccessFrequencyPolicy(1),
                    frequency_threshold=LocalAccessFrequencyPolicy(8))
    protocol = dict(capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(), workers=args.workers,
        repeat_factor=args.repeat_factor, tasks=len(source["tasks"]) * args.repeat_factor, time_scale=args.time_scale,
        max_attempts=3, arms=arms, new_api_calls=0, independent_evaluation_captures=len(source["tasks"]),
        policy_sha256={arm: hashlib.sha256(json.dumps(p.to_dict(), sort_keys=True).encode()).hexdigest()
                       for arm, p in policies.items() if isinstance(p, ResidualTablePolicy)},
        rule="one-second recent per-key counts; maximum among observed keys >= 1 or 8 triggers observed protection classes",
        rule_caveat="existing LocalAccessFrequencyPolicy is a class-mask rule, not exact per-item locking; threshold fixed before evaluation",
        baseline_caveat="Silo and 2PL are shared-runtime adapters, not native paper implementations",
        replay_caveat="repeated two independent live captures, no new online API calls; small pressure pilot",
        audit="hash-only version/value source, final state and real-time serialization graph; instrumentation in all arms")
    protocol["runner_source_sha256"] = hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    protocol["runner_module"] = runner.__name__
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    runner.AgentTransactionManager = AuditedManager
    rows = []
    for arm in arms:
        policy = policies.get(arm)
        decisions, attempts = [], []
        row = runner.run_system(source, "paper-atcc" if policy else arm, workers=args.workers,
            max_attempts=3, time_scale=args.time_scale, repeat_factor=args.repeat_factor, policy=policy,
            decision_sink=decisions if policy else None, attempt_sink=attempts)
        history = AuditedManager.instances[-1].history()
        (args.output / f"{arm}_history.json").write_text(json.dumps(history, indent=2))
        try:
            verdict = verify_history(**history)
        except AssertionError as exc:
            verdict = dict(verdict="failed", error=str(exc), scope="do not use this arm as correct performance evidence")
        (args.output / f"{arm}_oracle.json").write_text(json.dumps(verdict, indent=2))
        (args.output / f"{arm}_attempts.json").write_text(json.dumps(attempts, indent=2))
        (args.output / f"{arm}_decisions.json").write_text(json.dumps(dict(transitions=decisions), indent=2))
        row.update(arm=arm, oracle_verdict=verdict["verdict"])
        rows.append(row)
        with (args.output / "runtime.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(dict(result=row, oracle=verdict)), flush=True)
    if any(row["oracle_verdict"] != "verified" for row in rows):
        raise SystemExit("history oracle failed; results retained for diagnosis")


if __name__ == "__main__":
    main()
