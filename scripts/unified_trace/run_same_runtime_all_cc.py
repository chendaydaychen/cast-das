#!/usr/bin/env python3
"""Historical commit-adapter runner retained for reproducing archived results.

For corrected timestamp TicToc, operation-level 2PL, explicit adapter scopes,
and per-task tail samples use run_corrected_tail_matrix_20261106.py. This
module intentionally retains the old routing used by earlier CSV artifacts.

The learned/MVCC managers are selected in-process; this deliberately avoids
the historical adapter matrix and keeps trace parsing, retry accounting,
latency measurement, and result serialization in run_castdas_trace_fair.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.unified_trace import run_castdas_trace_fair as fair
CORE = ("occ", "2pl-nowait", "mvcc", "silo", "tictoc", "bamboo", "polaris", "paper-atcc")
ADAPTERS = ("mocc", "hekaton", "polyjuice", "neurcc")
LABELS = {
    "occ": "OCC", "2pl-nowait": "2PL", "mvcc": "MVCC", "silo": "Silo",
    "tictoc": "TicToc", "bamboo": "Bamboo", "polaris": "Polaris",
    "paper-atcc": "Aegis Full", "mocc": "MOCC", "hekaton": "Hekaton",
    "polyjuice": "Polyjuice", "neurcc": "NeurCC",
}


def adapter_class(name):
    if name == "mocc":
        from agent.runtime.mocc_runtime import MoccRuntimeManager
        return MoccRuntimeManager
    if name == "hekaton":
        from agent.runtime.hekaton_runtime import HekatonRuntimeManager
        return HekatonRuntimeManager
    from agent.runtime.audited_learned_adapter import AuditedLearnedAdapter
    return AuditedLearnedAdapter


def run_one(rows, *, name, policy, learned_text, max_attempts, execution_workers,
            measure_seconds, phase_sensitive=False):
    manager_type = None
    old = fair.AgentTransactionManager
    kwargs = dict(max_attempts=max_attempts, tokens_per_operation=(64 if phase_sensitive else 2703),
                  execution_workers=execution_workers, measure_seconds=measure_seconds,
                  cycle_trace=(False if phase_sensitive else True), paper_priority="enabled",
                  paper_delayed_write_apply="enabled", paper_performance_guards="enabled",
                  paper_deferred_replay_enabled=True, atcc_retry_cache_enabled=False,
                  tpcc_replay_capacity=1, ycsb_replay_capacity=1)
    if phase_sensitive:
        # The main Agent-YCSB curve uses the same observed-cost priority
        # contract as the phase-sensitive training run.  This is enabled only
        # for Aegis; traditional/adapted CC managers remain unchanged.
        kwargs.update(paper_inflight_priority=True,
                      paper_observed_contract=False,
                      paper_deferred_replay_enabled=False,
                      paper_cost_aware_observed_read_protection=False)
    if name == "paper-atcc":
        if phase_sensitive:
            # Reuse the same concrete-object protection manager as the
            # validated phase-sensitive YCSB run.  This is a runtime semantic
            # setting for Aegis, not an ablation arm.
            from scripts.revision.run_ycsb_fresh_dynamic_2401 import NoLowRiskGuardManager
            manager_type = NoLowRiskGuardManager
            fair.AgentTransactionManager = manager_type
        kwargs.update(paper_policy=policy)
        if phase_sensitive:
            kwargs.update(paper_deferred_replay_enabled=False,
                          atcc_retry_cache_enabled=False,
                          paper_performance_guards="disabled",
                          paper_delayed_write_apply="disabled",
                          paper_inflight_priority=True,
                          trajectory_output=Path("/tmp/aegis_phase_sensitive_trace.json"))
        try:
            row = fair.run_trace(rows, cc=name, policy=None, **kwargs)
        finally:
            fair.AgentTransactionManager = old
    elif name in ADAPTERS:
        manager_type = adapter_class(name)
        if name in ("polyjuice", "neurcc"):
            from agent.runtime.learned_policy_adapters import AdaptedLearnedPolicy
            p = AdaptedLearnedPolicy.polyjuice(learned_text) if name == "polyjuice" else AdaptedLearnedPolicy.neurcc(learned_text)
            class BoundLearnedAdapter(manager_type):
                def __init__(self, **kw):
                    super().__init__(speculative_policy=p, **kw)
            manager_type = BoundLearnedAdapter
        fair.AgentTransactionManager = manager_type
        kwargs.update(paper_policy=None, paper_priority="disabled",
                      paper_delayed_write_apply="disabled",
                      paper_performance_guards="disabled",
                      paper_deferred_replay_enabled=False, atcc_retry_cache_enabled=False)
        try:
            row = fair.run_trace(rows, cc="silo", policy=None, **kwargs)
        finally:
            fair.AgentTransactionManager = old
    else:
        row = fair.run_trace(rows, cc=name, policy=None, **kwargs)
    row["cc"] = LABELS[name]
    row["same_runtime_manager"] = name
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--paper-policy", type=Path, required=True)
    ap.add_argument("--learned-policy-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--execution-workers", type=int, required=True)
    ap.add_argument("--measure-seconds", type=float, default=0.0)
    ap.add_argument("--phase-sensitive", action="store_true",
                    help="use the observed-cost priority contract for Aegis on Agent-YCSB phase traces")
    args = ap.parse_args()
    rows = fair.read_trace(args.trace)
    learned = {}
    for n in ("polyjuice", "neurcc"):
        p = args.learned_policy_dir / f"{n}.txt"
        learned[n] = p.read_text(encoding="utf-8")
    out = []
    for n in CORE + ADAPTERS:
        out.append(run_one(rows, name=n, policy=args.paper_policy,
                           learned_text=learned.get(n, ""),
                           max_attempts=args.max_attempts,
                           execution_workers=args.execution_workers,
                           measure_seconds=args.measure_seconds,
                           phase_sensitive=args.phase_sensitive))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fair.FIELDS) + ["same_runtime_manager"]
    with args.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for row in out: w.writerow({k: row.get(k, "") for k in fields})
    print(args.output)


if __name__ == "__main__":
    main()
