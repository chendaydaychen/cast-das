#!/usr/bin/env python3
"""Bounded P0 supplement. All measurements must run on node1.

No paper outputs are overwritten. Full/NoPhase share behavior data and PPO
configuration, but are independently trained. Explicit phase is the only ablation.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOKUP_OPTIMIZED = False
sys.path.insert(0, str(ROOT))
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy, StaticThresholdPhasePolicy
from agent.runtime.context import LockAction, LockClass
from agent.runtime.state_collector import phase_aware_state_from_dict, policy_observation
from agent.cc.atcc.ppo import state_key, policy_group_key, state_features
from agent.workloads.credit_review import CreditReviewConfig
from scripts.unified_trace.run_credit_review_experiment import run_system


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader()
        w.writerows(rows)


class MeteredPolicy:
    def __init__(self, base):
        self.base = base
        self.reset()

    def reset(self):
        self.calls = self.ns = self.changes = self.up = self.down = 0
        self.phase_actions = Counter()

    def select(self, state):
        started = time.perf_counter_ns()
        result = self.base.select(state)
        self.ns += time.perf_counter_ns() - started
        action = int(result.protected)
        self.calls += 1
        self.changes += action != state.current_action
        self.up += state.current_action == 0 and action != 0
        self.down += state.current_action != 0 and action == 0
        self.phase_actions[(state.phase, action)] += 1
        return result


class FrozenPolicy:
    """Same Full artifact, freeze the first observed decision per attempt.

    Retains identical OCC startup/checkpoint schedule; no phase-dependent updates.
    Runtime's mandatory monotonic protection and retry safety are unchanged.
    """
    def __init__(self, base):
        self.base = base

    def select(self, state):
        if state.current_action:
            return LockAction(LockClass(state.current_action))
        # In CreditReview the first decision is after the Explore batch.
        if state.completed_rounds > 1:
            return LockAction()
        return self.base.select(state)


class ConstantProtectionPolicy:
    """Behavior sampler anchor, never a learned/evaluation policy."""
    def __init__(self, mask):
        self.mask=mask

    def select(self,state):
        return LockAction(LockClass(int(state.current_action)|self.mask))


def policy_for(out, variant, threshold):
    if variant == "silo":
        return None
    if variant == "hotrule_always":
        return HotRulePolicy()
    if variant == "hotrule_threshold":
        return HotRulePolicy(threshold)
    if variant == "static_threshold":
        return StaticThresholdPhasePolicy()
    base = CompiledPhasePolicy.load(out / "training" / ("nophase.json" if variant == "nophase" else "full.json"))
    base.prebucketed_lookup = LOOKUP_OPTIMIZED
    return FrozenPolicy(base) if variant == "frozen_full" else base


def run(out, variant, clients, seed, repeat=0, threshold=.1, config=None,
        override=None, trajectory=None, exploration=None, measure=2., warmup=.2):
    base = override if override is not None else policy_for(out, variant, threshold)
    meter = MeteredPolicy(base) if base is not None and exploration is None else None
    row = run_system(config=config or CreditReviewConfig(), clients=clients, seed=seed,
        repeat=repeat, system="silo" if variant == "silo" else "paper-atcc",
        paper_policy=out / "training/full.json", policy_override=meter or base,
        warmup_seconds=warmup, measure_seconds=measure, max_attempts=6,
        paper_priority="disabled" if variant == "no_priority" else "enabled",
        performance_guards=False, policy_commit_batch=True,
        trajectory_output=trajectory, paper_exploration_seed=exploration,
        paper_exploration_epsilon=.3)
    row.update(variant=variant, lookup_optimized=LOOKUP_OPTIMIZED,
               config_json=json.dumps(dataclasses.asdict(config or CreditReviewConfig())))
    if meter:
        row.update(policy_lookup_mean_us=meter.ns/max(1, meter.calls)/1000,
            policy_calls=meter.calls, decisions_per_task=meter.calls/max(1, row["logical_tasks"]),
            changes_per_task=meter.changes/max(1, row["logical_tasks"]),
            decisions_per_attempt=meter.calls/max(1, row["attempts"]),
            occ_to_protected=meter.up, protected_to_occ=meter.down,
            phase_actions_json=json.dumps({f"{p}:{a}": n for (p,a),n in meter.phase_actions.items()}))
    assert row["attempts"] == row["commits"] + row["aborts"]
    assert row["logical_tasks"] == row["commits"] + row["failed_tasks"]
    print(json.dumps({k:row[k] for k in ("variant", "clients", "repeat", "seed", "elapsed_task_tps", "abort_rate", "p99_latency_ms")}), flush=True)
    return row


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("level", "default"), r["variant"], int(r["clients"]))].append(r)
    metrics = ("elapsed_task_tps", "agent_tps", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "abort_rate", "commit_rate", "task_completion_rate", "wasted_reasoning_ms_per_commit",
        "failed_tasks", "logical_tasks", "commits", "aborts", "attempts", "drain_seconds",
        "wasted_tokens_per_commit", "blocked_ms_total", "lock_hold_ms_total", "paper_lock_wait_ms",
        "policy_lookup_mean_us", "decisions_per_task", "decisions_per_attempt", "changes_per_task")
    output=[]
    for (level,variant,clients), group in sorted(groups.items()):
        row=dict(level=level, variant=variant, clients=clients, repeats=len(group))
        for key in metrics:
            values=[float(r[key]) for r in group if r.get(key) not in (None,"")]
            if values:
                row[key+"_mean"]=statistics.fmean(values)
                row[key+"_std"]=statistics.stdev(values) if len(values)>1 else 0.
        output.append(row)
    return output


def train(out, extended=False, reuse=None):
    paths=[]
    collection=[]
    started=time.perf_counter()
    for clients in (8,24,40):
        path=out / "training" / f"behavior_c{clients}.json"
        if reuse is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(reuse/f"behavior_c{clients}.json",path)
        else:
            collection.append(run(out,"behavior",clients,6100000+clients,override=HotRulePolicy(.1),
                trajectory=path,exploration=6110000+clients,measure=2.,warmup=0.))
        paths.append(path)
    if extended:
        # Broaden training action support, rather than fit the evaluation seeds.
        # Identical added behavior data and PPO settings for Full and NoPhase.
        for mask in (0,5,15):
            path=out/"training"/f"behavior_anchor{mask}.json"
            collection.append(run(out,"behavior",24,6810000+mask,override=ConstantProtectionPolicy(mask),
                trajectory=path,exploration=6820000+mask,measure=3.,warmup=0.))
            paths.append(path)
    csv_write(out / "training/collection.csv",collection)
    collection_seconds=time.perf_counter()-started
    commands=[]
    for name, awareness in (("full","enabled"),("nophase","disabled")):
        cmd=[sys.executable,str(ROOT/"scripts/train_paper_atcc.py"),"--output",str(out/f"training/{name}.json"),
             "--report",str(out/f"training/{name}_report.json"),"--generation","1","--seed","6120000",
             "--epochs","4","--phase-awareness",awareness]
        for path in paths:
            cmd.extend(("--trajectory",str(path)))
        subprocess.run(cmd,check=True)
        commands.append(cmd)
    policy=CompiledPhasePolicy.load(out/"training/nophase.json")
    checks=0
    for path in paths:
        for t in json.loads(path.read_text())["transitions"]:
            state=phase_aware_state_from_dict(t["state"])
            outputs=[]
            for phase in ("explore","refine","commit"):
                actual=dataclasses.replace(state,phase=phase)
                observed=policy_observation(actual,"disabled")
                assert state_features(observed)[1:4] == (0.,0.,0.)
                outputs.append((state_key(observed),policy_group_key(observed),state_features(observed),int(policy.select(actual).protected)))
            assert outputs[0]==outputs[1]==outputs[2]
            checks+=1
    dump(out/"training/policy_audit.json",dict(explicit_phase_invariance_checks=checks,
        compiled_phases=sorted({e.phase for e in policy.entries}),collection_seconds=collection_seconds,
        full_sha256=digest(out/"training/full.json"),nophase_sha256=digest(out/"training/nophase.json"),
        independent_training=True,common_behavior_data=True,commands=commands,
        caveat="Observed access/progress/cost features remain; only explicit phase labels are ablated."))


def main():
    global LOOKUP_OPTIMIZED
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=("train","train-supported","validate","calibrate","smoke","evaluate","low","behavior"),required=True)
    p.add_argument("--clients",default="8,16,24,32,40")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--measure",type=float,default=2.)
    p.add_argument("--optimized-lookup",action="store_true")
    p.add_argument("--reuse-behavior",type=Path)
    p.add_argument("--reference-output",type=Path)
    p.add_argument("--eval-seed-base",type=int,default=6900000)
    a=p.parse_args()
    LOOKUP_OPTIMIZED=a.optimized_lookup
    if platform.node().split(".")[0] != "node1":
        raise SystemExit("Measurements must run on node1 via WSL SSH")
    out=a.output.resolve()
    destinations={"train":"training/full.json","train-supported":"training/full.json",
        "calibrate":"calibration/raw.csv","smoke":"smoke/raw.csv","validate":"validation/raw.csv",
        "evaluate":"evaluate/raw.csv","low":"low/raw.csv","behavior":"behavior/raw.csv"}
    if (out/destinations[a.stage]).exists():
        raise SystemExit(f"Refuse to overwrite existing {a.stage} evidence; use a fresh output namespace")
    if a.stage in {"train","train-supported"}:
        train(out,extended=a.stage=="train-supported",reuse=a.reuse_behavior)
        return
    if a.stage=="calibrate":
        rows=[run(out,"hotrule_threshold",24,6200024,threshold=t,measure=a.measure) | {"threshold":t}
              for t in (.05,.1,.2,.3)]
        csv_write(out/"calibration/raw.csv",rows)
        best=max(rows,key=lambda r:(r["elapsed_task_tps"],-r["threshold"]))
        dump(out/"calibration/frozen.json",dict(threshold=best["threshold"],seed=6200024,
            selection="maximum drain-inclusive TPS; frozen before all evaluation",candidates=[.05,.1,.2,.3]))
        return
    threshold=json.loads((out/"calibration/frozen.json").read_text())["threshold"]
    variants=("silo","static_threshold","hotrule_always","hotrule_threshold","nophase","full","frozen_full","no_priority")
    rows=[]
    if a.stage=="validate":
        if a.reference_output is None:
            raise SystemExit("validation requires reference-output")
        for c in (16,32):
            for rep in range(2):
                order=[("reference",a.reference_output),("candidate",out)]
                random.Random(6830000+c+rep).shuffle(order)
                for name,target in order:
                    row=run(target,"full",c,6830000+c*100+rep,rep,threshold=threshold,measure=a.measure)
                    rows.append(row|{"variant":name})
                    csv_write(out/"validation/raw.csv",rows)
        csv_write(out/"validation/summary.csv",summary(rows))
        scores={name:statistics.fmean(r["elapsed_task_tps"] for r in rows if r["variant"]==name)
                for name in ("reference","candidate")}
        dump(out/"validation/selection.json",dict(scores=scores,candidate_improves=scores["candidate"]>scores["reference"],
            evaluation_seeds_seen=False,selection="paired separate validation only; no threshold changes"))
    elif a.stage in {"smoke","evaluate"}:
        clients=(8,24,40) if a.stage=="smoke" else tuple(map(int,a.clients.split(",")))
        repeats=1 if a.stage=="smoke" else a.repeats
        dest=out/a.stage
        for c in clients:
            for rep in range(repeats):
                order=list(variants)
                base=6300000 if a.stage=="smoke" else a.eval_seed_base
                random.Random(base+c*100+rep).shuffle(order)
                for v in order:
                    rows.append(run(out,v,c,base+c*100+rep,rep,threshold=threshold,measure=a.measure))
                    csv_write(dest/"raw.csv",rows)
        csv_write(dest/"summary.csv",summary(rows))
        csv_write(dest/"plot_data.csv",summary(rows))
    elif a.stage=="low":
        # Shared compliance shards must also scale: enlarging companies alone
        # does not remove this global write hotspot.
        configs={"very-low":CreditReviewConfig(2048,128,64,0.,1.,24,512,128),
                 "low":CreditReviewConfig(1024,64,32,.3,1.,24,128,32),
                 "medium":CreditReviewConfig(512,16,8,.7,1.,24,16,4),
                 "high":CreditReviewConfig()}
        for level,conf in configs.items():
            for rep in range(a.repeats):
                order=["silo","hotrule_threshold","full"]
                random.Random(6400000+rep).shuffle(order)
                for v in order:
                    rows.append(run(out,v,24,6400000+rep,rep,threshold=threshold,config=conf,measure=a.measure)|{"level":level})
                    csv_write(out/"low/raw.csv",rows)
        csv_write(out/"low/summary.csv",summary(rows))
    else:
        for v in ("full","nophase"):
            path=out/f"behavior/{v}.json"
            rows.append(run(out,v,24,6500024,threshold=threshold,trajectory=path,measure=a.measure,warmup=0.))
        csv_write(out/"behavior/raw.csv",rows)


if __name__=="__main__":
    main()
