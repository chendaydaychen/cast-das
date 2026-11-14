#!/usr/bin/env python3
"""Audit raw pilot evidence, not just green acceptance markers.

Report complete curves and every paired direction. A positive point is NOT a
significance test, a universal claim, or numerical reproduction of the paper.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if not path.exists():return []
    with path.open(newline="") as stream:return list(csv.DictReader(stream))


def groups(rows,keys):
    result=defaultdict(list)
    for row in rows:result[tuple(row[k] for k in keys)].append(row)
    return result


def mean(rows,key):return statistics.fmean(float(row[key]) for row in rows)


def validate_rows(rows):
    errors=[]
    for row in rows:
        counts={key:int(row[key]) for key in ("attempts","commits","aborts","logical_tasks","failed_tasks")}
        if any(value<0 for value in counts.values()):errors.append("negative counter")
        if counts["attempts"]!=counts["commits"]+counts["aborts"]:errors.append("attempt invariant")
        if counts["logical_tasks"]!=counts["commits"]+counts["failed_tasks"]:errors.append("task invariant")
        denominator=float(row["measurement_seconds"])+float(row["drain_seconds"])
        if not math.isclose(float(row["elapsed_task_tps"]),counts["commits"]/denominator,rel_tol=1e-9):
            errors.append("elapsed TPS denominator")
        if not math.isclose(float(row["agent_tps"]),counts["commits"]/float(row["measurement_seconds"]),rel_tol=1e-9):
            errors.append("window TPS denominator")
        if int(row["max_attempts"])!=6:errors.append("attempt budget")
    return errors


def compare(rows,group_key,full,baseline):
    grouped=groups(rows,(group_key,"variant"))
    output=[]
    for key in sorted({r[group_key] for r in rows},
                      key=lambda k:((0,int(k)) if str(k).isdigit() else (1,str(k)))):
        first=grouped.get((key,full),[]);second=grouped.get((key,baseline),[])
        if not first or not second:continue
        left={(r["seed"],r["repeat"]):r for r in first}
        right={(r["seed"],r["repeat"]):r for r in second}
        pairs=sorted(set(left)&set(right))
        ratio=mean(first,"elapsed_task_tps")/mean(second,"elapsed_task_tps")
        pair_ratios=[float(left[p]["elapsed_task_tps"])/float(right[p]["elapsed_task_tps"]) for p in pairs]
        output.append(dict(group=str(key),full=full,baseline=baseline,repeats_full=len(first),
            repeats_baseline=len(second),matched_pairs=len(pairs),tps_ratio=ratio,
            window_tps_ratio=mean(first,"agent_tps")/mean(second,"agent_tps"),
            p99_ratio=mean(first,"p99_latency_ms")/max(1e-9,mean(second,"p99_latency_ms")),
            abort_rate_full=mean(first,"abort_rate"),abort_rate_baseline=mean(second,"abort_rate"),
            failed_tasks_full=sum(int(r["failed_tasks"]) for r in first),
            failed_tasks_baseline=sum(int(r["failed_tasks"]) for r in second),
            paired_wins=sum(r>1 for r in pair_ratios),paired_losses=sum(r<1 for r in pair_ratios),
            paired_ratios=pair_ratios,
            all_paired_directions_positive=bool(pair_ratios) and all(r>1 for r in pair_ratios)))
    return output


def matrix(rows,points,variants,repeats,group_key,seed_base):
    expected={(str(point),variant,str(rep)) for point in points for variant in variants for rep in range(repeats)}
    actual=[(r[group_key],r["variant"],r["repeat"]) for r in rows]
    seed_errors=[]
    for row in rows:
        index=int(row[group_key])*100 if group_key=="clients" else points.index(row[group_key])*100
        if int(row["seed"])!=seed_base+index+int(row["repeat"]):seed_errors.append(row["seed"])
    return dict(rows=len(rows),expected_rows=len(expected),missing=sorted(expected-set(actual)),
        unexpected=sorted(set(actual)-expected),duplicates=len(actual)-len(set(actual)),
        seed_errors=seed_errors,counter_or_metric_errors=validate_rows(rows),
        complete=len(actual)==len(expected) and set(actual)==expected and not seed_errors and not validate_rows(rows))


def initial_evidence(root):
    """Verify archived raw matrices rather than trusting their acceptance counts."""
    internal=read(root/"ablation/raw.csv")
    variants=["silo","static_threshold","hotrule_always","hotrule_threshold",
              "full","nophase","frozen_full","no_priority"]
    internal_check=matrix(internal,[8,16,24,32,40],variants,3,"clients",6900000)
    low=read(root/"low_contention/raw.csv")
    low_expected={(level,name,str(rep)) for level in ("very-low","low","medium","high")
                  for name in ("silo","hotrule_threshold","full") for rep in range(3)}
    low_actual=[(row["level"],row["variant"],row["repeat"]) for row in low]
    low_seed_valid=all(int(row["seed"])==6400000+int(row["repeat"]) for row in low)
    low_errors=validate_rows(low)
    low_check=dict(rows=len(low),expected_rows=36,missing=sorted(low_expected-set(low_actual)),
        unexpected=sorted(set(low_actual)-low_expected),duplicates=len(low_actual)-len(set(low_actual)),
        seed_valid=low_seed_valid,counter_or_metric_errors=low_errors,
        complete=len(low)==36 and set(low_actual)==low_expected and low_seed_valid and not low_errors)
    native=read(root/"dbx1000_baselines/raw.csv")
    native_expected={(workload,level,algorithm,str(rep))
        for workload in ("ycsb","tpcc") for level in ("low","medium","high")
        for algorithm in ("SILO","HEKATON","TICTOC","NO_WAIT")
        for rep in range(1 if workload=="tpcc" and level=="high" and algorithm!="SILO" else 3)}
    native_actual=[(row["workload"],row["level"],row["algorithm"],row["repeat"]) for row in native]
    native_errors=[]
    native_cases=groups(native,("workload","level"))
    for (workload,level),case_rows in native_cases.items():
        expected_seed=6600000+(0 if workload=="ycsb" else 3)+("low","medium","high").index(level)
        if any(int(row["seed"])!=expected_seed for row in case_rows):native_errors.append("native case seed")
        if len({row["trace_sha256"] for row in case_rows})!=1:native_errors.append("native case trace mismatch")
    for row in native:
        commits=int(row["commits"]);aborts=int(row["aborts"])
        if commits!=128 or aborts<0 or int(row["exit_code"])!=0:native_errors.append("commit/exit invariant")
        # all_debug1 is indexed by committed txn count. Aborted attempts write
        # the same slot; the eventual successful attempt overwrites them.
        if int(row["latency_samples"])!=commits:native_errors.append("successful attempt sample count")
        if not math.isclose(float(row["elapsed_tps"]),commits/float(row["run_seconds"]),rel_tol=1e-9):
            native_errors.append("subprocess TPS denominator")
    native_check=dict(rows=len(native),expected_rows=66,
        missing=sorted(native_expected-set(native_actual)),unexpected=sorted(set(native_actual)-native_expected),
        duplicates=len(native_actual)-len(set(native_actual)),counter_or_metric_errors=native_errors,
        complete=len(native)==66 and set(native_actual)==native_expected and not native_errors,
        scope="same native engine; unequal repetitions; final successful attempt latency only",
        aborted_attempt_latencies_retained=False,logical_task_latency_available=False)
    return dict(internal=internal_check,low_contention=low_check,native=native_check,
        initial_and_new_models_not_pooled=True,
        all_raw_matrices_verified=all(check["complete"] for check in (internal_check,low_check,native_check)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial",type=Path,required=True)
    parser.add_argument("--improvement",type=Path,required=True)
    parser.add_argument("--deployment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args()
    assert not args.output.exists(),"fresh audit output required"
    holdout=read(args.improvement/"holdout/raw.csv")
    deployment=read(args.deployment/"raw.csv")
    holdout_variants=["silo","hotrule_always","hotrule_threshold","full","nophase",
        "frozen_full","no_priority","reference_full"]
    levels=["disjoint-zero","very-low","low","medium","high"]
    deployment_variants=["silo","full_legacy_guard","full_conflict_fast","hotrule_conflict_fast"]
    audit=dict(holdout=matrix(holdout,[8,16,24,32,40],holdout_variants,2,"clients",7300000),
        deployment=matrix(deployment,levels,deployment_variants,2,"level",7400000),
        scope="two-repeat synthetic pilot; no significance or universal superiority claim",
        original_paper_numerical_reproduction_proven=False,goal_completion_proven=False)
    comparisons=[]
    for baseline in holdout_variants:
        if baseline!="full":comparisons+=compare(holdout,"clients","full",baseline)
    for baseline in ("silo","full_legacy_guard","hotrule_conflict_fast"):
        comparisons+=compare(deployment,"level","full_conflict_fast",baseline)
    native=read(args.improvement/"native_wall/raw.csv")
    native_expected={(alg,str(rep)) for alg in ("SILO","HEKATON","TICTOC","NO_WAIT") for rep in range(2)}
    audit["native_wall"]=dict(rows=len(native),expected_rows=8,
        matrix_complete={(r["algorithm"],r["repeat"]) for r in native}==native_expected and len(native)==8,
        commits_128=bool(native) and all(int(r["commits"])==128 for r in native),
        exits_zero=bool(native) and all(int(r["exit_code"])==0 for r in native),
        identical_fixed_trace=len({r["trace_sha256"] for r in native})==1,
        worker_denominators_valid=bool(native) and all(math.isclose(float(r["worker_tps"]),
            int(r["commits"])/float(r["worker_seconds"]),rel_tol=1e-9) for r in native),
        clock_scales=[float(r["configured_clock_scale"]) for r in native],
        initialization_not_in_primary=True,
        retry_latency_scope="native unbounded / final successful attempt; preceding abort latencies omitted")
    audit["initial_package"]=initial_evidence(args.initial)
    selection=json.loads((args.improvement/"validation/selection.json").read_text())
    mode=selection["selected_mode"]
    reports={name:json.loads((args.improvement/"training"/mode/f"{name}_report.json").read_text())
        for name in ("full","nophase")}
    policies={name:json.loads((args.improvement/"training"/mode/f"{name}.json").read_text())
        for name in ("full","nophase")}
    audit["nophase"]=dict(common_inputs=reports["full"]["inputs"]==reports["nophase"]["inputs"],
        common_seed=reports["full"]["seed"]==reports["nophase"]["seed"],
        common_config=reports["full"]["config"]==reports["nophase"]["config"],
        explicit_phase_disabled=policies["nophase"]["phase_awareness"]=="disabled",
        all_compiled_phases_all=bool(policies["nophase"]["entries"]) and
            all(row["phase"]=="all" for row in policies["nophase"]["entries"]),
        artifact_distinct=digest(args.improvement/"training"/mode/"full.json")!=
            digest(args.improvement/"training"/mode/"nophase.json"),
        implicit_phase_cues_remain=True)
    missing=not audit["holdout"]["complete"] or not audit["deployment"]["complete"]
    if missing and not args.partial:raise AssertionError("Matrix incomplete; use --partial only for diagnostics")
    if not audit["initial_package"]["all_raw_matrices_verified"] and not args.partial:
        raise AssertionError("Initial raw evidence incomplete or inconsistent")
    args.output.mkdir(parents=True)
    (args.output/"audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    (args.output/"comparisons.json").write_text(json.dumps(comparisons,indent=2,sort_keys=True)+"\n")
    lines=["# P0 raw-evidence audit","",audit["scope"],"",
        "All positive/negative points are listed; models and throughput denominators are not pooled.",
        "p99 covers committed logical tasks only. Failures are reported separately.",
        "Directional consistency in two pairs is not statistical significance.","",
        f"Holdout rows: {len(holdout)}/80; deployment: {len(deployment)}/40; native wall: {len(native)}/8.","",
        "|Group|Full variant|Baseline|TPS ratio|Window TPS ratio|p99 ratio|Paired wins|Failures Full/Base|",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"|{row['group']}|{row['full']}|{row['baseline']}|{row['tps_ratio']:.3f}|"
            f"{row['window_tps_ratio']:.3f}|{row['p99_ratio']:.3f}|{row['paired_wins']}/{row['matched_pairs']}|"
            f"{row['failed_tasks_full']}/{row['failed_tasks_baseline']}|")
    lines += ["","Artifact completion and favorable hypotheses are different gates.",
        "This audit does not prove original paper-scale numerical performance or overall goal completion."]
    (args.output/"AUDIT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(audit,indent=2,sort_keys=True))


if __name__=="__main__":main()
