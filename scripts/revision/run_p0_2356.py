#!/usr/bin/env python3
"""Bounded P0-2/3/5/6 follow-up, fixed before new holdout performance is observed."""
import argparse
import contextlib
import dataclasses
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import (DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig,
    state_features, state_key, policy_group_key, policy_transition_exclusion_reason)
from agent.runtime import PaperRewardConfig, apply_paper_rewards
from agent.runtime.paper_policy import CompiledPhasePolicy, HotRulePolicy
from agent.runtime.state_collector import policy_observation
from agent.runtime.version_manager import VersionManager
from agent.runtime.paper_hooks import PaperATCCHooks
from agent.workloads.credit_review import CreditReviewConfig
from scripts.revision.run_p0_actor_sampling import read_transitions
from scripts.revision.run_p0_revision import MeteredPolicy, csv_write, digest, dump, summary
from scripts.unified_trace.run_credit_review_experiment import run_system
from scripts.analysis.audit_p0_claims import compare, validate_rows

# Fixed, declared feature family. Generic footprint, hotness, conflict and
# observed blocking remain; this is not a claim of zero statistical phase proxy.
REMOVED_PROGRESS = ('inter_round_interval_ms', 'read_set_growth', 'write_set_growth',
    'access_overlap_ratio', 'completed_rounds', 'completed_operations', 'recent_write_ratio')


def without_phase_progress(state):
    return dataclasses.replace(state, phase='all', **{k: 0 for k in REMOVED_PROGRESS})


class PhaseProgressPolicy:
    """Explicit projection used for both independent training and inference."""
    def __init__(self, base):
        self.base = base

    def select(self, state):
        return self.base.select(without_phase_progress(state))


def train_models(behavior, output):
    paths = sorted(behavior.glob('behavior_*.json'))
    assert len(paths) == 6, 'fixed original common corpus only'
    old = json.loads((behavior/'full_report.json').read_text())
    raw = read_transitions(paths)
    assert len(raw) == 2424
    rows, reward = apply_paper_rewards(raw, PaperRewardConfig(**old['reward']['config']))
    cfg = PPOConfig(**old['config'])
    assert cfg.epochs == 4 and old['seed'] == 6120000
    records, models, reports = [], {}, {}
    for name in ('full', 'nophase_label', 'nophase_progress'):
        source_rows = (rows if name != 'nophase_progress' else [dataclasses.replace(t,
            state=without_phase_progress(t.state), next_state=without_phase_progress(t.next_state))
            for t in rows])
        # Never change logged q/actions/outcomes/reward or action-choice eligibility.
        for before, after in zip(rows, source_rows):
            # Removing operation progress can relabel an already deterministic
            # excluded row as fixed-initial OCC; optimizer eligibility must not
            # change, but that diagnostic reason name need not stay identical.
            assert bool(policy_transition_exclusion_reason(before)) == bool(policy_transition_exclusion_reason(after))
            for field in dataclasses.fields(before):
                if field.name not in ('state', 'next_state'):
                    assert getattr(before, field.name) == getattr(after, field.name)
        started = time.perf_counter()
        policy = DiscretePPOPolicy(seed=old['seed'], phase_awareness=(
            'enabled' if name == 'full' else 'disabled'))
        trained = DiscretePPOTrainer(cfg).train(policy, source_rows)
        fit_seconds = time.perf_counter()-started
        started = time.perf_counter()
        compiled = policy.compile(generation=1, medoids_per_group=old['medoids_per_group'],
                                  occ_cold_start_guard=old['occ_cold_start_guard'])
        compile_seconds = time.perf_counter()-started
        path = output/'training'/f'{name}.json'
        dump(path, compiled.to_dict())
        wrapper = PhaseProgressPolicy(compiled) if name == 'nophase_progress' else compiled
        models[name] = wrapper
        reports[name] = dict(config=dataclasses.asdict(cfg), reward=reward,
            seed=old['seed'], training=trained, removed_progress=(
                list(REMOVED_PROGRESS) if name == 'nophase_progress' else []),
            explicit_phase_removed=name != 'full', fit_seconds=fit_seconds,
            compile_seconds=compile_seconds, artifact_sha256=digest(path),
            runtime_loader=('PhaseProgressPolicy(CompiledPhasePolicy.load(path))'
                if name == 'nophase_progress' else 'CompiledPhasePolicy.load(path)'),
            trajectory_count=len({(t.source_id, t.txn_id) for t in rows}), transitions=len(rows),
            compiled_entries=len(compiled.entries), policy_bytes=path.stat().st_size,
            original_artifact_identical=(digest(path) == digest(behavior/(
                'full.json' if name == 'full' else 'nophase.json'))) if name != 'nophase_progress' else None)
        dump(output/'training'/f'{name}_report.json', reports[name])
        records.append(dict(variant=name, fit_seconds=fit_seconds, compile_seconds=compile_seconds,
            compiled_entries=len(compiled.entries), policy_bytes=path.stat().st_size,
            transitions=len(rows), trajectories=reports[name]['trajectory_count'], seed=old['seed'], epochs=4))
    checks = 0
    for t in rows:
        variants = [dataclasses.replace(t.state, phase=p, completed_rounds=i+1,
            completed_operations=(i+1)*17, read_set_growth=i*4, write_set_growth=i,
            access_overlap_ratio=i/2, recent_write_ratio=i/2,
            inter_round_interval_ms=i*100.) for i, p in enumerate(('explore', 'refine', 'commit'))]
        encoded = [(state_key(without_phase_progress(s)),
                    policy_group_key(without_phase_progress(s)),
                    state_features(without_phase_progress(s)), models['nophase_progress'].select(s))
                   for s in variants]
        assert encoded[0] == encoded[1] == encoded[2]
        checks += 1
    dump(output/'training/audit.json', dict(passed=True, progress_invariance_checks=checks,
        rewards_q_actions_and_eligibility_preserved=True,
        original_input_sha256={p.name: digest(p) for p in paths},
        architecture='original factorized four-bit linear actor and linear critic',
        common_phase_checkpoints_retained=True, all_statistical_phase_proxies_removed=False))
    csv_write(output/'training/costs.csv', records)
    return rows, reports


def load_model(output, name):
    base = CompiledPhasePolicy.load(output/'training'/f'{name}.json')
    base.prebucketed_lookup = True
    # All arms retain exactly the same guard configuration as training.
    return PhaseProgressPolicy(base) if name == 'nophase_progress' else base


def measure(output, name, case, config, seed, rep, clients, tracked, seconds=1.5, warmup=.1):
    path = output/'training/full.json'
    policy = (None if name == 'silo' else HotRulePolicy(.05) if name == 'hotrule_threshold'
              else load_model(output, name if name.startswith('nophase') else 'full'))
    meter = MeteredPolicy(policy) if policy is not None else None
    row = run_system(config=config, clients=clients, seed=seed, repeat=rep,
        system='silo' if name == 'silo' else 'paper-atcc', paper_policy=path,
        policy_override=meter, measure_seconds=seconds, warmup_seconds=warmup,
        max_attempts=6, performance_guards=False, policy_commit_batch=True,
        tracked_history_import=tracked)
    row.update(variant=name, case=case, config_json=json.dumps(dataclasses.asdict(config)),
        history_optimization=tracked, phase_checkpoints_common=True)
    if meter:
        row.update(policy_calls=meter.calls, decisions_per_attempt=meter.calls/max(1,row['attempts']),
            confirmed_mask_installation_not_claimed=True,
            selected_changes=meter.changes, policy_lookup_mean_us=meter.ns/max(1,meter.calls)/1000,
            cumulative_policy_lookup_ms=meter.ns/1e6,
            phase_actions_json=json.dumps({f'{p}:{a}':n for (p,a),n in meter.phase_actions.items()}))
    assert not validate_rows([row])
    print(json.dumps({k:row[k] for k in ('case','variant','repeat','elapsed_task_tps',
        'agent_tps','p99_latency_ms','failed_tasks')}), flush=True)
    return row


def lookup_costs(rows, output):
    states = [t.state for t in rows[:512]]
    samples = []
    for name in ('full','nophase_label','nophase_progress'):
        policy = load_model(output, name)
        base = policy.base if isinstance(policy, PhaseProgressPolicy) else policy
        for temperature in ('cold','warm'):
            values = []
            for s in states:
                if temperature == 'cold': base._fallback_cache.clear()
                else: policy.select(s)
                started = time.perf_counter_ns()
                policy.select(s)
                values.append((time.perf_counter_ns()-started)/1000.)
            samples.append(dict(variant=name, temperature=temperature, samples=len(values),
                mean_us=statistics.fmean(values), median_us=statistics.median(values)))
    csv_write(output/'lookup.csv', samples)


@contextlib.contextmanager
def profile_paths():
    totals, originals, latch = {}, [], threading.Lock()
    targets = [(VersionManager,'pin_lazy'),(VersionManager,'snapshot_and_pin'),
        (VersionManager,'read_at'),(VersionManager,'atomic_publish'),
        (PaperATCCHooks,'_state')]
    def wrap(name, original):
        def called(*a, **kw):
            started = time.perf_counter_ns()
            try: return original(*a, **kw)
            finally:
                with latch:
                    record = totals.setdefault(name, {'calls':0,'wall_ms':0.})
                    record['calls'] += 1
                    record['wall_ms'] += (time.perf_counter_ns()-started)/1e6
        return called
    try:
        for cls, method in targets:
            original = getattr(cls, method)
            originals.append((cls,method,original))
            setattr(cls,method,wrap(f'{cls.__name__}.{method}',original))
        yield totals
    finally:
        for cls,method,original in originals: setattr(cls,method,original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--behavior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','phase','low','profile'),required=True)
    args = parser.parse_args()
    assert platform.node().split('.')[0] == 'node1'
    if args.stage == 'train':
        assert not args.output.exists()
        dump(args.output/'protocol.json',dict(scope=__doc__,
            behavior_sha256={p.name:digest(p) for p in sorted(args.behavior.glob('behavior_*.json'))},
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'agent').rglob('*.py')},
            native_sha256=digest(next(ROOT.glob('cast_core*.so'))), script_sha256=digest(Path(__file__)),
            removed_progress=list(REMOVED_PROGRESS), phases=['all'], training_seed=6120000, epochs=4,
            phase_clients=16, phase_variants=['full','nophase_label','nophase_progress'],
            phase_seed_base=20521600, low_clients=24, low_cases=['low','high'],
            low_seed_base=20562400, repeats=2, seconds=1.5, warmup=.1, max_attempts=6,
            low_variants=['reference_full','full','hotrule_threshold','silo'],
            low_history_contract='all physical writes via this manager or explicitly synchronized',
            independent_threshold=.05, threshold_not_retuned=True, performance_guards=False,
            current_cpu_affinity=sorted(os.sched_getaffinity(0)), real_llm=False,
            default_model_replaced=False, input_ablation_not_phase_scheduling_ablation=True,
            prospective_screen='mean main/window>=1.03,p99<=1.05,2/2 main wins,fail0; report all costs',
            source_or_models_change_after_train_not_allowed=True))
        rows,reports = train_models(args.behavior,args.output)
        lookup_costs(rows,args.output)
        print(json.dumps(reports),flush=True)
        return
    protocol = json.loads((args.output/'protocol.json').read_text())
    assert digest(Path(__file__)) == protocol['script_sha256']
    assert protocol['source_sha256'] == {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'agent').rglob('*.py')}
    target = args.output/args.stage
    assert not target.exists(), 'one fixed run only; no overwrites'
    raw = []
    if args.stage == 'phase':
        for rep in range(2):
            order = ['full','nophase_label','nophase_progress']
            random.Random(20531600+rep).shuffle(order)
            for name in order:
                raw.append(measure(args.output,name,'phase',CreditReviewConfig(),
                    20521600+rep,rep,16,True))
                csv_write(target/'raw.csv',raw)
    elif args.stage == 'low':
        configs = {'low':CreditReviewConfig(1024,64,32,.3,1.,24,128,32),
                   'high':CreditReviewConfig()}
        for index,(case,config) in enumerate(configs.items()):
            for rep in range(2):
                order = ['reference_full','full','hotrule_threshold','silo']
                random.Random(20572400+index*100+rep).shuffle(order)
                for name in order:
                    raw.append(measure(args.output,name,case,config,20562400+index*100+rep,
                        rep,24,name!='reference_full'))
                    csv_write(target/'raw.csv',raw)
    else:
        config = CreditReviewConfig(1024,64,32,.3,1.,24,128,32)
        profiles = []
        for tracked in (False,True):
            with profile_paths() as costs:
                row = measure(args.output,'full','profile',config,20582400,0,24,tracked,
                              seconds=.5,warmup=0.)
            profiles.append(dict(tracked_history_import=tracked,costs=costs,row=row,
                scope='instrumented wall time, includes waiting and overlaps; NOT formal TPS or CPU breakdown'))
        dump(target/'profiles.json',profiles)
        return
    csv_write(target/'summary.csv',summary(raw))
    comparisons = [c for v in sorted({r['variant'] for r in raw}) if v!='full'
                   for c in compare(raw,'case','full',v)]
    for c in comparisons:
        c['direction_screen'] = (c['tps_ratio']>=1.03 and c['window_tps_ratio']>=1.03
            and c['p99_ratio']<=1.05 and c['paired_wins']==2
            and not c['failed_tasks_full'] and not c['failed_tasks_baseline'])
    dump(target/'completed.json',dict(rows=len(raw),raw_sha256=digest(target/'raw.csv'),
        comparisons=comparisons,significance_claimed=False,
        original_paper_preservation_proved=False,goal_completed=False))
    print(json.dumps(comparisons),flush=True)


if __name__ == '__main__': main()
