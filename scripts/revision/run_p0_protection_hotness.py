#!/usr/bin/env python3
"""Fresh actual protection-hotness encoder data; no relabeling of historical probabilities."""
import argparse
import dataclasses
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig
from agent.runtime import PaperRewardConfig, apply_paper_rewards
from agent.runtime.paper_policy import HotRulePolicy
from scripts.revision.run_p0_actor_sampling import ActorMixtureBehaviorPolicy, freeze_actor, read_transitions
from scripts.revision import run_p0_read_exposure as study
from scripts.unified_trace import run_credit_review_experiment as engine
from scripts.revision.run_p0_revision import dump, digest, csv_write
from scripts.analysis.audit_p0_claims import compare


class CanonicalMaskProposal:
    """Proposal only: equally cover OCC/hot/cold/all, retain monotonic masks.

    ActorMixtureBehaviorPolicy additionally explores ALL valid masks. This
    class never uses outcomes, labels, future targets, or test performance.
    """
    protection_relevant_hotness = True

    def probabilities(self, state):
        masks = {int(state.current_action) | mask for mask in (0, 5, 10, 15)}
        return tuple(1 / len(masks) if action in masks else 0. for action in range(16))


class PhaseContinuationProposal:
    """Collector only: explore Refine masks, follow frozen greedy continuation.

    The outer .2 uniform mixture still explores every legal action at every
    choice. No rollout result or future access enters this proposal.
    """
    protection_relevant_hotness = True

    def __init__(self, continuation):
        if not continuation.protection_relevant_hotness:
            raise ValueError('continuation must use the actual protection-hotness encoder')
        self.continuation = continuation

    def probabilities(self, state):
        if state.phase == 'refine':
            masks = {int(state.current_action) | mask for mask in (0, 5, 10, 15)}
            return tuple(1 / len(masks) if action in masks else 0. for action in range(16))
        selected = int(self.continuation.select(state).protected)
        if selected < 0 or selected >= 16 or selected | int(state.current_action) != selected:
            raise ValueError('continuation must select a legal monotonic full mask')
        return tuple(float(action == selected) for action in range(16))


class DeploymentMixtureProposal(PhaseContinuationProposal):
    """Collector only: half frozen deployment, half canonical exploration in Refine.

    Fixed .5 balance, not a tunable performance search. Outer .2 valid-action
    exploration is unchanged and the actual marginal q is logged before draw.
    """
    def probabilities(self, state):
        selected = int(self.continuation.select(state).protected)
        if selected < 0 or selected >= 16 or selected | int(state.current_action) != selected:
            raise ValueError('continuation must select a legal monotonic full mask')
        if state.phase != 'refine':
            return tuple(float(action == selected) for action in range(16))
        masks = {int(state.current_action) | mask for mask in (0, 5, 10, 15)}
        return tuple((.5 / len(masks) if action in masks else 0.)
                     + (.5 if action == selected else 0.) for action in range(16))


class PhaseExpansionMixtureProposal(DeploymentMixtureProposal):
    """Collector only: retain Refine mixture, cover every valid Commit expansion.

    Uses fixed half deployment/half uniform legal-action mixture at Commit.
    The outer .2 exploration still logs the complete actual marginal q. This
    changes shared collection coverage, never a deployed Full/NoPriority arm.
    """
    def probabilities(self, state):
        if state.phase != 'commit':
            return super().probabilities(state)
        current = int(state.current_action)
        selected = int(self.continuation.select(state).protected)
        if selected < 0 or selected >= 16 or selected | current != selected:
            raise ValueError('continuation must select a legal monotonic full mask')
        valid = tuple(action for action in range(16) if action | current == action)
        return tuple((.5 / len(valid) if action in valid else 0.)
                     + (.5 if action == selected else 0.) for action in range(16))


class StageBalancedProposal:
    """Collector only: cover initial hot/cold paths and their later expansions.

    Fixed observed-state schedule, no outcome-dependent adaptation. Retain
    the current mask before writes so exploring an early expansion does not
    consume all later-stage support. The outer .2 legal exploration remains.
    """
    protection_relevant_hotness = True

    def probabilities(self, state):
        current = int(state.current_action)
        if current < 0 or current >= 16:
            raise ValueError('current mask must be a legal full mask')
        valid = tuple(action for action in range(16) if action | current == action)
        if state.phase == 'refine':
            canonical = {current | mask for mask in (0, 5, 10, 15)}
            hot_cold = {current | mask for mask in (5, 10)}
            return tuple((.5 / len(canonical) if action in canonical else 0.)
                + (.5 / len(hot_cold) if action in hot_cold else 0.) for action in range(16))
        if state.phase == 'commit' and state.write_set_size > 0:
            return tuple(1. / len(valid) if action in valid else 0. for action in range(16))
        return tuple(float(action == current) for action in range(16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--baseline-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-family', type=int, default=106)
    parser.add_argument('--proposal-strategy', choices=('frozen_actor', 'canonical_masks', 'phase_continuation', 'deployment_mixture', 'phase_expansion_mixture', 'stage_balanced'), default='frozen_actor')
    parser.add_argument('--continuation-policy', type=Path)
    parser.add_argument('--advantage-baseline', choices=('linear-critic', 'deployment-group'), default='linear-critic')
    parser.add_argument('--regression-protocol', type=Path)
    parser.add_argument('--unique-attempt-outcomes', action='store_true')
    parser.add_argument('--carry-prior-restart-cost', action='store_true')
    parser.add_argument('--statewise-supported-actor', action='store_true')
    parser.add_argument('--exact-categorical-lookup', action='store_true')
    parser.add_argument('--round-interval-features', action='store_true')
    parser.add_argument('--categorical-initialization', choices=('behavior', 'qualified-return'), default='behavior')
    parser.add_argument('--ppo-reference-mode', choices=('behavior', 'frozen-actor'), default='behavior')
    parser.add_argument('--collect-and-fit-only', action='store_true', help='Stop before actual probe or performance; independently audit before deployment')
    parser.add_argument('--collect-only', action='store_true', help='Stop after raw collection and helper ledger; no training before separate input/cost audits')
    parser.add_argument('--phase-expansion-plan', type=Path)
    parser.add_argument('--stage-balanced-plan', type=Path)
    parser.add_argument('--coherent-footprint-witness', action='store_true')
    args = parser.parse_args()
    if args.exact_categorical_lookup and not args.statewise_supported_actor:
        parser.error('exact categorical lookup requires statewise supported actor')
    continuation_strategy = args.proposal_strategy in ('phase_continuation', 'deployment_mixture', 'phase_expansion_mixture')
    if (args.proposal_strategy == 'phase_expansion_mixture') != (args.phase_expansion_plan is not None):
        parser.error('phase expansion mixture requires its own frozen plan; other strategies must not use it')
    if (args.proposal_strategy == 'stage_balanced') != (args.stage_balanced_plan is not None):
        parser.error('stage balanced collection requires its own frozen plan; other strategies must not use it')
    if args.coherent_footprint_witness and args.proposal_strategy != 'stage_balanced':
        parser.error('coherent footprint witness requires its separately frozen stage-balanced collection plan')
    if args.proposal_strategy == 'stage_balanced' and not (args.collect_only and args.collect_and_fit_only
            and args.round_interval_features and args.statewise_supported_actor
            and args.advantage_baseline == 'deployment-group'
            and args.categorical_initialization == 'qualified-return' and args.ppo_reference_mode == 'frozen-actor'):
        parser.error('stage balanced protocol requires audited collection-only and the unchanged supported replay configuration')
    if continuation_strategy != (args.continuation_policy is not None):
        parser.error('continuation strategies require --continuation-policy, other strategies must not use it')
    if continuation_strategy and args.advantage_baseline != 'deployment-group':
        parser.error('continuation protocol requires --advantage-baseline deployment-group')
    assert platform.node().split('.')[0] == 'node1' and not args.output.exists()
    assert args.seed_family >= 106
    prefix = args.seed_family * 100000
    prior = json.loads((args.prior / 'protocol.json').read_text())
    assert json.loads((args.prior.parent / 'actor_sampling_audit_1315.json').read_text())['structural_checks_passed']
    baseline = json.loads(args.baseline_audit.read_text())
    assert baseline['passed']
    source = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'agent').rglob('*.py')}
    assert set(source) == set(baseline['source_sha256'])
    assert sorted(k for k in source if source[k] != baseline['source_sha256'][k]) == [
        'agent/cc/atcc/ppo.py', 'agent/runtime/paper_hooks.py', 'agent/runtime/paper_policy.py',
        'agent/runtime/state_collector.py', 'agent/runtime/trajectory.py', 'agent/runtime/transaction.py']
    native = digest(next(ROOT.glob('cast_core*.so')))
    assert native == baseline['native_sha256']
    if args.regression_protocol:
        regression = json.loads(args.regression_protocol.read_text())
        assert regression['passed'] and not regression['errors'] and not regression['failures']
        assert source == regression['source_sha256'] and native == regression['native_sha256']
    flags = dict(prior['runtime_flags'], protection_relevant_hotness=True)
    if args.unique_attempt_outcomes:
        flags['unique_attempt_outcomes'] = True
    if args.carry_prior_restart_cost:
        flags['carry_prior_restart_cost'] = True
    if args.round_interval_features:
        flags['round_interval_features'] = True
    ids = [f'behavior_{i}' for i in range(6)]
    dump(args.output / 'protocol.json', dict(scope=__doc__, prior=str(args.prior),
        prior_protocol_sha256=digest(args.prior / 'protocol.json'), baseline_audit=str(args.baseline_audit),
        baseline_audit_sha256=digest(args.baseline_audit),
        source_sha256=source, native_sha256=native, runtime_flags=flags,
        regression_protocol=str(args.regression_protocol) if args.regression_protocol else None,
        regression_protocol_sha256=digest(args.regression_protocol) if args.regression_protocol else None,
        hotrule_protection_relevant_hotness=False,
        scripts_sha256={p: digest(ROOT / p) for p in ('scripts/revision/run_p0_protection_hotness.py',
            'scripts/analysis/audit_p0_protection_hotness_inputs.py', 'scripts/analysis/audit_p0_actor_deployment_influence.py',
            'scripts/revision/run_p0_actor_sampling.py', 'scripts/analysis/audit_p0_actor_sampling_inputs.py',
            'scripts/revision/run_p0_read_exposure.py', 'scripts/analysis/audit_p0_helper_ledger.py',
            'scripts/analysis/diagnose_p0_deployed_actor_reachability.py',
            'scripts/analysis/audit_p0_deployed_actor_reachability.py',
            'scripts/analysis/audit_p0_restart_cost_inputs.py',
            'scripts/unified_trace/run_credit_review_experiment.py')},
        hotness_encoding='eligible_observed_reads_plus_observed_writes',
        collection_runs=6, collection_clients=[16, 32, 16, 32, 16, 32], collection_seconds=.75, collection_warmup=0.,
        collection_seed_base=prefix + 10000, sampler_seed_base=prefix + 11000, sampler_rebuild_seed=10030000,
        training_seed=prefix + 30000, ppo_training_source_ids=ids, gradient_epochs=4, support_files=6,
        proposal_strategy=args.proposal_strategy,
        **(dict(phase_expansion_plan=str(args.phase_expansion_plan),
                phase_expansion_plan_sha256=digest(args.phase_expansion_plan),
                collect_only=args.collect_only) if args.phase_expansion_plan is not None else {}),
        **(dict(stage_balanced_plan=str(args.stage_balanced_plan),
                stage_balanced_plan_sha256=digest(args.stage_balanced_plan),
                collect_only=args.collect_only) if args.stage_balanced_plan is not None else {}),
        **(dict(coherent_collection_footprint_witness=True) if args.coherent_footprint_witness else {}),
        continuation_policy=str(args.continuation_policy) if args.continuation_policy else None,
        continuation_policy_sha256=digest(args.continuation_policy) if args.continuation_policy else None,
        actor_advantage_baseline=args.advantage_baseline,
        actual_path_preflight_required=continuation_strategy or args.statewise_supported_actor,
        statewise_supported_actor=args.statewise_supported_actor,
        exact_categorical_lookup=args.exact_categorical_lookup,
        attempt_restart_cost_ledger=args.carry_prior_restart_cost,
        terminal_trajectory_guard=True,
        **(dict(round_cost_witness=args.round_interval_features, round_interval_features=args.round_interval_features,
                categorical_initialization=args.categorical_initialization, ppo_reference_mode=args.ppo_reference_mode,
                collect_and_fit_only=args.collect_and_fit_only) if (args.round_interval_features or args.collect_and_fit_only
                    or args.categorical_initialization != 'behavior' or args.ppo_reference_mode != 'behavior') else {}),
        **(dict(round_cost_plan_sha256=digest(ROOT/'OBSERVED_ROUND_COST_1885_PLAN.md')) if args.round_interval_features else {}),
        epsilon=.2, actor_mixture_weights={'full': .5, 'nophase': .5}, outer_exploration_wrapping=False,
        historical_rows_used_for_new_training=False,
        sampler_members_are_frozen_old_proposal_not_new_encoder_models=args.proposal_strategy == 'frozen_actor',
        reward_formula_changed=False, priority_formula_changed=False, minimum_action_support=8,
        sql_reasoning_token_retry_budget_changed=False, packed_lookup_enabled=False,
        validation_seed_base=prefix + 50000, order_seed_base=prefix + 60000, clients=[16], repeats=3, seconds=4., warmup=.2,
        variants=(['full', 'no_actor', 'nophase', 'empirical', 'hotrule_always', 'no_priority', 'frozen_full']
            if args.round_interval_features and args.collect_and_fit_only
            else ['full', 'no_actor', 'nophase', 'hotrule_always']), max_attempts=6,
        rule='Full vs each: main/window TPS>=1.03, p99<=1.05, wasted reasoning/tokens<=1, >=2/3 main wins, zero failures',
        performance_preflight=('Full actual path actor influence>0 AND fresh actor influence>0; both models rebuilt'
            if continuation_strategy
            else 'Full fresh actor influence>0; both independent NoPhase and Full rebuilt'),
        default_adopted=False, real_llm=False))
    old_paths = sorted((args.prior / 'training').glob('behavior_*.json'))
    old_audit = json.loads((args.prior / 'training/audit.json').read_text())
    assert {p.name: digest(p) for p in old_paths} == old_audit['common_input_sha256']
    old_raw = read_transitions(old_paths)
    reference = Path(prior['reference_models'])
    samplers, started = {}, time.monotonic()
    for name, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        report = json.loads((args.prior / f'training/{name}_report.json').read_text())
        rows, reward = apply_paper_rewards(old_raw, PaperRewardConfig(**report['reward']['config']))
        assert reward == report['reward']
        actor = DiscretePPOPolicy(seed=report['seed'], phase_awareness=awareness, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True)
        fresh = [t for t in rows if t.source_id in report['ppo_training_source_ids']]
        trained = DiscretePPOTrainer(PPOConfig(**report['config'])).train(actor, fresh, calibration_transitions=rows)
        assert trained['epochs'] == report['training']['epochs']
        rebuilt = actor.compile(generation=4, medoids_per_group=4, preserve_calibrated_groups=True)
        dump(args.output / f'sampler/{name}_rebuilt_prior.json', rebuilt.to_dict())
        assert digest(args.output / f'sampler/{name}_rebuilt_prior.json') == digest(args.prior / f'training/{name}.json')
        dump(args.output / f'sampler/{name}_parameters.json', freeze_actor(actor))
        samplers[name] = actor
        gold_report = json.loads((reference / f'{name}_report.json').read_text())
        rewarded, gold_reward = apply_paper_rewards(read_transitions([p for p in old_paths if p.stem in
            {f'behavior_{i}' for i in range(8)}]), PaperRewardConfig(**gold_report['reward']['config']))
        assert gold_reward == gold_report['reward']
        default = DiscretePPOPolicy(seed=9610000, phase_awareness=awareness)
        DiscretePPOTrainer(PPOConfig(**gold_report['config'])).train(default, rewarded)
        dump(args.output / f'default_rebuild/{name}.json', default.compile(generation=4, medoids_per_group=4,
            preserve_calibrated_groups=True).to_dict())
        assert digest(args.output / f'default_rebuild/{name}.json') == digest(reference / f'{name}.json')
    dump(args.output / 'sampler/audit.json', dict(rebuild_seconds=time.monotonic() - started,
        frozen_sha256={n: digest(args.output / f'sampler/{n}_parameters.json') for n in samplers}))
    if args.proposal_strategy == 'canonical_masks':
        dump(args.output / 'proposal/parameters.json', dict(type='canonical_monotone_mask_proposal',
            masks=[0, 5, 10, 15], all_valid_exploration_epsilon=.2, outcomes_or_future_targets_used=False))
        samplers = dict(full=CanonicalMaskProposal(), nophase=CanonicalMaskProposal())
    elif args.proposal_strategy == 'stage_balanced':
        dump(args.output/'proposal/parameters.json', dict(type='stage_balanced_observed_write_schedule',
            canonical_masks=[0, 5, 10, 15], hot_cold_masks=[5, 10], refine_canonical_weight=.5,
            before_observed_writes='retain_current', after_observed_writes='uniform_valid',
            all_valid_exploration_epsilon=.2, outcomes_or_future_targets_used=False,
            deployed_action_overridden=False))
        member = StageBalancedProposal()
        samplers = dict(full=member, nophase=member)
    elif continuation_strategy:
        continuation = study.load(args.continuation_policy)
        assert continuation.phase_awareness == 'enabled'
        continuation.unanimous_fallback_lookup = True
        continuation.packed_fallback_lookup = False
        (args.output / 'proposal').mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.continuation_policy, args.output / 'proposal/continuation.json')
        proposal_type = ('refine_deployment_canonical_commit_deployment_uniform_valid' if args.proposal_strategy == 'phase_expansion_mixture'
                         else 'refine_half_deployment_half_canonical' if args.proposal_strategy == 'deployment_mixture'
                         else 'refine_exploration_greedy_continuation')
        dump(args.output / 'proposal/parameters.json', dict(type=proposal_type,
            masks=[0, 5, 10, 15], exploration_phase='refine', all_valid_exploration_epsilon=.2,
            continuation_sha256=digest(args.output / 'proposal/continuation.json'),
            outcomes_or_future_targets_used=False))
        member = (PhaseExpansionMixtureProposal if args.proposal_strategy == 'phase_expansion_mixture'
                  else DeploymentMixtureProposal if args.proposal_strategy == 'deployment_mixture'
                  else PhaseContinuationProposal)(continuation)
        samplers = dict(full=member, nophase=member)
    original_run, original_manager = study.run_system, engine.AgentTransactionManager
    frames = []
    def manager_factory(*a, **kw):
        manager = original_manager(*a, **kw)
        original_snapshot = manager.state_collector.snapshot
        starts, start_operations, completed = {}, {}, {}
        original_phase = manager.interceptor.phase_change
        def phase_change(txn, phase):
            context = txn.context
            if phase != context.phase:
                completed[context.tid] = context.agent_cost_ms-starts.get(context.tid, 0.)
                starts[context.tid] = context.agent_cost_ms
                start_operations[context.tid] = context.completed_operations
            return original_phase(txn, phase)
        if args.round_interval_features:
            manager.interceptor.phase_change = phase_change
        def snapshot(context, **kwargs):
            before_cost = context.agent_cost_ms
            captured = []
            if args.coherent_footprint_witness:
                kwargs['footprint_witness'] = lambda state, footprint: captured.append((state, footprint))
            result = original_snapshot(context, **kwargs)
            if args.coherent_footprint_witness:
                assert len(captured) == 1 and captured[0][0] == result
                footprint = captured[0][1]
                reads, writes, hot, targets = (set(footprint[k]) for k in
                    ('reads', 'writes', 'hot', 'protection_relevant_targets'))
            else:
                reads, writes = set(context.read_versions), set(context.write_targets)
                hot = (set(context.hot_read_targets) | set(context.hot_write_targets)) & (reads | writes)
                targets = set(kwargs['protection_relevant_targets'])
            frame = dict(txn_id=context.tid, state=dataclasses.asdict(result), reads=sorted(reads),
                writes=sorted(writes), hot=sorted(hot), protection_relevant_targets=sorted(targets))
            if args.coherent_footprint_witness:
                frame['footprint_witness_from_collector'] = True
            if args.carry_prior_restart_cost:
                frame.update(prior_retry_cost_ms=context.prior_retry_cost_ms,
                             agent_cost_ms=context.agent_cost_ms, operation_cost_ms=context.operation_cost_ms)
            if args.round_interval_features:
                frame.update(round_start_agent_cost_ms=starts.get(context.tid, 0.),
                    round_start_operations=start_operations.get(context.tid, 0),
                    previous_round_agent_cost_ms=completed.get(context.tid, 0.),
                    cumulative_agent_cost_ms=context.agent_cost_ms,
                    completed_operations=context.completed_operations,
                    witness_stable_before_after=before_cost == context.agent_cost_ms)
            frames.append(frame)
            return result
        manager.state_collector.snapshot = snapshot
        return manager
    def run_system(**kwargs):
        kwargs.update(flags)
        actual_policy = kwargs.get('policy_override')
        if isinstance(getattr(actual_policy, 'base', actual_policy), HotRulePolicy):
            # The rule never consumes hotspot_access_ratio. Preserve its
            # original cheaper encoder; do not charge it unused feature work.
            kwargs['protection_relevant_hotness'] = False
        if kwargs.get('trajectory_output') is not None:
            kwargs['paper_exploration_seed'] = None
        return original_run(**kwargs)
    study.run_system, engine.AgentTransactionManager = run_system, manager_factory
    collection, started = [], time.monotonic()
    try:
        for i, c in enumerate((16, 32, 16, 32, 16, 32)):
            frames.clear()
            sampler = ActorMixtureBehaviorPolicy(prefix + 11000 + i, (samplers['full'], samplers['nophase']))
            row = study.measure(args.output / 'sampler/full_rebuilt_prior.json', sampler, c, prefix + 10000 + i,
                ids[i], i, trajectory=args.output / f'training/{ids[i]}.json', seconds=.75, warmup=0.)
            assert row['failed_tasks'] == 0
            dump(args.output / f'feature_frames/{ids[i]}.json', frames)
            collection.append(row)
            csv_write(args.output / 'collection.csv', collection)
    finally:
        engine.AgentTransactionManager = original_manager
    paths = sorted((args.output / 'training').glob('behavior_*.json'))
    dump(args.output / 'training/audit.json', dict(common_input_sha256={p.name: digest(p) for p in paths},
        collection_seconds=time.monotonic() - started))
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_helper_ledger.py'), '--training',
        str(args.output / 'training'), '--expected-files', '6', '--output', str(args.output / 'helper_ledger_gate.json')], check=True)
    if args.collect_only:
        dump(args.output/'collection_completed.json', dict(raw_collection_complete=True,
            input_and_cost_audits_required_before_training=True, performance_run=False,
            default_adopted=False, goal_completed=False))
        return
    if args.carry_prior_restart_cost:
        subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_restart_cost_inputs.py'),
            '--input', str(args.output), '--output', str(args.output/'restart_cost_inputs_audit.json')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_protection_hotness_inputs.py'), '--input',
        str(args.output), '--output', str(args.output / 'inputs_audit.json')], check=True)
    raw, started = read_transitions(paths), time.monotonic()
    for name, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        gold_report = json.loads((reference / f'{name}_report.json').read_text())
        config = dataclasses.replace(PPOConfig(**gold_report['config']), advantage_baseline=args.advantage_baseline,
            categorical_initialization=args.categorical_initialization, ppo_reference_mode=args.ppo_reference_mode)
        rows, reward = apply_paper_rewards(raw, PaperRewardConfig(**gold_report['reward']['config']))
        actor = DiscretePPOPolicy(seed=prefix + 30000, phase_awareness=awareness, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True,
            protection_relevant_hotness=True)
        trained = DiscretePPOTrainer(config).train(actor, rows, calibration_transitions=rows)
        model = actor.compile(generation=4, medoids_per_group=4, preserve_calibrated_groups=True,
                              statewise_supported_actor=args.statewise_supported_actor)
        dump(args.output / f'training/{name}.json', model.to_dict())
        dump(args.output / f'training/{name}_report.json', dict(seed=prefix + 30000, phase_awareness=awareness,
            config=dataclasses.asdict(config), reward=reward, training=trained, generation=4, medoids_per_group=4,
            preserve_calibrated_groups=True, occ_cold_start_guard=False, categorical_actor=True,
            ppo_supported_projection=True, local_qualified_action_support=True, local_context_groups=True,
            protection_relevant_hotness=True, statewise_supported_actor=args.statewise_supported_actor,
            ppo_training_source_ids=ids, inputs=[str(p) for p in paths]))
    nophase = study.load(args.output / 'training/nophase.json')
    for t in raw:
        assert len({int(nophase.select(dataclasses.replace(t.state, phase=p)).protected)
                    for p in ('explore', 'refine', 'commit')}) == 1
    training_audit = json.loads((args.output / 'training/audit.json').read_text())
    training_audit.update(training_seconds=time.monotonic() - started, phase_checks=len(raw),
        model_sha256={n: digest(args.output / f'training/{n}.json') for n in ('full', 'nophase')})
    dump(args.output / 'training/audit.json', training_audit)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_actor_deployment_influence.py'),
        '--training', str(args.output / 'training'), '--output', str(args.output / 'actor_influence')], check=True)
    influence = json.loads((args.output / 'actor_influence/diagnostic.json').read_text())['results']
    if args.collect_and_fit_only:
        dump(args.output/'fit_completed.json', dict(offline_actor_qualified=influence['full']['fresh_gradient_observations_changed'] > 0,
            influence=influence, performance_run=False, default_adopted=False, goal_completed=False))
        return
    preflight = influence['full']['fresh_gradient_observations_changed'] > 0
    if preflight and (continuation_strategy or args.statewise_supported_actor):
        subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/diagnose_p0_deployed_actor_reachability.py'),
            '--input', str(args.output), '--output', str(args.output / 'actual_path_preflight'),
            '--seed', str(prefix + 71600)], check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/audit_p0_deployed_actor_reachability.py'),
            '--input', str(args.output / 'actual_path_preflight'), '--models', str(args.output),
            '--output', str(args.output / 'actual_path_preflight_audit.json')], check=True)
        path_audit = json.loads((args.output / 'actual_path_preflight_audit.json').read_text())
        assert path_audit['passed']
        preflight = path_audit['measurement_actor_differences'] > 0
    dump(args.output / 'preflight_completed.json', dict(performance_preflight_passed=preflight,
        actor_influence=influence, default_adopted=False, goal_completed=False))
    if not preflight:
        return
    rows = []
    for rep in range(3):
        order = ['full', 'no_actor', 'nophase', 'hotrule_always']
        random.Random(prefix + 61600 + rep).shuffle(order)
        for v in order:
            path = args.output / ('actor_influence/full_no_actor_updates.json' if v == 'no_actor'
                                  else f'training/{"nophase" if v == "nophase" else "full"}.json')
            policy = HotRulePolicy() if v == 'hotrule_always' else study.load(path)
            if v != 'hotrule_always':
                policy.unanimous_fallback_lookup = True
                policy.packed_fallback_lookup = False
                policy.exact_categorical_lookup = args.exact_categorical_lookup
            row = study.measure(path, policy, 16, prefix + 51600 + rep, v, rep, seconds=4., warmup=.2)
            row.update(unanimous_fallback_lookup=v != 'hotrule_always', packed_fallback_lookup=False)
            if args.exact_categorical_lookup:
                row['exact_categorical_lookup'] = v != 'hotrule_always'
            assert not row['helper_externality_recording_active']
            rows.append(row)
            csv_write(args.output / 'validation/raw.csv', rows)
    dump(args.output / 'completed.json', dict(rows=len(rows), raw_sha256=digest(args.output / 'validation/raw.csv'),
        comparisons={v: compare(rows, 'clients', 'full', v) for v in ('no_actor', 'nophase', 'hotrule_always')},
        benefit_gate_pending_independent_audit=True, default_adopted=False, model_promoted=False, goal_completed=False))


if __name__ == '__main__':
    main()
