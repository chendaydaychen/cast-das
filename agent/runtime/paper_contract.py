"""Explicit opt-in runtime contract for paper §§4.1/5/7.

Legacy experiments keep their original hooks and definitions. This contract
does not claim learned priority parameters, distributed commit or opacity.
"""
import dataclasses
import json
import math
import threading
from agent.cc.locks import CONFLICT_SIGNAL_SCHEMA

from .paper_hooks import PaperATCCHooks
from .priority import PriorityConfig
from .window_hotness import WindowFrequencyHotnessTracker


class PaperObservedHooks(PaperATCCHooks):
    """Invoke after an observed reasoning round or five completed operations.

    Explicit benchmark phase changes are telemetry only, never decision gates.
    Guards keyed to workload names are not part of this contract.
    """
    def _select_and_apply(self,txn,state):
        refinement=getattr(self.manager,'selective_refinement',None)
        bundle=self.manager.paper_policy.snapshot()
        if refinement is not None:refinement.before_lookup(txn)
        super()._select_and_apply(txn,state)
        if refinement is not None:refinement.after_lookup(txn,state,bundle)

    def on_begin(self, txn):
        super().on_begin(txn)
        if not self.enabled(txn):return
        if txn.context.planned_write_targets:
            raise ValueError('Paper observed contract forbids planned write sets')
        txn.metadata['_defer_policy_write_locks']=False
        txn.metadata['_paper_last_decision_operations']=0

    def on_phase_change(self, txn, phase):
        pass

    def _start_online_observed_prefix(self, txn):
        # No global publication gate spanning a transaction's reasoning.
        pass

    def _decide(self, txn, source):
        if not self.enabled(txn):return
        txn.check_active()
        self.manager.interceptor.account_agent_interval(txn)
        self.manager.refresh_atcc_priority(txn)
        refinement=getattr(self.manager,'selective_refinement',None)
        if refinement is not None:refinement.poll(txn)
        self._select_and_apply(txn,self._state(txn))
        txn.metadata['_paper_last_decision_operations']=txn.context.completed_operations
        self._record_boundary(txn,source)

    def _record_boundary(self,txn,source):
        detail=dict(source=source,completed_operations=txn.context.completed_operations,
            completed_rounds=self.manager.state_collector._agent_round_counts.get(txn.context.tid,0))
        txn._event('paper-policy-boundary',detail)
        with self.manager._lock:
            self.manager.paper_contract_boundaries.append(dict(task_id=txn.task_id,**detail))

    def on_operation_finished(self, txn):
        last=txn.metadata.get('_paper_last_decision_operations',0)
        if txn.context.completed_operations-last>=5:
            self._decide(txn,'five-completed-operations')

    def on_agent_round_finished(self, txn):
        self._decide(txn,'observed-agent-round-finished')

    def before_expensive_operation(self, txn, delay_ms):
        # No future duration input and no additional pre-call decision.
        pass

    def before_commit(self, txn):
        if self.enabled(txn):
            refinement=getattr(self.manager,'selective_refinement',None)
            if refinement is not None:refinement.poll(txn)
            self.manager.interceptor.account_agent_interval(txn)
            self.manager.refresh_atcc_priority(txn)

    def _apply_protection_guard(self, txn, state, selected):
        return selected

    @staticmethod
    def _tpcc_all_agent_exact_retry(txn):
        return False

    def on_finish(self,txn):
        super().on_finish(txn)
        refinement=getattr(self.manager,'selective_refinement',None)
        if refinement is not None:refinement.forget(txn)
        self.publish_completed_trajectory(txn)

    def publish_completed_trajectory(self,txn):
        refresh=getattr(self.manager,'paper_policy_refresh',None)
        if refresh is not None and self.enabled(txn):
            if txn.result is None:return
            with txn._agent_work_gate:
                if txn.context.inflight_agent_started_ns:
                    txn.metadata['_offline_refresh_waiting_for_paid_call']=True
                    return
                if txn.metadata.get('_offline_refresh_published'):return
                txn.metadata['_offline_refresh_published']=True
                txn.metadata['_offline_refresh_waiting_for_paid_call']=False
            rows=tuple(t for t in self.manager.trajectory_collector.snapshot() if t.txn_id==txn.context.tid)
            replay=dict(task_id=txn.task_id,committed=bool(txn.result and txn.result.committed),
                operations=list(txn.metadata.get('_offline_replay_operations',[])),
                initial=dict(txn.metadata.get('_offline_replay_initial',{})),
                trailing_paid_agent_ms=max(0.,txn.context.agent_cost_ms-
                    float(txn.metadata.get('_offline_replay_agent_cost_ms',0.))),
                gap_scope='accounted already-paid agent work, excluding lock wait; not live replanning')
            try:
                refresh.observe_transaction(txn.context.tid,rows,self.manager.paper_runtime_metrics(),
                    replay_input=json.dumps(replay,sort_keys=True))
            except Exception:
                with txn._agent_work_gate:txn.metadata['_offline_refresh_published']=False
                raise


class PaperFirstAccessHotHooks(PaperObservedHooks):
    """Reviewer rule may act at an issued hot access before returning its value.

    This is not a future access oracle. Frequency counting already happened in
    the interceptor for the operation currently being requested by the agent.
    """
    def _issued_hot(self,txn,key):
        if not self.enabled(txn) or not self.manager.peek_hot(key):return
        if int(txn.context.action.protected)&5==5:return
        state=dataclasses.replace(self._state(txn),hotspot_access_ratio=1.)
        self.manager.refresh_atcc_priority(txn)
        self._select_and_apply(txn,state)
        self._record_boundary(txn,'reviewer-first-eligible-hot-access')

    def before_read(self,txn,object_id):
        self._issued_hot(txn,object_id)
        return super().before_read(txn,object_id)

    def before_write(self,txn,object_id):
        self._issued_hot(txn,object_id)
        return super().before_write(txn,object_id)


class ContentionEWMA:
    """Smooth sampled window conflict rates once per new outcome generation.

    Repeated lookups without another completed attempt do not alter EWMA.
    Other measurements (especially cost/reward telemetry) remain unmodified.
    """
    fields=('abort_rate','conflict_abort_rate','background_abort_rate')

    def __init__(self, manager, alpha=.25):
        self.manager=manager;self.raw=manager.paper_runtime_metrics
        self.alpha=alpha;self.values={};self.generation=None
        self.lock=threading.RLock()

    def __call__(self):
        # Serialize sampling as well as smoothing so concurrent queries cannot
        # install an older observation after a newer one.
        with self.lock:
            result=self.raw()
            generation=self.manager.runtime_outcome_diagnostics()['recorded']
            if generation!=self.generation:
                for field in self.fields:
                    value=float(result[field])
                    self.values[field]=(value if field not in self.values else
                        self.alpha*value+(1-self.alpha)*self.values[field])
                self.generation=generation
            return dict(result,**self.values)


def configure_paper_observed_contract(manager, *, priority_config=None,priority_training_artifact=None,
                                      hot_window_ms=1000., rank_refresh_ms=10.,metrics_window_ms=1000.,
                                      allow_action_mask_reduction=False,collect_reviewer_contention_statistics=True,
                                      explicit_phase_label_ablation=False,
                                      explicit_paid_agent_intervals=False):
    if getattr(manager,'paper_observed_contract',None):
        raise ValueError('Paper observed contract already configured')
    if not math.isfinite(metrics_window_ms) or metrics_window_ms<=0:
        raise ValueError('Positive finite runtime metrics window required')
    if type(explicit_paid_agent_intervals) is not bool:
        raise ValueError('Boolean explicit paid-agent accounting option required')
    from .paper_policy import StaticThresholdPhasePolicy
    policy=manager.paper_policy.snapshot()
    if type(collect_reviewer_contention_statistics) is not bool:
        raise ValueError('Boolean reviewer contention telemetry option required')
    if not collect_reviewer_contention_statistics and getattr(policy,'requires_reviewer_contention_statistics',True):
        raise ValueError('Policy may consume reviewer contention summaries; cannot omit them')
    manager.collect_reviewer_contention_statistics=collect_reviewer_contention_statistics
    if getattr(policy,'monotone_actions',True)!=(not allow_action_mask_reduction):
        raise ValueError('Policy and runtime action domains differ')
    if type(explicit_phase_label_ablation) is not bool:
        raise ValueError('Boolean explicit phase-label ablation option required')
    if not explicit_phase_label_ablation and (
            getattr(policy,'phase_awareness','disabled')!='disabled'
            or isinstance(policy,StaticThresholdPhasePolicy)):
        raise ValueError('Paper observed contract rejects explicit phase-dependent policies')
    priority_provenance=None
    if priority_training_artifact is not None:
        from agent.cc.atcc.priority_learning import validate_priority_artifact,digest
        learned=validate_priority_artifact(priority_training_artifact)
        if priority_config is not None and priority_config!=learned:
            raise ValueError('Runtime priority differs from its learned artifact')
        priority_config=learned
        priority_provenance=dict(artifact_sha256=digest(priority_training_artifact),
            method=priority_training_artifact['method'],updates=len(priority_training_artifact['steps']))
    manager.paper_observed_contract=dict(version=1,explicit_phase_input=False,
        reviewer_contention_statistics_collected=collect_reviewer_contention_statistics,
        omitted_reviewer_fields=[] if collect_reviewer_contention_statistics else
            ['max_observed_hot_accesses','max_observed_hot_contention_rate'],
        policy_boundaries='after-observed-round-or-five-operations',
        hotspot='top10pct-of-window-active-objects',hot_window_ms=hot_window_ms,
        hot_rank_refresh_ms=rank_refresh_ms,hot_ties='lexical-object-key',
        metrics_window_ms=metrics_window_ms,
        contention_ewma_alpha=.25,contention_ewma_sampling='one update per queried new outcome generation',
        conflict_classification_schema=CONFLICT_SIGNAL_SCHEMA,
        protected_reads_bypassable=False,performance_guards=False,
        category_write_lock_timing='on issued write, not deferred to commit',
        priority_operation_input='measured service milliseconds excluding lock wait',
        priority_parameters_learned=priority_training_artifact is not None,
        priority_training=priority_provenance,offline_refresh_enabled=False)
    manager.paper_observed_contract.update(
        write_publication_mode='transaction-private-write-set-until-atomic-commit',
        legacy_delay_flag_controls_publication=False,
        legacy_write_lock_deferral_overridden=True,
        cost_priority_enabled=bool(manager.priority_enabled))
    manager.paper_contract_boundaries=[]
    if explicit_phase_label_ablation:
        manager.paper_observed_contract.update(version=2,
            observation_experiment='explicit-phase-label-only-ablation',
            explicit_phase_input=getattr(policy,'phase_awareness','disabled')=='enabled',
            phase_source='current execution context only; no future trace access',
            phase_changes_trigger_policy=False,derived_phase_features_retained=True)
    manager.allow_future_action_mask_reduction=bool(allow_action_mask_reduction)
    if allow_action_mask_reduction:
        manager.paper_observed_contract.update(action_mask_domain='revisable-future-mask-held-locks-retained',
            action_change_unlocks_held_locks=False,experimental_interpretation=True)
    manager.runtime_metrics_window_seconds=metrics_window_ms/1000.
    manager._paper_metrics_cache_at=0.
    manager.hotness_tracker=WindowFrequencyHotnessTracker(window_ms=hot_window_ms,
        rank_refresh_ms=rank_refresh_ms)
    # The lock callback captures the previous tracker during construction.
    manager.atcc_locks._contention_callback=manager.hotness_tracker.observe_contention
    manager.exact_hotness_accounting=True
    manager.consistent_hotness_classification=True
    manager.unique_attempt_outcomes=True
    manager.performance_guards_enabled=False
    manager.low_conflict_occ_guard=False
    manager.first_access_read_refresh=True
    manager.complete_category_read_protection=True
    manager.policy_controlled_retry_protection=True
    manager.reconcile_selected_reads_at_checkpoints=True
    manager.policy_selected_write_admission_only=True
    manager.expensive_operation_checkpoint_enabled=False
    manager.pre_expensive_validation_enabled=False
    manager.defer_policy_protection_until_commit=False
    manager.paper_static_initial_mask=0
    manager.track_inflight_agent_cost=True
    if explicit_paid_agent_intervals:
        from .explicit_agent_cost_accounting import install as install_explicit_paid
        install_explicit_paid(manager)
    manager.paper_observed_contract['agent_interval_accounting']=(
        'explicit-paid-agent-intervals' if explicit_paid_agent_intervals else 'legacy-between-operation-wall-time')
    # A wounded attempt may finish its in-flight external call, but must not
    # launch another LLM/tool call after the runtime has already aborted it.
    manager.stop_cancelled_external_work=True
    manager.paper_observed_contract['stop_cancelled_external_work']=True
    manager.atcc_locks.refresh_priorities_before_wound=True
    # Equation (3) is integer-valued: a one-unit increase can change the
    # priority/start-time order of an already queued request. Do not inherit
    # the legacy two-unit queue-update hysteresis in the paper contract.
    manager.atcc_locks._priority_reorder_threshold=1
    manager.paper_observed_contract['pending_priority_update_quantum']=1
    manager.atcc_locks.strict_read_protection=True
    manager.state_collector.observed_agent_rounds=True
    manager.state_collector.round_interval_features=True
    manager.state_collector.observed_restart_cost_features=True
    manager.priority_manager.config=dataclasses.replace(priority_config or PriorityConfig(),
        operation_cost_based=True,include_inflight_agent_cost=True,include_prior_retry_cost=False)
    manager.paper_observed_contract['priority_config']=dataclasses.asdict(manager.priority_manager.config)
    manager.interceptor.hooks=(PaperFirstAccessHotHooks(manager)
        if getattr(policy,'first_access_hot_rule',False) else PaperObservedHooks(manager))
    manager.paper_observed_contract['reviewer_first_eligible_rule']=isinstance(
        manager.interceptor.hooks,PaperFirstAccessHotHooks)
    local_threshold=getattr(policy,'item_frequency_threshold',None)
    if local_threshold is not None:
        from .item_frequency_hooks import ItemFrequencyHooks
        item=ItemFrequencyHooks(manager,minimum_accesses=local_threshold,
            window_ms=getattr(policy,'item_frequency_window_ms',1000.))
        manager.interceptor.hooks=item
        manager.paper_item_frequency_hooks=item
        manager.paper_observed_contract.update(policy_boundaries='issued-access-local-frequency-no-PE',
            local_frequency_rule=dict(threshold=item.minimum_accesses,window_ms=item.window_ns/1e6,
                action_unit='current object; category mask unchanged'))
    manager.paper_runtime_metrics=ContentionEWMA(manager)
    return dict(manager.paper_observed_contract)
