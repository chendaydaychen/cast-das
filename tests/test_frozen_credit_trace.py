import dataclasses
import pytest

from agent.runtime import AgentTransactionManager
from agent.workloads.credit_reasoner import CapturingReasoner, ReasoningResult
from agent.workloads.credit_review import CreditReviewConfig, CreditReviewWorkload
from scripts.revision.compare_frozen_credit_trace import run_system, FixedProtectionPolicy
from scripts.revision.audit_phase_policy_choices import audit_propensities
from agent.runtime.paper_policy import CompiledPhasePolicy


class DeterministicReasoner:
    def reason(self, *, label, phase, context):
        text = context.rsplit("|", 1)[-1] if label == "evidence-route" else "reviewed"
        return ReasoningResult(text, 1, 1, "test-only")


def test_exploration_rejects_action_rewriting_guards_before_execution():
    with pytest.raises(ValueError, match="unmodified policy actions"):
        run_system({}, "paper-atcc", workers=1, max_attempts=1,
                   time_scale=1, exploration_epsilon=.6, performance_guards=True)


def test_same_frozen_access_path_runs_across_two_cc_strategies():
    config = CreditReviewConfig(company_count=12, commit_apply_ms=0)
    workload = CreditReviewWorkload(config)
    manager = AgentTransactionManager(record_traces=False, collect_trajectories=False)
    workload.register(manager)
    task = workload.task_for(seed=57, worker_id=0, sequence=0)
    reasoner = CapturingReasoner(DeterministicReasoner())
    cursor = workload.cursor(task, reasoner=reasoner, sleep_fn=lambda _: None)
    txn = manager.begin("capture", strategy="silo", snapshot_object_ids=None)
    execution = cursor.execute(txn)
    assert txn.commit("silo").committed
    source = {"mode": "live", "multi_storage": False,
              "config": dataclasses.asdict(config),
              "tasks": [{"task": dataclasses.asdict(task),
                         "branch": execution.branch,
                         "access_targets": list(execution.revealed_targets),
                         "llm_calls": reasoner.records, "tool_calls": []}]}
    for system in ("silo", "2pl-wait-die"):
        attempts = []
        result = run_system(source, system, workers=1, max_attempts=1,
                            time_scale=.001, attempt_sink=attempts)
        assert result["status"] == "ok"
        assert result["performance_guards"] is False
        assert result["commits"] == 1
        assert result["llm_calls"] == len(reasoner.records)
        assert result["input_tokens"] == len(reasoner.records)
        assert result["output_tokens"] == len(reasoner.records)
        assert result["wasted_input_tokens"] == 0
        assert result["wasted_output_tokens"] == 0
        assert len(attempts) == 1
        detail = attempts[0]["attempt_details"][0]
        assert detail["elapsed_ms"] <= attempts[0]["elapsed_ms"]
        assert detail["committed"]
        assert result["action_changes"] == 0
    static = run_system(source, "paper-atcc", workers=1, max_attempts=1,
                        time_scale=.001, policy=FixedProtectionPolicy(),
                        static_protection=True)
    assert static["commits"] == 1
    assert static["action_changes"] == 0
    switches = []
    def inspect_manager(**kwargs):
        switches.append((kwargs["delayed_write_apply_enabled"], kwargs["priority_enabled"]))
        return AgentTransactionManager(**kwargs)
    disabled = run_system(source, "paper-atcc", workers=1, max_attempts=1,
                          time_scale=.001, policy=FixedProtectionPolicy(),
                          static_protection=True, manager_factory=inspect_manager,
                          delayed_write_apply_enabled=False, priority_enabled=False)
    assert switches == [(False, False)]
    assert disabled["commits"] == 1
    assert disabled["delayed_write_apply_enabled"] is False
    assert disabled["priority_enabled"] is False
    decisions = []
    recorded = run_system(source, "paper-atcc", workers=1, max_attempts=1,
                          time_scale=.001, policy=FixedProtectionPolicy(),
                          static_protection=True, decision_sink=decisions)
    assert recorded["commits"] == 1
    assert recorded["policy_calls"] == len(decisions) > 0
    assert all(record["action"] == 4 for record in decisions)
    assert recorded["lookup_ms"] >= 0
    repeated = run_system(source, "silo", workers=2, max_attempts=2,
                          time_scale=.001, repeat_factor=2)
    assert repeated["tasks"] == 2
    assert repeated["repeat_factor"] == 2
    assert repeated["independent_recorded_tasks"] == 1
    transitions = []
    explored = run_system(source, "paper-atcc", workers=1, max_attempts=1,
                          time_scale=.001, policy=CompiledPhasePolicy(),
                          trajectory_sink=transitions, exploration_epsilon=1.0)
    assert explored["commits"] == 1
    audit = audit_propensities(transitions)
    assert audit["checked"] > 0
    assert audit["recorded_action_probability_mismatches"] == 0
    assert audit["invalid_distribution"] == 0
    class AbortFirstManager(AgentTransactionManager):
        logical_ids = []

        def begin(self, *args, **kwargs):
            txn = super().begin(*args, **kwargs)
            self.logical_ids.append(txn.task_id)
            return txn

        def commit(self, txn, **kwargs):
            if txn.metadata["retry_count"] == 0:
                return txn.abort("forced identity regression test", strategy=kwargs["strategy"])
            return super().commit(txn, **kwargs)

    retry = run_system(source, "silo", workers=1, max_attempts=2, time_scale=.001,
                       manager_factory=AbortFirstManager)
    assert retry["commits"] == 1
    assert retry["attempts"] == 2
    assert AbortFirstManager.logical_ids == ["frozen:silo:0", "frozen:silo:0"]
    carried = []
    run_system(source, "silo", workers=1, max_attempts=2, time_scale=.001,
               manager_factory=AbortFirstManager, carry_prior_restart_cost=True,
               attempt_sink=carried)
    first, second = carried[0]["attempt_details"]
    assert first["carried_prior_restart_cost_ms"] == 0
    assert first["paid_replay_work_ms"] > 0
    assert second["carried_prior_restart_cost_ms"] == first["paid_replay_work_ms"]

    from agent.runtime.paper_contract import configure_paper_observed_contract
    from agent.runtime.paper_observation_policy import PaperObservationPolicy
    def paper_manager(**kwargs):
        m=AbortFirstManager(**kwargs)
        configure_paper_observed_contract(m)
        return m
    labels=[];paper_attempts=[]
    run_system(source,"paper-atcc",workers=1,max_attempts=2,time_scale=.001,
        manager_factory=paper_manager,carry_prior_restart_cost=True,
        policy=PaperObservationPolicy(CompiledPhasePolicy(phase_awareness='disabled')),
        trajectory_sink=labels,attempt_sink=paper_attempts)
    first,second=paper_attempts[0]['attempt_details']
    terminals={int(t['txn_id'].rsplit(':',2)[1]):t for t in labels if t['done']}
    assert first['paid_replay_work_ms']>0
    assert second['carried_prior_restart_cost_ms']==first['paid_replay_work_ms']
    assert terminals[1]['retry_cost_ms']==first['paid_replay_work_ms']
