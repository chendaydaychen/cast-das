#!/usr/bin/env python3
"""Offline actor-learning influence; no counterfactual deployment/performance."""
import argparse
import dataclasses
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.cc.atcc.ppo import DiscretePPOPolicy, DiscretePPOTrainer, PPOConfig
from agent.runtime import PaperRewardConfig, apply_paper_rewards, policy_transition_from_dict
from scripts.revision.run_p0_revision import digest,dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert platform.node().split(".")[0] == "node1" and not args.output.exists()
    audit = json.loads((args.training / "audit.json").read_text())
    raw = []
    paths = sorted(args.training.glob("behavior_*.json"))
    assert {p.name:digest(p) for p in paths} == audit["common_input_sha256"]
    for path in paths:
        raw.extend(policy_transition_from_dict(t,source_id=str(t.get("source_id") or path.stem))
                   for t in json.loads(path.read_text())["transitions"])
    result = {}
    for name, awareness in (("full","enabled"),("nophase","disabled")):
        report = json.loads((args.training / f"{name}_report.json").read_text())
        config = PPOConfig(**report["config"])
        rows,reward = apply_paper_rewards(raw,PaperRewardConfig(**report["reward"]["config"]))
        assert reward == report["reward"]
        support_rows = None
        if report.get("ppo_training_source_ids"):
            support_rows = rows
            ids = set(report["ppo_training_source_ids"])
            rows = [t for t in rows if t.source_id in ids]
            assert rows and {t.source_id for t in rows} == ids
        policies,compiled = {},{}
        for mode in ("original","no_actor_updates"):
            policy = DiscretePPOPolicy(seed=report["seed"],phase_awareness=awareness,
                protection_relevant_hotness=report.get("protection_relevant_hotness",False),
                growth_conditioned_groups=bool(report.get("growth_conditioned_groups",False)),
                pooled_sparse_action_support=bool(report.get("pooled_sparse_action_support",False)),
                strict_local_action_support=bool(report.get("strict_local_action_support",False)),
                ppo_sparse_fallback=bool(report.get("ppo_sparse_fallback",False)),
                ppo_supported_projection=bool(report.get("ppo_supported_projection",False)),
                categorical_actor=bool(report.get("categorical_actor",False)),
                local_qualified_action_support=bool(report.get("local_qualified_action_support",False)),
                local_context_groups=bool(report.get("local_context_groups",False)))
            used_config = config if mode == "original" else dataclasses.replace(config,learning_rate=0.,group_learning_rate=0.)
            DiscretePPOTrainer(used_config).train(policy,rows,calibration_transitions=support_rows)
            model = policy.compile(generation=report["generation"],medoids_per_group=report["medoids_per_group"],
                occ_cold_start_guard=report["occ_cold_start_guard"],
                preserve_calibrated_groups=report["preserve_calibrated_groups"],
                statewise_supported_actor=report.get("statewise_supported_actor", False),
                refinement_distance_threshold=report.get("refinement_distance_threshold"))
            target = args.output / f"{name}_{mode}.json"
            dump(target,model.to_dict())
            if mode == "original":
                assert digest(target) == audit["model_sha256"][name]
            policies[mode],compiled[mode] = policy,model
        a,b = policies["original"],policies["no_actor_updates"]
        actor_weights_a = a.categorical_actor_weights if a.categorical_actor else a.actor_weights
        actor_weights_b = b.categorical_actor_weights if b.categorical_actor else b.actor_weights
        group_logits_a = a.categorical_group_logits if a.categorical_actor else a.actor_group_logits
        group_logits_b = b.categorical_group_logits if b.categorical_actor else b.actor_group_logits
        actor_diffs = [abs(x-y) for ar,br in zip(actor_weights_a,actor_weights_b) for x,y in zip(ar,br)]
        group_diffs = [abs(x-y) for key in group_logits_a for x,y in zip(group_logits_a[key],group_logits_b[key])]
        result[name] = dict(original_rebuilt_byte_identical=True,
            compiled_byte_identical=digest(args.output / f"{name}_original.json") == digest(args.output / f"{name}_no_actor_updates.json"),
            deployed_actions_changed=sum(int(compiled["original"].select(t.state).protected) != int(compiled["no_actor_updates"].select(t.state).protected) for t in (support_rows if support_rows is not None else rows)),
            **({"fresh_gradient_observations_changed":sum(int(compiled["original"].select(t.state).protected) != int(compiled["no_actor_updates"].select(t.state).protected) for t in rows),
                "ppo_gradient_transitions":len(rows),"support_transitions":len(support_rows)}
               if support_rows is not None else {}),
            max_abs_actor_weight_difference=max(actor_diffs,default=0.),max_abs_group_logit_difference=max(group_diffs,default=0.),
            calibrated_heads_identical=compiled["original"].calibrated_group_actions == compiled["no_actor_updates"].calibrated_group_actions,
            model_sha256={mode:digest(args.output / f"{name}_{mode}.json") for mode in policies})
    dump(args.output / "diagnostic.json",dict(scope="same measured inputs; actor learning disabled only offline; not RL performance proof",
        transitions=len(raw),results=result,common_input_sha256=audit["common_input_sha256"],
        analysis_source_sha256=digest(Path(__file__)),counterfactual_deployed=False,reward_changed=False,goal_completed=False))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
