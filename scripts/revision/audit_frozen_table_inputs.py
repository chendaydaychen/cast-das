"""Compare a previously frozen actor and pure table on identical runtime inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.cc.atcc.ppo import valid_actions
from agent.runtime.residual_policy import ResidualTablePolicy
from agent.runtime.state_collector import phase_aware_state_from_dict
from scripts.revision.evaluate_frozen_agent_unscaled_20261012 import load_pure_table
from scripts.revision.frozen_actor_collection import FrozenActorDistribution


def compare_inputs(rows, original, scoring, table):
    if not rows:
        raise ValueError("Nonempty runtime decision inputs required")
    totals = Counter()
    phases = {}
    mismatches = []
    for index, row in enumerate(rows):
        state = phase_aware_state_from_dict(row["state"])
        legal = valid_actions(state.current_action)
        probabilities = original.probabilities(state)
        expected = max(legal, key=lambda action: probabilities[action])
        score_action = scoring.select(state).encoded
        table_action = table.select(state).encoded
        logged_action = row["action"]
        if any(action not in legal for action in (expected, score_action, table_action, logged_action)):
            raise ValueError("Illegal or non-monotone protection action")
        phase = phases.setdefault(state.phase, Counter())
        for counts in (totals, phase):
            counts.update(inputs=1, scoring_matches=int(score_action == expected),
                          table_matches=int(table_action == expected),
                          logged_matches=int(logged_action == expected))
        if table_action != expected:
            mismatches.append(dict(index=index, phase=state.phase, current_action=state.current_action,
                                   retry_count=state.retry_count, actor_action=expected,
                                   table_action=table_action))

    def rates(counts):
        return dict(counts, scoring_agreement=counts["scoring_matches"] / counts["inputs"],
                    pure_table_agreement=counts["table_matches"] / counts["inputs"],
                    logged_agreement=counts["logged_matches"] / counts["inputs"])

    return dict(overall=rates(totals), phases={phase: rates(counts) for phase, counts in phases.items()},
                pure_table_mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-folder", type=Path, required=True)
    parser.add_argument("--table-audit", type=Path, required=True)
    parser.add_argument("--table-budget", type=int, default=4)
    parser.add_argument("--evaluation-folder", type=Path, required=True)
    parser.add_argument("--record-template", default="{arm}_scoring_records.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous audits")
    protocol = json.loads((args.training_folder / "protocol.json").read_text())
    report = dict(arms={}, table_audit_sha256=hashlib.sha256(args.table_audit.read_bytes()).hexdigest(),
        source="Frozen actor parameters reconstructed exactly from saved coefficient artifact",
        table_source="Existing hash-checked training-labelled artifact; no recompilation or evaluation labels",
        scope="Same-input action preservation only, not performance or universal coverage")
    for arm in ("full", "nophase"):
        model_path = args.training_folder / f"{arm}_policy.json"
        payload = json.loads(model_path.read_text())
        original = FrozenActorDistribution(payload, seed=protocol["actor_seed"])
        scoring = ResidualTablePolicy.from_dict(payload)
        table, table_path = load_pure_table(args.table_audit, args.training_folder, arm, args.table_budget)
        filename = args.record_template.format(arm=arm)
        if Path(filename).name != filename:
            parser.error("Record template must name a file in evaluation folder")
        input_path = args.evaluation_folder / filename
        result = compare_inputs(json.loads(input_path.read_text())["decisions"], original, scoring, table)
        result.update(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                      table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
                      records_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
        report["arms"][arm] = result
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({arm: result["overall"] for arm, result in report["arms"].items()}, indent=2))
    if any(result["overall"]["scoring_agreement"] != 1.0 for result in report["arms"].values()):
        raise RuntimeError("Scoring representation does not preserve the reconstructed actor")


if __name__ == "__main__":
    main()
