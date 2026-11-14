"""Reconstruct frozen coefficient actors for measured continuation rollouts."""
import dataclasses
import hashlib
import json
from pathlib import Path

from agent.cc.atcc.ppo import DiscretePPOPolicy
from agent.runtime.residual_policy import ResidualTablePolicy
from scripts.revision.run_p0_actor_sampling import ActorMixtureBehaviorPolicy


def actor_from_payload(payload, *, seed):
    compiled = ResidualTablePolicy.from_dict(payload)
    actor = DiscretePPOPolicy(seed=seed, phase_awareness=compiled.phase_awareness,
        stay_probability=compiled.stay_probability,
        growth_conditioned_groups=compiled.growth_conditioned_groups,
        local_context_groups=compiled.local_context_groups)
    actor.actor_weights = [list(row) for row in compiled.weights]
    actor.actor_group_logits.update({key:list(row) for key, row in compiled.groups.items()})
    actor.observed_group_actions.update({key:set(actions) for key, actions in compiled.support.items()})
    reconstructed = ResidualTablePolicy.compile(actor, generation=compiled.generation,
        remove_phase_index_proxy=compiled.remove_phase_index_proxy)
    if reconstructed.to_dict() != payload:
        raise ValueError('Frozen actor does not reconstruct its coefficient artifact')
    return actor


class FrozenActorDistribution:
    def __init__(self, payload, *, seed):
        self.actor = actor_from_payload(payload, seed=seed)
        self.remove_phase_index_proxy = payload['remove_phase_index_proxy']

    def probabilities(self, state):
        if self.remove_phase_index_proxy:
            state = dataclasses.replace(state, completed_rounds=0)
        return tuple(self.actor.probabilities(state))


def load_distributions(folder, *, seed):
    result = []
    for arm, awareness in (('full', 'enabled'), ('nophase', 'disabled')):
        payload = json.loads((Path(folder)/f'{arm}_policy.json').read_text())
        if payload['phase_awareness'] != awareness or payload['remove_phase_index_proxy'] != (arm == 'nophase'):
            raise ValueError('Expected Full and proxy-free NoPhase actors')
        result.append(FrozenActorDistribution(payload, seed=seed))
    return tuple(result)


class LateActorMixtureBehavior(ActorMixtureBehaviorPolicy):
    """Collector only: retain late choices while using actual frozen distributions."""
    stage_weights = {'explore': .02, 'refine': .2, 'commit': 1.}

    def probabilities(self, state):
        if state.phase not in self.stage_weights or not 0 <= int(state.current_action) < 16:
            raise ValueError('Original phase and legal mask required')
        base = super().probabilities(state)
        weight = self.stage_weights[state.phase]
        return tuple(weight*p + (1.-weight if action == state.current_action else 0.)
                     for action,p in enumerate(base))


def training_ancestry_batches(folder):
    """Give Linear the same measured information used by continued actors."""
    batches, seen, hashes = [], set(), set()
    expected_source = None
    while folder is not None:
        folder = Path(folder).resolve()
        if folder in seen:
            raise ValueError('Cyclic training model ancestry')
        seen.add(folder)
        protocol = json.loads((folder/'protocol.json').read_text())
        if expected_source is None:
            expected_source = protocol['source_sha256']
        elif protocol['source_sha256'] != expected_source:
            raise ValueError('Training ancestry changes capture sources')
        path = folder/'training_transitions.json'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in hashes:
            batches.append((path, folder/'training_report.json'))
            hashes.add(sha)
        # A fresh run inherits a capture/schema, not learned parent information.
        # Its future Linear continuation must stop at this reset boundary too.
        parent = None if protocol.get('fresh_initialization') else protocol.get('sampling_model_folder')
        if parent:
            for filename, expected in protocol['sampling_model_sha256'].items():
                if filename not in {'full_policy.json', 'nophase_policy.json'}:
                    raise ValueError('Unexpected parent model filename')
                if hashlib.sha256((Path(parent)/filename).read_bytes()).hexdigest() != expected:
                    raise ValueError('Training ancestor model changed')
        folder = parent
    return list(reversed(batches))
