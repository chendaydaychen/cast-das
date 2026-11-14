"""Training-only temperature initialization, without outcome/action labels.

After initialization this is the actual actor, sampled and differentiated by
ordinary categorical PPO. It is not a behavior mixture or replay estimator.
"""
import math
from agent.runtime.joint_mask_policy import convert_factorized_actor


def initialize_joint_actor(payload, *, seed, temperature=1.):
    temperature=float(temperature)
    if not math.isfinite(temperature) or temperature<1.:
        raise ValueError('Exploration initialization temperature must be finite and >=1')
    actor=convert_factorized_actor(payload,seed=seed)
    if temperature!=1.:
        actor.categorical_actor_weights=[[v/temperature for v in row]
            for row in actor.categorical_actor_weights]
        for key,row in actor.categorical_group_logits.items():
            actor.categorical_group_logits[key]=[v/temperature for v in row]
    return actor
