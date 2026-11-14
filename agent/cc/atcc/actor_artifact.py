"""Versioned snapshots of paper-mask PPO actors (not deployment table fallback)."""
import copy
from .ppo import DiscretePPOPolicy

def actor_snapshot(actor):
    if actor.timing_actions:raise ValueError('Paper actor snapshots do not support extra timing actions')
    data=dict(phase_awareness=actor.phase_awareness,actor_weights=actor.actor_weights,
        actor_group_logits=dict(actor.actor_group_logits),critic_weights=actor.critic_weights,
        observed_group_actions={k:sorted(v) for k,v in actor.observed_group_actions.items()})
    if actor.categorical_actor:
        data.update(actor_artifact_version=2,categorical_actor=True,
            categorical_actor_weights=actor.categorical_actor_weights,
            categorical_group_logits=dict(actor.categorical_group_logits))
    if not actor.monotone_actions:
        data.update(actor_artifact_version=3,monotone_actions=False)
    if actor.group_mode!='legacy':
        data.update(actor_artifact_version=4,group_mode=actor.group_mode)
    if getattr(actor,'initial_bit_probability',.5)!=.5:
        data['initial_bit_probability']=actor.initial_bit_probability
    return copy.deepcopy(data)

def actor_from_snapshot(data,*,seed=0):
    data=copy.deepcopy(data)
    categorical=bool(data.get('categorical_actor',False))
    monotone=data.get('monotone_actions',True)
    group_mode=data.get('group_mode','legacy')
    if (data.get('actor_artifact_version')==4)!=(group_mode!='legacy'):
        raise ValueError('Nonlegacy residual grouping requires explicit version4 actor')
    if data.get('actor_artifact_version')==3 and monotone is not False:
        raise ValueError('Version3 actor must explicitly retain its revisable domain')
    if type(monotone) is not bool or (not monotone and data.get('actor_artifact_version') not in (3,4)):
        raise ValueError('Revisable actor requires explicit version3 action domain')
    if categorical and data.get('actor_artifact_version') not in (2,3,4):
        raise ValueError('Categorical actor requires an explicit version2 snapshot')
    actor=DiscretePPOPolicy(seed=seed,phase_awareness=data.get('phase_awareness','disabled'),
        categorical_actor=categorical,monotone_actions=monotone,group_mode=group_mode,
        initial_bit_probability=data.get('initial_bit_probability',.5))
    actor.actor_weights=data['actor_weights'];actor.actor_group_logits.update(data['actor_group_logits'])
    actor.critic_weights=data.get('critic_weights',actor.critic_weights)
    actor.observed_group_actions.update({k:set(v) for k,v in data.get('observed_group_actions',{}).items()})
    if categorical:
        actor.categorical_actor_weights=data['categorical_actor_weights']
        actor.categorical_group_logits.update(data['categorical_group_logits'])
    return actor
