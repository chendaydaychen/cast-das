"""Offline return-based learning of the seven positive priority parameters.

The four-bit mask is still trained with PPO. This auxiliary optimizer is a
paired, antithetic evolution-strategy update over complete environment returns;
it is NOT a claim that PPO differentiates the priority formula. Every proposal,
score and update is replayable before a configuration may be installed.
"""
import dataclasses
import hashlib
import json
import math
import random

from agent.runtime.priority import PriorityConfig

PARAMETERS=('sql_weight','interval_weight','retry_weight','blocked_weight',
    'sql_quantum_ms','interval_quantum_ms','blocked_quantum_ms')
ARTIFACT_TYPE='cast-das-return-trained-priority'


def calibrated_priority_quantums(transitions,*,quantile=.75,target_quanta=8.):
    """Initialize units from training costs before measured-return optimization.

    This estimates scales, not superiority or seven final optimal parameters.
    Positive-event waits avoid zero dilution. Empty components keep an explicit
    default fallback. No evaluation data or policy action labels are used.
    """
    if not math.isfinite(quantile) or not 0<quantile<=1:
        raise ValueError('Cost quantile must be in (0,1]')
    if not math.isfinite(target_quanta) or target_quanta<=0:
        raise ValueError('Positive finite target quanta required')
    rows=list(transitions)
    terminal=[r for r in rows if r.done]
    completed=[r for r in terminal if r.commit_indicator]
    cfg=PriorityConfig(operation_cost_based=True,include_inflight_agent_cost=True,include_prior_retry_cost=False)
    populations=dict(sql_quantum_ms=[r.operation_cost_ms for r in completed],
        interval_quantum_ms=[r.agent_cost_ms for r in terminal],
        blocked_quantum_ms=[r.lock_wait_ms for r in rows])
    values={};report={}
    for name,population in populations.items():
        if any(not math.isfinite(v) or v<0 for v in population):
            raise ValueError('Training costs must be finite and nonnegative')
        positive=sorted(v for v in population if v>0)
        scale=positive[max(0,math.ceil(quantile*len(positive))-1)] if positive else None
        values[name]=max(1e-6,scale/target_quanta) if scale is not None else getattr(cfg,name)
        report[name]=dict(positive_samples=len(positive),population_samples=len(population),
            quantile_value_ms=scale,quantum_ms=values[name],default_fallback=scale is None)
    return dataclasses.replace(cfg,**values),dict(method='training-positive-cost-quantile initialization',
        quantile=quantile,target_quanta=target_quanta,components=report,
        weights_learned_here=False,evaluation_inputs_used=False)


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked_config(data):
    cfg=PriorityConfig(**data) if isinstance(data,dict) else data
    if not isinstance(cfg,PriorityConfig):raise ValueError('PriorityConfig required')
    if any(not math.isfinite(getattr(cfg,k)) or getattr(cfg,k)<=0 for k in PARAMETERS):
        raise ValueError('Positive finite learned priority parameters required')
    if not cfg.operation_cost_based or not cfg.include_inflight_agent_cost or cfg.include_prior_retry_cost:
        raise ValueError('Priority training requires the observed-service-cost runtime contract')
    return cfg


class PriorityReturnOptimizer:
    def __init__(self,initial=None,*,seed=20261022573,sigma=.15,learning_rate=.05):
        self.initial=checked_config(initial or PriorityConfig(operation_cost_based=True,
            include_inflight_agent_cost=True,include_prior_retry_cost=False))
        if not math.isfinite(sigma) or not 0<sigma<=1:
            raise ValueError('Finite sigma in (0,1] required')
        if not math.isfinite(learning_rate) or not 0<learning_rate<=1:
            raise ValueError('Finite learning rate in (0,1] required')
        self.seed=int(seed);self.sigma=float(sigma);self.learning_rate=float(learning_rate)
        self.rng=random.Random(self.seed);self.current=self.initial;self.pending=None;self.steps=[]

    def propose(self):
        if self.pending is not None:raise ValueError('Previous return pair not consumed')
        noise=[self.rng.gauss(0.,1.) for _ in PARAMETERS]
        before=dataclasses.asdict(self.current)
        def candidate(sign):
            return dataclasses.replace(self.current,**{k:getattr(self.current,k)*math.exp(
                max(-2.,min(2.,sign*self.sigma*z))) for k,z in zip(PARAMETERS,noise)})
        plus,minus=candidate(1.),candidate(-1.)
        self.pending=dict(iteration=len(self.steps),before=before,noise=noise,
            plus=dataclasses.asdict(plus),minus=dataclasses.asdict(minus))
        return plus,minus

    def update(self,plus_score,minus_score):
        if self.pending is None:raise ValueError('Return pair has no pending proposal')
        if not all(math.isfinite(s) for s in (plus_score,minus_score)):
            raise ValueError('Finite measured environment returns required')
        advantage=(plus_score-minus_score)/max(1.,abs(plus_score)+abs(minus_score))
        values={}
        for k,z in zip(PARAMETERS,self.pending['noise']):
            step=self.learning_rate*advantage*z/(2*self.sigma)
            proposal=getattr(self.current,k)*math.exp(max(-.5,min(.5,step)))
            values[k]=max(getattr(self.initial,k)/4.,min(getattr(self.initial,k)*4.,proposal))
        self.current=dataclasses.replace(self.current,**values)
        row=dict(self.pending,plus_score=float(plus_score),minus_score=float(minus_score),
            normalized_advantage=advantage,after=dataclasses.asdict(self.current))
        self.steps.append(row);self.pending=None
        return row

    def artifact(self,training_inputs):
        if self.pending is not None or not self.steps:raise ValueError('Completed priority updates required')
        return dict(artifact_type=ARTIFACT_TYPE,version=1,method='paired-antithetic-environment-return-ES',
            parameters=list(PARAMETERS),initial_config=dataclasses.asdict(self.initial),
            config=dataclasses.asdict(self.current),seed=self.seed,sigma=self.sigma,
            learning_rate=self.learning_rate,steps=self.steps,training_inputs=dict(training_inputs),
            steps_sha256=digest(self.steps),
            evaluation_inputs_used=False,score='sum of jointly normalized paper rewards / actual replay wall seconds',
            effective_update=any(r['before']!=r['after'] for r in self.steps))


def validate_priority_artifact(data):
    if data.get('artifact_type')!=ARTIFACT_TYPE or data.get('version')!=1:
        raise ValueError('Unsupported learned-priority artifact')
    if data.get('method')!='paired-antithetic-environment-return-ES' or data.get('steps_sha256')!=digest(data.get('steps',[])):
        raise ValueError('Priority measured-score chain or optimizer method differs')
    if (data.get('parameters')!=list(PARAMETERS) or data.get('evaluation_inputs_used') is not False
            or not data.get('training_inputs')):
        raise ValueError('Missing training-only priority provenance')
    if any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
            for v in data['training_inputs'].values()):raise ValueError('Invalid training input digest')
    optimizer=PriorityReturnOptimizer(checked_config(data['initial_config']),seed=data['seed'],
        sigma=data['sigma'],learning_rate=data['learning_rate'])
    for row in data['steps']:
        optimizer.propose()
        replay=optimizer.update(row['plus_score'],row['minus_score'])
        if replay!=row:raise ValueError('Priority numerical update chain differs')
    if not optimizer.steps or dataclasses.asdict(optimizer.current)!=data['config']:
        raise ValueError('Priority configuration differs from measured updates')
    if data.get('effective_update')!=any(r['before']!=r['after'] for r in optimizer.steps):
        raise ValueError('Priority effective-update flag differs')
    if not data['effective_update']:raise ValueError('No effective learned priority update')
    return checked_config(data['config'])
