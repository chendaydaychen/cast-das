"""Immutable training-derived quantile keys for an explicit future table format.

The legacy encoder remains unchanged. This encoder is not silently installed
in existing models and does not fit on queries. Policy wrappers must apply
their declared phase-feature ablation before fitting and encoding.
"""
from bisect import bisect_left
import dataclasses
import hashlib
import json
import math
from types import MappingProxyType
from .paper_policy import normalize_conflict_kind


KEY_FIELDS=('phase','inter_round_interval_ms','read_set_size','write_set_size',
    'read_set_growth','write_set_growth','access_overlap_ratio','completed_rounds',
    'completed_operations','recent_write_ratio','hotspot_access_ratio','blocked_time_ms',
    'retry_count','current_action','recent_conflict_kind','global_active_transactions',
    'global_waiter_count','global_abort_rate','global_throughput','global_avg_latency_ms',
    'global_tail_latency_ms','global_agent_task_throughput','global_agent_task_avg_latency_ms',
    'global_agent_task_tail_latency_ms','global_conflict_abort_rate','global_background_throughput',
    'global_background_abort_rate','observed_restart_cost_ms')
CATEGORICAL=frozenset(('phase','retry_count','current_action','recent_conflict_kind'))
NUMERIC=tuple(name for name in KEY_FIELDS if name not in CATEGORICAL)


def raw_value(state,name):
    value=getattr(state,name)
    if name=='phase':return str(value)
    if name=='recent_conflict_kind':return normalize_conflict_kind(value)
    if name=='current_action':
        if not 0<=int(value)<=15:raise ValueError('Four-bit current action required')
        return int(value)
    if name=='retry_count':return min(3,max(0,int(value)))
    value=float(value)
    if not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative encoder input required')
    return value


@dataclasses.dataclass(frozen=True)
class FittedPaperEncoder:
    boundaries: object
    bins: int
    state_samples: int
    state_sample_sha256: str

    def __post_init__(self):
        object.__setattr__(self,'boundaries',MappingProxyType({name:tuple(values)
            for name,values in self.boundaries.items()}))

    @classmethod
    def fit(cls,states,*,bins=8):
        if type(bins) is not int or not 2<=bins<=64:raise ValueError('Encoder bins must be an integer in [2,64]')
        rows=[tuple(raw_value(s,name) for name in KEY_FIELDS) for s in states]
        if not rows:raise ValueError('Observed training states required')
        boundaries={}
        for name in NUMERIC:
            column=KEY_FIELDS.index(name)
            values=sorted(row[column] for row in rows if row[column]>0)
            cuts=(sorted({values[math.ceil(k*len(values)/bins)-1] for k in range(1,bins)})
                  if values else [])
            boundaries[name]=tuple(v for v in cuts if v<values[-1]) if values else ()
        digest=hashlib.sha256(json.dumps(rows,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return cls(MappingProxyType(boundaries),bins,len(rows),digest)

    def key(self,state):
        result=[]
        for name in KEY_FIELDS:
            value=raw_value(state,name)
            result.append(value if name in CATEGORICAL else
                (0 if value==0 else 1+bisect_left(self.boundaries[name],value)))
        return tuple(result)

    def to_dict(self):
        return dict(artifact_type='cast-das-training-quantile-encoder',version=1,
            method='zero-separated-training-quantiles',key_fields=list(KEY_FIELDS),bins=self.bins,
            boundaries={name:list(self.boundaries[name]) for name in NUMERIC},
            state_samples=self.state_samples,state_sample_sha256=self.state_sample_sha256,
            fits_during_lookup=False)

    def validate_key(self,key):
        if len(key)!=len(KEY_FIELDS):raise ValueError('Fitted key dimension mismatch')
        for name,value in zip(KEY_FIELDS,key):
            if name in ('phase','recent_conflict_kind'):
                if not isinstance(value,str):raise ValueError('Categorical string required')
            else:
                limit=(15 if name=='current_action' else 3 if name=='retry_count'
                       else len(self.boundaries[name])+1)
                if type(value) is not int or not 0<=value<=limit:
                    raise ValueError('Invalid fitted categorical/bin value')
        return tuple(key)

    @classmethod
    def from_dict(cls,data):
        if (data.get('artifact_type')!='cast-das-training-quantile-encoder' or data.get('version')!=1
                or data.get('method')!='zero-separated-training-quantiles'
                or data.get('key_fields')!=list(KEY_FIELDS) or data.get('fits_during_lookup') is not False):
            raise ValueError('Unsupported fitted encoder schema')
        bins=data['bins'];count=data['state_samples'];digest=data['state_sample_sha256']
        if type(bins) is not int or not 2<=bins<=64 or type(count) is not int or count<1:
            raise ValueError('Invalid encoder fit population')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid encoder fit digest')
        if set(data['boundaries'])!=set(NUMERIC):raise ValueError('Encoder fields differ')
        cuts={name:tuple(map(float,data['boundaries'][name])) for name in NUMERIC}
        for values in cuts.values():
            if (len(values)>bins-1 or any(not math.isfinite(v) or v<=0 for v in values)
                    or tuple(sorted(set(values)))!=values):raise ValueError('Invalid quantile cuts')
        return cls(MappingProxyType(cuts),bins,count,digest)
