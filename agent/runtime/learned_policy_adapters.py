"""Explicit shared-runtime adaptations, NOT complete original learned systems.

Transfer frozen learned operation rows. Access positions beyond the training
descriptor use its final row (reported, not hidden). Guard/chop operations map
to full observed-prefix validation, and task-level retry replaces piece retry.
Original priority, guard scheduling, buffering and backoff are NOT reproduced.
"""
from .speculative_policy_runtime import SpeculativeDecision


class AdaptedLearnedPolicy:
    def __init__(self, rows, *, family):
        if not rows: raise ValueError('Empty learned policy')
        self.rows=tuple(rows); self.family=family
        self.calls=0; self.tail_calls=0

    def select(self, state):
        index=state['access_index']
        if type(index) is not int or index<0: raise ValueError('Invalid access index')
        self.calls+=1; self.tail_calls+=int(index>=len(self.rows))
        return self.rows[min(index,len(self.rows)-1)]

    @classmethod
    def polyjuice(cls,text):
        lines=text.strip().splitlines()
        if lines.count('normal access')!=1: raise ValueError('Missing Polyjuice access rows')
        rows=[]
        for line in lines[lines.index('normal access')+1:]:
            fields=line.split()
            if len(fields)!=4: raise ValueError('Expected single-type Polyjuice row')
            a,w,p,g=map(int,fields)
            if a not in (0,1) or w not in (0,1) or p not in (0,1) or g<0:
                raise ValueError('Invalid Polyjuice action')
            # Author access=0 permits public/dirty accesses; piece marks
            # cumulative exposure, wait is adapted to early validation.
            rows.append(SpeculativeDecision(read_dirty=not bool(a),
                expose=bool(p),early_validate=bool(w or p)))
        return cls(rows,family='Polyjuice-adapted')

    @classmethod
    def neurcc(cls,text):
        # Author training appends a human-readable diagnostic after the six
        # policy fields. It is not executable policy data.
        lines=text.strip().splitlines()
        labels=('conflict detection:','wait priorities:','timeout:','expose:','chop wait:','extra:')
        if len(lines)<12 or tuple(lines[:12:2])!=labels: raise ValueError('Invalid NeurCC framing')
        detection=lines[1].strip(); exposed=lines[7].strip()
        priorities=lines[3].split(); timeouts=lines[5].split(); guards=lines[9].split()
        if not detection or any(len(x)!=len(detection) for x in (exposed,priorities,timeouts,guards)):
            raise ValueError('NeurCC row counts differ')
        rows=[]
        for d,e,g in zip(detection,exposed,guards):
            if d not in '0123' or e not in '01' or int(g)<0: raise ValueError('Invalid NeurCC action')
            rows.append(SpeculativeDecision(read_dirty=d!='0',expose=e=='1',
                early_validate=d!='0' or int(g)>0))
        return cls(rows,family='NeurCC-adapted')

    def scope(self):
        return dict(family=self.family,policy_rows=len(self.rows),calls=self.calls,
            final_row_extension_calls=self.tail_calls,frozen_transfer=True,
            native_guard_priority_buffer_backoff=False,partial_piece_retry=False,
            guarded_actions='observed-prefix validation',retry_scope='whole task')
