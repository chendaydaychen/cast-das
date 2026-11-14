"""Opt-in Credit fixture with a shared sector credit-capacity constraint.

This is an explicit business extension, not a relabeling of existing captures.
The planner chooses when to inspect capacity; the runtime adds no access plan.
All company and capacity writes remain private until transaction commit.
"""
import json

from .base import ObjectSpec
from .credit_phase_independent import PhaseIndependentCreditCursor, PHASE_INDEPENDENT_INSTRUCTIONS
from .credit_review import CreditReviewWorkload, parse_profile
from .frozen_credit_agent import FrozenCreditAgentCursor


PORTFOLIO_INSTRUCTIONS = PHASE_INDEPENDENT_INSTRUCTIONS + """
Additional business constraint: companies in the same sector share credit capacity.
- read_portfolio: args {}; requires an observed company profile and returns the
  sector's currently available capacity. Choose when to inspect this information.
  This query keeps the candidate proposal but invalidates its verification receipt.
- verify also requires an observed portfolio capacity. The proposed new_limit
  minus the observed company limit must not exceed that available capacity.
  If capacity is insufficient, revise the proposal before verification.
- finish atomically publishes the company result and updates remaining capacity
  by subtracting (new_limit - old_limit). Reducing a limit returns capacity.
These are simulated accounting operations. Verification never reserves capacity
outside the transaction; a concurrent change can require a transaction retry.
"""


class PortfolioCreditWorkload(CreditReviewWorkload):
    def __init__(self, config=None, *, available_capacity=500000):
        super().__init__(config)
        if type(available_capacity) is not int or available_capacity < 0:
            raise ValueError('available_capacity must be a nonnegative integer')
        self.available_capacity = available_capacity

    @staticmethod
    def capacity_key(sector):
        return f'credit:portfolio:{sector}:available_capacity'

    def objects(self):
        yield from super().objects()
        for sector in range(self.config.sector_count):
            yield ObjectSpec(self.capacity_key(sector),str(self.available_capacity),kind='row')


def portfolio_reservation_hooks(manager,delegate,*,always=False,clock=None,checkpoint_only=False):
    """Experimental common action binding for the declared pool RMW tool.

    A selected read-protection class additionally reserves this issued RMW
    suffix. Ordinary reads keep their original semantics. This action binding
    is an extension, not the original paper runtime or a learned decision.
    """
    from agent.runtime.declared_rmw import DeclaredRMWReadHooks
    from agent.runtime.item_frequency_hooks import ItemFrequencyHooks
    item=getattr(delegate,'item',None)
    frequency=isinstance(item,ItemFrequencyHooks)
    if checkpoint_only:
        from agent.runtime.rmw_policy_checkpoint import bind_issued_rmw_checkpoint
        bind_issued_rmw_checkpoint(delegate)
    def declared(txn,key):
        return key.startswith('credit:portfolio:') and key.endswith(':available_capacity')
    def select(txn,key):
        if always:return True
        if frequency:return item._eligible(txn,key,False)
        if checkpoint_only:
            from agent.runtime.rmw_policy_checkpoint import select_at_issued_rmw_request
            select_at_issued_rmw_request(delegate,txn,key)
        # A hot-only rule must see the currently issued hot key before native
        # read protection/reservation; cold keys remain optimistic.
        issued=getattr(delegate,'_issued_hot',None)
        if issued is not None and not checkpoint_only:issued(txn,key)
        return txn.context.action.protects(hot=manager.is_hot(key),write=False)
    def read_selected(txn,key,selected):
        return item.protect_selected(txn,key,write=False,selected=selected)
    return DeclaredRMWReadHooks(manager,delegate,declared=declared,select=select,
        read_selected=read_selected if frequency else None,clock=clock)


class PortfolioCreditCursor(PhaseIndependentCreditCursor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.portfolio = None

    def planner_context(self, turns_remaining):
        state=json.loads(super().planner_context(turns_remaining))
        state['observed_portfolio']=self.portfolio
        return json.dumps(state,sort_keys=True)

    def _capacity_after_proposal(self):
        if self.proposal is None or self.portfolio is None or 'limit' not in self.company:
            raise ValueError('a proposal, observed company limit and read_portfolio are required')
        delta=int(self.proposal['new_limit'])-int(self.company['limit'])
        remaining=int(self.portfolio['available_capacity'])-delta
        if remaining < 0:
            raise ValueError('insufficient portfolio capacity; revise the proposal')
        return remaining

    def _dispatch(self,command):
        if isinstance(command,dict) and command.get('tool')=='read_portfolio':
            if set(command)!={'phase','tool','args'} or command['args']!={}:
                raise ValueError('read_portfolio takes no arguments')
            if command['phase'] != self.phase or self.phase=='commit':
                raise ValueError('read_portfolio preserves the current business annotation')
            if 'profile' not in self.company:
                raise ValueError('an observed company profile is required')
            sector,_,_=parse_profile(self.company['profile'])
            key=PortfolioCreditWorkload.capacity_key(sector)
            value=int(self._read(key))
            if value < 0:
                raise ValueError('invalid negative portfolio capacity')
            self.portfolio=dict(sector=sector,object_id=key,available_capacity=value)
            self.receipt=None
            return dict(self.portfolio)
        if isinstance(command,dict) and command.get('tool') in ('verify','finish'):
            remaining=self._capacity_after_proposal()
            result=super()._dispatch(command)
            if result.get('finished'):
                # Even a zero delta records a capacity checkpoint alongside the
                # company result; disclose this footprint in capture metadata.
                self._write(self.portfolio['object_id'],str(remaining),role='portfolio_capacity')
                result['portfolio_remaining']=remaining
            return result
        return super()._dispatch(command)


class FrozenPortfolioCreditCursor(FrozenCreditAgentCursor):
    """Freeze model-selected accesses/proposal; recompute accounting on real reads.

    This does not regenerate the proposal or replan after changed observations.
    Remaining capacity is derived, not copied from the captured absolute value.
    """
    def __init__(self,record,**kwargs):
        roles=[event for event in record['events']
               if event.get('write_role')=='portfolio_capacity']
        if len(roles)!=1 or roles[0]['event']!='write':
            raise ValueError('one captured portfolio accounting write is required')
        super().__init__(record,**kwargs)

    def _write_value(self,txn,event):
        if event.get('write_role')!='portfolio_capacity':
            return super()._write_value(txn,event)
        company=self.record['task']['company_id']
        prefix=CreditReviewWorkload.company_prefix(company)
        profile_key=f'{prefix}:profile'
        limit_key=f'{prefix}:limit'
        key=event['object_id']
        if not {profile_key,limit_key,key} <= set(txn.read_set) or limit_key not in txn.write_set:
            raise ValueError('portfolio replay requires observed profile, limit, capacity and staged company limit')
        sector,_,_=parse_profile(txn.snapshot[profile_key].value)
        if key != PortfolioCreditWorkload.capacity_key(sector):
            raise ValueError('portfolio accounting key differs from observed company sector')
        delta=int(txn.write_set[limit_key].value)-int(txn.snapshot[limit_key].value)
        remaining=int(txn.snapshot[key].value)-delta
        if remaining < 0:
            raise ValueError('frozen proposal exceeds current portfolio capacity; replanning required')
        return str(remaining)
