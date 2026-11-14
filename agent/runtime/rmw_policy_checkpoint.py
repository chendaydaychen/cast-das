"""Optional issued-RMW decision schedule, not the original general policy loop.

This isolates the actual reservation choice from stochastic monotonic mask
expansions on unrelated earlier reads. All algorithms retain normal data hooks;
the selection checkpoint opens only for the currently issued declared RMW read.
"""


def bind_issued_rmw_checkpoint(delegate):
    original=getattr(delegate,'_select_and_apply',None)
    if original is None:return
    def select_only_at_request(txn,state):
        if txn.metadata.get('_declared_rmw_checkpoint_open',False):
            return original(txn,state)
    delegate._select_and_apply=select_only_at_request


def select_at_issued_rmw_request(delegate,txn,key):
    select=getattr(delegate,'_select_and_apply',None)
    if select is None:return
    marker='_declared_rmw_checkpoint_open'
    previous=txn.metadata.get(marker,False)
    txn.metadata[marker]=True
    try:
        issued=getattr(delegate,'_issued_hot',None)
        if issued is not None:issued(txn,key)
        select(txn,delegate._state(txn))
    finally:
        if previous:txn.metadata[marker]=previous
        else:txn.metadata.pop(marker,None)
