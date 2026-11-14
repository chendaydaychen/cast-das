"""Opt-in common-snapshot certification, using retained committed versions."""
def install(manager):
    if getattr(manager,'read_only_snapshot_certification_enabled',False):raise ValueError('Install once')
    manager._enable_paper_versioning()
    manager.read_only_snapshot_certification_enabled=True
    manager.read_only_snapshot_certificate_metrics=dict(accepted=0,rejected=0)
    manager.read_only_snapshot_certificates=[]
    original=manager.begin
    def begin(*v,**kw):
        tx=original(*v,**kw)
        if tx.metadata.get('paper_atcc',False) and not tx.context.is_background:
            epoch=manager.version_manager.current_epoch()
            tx.metadata['_read_certificate_begin_epoch']=epoch
            # Keep history from begin; do not change online read values or set
            # the transaction's snapshot_epoch. Existing finish/abort unpins.
            manager.version_manager.pin(tx.context.tid,epoch)
        return tx
    manager.begin=begin
    return manager.read_only_snapshot_certificate_metrics
