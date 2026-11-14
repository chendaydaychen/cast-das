"""Opt-in undo-only recovery on the existing local heterogeneous participants.

Before-images are forced before any participant update. Every participant
installation is forced before the durable commit marker. Recovery never redoes
a committed update: it retains committed images and reverses incomplete ones.
Single coordinator/exclusive owner only, not distributed 2PC or tool recovery.
"""
from __future__ import annotations

import json

from .durable_storage import DurableRoutedVersionedKVStore


class UndoRoutedVersionedKVStore(DurableRoutedVersionedKVStore):
    backend_name='undo-durable-routed:default=sqlite,audit=atomic-json'
    journal_format='cast-das-local-undo-only-v1'

    def _batch(self,checks,writes):
        with self._mutex:
            self._gate();checks=tuple(checks);writes=tuple(writes)
            if len({str(k) for k,_,_ in writes})!=len(writes):raise ValueError('Duplicate write')
            if not self.batch_validate_versions(checks):return False
            rows=[]
            for key,value,exists in writes:
                key=str(key);before=self._image(key)
                rows.append(dict(key=key,storage=self.storage_id_for(key),before=before,
                    after=dict(value=str(value),version=int(before['version'])+1,exists=bool(exists))))
            if not rows:return True
            # The journal intentionally contains NO after-images. New values
            # are in memory until install and must be forced before COMMIT.
            undo=[{name:row[name] for name in ('key','storage','before')} for row in rows]
            try:
                with self.db:
                    cursor=self.db.execute("INSERT INTO decisions(status,body) VALUES ('UNDO_PREPARED',?)",
                        (json.dumps(undo,sort_keys=True),))
                txid=cursor.lastrowid;self._fault('after_undo_flush',txid)
                self._install(txid,rows)
                self._fault('after_all_installs_before_commit',txid)
                self._set_status(txid,'UNDO_COMMITTED');self._fault('after_commit_record',txid)
                self._set_status(txid,'UNDO_DONE');self._fault('after_done',txid)
                return True
            except BaseException:
                # No transaction or reader may observe a partial installation
                # until restart has completed recovery under the owner gate.
                self._ready=False;raise

    def _recover(self):
        report=dict(undone_incomplete=[],retained_committed=[],redo_installs=0)
        rows=self.db.execute("SELECT id,status,body FROM decisions WHERE status IN "
            "('UNDO_PREPARED','UNDO_COMMITTED') ORDER BY id DESC").fetchall()
        for txid,status,body in rows:
            undo=json.loads(body)
            if any(set(row)!={'key','storage','before'} for row in undo):
                raise RuntimeError('Invalid undo-only record or unexpected redo image')
            if status=='UNDO_COMMITTED':
                # The commit marker was forced after all participants. Do not
                # replay their writes or increment their versions again.
                for row in undo:
                    current=self._engines[row['storage']].get(row['key'])
                    if int(current['version'])<int(row['before']['version'])+1:
                        raise RuntimeError('Durable committed participant is missing')
                self._set_status(txid,'UNDO_DONE');report['retained_committed'].append(txid)
                continue
            for row in reversed(undo):
                participant=self._engines[row['storage']]
                current=participant.get(row['key']);before=row['before']
                if current!=before:
                    if int(current['version'])!=int(before['version'])+1:
                        raise RuntimeError('Cannot undo an unexpected participant version')
                    participant.install([dict(key=row['key'],before=current,after=before)])
                self._fault('after_undo_restore:'+row['storage'],txid)
            self._fault('before_undo_abort_record',txid)
            self._set_status(txid,'UNDO_ABORTED');report['undone_incomplete'].append(txid)
        return report
