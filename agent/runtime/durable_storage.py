"""Opt-in single-owner durable router for SQLite + atomic-file KV participants.

All access must go through this router. A durable coordinator COMMITTED decision
precedes participant installation; recovery redoes decided batches and discards
undecided PREPARED batches before opening the read gate. This is not network 2PC,
and does not recover arbitrary external stores or external tool side effects.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Callable
import uuid

from .types import SnapshotValue


def sync_directory(path: Path) -> None:
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def sqlite_connection(path: Path):
    db=sqlite3.connect(str(path),check_same_thread=False)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=EXTRA')
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
        db.close();raise RuntimeError('SQLite integrity check failed')
    return db


EMPTY=dict(value='',version=0,exists=False)


class SQLiteParticipant:
    """Private participant; callers must not bypass the coordinator."""
    def __init__(self,path: Path):
        self.path=path;self.db=sqlite_connection(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, image TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS identity (id TEXT NOT NULL)')
        if not self.db.execute('SELECT id FROM identity').fetchone():
            self.db.execute('INSERT INTO identity VALUES (?)',(uuid.uuid4().hex,))
        self.db.commit();sync_directory(path.parent)
        self.identity=self.db.execute('SELECT id FROM identity').fetchone()[0]

    def get(self,key):
        row=self.db.execute('SELECT image FROM kv WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else dict(EMPTY)

    def install(self,rows):
        with self.db:
            for row in rows:
                current=self.get(row['key'])
                if current==row['after']:continue
                if current!=row['before']:raise RuntimeError('Participant version/value diverged')
                self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',
                    (row['key'],json.dumps(row['after'],sort_keys=True)))

    def close(self):self.db.close()


class FileParticipant:
    """Whole-image JSON KV; atomic replace and file/directory fsync."""
    def __init__(self,path: Path):
        self.path=path
        if not path.exists():self._save(dict(identity=uuid.uuid4().hex,kv={}))
        self.state=json.loads(path.read_text(encoding='utf-8'))
        self.identity=self.state['identity']

    def _save(self,state):
        temporary=self.path.with_suffix(self.path.suffix+'.next')
        with temporary.open('w',encoding='utf-8') as handle:
            json.dump(state,handle,sort_keys=True);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,self.path);sync_directory(self.path.parent)

    def get(self,key):return dict(self.state['kv'].get(key,EMPTY))

    def install(self,rows):
        state=dict(identity=self.identity,kv=dict(self.state['kv']))
        for row in rows:
            current=state['kv'].get(row['key'],EMPTY)
            if current==row['after']:continue
            if current!=row['before']:raise RuntimeError('Participant version/value diverged')
            state['kv'][row['key']]=row['after']
        self._save(state);self.state=state

    def close(self):pass


class DurableRoutedVersionedKVStore:
    """Linux-only, exclusive-owner crash-recoverable routed versioned KV.

    The coordinator directory owns both participant files and the journal. No
    direct participant access is supported while open. Commits are serialized.
    Any install/fault-hook error poisons the read gate until close and restart.
    A commit whose response was lost may nevertheless have a durable decision.
    """
    backend_name='durable-routed:default=sqlite,audit=atomic-json'
    journal_format='cast-das-local-decision-redo-v1'
    owns_recovery=True

    def __init__(self,directory: str|Path,*,fault_hook: Callable|None=None):
        if os.name!='posix':raise RuntimeError('Durable experimental router requires Linux/POSIX')
        import fcntl
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        sync_directory(self.directory.parent)
        self._mutex=threading.RLock();self._ready=False;self._closed=False
        self._fault_hook=fault_hook;self._engines={};self.db=None
        self._owner=(self.directory/'owner.lock').open('a+b')
        try:
            fcntl.flock(self._owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self._owner.close();raise RuntimeError('Durable store already has an owner')
        try:
            self._engines['default']=SQLiteParticipant(self.directory/'records.sqlite')
            self._engines['audit']=FileParticipant(self.directory/'audit.json')
            self.db=sqlite_connection(self.directory/'coordinator.sqlite')
            self.db.execute('CREATE TABLE IF NOT EXISTS routes (key TEXT PRIMARY KEY, storage TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL, body TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS journal_format (id INTEGER PRIMARY KEY CHECK(id=1), name TEXT NOT NULL)')
            journal=self.db.execute('SELECT name FROM journal_format WHERE id=1').fetchone()
            if journal is not None and journal[0]!=self.journal_format:
                raise RuntimeError('Durable journal protocol differs; refusing reinterpretation')
            if journal is None:
                legacy=self.db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
                if legacy and self.journal_format!='cast-das-local-decision-redo-v1':
                    raise RuntimeError('Cannot reinterpret a legacy decision-redo journal as undo-only')
                self.db.execute('INSERT INTO journal_format VALUES (1,?)',(self.journal_format,))
            binding=json.dumps({k:dict(identity=v.identity,path=str(v.path)) for k,v in self._engines.items()},sort_keys=True)
            previous=self.db.execute('SELECT body FROM binding WHERE id=1').fetchone()
            if previous and previous[0]!=binding:raise RuntimeError('Participant identity/path changed')
            if not previous:self.db.execute('INSERT INTO binding VALUES (1,?)',(binding,))
            self.db.commit();sync_directory(self.directory)
            self.recovery_report=self._recover()
            self._ready=True
        except BaseException:
            self.close();raise

    def _gate(self):
        if not self._ready or self._closed:raise RuntimeError('Store unavailable: recovery/restart required')

    def _fault(self,stage,txid):
        if self._fault_hook:self._fault_hook(stage,txid)

    def _set_status(self,txid,status):
        with self.db:self.db.execute('UPDATE decisions SET status=? WHERE id=?',(status,txid))

    def _install(self,txid,rows):
        for storage in sorted({r['storage'] for r in rows}):
            self._engines[storage].install([r for r in rows if r['storage']==storage])
            self._fault('after_participant:'+storage,txid)

    def _recover(self):
        report=dict(discarded_prepared=[],redone_committed=[])
        for txid,status,body in self.db.execute("SELECT id,status,body FROM decisions WHERE status IN ('PREPARED','COMMITTED') ORDER BY id").fetchall():
            if status=='PREPARED':
                self._set_status(txid,'ABORTED');report['discarded_prepared'].append(txid)
            else:
                self._install(txid,json.loads(body));self._set_status(txid,'DONE')
                report['redone_committed'].append(txid)
        return report

    def register_route(self,object_id,storage_id='default'):
        key=str(object_id)
        with self._mutex:
            self._gate()
            if storage_id not in self._engines:raise ValueError('Unknown storage identifier')
            if self.db.execute('SELECT 1 FROM routes WHERE key=?',(key,)).fetchone():
                raise ValueError('Object already routed')
            with self.db:self.db.execute('INSERT INTO routes VALUES (?,?)',(key,storage_id))

    def unregister_route(self,object_id):
        with self._mutex:
            self._gate()
            with self.db:self.db.execute('DELETE FROM routes WHERE key=?',(str(object_id),))

    def storage_id_for(self,object_id):
        with self._mutex:
            self._gate()
            row=self.db.execute('SELECT storage FROM routes WHERE key=?',(str(object_id),)).fetchone()
            if not row:raise ValueError('Object has no durable route')
            return row[0]

    def _image(self,key):return self._engines[self.storage_id_for(key)].get(key)

    def get(self,key):
        with self._mutex:
            self._gate();row=self._image(str(key))
            return SnapshotValue(str(row['value']),int(row['version']),bool(row['exists']))

    def get_version(self,key):return self.get(key).version

    def batch_validate_versions(self,checks):
        with self._mutex:
            self._gate()
            return all(self.get_version(str(key))==int(version) for key,version in checks)

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
            try:
                with self.db:
                    cursor=self.db.execute("INSERT INTO decisions(status,body) VALUES ('PREPARED',?)",(json.dumps(rows,sort_keys=True),))
                txid=cursor.lastrowid;self._fault('after_prepare',txid)
                self._set_status(txid,'COMMITTED');self._fault('after_decision',txid)
                self._install(txid,rows);self._fault('before_done',txid)
                self._set_status(txid,'DONE');self._fault('after_done',txid)
                return True
            except BaseException:
                self._ready=False;raise

    def batch_put_if_version(self,checks,writes):
        return self._batch(checks,[(k,v,True) for k,v in writes])

    def put(self,key,value):
        self.batch_put_if_version((),[(str(key),str(value))]);return self.get(str(key))

    def put_if_version(self,key,version,value):
        return self.batch_put_if_version([(str(key),int(version))],[(str(key),str(value))])

    def delete_if_version(self,key,version):
        return self._batch([(str(key),int(version))],[(str(key),'',False)])

    def close(self):
        with self._mutex:
            if self._closed:return
            self._ready=False;self._closed=True
            for engine in self._engines.values():engine.close()
            if self.db is not None:self.db.close()
            self._owner.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
