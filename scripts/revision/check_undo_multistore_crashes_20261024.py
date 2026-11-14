"""Real SIGKILL cuts of undo-only routed commits and interrupted recovery.

Two local heterogeneous participants, one coordinator, exclusive owner.
The paper arm uses a scripted safety-only mask, NOT a trained Full policy or
performance comparator. No distributed 2PC, external tool or native DB durability
claim. Only subprocesses created by this script are killed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.runtime import AgentTransactionManager
from agent.runtime.context import LockAction
from agent.runtime.paper_contract import configure_paper_observed_contract
from agent.runtime.paper_observation_policy import PaperObservationPolicy
from agent.runtime.paper_policy import CompiledPhasePolicy
from agent.runtime.undo_storage import UndoRoutedVersionedKVStore

STAGES=('after_undo_flush','after_participant:audit','after_participant:default',
    'after_all_installs_before_commit','after_commit_record','after_done')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())


def physical(directory):
    with sqlite3.connect('file:'+str(directory/'records.sqlite')+'?mode=ro',uri=True) as db:
        source=json.loads(db.execute("SELECT image FROM kv WHERE key='source'").fetchone()[0])
    return dict(source=source,audit=read(directory/'audit.json')['kv']['audit'])


def child(a):
    def cut(stage,txid):
        if stage!=a.cut:return
        save(a.case/(a.child+'_cut.json'),dict(pid=os.getpid(),stage=stage,txid=txid))
        while True:time.sleep(1.)
    with UndoRoutedVersionedKVStore(a.case,fault_hook=cut) as store:
        if a.child=='recover':return
        manager=AgentTransactionManager(store=store,collect_trajectories=False,
            paper_policy=PaperObservationPolicy(CompiledPhasePolicy(phase_awareness='disabled')))
        if a.arm=='paper-atcc':configure_paper_observed_contract(manager)
        manager.attach_existing_object('source');manager.attach_existing_object('audit',storage_id='audit')
        txn=manager.begin('crash-integrity-only',{},strategy=a.arm)
        assert txn.read('source').value=='10' and txn.read('audit').value=='20'
        if a.arm=='paper-atcc':manager.transition_atcc_action(txn,LockAction.from_encoded(15))
        txn.write('source','11');txn.write('audit','21')
        result=txn.commit(a.arm)
        save(a.case/'commit_acknowledged.json',dict(committed=result.committed))


def kill_at(case,arm,mode,stage):
    marker=case/(mode+'_cut.json')
    command=[sys.executable,str(Path(__file__).resolve()),'--child',mode,
        '--case',str(case),'--arm',arm,'--cut',stage]
    with (case/(mode+'.log')).open('xb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20.;receipt=None
            while time.monotonic()<deadline:
                if marker.exists():
                    try:receipt=read(marker)
                    except ValueError:time.sleep(.01);continue
                    break
                if process.poll() is not None:raise RuntimeError(f'{mode} exited before requested cut; log retained')
                time.sleep(.01)
            if receipt is None:raise RuntimeError('Requested crash hook was not reached')
            assert receipt['pid']==process.pid and receipt['stage']==stage and process.poll() is None
            # The Popen handle identifies this script's own live child exactly.
            process.kill();code=process.wait(timeout=10.)
            assert code==-signal.SIGKILL
            return dict(command=command,marker=receipt,returncode=code)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10.)


def audit(output):
    plan=read(output/'protocol.json');cases=read(output/'cases.json');kills=0
    for name,h in plan['sources'].items():assert sha(output/'sources'/name)==h,name
    assert len(cases)==14
    for record in cases:
        folder=output/record['case'];e=read(folder/'evidence.json')
        assert e==record['evidence']
        for name in ('commit','recover'):
            cut=e.get(name+'_kill')
            if cut is None:assert name=='recover' and not record['recovery_interrupted'];continue
            assert cut['returncode']==-signal.SIGKILL
            assert cut['marker']['txid']==3
            assert cut['marker']==read(folder/(name+'_cut.json'));kills+=1
        committed=record['stage'] in ('after_commit_record','after_done')
        expected=dict(source=dict(value='11' if committed else '10',version=2 if committed else 1,exists=True),
            audit=dict(value='21' if committed else '20',version=2 if committed else 1,exists=True))
        assert physical(folder)==expected==e['recovered']
        old=dict(value='10',version=1,exists=True);old_audit=dict(value='20',version=1,exists=True)
        installed=dict(source=dict(value='11',version=2,exists=True),audit=dict(value='21',version=2,exists=True))
        before=(dict(source=old,audit=old_audit) if record['stage']=='after_undo_flush' else
            dict(source=old,audit=installed['audit']) if record['stage']=='after_participant:audit' else installed)
        assert e['physical_after_commit_kill']==before
        assert e['recovery_report']['redo_installs']==0
        if not committed:assert e['recovery_report']['undone_incomplete']==[3]
        if record['stage']=='after_commit_record':assert e['recovery_report']['retained_committed']==[3]
        assert not e['reopened_again_report']['undone_incomplete']
        assert not e['reopened_again_report']['retained_committed']
        with sqlite3.connect('file:'+str(folder/'coordinator.sqlite')+'?mode=ro',uri=True) as db:
            assert db.execute('SELECT name FROM journal_format WHERE id=1').fetchone()[0]=='cast-das-local-undo-only-v1'
            assert not db.execute("SELECT id FROM decisions WHERE status='UNDO_PREPARED'").fetchall()
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            for body, in db.execute('SELECT body FROM decisions'):
                assert all(set(row)=={'key','storage','before'} for row in json.loads(body))
    assert kills==16
    result=dict(passed=True,cases=len(cases),verified_sigkills=kills,
        interrupted_recoveries=2,undo_only_journal=True,redo_installs=0,
        values_versions_and_second_restart_verified=True,
        scope=__doc__)
    save(output/'independent_audit.json',result);print(json.dumps(result),flush=True)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);a.output=a.output.resolve()
    source_names=['scripts/revision/check_undo_multistore_crashes_20261024.py']
    source_names += [str(p.relative_to(ROOT)) for p in (ROOT/'agent').rglob('*.py')]
    sources={name:sha(ROOT/name) for name in source_names}
    for name in sources:
        dest=a.output/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    save(a.output/'protocol.json',dict(scope=__doc__,sources=sources,
        arms=['silo','paper-atcc'],stages=STAGES,scripted_mask=15,
        trained_policy_performance_claimed=False,distributed_commit_claimed=False))
    records=[]
    for arm in ('silo','paper-atcc'):
        for stage in STAGES:
            for interrupted in ([False,True] if stage=='after_all_installs_before_commit' else [False]):
                name=arm+'_'+stage.replace(':','_')+('_recovery_cut' if interrupted else '')
                case=a.output/name;assert case.parent==a.output
                with UndoRoutedVersionedKVStore(case) as store:
                    store.register_route('source');store.register_route('audit','audit')
                    store.put('source','10');store.put('audit','20')
                commit_kill=kill_at(case,arm,'commit',stage)
                forensic=physical(case)
                recovery_kill=kill_at(case,arm,'recover','after_undo_restore:audit') if interrupted else None
                started=time.perf_counter()
                with UndoRoutedVersionedKVStore(case) as recovered:
                    report=recovered.recovery_report
                    images={key:dict(value=recovered.get(key).value,version=recovered.get_version(key),exists=recovered.get(key).exists)
                        for key in ('source','audit')}
                elapsed=(time.perf_counter()-started)*1000.
                with UndoRoutedVersionedKVStore(case) as again:second=again.recovery_report
                evidence=dict(commit_kill=commit_kill,recover_kill=recovery_kill,
                    physical_after_commit_kill=forensic,recovered=images,
                    recovery_report=report,reopened_again_report=second,recovery_ms=elapsed)
                save(case/'evidence.json',evidence)
                records.append(dict(case=name,arm=arm,stage=stage,recovery_interrupted=interrupted,evidence=evidence))
                save(a.output/('case_count_'+str(len(records))+'.json'),dict(completed_cases=len(records)))
                print(dict(case=name,recovered=images,redo=report['redo_installs']),flush=True)
    save(a.output/'cases.json',records)
    for name,h in sources.items():assert sha(ROOT/name)==h,name
    audit(a.output)
    save(a.output/'completion.json',dict(completed=True,all_results_retained=True,independent_audit_passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--case',type=Path)
    p.add_argument('--child',choices=('commit','recover'));p.add_argument('--arm',choices=('silo','paper-atcc'))
    p.add_argument('--cut');p.add_argument('--audit',action='store_true')
    a=p.parse_args();assert platform.node().split('.')[0]=='node1'
    if a.child:child(a)
    elif a.audit:audit(a.output)
    else:run(a)
