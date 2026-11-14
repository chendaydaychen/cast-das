"""Real SIGKILL cuts through Aegis/Silo commits on persistent heterogeneous KV."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.runtime import AgentTransactionManager
from agent.runtime.durable_storage import DurableRoutedVersionedKVStore
from scripts.revision.run_current_credit_ablations import read,sha
from scripts.revision.evaluate_phase_pipeline_frozen import load_models
from scripts.revision.run_agent_schedule_comparison_20261012 import configure_frozen_runtime
from scripts.revision.train_phase_pipeline_pilot import ObservedAccessHooks

STAGES=('after_prepare','after_decision','after_participant:audit',
        'after_participant:default','before_done','after_done')


def audit(output):
    plan=read(output/'protocol.json');reference=Path(plan['reference'])
    if sha(output/'runner_executed.py')!=plan['source_sha256']:raise ValueError('Executed source changed')
    for name,digest in plan.get('runtime_sources',{}).items():
        if sha(output/'sources'/name)!=digest:raise ValueError('Runtime source snapshot changed')
    for name,digest in plan['artifact_sha256'].items():
        if sha(reference/name)!=digest:raise ValueError('Frozen artifact changed')
    rebuilt=[];kills=0
    for arm in plan['arms']:
        for stage in plan['stages']:
            for interrupted in ([False,True] if stage=='after_decision' else [False]):
                case=output/(arm+'_'+stage.replace(':','_')+('_recovery_kill' if interrupted else ''))
                evidence=read(case/'evidence.json');row=evidence['row']
                if not row['passed'] or row['recovery_ms']<0:raise ValueError('Invalid case result')
                if (row['arm'],row['stage'],row['recovery_interrupted'])!=(arm,stage,interrupted):raise ValueError('Case differs')
                for cut,name,expected_stage in ((evidence['commit_cut'],'commit_cut.json',stage),
                        (evidence['recovery_cut'],'recovery_cut.json','after_participant:audit')):
                    if cut is None:
                        if name=='recovery_cut.json' and not interrupted:continue
                        raise ValueError('Missing kill')
                    if cut['returncode']!=-signal.SIGKILL or read(case/name)!=cut['marker'] or cut['marker']['stage']!=expected_stage or cut['marker']['txid']!=4:
                        raise ValueError('SIGKILL evidence differs')
                    kills+=1
                undecided=stage=='after_prepare';version=2 if undecided else 3
                if (row['recovered_source'],row['recovered_ledger'],row['version'])!=('10' if undecided else '11','20' if undecided else '21',version):
                    raise ValueError('Recovered result differs from decision')
                expected_report=dict(discarded_prepared=[4] if undecided else [],
                    redone_committed=[4] if stage not in ('after_prepare','after_done') else [])
                if evidence['recovery_report']!=expected_report:raise ValueError('Recovery log classification differs')
                if evidence['acknowledged']!=read(case/'acknowledged.json') or not evidence['acknowledged']['committed']:
                    raise ValueError('Prior acknowledgement missing')
                if plan.get('paper_observed_contract'):
                    for record in (evidence['acknowledged'],evidence['post_recovery_transaction']):
                        expected=1 if arm=='full' else 0
                        if record.get('observed_policy_boundaries')!=expected:
                            raise ValueError('Observed-contract policy boundary differs')
                if (evidence['acknowledged']['source'],evidence['acknowledged']['ledger'])!=('10','20'):
                    raise ValueError('Earlier acknowledged values differ')
                with sqlite3.connect('file:'+str(case/'coordinator.sqlite')+'?mode=ro',uri=True) as db:
                    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Journal corrupt')
                    decisions=db.execute('SELECT id,status,body FROM decisions ORDER BY id').fetchall()
                    routes=dict(db.execute('SELECT key,storage FROM routes').fetchall())
                if routes!={'source':'default','ledger':'audit'} or [d[0] for d in decisions]!=[1,2,3,4,5]:
                    raise ValueError('Route or decision history differs')
                if [d[1] for d in decisions]!=['DONE','DONE','DONE','ABORTED' if undecided else 'DONE','DONE']:
                    raise ValueError('Durable decision status differs')
                victim={r['key']:r for r in json.loads(decisions[3][2])}
                if set(victim)!={'source','ledger'}:raise ValueError('Missing cross-store write')
                for key,value in [('source','11'),('ledger','21')]:
                    if victim[key]['after']!=dict(value=value,version=3,exists=True):raise ValueError('Victim after-image differs')
                    if victim[key]['before']!=dict(value='10' if key=='source' else '20',version=2,exists=True):raise ValueError('Acknowledged before-image differs')
                with sqlite3.connect('file:'+str(case/'records.sqlite')+'?mode=ro',uri=True) as db:
                    actual_source=json.loads(db.execute('SELECT image FROM kv WHERE key=?',('source',)).fetchone()[0])
                actual_ledger=read(case/'audit.json')['kv']['ledger']
                if actual_source!=dict(value='12',version=version+1,exists=True) or actual_ledger!=dict(value='22',version=version+1,exists=True):
                    raise ValueError('Raw persisted state differs after resumed transaction')
                rebuilt.append(row)
    if rebuilt!=read(output/'results.json'):raise ValueError('Results matrix differs')
    result=dict(passed=True,cases=len(rebuilt),actual_sigkills=kills,
        recovered_values_and_versions_verified=True,raw_journal_and_participants_verified=True,
        caveat='Controlled Linux process crashes, single serialized coordinator; not power loss, remote failures or external tool recovery')
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    (output/'auditor_executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result),flush=True);return result


def manager_for(store,reference,arm,attach=False,paper_observed=False):
    if paper_observed:
        from agent.runtime.paper_policy import CompiledPhasePolicy
        from agent.runtime.paper_contract import configure_paper_observed_contract
        manager=AgentTransactionManager(store=store,
            paper_policy=CompiledPhasePolicy.load(reference/'full_table.json'),
            record_traces=False,collect_trajectories=False,performance_guards_enabled=False,
            low_conflict_occ_guard=False,delayed_write_apply_enabled=True)
        configure_paper_observed_contract(manager,
            priority_training_artifact=read(reference/'priority.json'))
    else:
        runtime=read(reference/'protocol.json')['runtime']
        policy=load_models(reference)['full_table'] if arm=='full' else None
        manager=AgentTransactionManager(store=store,paper_policy=policy,record_traces=False,
            collect_trajectories=False,performance_guards_enabled=False,low_conflict_occ_guard=False,
            delayed_write_apply_enabled=True,priority_enabled=True,
            policy_selected_write_admission_only=runtime['policy_selected_write_admission_only'])
        configure_frozen_runtime(manager,runtime)
        if arm=='full':manager.interceptor.hooks=ObservedAccessHooks(manager)
    for key,storage,value in (('source','default','0'),('ledger','audit','0')):
        if attach:manager.attach_existing_object(key,kind='row',storage_id=storage)
        else:manager.register_object(key,value,kind='row',storage_id=storage)
    return manager


def update(manager,arm,values):
    strategy='paper-atcc' if arm=='full' else 'silo'
    boundaries_before=len(getattr(manager,'paper_contract_boundaries',()))
    txn=manager.begin('durable-cross-storage',strategy=strategy)
    txn.enter_phase('explore');txn.read('source');txn.read('ledger');txn.enter_phase('refine')
    txn.write('source',values[0]);txn.write('ledger',values[1])
    if getattr(manager,'paper_observed_contract',None):
        # Actual fifth/sixth operations exercise the current policy boundary
        # and private read-your-writes, without inventing a reasoning delay.
        if txn.read('source').value!=values[0] or txn.read('ledger').value!=values[1]:
            raise ValueError('Cross-store private read-your-writes failed')
    result=txn.commit(strategy)
    if not result.committed:raise ValueError('Uncontended durable transaction aborted')
    record=dict(committed=True,source=manager.store.get('source').value,
        ledger=manager.store.get('ledger').value,changes=sum(e.kind=='action-change' for e in txn.events))
    if getattr(manager,'paper_observed_contract',None):
        record['observed_policy_boundaries']=len(manager.paper_contract_boundaries)-boundaries_before
        if arm=='full' and record['observed_policy_boundaries']<1:
            raise ValueError('Current learned policy was not invoked')
    return record


def kill_hook(target,marker):
    def hook(stage,txid):
        if stage==target:
            with marker.open('w') as handle:
                json.dump(dict(stage=stage,txid=txid,signal='SIGKILL'),handle);handle.flush();os.fsync(handle.fileno())
            os.kill(os.getpid(),signal.SIGKILL)
    return hook


def worker(args):
    protocol=read(args.directory.parent/'protocol.json')
    for name,digest in protocol.get('runtime_sources',{}).items():
        if sha(ROOT/name)!=digest:raise ValueError('Worker runtime differs from frozen sources')
    if args.worker=='recovery':
        DurableRoutedVersionedKVStore(args.directory,fault_hook=kill_hook('after_participant:audit',args.marker))
        raise ValueError('Recovery fault hook not reached')
    with DurableRoutedVersionedKVStore(args.directory) as store:
        manager=manager_for(store,args.reference,args.arm,paper_observed=args.paper_observed_contract)
        acknowledged=update(manager,args.arm,('10','20'))
        (args.directory/'acknowledged.json').write_text(json.dumps(acknowledged))
        store._fault_hook=kill_hook(args.stage,args.marker)
        update(manager,args.arm,('11','21'))
        raise ValueError('Commit fault hook not reached')


def launch(args,case,mode,marker):
    command=[sys.executable,str(Path(__file__)),'--worker',mode,'--reference',str(args.reference),
        '--directory',str(case),'--arm',args.arm,'--stage',args.stage,'--marker',str(marker)]
    if args.paper_observed_contract:command.append('--paper-observed-contract')
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    if result.returncode!=-signal.SIGKILL or not marker.exists():
        raise ValueError('Worker not killed at declared cut: '+result.stderr)
    return dict(returncode=result.returncode,marker=read(marker),command=command)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--worker',choices=('commit','recovery'));p.add_argument('--directory',type=Path)
    p.add_argument('--marker',type=Path);p.add_argument('--arm',choices=('full','silo'))
    p.add_argument('--stage',choices=STAGES);p.add_argument('--audit-only',action='store_true')
    p.add_argument('--paper-observed-contract',action='store_true')
    args=p.parse_args()
    if args.worker:return worker(args)
    if not args.output:p.error('output required')
    if args.audit_only:return audit(args.output)
    args.output.mkdir(parents=True,exist_ok=False)
    model=(dict(artifact_sha256={name:sha(args.reference/name) for name in ('full_table.json','priority.json')})
           if args.paper_observed_contract else read(args.reference/'protocol.json'))
    for name,digest in model['artifact_sha256'].items():
        if sha(args.reference/name)!=digest:raise ValueError('Frozen artifact differs')
    plan=dict(reference=str(args.reference.resolve()),artifact_sha256=model['artifact_sha256'],
        stages=STAGES,arms=['full','silo'],source_sha256=sha(Path(__file__)),new_api_calls=0,
        paper_observed_contract=args.paper_observed_contract,
        scope='Single Linux coordinator; SQLite KV + atomic-file KV; process crash, not power loss/network partitions; commits serialized')
    if args.paper_observed_contract:
        files=list((ROOT/'agent').rglob('*.py'))+list(ROOT.glob('cast_core*.so'))
        plan['runtime_sources']={str(f.relative_to(ROOT)):sha(f) for f in files}
        for name in plan['runtime_sources']:
            target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    (args.output/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for arm in plan['arms']:
        for stage in STAGES:
            for interrupt_recovery in ([False,True] if stage=='after_decision' else [False]):
                case=args.output/(arm+'_'+stage.replace(':','_')+('_recovery_kill' if interrupt_recovery else ''))
                case.mkdir();args.arm=arm;args.stage=stage
                cut=launch(args,case,'commit',case/'commit_cut.json');recovery_cut=None
                if interrupt_recovery:recovery_cut=launch(args,case,'recovery',case/'recovery_cut.json')
                start=time.perf_counter()
                with DurableRoutedVersionedKVStore(case) as store:
                    recovery_ms=(time.perf_counter()-start)*1000
                    expected=('10','20') if stage=='after_prepare' else ('11','21')
                    expected_version=2 if stage=='after_prepare' else 3
                    actual=(store.get('source').value,store.get('ledger').value)
                    versions=(store.get_version('source'),store.get_version('ledger'))
                    if actual!=expected or versions!=(expected_version,expected_version):raise ValueError('Partial/lost/duplicate recovered writes')
                    recovery=dict(store.recovery_report)
                with DurableRoutedVersionedKVStore(case) as store:
                    if any(store.recovery_report.values()):raise ValueError('Recovery not idempotent')
                    if (store.get('source').value,store.get('ledger').value)!=expected:raise ValueError('Second open changed data')
                    manager=manager_for(store,args.reference,arm,attach=True,
                        paper_observed=args.paper_observed_contract)
                    post=update(manager,arm,('12','22'))
                    if store.get_version('source')!=expected_version+1 or store.get_version('ledger')!=expected_version+1:raise ValueError('Resumed transaction version differs')
                row=dict(arm=arm,stage=stage,recovery_interrupted=interrupt_recovery,
                    recovered_source=actual[0],recovered_ledger=actual[1],version=expected_version,
                    recovery_ms=recovery_ms,passed=True)
                evidence=dict(row=row,commit_cut=cut,recovery_cut=recovery_cut,recovery_report=recovery,
                    acknowledged=read(case/'acknowledged.json'),post_recovery_transaction=post)
                (case/'evidence.json').write_text(json.dumps(evidence,indent=2));rows.append(row)
                print(json.dumps(row),flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2))
    with (args.output/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(passed=True,cases=len(rows),actual_sigkills=16,scope=plan['scope'])),flush=True)
    for name,digest in plan.get('runtime_sources',{}).items():
        if sha(ROOT/name)!=digest:raise ValueError('Runtime changed during fault cases')
    audit(args.output)


if __name__=='__main__':main()
