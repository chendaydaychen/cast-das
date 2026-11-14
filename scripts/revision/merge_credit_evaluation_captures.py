"""Combine genuine held-out intents verbatim; never claim a merged file is a new live capture."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.workloads.frozen_credit_agent import is_complete_aborted_intent
from scripts.revision.merge_agent_training_captures import COMPATIBILITY_FIELDS, digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate(source):
    if source.get('mode')!='live' or source.get('agent_directed') is not True or not source.get('tasks'):
        raise ValueError('Original model-directed live capture required')
    for task in source['tasks']:
        if not task.get('committed') and not is_complete_aborted_intent(task):
            raise ValueError('Incomplete agent planning cannot enter evaluation')
        accesses=[e['object_id'] for e in task['events'] if e['event'] in ('read','write')]
        if accesses!=task['access_targets'] or len(accesses)!=task['operation_count']:
            raise ValueError('Captured access sequence differs')
        for event,field in (('llm','llm_calls'),('external_tool','tool_calls')):
            if sum(e['event']==event for e in task['events'])!=len(task[field]):
                raise ValueError('Incomplete recorded call stream')
            for call in task[field]:
                duration=call['elapsed_ms']
                if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not math.isfinite(duration) or duration<0:
                    raise ValueError('Invalid recorded call duration')
        if sum(e['event']=='agent_tool' for e in task['events'])!=len(task['llm_calls']):
            raise ValueError('Incomplete model action stream')


def merge(sources, forbidden_companies):
    if len(sources)<2:
        raise ValueError('At least two recorded sources required')
    tasks,origins,seen=[],[],set()
    for source_index,source in enumerate(sources):
        validate(source)
        for name in (*COMPATIBILITY_FIELDS,'phase_independent_planning'):
            if name not in source or source[name]!=sources[0].get(name):
                raise ValueError('Incompatible source field: '+name)
        for task_index,task in enumerate(source['tasks']):
            company=task['task']['company_id']
            if company in forbidden_companies:
                raise ValueError('Evaluation overlaps training/calibration company')
            fingerprint=digest(task)
            if fingerprint in seen:
                raise ValueError('Duplicate intent is not an independent recorded task')
            seen.add(fingerprint)
            tasks.append(copy.deepcopy(task))
            origins.append(dict(source_index=source_index,task_index=task_index,
                task_sha256=fingerprint,company_id=company,original_committed=bool(task['committed'])))
    # Keep only compatible metadata: aggregated counters/start times would be misleading.
    merged={name:copy.deepcopy(sources[0][name]) for name in (*COMPATIBILITY_FIELDS,'phase_independent_planning')}
    merged.update(tasks=tasks,workers=len(tasks),merged_evaluation_only=True,
        capture_role='Verbatim union of prior real intents for frozen replay; not one new live capture')
    return merged,origins


def footprint(source,repeat_factor=1):
    templates=source['tasks']
    tasks=templates*repeat_factor
    sets=[]
    for task in tasks:
        reads={e['object_id'] for e in task['events'] if e['event']=='read'}
        writes={e['object_id'] for e in task['events'] if e['event']=='write'}
        sets.append((reads,writes))
    edges=[]
    for i,(reads,writes) in enumerate(sets):
        for j in range(i):
            other_reads,other_writes=sets[j]
            shared=(writes&(other_reads|other_writes))|(other_writes&reads)
            if shared:
                edges.append(dict(first=j,second=i,shared_mutable_items=len(shared)))
    return dict(logical_tasks=len(tasks),recorded_intents=len(templates),repeat_factor=repeat_factor,
        companies=sorted({t['task']['company_id'] for t in tasks}),
        conflict_edges=edges,total_pairs=len(tasks)*(len(tasks)-1)//2,
        caveat='Potential read/write conflicts from recorded footprints, not actual overlap or a causal performance bound')


def load_merged(folder,training_folder,capture_folder):
    manifest=read(folder/'manifest.json')
    original=read(training_folder/'protocol.json')
    canonical_train=read(capture_folder/'train_live.json')
    forbidden={t['task']['company_id'] for t in canonical_train['tasks']}
    forbidden.update(original.get('additional_training_company_ids',[]))
    sources=[]
    for record in manifest['sources']:
        name=record['copied_filename']
        if Path(name).name!=name or sha(folder/name)!=record['sha256']:
            raise ValueError('Original evaluation source changed')
        sources.append(read(folder/name))
    reconstructed,origins=merge(sources,forbidden)
    path=folder/'evaluation_live.json'
    if (reconstructed!=read(path) or sha(path)!=manifest['merged_sha256']
            or origins!=manifest['task_origins']):
        raise ValueError('Merged task payload or provenance changed')
    canonical=read(capture_folder/'evaluation_live.json')
    for field in ('model','config','multi_storage','agent_directed'):
        if reconstructed[field]!=canonical[field]:
            raise ValueError('Merged input differs from frozen runtime contract: '+field)
    return reconstructed,dict(merged_sha256=sha(path),manifest_sha256=sha(folder/'manifest.json'),
        companies=sorted({t['task']['company_id'] for t in reconstructed['tasks']}),
        independent_recorded_intents=len(reconstructed['tasks']),new_api_calls=0,
        independent_fresh_confirmation=False,source_payloads_preserved=True,
        scope='Previously inspected held-out real intents combined; no phase/access/delay/value rewriting')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',type=Path,required=True)
    p.add_argument('--expected-sha256',nargs='+',required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--prior-capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if len(args.sources)!=len(args.expected_sha256) or len(set(p.resolve() for p in args.sources))!=len(args.sources):
        p.error('Unique sources and matching declared hashes required')
    for path,expected in zip(args.sources,args.expected_sha256):
        if path.name!='evaluation_live.json' or sha(path)!=expected:
            p.error('Declared original held-out source hash required')
    protocol=read(args.training/'protocol.json')
    forbidden={t['task']['company_id'] for t in read(args.prior_capture/'train_live.json')['tasks']}
    forbidden.update(protocol.get('additional_training_company_ids',[]))
    sources=[read(path) for path in args.sources]
    combined,origins=merge(sources,forbidden)
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'evaluation_live.json'
    path.write_text(json.dumps(combined,indent=2))
    records=[]
    for i,source in enumerate(args.sources):
        name=f'source_{i}.json'
        (args.output/name).write_bytes(source.read_bytes())
        records.append(dict(original_path=str(source),copied_filename=name,sha256=sha(source)))
    manifest=dict(sources=records,task_origins=origins,merged_sha256=sha(path),
        no_new_live_run_claimed=True,source_payloads_preserved=True,
        before_distribution=[footprint(source,2) for source in sources],after_distribution=footprint(combined),
        source_runner_sha256=sha(Path(__file__)))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    _,proof=load_merged(args.output,args.training,args.prior_capture)
    print(json.dumps(dict(proof=proof,footprint=manifest['after_distribution'])),flush=True)


if __name__=='__main__':
    main()
