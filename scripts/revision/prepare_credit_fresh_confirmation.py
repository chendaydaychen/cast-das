"""Bind new CreditCrew intents to pre-capture frozen models and a replay catalog.

The catalog contains an older source solely to reuse the existing verbatim-union
validator. Only the declared NEW company is executed; selection is verified.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.revision.independent_agent_evaluation import load_independent_capture
from scripts.revision.audit_credit_capture_timing import audit as audit_timing
from scripts.revision.merge_credit_evaluation_captures import merge,footprint,load_merged,sha,read


def verify_fresh_selection(capture,training,canonical,selected_tasks):
    summary=read(capture/'summary.json')
    records=summary['captures']
    if (summary.get('evaluation_only') is not True or len(records)!=1
            or records[0]['split']!='evaluation' or summary.get('phase_independent_planning') is not True):
        raise ValueError('One phase-independent evaluation-only capture required')
    source,proof=load_independent_capture(capture/'evaluation_live.json',
        expected_sha256=records[0]['capture_sha256'],
        manifest_path=capture/'pre_capture_frozen_manifest.json',training_folder=training,
        capture_folder=canonical,allow_complete_aborted_capture=True)
    manifest=read(capture/'pre_capture_frozen_manifest.json')
    if (not manifest.get('phase_independent_planning') or not source.get('phase_independent_planning')
            or selected_tasks!=source['tasks'] or len(selected_tasks)!=2):
        raise ValueError('Selected replay tasks differ from the new phase-independent capture')
    timing=audit_timing(source,read(capture/'evaluation_replay.json'))
    proof.update(all_selected_tasks_from_new_capture=True,phase_independent_planning=True,
        capture_summary_sha256=sha(capture/'summary.json'),
        replay_sha256=sha(capture/'evaluation_replay.json'),
        new_llm_calls=sum(len(t['llm_calls']) for t in source['tasks']),
        new_tool_calls=sum(len(t.get('tool_calls',[])) for t in source['tasks']),
        timing=timing,source_sha256=sha(Path(__file__)),
        caveat='Fresh model-selected intents; frozen replay with constructed short updates, not online replanning')
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','training','canonical','reference-catalog','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--company',type=int,required=True)
    parser.add_argument('--arrival',type=float,required=True)
    args=parser.parse_args()
    source=read(args.capture/'evaluation_live.json')
    proof=verify_fresh_selection(args.capture,args.training,args.canonical,source['tasks'])
    if set(proof['evaluation_company_ids'])!={args.company} or args.arrival<0:
        raise ValueError('Captured company differs from the predeclared replay plan')
    old=read(args.reference_catalog/'manifest.json')['sources'][0]
    old_path=Path(old['original_path'])
    if sha(old_path)!=old['sha256']:
        raise ValueError('Reference catalog source changed')
    paths=[old_path,args.capture/'evaluation_live.json']
    forbidden=set(proof['training_company_ids'])
    combined,origins=merge([read(p) for p in paths],forbidden)
    selected=[t for t in combined['tasks'] if t['task']['company_id']==args.company]
    verify_fresh_selection(args.capture,args.training,args.canonical,selected)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'evaluation_live.json').write_text(json.dumps(combined,indent=2))
    sources=[]
    for i,path in enumerate(paths):
        name=f'source_{i}.json'
        (args.output/name).write_bytes(path.read_bytes())
        sources.append(dict(original_path=str(path.resolve()),copied_filename=name,sha256=sha(path)))
    manifest=dict(sources=sources,task_origins=origins,
        merged_sha256=sha(args.output/'evaluation_live.json'),source_payloads_preserved=True,
        no_new_live_run_claimed=True,source_runner_sha256=sha(Path(__file__)),
        after_distribution=footprint(combined),
        selection='Only the new company is executed; old catalog source is never scheduled')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    load_merged(args.output,args.training,args.canonical)
    declaration=dict(company=args.company,arrival_seconds=[args.arrival],workers=2,
        arms=['full','nophase','linear','hot_always','item_threshold3','silo'],
        fresh_input=proof,training_audit_sha256=sha(args.training/'training_audit.json'),
        catalog_sha256=sha(args.output/'manifest.json'),
        scope='Declared before any CC evaluation; no training or parameter selection on these intents')
    (args.output/'independent_confirmation_plan.json').write_text(json.dumps(declaration,indent=2))
    print(json.dumps(dict(passed=True,new_llm_calls=proof['new_llm_calls'],
        new_tool_calls=proof['new_tool_calls'],companies=proof['evaluation_company_ids'],
        catalog=str(args.output))),flush=True)


if __name__=='__main__':
    main()
