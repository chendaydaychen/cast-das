"""LLM-selected read-only SQL and task-private publication on official inputs.

No preset query/solution or gold enters the model context. SQL is restricted to
SELECT/READ/FUNCTION by SQLite's authorizer. One independently executed task
per public benchmark, with final Full deployment and recorded timings.
"""
import argparse
import csv
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    krama=next(x for x in json.loads((ROOT/'external_sources/KramaBench_metadata_20261109/workload/archeology.json').read_text()) if x['id']=='archeology-easy-3')
    agentic=next(x for x in json.loads((ROOT/'revision_experiments/public_task_selection_20261109/selected_agent_inputs.json').read_text()) if x['task_id']=='social_network_18')
    records=[dict(benchmark='KramaBench',task_id=krama['id'],question=krama['query'],
        input='external_sources/KramaBench_metadata_20261109/data/archeology/input/roman_cities.csv',
        outputs=['answer.csv'],output_contract='answer.csv with one numeric column named answer'),
        dict(benchmark='AgenticDataBench',task_id=agentic['task_id'],question=agentic['question'],
        input='external_sources/AgenticDataBench_inputs_20261109/social_network/Twitter_US_Airline_Sentiment.sqlite',
        outputs=agentic['output_files'],output_contract='Use the output file names and columns in the question')]
    for row in records:row['input_sha256']=sha(ROOT/row['input'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(records,indent=2))
    print('Prepared two public task prompts without answers or reference queries',flush=True)


INSTRUCTIONS='''You are a data-analysis agent. Complete the supplied public task from its input.
Reply with ONE JSON object and no Markdown, using exactly phase, tool, args.
phase must be explore, refine, or commit, and cannot go backwards.
Tools:
inspect: args {}. Loads the registered input and returns table schemas and sample rows.
query: args {"sql": "SELECT ..."}. Runs your read-only SQLite query and shows up to 40 rows.
publish: args {"name": "output.csv", "sql": "SELECT ..."}. Executes your query and publishes all rows as a task-private CSV. Use refine or commit.
finish: args {}. Use commit after all required output files have been published.
Use inspect first. Use actual column names. CSV input is a table named data, with text columns.
SQLite functions include instr, substr, trim, lower, replace, round, printf, CAST, CASE, avg, count and WITH.
Check NULL, empty and nonnumeric sentinel values rather than treating them as zero.
The task question and output contract define your goal. No hidden answer is available.
Only inspect/query/publish/finish tools are available. Do not request shell, network or other files.
SQL queries are written by you, not supplied by a fixed task solution.'''


def execute(spec,output,key,model,mode='live',frozen=None):
    from agent.runtime import AgentTransactionManager
    from agent.runtime.paper_contract import configure_paper_observed_contract
    from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
    from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
    from agent.runtime.single_writer_grace_20261109 import install
    from agent.workloads.credit_reasoner import OpenAIResponsesReasoner,CapturingReasoner,ReplayReasoner
    from agent.workloads.frozen_credit_agent import tracked_agent_wait
    from scripts.revision.progressive_agent_environment_20261109 import ObservedBoundaryHooks
    from scripts.revision.grace_phase_environment_20261109 import MeterPolicy
    from scripts.revision.bridge_profile import ExclusiveClock,instrument_manager,summarize
    from agent.runtime.context import TransactionStatus
    output.mkdir(parents=True,exist_ok=False)
    path=ROOT/spec['input'];assert sha(path)==spec['input_sha256']
    table_path=ROOT/'revision_experiments/progressive_agent_final_20261109/selected_models/full_int16_table.json'
    policy=MeterPolicy(QuantizedSharedBitTable(json.loads(table_path.read_text())))
    manager=AgentTransactionManager(paper_policy=policy,record_traces=False,collect_trajectories=True,
        performance_guards_enabled=False,low_conflict_occ_guard=False)
    configure_paper_observed_contract(manager,explicit_phase_label_ablation=True)
    manager.interceptor.hooks=ObservedBoundaryHooks(ObservedWriteReservationHooks(manager.interceptor.hooks))
    install(manager,4500.,commit_grace_ms=50.)
    manager.paper_deferred_replay_enabled=False;manager.atcc_retry_cache_enabled=False
    manager.priority_manager.config=dataclasses.replace(manager.priority_manager.config,blocked_quantum_ms=10.)
    input_key='input:'+spec['task_id'];manager.register_object(input_key,spec['input_sha256'])
    for name in spec['outputs']:manager.register_object('output:'+spec['task_id']+':'+name,'EMPTY')
    clock=ExclusiveClock();instrument_manager(manager,clock,'paper-atcc')
    original_select=policy.select
    def select(state):
        with clock.scope('lookup'):return original_select(state)
    policy.select=select
    changes=[];transition=manager.transition_atcc_action
    def change(tx,action,**kw):
        before=int(tx.context.action.protected);tick=time.perf_counter_ns()
        try:return transition(tx,action,**kw)
        finally:changes.append(dict(actual=before!=int(tx.context.action.protected),wall_ns=time.perf_counter_ns()-tick))
    manager.transition_atcc_action=change
    delegate=(ReplayReasoner(frozen['llm_calls']) if mode=='replay' else OpenAIResponsesReasoner(model,api_key=key,
        endpoint='http://nas.flytown.cc:35200/v1/chat/completions',allow_insecure_http=True,api_style='chat',
        instructions=INSTRUCTIONS,timeout_s=120))
    reasoner=CapturingReasoner(delegate);events=[];history=[];published={};connection=None
    tx=manager.begin(spec['task_id'],dict(access_set_visibility='online_observed'),strategy='paper-atcc')
    tx.enter_phase('explore');started=time.monotonic_ns();versions={}
    def event(kind,tick,**kwargs):
        events.append(dict(event_id=len(events),kind=kind,start_ns=tick-started,end_ns=time.monotonic_ns()-started,**kwargs))
    def input_read():
        tick=time.monotonic_ns();snapshot=tx.read(input_key);raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==snapshot.value
        versions[input_key]=snapshot.version
        event('db_read',tick,object_id=input_key,storage_id='default',version=snapshot.version)
        return raw
    def load():
        nonlocal connection
        raw=input_read()
        if connection is None:
            if path.suffix=='.sqlite':
                # node1's SQLite build has no deserialize API. Open an exact
                # private copy of the already checked input bytes read-only.
                snapshot=output/'_input_snapshot.sqlite';snapshot.write_bytes(raw)
                connection=sqlite3.connect('file:'+str(snapshot.resolve())+'?mode=ro&immutable=1',uri=True)
            else:
                connection=sqlite3.connect(':memory:')
                rows=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))));columns=rows[0]
                quoted=','.join('"'+c.replace('"','""')+'" TEXT' for c in columns)
                connection.execute('CREATE TABLE data ('+quoted+')')
                connection.executemany('INSERT INTO data VALUES ('+','.join('?' for _ in columns)+')',rows[1:])
                connection.commit()
        return connection
    def query(sql):
        db=load()
        if not isinstance(sql,str) or len(sql)>24000:raise ValueError('SQL string required, at most 24000 characters')
        allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION}
        if hasattr(sqlite3,'SQLITE_RECURSIVE'):allowed.add(sqlite3.SQLITE_RECURSIVE)
        def authorize(action,a,b,c,d):
            if action==sqlite3.SQLITE_FUNCTION and str(b).lower() in {'load_extension','readfile','writefile'}:return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        db.set_authorizer(authorize);deadline=time.monotonic()+15
        db.set_progress_handler(lambda:int(time.monotonic()>deadline),10000)
        try:
            cursor=db.execute(sql)
            if cursor.description is None:raise ValueError('Read-only result query required')
            rows=cursor.fetchmany(20001)
            if len(rows)>20000:raise ValueError('Query output exceeds 20000 rows; aggregate first')
            return [c[0] for c in cursor.description],rows
        finally:db.set_authorizer(None);db.set_progress_handler(None,0)
    finished=False;error=None;previous_phase=0;order={'explore':0,'refine':1,'commit':2}
    try:
        with clock.scope('other',root=True):
            for turn in range(16):
                context=json.dumps(dict(task=spec['question'],output_contract=spec['output_contract'],
                    published=list(published),turns_remaining=16-turn,transcript=history),sort_keys=True,ensure_ascii=False)
                tick=time.monotonic_ns()
                with tracked_agent_wait(tx),clock.scope('reasoning'):
                    answer=reasoner.reason(label='public-next-action',phase=tx.context.phase.value,context=context)
                event('llm_api',tick,phase=tx.context.phase.value,input_read_versions=dict(versions))
                command=None;tick=time.monotonic_ns()
                try:
                    command=json.loads(answer.text)
                    if set(command)!={'phase','tool','args'}:raise ValueError('Use exactly phase/tool/args')
                    phase,tool,args=command['phase'],command['tool'],command['args']
                    if phase not in order or order[phase]<previous_phase:raise ValueError('Invalid or backward phase')
                    if tool in {'publish','finish'} and phase=='explore':raise ValueError('Publish requires refine or commit')
                    if not isinstance(args,dict):raise ValueError('Tool args must be object')
                    tx.enter_phase(phase);previous_phase=order[phase]
                    with clock.scope('external_tool'):
                        if tool=='inspect':
                            db=load();tables=db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                            result=[]
                            for (name,) in tables:
                                escaped='"'+name.replace('"','""')+'"'
                                result.append(dict(table=name,columns=db.execute('PRAGMA table_info('+escaped+')').fetchall(),
                                    sample=db.execute('SELECT * FROM '+escaped+' LIMIT 3').fetchall()))
                        elif tool in {'query','publish'}:
                            columns,rows=query(args['sql']);result=dict(columns=columns,row_count=len(rows),sample=rows[:40])
                            if tool=='publish':
                                name=args['name']
                                if name not in spec['outputs']:raise ValueError('Undeclared task output')
                                file=output/name
                                with file.open('w',encoding='utf-8',newline='') as stream:
                                    writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)
                                write_tick=time.monotonic_ns();output_key='output:'+spec['task_id']+':'+name
                                tx.write(output_key,sha(file),replace_staged=output_key in tx.write_set)
                                event('db_write',write_tick,object_id=output_key,storage_id='default',value=sha(file))
                                published[name]=dict(sha256=sha(file),rows=len(rows),sql=args['sql'])
                        elif tool=='finish':
                            if phase!='commit' or set(published)!=set(spec['outputs']):raise ValueError('Publish required files then finish in commit')
                            result=dict(finished=True);finished=True
                        else:raise ValueError('Unknown tool')
                except (ValueError,KeyError,TypeError,sqlite3.Error) as exc:
                    result=dict(error=type(exc).__name__+': '+str(exc))
                event('agent_tool',tick,action=command,result=result)
                history.append(dict(action_text=answer.text,result=result))
                manager.interceptor.agent_round_finished(tx)
                if finished:break
            if not finished:raise RuntimeError('Task turn budget exhausted')
            commit=tx.commit('paper-atcc')
            if not commit.committed:raise RuntimeError('Task commit failed: '+str(commit.reason))
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        if tx.context.status!=TransactionStatus.COMMITTED:tx.abort_or_observe_wound(error,strategy='paper-atcc')
    elapsed=(time.monotonic_ns()-started)/1e6
    if connection:connection.close()
    committed=tx.context.status==TransactionStatus.COMMITTED
    if committed:
        assert all(manager.store.get('output:'+spec['task_id']+':'+name).value==row['sha256'] for name,row in published.items())
    record=dict(task_id=spec['task_id'],benchmark=spec['benchmark'],mode=mode,model=model,committed=committed,
        finished=finished,error=error,elapsed_ms=elapsed,llm_calls=reasoner.records,llm_call_count=len(reasoner.records),
        policy_calls=len(policy.calls),actual_changes=sum(c['actual'] for c in changes),
        lookup_mean_us=sum(c['wall_ns'] for c in policy.calls)/max(1,len(policy.calls))/1000,
        action_change_ms=sum(c['wall_ns'] for c in changes if c['actual'])/1e6,
        events=events,outputs=published,input_sha256=spec['input_sha256'],model_sha256=sha(table_path),
        timing=summarize(clock,commits=int(committed),tasks=[dict(elapsed_ms=elapsed)]),
        scope='One LLM-selected SQL task per benchmark; official input, bounded tools, Full runtime, private outputs; no reference answer in context')
    for name,data in [('capture',record),('decisions',policy.questions),('lookup_samples',policy.calls),('changes',changes)]:
        (output/(name+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False))
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--key-stdin',action='store_true');p.add_argument('--model',default='gpt-5.6-luna')
    p.add_argument('--replay-source',type=Path)
    p.add_argument('--task-id',choices=['archeology-easy-3','social_network_18'])
    a=p.parse_args()
    if a.prepare:prepare(a.prepare);return
    if not a.manifest or not a.output:p.error('manifest and output required')
    key=sys.stdin.readline().strip() if a.key_stdin else os.environ.get('OPENAI_API_KEY')
    if not a.replay_source and not key:p.error('Live mode requires credential')
    a.output.mkdir(parents=True,exist_ok=False);specs=json.loads(a.manifest.read_text());rows=[]
    if a.task_id:specs=[s for s in specs if s['task_id']==a.task_id]
    for spec in specs:
        frozen=json.loads((a.replay_source/spec['task_id']/'capture.json').read_text()) if a.replay_source else None
        row=execute(spec,a.output/spec['task_id'],key,a.model,mode='replay' if frozen else 'live',frozen=frozen)
        rows.append(dict(task_id=row['task_id'],committed=row['committed'],llm_calls=row['llm_call_count'],
            elapsed_s=row['elapsed_ms']/1000,error=row['error']))
        print(json.dumps(rows[-1]),flush=True)
    (a.output/'completion.json').write_text(json.dumps(rows,indent=2))
    if not all(r['committed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
