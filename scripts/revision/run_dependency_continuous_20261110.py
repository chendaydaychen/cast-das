"""Continuous agent YCSB: actual read-reason-update dependence, frozen policies."""
import argparse
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.revision import run_current_dynamic_matrix_20261110 as base

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--arm',choices=['full','static','silo'],required=True)
    p.add_argument('--factor',choices=['reasoning','zipf','rw'],required=True)
    p.add_argument('--workers',type=int,default=100);p.add_argument('--objects',type=int,default=32)
    p.add_argument('--period',type=float,default=20);p.add_argument('--rate',type=float,default=10)
    p.add_argument('--static-threshold',type=float,default=.2)
    p.add_argument('--all-operation-rw',action='store_true',help='Exactly 1/5/9 writes across ten mutable operations; op8 is a dependency-bearing read')
    p.add_argument('--task-local-workspace',action='store_true',help='First eight operations use per-task mutable workspace; last read/write pair shares an account')
    args=p.parse_args()
    args.mode='continuous';args.tasks=0;args.writer_fraction=.2;args.position='early'
    args.attempts=32;args.seed=61011041;args.dependency_guard=True
    original=base.make_plan
    def make_plan(a):
        requests=original(a);rng=random.Random(a.seed+1)
        for spec in requests:
            probability=[.1,.5,.9][spec['cohort']] if a.factor=='rw' else .5
            if args.all_operation_rw:
                count=round(10*probability)
                other_keys=rng.sample([f'account:{k}' for k in range(a.objects) if f'account:{k}'!=spec['key']],8)
                if args.task_local_workspace:other_keys=[f'workspace:{spec["index"]}:{j}' for j in range(8)]
                write_indices=set(rng.sample(range(8),count-1))|{9}
                spec['operations']=[dict(kind='write' if j in write_indices else 'read',key=other_keys[j] if j<8 else spec['key']) for j in range(10)]
                spec['is_writer']=True
                spec['eligible_write_probability']=probability
                spec['dependency_contract']='10 mutable API operations;exactly 1/5/9 writes;op8 reads critical key;Refine consumes observed value;op9 updates same key'
                continue
            write=rng.random()<probability
            spec['operations'][8]=dict(kind='read',key=spec['key'])
            spec['operations'][9]=dict(kind='write' if write else 'read',key=spec['key'])
            spec['is_writer']=write
            spec['eligible_write_probability']=probability
            spec['dependency_contract']='op8 reads key;reasoning consumes value;op9 updates same key or rereads;10 total API ops'
        return requests
    base.make_plan=make_plan
    base.run(args)
    protocol=base.read(args.output/'protocol.json')
    protocol.update(continuous_scope='8catalog reads +read(key)+reasoning+Bernoulli update(key)/read(key);10 actual API operations;RW expected API write fractions1/5/9percent',
                    plan_source_sha256=base.sha(Path(__file__)),dependency_assertion='No dependence-bearing reasoning is moved before critical read',
                    comparison_scope='new controlled read-reason-update workload;not a replacement for previous continuous data')
    if args.all_operation_rw:
        protocol.update(continuous_scope='10 mutable API operations;1/5/9 writes per task by cohort;op8 dependent read then Refine then op9 RMW',
                        all_operation_rw=True,planned_api_write_fractions=[.1,.5,.9])
        if args.task_local_workspace:
            protocol.update(workspace_scope='Eight per-task mutable workspace objects plus shared read/RMW pair;9/5/1 reads and1/5/9 writes;shared write count stays one per task',
                            task_local_workspace=True)
    base.save(args.output/'protocol.json',protocol)
if __name__=='__main__':main()
