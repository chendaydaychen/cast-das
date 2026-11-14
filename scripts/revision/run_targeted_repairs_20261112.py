"""Small, separately reported audits and paired repairs. No API or test fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['dwa', 'continuous'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', default='full')
    p.add_argument('--reservation', action='store_true')
    p.add_argument('--position', default='early')
    p.add_argument('--writer-fraction', type=float, default=.8)
    p.add_argument('--tasks', type=int, default=64)
    p.add_argument('--objects', type=int, default=16)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--period', type=float, default=16)
    p.add_argument('--rate', type=float, default=2)
    p.add_argument('--seed', type=int, default=61011010)
    a = p.parse_args()
    from agent.runtime import AgentTransactionManager
    from agent.runtime.final_attempt_admission_20261111 import install
    old_init = AgentTransactionManager.__init__
    managers = []
    local = threading.local()
    waits = []
    def init(self, *v, **kw):
        old_init(self, *v, **kw)
        install(self, 5)
        managers.append(self)
        old_begin = self.begin
        def begin(task_id, *v, **kw):
            local.task = str(task_id)
            local.retry = int((kw.get('metadata', v[0] if v else {}) or {}).get('retry_count', 0))
            tick = time.monotonic()
            tx = old_begin(task_id, *v, **kw)
            if not str(task_id).startswith('maintenance:'):
                waits.append(dict(kind='begin_admission', task=local.task, retry=local.retry, seconds=time.monotonic()-tick))
            return tx
        self.begin = begin
        old_wait = self.atcc_locks._condition.wait
        def wait(*v, **kw):
            tick = time.monotonic()
            try:
                return old_wait(*v, **kw)
            finally:
                waits.append(dict(kind='lock_condition', task=getattr(local, 'task', None),
                                  retry=getattr(local, 'retry', None), seconds=time.monotonic()-tick))
        self.atcc_locks._condition.wait = wait
    AgentTransactionManager.__init__ = init
    models = ROOT/'revision_experiments/progressive_agent_final_20261109/selected_models/full_int16_table.json'
    reservations = []
    if a.kind == 'dwa':
        from scripts.revision import run_dwa_arrival_separated_20261110 as b
        old_boundary = b.ObservedBoundaryHooks
        if a.reservation:
            # Same repair in both physical-publication arms. Uses only history
            # from issued writes; no preloaded future write set or forced mask.
            from agent.runtime.observed_write_reservation import ObservedWriteReservationHooks
            def boundary(delegate):
                reservation = ObservedWriteReservationHooks(delegate)
                reservations.append(reservation)
                return old_boundary(reservation)
            b.ObservedBoundaryHooks = boundary
        old_config = b.configure_paper_observed_contract
        def configure(m, **kw):
            return old_config(m, explicit_paid_agent_intervals=True, **kw)
        b.configure_paper_observed_contract = configure
        sys.argv = [sys.argv[0], '--mode', 'dwa', '--output', str(a.output), '--model', str(models),
                    '--arm', a.arm, '--tasks', str(a.tasks), '--workers', str(a.workers),
                    '--objects', str(a.objects), '--attempts', '6', '--seed', str(a.seed),
                    '--position', a.position, '--writer-fraction', str(a.writer_fraction)]
        b.main()
    else:
        from scripts.revision import run_current_dynamic_matrix_20261110 as base
        old_plan = base.make_plan
        def plan(options):
            requests = old_plan(options)
            rng = random.Random(options.seed+702)
            for spec in requests:
                overall = [.1, .5, .9][spec['cohort']]
                # Nine independent eligible API operations; op8 must read the
                # key whose observed value the subsequent Refine consumes.
                probability = overall/0.9
                other = rng.sample([f'account:{k}' for k in range(options.objects)
                                    if f'account:{k}' != spec['key']], 8)
                ops = [dict(kind='write' if rng.random()<probability else 'read', key=k) for k in other]
                ops += [dict(kind='read', key=spec['key']),
                        dict(kind='write' if rng.random()<probability else 'read', key=spec['key'])]
                spec.update(operations=ops, is_writer=any(o['kind']=='write' for o in ops),
                            eligible_write_probability=probability, overall_expected_write_fraction=overall,
                            dependency_contract='9 independent eligible read/write choices + op8 dependent read; Refine remains after op8')
            return requests
        # Closure imports the same scenario module; replace only its main so
        # the outer engineering wrapper keeps identical admission/accounting.
        from scripts.revision import run_dependency_continuous_20261110 as scenario
        def scenario_main():
            q=argparse.ArgumentParser()
            q.add_argument('--output', type=Path);q.add_argument('--model', type=Path)
            q.add_argument('--arm');q.add_argument('--factor');q.add_argument('--objects', type=int)
            q.add_argument('--workers', type=int);q.add_argument('--period', type=float)
            q.add_argument('--rate', type=float);q.add_argument('--static-threshold', type=float)
            opts=q.parse_args();opts.mode='continuous';opts.tasks=0;opts.writer_fraction=.2
            opts.position='early';opts.attempts=6;opts.seed=a.seed;opts.dependency_guard=True
            base.make_plan=plan
            base.run(opts)
        scenario.main=scenario_main
        from scripts.revision import run_closure_continuous_20261110 as b
        sys.argv=[sys.argv[0], '--output', str(a.output), '--arm', a.arm, '--factor', 'rw',
                  '--objects', str(a.objects), '--workers', str(a.workers), '--period', str(a.period),
                  '--rate', str(a.rate), '--staged-blind-deferral', '--blocked-quantum-ms', '100']
        b.main()
    final = [dict(m.final_attempt_admission) for m in managers]
    assert all(x['active']==0 and x['waiting_recoveries']==0 for x in final)
    save(a.output/'repair_waits.json', waits)
    save(a.output/'repair_reservations.json', [e for r in reservations for e in r.reservation_events])
    save(a.output/'repair_protocol.json', dict(arguments=vars(a), final_recovery=final,
         model_changed=False, repair_is_runtime_not_new_learned_action=True,
         future_accesses_visible=False, publication_choice='fixed DWA/NoDWA, not a learned selector',
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
