"""Paired attribution follow-up; keep original studies and defaults unchanged."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['dwa', 'continuous'], required=True)
    p.add_argument('--arm', default='full')
    p.add_argument('--position', default='early')
    p.add_argument('--writer-fraction', type=float, default=.8)
    p.add_argument('--factor', default='rw')
    p.add_argument('--fixed-mask', type=int)
    p.add_argument('--no-recovery', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from agent.runtime import final_attempt_admission_20261111 as admission
    if a.no_recovery:
        def no_recovery(manager, final_attempt_index=5):
            manager.final_attempt_admission = dict(active=0, waiting_recoveries=0,
                recovery_admissions=0, normal_admissions=0, disabled=True)
            return manager.final_attempt_admission
        admission.install = no_recovery
    from scripts.revision import run_targeted_repairs_20261112 as target
    if a.kind == 'dwa':
        from scripts.revision import run_dwa_arrival_separated_20261110 as base
        install = base.install_eager_pcc_control
        def matched(manager, **kw):
            return install(manager, blocking_eager_reads=False, **kw)
        base.install_eager_pcc_control = matched
        sys.argv = [sys.argv[0], '--kind', 'dwa', '--arm', a.arm,
            '--position', a.position, '--writer-fraction', str(a.writer_fraction),
            '--reservation', '--output', str(a.output)]
    else:
        from scripts.revision import run_current_dynamic_matrix_20261110 as base
        if a.fixed_mask is not None:
            from agent.runtime.context import LockAction
            class Fixed:
                phase_awareness = 'disabled'
                monotone_actions = True
                def select(self, state):
                    return LockAction.from_encoded(int(state.current_action) | a.fixed_mask)
            base.QuantizedSharedBitTable = lambda payload: Fixed()
        # target implements actual ten-operation RW changes; for other factors,
        # reuse its RW plan without rewriting it, then make RW composition fixed.
        original_run = base.run
        def run(options):
            options.factor = a.factor
            if a.factor != 'rw':
                original_plan = base.make_plan
                def plan(opts):
                    requests = original_plan(opts)
                    # Regenerate only operation kinds with identical deterministic
                    # draws across cohorts; arrival/Zipf/reasoning remain dynamic.
                    import random
                    rng = random.Random(opts.seed + 702)
                    for spec in requests:
                        for j, op in enumerate(spec['operations']):
                            if j != 8:
                                op['kind'] = 'write' if rng.random() < 5/9 else 'read'
                        spec['is_writer'] = any(o['kind'] == 'write' for o in spec['operations'])
                        spec['overall_expected_write_fraction'] = .5
                    return requests
                base.make_plan = plan
            return original_run(options)
        base.run = run
        sys.argv = [sys.argv[0], '--kind', 'continuous', '--arm', 'full',
            '--objects', '512', '--workers', '32', '--period', '16', '--rate', '2',
            '--output', str(a.output)]
    target.main()
    protocol = json.loads((a.output/'repair_protocol.json').read_text())
    protocol.update(attribution_kind=a.kind, fixed_mask=a.fixed_mask,
        same_full_runtime=a.kind == 'continuous', factor=a.factor,
        read_path_matched=a.kind == 'dwa', no_final_recovery=a.no_recovery,
        future_accesses_visible=False)
    (a.output/'attribution_protocol.json').write_text(json.dumps(protocol, indent=2))
    result = json.loads((a.output/'result.json').read_text())
    assert result['values_correct'] and not result['unexpected_errors'], result
    assert result['physical_pending'] == 0 and not result['physical_poisoned']
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
