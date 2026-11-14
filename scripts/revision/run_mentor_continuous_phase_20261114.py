"""Matched Full runtime, same arrivals and data, only frozen phase policy differs."""
import argparse
import json
import sys
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['full','nophase'],required=True)
    p.add_argument('--factor',choices=['reasoning','zipf','rw'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from scripts.revision import run_current_dynamic_matrix_20261110 as base
    from agent.runtime.quantized_mask_table import QuantizedSharedBitTable
    model=ROOT/f'revision_experiments/progressive_agent_final_20261109/selected_models/{a.arm}_int16_table.json'
    payload=json.loads(model.read_text())
    base.QuantizedSharedBitTable=lambda ignored:QuantizedSharedBitTable(payload)
    original_run=base.run
    def run(options):
        options.model=model
        return original_run(options)
    base.run=run
    from scripts.revision import run_attribution_repairs_20261113 as study
    # All runtime branches see arm=full; no certificate/admission/priority toggle
    # differs between arms. Only the policy constructor above changes.
    sys.argv=[sys.argv[0],'--kind','continuous','--arm','full','--factor',a.factor,'--output',str(a.output)]
    study.main()
    protocol=dict(arm=a.arm,factor=a.factor,same_full_runtime=True,retrained_at_change=False,
        model=str(model.relative_to(ROOT)),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        no_api_calls=True,scope='Frozen paired Full/NoPhase models;same manager/store/history within each run')
    (a.output/'phase_comparison_protocol.json').write_text(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
