"""Summarize recorded actions; never change measured performance rows."""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'revision_experiments/mentor_additions_20261114'
OUT = ROOT / 'revision_experiments/mentor_additions_delivery_20261114'
rows = []
for folder in sorted(RAW.glob('ablation_*')):
    if not folder.is_dir():
        continue
    result = json.loads((folder / 'result.json').read_text())
    decisions = json.loads((folder / 'decisions.json').read_text())
    counts = collections.Counter((q['state']['phase'], q['action']) for q in decisions)
    for (phase, mask), count in sorted(counts.items()):
        rows.append(dict(workload=result['workload'], conflict=result['conflict'],
                         workers=result['workers'], arm=result['arm'],
                         recorded_phase=phase, action_mask=mask, decisions=count))
with (OUT / 'action_by_phase.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps(dict(cells=72, action_groups=len(rows), output='action_by_phase.csv')))
