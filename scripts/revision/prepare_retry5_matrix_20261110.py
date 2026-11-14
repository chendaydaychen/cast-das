from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'scripts/revision/launch_regular_worker_curves_20261110.py'
s=p.read_text().replace('agent_regular_worker_curves_20261110','agent_retry5_curves_20261110')
s=s.replace('max_attempts=32,','max_attempts=6,max_retries=5,')
s=s.replace("'--attempts','32'","'--attempts','6'")
s=s.replace('only concurrency grid changed','only retry cap changed from32attempts to6attempts=5retries')
s=s.replace("and r['lock_timeout_s']==120.","and r['lock_timeout_s']==120. and r['max_attempts']==6")
(root/'scripts/revision/launch_retry5_curves_20261110.py').write_text(s)
s=(root/'scripts/analysis/plot_regular_worker_curves_20261110.py').read_text()
s=s.replace('agent_regular_worker_curves_20261110','agent_retry5_curves_20261110').replace('agent_regular_worker_delivery_20261110','agent_retry5_delivery_20261110')
s=s.replace('REGULAR_WORKER_CURVES_20261110.md','RETRY5_WORKER_CURVES_20261110.md')
s=s.replace('最多32次尝试','首次执行后最多5次重试（最多6次尝试）')
s=s.replace("assert not r['grace_leaks']","assert not r['grace_leaks'] and r['max_attempts']==6")
(root/'scripts/analysis/plot_retry5_curves_20261110.py').write_text(s)
print('Prepared isolated retry-five launcher and report')
