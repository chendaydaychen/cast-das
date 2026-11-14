import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--factor',required=True)
 p.add_argument('--prospective-reads',action='store_true');p.add_argument('--cooldown',type=float,default=60)
 p.add_argument('--reservation-probability',type=float,default=.5);p.add_argument('--seed',type=int)
 p.add_argument('--optimistic-mixed-reads',action='store_true');p.add_argument('--all-operation-rw',action='store_true')
 p.add_argument('--task-local-workspace',action='store_true');p.add_argument('--staged-blind-deferral',action='store_true')
 p.add_argument('--blocked-quantum-ms',type=float,default=10.);p.add_argument('--disable-priority',action='store_true')
 p.add_argument('--max-active-tasks',type=int,default=0,help='Explicit bounded admission; waiting remains in arrival latency, does not increase retry budget')
 p.add_argument('--final-attempt-admission',action='store_true')
 p.add_argument('--objects',type=int,default=64);p.add_argument('--workers',type=int,default=100)
 p.add_argument('--period',type=float,default=15);p.add_argument('--rate',type=float,default=8);a=p.parse_args()
 from scripts.revision import run_current_dynamic_matrix_20261110 as b
 from scripts.revision import run_dependency_continuous_20261110 as scenario
 if a.max_active_tasks:
  if a.max_active_tasks<1:raise ValueError('Admission bound must be positive')
  from threading import BoundedSemaphore
  original_executor=b.ThreadPoolExecutor
  admission_gate=BoundedSemaphore(a.max_active_tasks)
  class BoundedAdmissionExecutor(original_executor):
   def submit(self,fn,*args,**kwargs):
    def admitted():
     with admission_gate:return fn(*args,**kwargs)
    return super().submit(admitted)
  b.ThreadPoolExecutor=BoundedAdmissionExecutor
 # Provisioning an immutable catalog is not evidence of a mutable business write.
 reservation=b.ObservedWriteReservationHooks
 class History(dict):
  def __setitem__(self,key,value):
   if str(key).startswith('catalog:'):return
   super().__setitem__(key,value)
  def move_to_end(self,*v,**kw):return None
  def popitem(self,last=False):return super().popitem()
 class BusinessHistory(reservation):
  def __init__(self,*v,**kw):super().__init__(*v,**kw);self.observed_writes=History()
 b.ObservedWriteReservationHooks=BusinessHistory
 from agent.runtime.observed_prefix_replay_20261110 import ObservedPrefixReplayClock
 from agent.runtime.read_only_snapshot_certificate_20261110 import install as certificate
 from agent.runtime.congested_read_admission_20261110 import install as admission
 from agent.runtime import observed_dependency_guard_20261110 as dependency
 original_guard=dependency.install
 def guard(manager,**kwargs):
  return original_guard(manager,min_reservation_write_probability=a.reservation_probability,
                        optimistic_mixed_reads=a.optimistic_mixed_reads,
                        defer_staged_blind_writes=a.staged_blind_deferral,**kwargs)
 dependency.install=guard
 replay=ObservedPrefixReplayClock(b.time);b.time=replay;old=b.configure_paper_observed_contract;refs={}
 def configure(m,**kw):
  result=old(m,explicit_paid_agent_intervals=True,**kw);replay.bind(m);refs['manager']=m
  if a.final_attempt_admission:
   from agent.runtime.final_attempt_admission_20261111 import install
   install(m,final_attempt_index=5)
  if a.arm=='full' and a.prospective_reads:m.prospective_category_read_protection=True
  return result
 physical_install=b.install_eager_pcc_control
 def physical(m,**kw):
  result=physical_install(m,**kw)
  if a.arm=='full':
   import dataclasses
   m.priority_manager.config=dataclasses.replace(m.priority_manager.config,blocked_quantum_ms=a.blocked_quantum_ms)
   if a.disable_priority:m.priority_enabled=False;m.atcc_locks._priority_callback=None
   certificate(m);m.replay_aware_admission_enabled=True
  return result
 b.install_eager_pcc_control=physical
 boundary=b.ObservedBoundaryHooks
 def boundary_hooks(delegate):
  result=boundary(delegate)
  if a.arm=='full':
   m=delegate.manager;m.interceptor.hooks=result
   admission(m,queue_threshold=1,cooldown_s=a.cooldown,min_read_share=.015)
   return m.interceptor.hooks
  return result
 b.ObservedBoundaryHooks=boundary_hooks
 b.configure_paper_observed_contract=configure;run=b.run
 def execute(options):
  options.attempts=6
  if a.seed is not None:options.seed=a.seed
  return run(options)
 b.run=execute
 model=ROOT/'revision_experiments/progressive_agent_final_20261109/selected_models/full_int16_table.json'
 threshold=json.loads((ROOT/'revision_experiments/revision_supplement_20261110/protocol.json').read_text())['static_thresholds']['ycsb']
 sys.argv=[sys.argv[0],'--output',str(a.output),'--model',str(model),'--arm',a.arm,'--factor',a.factor,'--objects',str(a.objects),
  '--workers',str(a.workers),'--period',str(a.period),'--rate',str(a.rate),'--static-threshold',str(threshold)]
 if a.all_operation_rw:sys.argv.append('--all-operation-rw')
 if a.task_local_workspace:sys.argv.append('--task-local-workspace')
 scenario.main();r=b.read(a.output/'result.json');r['prefix_replay_metrics']=replay.metrics;r['max_retries']=5
 r['snapshot_certificate_metrics']=getattr(refs['manager'],'read_only_snapshot_certificate_metrics',{})
 r['congestion_metrics']=refs['manager'].__dict__.get('congested_read_admission',{}).get('metrics',{})
 r['final_attempt_admission']=getattr(refs['manager'],'final_attempt_admission',{})
 if a.final_attempt_admission:assert r['final_attempt_admission']['active']==0
 b.save(a.output/'result.json',r);b.save(a.output/'prefix_replay_events.json',replay.events)
 protocol=b.read(a.output/'protocol.json');protocol.update(shared_prefix_replay=True,max_retries=5,explicit_paid_intervals=True,
  immutable_catalog_provisioning_not_mutable_history=True,
  prospective_reads=a.prospective_reads,admission_cooldown_s=a.cooldown,
  reservation_write_probability=a.reservation_probability,
  optimistic_mixed_reads=a.optimistic_mixed_reads,
  staged_blind_deferral=a.staged_blind_deferral,
  blocked_quantum_ms=a.blocked_quantum_ms,disable_priority=a.disable_priority,
  max_active_tasks=a.max_active_tasks,admission_queue_included_in_arrival_latency=True,
  final_attempt_admission=a.final_attempt_admission,
  deployment='Full extra certificate/admission;all arms version-validated prefix replay;frozen model;Refine never replayed')
 b.save(a.output/'protocol.json',protocol)
if __name__=='__main__':main()
