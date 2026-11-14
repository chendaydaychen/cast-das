"""Plot every fresh observation on the regular worker grid, without smoothing."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,ScalarFormatter,NullFormatter

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'revision_experiments/agent_regular_worker_curves_20261110'
OUT=ROOT/'revision_experiments/agent_regular_worker_delivery_20261110'
WORKERS=[40,80,120,160,200]
LABELS={'paper-atcc':'Aegis Full','occ':'OCC','2pl-nowait':'2PL','mvcc':'MVCC','silo':'Silo','tictoc':'TicToc','bamboo':'Bamboo','polaris':'Polaris','mocc':'MOCC','hekaton':'Hekaton','polyjuice':'Polyjuice'}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)

def main():
    assert read(BASE/'completion.json')['complete'],'Matrix not complete'
    protocol=read(BASE/'protocol.json');rows=[]
    assert protocol['workers']==WORKERS
    for w in ['ycsb','tpcc']:
        for n in WORKERS:
            for cc in LABELS:
                p=BASE/'runs'/f'{w}_{n}_{cc}'/'result.json';r=read(p)
                assert r['values_correct'] and not r['unexpected_errors'] and r['workers']==n and r['cc']==cc
                assert r['input_sha256']==protocol['input_sha256'][w] and r['submitted']==240
                assert not r['grace_leaks']
                if cc=='paper-atcc':assert r['model_sha256']==protocol['model_sha256']
                rows.append(dict(workload=w,workers=n,strategy=LABELS[cc],cc=cc,tps=r['agent_tps'],p99_s=r['retry_inclusive_p99_s'],
                    completion_rate=r['completion_rate'],submitted=r['submitted'],completed=r['completed'],aborts=r['aborts'],elapsed_s=r['elapsed_s'],
                    arrival_p99_s=r['arrival_to_finish_p99_s'],queue_mean_s=r['queue_mean_s'],policy_calls=r['policy_calls'],actual_changes=r['actual_changes'],
                    input_sha256=r['input_sha256'],model_sha256=r['model_sha256'],source=str(p.relative_to(ROOT))))
    assert len(rows)==110
    OUT.mkdir(exist_ok=True)
    write('all_cells_long.csv',rows)
    for w in ['ycsb','tpcc']:
        for metric in ['tps','p99_s','completion_rate','aborts','arrival_p99_s']:
            wide=[]
            for n in WORKERS:
                values={r['strategy']:r[metric] for r in rows if r['workload']==w and r['workers']==n}
                wide.append(dict(workers=n,**values))
            write(f'{w}_{metric}_wide.csv',wide)
    gains=[];best=[]
    for w in ['ycsb','tpcc']:
        for n in WORKERS:
            rr=[r for r in rows if r['workload']==w and r['workers']==n];full=next(r for r in rr if r['cc']=='paper-atcc')
            controls=[r for r in rr if r['cc']!='paper-atcc'];b=max(controls,key=lambda r:r['tps'])
            for c in controls:
                gains.append(dict(workload=w,workers=n,baseline=c['strategy'],tps_ratio=full['tps']/c['tps'],
                    p99_reduction_fraction=1-full['p99_s']/c['p99_s'],full_completion=full['completion_rate'],baseline_completion=c['completion_rate']))
            best.append(dict(workload=w,workers=n,full_tps=full['tps'],full_p99_s=full['p99_s'],full_completion=full['completion_rate'],
                strongest_tps_baseline=b['strategy'],baseline_tps=b['tps'],baseline_p99_s=b['p99_s'],baseline_completion=b['completion_rate'],
                tps_gain_percent=100*(full['tps']/b['tps']-1),p99_reduction_percent=100*(1-full['p99_s']/b['p99_s'])))
    write('all_pairwise_gains.csv',gains);write('full_vs_strongest_tps.csv',best)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#c32232','#4778ab','#e09735','#4c956c','#8c6bb1','#7d6257','#c36b9c','#6b7b83','#8c993f','#2d9aab','#4b4b4b']
    markers=['o','s','^','v','D','P','X','<','>','h','*']
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.0),sharex=True)
    handles=[]
    for i,w in enumerate(['ycsb','tpcc']):
        for j,(metric,label) in enumerate([('tps','Agent throughput (TPS)'),('p99_s','Retry-inclusive P99 (s)')]):
            ax=axes[i,j]
            for k,(cc,name) in enumerate(LABELS.items()):
                rr=[next(r for r in rows if r['workload']==w and r['workers']==n and r['cc']==cc) for n in WORKERS]
                line,=ax.plot(WORKERS,[r[metric] for r in rr],color=colors[k],marker=markers[k],
                    linestyle='-' if cc=='paper-atcc' else ['--','-.',':'][k%3],linewidth=2.8 if cc=='paper-atcc' else 1.35,
                    markersize=6 if cc=='paper-atcc' else 4.8,label=name,zorder=20 if cc=='paper-atcc' else 3)
                if i==0 and j==0:handles.append(line)
            ax.set_title('Agent-YCSB (64 shared accounts)' if w=='ycsb' else 'Agent-TPC-C (128 districts/customers)')
            ax.set_ylabel(label);ax.set_xticks(WORKERS);ax.set_xlim(35,205);ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.18);ax.set_xlabel('Concurrent workers')
    fig.legend(handles=handles,loc='upper center',ncol=6,bbox_to_anchor=(.5,1.02),frameon=False,columnspacing=1.4)
    fig.tight_layout(rect=[0,0,1,.92])
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('main_tps_p99_linear.'+ext),dpi=200,bbox_inches='tight')
    for i,w in enumerate(['ycsb','tpcc']):
        ax=axes[i,1];values=[r['p99_s'] for r in rows if r['workload']==w]
        low=min(values)*.85;high=max(values)*1.15
        ax.set_yscale('log');ax.set_ylim(low,high)
        ax.yaxis.set_major_locator(FixedLocator([v for v in [5,10,15,20,30,50,100,200,300,500] if low<=v<=high]))
        ax.yaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel('Retry-inclusive P99 (s, log scale)')
    save(fig,'main_tps_p99_curves')
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    for ax,w in zip(axes,['ycsb','tpcc']):
        for k,(cc,name) in enumerate(LABELS.items()):
            rr=[next(r for r in rows if r['workload']==w and r['workers']==n and r['cc']==cc) for n in WORKERS]
            ax.plot(WORKERS,[r['completion_rate']*100 for r in rr],marker=markers[k],color=colors[k],label=name,lw=2.5 if cc=='paper-atcc' else 1)
        ax.set_title(w.upper());ax.set_xticks(WORKERS);ax.set_xlabel('Concurrent workers');ax.set_ylabel('Task completion (%)');ax.set_ylim(0,105)
    axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8);fig.tight_layout();save(fig,'completion_curves')
    shutil.copy2(BASE/'protocol.json',OUT/'protocol.json')
    text=['# 40/80/120/160/200 workers 新测主曲线','',
          '两种负载×五档并发×11种CC=110个新测单元，未拼接旧40/200点。冻结任务、时延、模型及运行协议与前版一致，仅改变并发采样网格。YCSB固定64共享账户，TPC-C固定128地区/客户，各240任务。每点2物理核/4SMT，最多32次尝试，锁等待120s，Full保留4.5s单写宽限及50ms active-commit宽限。',
          '', 'TPS=成功agent任务数/整批wall（含排空）；P99从首次执行到最终成功，含重试，初始队列另报；同时提供完成率。使用同runtime适配CC，不声称原生产品排名或标准SQL TPC-C。',
          '', '图中的线段只连接实测点，没有插值、滤波或修改数值。主图P99采用对数秒轴以容纳大尾延迟对照，另附完全相同数据的线性轴版本main_tps_p99_linear。等间距横轴便于观察高并发趋势，不保证测量值单调；240任务有限批次的启动、排空、冲突时序和热点分配仍可能产生局部起伏。',
          '', '|负载|workers|Full TPS|Full P99 s|吞吐最强对照|对照TPS|吞吐提升|P99降低|Full完成率|','|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in best:text.append(f"|{r['workload']}|{r['workers']}|{r['full_tps']:.3f}|{r['full_p99_s']:.3f}|{r['strongest_tps_baseline']}|{r['baseline_tps']:.3f}|{r['tps_gain_percent']:.1f}%|{r['p99_reduction_percent']:.1f}%|{r['full_completion']:.1%}|")
    text+=['','完整数值见all_cells_long.csv及各指标宽表；all_pairwise_gains.csv给出Full相对每个CC的结果。原始输入、逐单元协议与结果存于raw_evidence.zip。旧曲线仍保留于agent_scaling_final_delivery_20261109。']
    (OUT/'README_ZH.md').write_text('\n'.join(text)+'\n')
    (ROOT/'docs/REGULAR_WORKER_CURVES_20261110.md').write_text('\n'.join(text)+'\n')
    audit=dict(cells=len(rows),all_values_correct=True,unexpected_errors=0,full_completion_all=all(r['completion_rate']==1 for r in rows if r['cc']=='paper-atcc'),
        all_strategy_completion_min=min(r['completion_rate'] for r in rows),grid=WORKERS,no_smoothing=True,all_points_new=True,
        input_hashes=protocol['input_sha256'],model_sha256=protocol['model_sha256'])
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    with zipfile.ZipFile(OUT/'raw_evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(BASE.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        for p in [ROOT/'scripts/revision/run_agent_main_matrix_20261109.py',ROOT/'scripts/revision/launch_regular_worker_curves_20261110.py',Path(__file__)]:
            z.write(p,str(p.relative_to(ROOT)))
    manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.csv']
    write('manifest.csv',manifest)
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,p.name)
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(dict(audit=audit,best=best),indent=2))

if __name__=='__main__':main()
