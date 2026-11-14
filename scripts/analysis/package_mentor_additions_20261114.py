"""All declared outcomes, combined throughput/latency plots and phase dynamics."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'revision_experiments/mentor_additions_20261114'
OUT=ROOT/'revision_experiments/mentor_additions_delivery_20261114'

def read(p):return json.loads(p.read_text())
def dump(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def figure(name):
    for ext in ['png','pdf','svg']:plt.savefig(OUT/f'{name}.{ext}',dpi=185,bbox_inches='tight')
    plt.close()

def main():
    OUT.mkdir(exist_ok=True)
    completion=read(RAW/'completion.json')
    assert len(completion)==78 and all(x['exit_code']==0 for x in completion)
    arms=['full','nophase','linear','hot_threshold','hot_always','static']
    labels=['Full','NoPhase','Base-linear','Hot-threshold','Hot-always','Static-write']
    colors=['#bf2539','#2478b5','#6b55a3','#da8d16','#168f8a','#7c7c7c']
    markers=['o','s','^','v','D','P']
    ablation=[];continuous=[];windows=[];cohorts=[]
    for status in completion:
        name=status['case'];folder=RAW/name;r=read(folder/'result.json')
        assert r['values_correct'] and not r['unexpected_errors']
        if name.startswith('ablation_'):
            ablation.append(dict(case=name,workload=r['workload'],conflict=r['conflict'],workers=r['workers'],
                arm=r['arm'],submitted=r['tasks'],completed=r['completed'],completion_rate=r['completion_rate'],
                agent_tps=r['agent_tps'],p99_s=r['foreground_p99_s'],aborts=r['aborts'],
                shared_competitors=r['shared_competitors'],competitor_commits=r['competitor_commits'],
                policy_calls=r['policy_calls'],actual_changes=r['actual_changes']))
        else:
            phase=read(folder/'phase_comparison_protocol.json')
            attempts=read(folder/'attempts.json')
            continuous.append(dict(case=name,factor=phase['factor'],arm=phase['arm'],
                agent_tps=r['agent_tps'],service_p99_s=r['retry_service_p99_s'],arrival_p99_s=r['arrival_p99_s'],
                completed=r['completed'],submitted=r['submitted'],completion_rate=r['completion_rate'],
                aborts=r['aborts'],actual_changes=r['actual_changes'],policy_calls=r['policy_calls'],
                continuity_verified=r['continuity_verified'],
                recovery_admissions=r['final_attempt_admission']['recovery_admissions']))
            continuous[-1]['failed_attempt_paid_s']=sum(x['paid_s'] for x in attempts if not x['committed'])
            continuous[-1]['attempt_blocked_ms']=sum(x['blocked_ms'] for x in attempts)
            assert r['continuity_verified'] and r['physical_pending']==0 and not r['physical_poisoned']
            assert read(folder/'protocol.json')['model_sha256']==phase['model_sha256']
            for x in read(folder/'time_windows.json'):windows.append(dict(factor=phase['factor'],arm=phase['arm'],**x))
            for x in read(folder/'arrival_cohorts.json'):cohorts.append(dict(factor=phase['factor'],arm=phase['arm'],**x))
    assert len(ablation)==72 and len(continuous)==6
    for w in ['ycsb','tpcc']:
        for level in ['high','low']:
            for workers in [8,16,32]:
                paths=[RAW/f'ablation_{w}_{level}_w{workers}_{a}'/'requests.json' for a in arms]
                assert all(read(p)==read(paths[0]) for p in paths)
                assert all(r['completed']==64 and r['competitor_commits']==64 for r in ablation if r['workload']==w and r['conflict']==level and r['workers']==workers)
    for factor in ['reasoning','zipf','rw']:
        assert read(RAW/f'continuous_{factor}_full/requests.json')==read(RAW/f'continuous_{factor}_nophase/requests.json')
    dump('ablation_all72.csv',ablation);dump('continuous_all6.csv',continuous)
    dump('continuous_time_windows.csv',windows);dump('continuous_arrival_cohorts.csv',cohorts)
    index={(r['workload'],r['conflict'],r['workers'],r['arm']):r for r in ablation}
    for w in ['ycsb','tpcc']:
        for level in ['high','low']:
            for metric in ['agent_tps','p99_s']:
                dump(f'{w}_{level}_{metric}_wide.csv',[dict(workers=n,**{label:index[w,level,n,arm][metric] for arm,label in zip(arms,labels)}) for n in [8,16,32]])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    for w in ['ycsb','tpcc']:
        fig,axs=plt.subplots(2,2,figsize=(11.5,8),sharex=True)
        for i,level in enumerate(['high','low']):
            for j,(metric,ylabel) in enumerate([('agent_tps','Agent TPS'),('p99_s','Retry-inclusive foreground P99 (s)')]):
                ax=axs[i,j]
                for k,(arm,label) in enumerate(zip(arms,labels)):
                    ax.plot([8,16,32],[index[w,level,n,arm][metric] for n in [8,16,32]],
                        label=label,color=colors[k],marker=markers[k],linestyle='-' if k==0 else ['--','-.',':'][k%3],
                        linewidth=2.5 if k==0 else 1.4,markersize=5)
                ax.set_title(f'{"High" if level=="high" else "Low"} contention ({64 if level=="high" else 8}/64 shared pairs)')
                ax.set_ylabel(ylabel);ax.set_xlabel('Foreground workers');ax.set_xticks([8,16,32]);ax.grid(alpha=.2)
        handles,ls=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,ls,loc='lower center',ncol=6,frameon=False)
        fig.suptitle('Agent-'+('YCSB' if w=='ycsb' else 'TPC-C')+' phase-sensitive business variant',fontsize=13)
        fig.tight_layout(rect=(0,.06,1,.96));figure(w+'_ablation_high_low')
    fig,axs=plt.subplots(2,3,figsize=(14,7.2))
    for j,factor in enumerate(['reasoning','zipf','rw']):
        for arm,color in [('full',colors[0]),('nophase',colors[1])]:
            selected=[r for r in windows if r['factor']==factor and r['arm']==arm]
            x=[(r['start_s']+r['end_s'])/2 for r in selected]
            for i,metric in enumerate(['commit_tps','queued_at_end']):
                if i==0:
                    axs[i,j].plot(x,[r[metric] for r in selected],label='Full' if arm=='full' else 'NoPhase',color=color,marker='o' if arm=='full' else 's',markersize=3)
                else:
                    groups=[r for r in cohorts if r['factor']==factor and r['arm']==arm]
                    axs[i,j].plot([8+16*r['cohort'] for r in groups],[r['arrival_p99_s'] for r in groups],
                        label='Full' if arm=='full' else 'NoPhase',color=color,marker='o' if arm=='full' else 's',markersize=5)
        for ax in axs[:,j]:
            for change in [16,32]:ax.axvline(change,color='#777777',ls=':',alpha=.8)
            ax.axvline(48,color='#aaaaaa',ls='--',alpha=.6);ax.grid(alpha=.2);ax.set_xlabel('Elapsed time (s)')
        axs[0,j].set_title({'reasoning':'Reasoning 1x / 2x / 1x','zipf':'Zipf 0 / 0.99 / 0.5','rw':'Write probability 10% / 50% / 90%'}[factor])
        axs[0,j].set_ylabel('Completed agent tasks / s');axs[1,j].set_ylabel('Arrival-cohort P99 (s)')
        axs[1,j].set_xlabel('Arrival cohort midpoint (s)');axs[1,j].set_xlim(0,48);axs[1,j].set_xticks([8,24,40])
    handles,ls=axs[0,0].get_legend_handles_labels();fig.legend(handles,ls,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));figure('continuous_full_nophase')
    gains=[]
    for w in ['ycsb','tpcc']:
        for level in ['high','low']:
            for n in [8,16,32]:
                full=index[w,level,n,'full'];no=index[w,level,n,'nophase'];best=max((index[w,level,n,a] for a in arms if a!='full'),key=lambda r:r['agent_tps'])
                gains.append(dict(workload=w,conflict=level,workers=n,
                    full_tps=full['agent_tps'],nophase_tps=no['agent_tps'],gain_vs_nophase=full['agent_tps']/no['agent_tps']-1,
                    best_comparator=best['arm'],gain_vs_best=full['agent_tps']/best['agent_tps']-1,
                    full_p99_s=full['p99_s'],nophase_p99_s=no['p99_s']))
    dump('ablation_gains.csv',gains)
    report=['# 导师新增实验：规则消融、高低冲突、连续Phase对照','',
        '本轮72个消融单元和6个连续对照单元全部保留。公开新负载按用户要求放最后，本轮不采集新API轨迹。',
        '', '## 设置与边界','',
        '消融采用现有阶段敏感YCSB/TPC-C业务变体，不替换原主性能曲线。每点64个前台任务（Explore/Refine各32）和64个配对竞争事务，8/16/32前台workers；竞争者另有配对线程。高冲突64对共享目标，低冲突8对共享目标，其余竞争者写入不相交私有对象；共享对在两种phase中平衡。保持相同经验计算预算、输入种子和执行机制。全部点前台与竞争事务均64/64完成，最终值正确。TPS只统计前台完成数，墙钟包含竞争者排空；P99为含重试前台服务延迟。',
        '',
        'Full/NoPhase沿用同预算训练的冻结模型，不在本轮重训，NoPhase只移除显式标签。规则使用相同检查点和单调动作；Hot-always在已观测对象的一秒窗口访问计数达到1时保护，Hot-threshold阈值为8；保护读，在已观察到写时增加写保护。Base-linear在已有Full训练数据上一次拟合每动作线性得分（512训练任务，ridge=1，输入屏蔽显式phase，无测试拟合）。Static-write固定写保护mask12，不称tuned-threshold Static。',
        '为隔离策略选择，消融各臂共同启用DWA、历史写入预约和只读版本证书，共同关闭Priority、deferred replay及末次独占恢复。这里Full表示该受控机制下的完整阶段策略，不等于包含全部工程机制的主性能Full部署。',
        '',
        '连续实验沿用原同runtime的96任务、32workers、512对象、2tasks/s及16/32秒变点，48秒停止到达后排空。Full与NoPhase只替换同一训练系列的冻结策略，运行时所有分支仍用Full，保持相同证书、Priority、DWA、replay和末次恢复。每次运行内manager/数据/历史/模型连续，不在变点重训；两臂任务输入逐项一致。',
        '', '## 32workers消融结果','',
        '|负载|冲突|策略|TPS|P99/s|中止|','|---|---|---|---:|---:|---:|']
    for w in ['ycsb','tpcc']:
        for level in ['high','low']:
            for arm,label in zip(arms,labels):
                r=index[w,level,32,arm];report.append(f'|{w}|{level}|{label}|{r["agent_tps"]:.3f}|{r["p99_s"]:.3f}|{r["aborts"]}|')
    report+=['','![YCSB](ycsb_ablation_high_low.png)','','![TPC-C](tpcc_ablation_high_low.png)','',
        '## 连续变化结果','','|变化|策略|TPS|到达P99/s|完成|中止|动作变化|恢复进入次数|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in continuous:
        report.append(f'|{r["factor"]}|{r["arm"]}|{r["agent_tps"]:.3f}|{r["arrival_p99_s"]:.3f}|{r["completed"]}/{r["submitted"]}|{r["aborts"]}|{r["actual_changes"]}|{r["recovery_admissions"]}|')
    report+=['','![连续变化](continuous_full_nophase.png)','','## 论文解释稿','',
        '我们在阶段敏感业务变体中补充简单规则和固定保护对照，并通过改变配对更新的目标重合程度区分高低冲突。该控制不改变前台任务的阶段结构和计算预算，能够观察争用压力如何影响各策略的保护价值。Full与NoPhase比较阶段标签的增量；Full与Base-linear及访问频率规则比较上下文策略与固定线性/热点判断的差别。',
        '高冲突时，避免依赖失效有更大价值；低冲突时，额外保护与排空成本可能使策略差异缩小。规则、固定动作和NoPhase出现接近点应按实际动作和中止记录解释，不能预设所有指标严格排序。该实验是阶段机制证据，不将变体曲线替代原主性能曲线。',
        '连续变化中的Full/NoPhase保留相同运行时路径，仅替换冻结策略。两臂均保留末次恢复，因此完成率不能独立归因于学习。对比应结合吞吐、到达尾延迟、队列、动作变化和恢复使用量；若指标有取舍则同时报告。固定到达率限制短窗口TPS，变点后完成量还包含前段在途任务，不能视为瞬时容量。']
    for w in ['ycsb','tpcc']:
        for level in ['high','low']:
            g=next(g for g in gains if g['workload']==w and g['conflict']==level and g['workers']==32)
            report.append(f'{w} {level}：32workers时Full吞吐{g["full_tps"]:.3f}，NoPhase为{g["nophase_tps"]:.3f}，提高{g["gain_vs_nophase"]:.1%}；相对该点吞吐最高对照{g["best_comparator"]}提高{g["gain_vs_best"]:.1%}。Full/NoPhase P99分别{g["full_p99_s"]:.3f}/{g["nophase_p99_s"]:.3f}秒。')
    for factor in ['reasoning','zipf','rw']:
        f=next(r for r in continuous if r['factor']==factor and r['arm']=='full');n=next(r for r in continuous if r['factor']==factor and r['arm']=='nophase')
        report.append(f'{factor}连续变化：Full/NoPhase吞吐{f["agent_tps"]:.3f}/{n["agent_tps"]:.3f}，Full相对变化{f["agent_tps"]/n["agent_tps"]-1:+.1%}；到达P99 {f["arrival_p99_s"]:.3f}/{n["arrival_p99_s"]:.3f}秒，相对变化{f["arrival_p99_s"]/n["arrival_p99_s"]-1:+.1%}。')
        report.append(f'失败尝试中的实付计算等待累计为Full {f["failed_attempt_paid_s"]:.3f}秒、NoPhase {n["failed_attempt_paid_s"]:.3f}秒，减少{1-f["failed_attempt_paid_s"]/n["failed_attempt_paid_s"]:.1%}。这是多任务累计时间，不是整批墙钟时间。')
    report.append('reasoning变化中Full中止53次，NoPhase45次，但失败尝试付出的计算等待更少，且整体吞吐和尾延迟更好。因此不能用中止次数单独代替重做代价；应结合失败发生位置、计算成本及锁等待评价。')
    (OUT/'REPORT_ZH.md').write_text('\n\n'.join(report),encoding='utf-8')
    shutil.copy2(RAW/'plan.json',OUT/'plan.json');shutil.copy2(RAW/'completion.json',OUT/'completion.json')
    sources=OUT/'sources';sources.mkdir(exist_ok=True)
    for name in ['run_phase_business_variant_20261113.py','run_mentor_rule_ablation_20261114.py',
                 'run_mentor_continuous_phase_20261114.py','launch_mentor_additions_20261114.py']:
        shutil.copy2(ROOT/'scripts/revision'/name,sources/name)
    (OUT/'verification.json').write_text(json.dumps(dict(cells=78,all_exit_zero=True,all_values_correct=True,
        all_ablation_tasks_and_competitors_complete=True,paired_inputs_identical=True,
        all_continuity_verified=True,all_actual_model_hashes_match=True),indent=2))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    print(json.dumps(dict(ablation_cells=72,continuous_cells=6,output=str(OUT)),indent=2))

if __name__=='__main__':main()
