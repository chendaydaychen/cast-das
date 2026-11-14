"""Add cost curves to the unchanged 110-cell main performance matrix."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'revision_experiments/six_task_paper_20261113/main_curve_equivalent_token_cost_all110.csv'
OUT = ROOT / 'revision_experiments/main_token_curves_20261113'
OUT.mkdir(exist_ok=True)
with SOURCE.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
names = [('paper-atcc','Aegis'), ('occ','OCC'), ('2pl-nowait','2PL'),
         ('mvcc','MVCC'), ('silo','Silo'), ('tictoc','TicToc'),
         ('bamboo','Bamboo'), ('polaris','Polaris'), ('mocc','MOCC'),
         ('hekaton','Hekaton'), ('polyjuice','Polyjuice')]
workers = [40,80,120,160,200]
index = {(r['workload'],r['cc'],int(r['workers'])):r for r in rows}
assert len(index) == len(rows) == 110
assert set(index) == {(w,c,n) for w in ['ycsb','tpcc'] for c,_ in names for n in workers}
for r in rows:
    assert abs(float(r['estimated_total_tokens']) / int(r['completed']) - float(r['estimated_tokens_per_success'])) < 1e-7

plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none'})
colors = ['#bd2333'] + [plt.get_cmap('tab10')(i) for i in range(10)]
markers = ['o','s','^','v','D','P','X','<','>','h','*']
styles = ['-','--','-.',':','--','-.',':','--','-.',':','--']

def draw(ax, workload, field, label, logarithmic=False):
    for i,(cc,name) in enumerate(names):
        vals = [float(index[workload,cc,n][field]) for n in workers]
        ax.plot(workers, vals, label=name, color=colors[i], marker=markers[i],
                linestyle=styles[i], linewidth=2.5 if i==0 else 1.2,
                markersize=6 if i==0 else 4, zorder=10 if i==0 else 2)
    ax.set_xticks(workers)
    ax.set_xlabel('Workers')
    ax.set_ylabel(label)
    if logarithmic:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(alpha=.23, which='both')

for log in [False,True]:
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,w,title in zip(axs,['ycsb','tpcc'],['Agent-YCSB','Agent-TPC-C']):
        draw(ax,w,'estimated_tokens_per_success','Equivalent tokens / successful task',log)
        ax.set_title(title + (' (log cost axis)' if log else ''))
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=6,bbox_to_anchor=(.5,.005),frameon=False)
    fig.tight_layout(rect=(0,.14,1,1))
    stem='token_cost_curves_log' if log else 'token_cost_curves_linear'
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(16,8))
for axrow,w,title in zip(axs,['ycsb','tpcc'],['Agent-YCSB','Agent-TPC-C']):
    for ax,field,label,log in zip(axrow,
            ['agent_tps','p99_s','estimated_tokens_per_success'],
            ['Agent TPS','Retry-inclusive P99 (s)','Equivalent tokens / successful task'],
            [False,False,True]):
        draw(ax,w,field,label,log)
        ax.set_title(title)
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=6,bbox_to_anchor=(.5,.005),frameon=False)
fig.tight_layout(rect=(0,.09,1,1))
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'main_tps_p99_token.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)

for w in ['ycsb','tpcc']:
    with (OUT/f'{w}_token_per_success_wide.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['workers']+[name for _,name in names])
        for n in workers:
            writer.writerow([n]+[float(index[w,c,n]['estimated_tokens_per_success']) for c,_ in names])
with (OUT/'all110_performance_and_cost.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)

summary={}
for w in ['ycsb','tpcc']:
    own=float(index[w,'paper-atcc',200]['estimated_tokens_per_success'])
    baseline=min((float(index[w,c,200]['estimated_tokens_per_success']),name) for c,name in names if c!='paper-atcc')
    summary[w]={'aegis_tokens_per_success_200':own,'minimum_baseline_200':baseline,
                'reduction_vs_minimum_baseline':1-own/baseline[0]}
(OUT/'verification.json').write_text(json.dumps({'cells':len(rows),'workers':workers,'summary':summary},indent=2))
(OUT/'README_ZH.md').write_text('''# 主性能曲线新增 token cost 图

与原主曲线使用相同110个实验单元：YCSB/TPC-C、11项CC、40/80/120/160/200workers、每点240任务、最多5次重试。

主图 main_tps_p99_token 为两行三列：吞吐、含重试P99、每成功任务等效token成本。独立cost图同时提供线性与对数纵轴，全量曲线无删点、无平滑。对数轴用于展示成本跨度；线性版本保留绝对差距。

成本=所有任务所有尝试的等效输入与输出token之和/成功任务数，包含失败尝试和最终失败任务消耗。完成率保留在all110表中。输入输出相加表示工作量，不表示等价单价。

计算沿用six_task_paper_20261113中的真实调用配对与阶段预算规则。本次仅生成图和宽表，不新增API调用、不改变原策略、不重跑性能。合成主曲线无真实LLM请求，必须标注Equivalent token cost，不可标成实际API账单。

建议图注：真实调用标定的Agent-YCSB与Agent-TPC-C中，不同并发度下各CC策略的任务吞吐、含重试P99及每成功任务等效token成本。成本包含全部尝试的执行工作量；cost纵轴采用对数刻度。
''',encoding='utf-8')
print(json.dumps(summary,indent=2))
print(OUT)
