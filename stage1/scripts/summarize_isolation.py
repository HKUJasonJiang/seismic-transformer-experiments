"""Separate roof, all-floor and oracle figures for the isolation experiment."""
from isolation_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(4)
    conf=json.loads((STAGE/'configs/isolation.json').read_text());prefix=conf['prefix']
    out=STAGE/'reports'/(prefix+'-summary');out.mkdir(parents=True,exist_ok=True)
    results={p.parent.name.removeprefix(prefix+'-'):json.loads(p.read_text()) for p in (STAGE/'reports').glob(prefix+'-*/result.json')}
    required=['reference','fixedA-roof','fixedA-full','fixedB-oracle','independent-roof','holdout-period-roof','holdout-period-full','holdout-stories-roof','holdout-stories-full','holdout-period-interpolation']
    assert all(k in results for k in required),set(required)-set(results)
    samples={};entries={}
    for split in ('train','validation'):
        ds,_,_=load_split(STAGE/conf['dataset'],split,'cpu')
        samples.update({v['case_id']:v for v in ds.items});entries.update({v['id']:v for v in ds.cases})
    cache={}
    def pred(method,split,case,key,floor):
        tag=(method,split)
        if tag not in cache:
            with np.load(STAGE/'runs'/(prefix+'-'+method)/(split+'_predictions.npz'),allow_pickle=False) as f:cache[tag]={k:f[k].copy() for k in f.files}
        a=cache[tag];i=a['case_ids'].tolist().index(case)
        return a[key][i,:,0 if results[method]['provenance'].get('roof_only',False) else floor-1]
    def error(method,split,case,key):
        r=next(x for x in results[method]['splits'][split]['cases'] if x['id']==case)
        return r[('u' if key=='displacement' else 'a')+'_roof_relative_l2']
    colors={'reference':'#246ca6','fixedA-roof':'#349b68','fixedA-full':'#349b68','independent-roof':'#9c4ba4','fixedB-oracle':'#d67920',
            'holdout-period-full':'#246ca6','holdout-period-roof':'#9c4ba4','holdout-stories-full':'#246ca6','holdout-stories-roof':'#9c4ba4','holdout-period-interpolation':'#349b68'}
    names={'reference':'Current A+B','fixedA-roof':'Frozen A + new roof head','fixedA-full':'Frozen A + new all-floor head','independent-roof':'Independent roof predictor','fixedB-oracle':'Frozen B + optimized Z (ORACLE)',
           'holdout-period-full':'Fresh A+B','holdout-period-roof':'Fresh roof predictor','holdout-stories-full':'Fresh A+B','holdout-stories-roof':'Fresh roof predictor','holdout-period-interpolation':'Training-response interpolation'}
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    def meta(case):
        e=entries[case];return f"{e['stories']} floors | T1={e['T1_s']:.3f}s | ratio={e['period_ratio']:.1f} | PGA={e['pga_g']:.3f}g | DS={e['DS_ref']}"
    def panel(ax,case,key,floor,methods,split,zoom=False):
        truth=samples[case][key][:,floor-1].numpy();t=np.arange(len(truth))*.02;scale=1000 if key=='displacement' else 1/9.80665
        ax.plot(t,truth*scale,color='#161616',lw=1.05,label='C++ reference')
        for method in methods:
            y=pred(method,split,case,key,floor)
            ax.plot(t,y[:len(t)]*scale,color=colors[method],lw=.85,alpha=.9,ls='--' if 'interpolation' in method else '-',label=names[method])
        ax.set(xlim=(0,25 if zoom else 75),ylabel='Displacement (mm)' if key=='displacement' else 'Absolute accel. (g)')
    # Choose 5F ratio 1.0 by metadata, not by prediction quality.
    choices=[r for r in results['reference']['splits']['held']['cases'] if r['stories']==5]
    case=min(choices,key=lambda r:abs(r['period_ratio']-1.))['id']
    methods=['reference','fixedA-roof','fixedA-full','independent-roof','fixedB-oracle']
    fig,axs=plt.subplots(len(methods),2,figsize=(13,10.5),sharex=True,constrained_layout=True)
    for i,m in enumerate(methods):
        for col,key in enumerate(('displacement','acceleration')):
            panel(axs[i,col],case,key,5,[m],'held')
            axs[i,col].set_title(f"{names[m]} | roof L2={error(m,'held',case,key):.1%}",fontsize=9)
    for ax in axs[-1]:ax.set_xlabel('Time (s)')
    axs[0,0].legend(fontsize=7);axs[-1,0].legend(fontsize=7)
    fig.suptitle('UNSEEN DEVELOPMENT RECORD: ROOF\n'+meta(case)+'\nOrange bottom row uses the held response truth; it is not a prediction.',fontsize=11)
    fig.savefig(out/'multiwave_roof.png',dpi=150);plt.close(fig)
    def floors(case,methods,split,title,filename):
        n=entries[case]['stories'];fig,axs=plt.subplots(n,2,figsize=(13,1.7*n+1),sharex=True,constrained_layout=True)
        for i in range(n):
            for j,key in enumerate(('displacement','acceleration')):
                panel(axs[i,j],case,key,i+1,methods,split);axs[i,j].set_title(f'Floor {i+1}',fontsize=8)
        axs[0,0].legend(fontsize=7,ncol=2)
        for ax in axs[-1]:ax.set_xlabel('Time (s)')
        fig.suptitle(title+'\n'+meta(case),fontsize=11);fig.savefig(out/filename,dpi=150);plt.close(fig)
    floors(case,['reference','fixedA-full','fixedB-oracle'],'held','UNSEEN DEVELOPMENT RECORD: EVERY FLOOR\nOrange = truth-assisted reconstruction; blue/green = ordinary predictions','multiwave_all_floors.png')
    selected={ 'multiwave':case }
    for fold in ('period','stories'):
        full='holdout-'+fold+'-full';roof_method='holdout-'+fold+'-roof';methods=[full,roof_method]
        if fold=='period':methods.append('holdout-period-interpolation')
        rows=[r for r in results[full]['splits']['held']['cases'] if abs(r['pga_g']-.6)<1e-6]
        rows.sort(key=lambda r:(r['stories'],r['period_ratio']));assert len(rows)==3
        fig,axs=plt.subplots(3,2,figsize=(13,7.5),sharex=True,constrained_layout=True)
        for i,r in enumerate(rows):
            for col,key in enumerate(('displacement','acceleration')):
                panel(axs[i,col],r['id'],key,r['stories'],methods,'held')
                axs[i,col].set_title(meta(r['id'])+'\n'+', '.join(f"{names[m]} {error(m,'held',r['id'],key):.1%}" for m in methods),fontsize=7)
        for ax in axs[-1]:ax.set_xlabel('Time (s)')
        axs[0,0].legend(fontsize=7)
        title='SAME RECORD: HELD-OUT T1 CONFIGURATIONS' if fold=='period' else 'SAME RECORD: ENTIRE 5-FLOOR GROUP HELD OUT'
        fig.suptitle(title+' | ROOF at common PGA=0.6g\nAll neural models initialized from scratch; checkpoints selected on training only.',fontsize=11)
        fig.savefig(out/('holdout_'+fold+'_roof.png'),dpi=150);plt.close(fig)
        selected_case=min(rows,key=lambda r:abs(r['stories']-5)+abs(r['period_ratio']-1))['id'];selected[fold]=selected_case
        floors(selected_case,[full]+(['holdout-period-interpolation'] if fold=='period' else []),'held',title+' | EVERY FLOOR','holdout_'+fold+'_floors.png')
    train_case=max([r for r in results['independent-roof']['splits']['train']['cases'] if r['stories']==5],key=lambda r:r['u_roof_relative_l2']+r['a_roof_relative_l2'])['id']
    methods=['reference','fixedA-roof','independent-roof'];fig,axs=plt.subplots(3,2,figsize=(13,7),sharex=True,constrained_layout=True)
    for i,m in enumerate(methods):
        for j,key in enumerate(('displacement','acceleration')):
            panel(axs[i,j],train_case,key,5,[m],'train');axs[i,j].set_title(f"{names[m]} | roof L2 {error(m,'train',train_case,key):.1%}")
    for ax in axs[-1]:ax.set_xlabel('Time (s)')
    axs[0,0].legend(fontsize=7);fig.suptitle('TRAINING FIT: ROOF\n'+meta(train_case)+'\nWorst 5F training case by independent roof u+a relative L2.')
    fig.savefig(out/'train_roof.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for m in required:
        path=STAGE/'reports'/(prefix+'-'+m)/'curve.json'
        if path.exists():
            c=json.loads(path.read_text());ax.plot([r['step'] for r in c],[r['criterion'] for r in c],label=m)
    ax.axhline(1,c='k',ls='--',label='Fit gate');ax.set(xlabel='Training step',ylabel='Worst normalized training error (gate = 1)',yscale='log',title='Training-only checkpoint selection')
    ax.legend(fontsize=7,ncol=2);fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    # Machine-readable and human-readable full roof/floor separation.
    summary={m:{'training_gate':r.get('training_gate','not_applicable'),'best_step':r.get('best_step'),
                'splits':{s:v['summary'] for s,v in r['splits'].items()},'frozen_check':r.get('frozen_module_check',r['provenance'].get('frozen_module_check'))} for m,r in results.items()}
    json_write(out/'summary.json',{'groups':summary,'selected_cases':selected,'test':'NOT_READ'})
    lines=['# 固定模块与结构留出结果','', '所有计算和作图在指定 GPU 主机完成；没有新增仿真，旧 test 未读取。开发验证波已经用于历史开发，因此不能当作新的最终测试。', '',
           '误差为逐算例时程相对 L2 的中位数；全楼指标联合有效楼层计算。固定 B 一行使用目标真值优化 Z，只说明可重建性，不是预测。','',
           '| 实验 | 划分 | n | 顶层位移 | 顶层加速度 | 全楼位移 | 全楼加速度 |','|---|---|---:|---:|---:|---:|---:|']
    for m in required:
        for split,v in summary[m]['splits'].items():
            s=v['all'];vals=[s.get(k) for k in ('u_roof_median','a_roof_median','u_median_relative_l2','a_median_relative_l2')]
            lines.append('| '+m+' | '+split+' | '+str(s['count'])+' | '+' | '.join(f'{x:.2%}' if x is not None else '不输出' for x in vals)+' |')
    lines+=['','## DS>2 的结构留出','', '| 实验 | n | 顶层位移 | 顶层加速度 | 全楼位移 | 全楼加速度 |','|---|---:|---:|---:|---:|---:|']
    for m in required:
        if not m.startswith('holdout'):continue
        s=summary[m]['splits']['held']['DS_gt_2'];vals=[s.get(k) for k in ('u_roof_median','a_roof_median','u_median_relative_l2','a_median_relative_l2')]
        lines.append('| '+m+' | '+str(s['count'])+' | '+' | '.join(f'{x:.2%}' if x is not None else '不输出' for x in vals)+' |')
    lines+=['','## 逐层误差','', '| 实验/划分 | 楼层数 | 查询楼层 | 位移中位数 | 加速度中位数 |','|---|---:|---:|---:|---:|']
    for m in required:
        for split,v in summary[m]['splits'].items():
            for n,fs in v['all'].get('by_floor',{}).items():
                for f,s in fs.items():lines.append(f"| {m}/{split} | {n} | {f} | {s['u']:.2%} | {s['a']:.2%} |")
    lines+=['','## 预测图','']
    for filename in ('train_roof','multiwave_roof','multiwave_all_floors','holdout_period_roof','holdout_period_floors','holdout_stories_roof','holdout_stories_floors','learning_curves'):
        lines+=['![ '+filename+' ]('+filename+'.png)','']
    lines+=['## 各组完整记录','']
    for m in required:
        lines.append(f'- [{m} 指标及来源](../{prefix}-{m}/result.json)；[全部留出顶层图](../{prefix}-{m}/held_roof.png)')
        for r in results[m]['splits']['held']['cases']:lines.append(f"  - [{r['id']}](../{prefix}-{m}/held/{r['id']}.png)")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':str(out),'selected_cases':selected,'groups':summary},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
