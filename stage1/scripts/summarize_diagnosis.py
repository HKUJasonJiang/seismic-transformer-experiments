"""Summarize preserved training-only intervention runs; never evaluate holdouts."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import numpy as np
from peer10_common import STAGE,json_write,load_split
from diagnose_nonlinear import selection


def main():
    p=argparse.ArgumentParser();p.add_argument('--selected',required=True);args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=STAGE/'reports'/'diagnosis-nonlinear-summary';output.mkdir(parents=True,exist_ok=True)
    paths=sorted((STAGE/'reports').glob('diagnosis-*/result.json'))
    results={path.parent.name:json.loads(path.read_text()) for path in paths}
    chosen=results[args.selected]
    assert chosen['provenance']['deployable_inputs'] and chosen['arm']!='fixed_head'
    lines=['# 3/5/8 层非线性训练拟合诊断','','只使用原 7 条训练波的 65 个 DS_ref>2 算例。没有生成新数据，没有读取验证或测试标签。以下全部为训练拟合，不能解释成泛化。','',
           '| 运行 | 位移 L2 中位数 | 加速度 L2 中位数 | 加速度 P90 | 每类层数门槛 | 步数 | 可训练参数 |',
           '| --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for name,r in results.items():
        s=r['summary']['all'];lines.append(f"| {name} | {s['u_median_relative_l2']:.2%} | {s['a_median_relative_l2']:.2%} | {s['a_p90_relative_l2']:.2%} | {r['status']} | {r['best_step']}/{r['last_step']} | {r['provenance']['trainable_parameters']:,} |")
    lines+=['','free_latent 使用算例身份直接优化时间 latent，仅检验 B 的重建能力；fixed_head 是固定 8 槽输出对照。frozen_A 冻结旧 A 并重新训练 B。参数量和计算预算不完全相等，不能把差值看成纯因果贡献。','',
            f"## 当前保留的查询模型：{args.selected}",'',f"训练拟合状态：**{chosen['status']}**。判定要求每类层数两通道中位数≤10%、P90≤20%。",'',
            '| 层数 | 数量 | 位移中位数 / P90 / 最差 | 加速度中位数 / P90 / 最差 | PIDR / PFA 逐层误差中位数 | DS 一致率 |',
            '| --- | ---: | --- | --- | --- | ---: |']
    for n in (3,5,8):
        s=chosen['summary']['by_stories'][str(n)]['all']
        triple=lambda c:' / '.join(f"{s[f'{c}_{q}_relative_l2']:.2%}" for q in ('median','p90','worst'))
        lines.append(f"| {n} | {s['count']} | {triple('u')} | {triple('a')} | {s['PIDR_floor_median_relative_error']:.2%} / {s['PFA_floor_median_relative_error']:.2%} | {s['DS_ref_accuracy']:.2%} |")
    q=chosen['query_checks'];lines+=['',f"独立楼层查询：{q['status']}，共 {q['single_floor_queries']} 次，另检查倒序查询。",'',
       '图中各层数选两通道误差之和居中的训练算例，另保留全体最差算例。完整指标、逐算例结果、源码版本、数据清单和权重哈希在各 result.json。','',
       '![对照](comparison.png)','','![典型顶层时程](roof_examples.png)','',
       f'![学习曲线](../{args.selected}/learning_curve.png)','']
    for n in (3,5,8):lines+=[f'![{n} 层全部时程](../{args.selected}/fit_{n:02d}F.png)','']
    lines+=[f'![最差算例](../{args.selected}/fit_worst.png)','','下一步只在训练拟合稳定后开展泛化试验；旧测试记录已使用，不能在反复调试后继续当作独立最终测试。']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    json_write(output/'summary.json',{'selected':args.selected,'runs':{name:{'arm':r['arm'],'status':r['status'],'best_step':r['best_step'],'summary':r['summary'],'provenance':r['provenance']} for name,r in results.items()},'validation':'NOT_READ','test':'NOT_READ'})
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=list(results)
    for ax,c,label in zip(axes,('u','a'),('Displacement','Absolute acceleration')):
        for name in names:
            curve=json.loads((STAGE/'reports'/name/'curve.json').read_text())
            ax.semilogy([x['step'] for x in curve],[x['summary']['all'][f'{c}_median_relative_l2'] for x in curve],label=name.replace('diagnosis-nonlinear-v1-',''))
        ax.axhline(.1,color='gray',ls=':');ax.set(xlabel='Optimizer steps in this run',ylabel='Training median relative L2',title=label);ax.legend(fontsize=6)
    fig.suptitle('A/B interventions | 65 nonlinear TRAINING cases | no held-out evaluation')
    fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)
    _,_,batch=load_split(STAGE/chosen['provenance']['config']['dataset'],'train','cpu')
    with np.load(STAGE/'runs'/args.selected/'train_predictions.npz',allow_pickle=False) as p:
        ids=list(p['case_ids']);pred={k:p[k] for k in ('displacement','acceleration')}
    fig,axes=plt.subplots(3,2,figsize=(13,8),constrained_layout=True)
    for row,n in enumerate((3,5,8)):
        candidates=sorted([x for x in chosen['cases'] if x['stories']==n],key=lambda x:x['u_relative_l2']+x['a_relative_l2']);entry=candidates[len(candidates)//2]
        i=batch['case_ids'].index(entry['id']);j=ids.index(entry['id']);t=np.arange(batch['ground'].shape[1])*.02
        for col,(key,scale,unit) in enumerate((('displacement',1000.,'mm'),('acceleration',1/9.80665,'g'))):
            ax=axes[row,col];ax.plot(t,batch[key][i,:,n-1].numpy()*scale,color='#303030',lw=.9,label='C++ reference')
            ax.plot(t,pred[key][j,:,n-1]*scale,color='#d6604d',lw=.7,alpha=.8,label='Training fit');ax.set(xlabel='Time (s)',ylabel=unit,xlim=(0,75),title=f"{n}F roof | DS {entry['DS_ref']} | building u {entry['u_relative_l2']:.1%}, a {entry['a_relative_l2']:.1%}");ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Nonlinear training examples | median-error case at each height | not generalization')
    fig.savefig(output/'roof_examples.png',dpi=170);plt.close(fig)
    print(json.dumps({'selected':args.selected,'status':chosen['status'],'by_stories':chosen['summary']['by_stories']}),flush=True)


if __name__=='__main__':main()
