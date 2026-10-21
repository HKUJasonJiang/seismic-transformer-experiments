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
    results={path.parent.name:json.loads(path.read_text()) for path in paths if (path.parent/'curve.json').exists()}
    chosen=results[args.selected]
    assert chosen['provenance']['deployable_inputs'] and chosen['arm']!='fixed_head'
    valpath=STAGE/'reports'/args.selected/'validation_result.json'
    validation=json.loads(valpath.read_text()) if valpath.exists() else None
    by_arm={r['arm']:r for r in results.values()}
    diagnoses=[]
    for key,label in (('joint','原 A+B'),('free_latent','绕过 A、直接优化 latent + 原 B'),('wide_joint','原 A + 加宽 B'),('basis_joint','原 A + 空间基查询 B')):
        if key in by_arm:
            z=by_arm[key];a=z['summary']['all']['a_median_relative_l2']
            diagnoses.append(f"- {label}：加速度时程相对 L2 中位数 {a:.2%}，{z['best_step']} 步，{z['provenance']['trainable_parameters']:,} 个可训练参数。")
    conclusion=['# 诊断结论','','**3/5/8 层非线性训练拟合已通过；新地震动泛化仍需单独解决。**','',
        '本轮已证明改进 B 可以显著改善训练拟合：保留 A 的结构并重新联合训练后，每类层数均通过训练门槛。此前把它概括成主要瓶颈在 B，归因偏强。自由 latent 对照也同时优化 B，既没有固定实际 A 的 latent，也没有固定同一个 B，因此不能据此证明实际 latent 信息充分，或单独定位新波泛化失败发生在 A 还是 B。加宽对照同样只证明当前预算下的拟合差异。','']+diagnoses+['',
        '保留模型的 A 主干、60 点 patch 和 128 维 latent 不变。B 预测随时间变化的响应系数，用完整的楼层坐标基组合查询层响应。当前 N≤8；这些坐标基不是识别得到的真实振型，也没有采用时间 FFT/FNO。模型从正常地震动/建筑输入独立推理，不需要样本身份、真实顶层曲线或响应标签。','',
        '训练数据仍为原 7 条波的 65 个 DS_ref>2 算例：3 层 22、5 层 22、8 层 21。每类层数两通道相对 L2 中位数≤10%、P90≤20%。没有增加数据。','',
        f"保留模型：`{args.selected}`。完整对照、逐层误差、EDP、最差算例和时程见 [报告](README.md)。",'']
    if validation:
        v=validation['summary']['all']
        conclusion += [f"冻结后的一次原验证波快照（9 个 DS>2 算例）：位移误差中位数 {v['u_median_relative_l2']:.2%}，加速度 {v['a_median_relative_l2']:.2%}，PIDR/PFA 逐层误差中位数 {v['PIDR_floor_median_relative_error']:.2%}/{v['PFA_floor_median_relative_error']:.2%}。因此只能说已经能够拟合训练中的非线性响应，不能说已经解决新波泛化。此验证结果没有再反馈到本轮模型；旧测试波未读取。",'']
    conclusion += ['下一步保持现有数据，针对地震动到动态响应系数的映射开展泛化诊断，优先检查时间位置依赖、相位与频率响应和非线性历史信息。不要把拟合通过直接写成泛化结论；扩增数据继续暂缓。','','![训练时程](roof_examples.png)','']
    (output/'CONCLUSION.md').write_text('\n'.join(conclusion),encoding='utf-8')
    lines=['# 3/5/8 层非线性训练拟合诊断','','所有训练对照只使用原 7 条训练波的 65 个 DS_ref>2 算例，没有生成新数据。模型/权重只按训练拟合选择。以下对照表全部为训练拟合；若存在模型冻结后的一次开发验证，单独列在末尾。旧测试波未读取。','',
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
    lines+=[f'![最差算例](../{args.selected}/fit_worst.png)','']
    if validation:
        lines+=['## 拟合通过并冻结模型后的一次开发验证','','只查看原 validation 波的 DS_ref>2 算例，未据此继续调模型。这条波在历史开发中已经使用，不能称为新的独立最终测试。','',
                '| 层数 | 样本 | 位移 L2 中位数 | 加速度 L2 中位数 | PIDR / PFA 逐层误差中位数 |','| --- | ---: | ---: | ---: | --- |']
        for n in (3,5,8):
            v=validation['summary']['by_stories'][str(n)]['all']
            lines.append(f"| {n} | {v['count']} | {v['u_median_relative_l2']:.2%} | {v['a_median_relative_l2']:.2%} | {v['PIDR_floor_median_relative_error']:.2%} / {v['PFA_floor_median_relative_error']:.2%} |")
    lines+=['','旧测试记录已使用，不能在反复调试后继续当作独立最终测试。当前单随机种子，未证明未见层数、其他结构类别或真实建筑泛化。']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    json_write(output/'summary.json',{'selected':args.selected,'runs':{name:{'arm':r['arm'],'status':r['status'],'best_step':r['best_step'],'summary':r['summary'],'provenance':r['provenance']} for name,r in results.items()},'development_validation_after_freeze':validation,'test':'NOT_READ'})
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
