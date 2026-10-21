"""Separate roof and whole-building errors using archived metrics only.

No inference, raw response labels, checkpoint selection, or retraining occurs here.
"""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import numpy as np

STAGE=Path(__file__).resolve().parents[1]


def aggregate(rows):
    out={'count':len(rows)}
    for c in ('u','a'):
        out[c+'_roof_median_relative_l2']=float(np.median([r[c+'_floor_relative_l2'][-1] for r in rows]))
        out[c+'_building_median_relative_l2']=float(np.median([r[c+'_relative_l2'] for r in rows]))
    return out


def main():
    selected='diagnosis-nonlinear-v1-basis_joint'
    inputs=[('current_train',STAGE/'reports'/selected/'result.json'),
            ('current_validation',STAGE/'reports'/selected/'validation_result.json'),
            ('historical_test',STAGE/'reports'/'peer10-358-patch60-seed20261021'/'result.json')]
    result={'scope':'Archived metric decomposition only; no new test inference or training','sources':{},'groups':{}}
    for name,path in inputs:
        data=json.loads(path.read_text());result['sources'][name]={'path':str(path.relative_to(STAGE)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        rows=data['splits']['test']['cases'] if name=='historical_test' else data['cases']
        strong=[r for r in rows if r['DS_ref']>2]
        result['groups'][name]={'DS_gt_2':aggregate(strong),'by_stories':{str(n):aggregate([r for r in strong if r['stories']==n]) for n in (3,5,8)}}
    lines=['# 顶层与全楼误差核对','','只拆分已归档指标，不重新预测、不训练，也不读取原始测试响应。以下均为 DS_ref>2 子集。','',
           '| 模型 / 数据 | 层数 | 算例 | 顶层位移 L2 | 全楼位移 L2 | 顶层加速度 L2 | 全楼加速度 L2 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,g in result['groups'].items():
        for n,s in [('全部',g['DS_gt_2']),*g['by_stories'].items()]:
            values=[s[c+'_'+scope+'_median_relative_l2'] for c in ('u','a') for scope in ('roof','building')]
            lines.append(f"| {name} | {n} | {s['count']} | "+' | '.join(f'{v:.2%}' for v in values)+' |')
    lines+=['','每项均为逐算例相对 L2 的中位数。顶层只对顶层时程计算，全楼在该算例全部有效楼层与时间上计算；二者不是相同聚合口径。',
            'current 为新的空间基查询模型；historical_test 为旧 attention 查询模型此前冻结的测试结果，不跨模型比较改进幅度。',
            '当前及历史分组均按地震动记录留出，结构配置在训练中出现过。它们验证未见波，不验证未见结构。',
            '历史单波 24 算例全部参与训练，只证明多结构联合拟合；没有单波下的结构留出验证。']
    folder=STAGE/'reports'/'diagnosis-response-scope';folder.mkdir(parents=True,exist_ok=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result['groups']),flush=True)


if __name__=='__main__':main()
