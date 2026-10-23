"""Collect finished runs of one stage into results/<stage>/SUMMARY.md; the carried-forward variant uses development data only.

  python codes/summarize.py                         # stage 1
  python codes/summarize.py --config configs/stage2.json
"""
import argparse
import numpy as np
from common import EXP, REPO, read, write

KEYS = (('u_roof_mse', 'roof displacement MSE (m²)'), ('a_roof_mse', 'roof acceleration MSE ((m/s²)²)'),
        ('u_all_mse', 'all-floor displacement MSE (m²)'), ('a_all_mse', 'all-floor acceleration MSE ((m/s²)²)'))


def cell(values):
    return f'{np.mean(values):.4g} [{min(values):.4g}–{max(values):.4g}]' if values else '—'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/run.json')
    args = parser.parse_args()
    base = read(EXP/args.config)
    stage = base['stage']
    splits = [s for s in base['evaluation_splits'] if s != 'train']
    names = [(v, v) for v in base['variants']]
    names += [(v, f'{v}/records{n}') for v in base['variants'] for n in base.get('training_record_counts', [])]
    found = {}
    for variant, name in names:
        for seed in base['seeds']:
            path = EXP/'results'/stage/name/f'seed{seed}'/'report.json'
            if path.exists():
                found.setdefault(name, {})[seed] = read(path)
    exp5 = REPO/'exp/exp5-normalized-floor-query/results/report.json'
    reference = read(exp5) if exp5.exists() and stage == 'stage1' else None
    lines = [f'# exp7 {stage} 汇总', '', '每格为各随机种子的平均值 [最小值–最大值]。MSE 为物理单位：每个算例先在有效时间和楼层上平均，再对算例等权平均。', '']
    summary = {'runs': {}, 'selection': None}
    for split in splits:
        for subset in ('all', 'DS_gt_2'):
            lines += [f'## {split} / {subset}', '',
                      '| 运行 | 种子数 | ' + ' | '.join(label for _, label in KEYS) + ' | 最大层间位移角 预测/真值 中位数 |',
                      '|---|---:|' + '---:|'*(len(KEYS)+1)]
            if reference and split in reference['splits']:
                s = reference['splits'][split]['summary'][subset]
                lines.append('| exp5 原模型（单种子） | 1 | ' + ' | '.join(f'{s[k]:.4g}' for k, _ in KEYS) + ' | — |')
            for _, name in names:
                rows = [r['splits'][split]['summary'][subset] for r in found.get(name, {}).values()
                        if split in r['splits'] and subset in r['splits'][split]['summary']]
                if not rows:
                    continue
                ratios = [r['PIDR_max_ratio']['median'] for r in rows if 'PIDR_max_ratio' in r]
                lines.append(f'| {name} | {len(rows)} | ' + ' | '.join(cell([r[k] for r in rows]) for k, _ in KEYS)
                             + f' | {cell(ratios)} |')
                summary['runs'].setdefault(name, {}).setdefault(split, {})[subset] = {k: [r[k] for r in rows] for k, _ in KEYS}
            lines.append('')
    complete = [v for v in base['variants'] if len(found.get(v, {})) == len(base['seeds'])]
    if complete:
        # Predeclared rule: lowest mean development objective among variants with every seed finished.
        score = {v: float(np.mean([r['splits']['development']['objective'] for r in found[v].values()])) for v in complete}
        chosen = min(score, key=score.get)
        summary['selection'] = {'rule': 'lowest seed-mean development objective; tests not consulted',
                                'development_objective': score, 'chosen': chosen}
        lines += ['## 按开发集选出的变体', '', f'开发集目标（各种子平均）最低：**{chosen}**。测试集结果不参与这一选择。', '',
                  '| 变体 | 开发集目标（种子平均） |', '|---|---:|'] + [f'| {v} | {s:.4g} |' for v, s in sorted(score.items(), key=lambda x: x[1])]
    missing = {name: [s for s in base['seeds'] if s not in found.get(name, {})] for _, name in names}
    lines += ['', '尚未完成的运行：' + ('无' if not any(missing.values()) else '; '.join(f'{v}: {s}' for v, s in missing.items() if s))]
    summary['missing_runs'] = missing
    out = EXP/'results'/stage
    out.mkdir(parents=True, exist_ok=True)
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'summary.json', summary)
    print('SUMMARY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
