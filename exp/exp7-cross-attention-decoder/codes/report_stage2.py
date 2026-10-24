"""Stage-2 analysis and figures for the exp7 report (execution host only).

Reads results/stage2 reports and summary.json, shard metadata and the saved test predictions; writes
figures/stage2/*.png and results/stage2/report_data.json. Case selection rules are fixed here, not chosen by eye:
- per variant, the seed with the median development objective is the "shown" seed;
- best / median / worst cases rank the shown exp5_decoder run by normalized all-floor error
  0.5*(MSE_u/mean(u_true^2) + MSE_a/mean(a_true^2)) within each test part (physical MSE is still what is reported).
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP/'codes'))
from common import read, write  # noqa: E402
from datagen.shards import load_part  # noqa: E402

VARIANTS = ('exp5_decoder', 'cross_attention')
COLORS = {'truth': '#243447', 'exp5_decoder': '#2b6cb0', 'cross_attention': '#e16a2d'}
LABELS = {'exp5_decoder': 'exp5-style decoder', 'cross_attention': 'cross-attention decoder'}
PARTS = ('unseen_record', 'seen_record_reference', 'cross_database', 'structure_only', 'joint')
FIG = EXP/'figures/stage2'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False, 'axes.spines.right': False})


def mean_range(values):
    return {'mean': float(np.mean(values)), 'min': float(np.min(values)), 'max': float(np.max(values))}


def main():
    conf = read(EXP/'configs/stage2.json')
    data_conf = read(EXP/'configs/data.json')
    seeds = conf['seeds']
    summary = read(EXP/'results/stage2/summary.json')['runs']
    FIG.mkdir(parents=True, exist_ok=True)
    out = {'seeds': seeds, 'main': {}, 'scaling': {}, 'groups': {}, 'stories': {}, 'cases': {}}
    reports = {v: {s: read(EXP/'results/stage2'/v/f'seed{s}'/'report.json') for s in seeds} for v in VARIANTS}
    shown = {}
    for v in VARIANTS:
        dev = {s: reports[v][s]['splits']['development']['objective'] for s in seeds}
        shown[v] = sorted(dev, key=dev.get)[1]
        out.setdefault('shown_seed', {})[v] = {'seed': shown[v], 'development_objective': dev}
        out.setdefault('training', {})[v] = {
            'parameters': reports[v][seeds[0]]['provenance']['parameters'],
            'steps': reports[v][seeds[0]]['provenance']['config']['steps'],
            'train_cases': len(reports[v][seeds[0]]['provenance']['train_case_ids']),
            'best_steps': [reports[v][s]['best_step'] for s in seeds]}
    keys = ('u_roof_mse', 'a_roof_mse', 'u_all_mse', 'a_all_mse', 'PIDR_max_mse', 'PIDR_max_rmse', 'PIDR_max_bias', 'PIDR_max_relative_bias')
    for v in VARIANTS:
        for split in ['development'] + list(PARTS):
            for subset in ('all', 'DS_gt_2'):
                row = summary[v][split][subset]
                out['main'].setdefault(v, {}).setdefault(split, {})[subset] = {k: mean_range(row[k]) for k in keys}
                out['main'][v][split][subset]['count'] = reports[v][seeds[0]]['splits'][split]['summary'][subset]['count']
    # Record-count scaling curve: --records N runs plus the main run (all training records).
    counts = conf['training_record_counts'] + [len(read(EXP/data_conf['record_split'])['training_order'])]
    for v in VARIANTS:
        for n, name in zip(counts, [f'{v}/records{n}' for n in conf['training_record_counts']] + [v]):
            for split in ('unseen_record', 'seen_record_reference', 'cross_database'):
                row = summary[name][split]['all']
                out['scaling'].setdefault(v, {}).setdefault(split, {})[n] = {k: mean_range(row[k]) for k in ('u_all_mse', 'a_all_mse', 'PIDR_max_mse')}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ('u_all_mse', 'a_all_mse', 'PIDR_max_mse'),
                              ('All-floor displacement MSE (m$^2$)', 'All-floor acceleration MSE ((m/s$^2$)$^2$)', 'Peak drift ratio MSE')):
        for v in VARIANTS:
            for split, style in (('unseen_record', '-o'), ('seen_record_reference', ':s'), ('cross_database', '--^')):
                pts = out['scaling'][v][split]
                y = [pts[n][key]['mean'] for n in counts]
                err = [[pts[n][key]['mean']-pts[n][key]['min'] for n in counts], [pts[n][key]['max']-pts[n][key]['mean'] for n in counts]]
                ax.errorbar(counts, y, yerr=err, fmt=style, ms=4, lw=1.2, capsize=2, color=COLORS[v],
                            label=f'{LABELS[v]} / {split.replace("_", " ")}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Training records (base ground motions)')
        ax.set_title(title)
    axes[0].legend(fontsize=6.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG/'scaling_curve.png', dpi=150)
    plt.close(fig)
    # Held-out structure groups and story counts (training-distribution stories come from the unseen-record part).
    meta = {}
    for part in ('unseen_record', 'structure_only', 'joint', 'cross_database', 'seen_record_reference'):
        for path in sorted((EXP/'runs/data'/data_conf['dataset_id']/'shards').glob(f'{part}_*.json')):
            for m in read(path)['cases']:
                meta[m['id']] = m
    for v in VARIANTS:
        for part in ('structure_only', 'joint'):
            for group in conf['holdout_groups']:
                vals = {k: [] for k in ('u_all_mse', 'a_all_mse', 'PIDR_max_mse')}
                count = 0
                for s in seeds:
                    rows = [r for r in reports[v][s]['splits'][part]['cases'] if meta[r['id']]['structure_group'] == group]
                    count = len(rows)
                    vals['u_all_mse'].append(np.mean([r['u_all_mse'] for r in rows]))
                    vals['a_all_mse'].append(np.mean([r['a_all_mse'] for r in rows]))
                    vals['PIDR_max_mse'].append(np.mean([(max(r['PIDR_pred'])-max(r['PIDR_true']))**2 for r in rows]))
                out['groups'].setdefault(v, {}).setdefault(part, {})[group] = {'count': count, **{k: mean_range(x) for k, x in vals.items()}}
        for part in ('unseen_record', 'joint'):
            for n in range(2, 15):
                vals = {k: [] for k in ('u_all_mse', 'a_all_mse', 'PIDR_max_mse')}
                count = 0
                for s in seeds:
                    rows = [r for r in reports[v][s]['splits'][part]['cases'] if r['stories'] == n
                            and meta[r['id']]['structure_group'] in ('training', 'story_interpolation', 'story_extrapolation')]
                    if not rows:
                        break
                    count = len(rows)
                    vals['u_all_mse'].append(np.mean([r['u_all_mse'] for r in rows]))
                    vals['a_all_mse'].append(np.mean([r['a_all_mse'] for r in rows]))
                    vals['PIDR_max_mse'].append(np.mean([(max(r['PIDR_pred'])-max(r['PIDR_true']))**2 for r in rows]))
                if count:
                    out['stories'].setdefault(v, {})[n] = {'count': count, 'part': part, **{k: mean_range(x) for k, x in vals.items()}}
    s_cfg = data_conf['structures']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title in zip(axes, ('u_all_mse', 'a_all_mse', 'PIDR_max_mse'),
                              ('All-floor displacement MSE (m$^2$)', 'All-floor acceleration MSE ((m/s$^2$)$^2$)', 'Peak drift ratio MSE')):
        ax.axvspan(10.5, 14.5, color='#f6e7d8', alpha=.6, lw=0)
        for n in s_cfg['interpolation_stories']:
            ax.axvspan(n-.4, n+.4, color='#e5eef8', alpha=.8, lw=0)
        for v in VARIANTS:
            ns = sorted(out['stories'][v])
            ax.errorbar(ns, [out['stories'][v][n][key]['mean'] for n in ns],
                        yerr=[[out['stories'][v][n][key]['mean']-out['stories'][v][n][key]['min'] for n in ns],
                              [out['stories'][v][n][key]['max']-out['stories'][v][n][key]['mean'] for n in ns]],
                        fmt='-o', ms=4, capsize=2, color=COLORS[v], label=LABELS[v])
        ax.set_yscale('log')
        ax.set_xticks(range(2, 15))
        ax.set_xlabel('Number of stories (blue band: interpolation, orange: extrapolation)')
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG/'stories.png', dpi=150)
    plt.close(fig)
    # Peak drift scatter, unseen records, shown seeds.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, v in zip(axes, VARIANTS):
        rows = reports[v][shown[v]]['splits']['unseen_record']['cases']
        t = np.array([max(r['PIDR_true']) for r in rows])
        p = np.array([max(r['PIDR_pred']) for r in rows])
        ax.scatter(t*100, p*100, s=5, alpha=.4, color=COLORS[v], lw=0)
        lim = [0.03, 6]
        ax.plot(lim, lim, color='#555', lw=.8)
        for th in data_conf['damage_thresholds']:
            ax.axvline(th*100, color='#999', lw=.5, ls=':')
            ax.axhline(th*100, color='#999', lw=.5, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('True peak drift ratio (%)')
        ax.set_title(f'{LABELS[v]}, unseen records (seed {shown[v]})')
    axes[0].set_ylabel('Predicted peak drift ratio (%)')
    fig.tight_layout()
    fig.savefig(FIG/'drift_scatter.png', dpi=150)
    plt.close(fig)
    # Learning curves of the shown seeds.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for v in VARIANTS:
        curve = read(EXP/'runs/stage2'/v/f'seed{shown[v]}'/'curve.json')
        steps = [r['step'] for r in curve]
        axes[0].plot(steps, [r['train_mse']['u_all_mse'] for r in curve], color=COLORS[v], ls='--', label=f'{LABELS[v]} train')
        axes[0].plot(steps, [r['development_mse']['u_all_mse'] for r in curve], color=COLORS[v], label=f'{LABELS[v]} development')
        axes[1].plot(steps, [r['train_mse']['a_all_mse'] for r in curve], color=COLORS[v], ls='--')
        axes[1].plot(steps, [r['development_mse']['a_all_mse'] for r in curve], color=COLORS[v])
    for ax, title in zip(axes, ('All-floor displacement MSE (m$^2$)', 'All-floor acceleration MSE ((m/s$^2$)$^2$)')):
        ax.set_yscale('log')
        ax.set_xlabel('Updates')
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG/'learning_curves.png', dpi=150)
    plt.close(fig)
    # Best / median / worst cases with both models overlaid.
    selections = [('unseen_record', None), ('joint', 'story_extrapolation'), ('cross_database', None)]
    for part, group in selections:
        cases, items, _ = load_part(data_conf, part)
        truth = {c['id']: (c, it) for c, it in zip(cases, items)}
        preds = {}
        for v in VARIANTS:
            with np.load(EXP/'runs/stage2'/v/f'seed{shown[v]}'/f'{part}_predictions.npz', allow_pickle=False) as f:
                # Each NpzFile access decompresses the whole array, so read every array once.
                u_all, a_all, ids_all = f['displacement'], f['acceleration'], f['case_ids']
            preds[v] = {str(cid): (u_all[i], a_all[i]) for i, cid in enumerate(ids_all)}
        rows = {r['id']: r for r in reports['exp5_decoder'][shown['exp5_decoder']]['splits'][part]['cases']}
        other = {r['id']: r for r in reports['cross_attention'][shown['cross_attention']]['splits'][part]['cases']}
        ids = [i for i in rows if group is None or meta[i]['structure_group'] == group]
        score = {}
        for i in ids:
            c, it = truth[i]
            n = c['stories']
            u, a = it['displacement'].numpy(), it['acceleration'].numpy()
            pu, pa = preds['exp5_decoder'][i][0][:len(u), :n], preds['exp5_decoder'][i][1][:len(a), :n]
            score[i] = .5*(np.mean((pu-u)**2)/np.mean(u**2) + np.mean((pa-a)**2)/np.mean(a**2))
        ranked = sorted(ids, key=score.get)
        picks = {'best': ranked[0], 'median': ranked[len(ranked)//2], 'worst': ranked[-1]}
        strong = [i for i in ranked if truth[i][0]['DS_ref'] >= 3]
        if strong:
            picks['worst_DS3'] = strong[-1]
        tag = part + ('_' + group if group else '')
        for label, cid in picks.items():
            c, it = truth[cid]
            n = c['stories']
            floors = sorted({n-1, (n-1)//2, 0}, reverse=True)
            t = np.arange(len(it['ground']))/data_conf['sample_rate_hz']
            fig, axes = plt.subplots(len(floors)+1, 2, figsize=(13, 1.9*(len(floors)+1)+.8), sharex=True)
            axes[0, 0].plot(t, it['ground'][:, 0].numpy(), color='#777', lw=.6)
            axes[0, 0].set_title('Ground acceleration input (m/s$^2$)')
            axes[0, 1].axis('off')
            mse_text = []
            for v, rr in (('exp5_decoder', rows[cid]), ('cross_attention', other[cid])):
                mse_text.append(f'{LABELS[v]}: MSE u={rr["u_all_mse"]:.3g} m$^2$, a={rr["a_all_mse"]:.3g} (m/s$^2$)$^2$, '
                                f'peak drift {max(rr["PIDR_pred"])*100:.2f}% (DS {rr["DS_pred"]})')
            axes[0, 1].text(0, .95, '\n'.join([f'True peak drift {max(rows[cid]["PIDR_true"])*100:.2f}% (DS {c["DS_ref"]})'] + mse_text),
                            va='top', fontsize=8, transform=axes[0, 1].transAxes)
            for j, f_ in enumerate(floors):
                for k, (key, unit) in enumerate((('displacement', 'm'), ('acceleration', 'm/s$^2$'))):
                    ax = axes[j+1, k]
                    ax.plot(t, it[key][:, f_].numpy(), color=COLORS['truth'], lw=1.0, label='MDOF reference')
                    for v in VARIANTS:
                        pv = preds[v][cid][0 if key == 'displacement' else 1][:len(t), f_]
                        ax.plot(t, pv, color=COLORS[v], lw=.7, alpha=.9, label=LABELS[v])
                    ax.set_ylabel(f'Floor {f_+1}/{n}\n{unit}')
                    if j == 0:
                        ax.set_title('Relative displacement' if key == 'displacement' else 'Absolute acceleration')
            axes[1, 0].legend(fontsize=7, loc='upper right')
            for ax in axes[-1]:
                ax.set_xlabel('Time (s)')
            fig.suptitle(f'{tag.replace("_", " ")} — {label.replace("_", " ")} case (ranked by exp5-style decoder): {cid}\n'
                         f'N={n}, fortification {c["fortification"]}, T1={c["T1_s"]:.3f} s, PGA={c["pga_g"]:.2f} g, '
                         f'Sa(T1)={c["sa_g"]:.2f} g, record {c["record_id"]}', fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, .95))
            fig.savefig(FIG/f'case_{tag}_{label}.png', dpi=130)
            plt.close(fig)
            out['cases'].setdefault(tag, {})[label] = {
                'id': cid, 'stories': n, 'fortification': c['fortification'], 'T1_s': c['T1_s'], 'pga_g': c['pga_g'],
                'sa_g': c['sa_g'], 'DS_ref': c['DS_ref'], 'normalized_error_exp5_decoder': float(score[cid]),
                **{v: {'u_all_mse': r['u_all_mse'], 'a_all_mse': r['a_all_mse'], 'PIDR_pred': max(r['PIDR_pred']), 'DS_pred': r['DS_pred']}
                   for v, r in (('exp5_decoder', rows[cid]), ('cross_attention', other[cid]))},
                'PIDR_true': max(rows[cid]['PIDR_true']), 'figure': f'case_{tag}_{label}.png'}
    write(EXP/'results/stage2/report_data.json', out)
    print('REPORT_DATA_COMPLETE', len(list(FIG.glob('*.png'))), 'figures', flush=True)


if __name__ == '__main__':
    main()
