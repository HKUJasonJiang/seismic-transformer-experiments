"""5090-only plots for one variant/seed: every test case, all physical floors and roof separately."""
import argparse
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import EXP, CHANNELS, read, write, load_split, resolve_config

TITLES = {'period': 'Unseen period configuration / known earthquake records',
          'earthquake': 'Unseen earthquake records / known structural configurations',
          'train': 'Training examples'}


def plot_case(row, item, pred, split, label, roof=False):
    n = row['stories']
    fs = [n-1] if roof else list(range(n))
    fig, axes = plt.subplots(len(fs), 2, figsize=(13, 2.1*len(fs)+1), squeeze=False)
    t = np.arange(len(item['ground']))/50.
    for j, f in enumerate(fs):
        for k, (channel, prefix, unit) in enumerate(zip(CHANNELS, ('u', 'a'), ('m', 'm/s2'))):
            ax = axes[j, k]
            ax.plot(t, item[channel][:, f], lw=.9, color='#243447', label='MDOF reference')
            ax.plot(t, pred[channel][:, f], lw=.8, color='#e16a2d', label=label)
            ax.set_title(f'Floor {f+1}/{n}, xi={(f+1)/n:.3f}: {channel}, MSE={row[prefix+"_floor_mse"][f]:.4g}', fontsize=9)
            ax.set_ylabel(unit)
            ax.grid(alpha=.2)
            if j == len(fs)-1:
                ax.set_xlabel('Time (s)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{TITLES[split]}\n{row["id"]} | T1={row["T1_s"]:.3f}s | reference DS={row["DS_ref"]}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93 if roof else .96))
    return fig


def save(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='full')
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    conf = resolve_config(read(EXP/'configs/run.json'), args.variant)
    run_dir = EXP/'runs'/args.variant/f'seed{args.seed}'
    out_dir = EXP/'results'/args.variant/f'seed{args.seed}'
    fig_dir = EXP/'figures'/f'{args.variant}_seed{args.seed}'
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = read(out_dir/'report.json')
    label = f'exp7 {args.variant}'
    selected, figure_count = {}, 0
    for split in ('train', 'period', 'earthquake'):
        cs, items, _ = load_split(split, conf)
        itemmap = {c['id']: it for c, it in zip(cs, items)}
        rows = report['splits'][split]['cases']
        rowmap = {r['id']: r for r in rows}
        with np.load(run_dir/f'{split}_predictions.npz', allow_pickle=False) as f:
            predictions = {str(name): {k: f[k][i, :len(itemmap[str(name)]['ground']), :rowmap[str(name)]['stories']] for k in CHANNELS}
                           for i, name in enumerate(f['case_ids'])}
        # Same predeclared representatives as exp5: never chosen by prediction quality.
        groups = ['peer_9', 'peer_10'] if split == 'earthquake' else ['peer_1']
        ratio = 1.0 if split == 'period' else 1.2
        selected[split] = []
        for rec in groups:
            for n in (3, 5, 8):
                eligible = [r for r in rows if r['record_id'] == rec and r['stories'] == n and abs(r['period_ratio']-ratio) < 1e-6]
                selected[split].append(max(eligible, key=lambda r: r['pga_g'])['id'])
        worst = {key: max(rows, key=lambda r: r[key])['id'] for key in ('u_all_mse', 'a_all_mse')}
        (fig_dir/split).mkdir(exist_ok=True)
        for i, r in enumerate(rows):
            if split == 'train' and r['id'] not in selected[split] and r['id'] not in worst.values():
                continue
            for roof in (True, False):
                suffix = 'roof' if roof else 'floors'
                path = fig_dir/split/f'{r["id"]}_{suffix}.png'
                save(plot_case(r, itemmap[r['id']], predictions[r['id']], split, label, roof), path)
                figure_count += 1
                if r['id'] in selected[split]:
                    shutil.copy2(path, fig_dir/f'{split}_{r["record_id"]}_{r["stories"]}F_{suffix}.png')
                for key, name in worst.items():
                    if r['id'] == name:
                        shutil.copy2(path, fig_dir/f'{split}_worst_{key}_{suffix}.png')
            if (i+1) % 20 == 0:
                print(f'{split}: figures {i+1}/{len(rows)}', flush=True)
        for rec in groups:
            fig, axes = plt.subplots(3, 2, figsize=(13, 8))
            for j, n in enumerate((3, 5, 8)):
                r = next(rowmap[name] for name in selected[split] if rowmap[name]['record_id'] == rec and rowmap[name]['stories'] == n)
                it, p = itemmap[r['id']], predictions[r['id']]
                t = np.arange(len(it['ground']))/50.
                for k, (channel, prefix, unit) in enumerate(zip(CHANNELS, ('u', 'a'), ('m', 'm/s2'))):
                    ax = axes[j, k]
                    ax.plot(t, it[channel][:, -1], lw=.9, color='#243447', label='MDOF reference')
                    ax.plot(t, p[channel][:, -1], lw=.8, color='#e16a2d', label=label)
                    ax.set_title(f'{n}F roof, T1={r["T1_s"]:.3f}s, PGA={r["pga_g"]:.3f}g, DS={r["DS_ref"]}\nMSE={r[prefix+"_roof_mse"]:.4g}', fontsize=9)
                    ax.set_ylabel(unit)
                    ax.grid(alpha=.2)
                    if j == 2:
                        ax.set_xlabel('Time (s)')
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(TITLES[split]+f'\n{rec}, ratio={ratio}; highest valid PGA per story count', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .93))
            save(fig, fig_dir/f'{split}_{rec}_roof_gallery.png')
        selected[split] = {'representative_case_ids': selected[split], 'worst_cases': worst}
    curve = read(run_dir/'curve.json')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(axes, ('u_all_mse', 'a_all_mse', 'objective'),
                              ('All-floor displacement MSE (m2)', 'All-floor acceleration MSE ((m/s2)2)', 'Normalized training objective')):
        for name in ('train', 'development'):
            y = [r[name+'_objective'] if key == 'objective' else r[name+'_mse'][key] for r in curve]
            ax.semilogy([r['step'] for r in curve], y, label=name)
        ax.axvline(report['best_step'], color='gray', ls='--')
        ax.set_xlabel('Updates')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    save(fig, fig_dir/'learning.png')
    write(out_dir/'figure_selection.json', {'selection': selected, 'unique_case_plot_count': figure_count,
                                            'all_png_count': len(list(fig_dir.rglob('*.png')))})
    print('REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
