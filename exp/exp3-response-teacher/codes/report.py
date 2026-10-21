"""Generate physical-unit MSE tables and actual waveform overlays on 5090."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import EXP, CHANNELS, read, write, load_data

METHODS = ['teacher', 'student', 'joint-matchedA', 'joint-matchedTotal']
LABELS = {'teacher': 'C+B reconstruction (TRUE responses supplied)', 'student': 'Two-stage A+B prediction',
          'joint-matchedA': 'Joint A+B: matched A updates', 'joint-matchedTotal': 'Joint A+B: matched total updates'}
COLORS = {'teacher': '#c98516', 'student': '#c83035', 'joint-matchedA': '#8455a5', 'joint-matchedTotal': '#2478ac'}
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2})


def load_predictions(protocol, split, b):
    predictions = {}
    for method in METHODS:
        with np.load(EXP / 'runs' / f'{protocol}-{method}' / f'{split}_predictions.npz', allow_pickle=False) as f:
            assert f['case_ids'].tolist() == b['case_ids']
            predictions[method] = {k: f[k].copy() for k in CHANNELS}
    return predictions


def waveform(path, b, pred, rows, indices, roof=True, methods=None, title=''):
    methods = methods or ['student', 'joint-matchedTotal']
    levels = [(i, int(b['attributes'][i, 0])) for i in indices] if roof else [(indices[0], f+1) for f in range(int(b['attributes'][indices[0], 0]))]
    fig, axes = plt.subplots(len(levels), 2, figsize=(13, 2.15*len(levels)+1.0), squeeze=False, sharex=True, constrained_layout=True)
    for j, (i, floor) in enumerate(levels):
        t = int(b['time_mask'][i].sum())
        times = np.arange(t) / 50.
        row = rows[i]
        for col, (key, prefix, unit) in enumerate(zip(CHANNELS, ('u', 'a'), ('m', 'm/s²'))):
            ax = axes[j, col]
            truth = b[key][i, :t, floor-1].cpu().numpy()
            ax.plot(times, truth, color='#252525', lw=1.05, label='C++ reference', alpha=.85)
            values = []
            for method in methods:
                value = pred[method][key][i, :t, floor-1]
                ax.plot(times, value, color=COLORS[method], lw=.8, alpha=.85, label=LABELS[method], linestyle='--' if method == 'teacher' else '-')
                mse = np.mean((value.astype('float64') - truth.astype('float64'))**2)
                short = {'student':'two-stage', 'teacher':'recon', 'joint-matchedA':'joint A', 'joint-matchedTotal':'joint total'}[method]
                values.append(f'{short} MSE={mse:.3g}')
            ax.set_title(f"{row['stories']}F floor {floor} | T ratio {row['period_ratio']:g} | PGA {row['pga_g']:.3g}g | DS {row['DS_ref']}\n" + '; '.join(values), fontsize=8)
            ax.set_ylabel(('Displacement ' if col == 0 else 'Absolute acceleration ') + f'({unit})')
            ax.set_xlim(0, times[-1])
    for ax in axes[-1]: ax.set_xlabel('Time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=12)
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    conf = read(EXP / 'configs/run.json')
    results, table = {}, []
    table += ['# exp3: response teacher experiment', '',
              'All reported errors are physical-unit MSE: displacement m²; acceleration (m/s²)². '
              'Average valid time/floors within each case, then give each case equal weight. '
              'C+B is reconstruction using true responses; only A+B is deployable prediction. Test split NOT READ.', '',
              '| Protocol | Split/subset | Method | Cases | Roof u MSE | Roof a MSE | All-floor u MSE | All-floor a MSE |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for protocol in conf['protocols']:
        available = [EXP / 'results' / f'{protocol}-{m}.json' for m in METHODS]
        if not all(p.exists() for p in available): continue
        results[protocol] = {m: read(p) for m, p in zip(METHODS, available)}
        train, held, entries, _, _ = load_data(conf, protocol)
        for split, batch in (('train', train), ('held', held)):
            predictions = load_predictions(protocol, split, batch)
            for subset in ('all', 'DS_gt_2'):
                if protocol == 'multiwave' and subset == 'all': continue
                for method in METHODS:
                    s = results[protocol][method]['splits'][split]['summary'][subset]
                    if not s['count']: continue
                    table.append(f"| {protocol} | {split}/{subset} | {method} | {s['count']} | " + ' | '.join(f'{s[k]:.6g}' for k in ('u_roof_mse', 'a_roof_mse', 'u_all_mse', 'a_all_mse')) + ' |')
            rows = results[protocol]['student']['splits'][split]['cases']
            if split == 'held':
                selected = list(range(len(rows)))
            else:
                selected = []
                # Predeclared train examples: median and worst acceleration MSE per height.
                for n in sorted({r['stories'] for r in rows}):
                    ordered = sorted([i for i, r in enumerate(rows) if r['stories'] == n], key=lambda i: rows[i]['a_all_mse'])
                    selected += [ordered[len(ordered)//2], ordered[-1]]
                selected = sorted(set(selected))
            for i in selected:
                root = EXP / 'figures' / protocol / split
                name = rows[i]['id']
                for roof, suffix in ((True, 'roof'), (False, 'floors')):
                    waveform(root / (name + '-' + suffix + '.png'), batch, predictions, rows, [i], roof,
                             title=f'{protocol} | {split} | {name} | prediction')
                waveform(root / (name + '-teacher.png'), batch, predictions, rows, [i], False, ['teacher', 'student'],
                         title=f'{protocol} | {split} | {name} | teacher uses TRUE responses')
            if split == 'held':
                chosen = []
                for n in sorted({r['stories'] for r in rows}):
                    candidates = [i for i, r in enumerate(rows) if r['stories'] == n and r['DS_ref'] > 2]
                    if candidates: chosen.append(max(candidates, key=lambda i: rows[i]['a_all_mse']))
                if len(chosen) == 1:
                    chosen = [i for i, r in enumerate(rows) if r['DS_ref'] > 2][:3]
                waveform(EXP / 'figures' / f'{protocol}-roof.png', batch, predictions, rows, chosen,
                         methods=['student', 'joint-matchedA', 'joint-matchedTotal'], title=f'{protocol}: held DS>2 roof predictions')
                n5 = next((i for i in chosen if rows[i]['stories'] == 5), chosen[0])
                waveform(EXP / 'figures' / f'{protocol}-floors.png', batch, predictions, rows, [n5], False,
                         title=f'{protocol}: held DS>2 all floors; {rows[n5]["id"]}')
                waveform(EXP / 'figures' / f'{protocol}-teacher-vs-student.png', batch, predictions, rows, [n5], False,
                         ['teacher', 'student'], title=f'{protocol}: reconstruction versus prediction (C reads TRUE responses)')
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for method in METHODS:
            curve = read(EXP / 'runs' / f'{protocol}-{method}' / 'curve.json')
            for ax, key in zip(axes.flat, ('u_roof_mse', 'a_roof_mse', 'u_all_mse', 'a_all_mse')):
                ax.semilogy([r['step'] for r in curve], [r['train_mse'][key] for r in curve], label=method, color=COLORS[method])
                ax.set_xlabel('Optimizer steps in this phase'); ax.set_ylabel(key.replace('_', ' '))
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(protocol + ': training MSE (teacher is reconstruction)')
        fig.savefig(EXP / 'figures' / (protocol + '-learning.png'), dpi=145); plt.close(fig)
    write(EXP / 'results/summary.json', {p: {m: {'seconds': r['seconds'], 'best_step': r['best_step'],
          'splits': {s: v['summary'] for s, v in r['splits'].items()}, 'checkpoint_sha256': r['checkpoint_sha256']}
          for m, r in rs.items()} for p, rs in results.items()})
    table += ['', 'Per-floor MSE, all case identifiers, audits and hashes are in the corresponding result JSON files.',
              'All held cases are plotted; train figures use median/worst acceleration MSE per height. '
              'The gallery chooses worst held acceleration MSE per height within DS>2; the stories gallery shows all three strong held cases.',
              'This is one seed and reused development validation; the final held records have not been opened.']
    (EXP / 'results/RESULTS.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    print('Reports generated for: ' + ', '.join(results), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
