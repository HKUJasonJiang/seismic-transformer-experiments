"""Audit a generated exp7 dataset (exp7.md section 5.7). Execution host only.

  python codes/datagen/audit.py
Writes results/data_audit/audit.json and 12 randomly chosen all-floor plots; exits nonzero on any failed rule.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

CODES = Path(__file__).resolve().parents[1]
EXP = CODES.parent
sys.path.insert(0, str(CODES))
from datagen.generate import periods, sha, read_json, write_json  # noqa: E402


def main():
    conf = read_json(EXP/'configs/data.json')
    root = EXP/'runs/data'/conf['dataset_id']
    manifest = read_json(root/'manifest.json')
    split = read_json(EXP/conf['record_split'])
    materials = read_json(EXP/conf['materials'])['materials']
    s, b = conf['structures'], conf['building']
    record_split = {r['id']: r['split'] for r in split['records']}
    failures, parts = [], {}
    sample_rng = np.random.default_rng(0)
    plotted = []
    for shard in manifest['shards']:
        path = root/'shards'/shard['npz']
        if sha(path) != shard['npz_sha256']:
            failures.append(f'hash mismatch {path.name}')
        meta = read_json(path.with_suffix('.json'))['cases']
        p = parts.setdefault(shard['part'], {'cases': [], 'shards': 0})
        p['cases'] += meta
        p['shards'] += 1
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        complete = [m for m in meta if m['status'] == 'complete']
        if [m['id'] for m in complete] != [str(x) for x in arrays['case_ids']]:
            failures.append(f'case order mismatch {path.name}')
        for key in ('ground', 'displacement', 'acceleration'):
            if not np.isfinite(arrays[key]).all() or arrays[key].shape[1] != conf['total_points']:
                failures.append(f'{key} nonfinite or wrong length in {path.name}')
        for i, m in enumerate(complete):
            n = m['stories']
            if int(arrays['stories'][i]) != n or np.abs(arrays['displacement'][i, :, n:]).max(initial=0) != 0:
                failures.append(f'floor padding/count mismatch {m["id"]}')
            mat = np.asarray(materials[f'type{b["structure_type"]}_N{n:02d}_IM{m["fortification"]}']['material'])
            t1 = periods(np.full(n, b['floor_area_m2']*b['mass_per_area_kg_m2']), mat[:, 0]*m['stiffness_coefficient'])[0]
            if not np.isclose(t1, m['T1_s'], rtol=1e-10) or not np.isclose(arrays['attributes'][i, 3], t1, rtol=1e-6):
                failures.append(f'T1 mismatch {m["id"]}')
        if len(plotted) < 12 and complete:
            i = int(sample_rng.integers(len(complete)))
            plotted.append((complete[i], {k: arrays[k][i] for k in ('ground', 'displacement', 'acceleration')}))
    summary = {}
    train_records = {c['record_id'] for c in parts['train']['cases']}
    for name, p in parts.items():
        cases = p['cases']
        ok = [c for c in cases if c['status'] == 'complete']
        groups = Counter(c['structure_group'] for c in cases)
        spec = conf['parts'][name]
        if spec['structures'] == 'training' and set(groups) != {'training'}:
            failures.append(f'{name} contains held-out structures')
        if spec['structures'] == 'holdout' and 'training' in groups:
            failures.append(f'{name} contains training-distribution structures')
        expected = {'train': 'train', 'train_first_300': 'train'}.get(spec['records'], spec['records'])
        if any(record_split[c['record_id']] != expected for c in cases):
            failures.append(f'{name} uses records outside its split')
        if expected != 'train' and {c['record_id'] for c in cases} & train_records:
            failures.append(f'{name} shares records with training')
        for c in cases:
            k, n = c['stiffness_coefficient'], c['stories']
            g = c['structure_group']
            in_gap = s['stiffness_gap'][0] <= k <= s['stiffness_gap'][1]
            if g == 'training' and (n not in s['training_stories'] or in_gap or not s['stiffness_range'][0] <= k <= s['stiffness_range'][1]):
                failures.append(f'training-distribution rule broken {c["id"]}')
            if g == 'story_interpolation' and n not in s['interpolation_stories']:
                failures.append(f'interpolation story rule broken {c["id"]}')
            if g == 'story_extrapolation' and n not in s['extrapolation_stories']:
                failures.append(f'extrapolation story rule broken {c["id"]}')
            if g == 'period_interpolation' and not in_gap:
                failures.append(f'period interpolation rule broken {c["id"]}')
            if g == 'period_extrapolation' and not any(np.isclose(k, v) for v in s['stiffness_extrapolation']):
                failures.append(f'period extrapolation rule broken {c["id"]}')
        ds = Counter(c['DS_ref'] for c in ok)
        summary[name] = {'cases': len(cases), 'complete': len(ok), 'shards': p['shards'],
                         'status_counts': dict(Counter(c['status'] for c in cases)),
                         'structure_groups': dict(groups), 'records': len({c['record_id'] for c in cases}),
                         'DS_counts_complete': {str(k): ds[k] for k in sorted(ds)},
                         'stories_complete': {str(k): v for k, v in sorted(Counter(c['stories'] for c in ok).items())},
                         'PGA_g_quantiles': np.quantile([c['pga_g'] for c in ok], [0, .1, .5, .9, 1]).round(4).tolist(),
                         'T1_s_quantiles': np.quantile([c['T1_s'] for c in ok], [0, .1, .5, .9, 1]).round(4).tolist()}
    out = EXP/'results/data_audit'
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for m, a in plotted:
        n = m['stories']
        fig, axes = plt.subplots(n, 2, figsize=(12, 1.6*n+1), squeeze=False)
        t = np.arange(conf['total_points'])/conf['sample_rate_hz']
        for j in range(n):
            axes[j, 0].plot(t, a['displacement'][:, j], lw=.7)
            axes[j, 1].plot(t, a['acceleration'][:, j], lw=.7)
            axes[j, 1].plot(t, -a['ground'], lw=.5, alpha=.5)
            axes[j, 0].set_ylabel(f'F{j+1}')
        axes[0, 0].set_title('displacement (m)')
        axes[0, 1].set_title('absolute acceleration (m/s2); faint: -ground')
        fig.suptitle(f'{m["id"]} | N={n} IM={m["fortification"]} T1={m["T1_s"]:.3f}s PGA={m["pga_g"]:.3f}g DS={m["DS_ref"]}', fontsize=9)
        fig.tight_layout()
        fig.savefig(out/f'{m["id"]}.png', dpi=110)
        plt.close(fig)
    result = {'status': 'PASS' if not failures else 'FAIL', 'failures': failures[:200], 'failure_count': len(failures),
              'manifest_sha256': sha(root/'manifest.json'), 'parts': summary, 'plots': [m['id'] for m, _ in plotted]}
    write_json(out/'audit.json', result)
    print(json.dumps({'status': result['status'], 'failure_count': len(failures),
                      'parts': {k: {x: v[x] for x in ('cases', 'complete', 'DS_counts_complete')} for k, v in summary.items()}}, indent=1))
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    main()
