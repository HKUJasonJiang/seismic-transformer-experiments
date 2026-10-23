"""exp7 dataset generator (exp7.md section 5). Execution host only.

Environment: EXP7_KNET_DIR and EXP7_PEER_DIR point at the processed text-record folders; the mdof-workbench
package must be importable (installed, or MDOF_WORKBENCH points at its checkout).

  python codes/datagen/generate.py preflight          # step size, ground-term identity, alignment, cost
  python codes/datagen/generate.py calibrate          # proposes the Sa(T1) range; write it into configs/data.json after review
  python codes/datagen/generate.py plan               # deterministic case list (needs intensity.sa_range_g)
  python codes/datagen/generate.py run [--parts a,b]  # simulate and write shards; resumable shard by shard
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

CODES = Path(__file__).resolve().parents[1]
EXP = CODES.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(CODES))
sys.path.insert(0, str(Path(os.environ.get('MDOF_WORKBENCH', REPO.parent/'mdof-workbench'))/'src'))
from datagen.records import read_library, number  # noqa: E402
from mdof_workbench.api import run_case  # noqa: E402
from mdof_workbench.response_spectra import compute_response_spectra  # noqa: E402

G = 9.80665
STATE = {}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def output_root(conf):
    return EXP/'runs/data'/conf['dataset_id']


def periods(mass, k):
    b = np.eye(len(k)) - np.eye(len(k), k=-1)
    stiffness = b.T @ np.diag(k) @ b
    invsqrt = 1/np.sqrt(mass)
    eig = np.linalg.eigvalsh(stiffness*invsqrt[:, None]*invsqrt[None, :])
    # Same pi literal as the C++ kernel, so T1 can be compared at 1e-10.
    return 2*float(np.float32(3.1415926))/np.sqrt(eig)


def load_state(conf):
    """Records, split and materials, loaded once in the parent and inherited by forked workers."""
    split = read_json(EXP/conf['record_split'])
    records = {}
    for folder, prefix in ((os.environ['EXP7_KNET_DIR'], 'knet'), (os.environ['EXP7_PEER_DIR'], 'peer')):
        for r in read_library(folder, prefix):
            records[r['id']] = r
    for row in split['records']:
        if records[row['id']]['sha256'] != row['sha256']:
            raise ValueError(f'Record hash differs from the frozen split: {row["id"]}')
    materials = read_json(EXP/conf['materials'])
    expected = {f'type{conf["building"]["structure_type"]}_N{n:02d}_IM{im}'
                for n in conf['structures']['training_stories'] + conf['structures']['interpolation_stories']
                + conf['structures']['extrapolation_stories'] for im in conf['structures']['fortification_levels']}
    missing = expected - set(materials['materials'])
    if missing:
        raise ValueError(f'materials.json lacks {sorted(missing)}')
    STATE.update(conf=conf, split=split, records=records, materials=materials['materials'],
                 materials_sha256=sha(EXP/conf['materials']))


def record_ids(selection):
    split = STATE['split']
    if selection == 'train':
        return list(split['training_order'])
    if selection == 'train_first_300':
        return list(split['training_order'][:300])
    ids = [r['id'] for r in split['records'] if r['split'] == selection]
    return sorted(ids, key=lambda x: (x.split('_')[0], number(x)))


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def training_stiffness(rng, s):
    lo, hi = s['stiffness_range']
    g0, g1 = s['stiffness_gap']
    a, b = np.log(g0/lo), np.log(hi/g1)
    u = rng.uniform(0., a+b)
    return float(lo*np.exp(u)) if u < a else float(g1*np.exp(u-a))


def draw_structure(rng, s, kind):
    if kind in ('training', 'period_interpolation', 'period_extrapolation'):
        stories = s['training_stories']
    else:
        stories = s['interpolation_stories'] if kind == 'story_interpolation' else s['extrapolation_stories']
    n = int(rng.choice(stories))
    if kind == 'period_interpolation':
        k = log_uniform(rng, *s['stiffness_gap'])
    elif kind == 'period_extrapolation':
        k = float(rng.choice(s['stiffness_extrapolation']))
    else:
        k = training_stiffness(rng, s)
    return {'stories': n, 'fortification': int(rng.choice(s['fortification_levels'])), 'stiffness_coefficient': k,
            'structure_group': kind}


def solver_ground(time, gt, ga):
    """Ground acceleration the kernel subtracts in its absolute-acceleration output (api.cpp output_ag).

    Linear interpolation on [t_k, t_k+1); at and after the final supplied sample the compatibility
    output keeps the last interpolated value."""
    g = np.interp(time, gt, ga)
    after = time >= gt[-1]
    if after.any():
        g[after] = g[np.flatnonzero(~after)[-1]]
    return g


def simulate(case, dt=None, keep_grid=False):
    conf, b = STATE['conf'], STATE['conf']['building']
    n = case['stories']
    base = np.asarray(STATE['materials'][f'type{b["structure_type"]}_N{n:02d}_IM{case["fortification"]}']['material'], dtype=np.float64)
    material = base.copy()
    material[:, :2] *= case['stiffness_coefficient']
    mass = np.full(n, b['floor_area_m2']*b['mass_per_area_kg_m2'])
    heights = np.full(n, b['story_height_m'])
    t = periods(mass, material[:, 0])
    record = STATE['records'][case['record_id']]
    rt, ra = record['time'], record['acceleration']
    sa_raw = float(compute_response_spectra(rt, ra, [t[0]], damping=b['damping_ratio']).Sa[0])
    scale = case['target_sa_g']*G/sa_raw
    ground = ra*scale
    zero_time = round(float(rt[-1] + (rt[1]-rt[0])), 10)
    gt, ga = np.r_[rt, zero_time], np.r_[ground, 0.]
    step = dt or conf['integration_dt_s']
    config = {
        'schema_version': 1, 'name': case['id'],
        'model': {'type': 'shear', 'mass': mass.tolist(), 'height': heights.tolist(), 'material': material.tolist(),
                  'collapse_drift': [conf['solver']['explicit_failure_drift']]*n},
        'ground_motion': {'time': gt.tolist(), 'acceleration': ga.tolist(), 'record_id': case['record_id']},
        'analysis': {'dt': step, 'duration': conf['total_points']/conf['sample_rate_hz'], 'damping_ratio': b['damping_ratio'],
                     'integrator': 'central_difference', 'alignment': conf['solver']['alignment'],
                     'compatibility': conf['solver']['compatibility']},
        'output': {},
        'provenance': {'dataset_id': conf['dataset_id'], 'record_sha256': record['sha256'],
                       'materials_sha256': STATE['materials_sha256'], 'case': case}}
    start = time.perf_counter()
    r = run_case(config)
    wall = time.perf_counter() - start
    if not np.isclose(r.periods[0], t[0], rtol=1e-10):
        raise ValueError('Independent T1 check failed')
    grid_ground = solver_ground(r.time, gt, ga)
    displacement = r.displacement
    if conf['solver']['alignment'] == 'legacy':
        # Legacy rows store u(t_{i+1}); shift so row i is u(t_i), matching the acceleration rows (as in stage 1).
        displacement = np.vstack((np.zeros((1, n)), displacement[:-1]))
    factor = round(1/(conf['sample_rate_hz']*step))
    if abs(factor*step*conf['sample_rate_hz'] - 1) > 1e-9:
        raise ValueError('Integration step must divide the output interval')

    def down(x):
        # Ground and responses share one anti-alias path, so absolute + ground = relative holds at 50 Hz.
        return resample_poly(x, 1, factor, axis=0, padtype='line')[:conf['total_points']]
    u, a, g = (down(x).astype(np.float32) for x in (displacement, r.acceleration, grid_ground))
    # Damage grade from the stored float32 values, so training-time recomputation agrees exactly.
    stored = u.astype(np.float64)
    drift = np.diff(stored, axis=1, prepend=np.zeros((len(u), 1)))/heights.astype(np.float32).astype(np.float64)
    pidr = float(np.abs(drift).max())
    dead = bool(r.material_dead.any())
    finite = all(np.isfinite(x).all() for x in (u, a, g))
    meta = {**case, 'T1_s': float(t[0]), 'T2_s': float(t[1]) if n > 1 else None, 'T3_s': float(t[2]) if n > 2 else None,
            'period_ratio': float(case['stiffness_coefficient']**-.5),
            'yield_base_shear_coefficient': float(material[0, 1]/(mass.sum()*G)),
            'sa_raw_m_s2': sa_raw, 'scale_factor': float(scale), 'sa_g': case['target_sa_g'],
            'pga_g': float(np.abs(ground).max()/G), 'max_PIDR_50Hz': pidr,
            'max_PIDR_integration': float(r.edp['maximum_interstory_drift_ratio']),
            'DS_ref': int(np.sum(pidr > np.asarray(conf['damage_thresholds']))),
            'material_failure': dead,
            'status': 'complete' if finite and not dead else 'excluded_material_failure' if dead else 'excluded_nonfinite',
            'wall_seconds': wall, 'config_hash': r.metadata['config_hash'], 'model_version': r.metadata['model_version']}
    arrays = {'ground': g, 'displacement': u, 'acceleration': a,
              'story_heights': heights.astype(np.float32),
              'attributes': np.asarray([n, heights.sum(), case['fortification'], t[0]], dtype=np.float32)}
    if keep_grid:
        return meta, arrays, r, grid_ground, displacement
    return meta, arrays


def worker(case):
    try:
        return simulate(case)
    except Exception as error:  # one failed case must not stop a shard; it is recorded, not hidden
        return {**case, 'status': 'solver_error', 'error': f'{type(error).__name__}: {error}'}, None


def pool(workers):
    return mp.get_context('fork').Pool(workers)


def run_cases(cases, workers):
    with pool(workers) as p:
        return p.map(worker, cases, chunksize=2)


def max_floors(conf):
    s = conf['structures']
    return max(s['training_stories'] + s['interpolation_stories'] + s['extrapolation_stories'])


def write_shard(path, results, conf):
    ok = [(m, a) for m, a in results if m['status'] == 'complete']
    count, t, f = len(ok), conf['total_points'], max_floors(conf)
    out = {'ground': np.zeros((count, t), np.float32), 'displacement': np.zeros((count, t, f), np.float32),
           'acceleration': np.zeros((count, t, f), np.float32), 'story_heights': np.zeros((count, f), np.float32),
           'attributes': np.zeros((count, 4), np.float32), 'stories': np.zeros(count, np.int64),
           'structure_type': np.full(count, conf['building']['structure_type'], np.int64),
           'case_ids': np.asarray([m['id'] for m, _ in ok])}
    for i, (m, a) in enumerate(ok):
        n = m['stories']
        out['ground'][i] = a['ground']
        out['displacement'][i, :, :n] = a['displacement']
        out['acceleration'][i, :, :n] = a['acceleration']
        out['story_heights'][i, :n] = a['story_heights']
        out['attributes'][i] = a['attributes']
        out['stories'][i] = n
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    write_json(path.with_suffix('.json'), {'npz': path.name, 'npz_sha256': sha(path), 'cases': [m for m, _ in results]})


def make_plan(conf):
    lo_hi = conf['intensity']['sa_range_g']
    if not lo_hi:
        raise ValueError('intensity.sa_range_g is not frozen; run calibrate and record the reviewed range first')
    s, seed = conf['structures'], conf['record_split_policy']['seed']
    cases = []
    for index, (part, spec) in enumerate(conf['parts'].items()):
        rng = np.random.default_rng([seed, index])
        for rid in record_ids(spec['records']):
            kinds = (['training']*spec['per_record'] if spec['structures'] == 'training'
                     else [g for g in conf['holdout_groups'] for _ in range(spec['per_holdout_group'])])
            for k, kind in enumerate(kinds):
                cases.append({'id': f'{part}_{rid}_{k:02d}', 'part': part, 'record_id': rid,
                              **draw_structure(rng, s, kind), 'target_sa_g': log_uniform(rng, *lo_hi)})
    return cases


def preflight(conf, workers):
    out = {}
    split = STATE['split']
    rid = split['training_order'][0]
    probes = [{'id': 'probe_N14', 'part': 'preflight', 'record_id': rid, 'stories': 14, 'fortification': 8,
               'stiffness_coefficient': 1.0, 'structure_group': 'probe', 'target_sa_g': 2.0},
              {'id': 'probe_N02_stiff', 'part': 'preflight', 'record_id': rid, 'stories': 2, 'fortification': 8,
               'stiffness_coefficient': 2.1, 'structure_group': 'probe', 'target_sa_g': 2.0}]
    for probe in probes:
        meta, arrays, r, grid_ground, displacement = simulate(probe, keep_grid=True)
        fine_meta, fine = simulate(probe, dt=conf['integration_dt_s']/2)
        dt = conf['integration_dt_s']
        # On the integration grid, absolute + ground must equal the second difference of the aligned displacement.
        relative = r.acceleration + grid_ground[:, None]
        second = (displacement[2:] - 2*displacement[1:-1] + displacement[:-2])/dt**2
        identity = {f'shift_{s}': float(np.abs(relative[1+s:len(relative)-1+s] - second).max()/np.abs(relative).max())
                    for s in (-1, 0, 1)}

        def rel(x, y):
            return float(np.abs(x-y).max()/np.abs(y).max())
        out[probe['id']] = {
            'meta': meta, 'step_halving_relative_difference': {k: rel(fine[k], arrays[k]) for k in ('displacement', 'acceleration')},
            'step_halving_PIDR': [meta['max_PIDR_50Hz'], fine_meta['max_PIDR_50Hz']],
            'relative_acceleration_identity_error': identity}
    s = conf['structures']
    out['T1_table_s'] = {key: float(periods(np.full(v['stories'], conf['building']['floor_area_m2']*conf['building']['mass_per_area_kg_m2']),
                                            np.asarray(v['material'])[:, 0])[0]) for key, v in STATE['materials'].items()}
    rng = np.random.default_rng(7)
    lo, hi = conf['intensity']['calibration']['search_range_g']
    sample = [{'id': f'cost_{i:03d}', 'part': 'preflight', 'record_id': rid_, **draw_structure(rng, s, 'training'),
               'target_sa_g': log_uniform(rng, lo, hi)}
              for i, rid_ in enumerate(split['training_order'][:200])]
    start = time.perf_counter()
    results = run_cases(sample, workers)
    elapsed = time.perf_counter() - start
    planned = sum((spec['per_record'] if spec['structures'] == 'training' else spec['per_holdout_group']*len(conf['holdout_groups']))
                  * len(record_ids(spec['records'])) for spec in conf['parts'].values())
    walls = [m['wall_seconds'] for m, _ in results if 'wall_seconds' in m]
    out['cost'] = {'cases': len(sample), 'workers': workers, 'elapsed_s': elapsed, 'mean_case_wall_s': float(np.mean(walls)),
                   'p90_case_wall_s': float(np.quantile(walls, .9)), 'planned_cases': planned,
                   'estimated_total_hours': planned*elapsed/len(sample)/3600,
                   'statuses': {k: sum(m['status'] == k for m, _ in results) for k in {m['status'] for m, _ in results}},
                   'estimated_npz_gigabytes_uncompressed': planned*conf['total_points']*(1+2*max_floors(conf))*4/1e9}
    write_json(EXP/'results/preflight/preflight.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'T1_table_s'}, indent=1, default=str)[:6000], flush=True)


def calibrate(conf, workers):
    c = conf['intensity']['calibration']
    s, seed = conf['structures'], conf['record_split_policy']['seed']
    rng = np.random.default_rng([seed, 99])
    cases = [{'id': f'calibration_{rid}_{k}', 'part': 'calibration', 'record_id': rid, **draw_structure(rng, s, 'training'),
              'target_sa_g': log_uniform(rng, *c['search_range_g'])}
             for rid in STATE['split']['training_order'][:c['records']] for k in range(c['cases_per_record'])]
    rows = [m for m, _ in run_cases(cases, workers)]
    target = np.array([c['target_fractions'][k] for k in ('DS_0_1', 'DS_2', 'DS_3_up')])
    grid = np.exp(np.linspace(*np.log(c['search_range_g']), 61))
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i+1:]:
            inside = [m for m in rows if lo <= m['target_sa_g'] <= hi]
            valid = [m for m in inside if m['status'] == 'complete']
            if hi/lo < 2 or len(valid) < 60:
                continue
            ds = np.array([m['DS_ref'] for m in valid])
            fractions = np.array([(ds <= 1).mean(), (ds == 2).mean(), (ds >= 3).mean()])
            failure = 1 - len(valid)/len(inside)
            score = float(np.abs(fractions-target).sum() + failure)
            if best is None or score < best['score']:
                best = {'sa_range_g': [float(lo), float(hi)], 'score': score, 'fractions': fractions.tolist(),
                        'failure_fraction': failure, 'cases_inside': len(inside)}
    write_json(EXP/'results/preflight/calibration.json', {'proposal': best, 'rule': 'minimize L1(fractions-target) + failure fraction',
                                                         'target': c['target_fractions'], 'cases': rows})
    print(json.dumps({'proposal': best}, indent=1), flush=True)


def run(conf, parts, workers):
    root = output_root(conf)
    plan = read_json(root/'plan.json')
    if plan['config_sha256'] != sha(EXP/'configs/data.json'):
        raise ValueError('configs/data.json changed after planning; create a new dataset_id instead')
    (root/'shards').mkdir(parents=True, exist_ok=True)
    size = conf['shard_size']
    for part in parts:
        cases = [c for c in plan['cases'] if c['part'] == part]
        for k in range(math.ceil(len(cases)/size)):
            path = root/'shards'/f'{part}_{k:04d}.npz'
            if path.with_suffix('.json').exists():
                continue
            start = time.perf_counter()
            results = run_cases(cases[k*size:(k+1)*size], workers)
            write_shard(path, results, conf)
            print(json.dumps({'part': part, 'shard': k, 'cases': len(results), 'seconds': round(time.perf_counter()-start, 1),
                              'complete': sum(m['status'] == 'complete' for m, _ in results)}), flush=True)
    shards = []
    for meta_path in sorted((root/'shards').glob('*.json')):
        meta = read_json(meta_path)
        shards.append({'part': meta_path.stem.rsplit('_', 1)[0], 'npz': meta['npz'], 'npz_sha256': meta['npz_sha256'],
                       'complete': sum(m['status'] == 'complete' for m in meta['cases']), 'cases': len(meta['cases'])})
    write_json(root/'manifest.json', {'dataset_id': conf['dataset_id'], 'config_sha256': plan['config_sha256'],
                                      'plan_sha256': sha(root/'plan.json'), 'record_split_sha256': sha(EXP/conf['record_split']),
                                      'materials_sha256': STATE['materials_sha256'], 'shards': shards})
    print('RUN_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'calibrate', 'plan', 'run'))
    parser.add_argument('--parts', default='')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    conf = read_json(EXP/'configs/data.json')
    load_state(conf)
    workers = args.workers or conf['workers']
    if args.command == 'preflight':
        preflight(conf, workers)
    elif args.command == 'calibrate':
        calibrate(conf, workers)
    elif args.command == 'plan':
        root = output_root(conf)
        if (root/'plan.json').exists():
            raise RuntimeError('plan.json exists; the plan is frozen for this dataset_id')
        cases = make_plan(conf)
        write_json(root/'plan.json', {'config_sha256': sha(EXP/'configs/data.json'), 'cases': cases})
        print(json.dumps({p: sum(c['part'] == p for c in cases) for p in conf['parts']}), flush=True)
    else:
        run(conf, [p for p in args.parts.split(',') if p] or list(conf['parts']), workers)


if __name__ == '__main__':
    main()
