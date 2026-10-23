"""Immutable data selection (same splits as exp5), configuration variants, output scales and SI-unit metrics."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.dataset import collate_responses
from legacy_metrics import Objective, metrics, file_hash, state_hash, read, write
from model import input_batch

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
CHANNELS = ('displacement', 'acceleration')


def resolve_config(base, variant):
    """Model = chosen preset updated by the variant's overrides; other keys are shared by all variants."""
    if variant not in base['variants']:
        raise ValueError(f'Unknown variant {variant}; declared: {sorted(base["variants"])}')
    conf = copy.deepcopy(base)
    model = dict(conf['model_presets'][conf['model_preset']])
    unknown = set(conf['variants'][variant]) - set(model)
    if unknown:
        raise ValueError(f'Variant overrides undeclared model keys: {sorted(unknown)}')
    model.update(conf['variants'][variant])
    for key in ('variants', 'model_presets'):
        conf.pop(key)
    conf.update(model=model, variant=variant)
    return conf


def read_selected(path, predicate):
    path = Path(path)
    meta = read(path)
    cases = [c for c in meta['cases'] if c['status'] == 'complete' and predicate(c)]
    items = []
    for c in cases:
        p = path.parent/c['file']
        if file_hash(p) != c['npz_sha256']:
            raise ValueError(f'NPZ hash mismatch: {c["id"]}')
        with np.load(p, allow_pickle=False) as f:
            item = {k: torch.from_numpy(f[k].copy()) for k in f.files}
        item['case_id'] = c['id']
        items.append(item)
    if not items:
        raise ValueError('Empty declared data split')
    return cases, items, meta


def known_ratio(c, conf):
    return any(abs(c['period_ratio']-v) < 1e-6 for v in conf['training_period_ratios'])


def split_predicate(name, conf):
    def choose(c):
        if c['stories'] not in conf['stories']:
            return False
        if name == 'period':
            return c['record_id'] in conf['training_records'] and abs(c['period_ratio']-conf['period_test_ratio']) < 1e-6
        records = conf[{'train': 'training_records', 'development': 'development_records', 'earthquake': 'earthquake_test_records'}[name]]
        return c['record_id'] in records and known_ratio(c, conf)
    return choose


def load_split(name, conf):
    if conf['dataset_format'] == 'exp7_shards':
        from datagen.shards import load_part
        data_conf = read(EXP/conf['data_config'])
        records = None
        if name == 'train' and conf.get('training_record_count'):
            # Nested prefixes of the frozen training order (record-count scaling curve).
            records = read(EXP/data_conf['record_split'])['training_order'][:conf['training_record_count']]
        return load_part(data_conf, name, records)
    if conf['dataset_format'] != 'peer10_npz':
        raise NotImplementedError(f'Unknown dataset_format {conf["dataset_format"]}')
    manifest = {'train': 'train', 'period': 'train', 'development': 'validation', 'earthquake': 'test'}[name]
    return read_selected(REPO/conf['dataset']/f'{manifest}.json', split_predicate(name, conf))


def batch(items, device):
    b = collate_responses(items)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}


def take(b, index):
    ids = index.detach().cpu().tolist()
    return {k: v[index] if torch.is_tensor(v) else [v[i] for i in ids] for k, v in b.items()}


def masked_mean_square(value, mask):
    return torch.where(mask, value.double().square(), 0.).sum((1, 2))/mask.sum((1, 2))


@torch.no_grad()
def fit_output_scales(b, conf):
    """Training-only RMS of the quantities the decoder emits, after optional PGA division."""
    m = conf['model']
    tm, fm = b['time_mask'], b['floor_mask']
    mask = tm[:, :, None] & fm[:, None, :]
    ground = torch.where(tm, b['ground'][..., 0], 0.)
    amplitude = ground.abs().amax(1).clamp_min(1e-6) if m['amplitude_factorization'] else torch.ones_like(ground[:, 0])
    relative = b['acceleration'] - m['ground_sign']*ground[:, :, None]
    opposite = b['acceleration'] + m['ground_sign']*ground[:, :, None]
    emitted = relative if m['ground_residual'] else b['acceleration']
    scales = [max(float(masked_mean_square(v/amplitude[:, None, None], mask).mean().sqrt()), 1e-8)
              for v in (b['displacement'], emitted)]
    absolute_ms = masked_mean_square(b['acceleration'], mask)
    relative_ratio = (masked_mean_square(relative, mask)/absolute_ms).sqrt()
    opposite_ratio = (masked_mean_square(opposite, mask)/absolute_ms).sqrt()
    # The configured sign must make the relative signal smaller than the opposite-sign combination.
    if not float(relative_ratio.median()) < float(opposite_ratio.median()):
        raise ValueError('ground_sign is inconsistent with the stored absolute acceleration')
    return scales, {'method': 'per-building RMS over valid time/floors, buildings averaged; training split only',
                    'amplitude_factorization': m['amplitude_factorization'], 'ground_residual': m['ground_residual'],
                    'relative_over_absolute_acceleration_rms_median': float(relative_ratio.median()),
                    'opposite_sign_over_absolute_acceleration_rms_median': float(opposite_ratio.median())}


@torch.no_grad()
def evaluate(model, b, conf, scales):
    output = {k: [] for k in CHANNELS}
    for start in range(0, len(b['case_ids']), conf['batch_size']):
        sub = {k: v[start:start+conf['batch_size']] for k, v in b.items()}
        p = model(input_batch(sub, conf))
        for k in CHANNELS:
            output[k].append(p[k])
    pred = {k: torch.cat(v, dim=0) for k, v in output.items()}
    objective = float(Objective(b, scales, conf)(pred))
    return pred, objective


def quantiles(values):
    return {'median': float(np.median(values)), 'p10': float(np.quantile(values, .1)), 'p90': float(np.quantile(values, .9))}


def rich_metrics(pred, b, entries):
    result = metrics(pred, b, entries)
    for i, r in enumerate(result['cases']):
        n, t = r['stories'], int(b['time_mask'][i].sum())
        height = b['story_heights'][i, :n].double()
        for label, source in [('true', b), ('pred', pred)]:
            u = source['displacement'][i, :t, :n].double()
            drift = (u-torch.cat((torch.zeros_like(u[:, :1]), u[:, :-1]), dim=-1))/height
            pidr = drift.abs().amax(0).cpu().numpy()
            pfa = source['acceleration'][i, :t, :n].double().abs().amax(0).cpu().numpy()
            r['PIDR_'+label] = pidr.tolist()
            r['PFA_'+label] = pfa.tolist()
            r['DS_'+label] = int(np.sum(pidr.max() > np.array([.005, .008, .02, .05])))
        assert r['DS_true'] == r['DS_ref']
        r['T1_s'] = entries[r['id']]['T1_s']
        r['PIDR_max_mse'] = (max(r['PIDR_pred'])-max(r['PIDR_true']))**2
        r['PIDR_max_ratio'] = max(r['PIDR_pred'])/max(r['PIDR_true'])
        r['roof_PFA_ratio'] = r['PFA_pred'][-1]/r['PFA_true'][-1]
    for name, predicate in [('all', lambda r: True), ('DS_gt_2', lambda r: r['DS_ref'] > 2), ('DS_le_2', lambda r: r['DS_ref'] <= 2)]:
        rows = [r for r in result['cases'] if predicate(r)]
        if rows:
            s = result['summary'][name]
            s['PIDR_max_mse'] = float(np.mean([r['PIDR_max_mse'] for r in rows]))
            s['PIDR_max_ratio'] = quantiles([r['PIDR_max_ratio'] for r in rows])
            s['roof_PFA_ratio'] = quantiles([r['roof_PFA_ratio'] for r in rows])
            s['damage_counts'] = {label: sum(op(r['DS_pred'], r['DS_ref']) for r in rows) for label, op in
                                  [('exact', lambda a, b: a == b), ('under', lambda a, b: a < b), ('over', lambda a, b: a > b)]}
    return result


def data_hashes(cases):
    """Shard-level hashes for shard datasets, per-case NPZ hashes for the exp5 data."""
    return dict(sorted({c.get('shard', c['id']): c['npz_sha256'] for c in cases}.items()))


def save_predictions(path, pred, b):
    np.savez_compressed(path, **{k: pred[k].detach().cpu().numpy() for k in CHANNELS},
                        case_ids=np.array(b['case_ids']), time_mask=b['time_mask'].cpu().numpy(), floor_mask=b['floor_mask'].cpu().numpy())


def structural_hash(c):
    obj = {'model': c['source_config']['model'], 'damping': c['source_config']['analysis']['damping_ratio'],
           'attributes': c['source_config']['provenance']['building_attributes'], 'T1_s': c['T1_s']}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
