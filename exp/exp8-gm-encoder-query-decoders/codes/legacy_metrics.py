"""Data splits, audited losses and physical-unit MSE. No test split is opened."""
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from src.dataset import ResponseDataset, collate_responses
from src.losses import ResponseScales, response_loss

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
CHANNELS = ('displacement', 'acceleration')


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def select(ds, predicate, device):
    items = [item for entry, item in zip(ds.cases, ds.items) if predicate(entry)]
    b = collate_responses(items)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}


def load_data(conf, protocol):
    data = REPO / conf['dataset']
    ds = ResponseDataset(data / 'train.json')
    entries = {c['id']: c for c in ds.cases}
    manifests = {'train': file_hash(data / 'train.json')}
    if protocol == 'multiwave':
        dv = ResponseDataset(data / 'validation.json')
        entries.update({c['id']: c for c in dv.cases})
        train = select(ds, lambda c: c['DS_ref'] > 2, conf['device'])
        held = select(dv, lambda c: c['DS_ref'] > 2, conf['device'])
        manifests['validation'] = file_hash(data / 'validation.json')
        assert (len(train['case_ids']), len(held['case_ids'])) == (65, 9)
    else:
        def eligible(c):
            return c['record_id'] == conf['single_record'] and '_fixed_' in c['id'] and any(abs(c['pga_g']-p) < 1e-6 for p in conf['shared_pga_g'])
        def isheld(c):
            return abs(c['period_ratio'] - 1.) < 1e-6 if protocol == 'period' else c['stories'] == 5
        train = select(ds, lambda c: eligible(c) and not isheld(c), conf['device'])
        held = select(ds, lambda c: eligible(c) and isheld(c), conf['device'])
        assert (len(train['case_ids']), len(held['case_ids'])) == (18, 9)
        structures = lambda b: {(entries[c]['stories'], entries[c]['period_ratio']) for c in b['case_ids']}
        assert not structures(train) & structures(held)
    assert not set(train['case_ids']) & set(held['case_ids'])
    scales, fit = ResponseScales.fit_training_batches([train], fit_source=f'exp3/{protocol}/train only')
    files = list((EXP / 'codes').rglob('*.py')) + list((EXP / 'configs').glob('*.json'))
    provenance = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_hashes': {str(p.relative_to(EXP)).replace('\\', '/'): file_hash(p) for p in files},
        'protocol': protocol, 'config': conf, 'scales': asdict(scales), 'scale_fit': fit,
        'manifest_sha256': manifests, 'npz_hashes_verified': True, 'test': 'NOT_READ',
        'train_case_ids': train['case_ids'], 'held_case_ids': held['case_ids'],
        'case_npz_sha256': {c: entries[c]['npz_sha256'] for c in train['case_ids'] + held['case_ids']},
        'held_role': 'Previously used development validation' if protocol == 'multiwave' else 'Single-motion structural holdout',
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
    }
    return train, held, entries, scales, provenance


class Objective:
    """Precompute train-only masks/denominators; audited against vendored loss."""
    def __init__(self, b, scales, conf):
        self.b, self.scales, self.conf = b, scales, conf
        self.mask = b['time_mask'][:, :, None] & b['floor_mask'][:, None, :]
        self.count = self.mask.sum((1, 2))
        self.heights = torch.where(b['floor_mask'], b['story_heights'], 1.)[:, None, :]
        self.truth = [b['displacement'], b['acceleration'], self.drift(b['displacement'])]
        self.denominators = []
        for truth, scale in zip(self.truth, asdict(scales).values()):
            den = self.mean(truth.square()).clamp_min((.01 * scale)**2) if conf['normalization'] == 'per_case' else scale**2
            self.denominators.append(den)

    def drift(self, u):
        return (u - torch.cat((torch.zeros_like(u[..., :1]), u[..., :-1]), -1)) / self.heights

    def mean(self, v):
        return torch.where(self.mask, v, 0.).sum((1, 2)) / self.count

    def __call__(self, pred):
        values = [pred['displacement'], pred['acceleration'], self.drift(pred['displacement'])]
        return sum(w * (self.mean((v - truth).square()) / den).mean()
                   for w, v, truth, den in zip(self.conf['response_weights'], values, self.truth, self.denominators))

    def audit(self, pred):
        b = self.b
        full = dict(pred, floor_mask=b['floor_mask'], time_mask=b['time_mask'], stories=b['attributes'][:, 0].long())
        ref = response_loss(full, b, self.scales, self.conf['response_weights'], normalization=self.conf['normalization'])['total']
        torch.testing.assert_close(self(pred), ref, rtol=2e-5, atol=1e-7)


@torch.no_grad()
def metrics(pred, b, entries):
    rows = []
    for i, name in enumerate(b['case_ids']):
        c = entries[name]
        row = {k: c[k] for k in ('id', 'record_id', 'stories', 'period_ratio', 'pga_g', 'DS_ref')}
        t, n = int(b['time_mask'][i].sum()), c['stories']
        for prefix, key in zip(('u', 'a'), CHANNELS):
            mse = (pred[key][i, :t, :n].double() - b[key][i, :t, :n].double()).square().mean(0).cpu().numpy()
            row[prefix + '_floor_mse'] = mse.tolist()
            row[prefix + '_roof_mse'] = float(mse[-1])
            row[prefix + '_all_mse'] = float(mse.mean())
        rows.append(row)
    def group(rs):
        out = {'count': len(rs)}
        if not rs: return out
        out.update({f'{c}_{s}_mse': float(np.mean([r[f'{c}_{s}_mse'] for r in rs])) for c in ('u', 'a') for s in ('roof', 'all')})
        out['by_stories'] = {str(n): {
            'count': sum(r['stories'] == n for r in rs),
            **{f'{c}_floor_mse': np.mean([r[f'{c}_floor_mse'] for r in rs if r['stories'] == n], 0).tolist() for c in ('u', 'a')}
        } for n in sorted({r['stories'] for r in rs})}
        return out
    return {'cases': rows, 'summary': {'all': group(rows), 'DS_gt_2': group([r for r in rows if r['DS_ref'] > 2]), 'DS_le_2': group([r for r in rows if r['DS_ref'] <= 2])}}


def save_prediction(path, pred, b):
    np.savez_compressed(path, **{k: pred[k].detach().cpu().numpy() for k in CHANNELS}, case_ids=np.asarray(b['case_ids']))
