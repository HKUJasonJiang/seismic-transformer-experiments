"""Read exp7 dataset shards into the same per-case items that exp5's collate_responses expects."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

EXP = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load_part(data_conf, part, records=None, data_experiment=EXP):
    """Return (cases, items, manifest) for one part of the dataset generated under `data_experiment`; `records` restricts base records."""
    root = Path(data_experiment)/"runs/data"/data_conf["dataset_id"]
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    keep = None if records is None else set(records)
    cases, items = [], []
    for shard in manifest['shards']:
        if shard['part'] != part:
            continue
        path = root/'shards'/shard['npz']
        if sha(path) != shard['npz_sha256']:
            raise ValueError(f'Shard hash mismatch: {path.name}')
        meta = {m['id']: m for m in json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['cases']}
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        for i, case_id in enumerate(arrays['case_ids']):
            m = dict(meta[str(case_id)], shard=shard['npz'], npz_sha256=shard['npz_sha256'])
            if keep is not None and m['record_id'] not in keep:
                continue
            n = int(arrays['stories'][i])
            items.append({'ground': torch.from_numpy(arrays['ground'][i][:, None].copy()),
                          'attributes': torch.from_numpy(arrays['attributes'][i].copy()),
                          'structure_type': torch.tensor(int(arrays['structure_type'][i])),
                          'displacement': torch.from_numpy(arrays['displacement'][i, :, :n].copy()),
                          'acceleration': torch.from_numpy(arrays['acceleration'][i, :, :n].copy()),
                          'story_heights': torch.from_numpy(arrays['story_heights'][i, :n].copy()),
                          'case_id': m['id']})
            cases.append(m)
    if not items:
        raise ValueError(f'No complete cases for part {part}')
    return cases, items, manifest
