"""Record inventory, near-duplicate grouping and the frozen KNET/PEER split (exp7.md section 5.3).

Run once on the execution host:
  python codes/datagen/inventory.py --knet-dir <KNET(3474)> --peer-dir <PEER(65)>
Writes configs/record_split.json; refuses to overwrite an existing split.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

CODES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES))
from datagen.records import read_library, number  # noqa: E402

EXP = CODES.parent


def groups_by_correlation(signals, threshold):
    z = signals - signals.mean(1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    gram = np.abs(z @ z.T)
    np.fill_diagonal(gram, 0.)
    parent = list(range(len(z)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(gram > threshold))):
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(z))]
    return roots, float(gram.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--knet-dir', required=True)
    parser.add_argument('--peer-dir', required=True)
    parser.add_argument('--out', default=str(EXP/'configs/record_split.json'))
    args = parser.parse_args()
    conf = json.loads((EXP/'configs/data.json').read_text(encoding='utf-8'))
    policy = conf['record_split_policy']
    out = Path(args.out)
    if out.exists():
        raise RuntimeError(f'{out} exists; the split is frozen')
    knet = read_library(args.knet_dir, 'knet')
    peer = read_library(args.peer_dir, 'peer')
    records = [dict(r, source='KNET') for r in knet] + [dict(r, source='PEER') for r in peer]
    roots, max_corr = groups_by_correlation(np.stack([r['acceleration'] for r in records]), policy['near_duplicate_correlation'])
    members = {}
    for r, root in zip(records, roots):
        members.setdefault(root, []).append(r)
    excluded_peer = set(policy['historically_evaluated_peer'])
    split = {}
    knet_groups = []
    for group in members.values():
        sources = {r['source'] for r in group}
        tainted = any(r['id'] in excluded_peer for r in group)
        for r in group:
            if r['source'] == 'PEER':
                split[r['id']] = 'excluded' if (tainted or 'KNET' in sources) else 'cross_database'
        knet_members = sorted((r['id'] for r in group if r['source'] == 'KNET'), key=lambda x: number(x))
        if knet_members:
            # A KNET group that shares a waveform with any PEER record may only train; it never tests.
            knet_groups.append((knet_members, 'PEER' in sources))
    rng = np.random.default_rng(policy['seed'])
    order = rng.permutation(len(knet_groups))
    counts = {'test': 0, 'development': 0}
    for index in order:
        ids, peer_linked = knet_groups[index]
        target = 'train'
        if not peer_linked:
            if counts['test'] < policy['test_records']:
                target = 'test'
            elif counts['development'] < policy['development_records']:
                target = 'development'
        if target != 'train':
            counts[target] += len(ids)
        for i in ids:
            split[i] = target
    train = sorted((i for i, s in split.items() if s == 'train'), key=number)
    training_order = [train[i] for i in rng.permutation(len(train))]
    rows = [{'id': r['id'], 'source': r['source'], 'sha256': r['sha256'], 'peak_m_s2': float(np.abs(r['acceleration']).max()),
             'group': int(root), 'split': split[r['id']]} for r, root in zip(records, roots)]
    duplicate_groups = [[r['id'] for r in g] for g in members.values() if len(g) > 1]
    result = {'policy': policy, 'created_by': 'codes/datagen/inventory.py',
              'counts': {s: sum(v == s for v in split.values()) for s in ('train', 'development', 'test', 'cross_database', 'excluded')},
              'max_offdiagonal_abs_correlation': max_corr, 'near_duplicate_groups': duplicate_groups,
              'event_metadata': 'unavailable; claims are record-level only',
              'training_order': training_order, 'records': rows}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'near_duplicate_groups': len(duplicate_groups),
                      'max_correlation': max_corr, 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
