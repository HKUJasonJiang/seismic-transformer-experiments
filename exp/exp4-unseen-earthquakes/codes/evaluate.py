"""Frozen A+B evaluation. Execute on 5090; no training or test-fitted scales."""
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import file_hash, state_hash, metrics, read, write
from models import fresh
from src.dataset import ResponseDataset, collate_responses
from src.losses import ResponseScales
from src.model import ModelConfig

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
CHANNELS = ('displacement', 'acceleration')
INPUTS = ('ground', 'attributes', 'structure_type', 'floor_ids', 'floor_mask', 'time_mask')


def fingerprint(c, item):
    obj = {'model': c['source_config']['model'],
           'damping_ratio': c['source_config']['analysis']['damping_ratio'],
           **{k: item[k].tolist() for k in ('attributes', 'structure_type', 'story_heights')}}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def summary(rows):
    out = {'count': len(rows)}
    if not rows:
        return out
    names = [f'{c}_{scope}_mse' for c in ('u', 'a') for scope in ('roof', 'all')]
    names += ['PIDR_max_mse', 'PIDR_floor_mse', 'PFA_roof_mse', 'PFA_floor_mse']
    out.update({k: float(np.mean([r[k] for r in rows])) for k in names})
    out['damage_counts'] = {name: sum(int(op(r['DS_pred'], r['DS_ref'])) for r in rows)
                            for name, op in [('exact', np.equal), ('under', np.less), ('over', np.greater)]}
    out['true_DS_counts'] = dict(Counter(str(r['DS_ref']) for r in rows))
    out['pred_DS_counts'] = dict(Counter(str(r['DS_pred']) for r in rows))
    out['PGA_outside_training_range_count'] = sum(not r['PGA_in_training_range'] for r in rows)
    out['by_stories'] = {str(n): {'count': len(rs), **{
        f'{c}_floor_mse': np.mean([r[f'{c}_floor_mse'] for r in rs], axis=0).tolist() for c in ('u', 'a')}}
        for n in sorted({r['stories'] for r in rows}) if (rs := [r for r in rows if r['stories'] == n])}
    return out


def curve_figure(c, item, pred, roof=False):
    n = c['stories']
    floors = [n-1] if roof else list(range(n))
    fig, axes = plt.subplots(len(floors), 2, figsize=(13, 2.1*len(floors)+1), squeeze=False)
    t = np.arange(len(item['ground'])) / 50.
    for row, f in enumerate(floors):
        for j, (key, unit) in enumerate(zip(CHANNELS, ('m', 'm/s2'))):
            ax = axes[row, j]
            y = item[key][:, f].numpy()
            yp = pred[key][:, f]
            ax.plot(t, y, color='#202c38', lw=.9, label='MDOF reference')
            ax.plot(t, yp, color='#e16a2d', lw=.85, alpha=.85, label='A+B prediction')
            mse = np.mean((yp.astype(float)-y.astype(float))**2)
            ax.set_title(f'Floor {f+1}: {key}, MSE={mse:.4g}', fontsize=9)
            ax.set_ylabel(unit)
            ax.grid(alpha=.2)
            if row == len(floors)-1:
                ax.set_xlabel('Time (s)')
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Known structure / unseen input record: {c["id"]}\nReference DS={c["DS_ref"]}; predicted DS={c["DS_pred"]}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94 if roof else .97))
    return fig


def main():
    conf = read(EXP / 'configs/evaluate.json')
    checkpoint = REPO / conf['checkpoint']
    assert file_hash(checkpoint) == conf['checkpoint_sha256']
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Own archived checkpoint; its immutable hash is fixed before test loading.
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['step'] == 14000
    assert ck['provenance']['name'] == 'multiwave-joint-matchedTotal'
    model = fresh(ModelConfig(**ck['config']), ResponseScales(**ck['scales']), conf)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().requires_grad_(False)
    before = state_hash(model)
    for folder in ('results', 'figures/cases', 'runs'):
        (EXP / folder).mkdir(parents=True, exist_ok=True)
    data = REPO / conf['dataset']
    train = ResponseDataset(data / 'train.json')
    train_ids = set(ck['provenance']['train_case_ids'])
    selected = [(c, it) for c, it in zip(train.cases, train.items) if c['id'] in train_ids]
    assert len(selected) == len(train_ids) == 65
    assert all(c['DS_ref'] > 2 for c, _ in selected)
    assert sorted({c['record_id'] for c, _ in selected}) == sorted(conf['training_records'])
    structures = {}
    for c, it in selected:
        key = fingerprint(c, it)
        structures.setdefault(key, []).append(c)
    assert len(structures) == 9
    # Check current train manifest against the checkpoint's actual training manifest.
    assert file_hash(data/'train.json') == ck['provenance']['manifest_sha256']['train']
    test = ResponseDataset(data / 'test.json')
    assert sorted({c['record_id'] for c in test.cases}) == sorted(conf['test_records'])
    assert not train_ids.intersection(c['id'] for c in test.cases)
    record_hashes = {r['record_id']: r['sha256'] for r in test.metadata['record_split']['records']}
    assert not {record_hashes[r] for r in conf['training_records']} & {record_hashes[r] for r in conf['test_records']}
    for c, it in zip(test.cases, test.items):
        key = fingerprint(c, it)
        assert key in structures, f'Unseen structural configuration: {c["id"]}'
        c['structure_sha256'] = key
        pgas = [x['pga_g'] for x in structures[key]]
        c['train_PGA_range'] = [min(pgas), max(pgas)]
        c['PGA_in_training_range'] = min(pgas)-1e-6 <= c['pga_g'] <= max(pgas)+1e-6
    entries = {c['id']: c for c in test.cases}
    predictions, rows, query_audits = {}, [], []
    with torch.inference_mode():
        for start in range(0, len(test), conf['batch_size']):
            items = test.items[start:start+conf['batch_size']]
            b = collate_responses(items)
            b = {k: v.to(conf['device']) if torch.is_tensor(v) else v for k,v in b.items()}
            # This dictionary deliberately cannot contain target response tensors.
            inputs = {k: b[k] for k in INPUTS}
            pred, _, _ = model(inputs)
            rows.extend(metrics(pred, b, entries)['cases'])
            roof_input = dict(inputs, floor_ids=b['attributes'][:, :1].long(),
                              floor_mask=torch.ones(len(items), 1, dtype=torch.bool, device=conf['device']))
            roof, _, _ = model(roof_input)
            reversed_input = dict(inputs, floor_ids=b['floor_ids'].flip(-1), floor_mask=b['floor_mask'].flip(-1))
            reversed_pred, _, _ = model(reversed_input)
            for j, it in enumerate(items):
                name = it['case_id']
                n, t = len(it['story_heights']), len(it['ground'])
                predictions[name] = {k: pred[k][j, :t, :n].cpu().numpy() for k in CHANNELS}
                for k in CHANNELS:
                    torch.testing.assert_close(roof[k][j, :t, 0], pred[k][j, :t, n-1], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(reversed_pred[k][j].flip(-1), pred[k][j], rtol=1e-5, atol=1e-6)
                    query_audits.append(float((roof[k][j, :t, 0].double()-pred[k][j, :t, n-1].double()).square().mean()))
            print(f'Inference {min(start+len(items),len(test))}/{len(test)}', flush=True)
    assert state_hash(model) == before
    assert file_hash(checkpoint) == conf['checkpoint_sha256']
    by_id = dict(zip((c['id'] for c in test.cases), test.items))
    thresholds = np.array(conf['damage_thresholds_strict_gt'])
    for r in rows:
        c, it = entries[r['id']], by_id[r['id']]
        pred = predictions[r['id']]
        height = it['story_heights'].numpy().astype(float)
        for label, source in [('true', {k: it[k].numpy() for k in CHANNELS}), ('pred', pred)]:
            u = source['displacement'].astype(float)
            drift = np.diff(np.pad(u, ((0, 0), (1, 0))), axis=1) / height
            pidr = np.max(np.abs(drift), axis=0)
            pfa = np.max(np.abs(source['acceleration'].astype(float)), axis=0)
            r[f'PIDR_{label}'] = pidr.tolist()
            r[f'PFA_{label}'] = pfa.tolist()
            r[f'DS_{label}'] = int(np.sum(pidr.max() > thresholds))
        assert r['DS_true'] == r['DS_ref']
        r['PIDR_max_mse'] = (max(r['PIDR_pred'])-max(r['PIDR_true']))**2
        r['PIDR_floor_mse'] = float(np.mean((np.array(r['PIDR_pred'])-r['PIDR_true'])**2))
        r['PFA_roof_mse'] = (r['PFA_pred'][-1]-r['PFA_true'][-1])**2
        r['PFA_floor_mse'] = float(np.mean((np.array(r['PFA_pred'])-r['PFA_true'])**2))
        for key in ('structure_sha256', 'train_PGA_range', 'PGA_in_training_range'):
            r[key] = c[key]
    subsets = {'all_valid': rows, 'DS_gt_2': [r for r in rows if r['DS_ref']>2],
               'DS_le_2': [r for r in rows if r['DS_ref']<=2]}
    report = {
        'config': conf, 'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_sha256': {p.relative_to(EXP).as_posix(): file_hash(p) for p in sorted((EXP/'codes').rglob('*.py'))},
        'config_sha256': file_hash(EXP/'configs/evaluate.json'),
        'manifest_sha256': {s: file_hash(data/f'{s}.json') for s in ('train','test')},
        'checkpoint_sha256': file_hash(checkpoint), 'checkpoint_step': ck['step'],
        'checkpoint_provenance': ck['provenance'], 'frozen_model_state_sha256': before,
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
        'audit': {'known_structures': len(structures), 'all_test_structures_seen': True,
                  'source_record_sha_disjoint': True, 'no_targets_in_model_input': True,
                  'weights_unchanged': True, 'max_roof_query_difference_mse': max(query_audits),
                  'reversed_floor_query_consistent': True, 'npz_sha_verified': True,
                  'test_simulation_status_counts': dict(Counter(c['status'] for c in test.metadata['cases']))},
        'record_sha256': record_hashes,
        'test_npz_sha256': {c['id']: c['npz_sha256'] for c in test.cases},
        'summary': {s: {'pooled': summary(rs), 'by_record': {rec: summary([r for r in rs if r['record_id']==rec]) for rec in conf['test_records']},
                        'within_training_PGA': summary([r for r in rs if r['PGA_in_training_range']]),
                        'outside_training_PGA': summary([r for r in rs if not r['PGA_in_training_range']])} for s,rs in subsets.items()},
        'cases': rows,
    }
    write(EXP/'results/report.json', report)
    np.savez_compressed(EXP/'results/predictions.npz', **{f'{name}__{key}': value for name,p in predictions.items() for key,value in p.items()})
    # Safe, self-contained inference weights and metadata (no original pickle provenance types).
    torch.save({'model': {k:v.cpu() for k,v in model.state_dict().items()}, 'config': ck['config'],
                'scales': ck['scales'], 'evaluation_config': conf}, EXP/'runs/ab_frozen.pt')
    rep_ids = []
    for rec in conf['test_records']:
        for n in (3,5,8):
            r = max((r for r in rows if r['record_id']==rec and r['stories']==n and abs(r['period_ratio']-1.)<1e-6), key=lambda r:r['pga_g'])
            rep_ids.append(r['id'])
    for i, r in enumerate(rows):
        for roof in (True, False):
            fig = curve_figure(r, by_id[r['id']], predictions[r['id']], roof)
            suffix = 'roof' if roof else 'floors'
            path = EXP/f'figures/cases/{r["id"]}_{suffix}.png'
            fig.savefig(path, dpi=125)
            plt.close(fig)
            if r['id'] in rep_ids:
                shutil.copy2(path, EXP/f'figures/{r["record_id"]}_{r["stories"]}F_{suffix}.png')
        if (i+1)%10==0:
            print(f'Figures {i+1}/{len(rows)} cases', flush=True)
    # Compact roof gallery, with identical references/predictions as complete plots.
    for rec in conf['test_records']:
        fig, axes = plt.subplots(3,2,figsize=(13,8))
        for j,n in enumerate((3,5,8)):
            r = next(r for r in rows if r['id'] in rep_ids and r['record_id']==rec and r['stories']==n)
            it, p = by_id[r['id']], predictions[r['id']]
            t = np.arange(len(it['ground']))/50
            for k,(key,prefix,unit) in enumerate(zip(CHANNELS,('u','a'),('m','m/s2'))):
                ax=axes[j,k]
                ax.plot(t,it[key][:,-1],lw=.9,color='#202c38',label='MDOF reference')
                ax.plot(t,p[key][:,-1],lw=.8,color='#e16a2d',label='A+B prediction')
                ax.set_title(f'{n}F roof / PGA={r["pga_g"]:.3f}g / MSE={r[prefix+"_roof_mse"]:.4g}',fontsize=10)
                ax.set_ylabel(unit); ax.grid(alpha=.2)
                if j==2: ax.set_xlabel('Time (s)')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Known structural configurations, unseen input {rec}\nPreselected r=1.0, highest valid PGA; A+B frozen at 14000 updates')
        fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(EXP/f'figures/{rec}_roof_gallery.png',dpi=140); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,5))
    for ax, field, label in zip(axes,('PIDR','PFA'),('Maximum interstory drift ratio','Maximum floor acceleration (m/s2)')):
        for rec, marker in zip(conf['test_records'],('o','^')):
            rs=[r for r in rows if r['record_id']==rec and r['DS_ref']>2]
            ax.scatter([max(r[field+'_true']) for r in rs],[max(r[field+'_pred']) for r in rs],marker=marker,label=rec)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([0,lim],[0,lim],'k--',lw=.8)
        ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.set_xlabel('Reference'); ax.set_ylabel('A+B prediction'); ax.set_title(label); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Unseen input records: nonlinear cases with reference DS > 2')
    fig.tight_layout(); fig.savefig(EXP/'figures/nonlinear_peaks.png',dpi=140); plt.close(fig)
    write(EXP/'results/figure_selection.json', {'representative_rule':conf['representative_rule'], 'case_ids':rep_ids,
                                               'case_figure_count':2*len(rows)})
    print(json.dumps(report['summary']['DS_gt_2'],indent=2),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
