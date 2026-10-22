"""Immutable data selection, SI-unit metrics, provenance and full-floor losses."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from src.dataset import collate_responses
from src.losses import ResponseScales
from legacy_metrics import Objective, metrics, file_hash, state_hash, read, write
from models import input_batch

EXP=Path(__file__).resolve().parents[1]
REPO=EXP.parents[1]
CHANNELS=('displacement','acceleration')


def read_selected(path,predicate):
    path=Path(path)
    meta=read(path)
    cases=[c for c in meta['cases'] if c['status']=='complete' and predicate(c)]
    items=[]
    for c in cases:
        p=path.parent/c['file']
        if file_hash(p)!=c['npz_sha256']: raise ValueError(f'NPZ hash mismatch: {c["id"]}')
        with np.load(p,allow_pickle=False) as f:
            item={k:torch.from_numpy(f[k].copy()) for k in f.files}
        item['case_id']=c['id']
        items.append(item)
    if not items: raise ValueError('Empty declared data split')
    return cases,items,meta


def known_ratio(c,conf):
    return any(abs(c['period_ratio']-v)<1e-6 for v in conf['training_period_ratios'])


def split_predicate(name,conf):
    def choose(c):
        if c['stories'] not in conf['stories']: return False
        if name=='period': return c['record_id'] in conf['training_records'] and abs(c['period_ratio']-conf['period_test_ratio'])<1e-6
        records=conf[{'train':'training_records','development':'development_records','earthquake':'earthquake_test_records'}[name]]
        return c['record_id'] in records and known_ratio(c,conf)
    return choose


def load_split(name,conf):
    manifest={'train':'train','period':'train','development':'validation','earthquake':'test'}[name]
    return read_selected(REPO/conf['dataset']/f'{manifest}.json',split_predicate(name,conf))


def batch(items,device):
    b=collate_responses(items)
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}


def take(b,index):
    ids=index.detach().cpu().tolist()
    return {k:v[index] if torch.is_tensor(v) else [v[i] for i in ids] for k,v in b.items()}


@torch.no_grad()
def evaluate(model,b,conf,scales):
    output={k:[] for k in CHANNELS}
    for start in range(0,len(b['case_ids']),conf['batch_size']):
        sub={k:v[start:start+conf['batch_size']] for k,v in b.items()}
        p=model(input_batch(sub,conf))
        for k in CHANNELS: output[k].append(p[k])
    pred={k:torch.cat(v,dim=0) for k,v in output.items()}
    objective=float(Objective(b,scales,conf)(pred))
    return pred,objective


def rich_metrics(pred,b,entries):
    result=metrics(pred,b,entries)
    for i,r in enumerate(result['cases']):
        n,t=r['stories'],int(b['time_mask'][i].sum())
        height=b['story_heights'][i,:n].double()
        for label,source in [('true',b),('pred',pred)]:
            u=source['displacement'][i,:t,:n].double()
            drift=(u-torch.cat((torch.zeros_like(u[:,:1]),u[:,:-1]),dim=-1))/height
            pidr=drift.abs().amax(0).cpu().numpy()
            pfa=source['acceleration'][i,:t,:n].double().abs().amax(0).cpu().numpy()
            r['PIDR_'+label]=pidr.tolist(); r['PFA_'+label]=pfa.tolist()
            r['DS_'+label]=int(np.sum(pidr.max()>np.array([.005,.008,.02,.05])))
        assert r['DS_true']==r['DS_ref']
        r['T1_s']=entries[r['id']]['T1_s']
        r['PIDR_max_mse']=(max(r['PIDR_pred'])-max(r['PIDR_true']))**2
    for name,predicate in [('all',lambda r:True),('DS_gt_2',lambda r:r['DS_ref']>2),('DS_le_2',lambda r:r['DS_ref']<=2)]:
        rows=[r for r in result['cases'] if predicate(r)]
        if rows:
            result['summary'][name]['PIDR_max_mse']=float(np.mean([r['PIDR_max_mse'] for r in rows]))
            result['summary'][name]['damage_counts']={label:sum(op(r['DS_pred'],r['DS_ref']) for r in rows) for label,op in [('exact',lambda a,b:a==b),('under',lambda a,b:a<b),('over',lambda a,b:a>b)]}
    return result


def save_predictions(path,pred,b):
    np.savez_compressed(path,**{k:pred[k].detach().cpu().numpy() for k in CHANNELS},
                        case_ids=np.array(b['case_ids']),time_mask=b['time_mask'].cpu().numpy(),floor_mask=b['floor_mask'].cpu().numpy())


def structural_hash(c):
    import hashlib
    obj={'model':c['source_config']['model'],'damping':c['source_config']['analysis']['damping_ratio'],
         'attributes':c['source_config']['provenance']['building_attributes'],'T1_s':c['T1_s']}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
