"""Shared masked metrics for the predeclared ten-record experiment."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import torch
STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from src.dataset import ResponseDataset,collate_responses
from src.losses import interstory_drift
from train_poc import json_write,to_device


def load_split(folder,split,device):
    dataset=ResponseDataset(folder/f"{split}.json")
    cpu=collate_responses(dataset.items)
    return dataset,cpu,to_device(cpu,device)


def slice_batch(batch,indices):
    return {k:v[indices] if torch.is_tensor(v) else [v[int(i)] for i in indices] for k,v in batch.items()}


@torch.no_grad()
def predict(model,batch,batch_size=48):
    model.eval();outputs=[]
    for start in range(0,len(batch["case_ids"]),batch_size):
        idx=torch.arange(start,min(start+batch_size,len(batch["case_ids"])),device=batch["ground"].device)
        b=slice_batch(batch,idx)
        out=model(b["ground"],b["attributes"],b["structure_type"],b["floor_ids"],b["time_mask"])
        outputs.append({k:out[k] for k in ("displacement","acceleration")})
    return {k:torch.cat([o[k] for o in outputs]) for k in outputs[0]}


@torch.no_grad()
def measure(pred,batch,entries):
    rows=[]
    u=pred["displacement"];a=pred["acceleration"]
    theta=interstory_drift(u,batch["story_heights"],batch["floor_mask"])
    truth_theta=interstory_drift(batch["displacement"],batch["story_heights"],batch["floor_mask"])
    for i,name in enumerate(batch["case_ids"]):
        meta=entries[name];n=meta["stories"];t=int(batch["time_mask"][i].sum())
        row={k:meta[k] for k in ("id","record_id","stories","period_ratio","pga_g","DS_ref","ductility_max")}
        for label,key in (("u","displacement"),("a","acceleration")):
            truth=batch[key][i,:t,:n];p=pred[key][i,:t,:n]
            error=(p-truth).square()
            denominator=truth.square().sum().clamp_min(1e-12)
            row[f"{label}_relative_l2"]=float((error.sum()/denominator).sqrt())
            row[f"{label}_mse"]=float(error.mean())
            baseline=torch.zeros_like(truth) if label=="u" else -batch["ground"][i,:t,:].expand(t,n)
            row[f"{label}_baseline_mse"]=float((truth-baseline).square().mean())
            row[f"{label}_floor_relative_l2"]=(error.sum(0)/truth.square().sum(0).clamp_min(1e-12)).sqrt().cpu().tolist()
        tp=theta[i,:t,:n].abs().amax(0);tt=truth_theta[i,:t,:n].abs().amax(0)
        ap=a[i,:t,:n].abs().amax(0);at=batch["acceleration"][i,:t,:n].abs().amax(0)
        row.update(true_PIDR=tt.cpu().tolist(),pred_PIDR=tp.cpu().tolist(),true_PFA=at.cpu().tolist(),pred_PFA=ap.cpu().tolist(),
                   PIDR_floor_relative_error=((tp-tt).abs()/tt.clamp_min(1e-6)).cpu().tolist(),
                   PFA_floor_relative_error=((ap-at).abs()/at.clamp_min(1e-4)).cpu().tolist(),
                   pred_DS_ref=int((tp.max()>torch.tensor([.005,.008,.02,.05],device=tp.device)).sum()),
                   true_critical_story=int(tt.argmax())+1,pred_critical_story=int(tp.argmax())+1)
        rows.append(row)
    return rows


def aggregate(rows):
    if not rows:return {"count":0}
    s={"count":len(rows),"base_records":sorted({x["record_id"] for x in rows})}
    for label in ("u","a"):
        z=[x[f"{label}_relative_l2"] for x in rows]
        s[f"{label}_median_relative_l2"]=float(np.median(z));s[f"{label}_p90_relative_l2"]=float(np.quantile(z,.9));s[f"{label}_worst_relative_l2"]=float(max(z))
        s[f"{label}_mse_improvement_vs_baseline"]=float(1-np.mean([x[f"{label}_mse"] for x in rows])/max(np.mean([x[f"{label}_baseline_mse"] for x in rows]),1e-15))
    for label in ("PIDR","PFA"):
        z=[v for x in rows for v in x[f"{label}_floor_relative_error"]]
        s[f"{label}_floor_median_relative_error"]=float(np.median(z));s[f"{label}_floor_p90_relative_error"]=float(np.quantile(z,.9))
    s["DS_ref_accuracy"]=float(np.mean([x["DS_ref"]==x["pred_DS_ref"] for x in rows]))
    s["critical_story_accuracy"]=float(np.mean([x["true_critical_story"]==x["pred_critical_story"] for x in rows]))
    return s


def grouped(rows):
    result={"all":aggregate(rows),"DS_gt_2":aggregate([x for x in rows if x["DS_ref"]>2]),
            "DS_le_2":aggregate([x for x in rows if x["DS_ref"]<=2])}
    result["by_stories"]={str(n):{"all":aggregate([x for x in rows if x["stories"]==n]),
                                      "DS_gt_2":aggregate([x for x in rows if x["stories"]==n and x["DS_ref"]>2])} for n in (3,5,8)}
    result["by_record"]={k:aggregate([x for x in rows if x["record_id"]==k]) for k in sorted({x["record_id"] for x in rows})}
    return result


def selection_score(s):
    # Predetermined equal emphasis on all validation cases and strong nonlinear validation cases.
    return float(np.mean([s[g][f"{c}_median_relative_l2"] for g in ("all","DS_gt_2") for c in ("u","a")]))
