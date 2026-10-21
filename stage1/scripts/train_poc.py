"""5090 training-set fit experiment; reports learnability, never test accuracy."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
from dataclasses import asdict
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STAGE))
from src.model import ModelConfig,SeismicQueryModel
from src.dataset import ResponseDataset,collate_responses
from src.losses import ResponseScales,response_loss,interstory_drift


def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def to_device(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


@torch.no_grad()
def evaluate(model,batch):
    model.eval()
    out=model(batch["ground"],batch["attributes"],batch["structure_type"],batch["floor_ids"],batch["time_mask"])
    mask=batch["time_mask"][:,:,None]&batch["floor_mask"][:,None,:]
    metrics=[]
    for i,name in enumerate(batch["case_ids"]):
        n=int(batch["floor_mask"][i].sum())
        t=int(batch["time_mask"][i].sum())
        row={"case_id":name,"stories":n,"T1_s":float(batch["attributes"][i,3])}
        for key,label in [("displacement","u"),("acceleration","a")]:
            pred=out[key][i,:t,:n]
            truth=batch[key][i,:t,:n]
            row[f"{label}_relative_l2"]=float(torch.linalg.vector_norm(pred-truth)/torch.linalg.vector_norm(truth).clamp_min(1e-10))
        dp=interstory_drift(out["displacement"][i:i+1,:t,:n],batch["story_heights"][i:i+1,:n],batch["floor_mask"][i:i+1,:n]).abs().amax(dim=1)[0]
        dt=interstory_drift(batch["displacement"][i:i+1,:t,:n],batch["story_heights"][i:i+1,:n],batch["floor_mask"][i:i+1,:n]).abs().amax(dim=1)[0]
        ap=out["acceleration"][i,:t,:n].abs().amax(dim=0)
        at=batch["acceleration"][i,:t,:n].abs().amax(dim=0)
        row["PIDR_median_relative_error"]=float(((dp-dt).abs()/dt.clamp_min(1e-6)).median())
        row["PFA_median_relative_error"]=float(((ap-at).abs()/at.clamp_min(1e-4)).median())
        row["true_max_PIDR"]=float(dt.max());row["pred_max_PIDR"]=float(dp.max())
        metrics.append(row)
    summary={f"{label}_median_relative_l2":float(np.median([x[f"{label}_relative_l2"] for x in metrics])) for label in ("u","a")}
    summary["selection_score"]=summary["u_median_relative_l2"]+summary["a_median_relative_l2"]
    summary["PIDR_median_relative_error"]=float(np.median([x["PIDR_median_relative_error"] for x in metrics]))
    summary["PFA_median_relative_error"]=float(np.median([x["PFA_median_relative_error"] for x in metrics]))
    return summary,metrics,out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    conf=json.loads((STAGE/"configs/poc_fit.json").read_text())
    if args.epochs:conf["epochs"]=args.epochs
    seed=conf["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(6)
    if not torch.cuda.is_available():raise RuntimeError("5090 CUDA is required for this run")
    device=torch.device(conf["device"])
    torch.cuda.set_device(device)
    gpu_name=torch.cuda.get_device_name(device)
    if "5090" not in gpu_name:raise RuntimeError(f"Expected RTX 5090, got {gpu_name}")
    torch.backends.cuda.matmul.allow_tf32=True
    dataset=ResponseDataset(STAGE/conf["dataset"])
    cpu_batch=collate_responses(dataset.items)
    scales,scale_info=ResponseScales.fit_training_batches([cpu_batch],fit_source="poc_fit24/training-only")
    batch=to_device(cpu_batch,device)
    model=SeismicQueryModel(ModelConfig(**json.loads((STAGE/conf["model_config"]).read_text()))).to(device)
    model.decoder.set_response_scales(scales.displacement_m,scales.acceleration_m_s2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=conf["learning_rate"],weight_decay=conf["weight_decay"],fused=True)
    run=STAGE/"runs"/conf["run_id"]
    report=STAGE/"reports"/conf["run_id"]
    run.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    checkpoint=run/"best.pt"
    if checkpoint.exists() and not args.resume:raise RuntimeError("Run exists; choose another run_id or --resume")
    git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=STAGE.parent,text=True).strip()
    provenance={"training":conf,"model":model.config.to_dict(),"scales":asdict(scales),"scale_info":scale_info,
                "git_sha":git_sha,"dataset_manifest_sha256":hashlib.sha256((STAGE/conf["dataset"]).read_bytes()).hexdigest(),
                "gpu":gpu_name,"torch":torch.__version__,"cuda":torch.version.cuda,"cases":len(dataset),
                "scope":"Training reconstruction only, no held-out evaluation; same waveform across all 24 cases"}
    json_write(run/"provenance.json",provenance)
    best=float("inf");best_epoch=0;start_epoch=0;curve=[]
    if args.resume:
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        start_epoch=state["epoch"];best=state["score"];best_epoch=start_epoch
        old=run/"curve.json"
        if old.exists():curve=json.loads(old.read_text())
    start=time.perf_counter()
    print(json.dumps({"status":"training","gpu":gpu_name,"cases":len(dataset),"parameters":sum(p.numel() for p in model.parameters()),"run_id":conf["run_id"]}),flush=True)
    for epoch in range(start_epoch+1,conf["epochs"]+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        output=model(batch["ground"],batch["attributes"],batch["structure_type"],batch["floor_ids"],batch["time_mask"])
        parts=response_loss(output,batch,scales)
        if not torch.isfinite(parts["total"]):raise RuntimeError("Nonfinite training loss")
        parts["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if epoch==1 or epoch%conf["evaluate_every"]==0 or epoch==conf["epochs"]:
            summary,_,_=evaluate(model,batch)
            row={"epoch":epoch,"loss":float(parts["total"].detach()),"wall_seconds":time.perf_counter()-start,**summary}
            curve.append(row);json_write(run/"curve.json",curve);json_write(run/"progress.json",row)
            if summary["selection_score"]<best:
                best=summary["selection_score"];best_epoch=epoch
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,
                            "score":best,"provenance":provenance},checkpoint)
            print(json.dumps(row),flush=True)
            if summary["u_median_relative_l2"]<conf["target_median_relative_l2"] and summary["a_median_relative_l2"]<conf["target_median_relative_l2"]:
                break
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state["model"])
    summary,details,prediction=evaluate(model,batch)
    result={"status":"PASS" if max(summary["u_median_relative_l2"],summary["a_median_relative_l2"])<conf["target_median_relative_l2"] else "PARTIAL",
            "best_epoch":best_epoch,"wall_seconds":time.perf_counter()-start,"peak_gpu_memory_gb":torch.cuda.max_memory_allocated()/2**30,
            "summary":summary,"cases":details,"provenance":provenance,
            "limitation":"Training-set fit only; no generalization result"}
    json_write(report/"result.json",result);json_write(report/"curve.json",curve)
    np.savez_compressed(run/"predictions.npz",displacement=prediction["displacement"].cpu().numpy(),
                        acceleration=prediction["acceleration"].cpu().numpy(),
                        truth_displacement=batch["displacement"].cpu().numpy(),truth_acceleration=batch["acceleration"].cpu().numpy(),
                        case_ids=np.asarray(batch["case_ids"]),floor_mask=batch["floor_mask"].cpu().numpy())
    plot_results(report,curve,result,batch,prediction)
    print(json.dumps({"status":result["status"],"best_epoch":best_epoch,**summary,"report":str(report)}),flush=True)


def plot_results(folder,curve,result,batch,prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.2,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].semilogy([x["epoch"] for x in curve],[x["loss"] for x in curve],color="#2166ac")
    for label,color in [("u","#2166ac"),("a","#d6604d")]:
        axes[1].semilogy([x["epoch"] for x in curve],[x[f"{label}_median_relative_l2"] for x in curve],label=label,color=color)
    axes[1].axhline(.1,color="#555555",ls="--",label="10% fit target")
    axes[0].set(xlabel="Epoch",ylabel="Normalized loss",title="Training loss")
    axes[1].set(xlabel="Epoch",ylabel="Median relative L2",title="Training-set waveform error");axes[1].legend()
    fig.suptitle("POC: training reconstruction only | RTX 5090 | patch 60")
    fig.savefig(folder/"learning_curve.png",dpi=170);plt.close(fig)
    for n in [3,5,8,12]:
        selected=[i for i,name in enumerate(batch["case_ids"]) if name.startswith(f"{n:02d}F_r1.00_")]
        if not selected:continue
        i=selected[-1];name=batch["case_ids"][i]
        floors=list(range(n)) if n<=5 else sorted(set([0,n//2,n-1]))
        fig,axes=plt.subplots(len(floors),2,figsize=(13,2.3*len(floors)),squeeze=False,sharex=True,constrained_layout=True)
        for row,f in enumerate(floors):
            for col,(key,scale,unit) in enumerate([("displacement",1000.,"mm"),("acceleration",1/9.80665,"g")]):
                truth=batch[key][i,:,f].cpu().numpy()*scale;pred=prediction[key][i,:,f].cpu().numpy()*scale
                t=np.arange(len(truth))*.02
                axes[row,col].plot(t,truth,color="#333333",lw=.9,label="C++ reference")
                axes[row,col].plot(t,pred,color="#d6604d",lw=.7,alpha=.85,label="Model fit")
                axes[row,col].set_ylabel(f"Floor {f+1}\n{key} ({unit})")
                axes[row,col].set_xlim(0,35)
        axes[0,0].legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel("Time (s)")
        fig.suptitle(f"{name} | training-set reconstruction | first 35 s shown")
        fig.savefig(folder/f"fit_{n:02d}F.png",dpi=160);plt.close(fig)
    lines=["# Stage 1: 5090 small-sample fit","",f"Status: {result['status']}","",
           "Training reconstruction only. All cases share the same base waveform; this is not a generalization test.","",
           f"Best epoch: {result['best_epoch']}. Median displacement relative L2: {result['summary']['u_median_relative_l2']:.2%}; acceleration: {result['summary']['a_median_relative_l2']:.2%}.","",
           "![Learning curve](learning_curve.png)",""]
    for n in [3,5,8,12]:lines += [f"![{n}-story fit](fit_{n:02d}F.png)",""]
    (folder/"README.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":main()
