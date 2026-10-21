"""Train jointly on seven records; select on validation only; never read test labels."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import random
import subprocess
import time
import numpy as np
import torch
from peer10_common import STAGE,load_split,slice_batch,predict,measure,grouped,selection_score,json_write
from src.model import SeismicQueryModel,ModelConfig
from src.losses import ResponseScales,response_loss


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/peer10_train.json");args=p.parse_args()
    conf=json.loads((STAGE/args.config).read_text())
    random.seed(conf["seed"]);np.random.seed(conf["seed"]);torch.manual_seed(conf["seed"]);torch.set_num_threads(6)
    device=torch.device(conf["device"])
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(device):raise RuntimeError("RTX 5090 required")
    torch.backends.cuda.matmul.allow_tf32=True
    folder=STAGE/conf["dataset"]
    td,cpu_train,train=load_split(folder,"train",device)
    vd,_,val=load_split(folder,"validation",device)
    train_records={x["record_id"] for x in td.cases};val_records={x["record_id"] for x in vd.cases}
    assert not train_records&val_records
    assert all(sum(x["DS_ref"]>2 and x["stories"]==n for x in td.cases)>0 for n in (3,5,8))
    scales,scale_info=ResponseScales.fit_training_batches([cpu_train],fit_source="peer10/train records only")
    model=SeismicQueryModel(ModelConfig(**json.loads((STAGE/conf["model_config"]).read_text()))).to(device)
    model.decoder.set_response_scales(scales.displacement_m,scales.acceleration_m_s2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=conf["learning_rate"],weight_decay=conf["weight_decay"],fused=True)
    run=STAGE/"runs"/conf["run_id"];report=STAGE/"reports"/conf["run_id"]
    run.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    if (run/"best.pt").exists():raise RuntimeError("Existing run; use a new run_id")
    provenance={"training":conf,"model":model.config.to_dict(),"scales":asdict(scales),"scale_info":scale_info,
                "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=STAGE.parent,text=True).strip(),
                "manifest_sha256":{s:hashlib.sha256((folder/f"{s}.json").read_bytes()).hexdigest() for s in ("train","validation")},
                "train_records":sorted(train_records),"validation_records":sorted(val_records),
                "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),
                "scope":"Random initialization; train+validation only; test not loaded by trainer",
                "selection":"mean of u/a relative-L2 medians on all validation and DS>2 validation"}
    json_write(run/"provenance.json",provenance)
    entries={x["id"]:x for x in td.cases+vd.cases}
    curve=[];best=float("inf");best_epoch=0;steps=0;start=time.perf_counter()
    for epoch in range(1,conf["epochs"]+1):
        if epoch<=conf["warmup_epochs"]:lr=conf["learning_rate"]*epoch/conf["warmup_epochs"]
        else:
            f=(epoch-conf["warmup_epochs"])/(conf["epochs"]-conf["warmup_epochs"])
            lr=conf["minimum_learning_rate"]+.5*(conf["learning_rate"]-conf["minimum_learning_rate"])*(1+math.cos(math.pi*f))
        for group in optimizer.param_groups:group["lr"]=lr
        order=torch.randperm(len(td),device=device);loss_sum=0.;seen=0
        model.train()
        for begin in range(0,len(td),conf["batch_size"]):
            b=slice_batch(train,order[begin:begin+conf["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            out=model(b["ground"],b["attributes"],b["structure_type"],b["floor_ids"],b["time_mask"])
            losses=response_loss(out,b,scales,normalization=conf["loss_normalization"])
            if not torch.isfinite(losses["total"]):raise RuntimeError("Nonfinite loss")
            losses["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();steps+=1
            count=len(b["case_ids"]);seen+=count;loss_sum+=float(losses["total"].detach())*count
        if epoch==1 or epoch%conf["evaluate_every"]==0 or epoch==conf["epochs"]:
            ts=grouped(measure(predict(model,train),train,entries));vs=grouped(measure(predict(model,val),val,entries))
            score=selection_score(vs)
            row={"epoch":epoch,"optimizer_steps":steps,"wall_seconds":time.perf_counter()-start,"loss":loss_sum/seen,"learning_rate":lr,
                 "train":{g:ts[g] for g in ("all","DS_gt_2")},"validation":{g:vs[g] for g in ("all","DS_gt_2")},"selection_score":score}
            curve.append(row);json_write(run/"curve.json",curve);json_write(run/"progress.json",row)
            if score<best:
                best=score;best_epoch=epoch
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"score":score,"provenance":provenance},run/"best.pt")
            print(json.dumps({"epoch":epoch,"seconds":row["wall_seconds"],"loss":row["loss"],"best_epoch":best_epoch,
                              "train_u":ts["all"]["u_median_relative_l2"],"train_a":ts["all"]["a_median_relative_l2"],
                              "train_DS3_u":ts["DS_gt_2"]["u_median_relative_l2"],"train_DS3_a":ts["DS_gt_2"]["a_median_relative_l2"],
                              "val_u":vs["all"]["u_median_relative_l2"],"val_a":vs["all"]["a_median_relative_l2"],
                              "val_DS3_u":vs["DS_gt_2"]["u_median_relative_l2"],"val_DS3_a":vs["DS_gt_2"]["a_median_relative_l2"]}),flush=True)
            if epoch>=conf["minimum_epochs"] and epoch-best_epoch>=conf["patience_epochs"]:break
    state=torch.load(run/"best.pt",map_location=device,weights_only=False);model.load_state_dict(state["model"])
    summary={}
    for name,b in (("train",train),("validation",val)):
        rows=measure(predict(model,b),b,entries);summary[name]={"summary":grouped(rows),"cases":rows}
    result={"best_epoch":state["epoch"],"last_epoch":epoch,"optimizer_steps":steps,"wall_seconds":time.perf_counter()-start,
            "peak_gpu_memory_gb":torch.cuda.max_memory_allocated()/2**30,"provenance":provenance,"splits":summary,
            "test_status":"NOT_EVALUATED"}
    json_write(report/"training_result.json",result);json_write(report/"curve.json",curve)
    print(json.dumps({"status":"TRAINING_COMPLETE","best_epoch":state["epoch"],"seconds":result["wall_seconds"],"test":"NOT_EVALUATED"}),flush=True)


if __name__=="__main__":main()
