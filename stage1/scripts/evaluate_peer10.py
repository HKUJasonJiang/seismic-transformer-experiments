"""One final held-record evaluation, nonlinear bins and explicit decoder query checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from peer10_common import STAGE,load_split,predict,measure,grouped,json_write,slice_batch
from src.model import SeismicQueryModel,ModelConfig


def passes(s):
    return bool(s.get("count",0)>0 and all(s[f"{c}_mse_improvement_vs_baseline"]>=.3 for c in ("u","a"))
                and all(s[f"{c}_floor_median_relative_error"]<=.2 for c in ("PIDR","PFA")))


@torch.no_grad()
def query_check(model,batch):
    """Every test floor queried independently; compare with full-floor decoding."""
    maxima={"displacement":0.,"acceleration":0.};count=0
    for i in range(len(batch["case_ids"])):
        b=slice_batch(batch,torch.tensor([i],device=batch["ground"].device))
        n=int(b["floor_mask"].sum());ids=b["floor_ids"][:,:n]
        full=model(b["ground"],b["attributes"],b["structure_type"],ids,b["time_mask"])
        for floor in range(n):
            q=ids[:,floor:floor+1]
            out=model.decoder(full["latent"],full["structure_latent"],full["patch_mask"],b["attributes"],q,q>0,b["time_mask"])
            for key in maxima:
                delta=float((out[key]-full[key][...,floor:floor+1]).abs().max())
                maxima[key]=max(maxima[key],delta)
                torch.testing.assert_close(out[key],full[key][...,floor:floor+1],atol=1e-7 if key=="displacement" else 1e-4,rtol=1e-4)
            count+=1
        order=torch.arange(n-1,-1,-1,device=ids.device);q=ids[:,order]
        out=model.decoder(full["latent"],full["structure_latent"],full["patch_mask"],b["attributes"],q,q>0,b["time_mask"])
        for key in maxima:torch.testing.assert_close(out[key],full[key][...,order],atol=1e-7 if key=="displacement" else 1e-4,rtol=1e-4)
    return {"status":"PASS","test_buildings":len(batch["case_ids"]),"single_floor_queries":count,
            "maximum_absolute_difference_SI":maxima,"meaning":"Query consistency only; accuracy is measured separately against C++ labels"}


def plots(folder,curve,result,test_batch,pred,test_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.2,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for col,label in enumerate(("u","a")):
        for split,color in (("train","#2166ac"),("validation","#d6604d")):
            for group,style in (("all","-"),("DS_gt_2","--")):
                ax[col].semilogy([x["epoch"] for x in curve],[x[split][group][f"{label}_median_relative_l2"] for x in curve],style,color=color,label=f"{split} {group}")
        ax[col].axhline(.1,color="gray",ls=":");ax[col].set(xlabel="Epoch",ylabel="Median relative L2",title="Displacement" if label=="u" else "Acceleration");ax[col].legend(fontsize=7)
    fig.suptitle("3/5/8 stories jointly | 7 training records, 1 validation record")
    fig.savefig(folder/"learning_curve.png",dpi=170);plt.close(fig)
    rows=result["splits"]["test"]["cases"];entry={x["id"]:x for x in test_data.cases}
    chosen=[]
    for n in (3,5,8):
        candidates=sorted([x for x in rows if x["stories"]==n and x["DS_ref"]>2],key=lambda x:x["u_relative_l2"]+x["a_relative_l2"])
        if candidates:chosen.append((f"test_DS3_{n:02d}F",candidates[len(candidates)//2]))
    worst=max((x for x in rows if x["DS_ref"]>2),key=lambda x:x["u_relative_l2"]+x["a_relative_l2"])
    chosen.append(("test_DS3_worst",worst))
    for filename,row in chosen:
        i=test_batch["case_ids"].index(row["id"]);n=row["stories"]
        fig,axes=plt.subplots(n,2,figsize=(14,2*n),squeeze=False,sharex=True,constrained_layout=True)
        for f in range(n):
            for col,(key,scale,unit) in enumerate((("displacement",1000.,"mm"),("acceleration",1/9.80665,"g"))):
                truth=test_batch[key][i,:,f].cpu().numpy()*scale;p=pred[key][i,:,f].cpu().numpy()*scale;t=np.arange(len(truth))*.02
                axes[f,col].plot(t,truth,color="#303030",lw=.8,label="C++ reference")
                axes[f,col].plot(t,p,color="#d6604d",lw=.7,alpha=.8,label="Unseen-record prediction")
                axes[f,col].set_ylabel(f"Floor {f+1} {unit}");axes[f,col].set_xlim(0,75)
        axes[0,0].set_title("Displacement");axes[0,1].set_title("Absolute acceleration");axes[0,0].legend(fontsize=7)
        for a in axes[-1]:a.set_xlabel("Time (s)")
        fig.suptitle(f"TEST | {row['id']} | true DS={row['DS_ref']} | u L2={row['u_relative_l2']:.1%}, a L2={row['a_relative_l2']:.1%}")
        fig.savefig(folder/f"{filename}.png",dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(_,row) in zip(axes,chosen[:3]):
        meta=entry[row["id"]]
        with np.load(test_data.manifest.parent/meta["file"],allow_pickle=False) as d:
            floor=row["true_critical_story"]-1;u=d["displacement"]
            deform=u[:,floor]-(u[:,floor-1] if floor else 0.)
            ax.plot(deform*1000,d["story_force"][:,floor]/1e6,color="#2166ac",lw=.8)
            mat=np.asarray(meta["source_config"]["model"]["material"]);uy=mat[floor,1]/mat[floor,0]
            z=np.linspace(-1.5*uy,1.5*uy,30);ax.plot(z*1000,z*mat[floor,0]/1e6,"--",color="gray",label="Initial stiffness")
            ax.set(xlabel="Story deformation (mm)",ylabel="Story force (MN)",title=f"{row['stories']}F, story {floor+1}, DS={row['DS_ref']}")
            ax.legend(fontsize=7)
    fig.suptitle("Reference hysteresis for the displayed nonlinear test cases (50 Hz saved histories)")
    fig.savefig(folder/"reference_hysteresis.png",dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/peer10_train.json");args=p.parse_args()
    conf=json.loads((STAGE/args.config).read_text());device=torch.device(conf["device"]);torch.set_num_threads(6)
    # Strict FP32 reduces batching-dependent roundoff in decoder equivalence checks.
    torch.backends.cuda.matmul.allow_tf32=False
    run=STAGE/"runs"/conf["run_id"];report=STAGE/"reports"/conf["run_id"]
    if (report/"result.json").exists():raise RuntimeError("Final test already evaluated; preserve its frozen result")
    state=torch.load(run/"best.pt",map_location=device,weights_only=False)
    model=SeismicQueryModel(ModelConfig(**state["provenance"]["model"])).to(device);model.load_state_dict(state["model"]);model.eval()
    provenance=state["provenance"]
    result={"best_epoch":state["epoch"],"checkpoint_sha256":hashlib.sha256((run/"best.pt").read_bytes()).hexdigest(),
            "provenance":provenance,"evaluation_git_sha":__import__("subprocess").check_output(["git","rev-parse","HEAD"],cwd=STAGE.parent,text=True).strip(),
            "splits":{},"test_tuning":"None; selected on one predeclared validation record"}
    record_sets=[];datafolder=STAGE/conf["dataset"]
    for split in ("train","validation","test"):
        d,_,b=load_split(datafolder,split,device);entries={x["id"]:x for x in d.cases}
        record_sets.append({x["record_id"] for x in d.cases});pred=predict(model,b)
        rows=measure(pred,b,entries);result["splits"][split]={"summary":grouped(rows),"cases":rows}
        if split=="test":test_data,test_batch,test_pred=d,b,pred
    assert all(not record_sets[i]&record_sets[j] for i in range(3) for j in range(i))
    result["record_split_disjoint"]=True
    result["query_check"]=query_check(model,test_batch)
    ts=result["splits"]["train"]["summary"];test=result["splits"]["test"]["summary"]
    fit=all(ts[g][f"{c}_median_relative_l2"]<=.1 for g in ("all","DS_gt_2") for c in ("u","a"))
    nonlinear_pass=passes(test["DS_gt_2"]) and all(passes(test["by_stories"][str(n)]["DS_gt_2"]) for n in (3,5,8))
    result["criteria"]={"training_fit":fit,"test_all":passes(test["all"]),"test_DS_gt_2_each_height":nonlinear_pass,
                        "query_interface":True,"threshold_source":"PEER10_PLAN.md frozen before training"}
    result["status"]="PASS" if fit and passes(test["all"]) and nonlinear_pass else "PARTIAL"
    np.savez_compressed(run/"test_predictions.npz",displacement=test_pred["displacement"].cpu().numpy(),acceleration=test_pred["acceleration"].cpu().numpy(),
                        truth_displacement=test_batch["displacement"].cpu().numpy(),truth_acceleration=test_batch["acceleration"].cpu().numpy(),
                        case_ids=np.asarray(test_batch["case_ids"]),floor_mask=test_batch["floor_mask"].cpu().numpy())
    curve=json.loads((run/"curve.json").read_text());plots(report,curve,result,test_batch,test_pred,test_data)
    lines=["# PEER-10：3/5/8 层非线性查询结果","",f"状态：**{result['status']}**。模型从头联合训练；checkpoint 只由验证波选择。","",
           "训练 peer_1～7；验证 peer_8；测试 peer_9～10。同条波的结构/调幅版本不跨组。记录无事件/台站标识，未证明事件独立。","",
           "| 分区/子集 | 算例 | 位移 L2 中位数 | 加速度 L2 中位数 | PIDR 楼层误差中位数 | PFA 楼层误差中位数 |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("train","validation","test"):
        for group in ("all","DS_gt_2"):
            s=result["splits"][split]["summary"][group]
            lines.append(f"| {split} / {group} | {s['count']} | {s['u_median_relative_l2']:.2%} | {s['a_median_relative_l2']:.2%} | {s['PIDR_floor_median_relative_error']:.2%} | {s['PFA_floor_median_relative_error']:.2%} |")
    lines += ["","## 未见波、DS>2，按层数","","| 层数 | 算例 | 位移 L2 中位数 | 加速度 L2 中位数 | 位移/加速度 MSE 相对基线改善 |","| --- | ---: | ---: | ---: | --- |"]
    for n in (3,5,8):
        s=test["by_stories"][str(n)]["DS_gt_2"]
        lines.append(f"| {n} | {s['count']} | {s['u_median_relative_l2']:.2%} | {s['a_median_relative_l2']:.2%} | {s['u_mse_improvement_vs_baseline']:.2%} / {s['a_mse_improvement_vs_baseline']:.2%} |")
    lines += ["",f"独立查询核验通过：{result['query_check']['test_buildings']} 个测试建筑、{result['query_check']['single_floor_queries']} 次逐层查询；乱序查询也通过。它证明接口一致性，预测准确度由上表判定。",
              "","DS>2 是既有参考阈值下 PIDR>2%，DS≤2 不代表线性；数据另存屈服变形比、逐层剪力—变形历史。下方滞回图是真值的非线性核验，不是模型预测的剪力。",
              "","![训练与验证](learning_curve.png)","","![真值滞回](reference_hysteresis.png)",""]
    for n in (3,5,8):lines += [f"![{n} 层非线性测试](test_DS3_{n:02d}F.png)",""]
    lines += ["![最差非线性测试](test_DS3_worst.png)","",
              "图中每类层数选 DS>2 测试算例中两通道 L2 之和居中的算例；另显示最差算例。全时程 0–75 s，包含衰减段。逐层误差、峰值、P90/最差、基线改善均见 result.json。",
              "","本轮只验证已见层数与结构配置的未见地震动，不检验未见层数。测试仅两条基础波；未用测试调参。若指标未通过，不据此宣称非线性泛化已解决。"]
    json_write(report/"result.json",result);(report/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"best_epoch":state["epoch"],"criteria":result["criteria"],"query_check":result["query_check"],
                      "test":test["all"],"test_DS_gt_2":test["DS_gt_2"],"nonlinear_by_stories":{k:v["DS_gt_2"] for k,v in test["by_stories"].items()}}),flush=True)


if __name__=="__main__":main()
