"""Remote data/metric QA and three refined-step strong-nonlinear reference checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import copy
import hashlib
import json
import numpy as np
import torch
from scipy.signal import resample_poly
from peer10_common import STAGE,measure,aggregate,json_write
from five_story_preview import periods,run_case


def main():
    folder=STAGE/"data/peer10_nonlinear_v1"
    manifest=json.loads((folder/"manifest.json").read_text())
    rows=[x for x in manifest["cases"] if x["status"]=="complete"]
    maximum_period_error=0.;sampling_error=[]
    for row in rows:
        path=folder/row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row["npz_sha256"]
        with np.load(path,allow_pickle=False) as f:
            n=row["stories"]
            assert f["ground"].shape==(3751,1) and f["displacement"].shape==(3751,n)
            assert all(np.isfinite(f[k]).all() for k in f.files)
            assert int(f["attributes"][0])==n and np.isclose(f["attributes"][1],f["story_heights"].sum())
            pidr=float(np.max(np.abs(np.diff(f["displacement"],axis=1,prepend=np.zeros((3751,1))))/f["story_heights"]))
            assert row["DS_ref"]==int((pidr>np.asarray([.005,.008,.02,.05])).sum())
            assert np.isclose(np.abs(f["ground"]).max()/9.80665,row["pga_g"],rtol=1e-6)
            sampling_error.append(abs(pidr/row["max_PIDR_integral"]-1))
        c=row["source_config"];mat=np.asarray(c["model"]["material"])
        period=periods(np.asarray(c["model"]["mass"]),mat[:,0])[0]
        maximum_period_error=max(maximum_period_error,abs(period/row["T1_s"]-1))
    sets=[{r["record_id"] for r in rows if r["split"]==s} for s in ("train","validation","test")]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    # Analytic metric contract: exact prediction must improve both nonzero baselines by 100%.
    fm=torch.ones(1,3,dtype=torch.bool);tm=torch.ones(1,20,dtype=torch.bool)
    u=torch.arange(20).float()[None,:,None].expand(1,20,3)*.01
    a=torch.sin(torch.arange(20).float())[None,:,None].expand(1,20,3)
    batch={"case_ids":["analytic"],"floor_mask":fm,"time_mask":tm,"story_heights":torch.full((1,3),3.),
           "displacement":u,"acceleration":a,"ground":torch.ones(1,20,1)}
    meta={"analytic":{"id":"analytic","record_id":"analytic","stories":3,"period_ratio":1.,"pga_g":.1,"DS_ref":3,"ductility_max":2.}}
    metrics=aggregate(measure({"displacement":u,"acceleration":a},batch,meta))
    for key in ("u","a"):
        assert metrics[f"{key}_median_relative_l2"]==0 and metrics[f"{key}_mse_improvement_vs_baseline"]==1
    for key in ("PIDR","PFA"):assert metrics[f"{key}_floor_median_relative_error"]==0
    refined=[]
    for n in (3,5,8):
        candidates=[r for r in rows if r["split"]=="train" and r["record_id"]=="peer_1" and r["stories"]==n and r["period_ratio"]==1. and r["DS_ref"]>2]
        if not candidates:raise ValueError("Missing predeclared nonlinear refinement case")
        row=min(candidates,key=lambda r:abs(r["max_PIDR_50Hz"]-.03))
        c=copy.deepcopy(row["source_config"]);c["analysis"]["dt"]*=.5;c["name"]+="_refined_dt"
        result=run_case(c)
        assert not result.material_dead.any()
        u=np.vstack((np.zeros((1,n)),result.displacement[:-1]))
        pidr=float((np.abs(np.diff(u,axis=1,prepend=np.zeros((len(u),1))))/np.asarray(c["model"]["height"])).max())
        pfa=float(np.abs(result.acceleration).max())
        down=round(.02/c["analysis"]["dt"])
        ur=resample_poly(u,1,down,axis=0,padtype="line")
        ar=resample_poly(result.acceleration,1,down,axis=0,padtype="line")
        with np.load(folder/row["file"],allow_pickle=False) as original:
            item={"case_id":row["id"],"stories":n,"original_dt":row["source_config"]["analysis"]["dt"],"refined_dt":c["analysis"]["dt"],
                  "PIDR_relative_difference":abs(pidr/row["max_PIDR_integral"]-1),
                  "PFA_relative_difference":abs(pfa/row["peak_PFA_integral_m_s2"]-1),
                  "u_waveform_relative_difference":float(np.linalg.norm(ur-original["displacement"])/np.linalg.norm(ur)),
                  "a_waveform_relative_difference":float(np.linalg.norm(ar-original["acceleration"])/np.linalg.norm(ar))}
        refined.append(item);print(json.dumps({"refinement":item}),flush=True)
    trials=[t for p in folder.glob("*_group.json") for t in json.loads(p.read_text())["trials"]]
    report={"status":"PASS","cases_checked":len(rows),"record_split_disjoint":True,"metric_analytic_check":"PASS",
            "strong_coverage_groups":manifest["strong_coverage_groups"],"total_groups":manifest["total_groups"],
            "nonlinear_ductility_range":[min(r["ductility_max"] for r in rows if r["DS_ref"]>2),max(r["ductility_max"] for r in rows if r["DS_ref"]>2)],
            "max_relative_eigen_error":maximum_period_error,"PIDR_saved_grid_peak_relative_difference_max":max(sampling_error),
            "simulation_trials":len(trials),"failed_trials":sum(t["status"]!="complete" for t in trials),
            "refined_step_cases":refined,"refinement_scope":"Three predeclared training strong-nonlinear cases; not every record"}
    if any(max(x["PIDR_relative_difference"],x["PFA_relative_difference"])>.1 for x in refined):report["status"]="REVIEW_NUMERICAL_SENSITIVITY"
    json_write(STAGE/"reports/peer10_data_quality.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
