"""Remote-only PEER-10 simulation, predeclared record splits and nonlinear coverage."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import sys
sys.dont_write_bytecode=True
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.signal import resample_poly
from five_story_preview import STAGE,PROJECT,baseline,case_config,run_case,load_record,periods,write_json


def audit(conf,folder):
    path=folder/"record_split.json"
    if path.exists():return json.loads(path.read_text())
    selected=[];signals=[];skipped=[]
    library=PROJECT/"mdof-workbench/data/GM/PEER"
    for p in sorted(library.glob("peer_*.txt"),key=lambda x:int(x.stem.split("_")[-1])):
        gm=load_record(p,unit="m/s^2")
        z=gm.acceleration-gm.acceleration.mean()
        z=z/np.linalg.norm(z)
        correlations=[float(abs(z@q)) for q in signals]
        if correlations and max(correlations)>conf["normalized_duplicate_correlation"]:
            skipped.append({"record":p.stem,"near_duplicate_of":selected[int(np.argmax(correlations))]["record_id"],"correlation":max(correlations)})
            continue
        selected.append({"record_id":p.stem,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                         "source_pga_m_s2":float(np.abs(gm.acceleration).max())})
        signals.append(z)
        if len(selected)==conf["record_count"]:break
    if len(selected)!=10:raise ValueError("Need ten nonduplicate processed PEER records")
    ntrain,nval,ntest=conf["split_record_counts"]
    for i,x in enumerate(selected):x["split"]="train" if i<ntrain else "validation" if i<ntrain+nval else "test"
    result={"records":selected,"near_duplicates_skipped":skipped,"event_metadata":"unavailable; record-level independence only",
            "pairwise_max_absolute_centered_correlation":float(np.max(np.abs(np.asarray(signals)@np.asarray(signals).T-np.eye(10)))),
            "policy":conf,"selection":"number order; before any response simulation or training"}
    write_json(path,result)
    return result


def generate_group(task):
    conf,record,n,ratio,material,source=task
    base=json.loads((STAGE/"configs/five_story_preview.json").read_text())
    base["stories"]=n
    folder=STAGE/"data"/conf["dataset_id"]
    group=f"{record['record_id']}_{n:02d}F_r{ratio:.2f}"
    marker=folder/f"{group}_group.json"
    if marker.exists():
        old=json.loads(marker.read_text())
        for x in old["cases"]:
            if x["status"]=="complete":
                if hashlib.sha256((folder/x["file"]).read_bytes()).hexdigest()!=x["npz_sha256"]:raise ValueError("Cached case hash mismatch")
        return old
    gm=load_record(PROJECT/f"mdof-workbench/data/GM/PEER/{record['record_id']}.txt",unit="m/s^2")
    rows=[];trials=[];known=[]
    def simulate(pga,save,label):
        pga=float(pga)
        c=case_config(base,np.asarray(material),gm,ratio,pga,source)
        name=f"{group}_{label}_pga{pga:.6f}g"
        c["name"]=name
        start=time.perf_counter()
        r=run_case(c)
        u=np.vstack((np.zeros((1,n)),r.displacement[:-1]))
        deformation=np.diff(u,axis=1,prepend=np.zeros((len(u),1)))
        mat=np.asarray(c["model"]["material"])
        ductility=float(np.max(np.abs(deformation)/(mat[:,1]/mat[:,0])))
        peak=float(np.max(np.abs(deformation)/np.asarray(c["model"]["height"])))
        dead=bool(r.material_dead.any())
        finite=all(np.isfinite(x).all() for x in (u,r.acceleration,r.velocity,r.story_force))
        expected=periods(np.asarray(c["model"]["mass"]),mat[:,0])[0]
        if not np.isclose(r.periods[0],expected,rtol=1e-10):raise ValueError("Independent T1 check failed")
        info={"id":name,"record_id":record["record_id"],"split":record["split"],"stories":n,"period_ratio":ratio,
              "pga_g":pga,"T1_s":float(r.periods[0]),"max_PIDR_integral":peak,"ductility_max":ductility,
              "DS_ref_integral":int(np.sum(peak>np.asarray(base["reference_drift_thresholds"]))),
              "material_failure":dead,"status":"complete" if finite and not dead else "excluded_material_failure" if dead else "excluded_nonfinite",
              "wall_seconds":time.perf_counter()-start}
        trials.append({**info,"retained":save})
        known.append((pga,peak,finite and not dead))
        if not save:return info
        if info["status"]=="complete":
            down=round(.02/base["integration_dt_s"])
            us=resample_poly(u,1,down,axis=0,padtype="line").astype(np.float32)
            acc=resample_poly(r.acceleration,1,down,axis=0,padtype="line").astype(np.float32)
            vel=resample_poly(r.velocity,1,down,axis=0,padtype="line").astype(np.float32)
            force=resample_poly(r.story_force,1,down,axis=0,padtype="line").astype(np.float32)
            t=np.arange(len(us))*.02
            ground=np.interp(t,c["ground_motion"]["time"],c["ground_motion"]["acceleration"],right=0.)
            heights=np.asarray(c["model"]["height"],dtype=np.float32)
            pidr50=float(np.max(np.abs(np.diff(us,axis=1,prepend=np.zeros((len(us),1))))/heights))
            f=folder/f"{name}.npz"
            # Additional fields support physical/hysteresis checks; model collate only uses declared inputs.
            np.savez_compressed(f,ground=ground.astype(np.float32)[:,None],
                                attributes=np.asarray([n,heights.sum(),base["fortification_intensity"],r.periods[0]],dtype=np.float32),
                                structure_type=np.asarray(base["structure_type"],dtype=np.int64),
                                displacement=us,acceleration=acc,velocity=vel,story_force=force,story_heights=heights)
            info.update(file=f.name,npz_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                        max_PIDR_50Hz=pidr50,DS_ref=int(np.sum(pidr50>np.asarray(base["reference_drift_thresholds"]))),
                        config_hash=r.metadata["config_hash"],source_config=c,model_version=r.metadata["model_version"],
                        T1_relative_eigen_error=abs(float(r.periods[0])/expected-1),
                        peak_PFA_integral_m_s2=float(np.abs(r.acceleration).max()))
        write_json(folder/f"{name}.json",info)
        rows.append(info)
        return info
    for pga in conf["fixed_pga_g"]:simulate(pga,True,"fixed")
    lo,hi=conf["strong_accept_PIDR"]
    def strong_present():return any(x["status"]=="complete" and x.get("DS_ref",0)>2 and x.get("max_PIDR_50Hz",0)<hi for x in rows)
    if not strong_present():
        for attempt in range(conf["maximum_search_trials"]):
            safe_low=[(p,d) for p,d,ok in known if ok and d<conf["strong_target_PIDR"]]
            high=[(p,d) for p,d,ok in known if not ok or d>=conf["strong_target_PIDR"]]
            if safe_low and high:
                lower=max(safe_low,key=lambda x:x[0])[0]
                upper=min((p for p,d in high if p>lower),default=None)
                pga=math.sqrt(lower*upper) if upper else lower*1.5
            elif safe_low:pga=max(p for p,d in safe_low)*1.5
            else:pga=min(p for p,d,ok in known)*.6
            pga=min(pga,conf["maximum_search_pga_g"])
            if any(abs(pga-p)<1e-6 for p,d,ok in known):break
            item=simulate(pga,False,f"pilot{attempt}")
            if item["status"]=="complete" and lo<=item["max_PIDR_integral"]<=hi:
                simulate(pga,True,"strong")
                if strong_present():break
    answer={"group":group,"record_id":record["record_id"],"split":record["split"],"stories":n,"period_ratio":ratio,
            "strong_coverage":strong_present(),"cases":rows,"trials":trials}
    write_json(marker,answer)
    return answer


def main():
    p=argparse.ArgumentParser();p.add_argument("--audit-only",action="store_true");args=p.parse_args()
    conf=json.loads((STAGE/"configs/peer10_data.json").read_text())
    folder=STAGE/"data"/conf["dataset_id"];folder.mkdir(parents=True,exist_ok=True)
    split=audit(conf,folder)
    print(json.dumps({"record_split":split}),flush=True)
    if args.audit_only:return
    tasks=[]
    for n in conf["stories"]:
        base=json.loads((STAGE/"configs/five_story_preview.json").read_text());base["stories"]=n
        mat,source=baseline(base,STAGE/f"runs/{conf['dataset_id']}_generation/{n}F")
        for record in split["records"]:
            for ratio in conf["period_ratios"]:tasks.append((conf,record,n,ratio,mat.tolist(),source))
    groups=[]
    with ProcessPoolExecutor(max_workers=conf["workers"]) as executor:
        futures=[executor.submit(generate_group,t) for t in tasks]
        for future in as_completed(futures):
            group=future.result();groups.append(group)
            print(json.dumps({"group":group["group"],"finished":len(groups),"total":len(tasks),"strong_coverage":group["strong_coverage"],
                              "valid":sum(x["status"]=="complete" for x in group["cases"])}),flush=True)
    allrows=sorted([x for g in groups for x in g["cases"]],key=lambda x:x["id"])
    common={"sample_rate_hz":50,"purpose":"Joint 3/5/8-story prediction and unseen-record nonlinear evaluation",
            "record_split":split,"configuration":conf,"full_floor_supervision":True,
            "strong_coverage_groups":sum(g["strong_coverage"] for g in groups),"total_groups":len(groups),
            "coverage_gaps":[g["group"] for g in groups if not g["strong_coverage"]]}
    write_json(folder/"manifest.json",{**common,"cases":allrows})
    for name in ("train","validation","test"):
        subset=[x for x in allrows if x["split"]==name]
        write_json(folder/f"{name}.json",{**common,"split":name,"cases":subset})
        valid=[x for x in subset if x["status"]=="complete"]
        print(json.dumps({"split":name,"valid":len(valid),"excluded":len(subset)-len(valid),
                          "DS_gt_2":sum(x["DS_ref"]>2 for x in valid),
                          "nonlinear_by_stories":{str(n):sum(x["DS_ref"]>2 and x["stories"]==n for x in valid) for n in conf["stories"]}}),flush=True)
    (folder/"README.md").write_text("# PEER-10 nonlinear query dataset\n\n7/1/2 base-record split; 3/5/8 stories. "
        "All scaling variants of a base record stay in one split. Constitutive-failure cases are excluded from ordinary response targets. "
        "DS_ref>2 means peak saved-grid drift>2%; record-level separation does not establish event-level separation.\n",encoding="utf-8")


if __name__=="__main__":main()
