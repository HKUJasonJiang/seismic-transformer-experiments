"""Generate 24 cross-height fit-only cases on 5090 using the production C++ DLL."""
from __future__ import annotations

import sys
sys.dont_write_bytecode=True
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from five_story_preview import STAGE, PROJECT, baseline, case_config, run_case, load_record, periods, write_json


def main():
    base=json.loads((STAGE/"configs/five_story_preview.json").read_text(encoding="utf-8"))
    folder=STAGE/"data/poc_fit24"
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for stories in [3,5,8,12]:
        config=copy.deepcopy(base)
        config["stories"]=stories
        material,source=baseline(config,STAGE/f"runs/poc_fit24_generation/{stories}F")
        gm=load_record(PROJECT/config["ground_motion_relative_path"],unit="m/s^2")
        for ratio in [.8,1.,1.2]:
            for pga in [.1,.35]:
                name=f"{stories:02d}F_r{ratio:.2f}_pga{pga:g}g"
                path=folder/f"{name}.npz"
                meta_path=folder/f"{name}.json"
                if path.exists() and meta_path.exists():
                    records.append(json.loads(meta_path.read_text(encoding="utf-8")))
                    print(json.dumps({"id":name,"status":"reused"}),flush=True)
                    continue
                c=case_config(config,material,gm,ratio,pga,source)
                # All legacy compatibility semantics remain explicit in source metadata.
                result=run_case(c)
                failure=bool(result.material_dead.any())
                if failure:
                    item={"id":name,"status":"excluded_material_failure","stories":stories,
                          "T1_s":float(result.periods[0]),"pga_g":pga,
                          "max_PIDR":float(result.edp["maximum_interstory_drift_ratio"])}
                    write_json(meta_path,item)
                    records.append(item)
                    print(json.dumps(item),flush=True)
                    continue
                # Restore displacement to physical current time; acceleration is already current.
                u=np.vstack((np.zeros((1,stories)),result.displacement[:-1]))
                down=round(.02/config["integration_dt_s"])
                u=resample_poly(u,1,down,axis=0,padtype="line").astype(np.float32)
                a=resample_poly(result.acceleration,1,down,axis=0,padtype="line").astype(np.float32)
                time=np.arange(len(u))*.02
                g=np.interp(time,c["ground_motion"]["time"],c["ground_motion"]["acceleration"],right=0.)
                heights=np.asarray(c["model"]["height"],dtype=np.float32)
                np.savez_compressed(path,ground=g.astype(np.float32)[:,None],
                                    attributes=np.asarray([stories,heights.sum(),config["fortification_intensity"],result.periods[0]],dtype=np.float32),
                                    structure_type=np.asarray(config["structure_type"],dtype=np.int64),
                                    displacement=u,acceleration=a,story_heights=heights)
                item={"id":name,"status":"complete","file":path.name,"stories":stories,
                      "T1_s":float(result.periods[0]),"pga_g":pga,"period_ratio":ratio,
                      "max_PIDR_integral":float(result.edp["maximum_interstory_drift_ratio"]),
                      "material_failure":False,"config_hash":result.metadata["config_hash"],
                      "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "source_config":c,"sampling":"50 Hz, anti-aliased from 0.00025 s integration; displacement physically reindexed"}
                write_json(meta_path,item)
                records.append(item)
                print(json.dumps({k:item[k] for k in ("id","stories","T1_s","max_PIDR_integral","status")}),flush=True)
    manifest={"purpose":"Training-set learnability only; same PEER-1 base waveform in all cases; no generalization claim",
              "sample_rate_hz":50,"planned_cases":24,"valid_cases":sum(x["status"]=="complete" for x in records),
              "cases":records}
    write_json(folder/"manifest.json",manifest)
    (folder/"README.md").write_text("# Stage 1 small-sample fit data\n\n"
        "New C++ simulations of 3/5/8/12-story buildings, three period variants and two PGA levels. "
        "All use the same processed PEER-1 waveform. This is a training-only learnability experiment, "
        "not a train/validation/test benchmark. Physical units and source configurations accompany every NPZ.\n",encoding="utf-8")
    print(json.dumps({"manifest":str(folder/"manifest.json"),"valid_cases":manifest["valid_cases"]}),flush=True)


if __name__=="__main__":
    main()
