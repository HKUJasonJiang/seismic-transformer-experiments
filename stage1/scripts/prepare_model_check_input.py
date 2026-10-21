"""Prepare three actual five-story cases for O2; run in the MDOF environment.

These inputs are only for interface checks, not a train/validation/test split.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly

STAGE = Path(__file__).resolve().parents[1]


def main():
    ids = ["r0.80_pga0.1g", "r1.00_pga0.35g", "r1.20_pga0.6g"]
    arrays = {key: [] for key in ("ground", "attributes", "structure_type", "floor_ids",
                                 "story_heights", "displacement", "acceleration")}
    provenance = []
    for case_id in ids:
        path = STAGE / "data/five_story_preview" / f"{case_id}.h5"
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["metadata_json"])
            c = meta["source_config"]
            dt = meta["integration_dt"]
            down = round(.02/dt)
            if not np.isclose(down*dt, .02):
                raise ValueError("This check requires integral resampling to 50 Hz")
            u, a = f["displacement"][...], f["acceleration"][...]
            if meta["time_alignment"] == "legacy":
                u = np.vstack((np.zeros((1,u.shape[1])), u[:-1]))
            elif meta["time_alignment"] != "physical":
                raise ValueError("Unknown time alignment")
            u = resample_poly(u, 1, down, axis=0, padtype="line")
            a = resample_poly(a, 1, down, axis=0, padtype="line")
            t = np.arange(len(u))*.02
            ground = np.interp(t, c["ground_motion"]["time"], c["ground_motion"]["acceleration"], right=0.)[:,None]
            heights = np.asarray(c["model"]["height"])
            n = len(heights)
            attrs = c["provenance"]["building_attributes"]
            arrays["ground"].append(ground)
            arrays["displacement"].append(u)
            arrays["acceleration"].append(a)
            arrays["attributes"].append([n, heights.sum(), attrs["fortification_intensity"], f["periods"][0]])
            arrays["structure_type"].append(attrs["structure_type"])
            arrays["floor_ids"].append(np.arange(1,n+1))
            arrays["story_heights"].append(heights)
            provenance.append({"case_id":case_id,"source":str(path),
                               "source_config_hash":meta["config_hash"],"source_integration_dt":dt,
                               "source_peaks":"Integration-grid peak labels excluded from waveform loss",
                               "response_processing":"Legacy displacement reindexed, then scipy resample_poly anti-aliasing to 50 Hz"})
    arrays = {key: np.asarray(value, dtype=np.int64 if key in ("floor_ids", "structure_type") else np.float32)
              for key,value in arrays.items()}
    arrays["time_mask"] = np.ones(arrays["ground"].shape[:2], dtype=bool)
    arrays["floor_mask"] = arrays["floor_ids"] > 0
    arrays["case_ids"] = np.asarray(ids)
    path = STAGE/"data/model_check/input.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    report = {"purpose":"O2 real-data forward/backward only; not a generalization experiment",
              "sample_rate_hz":50,"ground_shape":list(arrays["ground"].shape),
              "response_shape":list(arrays["displacement"].shape),"cases":provenance,
              "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (STAGE/"reports/model_check_input.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(path),"ground_shape":report["ground_shape"]}))


if __name__ == "__main__":
    main()
