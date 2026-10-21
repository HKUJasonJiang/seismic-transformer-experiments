"""Read NPZ cases and pad variable floor/time counts; no simulation dependency."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class ResponseDataset(torch.utils.data.Dataset):
    def __init__(self,manifest):
        self.manifest=Path(manifest)
        self.metadata=json.loads(self.manifest.read_text(encoding="utf-8"))
        self.cases=[c for c in self.metadata["cases"] if c["status"]=="complete"]
        if not self.cases:
            raise ValueError("No valid simulation cases")
        self.items=[]
        for entry in self.cases:
            path=self.manifest.parent/entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["npz_sha256"]:
                raise ValueError(f"Dataset hash mismatch: {path}")
            with np.load(path,allow_pickle=False) as f:
                item={k:torch.from_numpy(f[k].copy()) for k in f.files}
            item["case_id"]=entry["id"]
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        return self.items[index]


def collate_responses(items):
    b=len(items)
    t=max(len(x["ground"]) for x in items)
    n=max(len(x["story_heights"]) for x in items)
    output={"ground":torch.zeros(b,t,1),"attributes":torch.stack([x["attributes"] for x in items]),
            "structure_type":torch.stack([x["structure_type"] for x in items]).long(),
            "displacement":torch.zeros(b,t,n),"acceleration":torch.zeros(b,t,n),
            "story_heights":torch.zeros(b,n),"floor_ids":torch.zeros(b,n,dtype=torch.long),
            "floor_mask":torch.zeros(b,n,dtype=torch.bool),"time_mask":torch.zeros(b,t,dtype=torch.bool),
            "case_ids":[x["case_id"] for x in items]}
    for i,item in enumerate(items):
        ti,ni=len(item["ground"]),len(item["story_heights"])
        output["ground"][i,:ti]=item["ground"]
        for key in ("displacement","acceleration"):
            output[key][i,:ti,:ni]=item[key]
        output["story_heights"][i,:ni]=item["story_heights"]
        output["floor_ids"][i,:ni]=torch.arange(1,ni+1)
        output["floor_mask"][i,:ni]=True
        output["time_mask"][i,:ti]=True
    return output
