"""One development-validation snapshot AFTER selecting a fit-passing model on train."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import numpy as np
import torch
from peer10_common import STAGE,load_split,slice_batch,predict,measure,grouped,json_write
from src.nonlinear_inference import load_query_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    report=STAGE/'reports'/args.run;r=json.loads((report/'result.json').read_text())
    if r['status']!='FIT_PASS':raise RuntimeError('Training gate must pass before validation')
    if (report/'validation_result.json').exists():raise RuntimeError('Preserve the frozen validation snapshot')
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name():raise RuntimeError('RTX 5090 required')
    path=STAGE/'runs'/args.run/'inference.pt';model=load_query_model(path,'cuda:0')
    folder=STAGE/r['provenance']['config']['dataset'];d,_,b=load_split(folder,'validation','cuda:0')
    chosen=[i for i,x in enumerate(d.cases) if x['DS_ref']>2]
    b=slice_batch(b,torch.tensor(chosen,device='cuda:0'));entries={x['id']:x for x in d.cases}
    assert not set(r['provenance']['train_records']) & {entries[x]['record_id'] for x in b['case_ids']}
    out=predict(model,b);rows=measure(out,b,entries);s=grouped(rows)
    result={'summary':s,'cases':rows,'selection':'Frozen after train-only fitting; no further model changes from this snapshot',
            'scope':'Existing development validation record, DS>2 only; not a new independent final test',
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'validation_manifest_sha256':hashlib.sha256((folder/'validation.json').read_bytes()).hexdigest(),'test':'NOT_READ'}
    json_write(report/'validation_result.json',result)
    np.savez_compressed(STAGE/'runs'/args.run/'validation_predictions.npz',displacement=out['displacement'].cpu().numpy(),acceleration=out['acceleration'].cpu().numpy(),case_ids=np.asarray(b['case_ids']))
    print(json.dumps({'scope':result['scope'],'summary':s['all'],'by_stories':s['by_stories']}),flush=True)


if __name__=='__main__':main()
