"""Export fit-passing deployable model and verify physical-input-only inference."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import numpy as np
import torch
from peer10_common import STAGE,load_split,slice_batch,json_write,predict
from src.nonlinear_inference import load_query_model
from evaluate_peer10 import query_check


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    folder=STAGE/'runs'/args.run;report=STAGE/'reports'/args.run;r=json.loads((report/'result.json').read_text())
    assert r['status']=='FIT_PASS' and r['provenance']['deployable_inputs']
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name():raise RuntimeError('RTX 5090 required')
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert all(k.startswith('base.') for k in state['model'])
    path=folder/'inference.pt'
    torch.save({'model':{k[5:]:v for k,v in state['model'].items()},'variant':r['arm'],'provenance':state['provenance'],
                'scope':'Training-fit checkpoint; no independent generalization validation'},path)
    model=load_query_model(path,'cuda:0')
    _,_,b=load_split(STAGE/r['provenance']['config']['dataset'],'train','cuda:0')
    with np.load(folder/'train_predictions.npz',allow_pickle=False) as saved:
        ix=torch.tensor([b['case_ids'].index(str(x)) for x in saved['case_ids']],device='cuda:0');b=slice_batch(b,ix)
        out=predict(model,b);maximum={}
        for key in ('displacement','acceleration'):
            reference=torch.tensor(saved[key],device='cuda:0');maximum[key]=float((out[key]-reference).abs().max())
            torch.testing.assert_close(out[key],reference,atol=2e-6 if key=='displacement' else 2e-4,rtol=1e-4)
    # This imported helper operates only on the supplied TRAINING batch.
    q=query_check(model,b);q['training_buildings']=q.pop('test_buildings')
    result={'status':'PASS','input_contract':['ground','attributes','structure_type','floor_ids','time_mask'],
            'requires_response_labels':False,'requires_case_identity':False,'requires_true_roof_curve':False,
            'inference_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'maximum_export_difference_SI':maximum,'training_query_check':q,
            'scope':'Train-only checkpoint export; this script never loads a validation/test split'}
    json_write(report/'inference_check.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
