"""Verify that the selected floor basis can reconstruct actual nonlinear histories.

This projects true responses ONLY for a representation check. It does not provide
coefficients to the prediction model or count as a prediction result.
"""
import sys
sys.dont_write_bytecode=True
import json
import math
import numpy as np
from peer10_common import STAGE,load_split,json_write


def main():
    data,_,b=load_split(STAGE/'data'/'peer10_nonlinear_v1','train','cpu')
    rows=[];orth=[]
    for i,meta in enumerate(data.cases):
        if meta['DS_ref']<=2:continue
        n=meta['stories'];f=np.arange(1,n+1)[:,None];k=np.arange(1,n+1)[None,:]
        phi=np.sqrt(2/(n+1))*np.sin(math.pi*f*k/(n+1))
        orth.append(float(np.max(np.abs(phi.T@phi-np.eye(n)))))
        row={'id':meta['id'],'stories':n}
        for key in ('displacement','acceleration'):
            truth=b[key][i,:,:n].numpy().astype(np.float64);coeff=truth@phi;reconstructed=coeff@phi.T
            row[key+'_reconstruction_l2']=float(np.linalg.norm(reconstructed-truth)/np.linalg.norm(truth))
            assert row[key+'_reconstruction_l2']<1e-12
        rows.append(row)
    result={'status':'PASS','count':len(rows),'max_orthogonality_error':max(orth),'max_response_reconstruction_l2':max(r[k] for r in rows for k in ('displacement_reconstruction_l2','acceleration_reconstruction_l2')),
            'scope':'True-response projection verifies representation only; no model prediction or generalization claim','cases':rows}
    json_write(STAGE/'reports'/'diagnosis-basis-check'/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
