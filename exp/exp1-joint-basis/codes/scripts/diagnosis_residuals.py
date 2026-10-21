"""Physical-frequency and patch-edge error audit of saved training predictions."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
import numpy as np
from peer10_common import STAGE,load_split,json_write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    folder=STAGE/'reports'/args.run;r=json.loads((folder/'result.json').read_text())
    _,_,b=load_split(STAGE/r['provenance']['config']['dataset'],'train','cpu')
    results={}
    with np.load(STAGE/'runs'/args.run/'train_predictions.npz',allow_pickle=False) as f:
        for key in ('displacement','acceleration'):
            band_sums=np.zeros((5,2));edge_sums=np.zeros((2,2));per_floor={}
            for j,name in enumerate(f['case_ids']):
                i=b['case_ids'].index(str(name));n=int(b['attributes'][i,0]);t=int(b['time_mask'][i].sum())
                truth=b[key][i,:t,:n].numpy();error=f[key][j,:t,:n]-truth
                tf=np.fft.rfft(truth,axis=0,norm='ortho');ef=np.fft.rfft(error,axis=0,norm='ortho');freq=np.fft.rfftfreq(t,d=.02)
                weights=np.full(len(freq),2.);weights[0]=1.
                if t%2==0:weights[-1]=1.
                for k,(lo,hi) in enumerate(((0,.5),(.5,2),(2,5),(5,10),(10,25.00001))):
                    mask=(freq>=lo)&(freq<hi);band_sums[k]+=[np.sum(abs(tf[mask])**2*weights[mask,None]),np.sum(abs(ef[mask])**2*weights[mask,None])]
                edge=(np.arange(t)%60<3)|(np.arange(t)%60>=57)
                for k,mask in enumerate((~edge,edge)):edge_sums[k]+=[float((error[mask]**2).sum()),int(mask.sum())*n]
                for level in range(n):per_floor.setdefault(f'{n}F_floor{level+1}',[]).append(float(np.linalg.norm(error[:,level])/max(np.linalg.norm(truth[:,level]),1e-12)))
            results[key]={'bands':[{'Hz':label,'truth_energy_fraction':float(v[0]/band_sums[:,0].sum()),'band_relative_l2':float(np.sqrt(v[1]/max(v[0],1e-15))),'error_energy_fraction':float(v[1]/band_sums[:,1].sum())} for label,v in zip(('0-.5','.5-2','2-5','5-10','10-25'),band_sums)],
                          'patch_edge_vs_interior_mse_ratio':float((edge_sums[1,0]/edge_sums[1,1])/(edge_sums[0,0]/edge_sums[0,1])),
                          'median_per_floor_l2':{k:float(np.median(v)) for k,v in per_floor.items()}}
    json_write(folder/'residual_audit.json',results);print(json.dumps(results),flush=True)


if __name__=='__main__':main()
