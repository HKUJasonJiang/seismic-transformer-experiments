"""Metrics, models, plots and training for strictly frozen-module interventions."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from dataclasses import asdict
import hashlib
import json
import math
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
from peer10_common import STAGE,load_split,slice_batch,measure,aggregate,json_write
from src.model import ModelConfig,StructuralEncoder,TemporalEncoder
from src.losses import ResponseScales,per_building_mean,interstory_drift
from src.nonlinear_inference import load_query_model,make_model


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def state_hash(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def subset(b,indices):
    return slice_batch(b,torch.tensor(indices,device=b['ground'].device,dtype=torch.long))


def roof(x,b):
    index=(b['attributes'][:,0].long()-1)[:,None,None].expand(-1,x.shape[1],1)
    return x.gather(2,index)


class RoofHead(nn.Module):
    def __init__(self,config,scales):
        super().__init__();self.p=config.patch_size
        self.net=nn.Sequential(nn.Linear(config.d_model,256),nn.GELU(),nn.Linear(256,2*self.p))
        self.register_buffer('scales',torch.tensor([scales.displacement_m,scales.acceleration_m_s2]))
    def forward(self,z,b):
        v=self.net(z).reshape(len(z),-1,2)[:,:b['ground'].shape[1]]*self.scales
        return {key:torch.where(b['time_mask'][...,None],v[...,i:i+1],0.) for i,key in enumerate(('displacement','acceleration'))}


class AllHead(nn.Module):
    """Eight fixed output slots; diagnostic readout, no floor query inference claim."""
    def __init__(self,config,scales):
        super().__init__();self.heads=nn.ModuleList([RoofHead(config,scales) for _ in range(8)])
    def forward(self,z,b):
        outputs=[h(z,b) for h in self.heads];n=b['floor_mask'].shape[1]
        return {key:torch.where(b['floor_mask'][:,None,:],torch.cat([x[key] for x in outputs],-1)[...,:n],0.) for key in outputs[0]}


class RoofPredictor(nn.Module):
    """An independent fresh encoder plus a simple roof head, no query decoder."""
    def __init__(self,config,scales):
        super().__init__();self.structure_encoder=StructuralEncoder(config);self.encoder=TemporalEncoder(config);self.head=RoofHead(config,scales)
    def forward(self,b):
        s=self.structure_encoder(b['attributes'],b['structure_type']);z,_=self.encoder(b['ground'],s,b['time_mask'])
        return self.head(z,b)


class FullPredictor(nn.Module):
    def __init__(self,config,scales):
        super().__init__();self.model=make_model(config,scales,'basis_joint')
    def forward(self,b):
        return self.model(b['ground'],b['attributes'],b['structure_type'],b['floor_ids'],b['time_mask'])


class FrozenReadout(nn.Module):
    def __init__(self,config,scales,roof_only):
        super().__init__();self.head=RoofHead(config,scales) if roof_only else AllHead(config,scales)
    def forward(self,b):return self.head(b['z'],b)


@torch.no_grad()
def cache_encoder(model,b):
    s=model.structure_encoder(b['attributes'],b['structure_type']);z,pm=model.encoder(b['ground'],s,b['time_mask'])
    return dict(b,z=z.detach(),structure=s.detach(),patch_mask=pm.detach())


def decode(model,z,b):
    return model.decoder(z,b['structure'],b['patch_mask'],b['attributes'],b['floor_ids'],b['floor_mask'],b['time_mask'])


def loss_vector(pred,b,scales,roof_only):
    mask=b['time_mask'][...,None] if roof_only else b['time_mask'][...,None]&b['floor_mask'][:,None,:]
    values=[]
    for key,scale in (('displacement',scales.displacement_m),('acceleration',scales.acceleration_m_s2)):
        truth=roof(b[key],b) if roof_only else b[key]
        v=pred[key]
        den=per_building_mean(truth.square(),mask).detach().clamp_min((.01*scale)**2)
        values.append(per_building_mean((v-truth).square(),mask)/den)
    if not roof_only:
        dt=interstory_drift(b['displacement'],b['story_heights'],b['floor_mask'])
        dp=interstory_drift(pred['displacement'],b['story_heights'],b['floor_mask'])
        den=per_building_mean(dt.square(),mask).detach().clamp_min((.01*scales.drift_ratio)**2)
        values.append(per_building_mean((dp-dt).square(),mask)/den)
    return sum(values)


def fit_scales(b,roof_only):
    if not roof_only:return ResponseScales.fit_training_batches([b],fit_source='current training subset only')[0]
    mask=b['time_mask'][...,None]
    u=per_building_mean(roof(b['displacement'],b).square(),mask).mean().sqrt()
    a=per_building_mean(roof(b['acceleration'],b).square(),mask).mean().sqrt()
    return ResponseScales(max(float(u),1e-6),max(float(a),1e-4),1.)


@torch.no_grad()
def metrics(pred,b,entries,roof_only):
    rows=[] if roof_only else measure(pred,b,entries)
    if roof_only:
        for name in b['case_ids']:
            rows.append({k:entries[name][k] for k in ('id','record_id','stories','period_ratio','pga_g','DS_ref')})
    for i,row in enumerate(rows):
        t=int(b['time_mask'][i].sum());n=row['stories']
        for c,key in (('u','displacement'),('a','acceleration')):
            true=b[key][i,:t,n-1];value=pred[key][i,:t,0 if roof_only else n-1]
            row[c+'_roof_relative_l2']=float(((value-true).square().sum()/true.square().sum().clamp_min(1e-12)).sqrt())
    return rows,summarize(rows,roof_only)


def summarize(rows,roof_only):
    def one(rs):
        if not rs:return {'count':0}
        s={'count':len(rs)} if roof_only else aggregate(rs)
        for c in ('u','a'):
            v=[x[c+'_roof_relative_l2'] for x in rs]
            for q,val in (('median',np.median(v)),('p90',np.quantile(v,.9)),('worst',max(v))):s[c+'_roof_'+q]=float(val)
        if not roof_only:
            s['by_floor']={str(n):{str(f+1):{c:float(np.median([r[c+'_floor_relative_l2'][f] for r in rs if r['stories']==n])) for c in ('u','a')} for f in range(n)} for n in sorted({r['stories'] for r in rs})}
        return s
    return {'all':one(rows),'DS_gt_2':one([r for r in rows if r['DS_ref']>2]),'DS_le_2':one([r for r in rows if r['DS_ref']<=2]),
            'by_stories':{str(n):one([r for r in rows if r['stories']==n]) for n in sorted({r['stories'] for r in rows})}}


def criterion(summary,roof_only):
    ratios=[]
    for s in summary['by_stories'].values():
        for c in ('u','a'):
            ratios += [s[c+'_roof_median']/.1,s[c+'_roof_p90']/.2]
            if not roof_only:ratios += [s[c+'_median_relative_l2']/.1,s[c+'_p90_relative_l2']/.2]
    return max(ratios)


def brief(s,roof_only):
    v=s['all'];out={k:v[k] for k in ('count','u_roof_median','a_roof_median')}
    if not roof_only:out.update({k:v[k] for k in ('u_median_relative_l2','a_median_relative_l2')})
    return out


def save_predictions(run,split,pred,b):
    np.savez_compressed(run/f'{split}_predictions.npz',**{k:pred[k].detach().cpu().numpy() for k in ('displacement','acceleration')},case_ids=np.asarray(b['case_ids']))


def plot_cases(folder,split,pred,b,rows,roof_only,oracle=False,all_cases=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.2})
    path=folder/split;path.mkdir(parents=True,exist_ok=True)
    chosen=rows if all_cases else [max([x for x in rows if x['stories']==n],key=lambda x:x['u_roof_relative_l2']+x['a_roof_relative_l2']) for n in sorted({x['stories'] for x in rows})]
    for row in chosen:
        i=b['case_ids'].index(row['id']);n=row['stories'];t=int(b['time_mask'][i].sum());times=np.arange(t)*.02
        levels=[n] if roof_only else list(range(1,n+1))
        fig,axes=plt.subplots(len(levels),2,figsize=(12,1.65*len(levels)+.5),sharex=True,squeeze=False,constrained_layout=True)
        for j,f in enumerate(levels):
            for col,(key,scale,unit) in enumerate((('displacement',1000.,'mm'),('acceleration',1/9.80665,'g'))):
                ax=axes[j,col];ax.plot(times,b[key][i,:t,f-1].cpu().numpy()*scale,color='#252525',lw=.8,label='C++ reference')
                ax.plot(times,pred[key][i,:t,0 if roof_only else f-1].detach().cpu().numpy()*scale,color='#d97920' if oracle else '#2878b5',lw=.7,alpha=.85,label='ORACLE: uses response truth' if oracle else 'Prediction')
                ax.set_ylabel(f'Floor {f} ({unit})');ax.set_xlim(0,75)
        axes[0,0].set_title('Displacement');axes[0,1].set_title('Absolute acceleration');axes[0,0].legend(fontsize=6)
        for ax in axes[-1]:ax.set_xlabel('Time (s)')
        fig.suptitle(f"{folder.name.replace('diagnosis-isolation-v1-','')} | {split} | {'ORACLE RECONSTRUCTION' if oracle else 'PREDICTION'}\n{row['id']} | DS {row['DS_ref']} | roof u {row['u_roof_relative_l2']:.1%}, a {row['a_roof_relative_l2']:.1%}",fontsize=8)
        fig.savefig(path/(row['id']+'.png'),dpi=135);plt.close(fig)
    # Every held case has a separate roof panel even when all floors are predicted.
    fig,axes=plt.subplots(len(chosen),2,figsize=(12,1.65*len(chosen)+.5),squeeze=False,constrained_layout=True)
    for row,axs in zip(chosen,axes):
        i=b['case_ids'].index(row['id']);n=row['stories'];t=int(b['time_mask'][i].sum());times=np.arange(t)*.02
        for ax,key,scale,unit in zip(axs,('displacement','acceleration'),(1000.,1/9.80665),('mm','g')):
            ax.plot(times,b[key][i,:t,n-1].cpu().numpy()*scale,'k',lw=.7)
            ax.plot(times,pred[key][i,:t,0 if roof_only else n-1].detach().cpu().numpy()*scale,color='#d97920' if oracle else '#2878b5',lw=.7)
            ax.set(xlim=(0,25),ylabel=unit,title=f"{n}F r={row['period_ratio']:.1f} PGA={row['pga_g']:.3f}g DS={row['DS_ref']} | roof L2 {row['u_roof_relative_l2']:.1%}/{row['a_roof_relative_l2']:.1%}")
    for ax in axes[-1]:ax.set_xlabel('Time (s); 0-25 s detail, full histories in individual plots')
    fig.suptitle(f"{folder.name.replace('diagnosis-isolation-v1-','')} | {split} | ROOF | "+('ORACLE: uses response truth' if oracle else 'PREDICTION'))
    fig.savefig(folder/(split+'_roof.png'),dpi=130);plt.close(fig)


def train_model(name,model,train,held,entries,scales,roof_only,conf,extra):
    run=STAGE/'runs'/name;report=STAGE/'reports'/name
    run.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    if (run/'best.pt').exists():raise RuntimeError('Refusing to overwrite '+name)
    prov={'config':conf,'roof_only':roof_only,'scales':asdict(scales),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=STAGE.parent,text=True).strip(),
          'train_case_ids':train['case_ids'],'held_case_ids':held['case_ids'],'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'selection':'Training only, worst normalized per-height roof/full median and P90','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),**extra}
    json_write(run/'provenance.json',prov)
    opt=torch.optim.AdamW(model.parameters(),lr=conf['learning_rate'],weight_decay=conf['weight_decay'],fused=True)
    curve=[];best=math.inf;best_step=0;start=time.perf_counter()
    for step in range(conf['steps']+1):
        if step:
            model.train();lr=conf['minimum_learning_rate']+.5*(conf['learning_rate']-conf['minimum_learning_rate'])*(1+math.cos(math.pi*step/conf['steps']))
            if step<50:lr*=step/50
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True);pred=model(train);loss=loss_vector(pred,train,scales,roof_only).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%conf['evaluate_every']==0 or step==conf['steps']:
            model.eval()
            with torch.no_grad():pred=model(train);_,s=metrics(pred,train,entries,roof_only)
            score=criterion(s,roof_only);row={'step':step,'seconds':time.perf_counter()-start,'summary':s,'criterion':score};curve.append(row)
            if score<best:
                best=score;best_step=step;torch.save({'model':model.state_dict(),'step':step,'provenance':prov},run/'best.pt')
            json_write(run/'progress.json',row);json_write(run/'curve.json',curve)
            print(json.dumps({'run':name,'step':step,'seconds':round(row['seconds'],1),'criterion':score,**brief(s,roof_only)}),flush=True)
            if score<=1:break
    model.load_state_dict(torch.load(run/'best.pt',map_location=train['ground'].device,weights_only=False)['model']);model.eval()
    torch.backends.cuda.matmul.allow_tf32=False
    result={'provenance':prov,'best_step':best_step,'last_step':step,'seconds':time.perf_counter()-start,'splits':{},'test':'NOT_READ'}
    for label,b in (('train',train),('held',held)):
        with torch.no_grad():pred=model(b);rows,s=metrics(pred,b,entries,roof_only)
        result['splits'][label]={'summary':s,'cases':rows};save_predictions(run,label,pred,b)
        plot_cases(report,label,pred,b,rows,roof_only,all_cases=label=='held')
    result['training_gate']='PASS' if criterion(result['splits']['train']['summary'],roof_only)<=1 else 'NOT_MET'
    result['checkpoint_sha256']=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    json_write(report/'result.json',result);json_write(report/'curve.json',curve)
    print(json.dumps({'run':name,'status':'COMPLETE','training_gate':result['training_gate'],'train':brief(result['splits']['train']['summary'],roof_only),'held':brief(result['splits']['held']['summary'],roof_only)}),flush=True)
    torch.backends.cuda.matmul.allow_tf32=True
    return result
