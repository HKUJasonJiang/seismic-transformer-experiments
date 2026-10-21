"""Train-only A/B interventions. All scientific execution belongs on the GPU host."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
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
from peer10_common import STAGE, load_split, slice_batch, measure, grouped, json_write
from src.model import SeismicQueryModel, ModelConfig
from src.losses import ResponseScales, response_loss
from src.basis_decoder import SineFloorQueryDecoder


class FixedHead(nn.Module):
    """Fixed eight-slot control, with per-slot MLPs and no shared query."""
    def __init__(self, config, scales):
        super().__init__()
        d,p=config.d_model,config.patch_size
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*p)) for _ in range(8)])
        self.register_buffer('scales',torch.tensor([scales.displacement_m,scales.acceleration_m_s2]))
        self.p=p

    def forward(self, latent, structure, patch_mask, attributes, floor_ids, floor_mask, time_mask):
        b,k,d=latent.shape;t=time_mask.shape[1]
        values=torch.stack([head(latent).reshape(b,k,self.p,2).reshape(b,-1,2)[:,:t] for head in self.heads],dim=2)
        ids=(floor_ids-1).clamp_min(0)[:,None,:,None].expand(b,t,-1,2)
        values=values.gather(2,ids)*self.scales
        mask=time_mask[:,:,None]&floor_mask[:,None,:]
        return {name:torch.where(mask,values[...,j],0.) for j,name in enumerate(('displacement','acceleration'))}


class Intervention(nn.Module):
    def __init__(self, config, scales, mode, batch, source_checkpoint=None):
        super().__init__();self.mode=mode;self.base=SeismicQueryModel(config)
        self.base.decoder.set_response_scales(scales.displacement_m,scales.acceleration_m_s2)
        if mode=='fixed_head':self.base.decoder=FixedHead(config,scales)
        if mode in ('basis_joint','free_basis'):self.base.decoder=SineFloorQueryDecoder(config,scales)
        if mode in ('wide_joint','free_wide'):
            d=config.d_model;w=2*d
            self.base.decoder.fusion=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,w),nn.GELU(),nn.Linear(w,w),nn.GELU())
            self.base.decoder.displacement_head=nn.Linear(w,config.patch_size)
            self.base.decoder.acceleration_head=nn.Linear(w,config.patch_size)
        if mode=='frozen_A':
            state=torch.load(source_checkpoint,map_location='cpu',weights_only=False)['model']
            for prefix,mod in (('encoder.',self.base.encoder),('structure_encoder.',self.base.structure_encoder)):
                mod.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
                mod.requires_grad_(False)
        if mode in ('free_latent','free_basis','free_wide'):
            self.base.encoder.requires_grad_(False)
            k=math.ceil(batch['ground'].shape[1]/config.patch_size)
            self.latents=nn.Parameter(torch.randn(len(batch['case_ids']),k,config.d_model)*.1)
            self.lookup={name:i for i,name in enumerate(batch['case_ids'])}

    def forward(self,b):
        structure=self.base.structure_encoder(b['attributes'],b['structure_type'])
        if self.mode in ('free_latent','free_basis','free_wide'):
            ix=torch.tensor([self.lookup[x] for x in b['case_ids']],device=structure.device)
            latent=self.latents[ix]
            p=self.base.config.patch_size
            patch_mask=torch.nn.functional.pad(b['time_mask'],(0,(-b['time_mask'].shape[1])%p)).reshape(len(ix),-1,p).any(-1)
        else:latent,patch_mask=self.base.encoder(b['ground'],structure,b['time_mask'])
        out=self.base.decoder(latent,structure,patch_mask,b['attributes'],b['floor_ids'],b['floor_mask'],b['time_mask'])
        out.update(latent=latent,structure_latent=structure,patch_mask=patch_mask,time_mask=b['time_mask'],floor_mask=b['floor_mask'],stories=b['attributes'][:,0].long())
        return out


def fit_pass(s):
    return all(s['by_stories'][str(n)]['all'][f'{c}_{q}_relative_l2']<=limit
               for n in (3,5,8) for c in ('u','a') for q,limit in (('median',.1),('p90',.2)))


def selection(s):
    return max(s['by_stories'][str(n)]['all'][f'{c}_{q}_relative_l2']/limit
               for n in (3,5,8) for c in ('u','a') for q,limit in (('median',.1),('p90',.2)))


@torch.no_grad()
def evaluate(model,batch,entries):
    model.eval();pred=model(batch);rows=measure(pred,batch,entries)
    return pred,rows,grouped(rows)


@torch.no_grad()
def query_checks(model,batch):
    model.eval();full=model(batch);mx={'displacement':0.,'acceleration':0.};queries=0
    for i in range(len(batch['case_ids'])):
        n=int(batch['attributes'][i,0]);idx=slice(i,i+1)
        for order in ([j] for j in range(1,n+1)):
            ids=torch.tensor([order],device=batch['ground'].device)
            out=model.base.decoder(full['latent'][idx],full['structure_latent'][idx],full['patch_mask'][idx],batch['attributes'][idx],ids,ids>0,batch['time_mask'][idx])
            for key in mx:
                ref=full[key][idx,:,order[0]-1:order[0]]
                mx[key]=max(mx[key],float((out[key]-ref).abs().max()))
                torch.testing.assert_close(out[key],ref,atol=2e-6 if key=='displacement' else 2e-4,rtol=1e-4)
            queries+=1
        ids=torch.arange(n,0,-1,device=batch['ground'].device)[None,:]
        out=model.base.decoder(full['latent'][idx],full['structure_latent'][idx],full['patch_mask'][idx],batch['attributes'][idx],ids,ids>0,batch['time_mask'][idx])
        for key in mx:torch.testing.assert_close(out[key],full[key][idx,:,:n].flip(-1),atol=2e-6 if key=='displacement' else 2e-4,rtol=1e-4)
    return {'status':'PASS','single_floor_queries':queries,'reverse_order':True,'maximum_difference_SI':mx}


def plot_run(folder,curve,rows,batch,pred,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for ax,c,label in zip(axes,('u','a'),('Displacement','Absolute acceleration')):
        for n in (3,5,8):
            ax.semilogy([x['step'] for x in curve],[x['summary']['by_stories'][str(n)]['all'][f'{c}_median_relative_l2'] for x in curve],label=f'{n} floors')
        ax.axhline(.1,color='gray',ls=':');ax.set(xlabel='Optimizer steps',ylabel='Training median relative L2',title=label);ax.legend()
    fig.suptitle(title+' | TRAINING ONLY');fig.savefig(folder/'learning_curve.png',dpi=160);plt.close(fig)
    chosen=[]
    for n in (3,5,8):
        candidates=sorted([x for x in rows if x['stories']==n],key=lambda x:x['u_relative_l2']+x['a_relative_l2'])
        chosen.append((f'fit_{n:02d}F',candidates[len(candidates)//2]))
    chosen.append(('fit_worst',max(rows,key=lambda x:max(x['u_relative_l2'],x['a_relative_l2']))))
    for name,row in chosen:
        i=batch['case_ids'].index(row['id']);n=row['stories'];t=np.arange(batch['ground'].shape[1])*.02
        fig,axes=plt.subplots(n,2,figsize=(13,1.7*n),sharex=True,squeeze=False,constrained_layout=True)
        for f in range(n):
            for j,(key,scale,unit) in enumerate((('displacement',1000.,'mm'),('acceleration',1/9.80665,'g'))):
                axes[f,j].plot(t,batch[key][i,:,f].cpu().numpy()*scale,color='#303030',lw=.8,label='C++ reference')
                axes[f,j].plot(t,pred[key][i,:,f].cpu().numpy()*scale,color='#d6604d',lw=.7,alpha=.85,label='Training fit')
                axes[f,j].set_ylabel(f'Floor {f+1} ({unit})');axes[f,j].set_xlim(0,75)
        axes[0,0].set_title('Displacement');axes[0,1].set_title('Absolute acceleration');axes[0,0].legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel('Time (s)')
        fig.suptitle(f"{title} | TRAIN | {row['id']} | DS {row['DS_ref']} -> {row['pred_DS_ref']} | u {row['u_relative_l2']:.1%}, a {row['a_relative_l2']:.1%}")
        fig.savefig(folder/f'{name}.png',dpi=145);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/diagnosis.json');p.add_argument('--arm');p.add_argument('--steps',type=int);p.add_argument('--suffix',default='');p.add_argument('--resume');args=p.parse_args()
    conf=json.loads((STAGE/args.config).read_text());steps=args.steps or conf['steps']
    torch.set_num_threads(6);device=torch.device(conf['device'])
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(device):raise RuntimeError('RTX 5090 required')
    torch.backends.cuda.matmul.allow_tf32=True
    data,cpu,batch=load_split(STAGE/conf['dataset'],'train',device)
    selected=[i for i,x in enumerate(data.cases) if x['DS_ref']>2]
    cpu=slice_batch(cpu,torch.tensor(selected));batch=slice_batch(batch,torch.tensor(selected,device=device))
    entries={x['id']:x for x in data.cases if x['DS_ref']>2}
    scales,scale_info=ResponseScales.fit_training_batches([cpu],fit_source='65 DS>2 training cases; no validation/test loaded')
    config=ModelConfig(**json.loads((STAGE/conf['model_config']).read_text()))
    for arm in ([args.arm] if args.arm else conf['arms']):
        random.seed(conf['seed']);np.random.seed(conf['seed']);torch.manual_seed(conf['seed'])
        name=conf['run_id']+'-'+arm+args.suffix;run=STAGE/'runs'/name;report=STAGE/'reports'/name
        run.mkdir(exist_ok=True,parents=True);report.mkdir(exist_ok=True,parents=True)
        if (run/'best.pt').exists():raise RuntimeError('Refusing to overwrite existing run '+name)
        model=Intervention(config,scales,arm,batch,STAGE/conf['source_checkpoint']).to(device)
        if args.resume:model.load_state_dict(torch.load(STAGE/args.resume,map_location=device,weights_only=False)['model'])
        params=[x for x in model.parameters() if x.requires_grad]
        opt=torch.optim.AdamW(params,lr=conf['learning_rate'],weight_decay=conf['weight_decay'],fused=True)
        prov={'config':conf,'arm':arm,'max_steps':steps,'resume':args.resume,'model':config.to_dict(),'scales':asdict(scales),'scale_info':scale_info,
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=STAGE.parent,text=True).strip(),
              'train_manifest_sha256':hashlib.sha256((STAGE/conf['dataset']/'train.json').read_bytes()).hexdigest(),
              'case_ids':batch['case_ids'],'train_records':sorted({x['record_id'] for x in entries.values()}),
              'trainable_parameters':sum(x.numel() for x in params),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device),
              'selection':'training-only worst normalized per-height median/P90; no validation or test read',
              'deployable_inputs':arm not in ('free_latent','free_basis','free_wide')}
        if arm=='frozen_A':prov['source_checkpoint_sha256']=hashlib.sha256((STAGE/conf['source_checkpoint']).read_bytes()).hexdigest()
        if args.resume:prov['resume_sha256']=hashlib.sha256((STAGE/args.resume).read_bytes()).hexdigest()
        json_write(run/'provenance.json',prov);curve=[];best=math.inf;best_step=0;start=time.perf_counter()
        for step in range(steps+1):
            if step:
                model.train();lr=conf['minimum_learning_rate']+.5*(conf['learning_rate']-conf['minimum_learning_rate'])*(1+math.cos(math.pi*step/steps))
                if step<=50:lr*=step/50
                for g in opt.param_groups:g['lr']=lr
                if len(batch['case_ids'])>conf['batch_size']:
                    indices=torch.randperm(len(batch['case_ids']),device=device)[:conf['batch_size']];b=slice_batch(batch,indices)
                else:b=batch
                opt.zero_grad(set_to_none=True);out=model(b);loss=response_loss(out,b,scales,normalization='per_case')['total']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            if step%conf['evaluate_every']==0 or step==steps:
                pred,rows,s=evaluate(model,batch,entries);score=selection(s)
                row={'step':step,'seconds':time.perf_counter()-start,'summary':s,'selection_score':score}
                curve.append(row);json_write(run/'curve.json',curve);json_write(run/'progress.json',row)
                if score<best:
                    best=score;best_step=step;torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'provenance':prov},run/'best.pt')
                print(json.dumps({'arm':arm,'step':step,'seconds':round(row['seconds'],1),'u':s['all']['u_median_relative_l2'],'a':s['all']['a_median_relative_l2'],
                                  'u_p90':s['all']['u_p90_relative_l2'],'a_p90':s['all']['a_p90_relative_l2'],'criterion_ratio':score,'best_step':best_step}),flush=True)
                if fit_pass(s):break
        torch.save({'model':model.state_dict(),'step':step,'provenance':prov},run/'last.pt')
        state=torch.load(run/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
        # Final metrics and equivalence in strict FP32, including independent queries.
        torch.backends.cuda.matmul.allow_tf32=False
        pred,rows,s=evaluate(model,batch,entries)
        result={'arm':arm,'status':'FIT_PASS' if fit_pass(s) else 'FIT_NOT_MET','best_step':best_step,'last_step':step,'seconds':time.perf_counter()-start,
                'provenance':prov,'summary':s,'cases':rows,'validation':'NOT_READ','test':'NOT_READ',
                'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),'query_checks':query_checks(model,batch)}
        json_write(report/'result.json',result);json_write(report/'curve.json',curve)
        np.savez_compressed(run/'train_predictions.npz',displacement=pred['displacement'].cpu().numpy(),acceleration=pred['acceleration'].cpu().numpy(),case_ids=np.asarray(batch['case_ids']))
        plot_run(report,curve,rows,batch,pred,arm)
        print(json.dumps({'arm':arm,'status':result['status'],'summary':s['all'],'seconds':result['seconds']}),flush=True)
        torch.backends.cuda.matmul.allow_tf32=True


if __name__=='__main__':main()
