"""Frozen A/B interventions and fresh single-motion structural holdouts.

No simulations and no test-manifest access. See ISOLATION_PLAN.md for interpretation.
"""
from isolation_common import *
import argparse
from pathlib import Path


def output(name,predictions,batches,entries,prov,oracle=False,roof_only=False):
    run=STAGE/'runs'/name;report=STAGE/'reports'/name
    run.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    result={'provenance':prov,'splits':{},'test':'NOT_READ'}
    for label,b in batches.items():
        p=predictions[label];rows,s=metrics(p,b,entries,roof_only)
        result['splits'][label]={'summary':s,'cases':rows}
        save_predictions(run,label,p,b);plot_cases(report,label,p,b,rows,roof_only,oracle,all_cases=label=='held')
    json_write(report/'result.json',result);json_write(run/'provenance.json',prov)
    print(json.dumps({'run':name,'status':'COMPLETE',**{s:brief(v['summary'],roof_only) for s,v in result['splits'].items()}}),flush=True)
    return result


def frozen_a(conf,config,source,train,held,entries,prov):
    before=state_hash(source);train=cache_encoder(source,train);held=cache_encoder(source,held)
    for isroof,label in ((True,'roof'),(False,'full')):
        name=conf['prefix']+'-fixedA-'+label
        if (STAGE/'reports'/name/'result.json').exists():continue
        seed_all(conf['seed']);scales=fit_scales(train,isroof)
        model=FrozenReadout(config,scales,isroof).to(conf['device'])
        result=train_model(name,model,train,held,entries,scales,isroof,conf,
                           dict(prov,intervention='Only a new readout is trained; actual A latent is cached',frozen_source_before=before))
        assert before==state_hash(source) and all(p.grad is None for p in source.parameters())
        result['frozen_source_after']=state_hash(source);result['frozen_module_check']='PASS'
        json_write(STAGE/'reports'/name/'result.json',result)


def fixed_b(conf,source,train,held,entries,prov,constrained=False):
    name=conf['prefix']+('-fixedB-norm-oracle' if constrained else '-fixedB-oracle');run=STAGE/'runs'/name;report=STAGE/'reports'/name
    if (report/'result.json').exists():return
    run.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    before=state_hash(source);train=cache_encoder(source,train);held=cache_encoder(source,held)
    b={k:torch.cat((train[k],held[k])) if torch.is_tensor(train[k]) else train[k]+held[k] for k in train}
    nz=len(train['case_ids']);scales=fit_scales(train,False);actual=b['z'].clone()
    mean=train['z'].mean((0,1),keepdim=True);std=train['z'].std((0,1),keepdim=True).clamp_min(1e-5)
    best_loss=torch.full((len(actual),),float('inf'),device=actual.device)
    best_z=actual.clone();best_start=torch.full((len(actual),),-1,device=actual.device,dtype=torch.long)
    starts=[];t0=time.perf_counter()
    for start in range(3):
        label=('A_output','random_1','random_2')[start];seed_all(conf['seed']+start)
        initial=actual.clone() if start==0 else mean+std*torch.randn_like(actual)
        norm=source.encoder.output_norm
        if constrained:
            assert norm.weight.abs().min()>1e-6
            initial=(initial-norm.bias)/norm.weight
        z=nn.Parameter(initial)
        def latent():return norm(z) if constrained else z
        opt=torch.optim.Adam([z],lr=conf['oracle_learning_rate']);curve=[]
        local_loss=torch.full_like(best_loss,float('inf'));local_z=latent().detach().clone()
        for step in range(conf['oracle_steps']+1):
            if step:
                lr=conf['oracle_minimum_learning_rate']+.5*(conf['oracle_learning_rate']-conf['oracle_minimum_learning_rate'])*(1+math.cos(math.pi*step/conf['oracle_steps']))
                opt.param_groups[0]['lr']=lr;opt.zero_grad(set_to_none=True)
                loss=loss_vector(decode(source,latent(),b),b,scales,False).mean()
                if not torch.isfinite(loss):raise RuntimeError('Oracle loss nonfinite')
                loss.backward();opt.step()
            if step%conf['evaluate_every']==0 or step==conf['oracle_steps']:
                with torch.no_grad():
                    zz=latent();pred=decode(source,zz,b);lv=loss_vector(pred,b,scales,False)
                    mask=lv<local_loss;local_loss=torch.minimum(local_loss,lv);local_z[mask]=zz.detach()[mask]
                    _,s=metrics(pred,b,entries,False)
                row={'step':step,'summary':s,'seconds':time.perf_counter()-t0};curve.append(row)
                json_write(run/'progress.json',dict(start=label,**row))
                print(json.dumps({'run':name,'start':label,'step':step,'seconds':round(row['seconds'],1),**brief(s,False)}),flush=True)
        with torch.no_grad():
            pred=decode(source,local_z,b);rows,s=metrics(pred,b,entries,False)
            mask=local_loss<best_loss;best_loss=torch.minimum(best_loss,local_loss);best_z[mask]=local_z[mask];best_start[mask]=start
        torch.save({'z':local_z.cpu(),'case_ids':b['case_ids'],'loss':local_loss.cpu(),'initialization':label},run/(label+'.pt'))
        starts.append({'initialization':label,'seed':conf['seed']+start,'summary':s,'cases':rows,'curve':curve})
    assert state_hash(source)==before and all(p.grad is None for p in source.parameters())
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.no_grad():pred=decode(source,best_z,b)
    predictions={'train':{k:v[:nz] for k,v in pred.items()},'held':{k:v[nz:] for k,v in pred.items()}}
    detail=[]
    for i,case in enumerate(b['case_ids']):
        detail.append({'id':case,'start':int(best_start[i]),'best_loss':float(best_loss[i]),
                       'delta_z_relative_l2':float((best_z[i]-actual[i]).norm()/actual[i].norm().clamp_min(1e-8)),
                       'actual_z_rms':float(actual[i].square().mean().sqrt()),'oracle_z_rms':float(best_z[i].square().mean().sqrt())})
    oracle_prov=dict(prov,intervention='Frozen B; optimize only per-case Z using that case response truth',roof_only=False,
                     warning='ORACLE RECONSTRUCTION; NOT PREDICTION OR GENERALIZATION',frozen_before=before,frozen_after=state_hash(source),
                     frozen_module_check='PASS',selection='Per-case best u+a+drift loss across checkpoints and three initializations',
                     latent_constraint='Frozen original A output LayerNorm applied to optimized pre-normalization tokens' if constrained else 'Unconstrained Z; same dimensions but not necessarily in the encoder output range',
                     config=conf,scales=asdict(scales),seconds=time.perf_counter()-t0)
    result=output(name,predictions,{'train':train,'held':held},entries,oracle_prov,oracle=True)
    result.update(starts=starts,latent_changes=detail);json_write(report/'result.json',result)
    torch.save({'z':best_z.cpu(),'case_ids':b['case_ids'],'best_start':best_start.cpu(),'provenance':oracle_prov},run/'oracle_latents.pt')
    torch.backends.cuda.matmul.allow_tf32=True


def fresh_run(name,conf,config,train,held,entries,prov,roof_only):
    if (STAGE/'reports'/name/'result.json').exists():return
    seed_all(conf['seed']);scales=fit_scales(train,roof_only)
    model=(RoofPredictor if roof_only else FullPredictor)(config,scales).to(conf['device'])
    train_model(name,model,train,held,entries,scales,roof_only,conf,dict(prov,initialization='Random; no inherited source weights'))


def structural_holdout(conf,config,all_train,entries,prov):
    selected=[i for i,c in enumerate(all_train['case_ids']) if entries[c]['record_id']==conf['single_record'] and '_fixed_' in c
              and any(abs(entries[c]['pga_g']-p)<1e-6 for p in conf['shared_pga_g'])]
    assert len(selected)==27
    for fold in ('period','stories'):
        def isheld(i):
            c=entries[all_train['case_ids'][i]]
            return abs(c['period_ratio']-1.)<1e-6 if fold=='period' else c['stories']==5
        ti=[i for i in selected if not isheld(i)];hi=[i for i in selected if isheld(i)]
        assert len(ti)==18 and len(hi)==9
        train=subset(all_train,ti);held=subset(all_train,hi)
        def structures(b):return {(entries[c]['stories'],entries[c]['period_ratio']) for c in b['case_ids']}
        assert not structures(train)&structures(held)
        audit=dict(prov,fold=fold,record=conf['single_record'],train_structures=sorted(structures(train)),held_structures=sorted(structures(held)),
                   shared_pga_g=conf['shared_pga_g'],structural_overlap=False,held_DS_gt_2=sum(entries[c]['DS_ref']>2 for c in held['case_ids']))
        for isroof,label in ((True,'roof'),(False,'full')):
            fresh_run(conf['prefix']+'-holdout-'+fold+'-'+label,conf,config,train,held,entries,audit,isroof)
        if fold=='period':
            name=conf['prefix']+'-holdout-period-interpolation'
            if (STAGE/'reports'/name/'result.json').exists():continue
            pred={k:torch.zeros_like(held[k]) for k in ('displacement','acceleration')};weights=[]
            for i,c in enumerate(held['case_ids']):
                m=entries[c];neighbors=sorted([j for j,n in enumerate(train['case_ids']) if entries[n]['stories']==m['stories'] and abs(entries[n]['pga_g']-m['pga_g'])<1e-6],key=lambda j:float(train['attributes'][j,3]))
                assert len(neighbors)==2
                lo,hi=neighbors;w=float((held['attributes'][i,3]-train['attributes'][lo,3])/(train['attributes'][hi,3]-train['attributes'][lo,3]))
                for k in pred:pred[k][i]=(1-w)*train[k][lo]+w*train[k][hi]
                weights.append({'held':c,'training_neighbors':[train['case_ids'][lo],train['case_ids'][hi]],'weight':w})
            output(name,{'held':pred},{'held':held},entries,dict(audit,roof_only=False,intervention='Linear response interpolation using only two neighboring training structures at the same PGA',weights=weights))


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['all','fixed_a','fixed_b','fixed_b_norm','roof','holdout'],default='all');args=p.parse_args()
    conf=json.loads((STAGE/'configs/isolation.json').read_text());config=ModelConfig(**json.loads((STAGE/conf['model_config']).read_text()))
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    data=STAGE/conf['dataset'];ds,_,all_train=load_split(data,'train',conf['device'])
    entries={c['id']:c for c in ds.cases}
    prov={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=STAGE.parent,text=True).strip(),
          'train_manifest_sha256':hashlib.sha256((data/'train.json').read_bytes()).hexdigest(),
          'npz_hashes_verified':True,'model':asdict(config),'seed':conf['seed'],'test':'NOT_READ'}
    if args.group=='holdout':
        structural_holdout(conf,config,all_train,entries,prov)
        return
    dv,_,val=load_split(data,'validation',conf['device']);entries.update({c['id']:c for c in dv.cases})
    train=subset(all_train,[i for i,c in enumerate(all_train['case_ids']) if entries[c]['DS_ref']>2])
    held=subset(val,[i for i,c in enumerate(val['case_ids']) if entries[c]['DS_ref']>2])
    assert len(train['case_ids'])==65 and len(held['case_ids'])==9
    prov.update(validation_manifest_sha256=hashlib.sha256((data/'validation.json').read_bytes()).hexdigest(),
                held_role='Previously used development validation record; not fresh final test',train_case_ids=train['case_ids'],held_case_ids=held['case_ids'])
    source=load_query_model(STAGE/conf['source_checkpoint'],conf['device']);source.eval();source.requires_grad_(False)
    prov.update(source_checkpoint=conf['source_checkpoint'],source_sha256=hashlib.sha256((STAGE/conf['source_checkpoint']).read_bytes()).hexdigest())
    reference=conf['prefix']+'-reference'
    if not (STAGE/'reports'/reference/'result.json').exists():
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.no_grad():
            batches={'train':train,'held':held}
            predictions={k:source(b['ground'],b['attributes'],b['structure_type'],b['floor_ids'],b['time_mask']) for k,b in batches.items()}
        output(reference,predictions,batches,entries,dict(prov,roof_only=False,intervention='Unchanged current A+B reference'))
        torch.backends.cuda.matmul.allow_tf32=True
    if args.group in ('all','fixed_a'):frozen_a(conf,config,source,train,held,entries,prov)
    if args.group in ('all','fixed_b'):fixed_b(conf,source,train,held,entries,prov)
    if args.group in ('all','fixed_b_norm'):fixed_b(conf,source,train,held,entries,prov,constrained=True)
    if args.group in ('all','roof'):fresh_run(conf['prefix']+'-independent-roof',conf,config,train,held,entries,prov,True)
    if args.group=='all':structural_holdout(conf,config,all_train,entries,{k:v for k,v in prov.items() if k not in ('source_checkpoint','source_sha256','train_case_ids','held_case_ids','held_role')})


if __name__=='__main__':main()
