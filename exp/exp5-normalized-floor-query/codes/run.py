"""5090-only experiment: one A+B, two independently isolated generalization tests."""
import argparse
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict
import torch
from common import *
from models import Model, predict, validate_inputs, load_model


def smoke(model,b,conf,scales):
    small=take(b,torch.arange(min(3,len(b['case_ids'])),device=conf['device']))
    inputs=input_batch(small,conf); validate_inputs(inputs)
    model.train(); pred=model(inputs)
    objective=Objective(small,scales,conf); objective.audit(pred)
    objective(pred).backward()
    for part in (model.structure,model.temporal,model.decoder):
        assert sum(float(p.grad.square().sum()) for p in part.parameters() if p.grad is not None)>0
    assert model.structure.intensity_embedding.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True); model.eval()
    with torch.no_grad():
        p=model(inputs)
        poisoned=dict(small,displacement=torch.full_like(small['displacement'],float('nan')),
                      acceleration=torch.full_like(small['acceleration'],float('nan')))
        other=model(input_batch(poisoned,conf))
        # Height is intentionally absent from the encoder, although retained for drift labels.
        changed=dict(small,attributes=small['attributes'].clone()); changed['attributes'][:,1]*=17
        noheight=model(input_batch(changed,conf))
        reverse=dict(inputs,positions=inputs['positions'].flip(-1),query_mask=inputs['query_mask'].flip(-1))
        rev=predict(model,reverse)
        roof=dict(inputs,positions=torch.ones(len(inputs['ground']),1,device=conf['device']),
                  query_mask=torch.ones(len(inputs['ground']),1,dtype=torch.bool,device=conf['device']))
        rp=predict(model,roof)
        padded=dict(inputs,positions=torch.cat((inputs['positions'],torch.full_like(inputs['positions'][:,:2],float('nan'))),dim=1),
                    query_mask=torch.cat((inputs['query_mask'],torch.zeros_like(inputs['query_mask'][:,:2])),dim=1))
        pp=predict(model,padded)
        for k in CHANNELS:
            torch.testing.assert_close(p[k],other[k],rtol=0,atol=0)
            torch.testing.assert_close(p[k],noheight[k],rtol=0,atol=0)
            torch.testing.assert_close(p[k],rev[k].flip(-1),rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(p[k],pp[k][:,:,:p[k].shape[2]],rtol=1e-5,atol=1e-6)
            for i,n in enumerate(small['floor_mask'].sum(-1)):
                torch.testing.assert_close(p[k][i,:,int(n)-1],rp[k][i,:,0],rtol=1e-5,atol=1e-6)
        for value in (0.,1.2,float('nan')):
            bad=dict(roof,positions=torch.full_like(roof['positions'],value))
            try: predict(model,bad)
            except ValueError: pass
            else: raise AssertionError('Invalid query was accepted')
    return {'status':'PASS','three_part_gradients':True,'categorical_intensity_gradient':True,
            'labels_not_inputs':True,'total_height_not_input':True,'reordered_query':True,
            'roof_only_query':True,'padded_queries':True,'invalid_coordinates_rejected':True,'loss_audited':True}


def main():
    args=argparse.ArgumentParser(); args.add_argument('--resume',action='store_true'); args=args.parse_args()
    conf=read(EXP/'configs/run.json')
    torch.set_num_threads(4); torch.set_num_interop_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
    torch.manual_seed(conf['seed'])
    for d in ('runs','results','figures'): (EXP/d).mkdir(exist_ok=True)
    cases,items,meta=load_split('train',conf)
    dc,di,dm=load_split('development',conf)
    b=batch(items,conf['device']); dev=batch(di,conf['device'])
    scales,scale_fit=ResponseScales.fit_training_batches([b],fit_source='exp5 training records and period configurations only')
    entries={c['id']:c for c in cases+dc}
    model=Model(conf,scales).to(conf['device'])
    initial=state_hash(model)
    sources={p.relative_to(EXP).as_posix():file_hash(p) for p in sorted((EXP/'codes').rglob('*.py'))}
    prov={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'config':conf,'source_sha256':sources,'config_sha256':file_hash(EXP/'configs/run.json'),
          'scales':asdict(scales),'scale_fit':scale_fit,'train_case_ids':b['case_ids'],'development_case_ids':dev['case_ids'],
          'manifest_sha256':{name:file_hash(REPO/conf['dataset']/f'{name}.json') for name in ('train','validation','test')},
          'train_npz_sha256':{c['id']:c['npz_sha256'] for c in cases},'initial_state_sha256':initial,
          'parameters':sum(p.numel() for p in model.parameters()),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
          'train_counts':dict(Counter(f'{c["stories"]}F_DS{c["DS_ref"]}' for c in cases))}
    optimizer=torch.optim.AdamW(model.parameters(),lr=conf['learning_rate'],weight_decay=conf['weight_decay'],fused=True)
    last=EXP/'runs/last.pt'; best=float('inf'); best_step=0; start_step=0; elapsed=0.; curve=[]
    if args.resume:
        ck=torch.load(last,map_location=conf['device'],weights_only=True)
        old=read(EXP/'runs/provenance.json')
        assert old['source_sha256']==sources and old['config']==conf
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start_step=ck['step']; best=ck['best']; best_step=ck['best_step']; elapsed=ck['elapsed']
        torch.set_rng_state(ck['rng'].cpu()); torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        curve=read(EXP/'runs/curve.json'); prov=old
    else:
        if last.exists(): raise RuntimeError('Existing run; use --resume or preserve it under another version')
        write(EXP/'runs/provenance.json',prov)
        write(EXP/'results/smoke.json',smoke(model,b,conf,scales))
        # Smoke has no optimizer steps and leaves weights unchanged.
        assert state_hash(model)==initial
    print(f'Train cases={len(cases)}, development cases={len(dc)}, parameters={prov["parameters"]}',flush=True)
    start=time.perf_counter()
    for step in range(start_step+1,conf['steps']+1):
        model.train(); torch.backends.cuda.matmul.allow_tf32=True
        index=torch.randint(len(cases),(conf['batch_size'],),device=conf['device'])
        sub=take(b,index)
        lr=conf['minimum_learning_rate']+.5*(conf['learning_rate']-conf['minimum_learning_rate'])*(1+math.cos(math.pi*step/conf['steps']))
        lr*=min(1.,step/conf['warmup_steps'])
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        pred=model(input_batch(sub,conf)); loss=Objective(sub,scales,conf)(pred)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),conf['max_grad_norm']); optimizer.step()
        if step%conf['evaluate_every']==0 or step==conf['steps']:
            model.eval(); torch.backends.cuda.matmul.allow_tf32=False
            with torch.no_grad():
                p,score=evaluate(model,b,conf,scales); dp,dscore=evaluate(model,dev,conf,scales)
                tm=metrics(p,b,entries)['summary']['all']; vm=metrics(dp,dev,entries)['summary']['all']
            if score<best:
                best,best_step=score,step
                torch.save({'model':model.state_dict(),'config':conf,'scales':asdict(scales),'step':step,'training_objective':score},EXP/'runs/best.pt')
            seconds=elapsed+time.perf_counter()-start
            row={'step':step,'seconds':seconds,'train_objective':score,'development_objective':dscore,'train_mse':tm,'development_mse':vm}
            curve.append(row); write(EXP/'runs/curve.json',curve); write(EXP/'runs/progress.json',row)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'best':best,'best_step':best_step,
                        'elapsed':seconds,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},EXP/'runs/last.tmp')
            (EXP/'runs/last.tmp').replace(last)
            print(json.dumps({'step':step,'seconds':round(seconds,1),'train_objective':score,'development_objective':dscore,
                              'train_u_mse':tm['u_all_mse'],'train_a_mse':tm['a_all_mse']}),flush=True)
    model,loaded_conf=load_model(EXP/'runs/best.pt',conf['device'])
    torch.backends.cuda.matmul.allow_tf32=False
    frozen=state_hash(model)
    ck=torch.load(EXP/'runs/best.pt',map_location='cpu',weights_only=True)
    report={'provenance':prov,'best_step':ck['step'],'checkpoint_sha256':file_hash(EXP/'runs/best.pt'),
            'state_sha256':frozen,'splits':{}}
    audit={'training_records':conf['training_records'],'training_period_ratios':conf['training_period_ratios'],
           'trained_structure_count':len({structural_hash(c) for c in cases}),'splits':{}}
    train_struct={structural_hash(c) for c in cases}
    hashes={r['record_id']:r['sha256'] for r in meta['record_split']['records']}
    assert not {hashes[r] for r in conf['training_records']} & {hashes[r] for r in conf['earthquake_test_records']}
    for name in ('train','development','period','earthquake'):
        cs,its,_=load_split(name,conf); bs=batch(its,conf['device'])
        if name in ('period','earthquake'):
            assert not set(bs['case_ids']) & set(b['case_ids'])
            if name=='period':
                assert not {structural_hash(c) for c in cs} & train_struct
                assert set(c['record_id'] for c in cs)<=set(conf['training_records'])
            else: assert {structural_hash(c) for c in cs}<=train_struct
        validate_inputs(input_batch(bs,conf))
        pred,score=evaluate(model,bs,conf,scales)
        result=rich_metrics(pred,bs,{c['id']:c for c in cs}); result['objective']=score
        # Matched fixed-intensity subset is supplementary, never used for selection.
        fixed=[i for i,c in enumerate(cs) if '_fixed_' in c['id']]
        if fixed:
            ix=torch.tensor(fixed,device=conf['device']); fb=take(bs,ix)
            result['fixed_PGA_summary']=metrics({k:v[ix] for k,v in pred.items()},fb,{c['id']:c for c in cs})['summary']
        report['splits'][name]=result
        save_predictions(EXP/f'results/{name}_predictions.npz',pred,bs)
        audit['splits'][name]={'count':len(cs),'nonlinear_count':sum(c['DS_ref']>2 for c in cs),
            'records':sorted({c['record_id'] for c in cs}),'period_ratios':sorted({c['period_ratio'] for c in cs}),
            'case_ids':bs['case_ids'],'npz_sha256':{c['id']:c['npz_sha256'] for c in cs},
            'counts_by_stories':dict(Counter(str(c['stories']) for c in cs))}
        print(name+' '+json.dumps(result['summary']['all']),flush=True)
    assert state_hash(model)==frozen
    # Repeat interface audits with trained weights; no optimizer changes.
    model.requires_grad_(True)
    trained_checks=smoke(model,b,conf,scales)
    model.requires_grad_(False)
    assert state_hash(model)==frozen
    # Safe checkpoint reload, tested against a representative batch.
    again,_=load_model(EXP/'runs/best.pt',conf['device'])
    with torch.no_grad():
        sample=take(b,torch.arange(3,device=conf['device'])); x=input_batch(sample,conf)
        p=model(x); q=again(x)
        export_mse={k:float((p[k].double()-q[k].double()).square().mean()) for k in CHANNELS}
        assert all(v==0 for v in export_mse.values())
    report['audit']={'trained_interface_checks':trained_checks,'safe_reload_difference_mse':export_mse,
                     'weights_unchanged_in_evaluation':True,'source_record_hash_disjoint':True,
                     'known_earthquake_test_structures':True,'unseen_period_structures':True,
                     'all_npz_hashes_verified':True}
    write(EXP/'results/report.json',report); write(EXP/'results/split_audit.json',audit)
    print('TRAIN_AND_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__': main()
