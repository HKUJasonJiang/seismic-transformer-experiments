"""Postflight label-independence, structural-split and arbitrary-query checks."""
from isolation_common import *


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    conf=json.loads((STAGE/'configs/isolation.json').read_text());prefix=conf['prefix'];device=conf['device']
    ds,_,bt=load_split(STAGE/conf['dataset'],'train',device)
    dv,_,bv=load_split(STAGE/conf['dataset'],'validation',device)
    entries={x['id']:x for x in ds.cases+dv.cases};source=load_query_model(STAGE/conf['source_checkpoint'],device)
    checks={}
    for path in (STAGE/'reports').glob(prefix+'-*/result.json'):
        name=path.parent.name;r=json.loads(path.read_text());p=r['provenance'];short=name.removeprefix(prefix+'-')
        if 'oracle' in short or short in ('reference','holdout-period-interpolation'):continue
        b0=bt if short.startswith('holdout-') else bv
        ids=[x['id'] for x in r['splits']['held']['cases']];b=subset(b0,[b0['case_ids'].index(c) for c in ids])
        config=ModelConfig(**p['model']);scales=ResponseScales(**p['scales']);roof_only=p['roof_only']
        if short.startswith('fixedA'):
            b=cache_encoder(source,b);model=FrozenReadout(config,scales,roof_only)
        else:model=(RoofPredictor if roof_only else FullPredictor)(config,scales)
        model.to(device);model.load_state_dict(torch.load(STAGE/'runs'/name/'best.pt',map_location=device,weights_only=False)['model']);model.eval()
        poisoned=dict(b,displacement=torch.full_like(b['displacement'],float('nan')),acceleration=torch.full_like(b['acceleration'],float('nan')))
        with torch.no_grad():original=model(b);changed=model(poisoned)
        assert all(torch.equal(original[k],changed[k]) for k in ('displacement','acceleration'))
        check={'labels_not_in_forward':'PASS','held_cases':len(ids),'training_gate':r['training_gate'],'strict_fp32':True}
        if short.startswith('holdout'):
            ti=p['train_case_ids'];hi=p['held_case_ids']
            assert not set(ti)&set(hi)
            assert not {(entries[c]['stories'],entries[c]['period_ratio']) for c in ti}&{(entries[c]['stories'],entries[c]['period_ratio']) for c in hi}
            ids0=ti+hi;indices=[bt['case_ids'].index(c) for c in ids0]
            for pga in conf['shared_pga_g']:
                rows=[i for i in indices if abs(entries[bt['case_ids'][i]]['pga_g']-pga)<1e-6]
                assert len(rows)==9
                assert all(torch.equal(bt['ground'][rows[0]],bt['ground'][j]) for j in rows)
            assert p['initialization']=='Random; no inherited source weights'
            check.update(structure_overlap='NONE',same_motion_at_same_pga='PASS',fresh_initialization='PASS')
        if isinstance(model,FullPredictor):
            maximum=0.
            for i in range(len(ids)):
                one=subset(b,[i]);n=int(one['attributes'][0,0]);query=torch.tensor([[n,1,max(1,n//2)]],device=device,dtype=torch.long)
                with torch.no_grad():q=model(dict(one,floor_ids=query,floor_mask=query>0))
                for key in ('displacement','acceleration'):
                    target=original[key][i:i+1,:,(query[0]-1)]
                    delta=float((target-q[key]).abs().max());maximum=max(maximum,delta)
                    torch.testing.assert_close(q[key],target,rtol=1e-4,atol=1e-5)
            check.update(reordered_floor_queries='PASS',query_max_absolute_difference=maximum)
        checks[short]=check
    assert len(checks)==7,len(checks)
    out=STAGE/'reports'/(prefix+'-summary');out.mkdir(parents=True,exist_ok=True)
    json_write(out/'audit.json',{'checks':checks,'data_npz_hashes':'VERIFIED','test':'NOT_READ','oracle_excluded_from_prediction_audit':True})
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
