"""5090-only exp7 run: one decoder variant and one seed per call, frozen evaluation on the exp5 splits."""
import argparse
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict
import torch
from common import (EXP, REPO, CHANNELS, Objective, metrics, file_hash, state_hash, read, write, input_batch,
                    resolve_config, load_split, batch, take, evaluate, rich_metrics, save_predictions,
                    structural_hash, fit_output_scales)
from model import Model, predict, load_model
from src.losses import ResponseScales


def smoke(model, b, conf, scales):
    device = conf['device']
    small = take(b, torch.arange(min(3, len(b['case_ids'])), device=device))
    inputs = input_batch(small, conf)
    model.validate(inputs)
    model.train()
    pred = model(inputs)
    objective = Objective(small, scales, conf)
    objective.audit(pred)
    objective(pred).backward()
    for part in (model.condition, model.temporal, model.decoder):
        assert sum(float(p.grad.square().sum()) for p in part.parameters() if p.grad is not None) > 0
    assert model.condition.intensity_embedding.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.eval()
    tol = dict(rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        p = model(inputs)
        poisoned = dict(small, displacement=torch.full_like(small['displacement'], float('nan')),
                        acceleration=torch.full_like(small['acceleration'], float('nan')))
        other = model(input_batch(poisoned, conf))
        # Height is not a model input; it is retained only for drift labels.
        changed = dict(small, attributes=small['attributes'].clone())
        changed['attributes'][:, 1] *= 17
        noheight = model(input_batch(changed, conf))
        reverse = predict(model, dict(inputs, positions=inputs['positions'].flip(-1), query_mask=inputs['query_mask'].flip(-1)))
        ones = torch.ones(len(inputs['ground']), 1, device=device)
        roof = predict(model, dict(inputs, positions=ones, query_mask=ones.bool()))
        padded = predict(model, dict(inputs,
                                     positions=torch.cat((inputs['positions'], torch.full_like(inputs['positions'][:, :2], float('nan'))), 1),
                                     query_mask=torch.cat((inputs['query_mask'], torch.zeros_like(inputs['query_mask'][:, :2])), 1)))
        extra = 2*model.patch_size
        tail = predict(model, dict(inputs,
                                   ground=torch.cat((inputs['ground'], torch.full_like(inputs['ground'][:, :extra], float('nan'))), 1),
                                   time_mask=torch.cat((inputs['time_mask'], torch.zeros_like(inputs['time_mask'][:, :extra])), 1)))
        t = inputs['ground'].shape[1]
        for k in CHANNELS:
            torch.testing.assert_close(p[k], other[k], rtol=0, atol=0)
            torch.testing.assert_close(p[k], noheight[k], rtol=0, atol=0)
            torch.testing.assert_close(p[k], reverse[k].flip(-1), **tol)
            torch.testing.assert_close(p[k], padded[k][:, :, :p[k].shape[2]], **tol)
            torch.testing.assert_close(p[k], tail[k][:, :t], **tol)
            for i, n in enumerate(small['floor_mask'].sum(-1)):
                torch.testing.assert_close(p[k][i, :, int(n)-1], roof[k][i, :, 0], **tol)
        ground = torch.where(inputs['time_mask'], inputs['ground'][..., 0], 0.)
        valid = inputs['time_mask'][:, :, None] & inputs['query_mask'][:, None, :]
        torch.testing.assert_close(p['acceleration'], torch.where(valid, p['relative_acceleration'] + model.ground_sign*ground[:, :, None], 0.), **tol)
        bad_inputs = [dict(roof, positions=torch.full_like(ones, value)) for value in (0., 1.2, float('nan'))]
        untrained = (~model.trained_intensities).nonzero().flatten()
        if len(untrained):
            bad_inputs.append(dict(inputs, intensity=torch.full_like(inputs['intensity'], int(untrained[0]))))
        for bad in bad_inputs:
            try:
                predict(model, bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid input was accepted')
    return {'status': 'PASS', 'three_part_gradients': True, 'categorical_intensity_gradient': True,
            'labels_not_inputs': True, 'total_height_not_input': True, 'reordered_query': True,
            'roof_only_query': True, 'padded_queries': True, 'padded_time_samples': True,
            'absolute_equals_relative_plus_ground_term': True, 'invalid_coordinates_rejected': True,
            'untrained_category_rejected': bool(len(untrained)), 'loss_audited': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='full')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    base = read(EXP/'configs/run.json')
    conf = resolve_config(base, args.variant)
    seed = base['seeds'][0] if args.seed is None else args.seed
    if seed not in base['seeds']:
        raise ValueError(f'Seed {seed} is not one of the predeclared seeds {base["seeds"]}')
    conf['seed'] = seed
    run_dir = EXP/'runs'/args.variant/f'seed{seed}'
    out_dir = EXP/'results'/args.variant/f'seed{seed}'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, items, meta = load_split('train', conf)
    dc, di, _ = load_split('development', conf)
    b = batch(items, conf['device'])
    dev = batch(di, conf['device'])
    scales, scale_fit = ResponseScales.fit_training_batches([b], fit_source='exp7 training records and period configurations only')
    output_scales, output_fit = fit_output_scales(b, conf)
    entries = {c['id']: c for c in cases+dc}
    model = Model(conf, output_scales).to(conf['device'])
    model.set_trained_categories(b['structure_type'], input_batch(b, conf)['intensity'])
    initial = state_hash(model)
    sources = {p.relative_to(EXP).as_posix(): file_hash(p) for p in sorted((EXP/'codes').rglob('*.py'))}
    prov = {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'variant': args.variant, 'seed': seed, 'config': conf, 'source_sha256': sources,
            'config_sha256': file_hash(EXP/'configs/run.json'),
            'loss_scales': asdict(scales), 'loss_scale_fit': scale_fit,
            'output_scales': output_scales, 'output_scale_fit': output_fit,
            'train_case_ids': b['case_ids'], 'development_case_ids': dev['case_ids'],
            'manifest_sha256': {name: file_hash(REPO/conf['dataset']/f'{name}.json') for name in ('train', 'validation', 'test')},
            'train_npz_sha256': {c['id']: c['npz_sha256'] for c in cases}, 'initial_state_sha256': initial,
            'parameters': sum(p.numel() for p in model.parameters()), 'torch': str(torch.__version__),
            'gpu': torch.cuda.get_device_name(),
            'train_counts': dict(Counter(f'{c["stories"]}F_DS{c["DS_ref"]}' for c in cases))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['learning_rate'], weight_decay=conf['weight_decay'], fused=True)
    last = run_dir/'last.pt'
    best, best_step, start_step, elapsed, curve = float('inf'), 0, 0, 0., []
    if args.resume:
        ck = torch.load(last, map_location=conf['device'], weights_only=True)
        old = read(run_dir/'provenance.json')
        assert old['source_sha256'] == sources and old['config'] == conf
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_step, best, best_step, elapsed = ck['step'], ck['best'], ck['best_step'], ck['elapsed']
        torch.set_rng_state(ck['rng'].cpu())
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        curve = read(run_dir/'curve.json')
        prov = old
    else:
        if last.exists():
            raise RuntimeError('Existing run; use --resume or preserve it under another version')
        write(run_dir/'provenance.json', prov)
        write(out_dir/'smoke.json', smoke(model, b, conf, scales))
        # Smoke has no optimizer steps and leaves weights unchanged.
        assert state_hash(model) == initial
    print(f'Variant={args.variant}, seed={seed}, train cases={len(cases)}, development cases={len(dc)}, '
          f'parameters={prov["parameters"]}', flush=True)
    start = time.perf_counter()
    for step in range(start_step+1, conf['steps']+1):
        model.train()
        torch.backends.cuda.matmul.allow_tf32 = True
        index = torch.randint(len(cases), (conf['batch_size'],), device=conf['device'])
        sub = take(b, index)
        lr = conf['minimum_learning_rate'] + .5*(conf['learning_rate']-conf['minimum_learning_rate'])*(1+math.cos(math.pi*step/conf['steps']))
        lr *= min(1., step/conf['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = Objective(sub, scales, conf)(model(input_batch(sub, conf)))
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), conf['max_grad_norm'])
        optimizer.step()
        if step % conf['evaluate_every'] == 0 or step == conf['steps']:
            model.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                p, score = evaluate(model, b, conf, scales)
                dp, dscore = evaluate(model, dev, conf, scales)
                tm = metrics(p, b, entries)['summary']['all']
                vm = metrics(dp, dev, entries)['summary']['all']
            chosen = score if conf['checkpoint_selection'] == 'train_objective' else dscore
            if chosen < best:
                best, best_step = chosen, step
                torch.save({'model': model.state_dict(), 'config': conf, 'output_scales': output_scales,
                            'loss_scales': asdict(scales), 'step': step, 'selection_score': chosen}, run_dir/'best.pt')
            seconds = elapsed + time.perf_counter() - start
            row = {'step': step, 'seconds': seconds, 'train_objective': score, 'development_objective': dscore,
                   'train_mse': tm, 'development_mse': vm}
            curve.append(row)
            write(run_dir/'curve.json', curve)
            write(run_dir/'progress.json', row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'best': best,
                        'best_step': best_step, 'elapsed': seconds, 'rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state()}, run_dir/'last.tmp')
            (run_dir/'last.tmp').replace(last)
            print(json.dumps({'step': step, 'seconds': round(seconds, 1), 'train_objective': score,
                              'development_objective': dscore, 'train_u_mse': tm['u_all_mse'],
                              'train_a_mse': tm['a_all_mse']}), flush=True)
    model, _ = load_model(run_dir/'best.pt', conf['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = state_hash(model)
    ck = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=True)
    report = {'provenance': prov, 'best_step': ck['step'], 'checkpoint_selection': conf['checkpoint_selection'],
              'checkpoint_sha256': file_hash(run_dir/'best.pt'), 'state_sha256': frozen, 'splits': {}}
    train_struct = {structural_hash(c) for c in cases}
    audit = {'training_records': conf['training_records'], 'training_period_ratios': conf['training_period_ratios'],
             'trained_structure_count': len(train_struct), 'splits': {}}
    hashes = {r['record_id']: r['sha256'] for r in meta['record_split']['records']}
    assert not {hashes[r] for r in conf['training_records']} & {hashes[r] for r in conf['earthquake_test_records']}
    for name in ('train', 'development', 'period', 'earthquake'):
        cs, its, _ = load_split(name, conf)
        bs = batch(its, conf['device'])
        if name in ('period', 'earthquake'):
            assert not set(bs['case_ids']) & set(b['case_ids'])
            if name == 'period':
                assert not {structural_hash(c) for c in cs} & train_struct
                assert set(c['record_id'] for c in cs) <= set(conf['training_records'])
            else:
                assert {structural_hash(c) for c in cs} <= train_struct
        model.validate(input_batch(bs, conf))
        pred, score = evaluate(model, bs, conf, scales)
        result = rich_metrics(pred, bs, {c['id']: c for c in cs})
        result['objective'] = score
        # Matched fixed-intensity subset is supplementary, never used for selection.
        fixed = [i for i, c in enumerate(cs) if '_fixed_' in c['id']]
        if fixed:
            ix = torch.tensor(fixed, device=conf['device'])
            result['fixed_PGA_summary'] = metrics({k: v[ix] for k, v in pred.items()}, take(bs, ix), {c['id']: c for c in cs})['summary']
        report['splits'][name] = result
        save_predictions(run_dir/f'{name}_predictions.npz', pred, bs)
        audit['splits'][name] = {'count': len(cs), 'nonlinear_count': sum(c['DS_ref'] > 2 for c in cs),
                                 'records': sorted({c['record_id'] for c in cs}),
                                 'period_ratios': sorted({c['period_ratio'] for c in cs}),
                                 'case_ids': bs['case_ids'], 'npz_sha256': {c['id']: c['npz_sha256'] for c in cs},
                                 'counts_by_stories': dict(Counter(str(c['stories']) for c in cs))}
        print(name+' '+json.dumps(result['summary']['all']), flush=True)
    assert state_hash(model) == frozen
    # Repeat interface audits with trained weights; no optimizer changes.
    model.requires_grad_(True)
    trained_checks = smoke(model, b, conf, scales)
    model.requires_grad_(False)
    assert state_hash(model) == frozen
    again, _ = load_model(run_dir/'best.pt', conf['device'])
    with torch.no_grad():
        x = input_batch(take(b, torch.arange(3, device=conf['device'])), conf)
        p, q = model(x), again(x)
        export_mse = {k: float((p[k].double()-q[k].double()).square().mean()) for k in CHANNELS}
        assert all(v == 0 for v in export_mse.values())
    report['audit'] = {'trained_interface_checks': trained_checks, 'safe_reload_difference_mse': export_mse,
                       'weights_unchanged_in_evaluation': True, 'source_record_hash_disjoint': True,
                       'known_earthquake_test_structures': True, 'unseen_period_structures': True,
                       'all_npz_hashes_verified': True}
    write(out_dir/'report.json', report)
    write(out_dir/'split_audit.json', audit)
    print('TRAIN_AND_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
