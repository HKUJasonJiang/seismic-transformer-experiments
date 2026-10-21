"""5090 runner: C+B reconstruction, frozen-B student, two joint controls."""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import math
import time
from dataclasses import asdict
import torch
from common import EXP, CHANNELS, Objective, load_data, read, write, state_hash, file_hash, metrics, save_prediction
from models import fresh, decode
from src.model import ModelConfig


def latent_loss(z, target, pm, std):
    point = ((z - target) / std).square().mean(-1)
    return (torch.where(pm, point, 0.).sum(1) / pm.sum(1)).mean()


def smoke(config, scales, conf, b):
    """Scientific invariants, not numerical performance thresholds."""
    b = {k: v[:3] if torch.is_tensor(v) or isinstance(v, list) else v for k, v in b.items()}
    teacher = fresh(config, scales, conf, True)
    objective = Objective(b, scales, conf)
    pred, z, pm = teacher(b)
    objective.audit(pred)
    objective(pred).backward()
    assert sum(float(p.grad.square().sum()) for p in teacher.encoder.parameters() if p.grad is not None) > 0
    assert sum(float(p.grad.square().sum()) for p in teacher.decoder.parameters() if p.grad is not None) > 0
    teacher.zero_grad(set_to_none=True)
    teacher.requires_grad_(False).eval()
    before = state_hash(teacher)
    student = fresh(config, scales, conf)
    student.decoder = copy.deepcopy(teacher.decoder)
    p, zs, ps = student(b)
    zs.retain_grad()
    objective(p).backward()
    assert zs.grad is not None and float(zs.grad.square().sum()) > 0
    assert all(p.grad is None for p in student.decoder.parameters())
    assert state_hash(teacher) == before
    student.eval()
    with torch.no_grad():
        clean, z0, _ = student(b)
        poisoned = dict(b, **{k: torch.full_like(b[k], float('nan')) for k in CHANNELS})
        p0, z1, _ = student(poisoned)
        torch.testing.assert_close(z0, z1, rtol=0, atol=0)
        for k in CHANNELS: torch.testing.assert_close(clean[k], p0[k], rtol=0, atol=0)
        # NaN padding and extra entirely masked patches/floors cannot affect C.
        pad = config.patch_size * 2
        bp = dict(b)
        for k in CHANNELS:
            bp[k] = torch.nn.functional.pad(b[k], (0, 2, 0, pad), value=float('nan'))
        bp['time_mask'] = torch.nn.functional.pad(b['time_mask'], (0, pad), value=False)
        bp['floor_mask'] = torch.nn.functional.pad(b['floor_mask'], (0, 2), value=False)
        bp['floor_ids'] = torch.nn.functional.pad(b['floor_ids'], (0, 2), value=0)
        z0, pm0 = teacher.encoder(b)
        zp, pmp = teacher.encoder(bp)
        torch.testing.assert_close(zp[:, :z0.shape[1]], z0, rtol=1e-5, atol=1e-6)
        assert (zp[~pmp] == 0).all()
        reverse = torch.arange(b['floor_ids'].shape[1]-1, -1, -1, device=conf['device'])
        br = dict(b, floor_ids=b['floor_ids'][:, reverse], floor_mask=b['floor_mask'][:, reverse])
        pr = decode(student.decoder, zs.detach(), ps, br)
        for k in CHANNELS: torch.testing.assert_close(pr[k], clean[k][:, :, reverse], rtol=1e-5, atol=1e-6)
    return {'status': 'PASS', 'loss_matches_vendored_loss': True, 'teacher_encoder_decoder_gradients': True,
            'frozen_decoder_passes_latent_gradients': True, 'response_label_poison_independence': True,
            'teacher_padding_invariance': True, 'floor_reorder_query': True}


def fit(name, model, train, held, entries, scales, conf, prov, steps, targets=None, std=None, frozen=None):
    run = EXP / 'runs' / name
    result_path = EXP / 'results' / (name + '.json')
    run.mkdir(parents=True, exist_ok=True)
    signature = dict(prov, name=name, steps=steps, initialization_hash=state_hash(model),
                     encoder_initialization_hash=state_hash(model.encoder), decoder_initialization_hash=state_hash(model.decoder),
                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     total_parameters=sum(p.numel() for p in model.parameters()),
                     checkpoint_selection='Minimum training objective, checked every evaluation interval; no held access',
                     teacher_decoder_sha256=frozen, latent_weight=conf['latent_weight'] if targets is not None else 0.)
    if result_path.exists():
        result = read(result_path)
        assert result['provenance']['source_hashes'] == signature['source_hashes'], 'Source changed; create a new experiment'
        assert result['provenance']['initialization_hash'] == signature['initialization_hash']
        assert file_hash(run / 'best.pt') == result['checkpoint_sha256']
        model.load_state_dict(torch.load(run / 'best.pt', map_location=conf['device'], weights_only=False)['model'])
        model.eval()
        print(name + ' already complete; verified and loaded', flush=True)
        return result
    if (run / 'progress.json').exists():
        raise RuntimeError(f'Incomplete run {name}; preserve it before explicitly retrying')
    write(run / 'provenance.json', signature)
    objective = Objective(train, scales, conf)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=conf['learning_rate'], weight_decay=conf['weight_decay'], fused=True)
    curve, best, best_step = [], math.inf, 0
    start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    def loss_on_train():
        pred, z, pm = model(train)
        response = objective(pred)
        ll = latent_loss(z, targets, pm, std) if targets is not None else torch.zeros((), device=z.device)
        return response + conf['latent_weight'] * ll, response, ll, pred
    for step in range(steps + 1):
        if step:
            model.train()
            if frozen: model.decoder.eval()
            lr = conf['minimum_learning_rate'] + .5 * (conf['learning_rate'] - conf['minimum_learning_rate']) * (1 + math.cos(math.pi * step / steps))
            if step < 50: lr *= step / 50
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = loss_on_train()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, conf['max_grad_norm'])
            optimizer.step()
        if step % conf['evaluate_every'] == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                total, response, ll, pred = loss_on_train()
                score = float(total)
                if not math.isfinite(score): raise RuntimeError(f'{name}: nonfinite loss at {step}')
                summary = metrics(pred, train, entries)['summary']['all']
            row = {'step': step, 'seconds': time.perf_counter()-start, 'training_objective': score,
                   'response_objective': float(response), 'latent_objective': float(ll), 'train_mse': summary}
            curve.append(row)
            if score < best:
                best, best_step = score, step
                torch.save({'model': model.state_dict(), 'step': step, 'config': asdict(model.encoder.config) if hasattr(model.encoder, 'config') else asdict(model.encoder.temporal.config),
                            'scales': asdict(scales), 'latent_std': std, 'provenance': signature}, run / 'best.pt')
            write(run / 'progress.json', row)
            write(run / 'curve.json', curve)
            print(__import__('json').dumps({'run': name, 'step': step, 'seconds': round(row['seconds'], 1), 'objective': score,
                                         'u_roof_mse': summary['u_roof_mse'], 'a_roof_mse': summary['a_roof_mse']}), flush=True)
    model.load_state_dict(torch.load(run / 'best.pt', map_location=conf['device'], weights_only=False)['model'])
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = {}
    if frozen:
        assert state_hash(model.decoder) == frozen
        assert all(p.grad is None for p in model.decoder.parameters())
        checks['frozen_decoder_hash_and_gradients'] = 'PASS'
    result = {'provenance': signature, 'best_step': best_step, 'completed_steps': steps,
              'seconds': time.perf_counter()-start, 'checkpoint_sha256': file_hash(run / 'best.pt'),
              'checks': checks, 'splits': {}, 'test': 'NOT_READ', 'oracle': name.endswith('teacher')}
    with torch.no_grad():
        for label, batch in (('train', train), ('held', held)):
            pred, z, pm = model(batch)
            result['splits'][label] = metrics(pred, batch, entries)
            save_prediction(run / (label + '_predictions.npz'), pred, batch)
        if not result['oracle']:
            original, z, pm = model(held)
            poisoned = dict(held, **{k: torch.full_like(held[k], float('nan')) for k in CHANNELS})
            check, _, _ = model(poisoned)
            order = torch.arange(held['floor_ids'].shape[1]-1, -1, -1, device=conf['device'])
            query = dict(held, floor_ids=held['floor_ids'][:, order], floor_mask=held['floor_mask'][:, order])
            reordered = decode(model.decoder, z, pm, query)
            for key in CHANNELS:
                torch.testing.assert_close(original[key], check[key], rtol=0, atol=0)
                torch.testing.assert_close(reordered[key], original[key][:, :, order], rtol=1e-5, atol=1e-6)
            checks.update(response_label_poison_independence='PASS', reordered_query='PASS')
    write(result_path, result)
    print(__import__('json').dumps({'run': name, 'status': 'COMPLETE', 'best_step': best_step, 'held_mse': result['splits']['held']['summary']['all']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['all', 'multiwave', 'period', 'stories'], default='all')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    conf = read(EXP / 'configs/run.json')
    config = ModelConfig(**read(EXP / 'configs/model.json'))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
    torch.backends.cudnn.allow_tf32 = True
    protocols = list(conf['protocols']) if args.protocol == 'all' else [args.protocol]
    for protocol in protocols:
        train, held, entries, scales, prov = load_data(conf, protocol)
        audit = smoke(config, scales, conf, train)
        write(EXP / 'results' / (protocol + '-smoke.json'), audit)
        print(protocol + ' scientific invariants PASS', flush=True)
        if args.smoke_only: continue
        steps = conf['protocols'][protocol]
        teacher = fresh(config, scales, conf, True)
        fit(protocol + '-teacher', teacher, train, held, entries, scales, conf, prov, steps['teacher_steps'])
        teacher.zero_grad(set_to_none=True)
        teacher.requires_grad_(False).eval()
        frozen_teacher = state_hash(teacher)
        with torch.no_grad():
            targets, pm = teacher.encoder(train)
            std = targets[pm].std(0, unbiased=False).clamp_min(conf['latent_std_floor'])[None, None, :]
        torch.save({'targets': targets.cpu(), 'std': std.cpu(), 'case_ids': train['case_ids'], 'teacher_hash': frozen_teacher},
                   EXP / 'runs' / (protocol + '-teacher') / 'training_latents.pt')
        student = fresh(config, scales, conf)
        student.decoder = copy.deepcopy(teacher.decoder)
        frozen_decoder = state_hash(student.decoder)
        fit(protocol + '-student', student, train, held, entries, scales, conf, prov, steps['student_steps'], targets, std, frozen_decoder)
        assert state_hash(teacher) == frozen_teacher and all(p.grad is None for p in teacher.parameters())
        write(EXP / 'results' / (protocol + '-freeze-audit.json'), {'status': 'PASS', 'teacher_hash_before_after': frozen_teacher,
              'decoder_hash_before_after': frozen_decoder, 'teacher_fit_cases': train['case_ids'], 'held_not_used_to_fit_teacher': True})
        del student, teacher, targets
        for label, budget in (('joint-matchedA', steps['student_steps']), ('joint-matchedTotal', sum(steps.values()))):
            model = fresh(config, scales, conf)
            fit(protocol + '-' + label, model, train, held, entries, scales, conf, prov, budget)
            del model
        del train, held
        torch.cuda.empty_cache()


if __name__ == '__main__': main()
