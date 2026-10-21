"""Post-training audit and label-free deployment export; no optimization."""
import sys
sys.dont_write_bytecode = True
from dataclasses import asdict
import subprocess
import numpy as np
import torch
from common import EXP, REPO, CHANNELS, read, write, file_hash, state_hash, load_data, metrics
from models import fresh
from run import latent_loss
from src.model import ModelConfig
from inference import load_model, predict


def main():
    conf = read(EXP / 'configs/run.json')
    config = ModelConfig(**read(EXP / 'configs/model.json'))
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = {'status': 'PASS', 'protocols': {}, 'test': 'NOT_READ',
             'audit_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}
    for protocol in conf['protocols']:
        train, held, entries, scales, _ = load_data(conf, protocol)
        methods = ['teacher', 'student', 'joint-matchedA', 'joint-matchedTotal']
        results = {m: read(EXP / 'results' / f'{protocol}-{m}.json') for m in methods}
        for m, result in results.items():
            for name, digest in result['provenance']['source_hashes'].items():
                assert file_hash(EXP / name) == digest, f'Changed original source: {name}'
            assert file_hash(EXP / 'runs' / f'{protocol}-{m}' / 'best.pt') == result['checkpoint_sha256']
            assert result['test'] == 'NOT_READ'
            for split, b in (('train', train), ('held', held)):
                with np.load(EXP / 'runs' / f'{protocol}-{m}' / f'{split}_predictions.npz') as f:
                    assert f['case_ids'].tolist() == b['case_ids']
                    pred = {k: torch.from_numpy(f[k]).to(conf['device']) for k in CHANNELS}
                recalculated = metrics(pred, b, entries)
                assert recalculated == result['splits'][split], 'Saved waveform MSE does not reproduce report'
        assert len({results[m]['provenance']['encoder_initialization_hash'] for m in methods[1:]}) == 1
        assert len({results[m]['provenance']['decoder_initialization_hash'] for m in ('teacher', 'joint-matchedA', 'joint-matchedTotal')}) == 1
        teacher = fresh(config, scales, conf, True)
        student = fresh(config, scales, conf)
        def restore(m, model):
            checkpoint = torch.load(EXP / 'runs' / f'{protocol}-{m}' / 'best.pt', map_location=conf['device'], weights_only=False)
            model.load_state_dict(checkpoint['model']); model.eval().requires_grad_(False)
            return checkpoint
        restore('teacher', teacher)
        checkpoint = restore('student', student)
        assert state_hash(teacher.decoder) == state_hash(student.decoder)
        parts = {}
        with torch.no_grad():
            for split, b in (('train', train), ('held', held)):
                zc, pm = teacher.encoder(b)
                za, _ = student.encoder(b)
                parts[split] = {'standardized_latent_mse': float(latent_loss(za, zc, pm, checkpoint['latent_std']))}
        export_path = EXP / 'runs' / f'{protocol}-student' / 'inference.pt'
        torch.save({'architecture': 'exp3-input-A-frozen-B', 'model_config': asdict(config),
                    'run_config': conf, 'scales': asdict(scales),
                    'model': {k: v.detach().cpu() for k, v in student.state_dict().items()},
                    'source_checkpoint_sha256': results['student']['checkpoint_sha256']}, export_path)
        loaded = load_model(export_path, conf['device'])
        assert state_hash(loaded) == state_hash(student)
        export_difference = {}
        with torch.no_grad():
            p = predict(loaded, held['ground'], held['attributes'], held['structure_type'], held['floor_ids'], held['time_mask'])
            original, _, _ = student(held)
            roof_ids = held['attributes'][:, :1].long()
            roof = predict(loaded, held['ground'], held['attributes'], held['structure_type'], roof_ids, held['time_mask'])
            for k in CHANNELS:
                # Identical weights; permit ordinary GPU floating-point dispatch differences.
                export_difference[k + '_mse'] = float((p[k].double()-original[k].double()).square().mean())
                assert export_difference[k + '_mse'] <= (1e-12 if k == 'displacement' else 1e-10)
                wanted = p[k].gather(2, (roof_ids-1)[:, None, :].expand(-1, p[k].shape[1], -1))
                torch.testing.assert_close(roof[k], wanted, rtol=1e-5, atol=1e-6)
            try:
                predict(loaded, held['ground'], held['attributes'], held['structure_type'], roof_ids+1, held['time_mask'])
            except ValueError:
                pass
            else:
                raise AssertionError('Out-of-building query was not rejected')
        parts.update(export_prediction_difference=export_difference, same_A_initialization='PASS', same_B_initialization='PASS', source_and_checkpoint_hashes='PASS',
                     saved_waveform_MSE_reproduction='PASS', frozen_B_equal_to_teacher_B='PASS',
                     deployment_without_labels_or_C='PASS', roof_only_query='PASS', invalid_floor_rejected='PASS', inference_sha256=file_hash(export_path))
        audit['protocols'][protocol] = parts
        del teacher, student, loaded, train, held
        torch.cuda.empty_cache()
    write(EXP / 'results/final_audit.json', audit)
    print(__import__('json').dumps(audit), flush=True)


if __name__ == '__main__': main()
