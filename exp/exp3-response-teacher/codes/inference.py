"""Deployment loader: A+B only; response labels and C are not accepted inputs."""
import torch
from src.model import ModelConfig
from src.losses import ResponseScales
from models import fresh


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['architecture'] != 'exp3-input-A-frozen-B':
        raise ValueError('Not a deployment A+B checkpoint')
    conf = dict(checkpoint['run_config'], device=device)
    model = fresh(ModelConfig(**checkpoint['model_config']), ResponseScales(**checkpoint['scales']), conf)
    model.load_state_dict(checkpoint['model'])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def predict(model, ground, attributes, structure_type, floor_ids, time_mask):
    """Ground [B,T,1], attributes [B,4], types [B], requested floor IDs [B,F]."""
    if ground.ndim != 3 or ground.shape[-1] != 1 or attributes.shape != (len(ground), 4):
        raise ValueError('Expected ground [B,T,1] and attributes [B,4] in SI units')
    if time_mask.shape != ground.shape[:2] or time_mask.dtype != torch.bool or not time_mask.any(1).all():
        raise ValueError('Each case needs a boolean time mask with valid observations')
    count = attributes[:, :1]
    if not torch.isfinite(attributes).all() or (count != count.round()).any() or (count < 1).any() or (count > model.decoder.n_basis).any():
        raise ValueError('This experiment supports integer story counts between 1 and 8')
    if floor_ids.ndim != 2 or len(floor_ids) != len(ground) or floor_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Floor queries must be an integer [B,F] tensor')
    if (floor_ids < 0).any() or (floor_ids > count).any() or not (floor_ids > 0).any(1).all():
        raise ValueError('Query a physical floor 1..N; zero is padding only')
    if structure_type.shape != (len(ground),) or not torch.isfinite(ground[..., 0][time_mask]).all():
        raise ValueError('Invalid structure types or nonfinite valid ground observations')
    b = dict(ground=ground, attributes=attributes, structure_type=structure_type,
             floor_ids=floor_ids, floor_mask=floor_ids > 0, time_mask=time_mask)
    result, _, _ = model(b)
    return result
