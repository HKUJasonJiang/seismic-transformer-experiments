"""Self-contained experiment models; src is the vendored exp3 snapshot."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.model import StructuralEncoder, TemporalEncoder
from src.basis_decoder import SineFloorQueryDecoder


class InputEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.structure = StructuralEncoder(config)
        self.temporal = TemporalEncoder(config)

    def forward(self, b):
        s = self.structure(b['attributes'], b['structure_type'])
        return self.temporal(b['ground'], s, b['time_mask'])


class ResponseEncoder(nn.Module):
    """C reads full response truth, never used at deployment."""
    def __init__(self, config, scales, max_floors=8, hidden=(512, 256)):
        super().__init__()
        self.config, self.max_floors = config, max_floors
        self.register_buffer('scales', torch.tensor([scales.displacement_m, scales.acceleration_m_s2]))
        self.register_buffer('feature_scales', torch.tensor([20., 60., 10., 2.]))
        self.register_buffer('basis_ids', torch.arange(1, max_floors + 1, dtype=torch.float32))
        width = max_floors * config.patch_size * 2 + 4 + config.num_structure_types + max_floors
        layers = []
        for next_width in (*hidden, config.d_model):
            layers.extend([nn.Linear(width, next_width), nn.GELU()])
            width = next_width
        self.net = nn.Sequential(*layers[:-1], nn.LayerNorm(config.d_model))

    def forward(self, b):
        count = b['attributes'][:, 0, None, None]
        phi = (math.pi * b['floor_ids'][:, :, None] * self.basis_ids[None, None, :] / (count + 1)).sin() * torch.sqrt(2 / (count + 1))
        active = self.basis_ids[None, :] <= count[:, 0, :]
        phi = torch.where(b['floor_mask'][:, :, None] & active[:, None, :], phi, 0.)
        mask = b['time_mask'][:, :, None] & b['floor_mask'][:, None, :]
        values = torch.stack([torch.where(mask, b[key], 0.) for key in ('displacement', 'acceleration')], -1) / self.scales
        coefficients = torch.einsum('btfc,bfk->btkc', values, phi)
        p = self.config.patch_size
        pad = (-values.shape[1]) % p
        coefficients = F.pad(coefficients, (0, 0, 0, 0, 0, pad)).reshape(len(values), -1, p * self.max_floors * 2)
        pm = F.pad(b['time_mask'], (0, pad), value=False).reshape(len(values), -1, p).any(-1)
        condition = torch.cat((b['attributes'] / self.feature_scales,
                               F.one_hot(b['structure_type'], self.config.num_structure_types).to(values.dtype),
                               active.to(values.dtype)), -1)
        z = self.net(torch.cat((coefficients, condition[:, None, :].expand(-1, coefficients.shape[1], -1)), -1))
        return torch.where(pm[:, :, None], z, 0.), pm


class Predictor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, b):
        z, pm = self.encoder(b)
        return decode(self.decoder, z, pm, b), z, pm


def decode(decoder, z, pm, b):
    return decoder(z, None, pm, b['attributes'], b['floor_ids'], b['floor_mask'], b['time_mask'])


def fresh(config, scales, conf, teacher=False):
    # Independent seeds keep A and B initial weights identical across interventions.
    torch.manual_seed(conf['seed'] + (2 if teacher else 0))
    encoder = (ResponseEncoder(config, scales, conf['max_floors'], conf['teacher_hidden'])
               if teacher else InputEncoder(config))
    torch.manual_seed(conf['seed'] + 1)
    decoder = SineFloorQueryDecoder(config, scales, conf['max_floors'])
    return Predictor(encoder, decoder).to(conf['device'])
