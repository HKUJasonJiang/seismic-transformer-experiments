"""Joint temporal encoder and shared floor-query decoder.

Inputs and outputs use SI units. This first POC uses a fixed sample interval,
temporal patches, FiLM structural conditioning, and parallel waveform output.
It has no autoregressive teacher forcing and no structure-ID embedding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    sample_rate_hz: float = 50.0
    patch_size: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.0
    num_structure_types: int = 4

    def __post_init__(self):
        if min(self.patch_size, self.d_model, self.n_heads, self.n_layers,
               self.ff_multiplier, self.num_structure_types) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_heads")
        if self.sample_rate_hz <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid sample rate or dropout")

    def to_dict(self):
        return asdict(self)


class StructuralEncoder(nn.Module):
    """Physical attributes: stories, total height, fortification code, initial T1."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        # Fixed engineering scales, not fitted on held-out buildings.
        self.register_buffer("feature_scales", torch.tensor([20., 60., 10., 2.]))
        self.numeric = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.type_embedding = nn.Embedding(config.num_structure_types, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, attributes: Tensor, structure_type: Tensor) -> Tensor:
        return self.norm(self.numeric(attributes / self.feature_scales)
                         + self.type_embedding(structure_type))


class TemporalEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.patch_embedding = nn.Linear(config.patch_size, d)
        self.patch_valid_fraction = nn.Linear(1, d, bias=False)
        self.input_norm = nn.LayerNorm(d)
        self.register_buffer("ground_scale", torch.tensor(9.80665))
        self.register_buffer("position_frequencies", torch.exp(
            torch.linspace(math.log(.01), math.log(5.), d // 2)))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, config.n_heads,
                                       dim_feedforward=config.ff_multiplier*d,
                                       dropout=config.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(config.n_layers)
        ])
        self.film = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, config.n_layers * 2 * d))
        self.output_norm = nn.LayerNorm(d)

    def forward(self, ground: Tensor, structure: Tensor, time_mask: Tensor):
        b, t, _ = ground.shape
        p, d = self.config.patch_size, self.config.d_model
        pad = (-t) % p
        # Remove padded values before any projection, including arbitrary/NaN padding.
        signal = torch.where(time_mask, ground[..., 0], 0.) / self.ground_scale
        signal = F.pad(signal, (0, pad)).reshape(b, -1, p)
        samples = F.pad(time_mask, (0, pad), value=False).reshape(b, -1, p)
        patch_mask = samples.any(dim=-1)
        times = (torch.arange(signal.shape[1], device=ground.device,
                              dtype=ground.dtype) * p + (p-1)/2) / self.config.sample_rate_hz
        phase = 2 * math.pi * times[:, None] * self.position_frequencies[None, :]
        positions = torch.cat((phase.sin(), phase.cos()), dim=-1)
        encoded = self.input_norm(self.patch_embedding(signal)
                                  + self.patch_valid_fraction(samples.float().mean(-1, keepdim=True))
                                  + positions[None, :, :])
        affine = self.film(structure).reshape(b, self.config.n_layers, 2, d)
        for index, layer in enumerate(self.layers):
            gamma, beta = affine[:, index, 0].tanh(), affine[:, index, 1]
            encoded = encoded * (1 + gamma[:, None, :]) + beta[:, None, :]
            encoded = layer(encoded, src_key_padding_mask=~patch_mask)
            encoded = torch.where(patch_mask[..., None], encoded, 0.)
        return self.output_norm(encoded), patch_mask


class FloorQueryDecoder(nn.Module):
    """Shared query for each floor, with direct access to every temporal token."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.floor_embedding = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.attention = nn.MultiheadAttention(d, config.n_heads,
                                               dropout=config.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d)
        self.fusion = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d), nn.GELU(),
                                    nn.Linear(d, d), nn.GELU())
        self.displacement_head = nn.Linear(d, config.patch_size)
        self.acceleration_head = nn.Linear(d, config.patch_size)
        # Physical output scales are configured from training data before O3.
        self.register_buffer("displacement_scale", torch.tensor(.01))
        self.register_buffer("acceleration_scale", torch.tensor(9.80665))

    def set_response_scales(self, displacement_m: float, acceleration_m_s2: float):
        if not all(math.isfinite(x) and x > 0 for x in (displacement_m, acceleration_m_s2)):
            raise ValueError("Response scales must be positive finite training scales")
        self.displacement_scale.fill_(displacement_m)
        self.acceleration_scale.fill_(acceleration_m_s2)

    def forward(self, latent, structure, patch_mask, attributes,
                floor_ids, floor_mask, time_mask):
        b, patches, d = latent.shape
        count = attributes[:, 0, None]
        floors = floor_ids.to(latent.dtype)
        # Equal-height first POC: floor/N is also cumulative height/total height.
        query_features = torch.stack((floors/count, floors/20., (floors == count).to(latent.dtype)), dim=-1)
        query_features = torch.where(floor_mask[..., None], query_features, 0.)
        query = self.floor_embedding(query_features) + structure[:, None, :]
        context, _ = self.attention(query, latent, latent,
                                     key_padding_mask=~patch_mask, need_weights=False)
        floor_state = self.query_norm(query + context)
        # Keep the whole time sequence: floor attention does not replace latent memory.
        nt = floor_ids.shape[1]
        temporal = latent[:, None, :, :].expand(b, nt, patches, d)
        spatial = floor_state[:, :, None, :].expand(b, nt, patches, d)
        fused = self.fusion(torch.cat((temporal, spatial), dim=-1))
        t = time_mask.shape[1]
        def unpatch(head, scale):
            value = head(fused).reshape(b, nt, -1)[..., :t].transpose(1, 2) * scale
            mask = time_mask[:, :, None] & floor_mask[:, None, :]
            return torch.where(mask, value, 0.)
        return {"displacement": unpatch(self.displacement_head, self.displacement_scale),
                "acceleration": unpatch(self.acceleration_head, self.acceleration_scale)}


class SeismicQueryModel(nn.Module):
    """A and B are jointly optimized modules in one model.

    ground: [batch, time, 1], m/s²
    attributes: [batch, 4] = [stories, total_height_m, fortification_code, T1_s]
    structure_type: [batch] categorical integer, no building identity
    floor_ids: [batch, query_floor], one-based IDs; zero denotes padding
    time_mask: [batch, time], valid prefix (free-decay tail is valid)
    output: [batch, time, query_floor], displacement m and absolute acceleration m/s²
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.structure_encoder = StructuralEncoder(self.config)
        self.encoder = TemporalEncoder(self.config)
        self.decoder = FloorQueryDecoder(self.config)

    def forward(self, ground: Tensor, attributes: Tensor, structure_type: Tensor,
                floor_ids: Tensor, time_mask: Tensor | None = None):
        if ground.ndim != 3 or ground.shape[-1] != 1:
            raise ValueError("ground must have shape [B,T,1]")
        b, t, _ = ground.shape
        if t < 1 or attributes.shape != (b, 4) or structure_type.shape != (b,):
            raise ValueError("Incorrect structural input dimensions")
        if floor_ids.ndim != 2 or floor_ids.shape[0] != b:
            raise ValueError("floor_ids must have shape [B,F]")
        if structure_type.dtype != torch.long or floor_ids.dtype != torch.long:
            raise ValueError("Structure codes and floor IDs must be int64")
        if not torch.isfinite(attributes).all() or (attributes <= 0).any():
            raise ValueError("All four structural attributes must be positive and finite")
        if not torch.equal(attributes[:, 0], attributes[:, 0].round()):
            raise ValueError("Story counts must be integers")
        if ((structure_type < 0) | (structure_type >= self.config.num_structure_types)).any():
            raise ValueError("Structure code out of vocabulary")
        if ((floor_ids < 0) | (floor_ids > attributes[:, 0, None])).any():
            raise ValueError("Floor query outside the building")
        floor_mask = floor_ids > 0
        if not floor_mask.any(dim=1).all():
            raise ValueError("Each building needs at least one valid floor query")
        if time_mask is None:
            time_mask = torch.ones((b, t), dtype=torch.bool, device=ground.device)
        if time_mask.shape != (b, t) or time_mask.dtype != torch.bool:
            raise ValueError("time_mask must be boolean [B,T]")
        if not time_mask.any(dim=1).all() or (time_mask[:, 1:] & ~time_mask[:, :-1]).any():
            raise ValueError("Valid time samples must be a nonempty prefix")
        if not torch.isfinite(ground[..., 0][time_mask]).all():
            raise ValueError("Valid ground samples must be finite")
        structure = self.structure_encoder(attributes, structure_type)
        latent, patch_mask = self.encoder(ground, structure, time_mask)
        result = self.decoder(latent, structure, patch_mask, attributes,
                              floor_ids, floor_mask, time_mask)
        result.update(latent=latent, structure_latent=structure,
                      patch_mask=patch_mask, time_mask=time_mask, floor_mask=floor_mask,
                      stories=attributes[:, 0].long())
        return result
