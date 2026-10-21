"""Per-building masked waveform/drift supervision, expressed in physical units."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor


def per_building_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.expand_as(value)
    count = mask.sum(dim=(1, 2))
    if (count == 0).any():
        raise ValueError("Every building must have valid time/floor observations")
    return torch.where(mask, value, 0.).sum(dim=(1, 2)) / count


def validate_full_floor_order(floor_ids: Tensor, floor_mask: Tensor, story_counts: Tensor):
    expected = torch.arange(1, floor_ids.shape[1]+1, device=floor_ids.device)[None, :]
    if not torch.equal(floor_ids > 0, floor_mask) or ((floor_ids != expected) & floor_mask).any():
        raise ValueError("Drift/loss evaluation requires all floors in ascending contiguous order")
    if (floor_mask[:, 1:] & ~floor_mask[:, :-1]).any():
        raise ValueError("Valid floors must precede padded floors")
    if not torch.equal(floor_mask.sum(-1), story_counts):
        raise ValueError("Global indicators and drift supervision require every physical floor")


def interstory_drift(displacement: Tensor, heights: Tensor, floor_mask: Tensor) -> Tensor:
    if heights.shape != floor_mask.shape or displacement.shape[::2] != heights.shape:
        raise ValueError("Expected displacement [B,T,F] and heights/mask [B,F]")
    if not torch.isfinite(heights[floor_mask]).all() or (heights[floor_mask] <= 0).any():
        raise ValueError("Valid story heights must be positive and finite")
    u = torch.where(floor_mask[:, None, :], displacement, 0.)
    previous = torch.cat((torch.zeros_like(u[..., :1]), u[..., :-1]), dim=-1)
    safe_height = torch.where(floor_mask, heights, 1.)
    return torch.where(floor_mask[:, None, :], (u-previous)/safe_height[:, None, :], 0.)


@dataclass(frozen=True)
class ResponseScales:
    displacement_m: float
    acceleration_m_s2: float
    drift_ratio: float

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Response scales must be positive finite numbers")

    @classmethod
    def fit_training_batches(cls, batches, *, fit_source: str):
        """Caller supplies ONLY training batches; fit_source identifies the split."""
        if not fit_source.strip():
            raise ValueError("Record the source of the training split")
        means = []
        for batch in batches:
            fm, tm = batch["floor_mask"], batch["time_mask"]
            mask = tm[:, :, None] & fm[:, None, :]
            drift = interstory_drift(batch["displacement"], batch["story_heights"], fm)
            values = [batch["displacement"], batch["acceleration"], drift]
            means.append(torch.stack([per_building_mean(v.square(), mask) for v in values], dim=1).detach().double().cpu())
        if not means:
            raise ValueError("Empty training split")
        rms = torch.cat(means).mean(0).sqrt()
        floors = [1e-6, 1e-4, 1e-7]
        return cls(*(max(float(v), limit) for v, limit in zip(rms, floors))), {
            "fit_source": fit_source, "method": "per-building equal-weight RMS; no mean removal",
            "buildings_seen": sum(len(v) for v in means), "minimum_scales": floors}


def response_loss(prediction, target, scales: ResponseScales,
                  weights=(1., 1., 1.)):
    """Target must contain ALL physical floors, with normalized mask weighting."""
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Need three nonnegative weights, at least one positive")
    fm, tm = prediction["floor_mask"], prediction["time_mask"]
    validate_full_floor_order(target["floor_ids"], fm, prediction["stories"])
    if target["displacement"].shape != prediction["displacement"].shape or target["acceleration"].shape != prediction["acceleration"].shape:
        raise ValueError("Targets must match output shapes")
    mask = tm[:, :, None] & fm[:, None, :]
    for key in ("displacement", "acceleration"):
        if not torch.isfinite(target[key][mask]).all():
            raise ValueError("Valid response labels must be finite")
    # Sanitize before subtraction/squaring so NaN padding cannot pollute gradients.
    def clean(x):
        return torch.where(mask, x, 0.)
    u = clean(prediction["displacement"])
    ut = clean(target["displacement"])
    a = clean(prediction["acceleration"])
    at = clean(target["acceleration"])
    drift = interstory_drift(u, target["story_heights"], fm)
    drift_target = interstory_drift(ut, target["story_heights"], fm)
    parts = {
        "displacement": per_building_mean(((u-ut)/scales.displacement_m).square(), mask).mean(),
        "acceleration": per_building_mean(((a-at)/scales.acceleration_m_s2).square(), mask).mean(),
        "drift": per_building_mean(((drift-drift_target)/scales.drift_ratio).square(), mask).mean(),
    }
    parts["total"] = sum(w*parts[k] for w, k in zip(weights, ("displacement", "acceleration", "drift")))
    return parts
