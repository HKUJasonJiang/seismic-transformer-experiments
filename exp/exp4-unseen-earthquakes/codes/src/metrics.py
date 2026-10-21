"""Response-derived EDPs and explicit reference grades, without a classification head."""
import torch

from .losses import interstory_drift, validate_full_floor_order


@torch.no_grad()
def response_indicators(prediction, story_heights, floor_ids,
                        thresholds=(.005, .008, .02, .05)):
    fm, tm = prediction["floor_mask"], prediction["time_mask"]
    validate_full_floor_order(floor_ids, fm, prediction["stories"])
    limits = torch.as_tensor(thresholds, dtype=prediction["displacement"].dtype,
                             device=prediction["displacement"].device)
    if limits.ndim != 1 or len(limits) != 4 or not torch.isfinite(limits).all() or (limits <= 0).any() or (limits[1:] <= limits[:-1]).any():
        raise ValueError("Four strictly increasing positive thresholds required")
    mask = tm[:, :, None] & fm[:, None, :]
    u = torch.where(mask, prediction["displacement"], 0.)
    a = torch.where(mask, prediction["acceleration"], 0.)
    drift = interstory_drift(u, story_heights, fm)
    pidr = drift.abs().amax(dim=1)
    pfa = a.abs().amax(dim=1)
    grades = (pidr[..., None] > limits).sum(-1)
    roof_index = fm.sum(-1)-1
    index = roof_index[:, None, None].expand(-1, u.shape[1], 1)
    return {"drift": drift, "PIDR": pidr, "PFA": pfa,
            "maximum_PIDR": pidr.amax(dim=1), "maximum_PFA": pfa.amax(dim=1),
            "critical_story": floor_ids.gather(1, pidr.argmax(dim=1, keepdim=True)).squeeze(1),
            "DS_ref_by_floor": grades, "DS_ref_building": grades.amax(dim=1),
            "roof_displacement": u.gather(2, index).squeeze(-1),
            "roof_acceleration": a.gather(2, index).squeeze(-1)}
