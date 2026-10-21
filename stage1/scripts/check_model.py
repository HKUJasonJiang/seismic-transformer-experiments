"""CPU contract checks; validates O2 without starting a training campaign.

Run from experiment with a Python environment containing torch and numpy.
The optional real-data check reads an explicitly prepared NPZ in stage1/data.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from src.model import ModelConfig, SeismicQueryModel
from src.losses import ResponseScales, interstory_drift, response_loss
from src.metrics import response_indicators


def inputs(length=43):
    torch.manual_seed(121)
    attributes = torch.tensor([[3., 9., 8., .23], [5., 15., 8., .29], [8., 24., 8., .46]])
    floors = torch.arange(1, 9)[None, :].repeat(3, 1)
    floors = torch.where(floors <= attributes[:, :1], floors, 0)
    mask = torch.ones((3, length), dtype=torch.bool)
    mask[0, 35:] = False
    ground = torch.randn(3, length, 1)
    return ground, attributes, torch.zeros(3, dtype=torch.long), floors, mask


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)
        self.model = SeismicQueryModel(ModelConfig(d_model=32, n_layers=2, patch_size=4)).eval()

    def test_mixed_buildings_backward(self):
        x = inputs()
        out = self.model(*x)
        self.assertEqual(out["displacement"].shape, (3, 43, 8))
        self.assertEqual(out["latent"].shape, (3, 11, 32))
        mask = out["time_mask"][:, :, None] & out["floor_mask"][:, None, :]
        for key in ("displacement", "acceleration"):
            self.assertTrue(torch.isfinite(out[key]).all())
            self.assertTrue((out[key][~mask] == 0).all())
        target = {"displacement": torch.randn(3,43,8)*.01,
                  "acceleration": torch.randn(3,43,8),
                  "story_heights": torch.where(out["floor_mask"], 3., 0.), "floor_ids": x[3]}
        loss = response_loss(out, target, ResponseScales(.01, 1., .003))
        loss["total"].backward()
        for module in (self.model.structure_encoder, self.model.encoder, self.model.decoder):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_padded_values_cannot_change_predictions(self):
        x = inputs()
        with torch.no_grad():
            expected = self.model(*x)
            altered = list(x)
            altered[0] = x[0].clone()
            altered[0][~x[4]] = float("nan")
            actual = self.model(*altered)
            for key in ("displacement", "acceleration"):
                torch.testing.assert_close(expected[key], actual[key], atol=1e-7, rtol=1e-6)
            # Additional fully padded time and floor queries must have no effect.
            extended = list(x)
            extended[0] = torch.cat((x[0], torch.full((3,13,1), float("nan"))), dim=1)
            extended[4] = torch.cat((x[4], torch.zeros(3,13,dtype=torch.bool)), dim=1)
            extended[3] = torch.cat((x[3], torch.zeros(3,2,dtype=torch.long)), dim=1)
            padded = self.model(*extended)
            for key in ("displacement", "acceleration"):
                # Different sequence dimensions may select different float32 GEMM paths.
                atol = 1e-8 if key == "displacement" else 1e-5
                torch.testing.assert_close(expected[key], padded[key][:,:43,:8], atol=atol, rtol=1e-5)

    def test_query_order_and_single_floor(self):
        x = inputs()
        with torch.no_grad():
            full = self.model(*x)
            altered = list(x)
            altered[3] = x[3][:, [2, 0]]
            subset = self.model(*altered)
            for key in ("displacement", "acceleration"):
                torch.testing.assert_close(full[key][..., [2,0]], subset[key], atol=1e-7, rtol=1e-6)
        with self.assertRaises(ValueError):
            response_indicators(subset, torch.full((3,2),3.), altered[3])

    def test_padding_excluded_from_loss_and_train_scales(self):
        x = inputs()
        out = self.model(*x)
        mask = out["time_mask"][:,:,None] & out["floor_mask"][:,None,:]
        target = {"displacement": torch.zeros(3,43,8), "acceleration": torch.ones(3,43,8),
                  "story_heights": torch.where(out["floor_mask"], 3., 0.), "floor_ids": x[3],
                  "floor_mask": out["floor_mask"], "time_mask": x[4]}
        scales, info = ResponseScales.fit_training_batches([target], fit_source="synthetic-training-check-only")
        self.assertEqual(info["buildings_seen"], 3)
        clean = response_loss(out, target, scales)
        target = {k:v.clone() for k,v in target.items()}
        for key in ("displacement", "acceleration"):
            target[key][~mask] = float("nan")
        target["story_heights"][~out["floor_mask"]] = float("nan")
        dirty = response_loss(out, target, scales)
        for key in clean:
            torch.testing.assert_close(clean[key], dirty[key])
        self.model.zero_grad()
        dirty["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_analytic_drift_grades_and_roof(self):
        # Building 1: floor drifts .5% and 1%; building 2: exactly 2%.
        u = torch.tensor([[[.015,.045]], [[.06,0.]]])
        fm = torch.tensor([[True,True],[True,False]])
        out = {"displacement":u, "acceleration":torch.tensor([[[2.,4.]], [[3.,0.]]]),
               "floor_mask":fm, "time_mask":torch.ones(2,1,dtype=torch.bool),
               "stories":torch.tensor([2,1])}
        floors = torch.tensor([[1,2],[1,0]])
        h = torch.tensor([[3.,3.],[3.,0.]])
        indicators = response_indicators(out,h,floors)
        torch.testing.assert_close(indicators["PIDR"], torch.tensor([[.005,.01],[.02,0.]]))
        self.assertEqual(indicators["DS_ref_building"].tolist(), [2,2])
        torch.testing.assert_close(indicators["roof_displacement"],torch.tensor([[.045],[.06]]))
        self.assertEqual(indicators["critical_story"].tolist(), [2,1])

    def test_reject_invalid_inputs(self):
        x = list(inputs())
        x[3] = x[3].clone()
        x[3][0,0] = 4
        with self.assertRaises(ValueError):
            self.model(*x)
        x = list(inputs())
        x[4] = x[4].clone()
        x[4][1,1] = False
        with self.assertRaises(ValueError):
            self.model(*x)


def real_shape_check():
    config = ModelConfig(**json.loads((STAGE/"configs/model.json").read_text()))
    torch.manual_seed(20261021)
    model = SeismicQueryModel(config)
    length = 3751  # 75 s on a 50 Hz grid including zero.
    ground = torch.zeros(2,length,1)
    attributes = torch.tensor([[5.,15.,8.,.2886761],[5.,15.,8.,.3464113]])
    floor_ids = torch.arange(1,6)[None,:].repeat(2,1)
    structure_type = torch.zeros(2,dtype=torch.long)
    out = model(ground, attributes, structure_type, floor_ids)
    loss = out["displacement"].square().mean() + out["acceleration"].square().mean()
    loss.backward()
    return {"input_shape":list(ground.shape), "latent_shape":list(out["latent"].shape),
            "displacement_shape":list(out["displacement"].shape),
            "acceleration_shape":list(out["acceleration"].shape),
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "device":"cpu", "torch_version":torch.__version__,
            "config":config.to_dict(), "finite_backward":all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
            "input_description":"Synthetic zero waveform; actual preview waveform check recorded separately if available",
            "training_status":"No training campaign or optimizer update; forward/backward interface check only"}


def real_data_check():
    source = STAGE/"data/model_check/input.npz"
    if not source.exists():
        return {"status":"NOT_RUN", "reason":"Run prepare_model_check_input.py first"}
    with np.load(source,allow_pickle=False) as data:
        batch = {key:torch.from_numpy(data[key].copy()) for key in data.files if key != "case_ids"}
        ids = data["case_ids"].tolist()
    scales, scale_info = ResponseScales.fit_training_batches([batch],fit_source="O2-smoke-input-only; not a frozen training split")
    config = ModelConfig(**json.loads((STAGE/"configs/model.json").read_text()))
    torch.manual_seed(20261021)
    model = SeismicQueryModel(config)
    model.decoder.set_response_scales(scales.displacement_m,scales.acceleration_m_s2)
    output = model(batch["ground"],batch["attributes"],batch["structure_type"],batch["floor_ids"],batch["time_mask"])
    losses = response_loss(output,batch,scales)
    losses["total"].backward()
    modules={}
    for name in ("structure_encoder","encoder","decoder"):
        gradients=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
        modules[name]={"finite":all(torch.isfinite(g).all().item() for g in gradients),
                       "gradient_L1":sum(float(g.abs().sum()) for g in gradients)}
        if not modules[name]["finite"] or modules[name]["gradient_L1"] <= 0:
            raise AssertionError(f"Invalid real-data backward gradient: {name}")
    indicators=response_indicators(output,batch["story_heights"],batch["floor_ids"])
    return {"status":"PASS", "case_ids":ids,"ground_shape":list(batch["ground"].shape),
            "latent_shape":list(output["latent"].shape),"response_shape":list(output["displacement"].shape),
            "untrained_loss_parts":{k:float(v.detach()) for k,v in losses.items()},
            "response_scales":{k:getattr(scales,k) for k in ("displacement_m","acceleration_m_s2","drift_ratio")},
            "scale_provenance":scale_info,"module_gradients":modules,
            "derived_indicator_shapes":{k:list(v.shape) for k,v in indicators.items()},
            "input_npz_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "performance_claim":"None; random initial weights, no optimizer step or training campaign"}


if __name__ == "__main__":
    torch.set_num_threads(4)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = {"checks_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
              "production_shape_check":real_shape_check(),"real_data_check":real_data_check()}
    patch_size = report["production_shape_check"]["config"]["patch_size"]
    destination = STAGE/"reports"/f"model_check_patch{patch_size}.json"
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
