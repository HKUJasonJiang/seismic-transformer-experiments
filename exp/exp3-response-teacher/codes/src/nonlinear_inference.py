"""Load an exported query model; inference takes only declared physical inputs."""
from torch import nn
import torch
from .model import SeismicQueryModel,ModelConfig
from .basis_decoder import SineFloorQueryDecoder
from .losses import ResponseScales


def make_model(config,scales,variant):
    model=SeismicQueryModel(config)
    model.decoder.set_response_scales(scales.displacement_m,scales.acceleration_m_s2)
    if variant=='basis_joint':model.decoder=SineFloorQueryDecoder(config,scales)
    elif variant=='wide_joint':
        d=config.d_model;w=2*d
        model.decoder.fusion=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,w),nn.GELU(),nn.Linear(w,w),nn.GELU())
        model.decoder.displacement_head=nn.Linear(w,config.patch_size)
        model.decoder.acceleration_head=nn.Linear(w,config.patch_size)
    elif variant!='joint':raise ValueError('Only ground-motion input models may be exported')
    return model


def load_query_model(checkpoint,device='cpu'):
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    p=state['provenance']
    model=make_model(ModelConfig(**p['model']),ResponseScales(**p['scales']),state['variant']).to(device)
    model.load_state_dict(state['model']);model.eval()
    return model
