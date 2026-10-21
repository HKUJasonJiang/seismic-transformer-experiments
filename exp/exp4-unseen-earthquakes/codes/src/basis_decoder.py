"""Coordinate-query decoder using a complete discrete sine basis for N<=K.

These are mathematical floor-coordinate functions, NOT identified structural modes.
The temporal coefficients depend on the input motion through the existing encoder.
"""
import math
import torch
from torch import nn


class SineFloorQueryDecoder(nn.Module):
    def __init__(self,config,scales,n_basis=8):
        super().__init__()
        self.config=config;self.n_basis=n_basis
        d,p=config.d_model,config.patch_size
        # Identical capacity to the fixed-head control, now predict basis coefficients.
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*p)) for _ in range(n_basis)])
        self.register_buffer('scales',torch.tensor([scales.displacement_m,scales.acceleration_m_s2]))
        self.register_buffer('basis_ids',torch.arange(1,n_basis+1,dtype=torch.float32))

    def forward(self,latent,structure,patch_mask,attributes,floor_ids,floor_mask,time_mask):
        b,k,d=latent.shape;p=self.config.patch_size;t=time_mask.shape[1]
        count=attributes[:,0,None,None]
        if (count>self.n_basis).any():raise ValueError('This diagnostic decoder requires stories <= n_basis')
        phi=(math.pi*floor_ids[:,:,None]*self.basis_ids[None,None,:]/(count+1)).sin()*torch.sqrt(2/(count+1))
        phi=torch.where(floor_mask[:,:,None]&(self.basis_ids[None,None,:]<=count),phi,0.)
        coefficients=torch.stack([head(latent).reshape(b,k,p,2).reshape(b,-1,2)[:,:t] for head in self.heads],dim=2)
        values=torch.einsum('btkc,bfk->btfc',coefficients,phi)*self.scales
        mask=time_mask[:,:,None]&floor_mask[:,None,:]
        return {key:torch.where(mask,values[...,j],0.) for j,key in enumerate(('displacement','acceleration'))}
