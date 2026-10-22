"""A: categorical/continuous concatenation + FiLM. B: shared normalized query MLP."""
import torch
from torch import nn
from src.model import ModelConfig, TemporalEncoder
from src.losses import ResponseScales


class StructuralEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        nd, cd, d = conf['numeric_embedding_dim'], conf['category_embedding_dim'], conf['model']['d_model']
        self.register_buffer('numeric_scales', torch.tensor(conf['numeric_scales']))
        self.numeric = nn.Sequential(nn.Linear(2, nd), nn.GELU(), nn.Linear(nd, nd))
        self.type_embedding = nn.Embedding(conf['model']['num_structure_types'], cd)
        self.intensity_embedding = nn.Embedding(len(conf['intensity_vocabulary']), cd)
        self.fusion = nn.Sequential(nn.Linear(nd+2*cd, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, numeric, structure_type, intensity):
        # numeric is [N, T1_seconds]. Height and ordinal intensity are NOT included.
        parts = (self.numeric(numeric/self.numeric_scales),
                 self.type_embedding(structure_type), self.intensity_embedding(intensity))
        return self.fusion(torch.cat(parts, dim=-1))


class NormalizedQueryDecoder(nn.Module):
    def __init__(self, conf, scales):
        super().__init__()
        d, q, h = conf['model']['d_model'], conf['query_embedding_dim'], conf['decoder_hidden_dim']
        self.patch_size = conf['model']['patch_size']
        self.query = nn.Sequential(nn.Linear(1,q),nn.GELU(),nn.Linear(q,q))
        self.fusion = nn.Sequential(nn.Linear(d+q,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,2*self.patch_size))
        self.register_buffer('scales', torch.tensor([scales.displacement_m, scales.acceleration_m_s2]))

    def forward(self, z, positions, query_mask, time_mask):
        # No N, floor ID embedding, structural vector, roof truth or fixed basis slots.
        clean = torch.where(query_mask, positions, 0.)
        q = self.query(clean[...,None])
        b, patches, d = z.shape
        floors = q.shape[1]
        x = torch.cat((z[:,None,:,:].expand(-1,floors,-1,-1),
                       q[:,:,None,:].expand(-1,-1,patches,-1)),dim=-1)
        y = self.fusion(x).reshape(b,floors,patches*self.patch_size,2)[:,:,:time_mask.shape[1]]
        y = y.permute(0,2,1,3)*self.scales
        mask = time_mask[:,:,None] & query_mask[:,None,:]
        return {key:torch.where(mask,y[...,j],0.) for j,key in enumerate(('displacement','acceleration'))}


class Model(nn.Module):
    def __init__(self, conf, scales):
        super().__init__()
        self.structure = StructuralEncoder(conf)
        self.temporal = TemporalEncoder(ModelConfig(**conf['model']))
        self.decoder = NormalizedQueryDecoder(conf, scales)

    def forward(self, inputs):
        s = self.structure(inputs['numeric'],inputs['structure_type'],inputs['intensity'])
        z, pm = self.temporal(inputs['ground'],s,inputs['time_mask'])
        return self.decoder(z,inputs['positions'],inputs['query_mask'],inputs['time_mask'])


def input_batch(batch, conf):
    """Adapter from immutable old NPZ schema; no label is passed to the model."""
    attributes = batch['attributes']
    vocabulary = attributes.new_tensor(conf['intensity_vocabulary'])
    matches = attributes[:,2,None] == vocabulary[None,:]
    if not (matches.sum(-1)==1).all():
        raise ValueError('Unknown fortification category; categorical IDs must be explicitly mapped')
    return {'ground':batch['ground'], 'numeric':attributes[:,[0,3]],
            'structure_type':batch['structure_type'], 'intensity':matches.long().argmax(-1),
            'positions':batch['floor_ids'].to(attributes.dtype)/attributes[:,:1],
            'query_mask':batch['floor_mask'], 'time_mask':batch['time_mask']}


def validate_inputs(inputs):
    x = inputs
    if x['numeric'].shape != (len(x['ground']),2) or not torch.isfinite(x['numeric']).all() or (x['numeric']<=0).any():
        raise ValueError('Expected positive finite N and T1')
    if (x['numeric'][:,0] != x['numeric'][:,0].round()).any():
        raise ValueError('N must be an integer')
    if x['positions'].shape != x['query_mask'].shape or x['query_mask'].dtype != torch.bool:
        raise ValueError('Query positions and boolean mask must have the same [B,Q] shape')
    valid = x['positions'][x['query_mask']]
    if not x['query_mask'].any(-1).all() or not torch.isfinite(valid).all() or (valid<=0).any() or (valid>1).any():
        raise ValueError('Physical floor queries require 0 < normalized position <= 1; zero is padding only')
    if x['time_mask'].dtype != torch.bool or x['time_mask'].shape != x['ground'].shape[:2] or not x['time_mask'].any(-1).all():
        raise ValueError('Invalid time mask')
    if not torch.isfinite(x['ground'][x['time_mask']]).all():
        raise ValueError('Nonfinite valid ground input')


@torch.no_grad()
def predict(model, inputs):
    validate_inputs(inputs)
    return model(inputs)


def load_model(path, device='cuda:0'):
    ck = torch.load(path,map_location=device,weights_only=True)
    model = Model(ck['config'],ResponseScales(**ck['scales'])).to(device)
    model.load_state_dict(ck['model'])
    return model.eval().requires_grad_(False), ck['config']
