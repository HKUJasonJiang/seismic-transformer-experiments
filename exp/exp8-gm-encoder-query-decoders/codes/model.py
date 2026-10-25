"""exp8 surrogate (from exp7). A: temporal Transformer, optionally reading the ground motion only.
B: per-(floor, time patch) query decoder: MLP (exp5 style, optionally structure-aware), cross-attention, or FNO.

Fixed 50 Hz input and output, SI units. Options (all set in configs/run.json):
- decoder_type 'cross_attention': each query attends to every encoder token; queries never
  attend to each other. 'concat' reproduces the exp5 decoder (token + floor query -> MLP).
- local_snippet: the query also carries the raw ground motion of its own patch (+-context).
- ground_residual: predict relative acceleration; absolute = relative + ground_sign * ag.
  The solver stores absolute = relative - ag, hence ground_sign = -1.
- amplitude_factorization: encoder sees ag / PGA, log(PGA/g) joins the structure condition,
  and outputs are multiplied by PGA. Without it the encoder sees ag / g as in exp5.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

G = 9.80665
FIELDS = {'stories': lambda b: b['attributes'][:, 0],
          'T1_s': lambda b: b['attributes'][:, 3],
          'yield_base_shear_coefficient': lambda b: b['yield_base_shear_coefficient']}


def sinusoid(times, frequencies):
    phase = 2*math.pi*times[:, None]*frequencies[None, :]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class ConditionEncoder(nn.Module):
    """Declared structure attributes (+ log PGA) -> one condition vector. No building identity."""

    def __init__(self, conf):
        super().__init__()
        m, features = conf['model'], conf['structure_features']
        d, nd, cd = m['d_model'], m['numeric_embedding_dim'], m['category_embedding_dim']
        if any(f['transform'] not in ('linear', 'log') for f in features):
            raise ValueError('Structure feature transform must be linear or log')
        self.register_buffer('feature_scales', torch.tensor([float(f['scale']) for f in features]))
        self.register_buffer('feature_log', torch.tensor([f['transform'] == 'log' for f in features]))
        count = len(features) + int(m['amplitude_factorization'])
        self.numeric = nn.Sequential(nn.Linear(count, nd), nn.GELU(), nn.Linear(nd, nd))
        self.type_embedding = nn.Embedding(m['num_structure_types'], cd)
        self.intensity_embedding = nn.Embedding(len(conf['intensity_vocabulary']), cd)
        self.fusion = nn.Sequential(nn.Linear(nd+2*cd, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, numeric, structure_type, intensity, log_pga=None):
        # All numeric features are validated positive, so log never sees a nonpositive value.
        x = numeric/self.feature_scales
        x = torch.where(self.feature_log, torch.log(x), x)
        if log_pga is not None:
            x = torch.cat((x, log_pga[:, None]), dim=-1)
        parts = (self.numeric(x), self.type_embedding(structure_type), self.intensity_embedding(intensity))
        return self.fusion(torch.cat(parts, dim=-1))


class TemporalEncoder(nn.Module):
    """Same tokenization as exp5: 60-sample patches, input LayerNorm, FiLM before every layer."""

    def __init__(self, conf):
        super().__init__()
        m = conf['model']
        d, self.patch_size, self.sample_rate, self.n_layers = m['d_model'], m['patch_size'], m['sample_rate_hz'], m['encoder_layers']
        self.patch_embedding = nn.Linear(self.patch_size, d)
        self.patch_valid_fraction = nn.Linear(1, d, bias=False)
        self.input_norm = nn.LayerNorm(d)
        self.register_buffer('position_frequencies', torch.exp(torch.linspace(math.log(.01), math.log(5.), d//2)))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, m['heads'], dim_feedforward=m['ff_multiplier']*d, dropout=m['dropout'],
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(self.n_layers)])
        # 'none': the encoder reads only the ground motion, so its tokens can be reused for any building.
        self.film = (nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.n_layers*2*d))
                     if m['encoder_conditioning'] == 'film' else None)
        self.output_norm = nn.LayerNorm(d)

    def time_features(self, patches, like):
        p = self.patch_size
        times = (torch.arange(patches, device=like.device, dtype=like.dtype)*p + (p-1)/2)/self.sample_rate
        return sinusoid(times, self.position_frequencies)

    def forward(self, signal, time_mask, condition=None):
        # signal [B,T] is dimensionless and already zero outside time_mask.
        b, t = signal.shape
        p = self.patch_size
        pad = (-t) % p
        x = F.pad(signal, (0, pad)).reshape(b, -1, p)
        samples = F.pad(time_mask, (0, pad), value=False).reshape(b, -1, p)
        patch_mask = samples.any(dim=-1)
        positions = self.time_features(x.shape[1], signal)
        h = self.input_norm(self.patch_embedding(x)
                            + self.patch_valid_fraction(samples.to(x.dtype).mean(-1, keepdim=True))
                            + positions[None])
        affine = self.film(condition).reshape(b, self.n_layers, 2, -1) if self.film is not None else None
        for index, layer in enumerate(self.layers):
            if affine is not None:
                h = h*(1 + affine[:, index, 0].tanh()[:, None]) + affine[:, index, 1][:, None]
            h = layer(h, src_key_padding_mask=~patch_mask)
            h = torch.where(patch_mask[..., None], h, 0.)
        return self.output_norm(h), patch_mask, positions


class CrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention + feed-forward. No self-attention between queries."""

    def __init__(self, d, heads, ff_multiplier, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_multiplier*d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_multiplier*d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, ignore):
        context, _ = self.attention(self.query_norm(query), memory, memory, key_padding_mask=ignore, need_weights=False)
        query = query + self.dropout(context)
        return query + self.dropout(self.ff(self.ff_norm(query)))


class CrossAttentionDecoder(nn.Module):
    """Query = floor position + patch time + local ground snippet + structure condition."""

    def __init__(self, conf):
        super().__init__()
        m = conf['model']
        d, p = m['d_model'], m['patch_size']
        self.patch_size = p
        self.register_buffer('floor_frequencies', math.pi*2.**torch.arange(m['floor_fourier_features'], dtype=torch.float32))
        self.floor = nn.Sequential(nn.Linear(1+2*m['floor_fourier_features'], d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Linear(d, d)
        self.memory_time = nn.Linear(d, d)
        self.snippet = nn.Linear(p+2*m['snippet_context'], d) if m['local_snippet'] else None
        self.condition = nn.Linear(d, d)
        self.layers = nn.ModuleList([CrossAttentionLayer(d, m['heads'], m['ff_multiplier'], m['dropout'])
                                     for _ in range(m['decoder_layers'])])
        self.output_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2*p)

    def forward(self, memory, patch_mask, time_features, positions, snippets, condition, signal=None):
        b, patches, d = memory.shape
        phase = positions[..., None]*self.floor_frequencies
        floor = self.floor(torch.cat((positions[..., None], phase.sin(), phase.cos()), dim=-1))
        local = self.time(time_features)[None]
        if self.snippet is not None:
            local = local + self.snippet(snippets)
        query = floor[:, :, None] + local[:, None] + self.condition(condition)[:, None, None]
        floors = query.shape[1]
        query = query.reshape(b, floors*patches, d)
        keys = memory + self.memory_time(time_features)[None]
        for layer in self.layers:
            query = layer(query, keys, ~patch_mask)
        y = self.head(self.output_norm(query)).reshape(b, floors, patches, 2*self.patch_size)
        return y[..., :self.patch_size], y[..., self.patch_size:]


class ConcatDecoder(nn.Module):
    """exp5 decoder: each patch token concatenated with the floor query, shared MLP, no attention.

    decoder_structure 'film' (exp8): the structure condition is also concatenated (projected to the query size)
    and scales/shifts both hidden layers (FiLM, zero-initialized, so training starts from the plain MLP)."""

    def __init__(self, conf):
        super().__init__()
        m = conf['model']
        d, q, h, p = m['d_model'], m['concat_query_dim'], m['concat_hidden_dim'], m['patch_size']
        self.patch_size = p
        self.query = nn.Sequential(nn.Linear(1, q), nn.GELU(), nn.Linear(q, q))
        self.snippet = nn.Linear(p+2*m['snippet_context'], q) if m['local_snippet'] else None
        self.structure = nn.Linear(d, q) if m['decoder_structure'] == 'film' else None
        self.film = None
        if self.structure is not None:
            self.film = nn.Linear(d, 4*h)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.hidden1 = nn.Linear(d + q*(2 if self.structure is not None else 1), h)
        self.hidden2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 2*p)

    def forward(self, memory, patch_mask, time_features, positions, snippets, condition, signal=None):
        b, patches, d = memory.shape
        query = self.query(positions[..., None])
        floors = query.shape[1]
        query = query[:, :, None, :].expand(-1, -1, patches, -1)
        if self.snippet is not None:
            query = query + self.snippet(snippets)[:, None]
        parts = [memory[:, None].expand(-1, floors, -1, -1), query]
        if self.structure is not None:
            parts.append(self.structure(condition)[:, None, None].expand(-1, floors, patches, -1))
        x = self.hidden1(torch.cat(parts, dim=-1))
        if self.film is not None:
            g1, b1, g2, b2 = self.film(condition)[:, None, None].chunk(4, dim=-1)
            x = x*(1 + g1.tanh()) + b1
        x = self.hidden2(F.gelu(x))
        if self.film is not None:
            x = x*(1 + g2.tanh()) + b2
        y = self.out(F.gelu(x))
        return y[..., :self.patch_size], y[..., self.patch_size:]


class SpectralBlock(nn.Module):
    """FNO block on [M, C, L]: every Fourier mode up to Nyquist has its own complex gain per channel
    (depthwise, so 25 Hz content is not truncated), the lowest `mix_modes` also mix channels as in FNO,
    plus a pointwise channel map. The FFT is zero-padded to a fixed n_fft >= 2L, so the convolution is linear:
    the end of a record never wraps onto its start, and added padding cannot change the valid samples."""

    def __init__(self, width, mix_modes, n_fft):
        super().__init__()
        self.mix_modes, self.n_fft = mix_modes, n_fft
        self.depthwise = nn.Parameter(torch.randn(width, n_fft//2 + 1, 2)*.02)
        self.mix = nn.Parameter(torch.randn(width, width, mix_modes, 2)/(width*math.sqrt(2)))
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, x):
        length = x.shape[-1]
        if 2*length > self.n_fft:
            raise ValueError('Sequence is longer than the configured fno_sequence_points')
        spectrum = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        out = spectrum*torch.view_as_complex(self.depthwise.contiguous())
        k = self.mix_modes
        mixed = torch.einsum('mck,cdk->mdk', spectrum[..., :k], torch.view_as_complex(self.mix.contiguous()))
        out = torch.cat((out[..., :k] + mixed, out[..., k:]), dim=-1)
        return torch.fft.irfft(out, n=self.n_fft, dim=-1)[..., :length] + self.pointwise(x)


class FNODecoder(nn.Module):
    """Per-floor 1D FNO over the full 50 Hz time axis.

    Channels at each time step: encoder tokens (projected, linearly interpolated from patch centres to every
    sample), the ground signal, and constant channels for the floor position and the structure condition.
    FiLM from (floor, structure) scales and shifts every block (zero-initialized)."""

    def __init__(self, conf):
        super().__init__()
        m = conf['model']
        d, w, nf = m['d_model'], m['fno_width'], m['floor_fourier_features']
        self.patch_size = m['patch_size']
        self.register_buffer('floor_frequencies', math.pi*2.**torch.arange(nf, dtype=torch.float32))
        self.token = nn.Linear(d, m['fno_token_channels'])
        self.floor = nn.Sequential(nn.Linear(1+2*nf, 32), nn.GELU(), nn.Linear(32, 16))
        self.condition = nn.Linear(d, 16)
        self.lift = nn.Conv1d(m['fno_token_channels'] + 1 + 32, w, 1)
        self.blocks = nn.ModuleList([SpectralBlock(w, m['fno_mix_modes'], 2*m['fno_sequence_points'])
                                     for _ in range(m['fno_layers'])])
        self.film = nn.Linear(32, 2*w*m['fno_layers'])
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.project = nn.Sequential(nn.Conv1d(w, 2*w, 1), nn.GELU(), nn.Conv1d(2*w, 2, 1))

    def forward(self, memory, patch_mask, time_features, positions, snippets, condition, signal):
        b, patches, d = memory.shape
        length = patches*self.patch_size
        floors = positions.shape[1]
        tokens = F.interpolate(self.token(memory).transpose(1, 2), size=length, mode='linear', align_corners=False)
        ground = F.pad(signal, (0, length - signal.shape[1]))[:, None]
        phase = positions[..., None]*self.floor_frequencies
        floor = self.floor(torch.cat((positions[..., None], phase.sin(), phase.cos()), dim=-1))
        static = torch.cat((floor, self.condition(condition)[:, None].expand(-1, floors, -1)), dim=-1)
        x = torch.cat((torch.cat((tokens, ground), dim=1)[:, None].expand(-1, floors, -1, -1),
                       static[..., None].expand(-1, -1, -1, length)), dim=2)
        # Zero every channel outside the valid patches before each spectral convolution.
        valid = patch_mask.repeat_interleave(self.patch_size, dim=1)[:, None].expand(-1, floors, -1)
        valid = valid.reshape(b*floors, 1, length).to(memory.dtype)
        x = self.lift(x.reshape(b*floors, -1, length))*valid
        affine = self.film(static.reshape(b*floors, -1)).reshape(b*floors, len(self.blocks), 2, -1)
        for index, block in enumerate(self.blocks):
            x = block(x)
            x = F.gelu(x*(1 + affine[:, index, 0, :, None].tanh()) + affine[:, index, 1, :, None])*valid
        y = self.project(x).reshape(b, floors, 2, patches, self.patch_size)
        return y[:, :, 0], y[:, :, 1]


class Model(nn.Module):
    def __init__(self, conf, output_scales):
        super().__init__()
        m = conf['model']
        decoders = {'cross_attention': CrossAttentionDecoder, 'concat': ConcatDecoder, 'fno': FNODecoder}
        if m['decoder_type'] not in decoders:
            raise ValueError(f'decoder_type must be one of {sorted(decoders)}')
        if m['encoder_conditioning'] not in ('film', 'none') or m['decoder_structure'] not in ('none', 'film'):
            raise ValueError('encoder_conditioning must be film/none and decoder_structure none/film')
        if m['encoder_conditioning'] == 'none' and m['decoder_structure'] == 'none' and m['decoder_type'] == 'concat':
            raise ValueError('With a ground-motion-only encoder the decoder must receive the structure')
        if m['ground_sign'] not in (-1, 1):
            raise ValueError('ground_sign must be +1 or -1')
        self.feature_names = [f['name'] for f in conf['structure_features']]
        self.patch_size, self.context = m['patch_size'], m['snippet_context']
        self.local_snippet, self.ground_residual = m['local_snippet'], m['ground_residual']
        self.amplitude, self.ground_sign = m['amplitude_factorization'], float(m['ground_sign'])
        self.condition = ConditionEncoder(conf)
        self.temporal = TemporalEncoder(conf)
        self.decoder = decoders[m['decoder_type']](conf)
        self.register_buffer('output_scales', torch.tensor([float(v) for v in output_scales]))
        # Categories that occurred in training; others have untrained embeddings and are rejected.
        self.register_buffer('trained_types', torch.zeros(m['num_structure_types'], dtype=torch.bool))
        self.register_buffer('trained_intensities', torch.zeros(len(conf['intensity_vocabulary']), dtype=torch.bool))

    def set_trained_categories(self, structure_type, intensity):
        self.trained_types.zero_()
        self.trained_intensities.zero_()
        self.trained_types[structure_type.unique()] = True
        self.trained_intensities[intensity.unique()] = True

    def snippets(self, signal, patches):
        p, c = self.patch_size, self.context
        total = patches*p
        padded = F.pad(signal, (c, total - signal.shape[1] + c))
        return padded.unfold(1, p + 2*c, p)

    def forward(self, inputs):
        mask = inputs['time_mask']
        ground = torch.where(mask, inputs['ground'][..., 0], 0.)
        b, t = ground.shape
        if self.amplitude:
            pga = ground.abs().amax(dim=1).clamp_min(1e-6)
            signal, amplitude, log_pga = ground/pga[:, None], pga, torch.log(pga/G)
        else:
            signal, amplitude, log_pga = ground/G, torch.ones_like(ground[:, 0]), None
        condition = self.condition(inputs['numeric'], inputs['structure_type'], inputs['intensity'], log_pga)
        # A ground-motion-only encoder ignores the condition, so its tokens depend on the record alone.
        memory, patch_mask, time_features = self.temporal(signal, mask, condition)
        snippets = self.snippets(signal, memory.shape[1]) if self.local_snippet else None
        positions = torch.where(inputs['query_mask'], inputs['positions'], 0.)
        raw_u, raw_a = self.decoder(memory, patch_mask, time_features, positions, snippets, condition, signal)

        def series(raw, scale):
            return raw.reshape(b, raw.shape[1], -1)[..., :t].transpose(1, 2)*(scale*amplitude[:, None, None])
        u, a = series(raw_u, self.output_scales[0]), series(raw_a, self.output_scales[1])
        term = self.ground_sign*ground[:, :, None]
        absolute, relative = (a + term, a) if self.ground_residual else (a, a - term)
        valid = mask[:, :, None] & inputs['query_mask'][:, None, :]
        return {'displacement': torch.where(valid, u, 0.), 'acceleration': torch.where(valid, absolute, 0.),
                'relative_acceleration': torch.where(valid, relative, 0.)}

    def validate(self, x):
        n = len(x['ground'])
        if x['numeric'].shape != (n, len(self.feature_names)) or not torch.isfinite(x['numeric']).all() or (x['numeric'] <= 0).any():
            raise ValueError(f'Expected positive finite structure features {self.feature_names}')
        if 'stories' in self.feature_names:
            stories = x['numeric'][:, self.feature_names.index('stories')]
            if (stories != stories.round()).any():
                raise ValueError('Number of stories must be an integer')
        for key, trained in (('structure_type', self.trained_types), ('intensity', self.trained_intensities)):
            value = x[key]
            if value.shape != (n,) or (value < 0).any() or (value >= len(trained)).any() or not trained[value].all():
                raise ValueError(f'{key} is outside the categories seen in training')
        if x['positions'].shape != x['query_mask'].shape or x['query_mask'].dtype != torch.bool:
            raise ValueError('Query positions and boolean mask must have the same [B,Q] shape')
        valid = x['positions'][x['query_mask']]
        if not x['query_mask'].any(-1).all() or not torch.isfinite(valid).all() or (valid <= 0).any() or (valid > 1).any():
            raise ValueError('Physical floor queries require 0 < normalized position <= 1; zero is padding only')
        tm = x['time_mask']
        if tm.dtype != torch.bool or tm.shape != x['ground'].shape[:2] or not tm.any(-1).all() or (tm[:, 1:] & ~tm[:, :-1]).any():
            raise ValueError('Time mask must be a nonempty valid prefix')
        if not torch.isfinite(x['ground'][..., 0][tm]).all():
            raise ValueError('Nonfinite valid ground input')


def input_batch(batch, conf):
    """Adapter from the stored NPZ schema; no response label is passed to the model."""
    attributes = batch['attributes']
    vocabulary = attributes.new_tensor(conf['intensity_vocabulary'])
    matches = attributes[:, 2, None] == vocabulary[None, :]
    if not (matches.sum(-1) == 1).all():
        raise ValueError('Unknown fortification category; categorical IDs must be explicitly mapped')
    return {'ground': batch['ground'], 'time_mask': batch['time_mask'],
            'numeric': torch.stack([FIELDS[f['name']](batch) for f in conf['structure_features']], dim=-1),
            'structure_type': batch['structure_type'], 'intensity': matches.long().argmax(-1),
            'positions': batch['floor_ids'].to(attributes.dtype)/attributes[:, :1], 'query_mask': batch['floor_mask']}


@torch.no_grad()
def predict(model, inputs):
    model.validate(inputs)
    return model(inputs)


def load_model(path, device='cuda:0'):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = Model(ck['config'], ck['output_scales']).to(device)
    model.load_state_dict(ck['model'])
    return model.eval().requires_grad_(False), ck['config']
