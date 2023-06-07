import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-6


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DAGrid(nn.Module):
    def __init__(self, **kwargs):
        """
        """
        super(DAGrid, self).__init__()
        n_levels, base_resolution, desired_resolution = (
            kwargs['n_levels'],
            kwargs['base_resolution'], kwargs['desired_resolution']
        )
        b, n_features_per_level, bbox = (
            kwargs['b'], kwargs['n_features_per_level'], kwargs['bbox']
        )

        self.n_levels = n_levels
        if desired_resolution != -1:
            self.b = (desired_resolution / base_resolution) ** (1 / (n_levels - 1))
        else:
            self.b = b
        self.base_resolution = base_resolution # 16
        self.f = n_features_per_level # 2
        self.out_dim = self.f * self.n_levels # 32
        self.output_dim = self.out_dim + 3 # 35
        self.bounds = torch.tensor(np.array(bbox).reshape((2, 3))).float().cuda()

        self.size = (self.bounds[1] - self.bounds[0]).max().item()
        self.bounds[1] = self.bounds[1] - eps # [0, 1)
        self.offsets = [0]
        self.scales = []
        for i in range(self.n_levels):
            res = int((self.base_resolution) * (self.b**i))
            self.scales.append(res)
            n_entrys = int((res + 1) ** 3)
            self.offsets.append(self.offsets[-1] + n_entrys)

        anchors_ = self._init_anchors(freq_num=self.n_levels)
        self.data = torch.nn.Parameter(anchors_,requires_grad=True)

        self.offsets_pos = torch.tensor([[0., 0., 0.],
                                     [0., 0., 1.],
                                     [0., 1., 0.],
                                     [0., 1., 1.],
                                     [1., 0., 0.],
                                     [1., 0., 1.],
                                     [1., 1., 0.],
                                     [1., 1., 1.]]).float().cuda()  # 8 x 3

        self.scales = torch.tensor(self.scales).cuda().float()
        self.offsets = torch.tensor(np.array(self.offsets)).cuda().long()

    def _init_anchors(self, freq_num=6):
        freq_bands = 2. ** torch.linspace(0., freq_num-1, freq_num)
        self.embed_fns = []
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
        anchors = []
        for i in range(self.n_levels):
            ti = [
                torch.linspace(self.bounds[0][_i], self.bounds[1][_i], self.scales[i] + 1)
                for _i in range(3)
            ]
            ti_ = torch.meshgrid(ti)
            xyz_ = torch.stack([ti_[0].flatten(), ti_[1].flatten(), ti_[2].flatten()], dim=-1) # N x 3
            anchors.append(xyz_)
        anchors = torch.cat(anchors, dim=0) # N' x 3
        assert len(anchors) == self.offsets[-1], f'anchors dims not match offset dims, anchors: {len(anchors)}, offset[-1]: {self.offsets[-1]}.'
        return anchors

    def forward(self, xyz):
        xyz_ = torch.max(torch.min(xyz, self.bounds[1]), self.bounds[0])
        xyz_ = (xyz_ - self.bounds[None, 0]) / self.size
        xyz_ = xyz_[None].repeat(self.n_levels, 1, 1) # N x 3  -> n_level x N x 3
        float_xyz = xyz_ * self.scales[:, None, None]
        int_xyz = (float_xyz[:, :, None] + self.offsets_pos[None, None]).long()
        offset_xyz = float_xyz - int_xyz[:, :, 0]

        ind = torch.zeros_like(int_xyz[..., 0])

        sh = self.n_levels
        ind[:sh] = int_xyz[:sh, ..., 0] * ((self.scales[:sh] + 1)**2)[:, None, None] + \
                int_xyz[:sh, ..., 1] * ((self.scales[:sh] + 1))[:, None, None] + \
                int_xyz[:sh, ..., 2]
        nl = self.n_levels
        
        ind = ind.reshape(nl, -1)
        ind += self.offsets[:-1, None]
        ind = ind.reshape(-1)

        val = torch.gather(self.data, 0, ind[:, None].repeat(1, 3))
        val = val.reshape(nl, -1, 8, 3)

        weights_xyz = torch.clamp((1 - self.offsets_pos[None, None]) + (2 * self.offsets_pos[None, None] - 1.) * offset_xyz[:, :, None], min=0., max=1.)
        weights_xyz = weights_xyz[..., 0] * weights_xyz[..., 1] * weights_xyz[..., 2]

        val = torch.cat(
            [
                self.embed_fns[_i](val[_i//2,:,:])  # N x 3, (N x level x 3) x (2 * 8)
                for _i in range(len(self.embed_fns))
            ], 
            dim=-1
        ) # level x N x 8 x 6
        val = val.view(-1, 8, self.n_levels, self.f) # N x 8 x level x 6
        val = val.permute(0, 2, 1, 3) # N x level x 8 x 6
        weights_xyz = weights_xyz.permute(1, 0, 2) # level x N x 8 --> N x level x 8
        val  = (weights_xyz[..., None] * val).sum(dim=-2)
        val = val.reshape(-1, self.out_dim)
        # 
        val = torch.cat([xyz, val], dim=-1)
        return val



def get_embedder(cfg, input_dims=3):
    if cfg.type == 'frequency':
        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': cfg.freq-1,
            'num_freqs': cfg.freq,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)
        def embed(x, eo=embedder_obj): return eo.embed(x)
        return embed, embedder_obj.out_dim
    elif cfg.type == 'deformable_anchor_grid':
        embedder = DAGrid(
                n_levels=cfg.n_levels,
                base_resolution=cfg.base_resolution,
                n_features_per_level=cfg.n_features_per_level,
                desired_resolution=cfg.desired_resolution,
                b=cfg.b,
                bbox=cfg.bbox)
        return embedder, embedder.output_dim
    else:
        assert False, f'Unknown embedder type: {cfg.type}'

