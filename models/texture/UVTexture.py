import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import pdb



class UVTexture(nn.Module):
    def __init__(self,
                 texture_path=None,
                 tex_dim=1024):
        super(UVTexture, self).__init__()
        self.tex_dim = tex_dim
        _texture = self.init_texture(texture_path) # 1 x 3 x H x W
        self.texture = torch.nn.Parameter(_texture, requires_grad=True)
        
    def init_texture(self, texture_file):
        if not texture_file is None:
            t = cv.imread(texture_file)
            t = (t / 255.0 * 2.0 - 1.0)
            t = np.reshape(t, (1, t.shape[0], t.shape[1], 3)).astype(np.float32)
        else:
            t = np.zeros((1,self.tex_dim, self.tex_dim,3))
        texture = torch.tensor(t).permute(0, 3, 1, 2)
        return texture

    def forward(self, uv, align_corners=True):
        texture_ = torch.repeat_interleave(self.texture, repeats=uv.shape[0], dim=0)
        return F.grid_sample(texture_, uv, align_corners=align_corners) # N, C, H, W

    def save_texture(self, out_path):
        _texture = self.texture.detach().squeeze(0).permute(1,2,0)
        _texture = (torch.clamp((_texture + 1) * 0.5, min=0, max=1) * 255.0).cpu().numpy().astype(np.uint8)
        cv.imwrite(out_path, _texture)

