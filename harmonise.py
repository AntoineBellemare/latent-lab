"""The material's colour and tone settled into the picture: learned image harmonisation.

A material laid into a mask keeps its own white balance and contrast, and reads as pasted on.
PCT-Net (Guerreiro et al., CVPR 2023, "PCT-Net: Full Resolution Image Harmonization Using
Pixel-Wise Color Transformations"; Rakuten, MPL-2.0) looks at the composite and its mask at 256 px
with a small vision transformer and predicts, for every pixel, an affine colour transform (a 3×3
matrix and an offset) that would make the masked part belong to the rest. The transforms are
smooth, so they are enlarged to the full frame and applied there: the material's own grain is
untouched, only its colour and tone move.

The network is written out here from the published code (its ViT_pct configuration) and loads
the published weights, tensors only, from <models>/pctnet/PCTNet_ViT.pth.
"""

import math
import os
import pathlib
import threading

import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

WEIGHTS = pathlib.Path(os.environ.get("LATENT_LAB_MODELS", pathlib.Path.home() / "latent-lab-models")) / "pctnet" / "PCTNet_ViT.pth"
SIDE = 256
_lock = threading.Lock()
_net = {}


class ViTHarmonizer(nn.Module):
    """PCT-Net's ViT backbone: 4×4 patches of image and mask, nine transformer layers, and a
    transposed convolution back to a parameter map."""

    def __init__(self, output_nc, ksize=4, heads=2, layers=9, input_nc=3, forward_mult=2):
        super().__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ksize, p2=ksize),
            nn.Linear(ksize * ksize * (input_nc + 1), dim))
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=heads, dim_feedforward=dim * forward_mult, activation="gelu"),
            num_layers=layers, enable_nested_tensor=False)
        self.dec = nn.Sequential(nn.ConvTranspose2d(dim, output_nc, kernel_size=ksize, stride=ksize), nn.Tanh())

    def forward(self, x):
        e = self.patch_to_embedding(x)
        h = self.transformer_enc(e.permute(1, 0, 2))
        b, n, c = e.shape
        return self.dec(h.permute(1, 2, 0).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n))))


class PCTNet(nn.Module):
    """The ViT_pct configuration: a linear colour transform with offset (12 numbers a pixel)."""

    def __init__(self):
        super().__init__()
        self.encoder = ViTHarmonizer(output_nc=12)
        self.get_params = nn.Conv2d(12, 12, kernel_size=1)

    def params(self, image, mask):
        return self.get_params(self.encoder(torch.cat([image, mask], 1)))


def transform(image, params):
    """Each pixel's colour through its own 3×3 matrix and offset, clamped to [0, 1]."""
    n, _, h, w = image.shape
    m = params[:, :9].permute(0, 2, 3, 1).reshape(n, h, w, 3, 3)
    b = params[:, 9:12].permute(0, 2, 3, 1)
    x = image.permute(0, 2, 3, 1)[..., None]
    return (torch.matmul(m, x)[..., 0] + b).clamp(0.0, 1.0).permute(0, 3, 1, 2)


def net(device):
    key = str(device)
    if key not in _net:
        model = PCTNet()
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        _net[key] = model.to(device).eval()
    return _net[key]


def harmonise(composite, mask, strength=1.0, device="cuda"):
    """The composite ((h, w, 3) in [0, 1]) with its masked part's colour and tone moved toward
    the rest of the picture, `strength` of the way."""
    if strength <= 0 or not (mask > 0.01).any():
        return composite
    with _lock, torch.no_grad():
        model = net(device)
        img = torch.from_numpy(np.ascontiguousarray(composite, np.float32)).permute(2, 0, 1)[None].to(device)
        m = torch.from_numpy(np.ascontiguousarray(mask, np.float32))[None, None].to(device)
        small = F.interpolate(img, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        small_m = F.interpolate(m, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        params = model.params(small, small_m)
        params = F.interpolate(params, size=img.shape[2:], mode="bicubic", align_corners=False)
        out = transform(img, params)
        out = out * m + img * (1.0 - m)
        out = img + float(strength) * (out - img)
    return out[0].permute(1, 2, 0).cpu().numpy()
