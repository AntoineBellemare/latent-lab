"""A picture as DINOv3's token grid, and back: RAEv2, beside rae.py's DINOv2 one, which stays as it was.

    python rae.py --decoder raev2/stage1/general/dinov3l-k23/decoder.pt --out recon/ photo-*.jpg
    python rae.py --decoder raev2/stage1/general/dinov3l-k23/decoder.pt --film --space texture-ip-sd15.npz \\
        --stops 24 --frame 3:2 --long 768 --upscale RealESRGAN_x4plus.pth --size 1920x1280 --seconds 180 --out morph-v3.mp4
    python bench_rae.py --out bench.json      # both, side by side

Singh, Zheng, Wu, Zhang, Shechtman and Xie, "Improved Baselines with Representation Autoencoders"
(2026). The encoder is DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, gated under Meta's
DINOv3 licence: each user requests it), frozen. Its token is not the last layer's alone but the mean
of blocks 1 to 23 of its 24, each through the final layer norm without its learned scale and shift,
plus the last layer's mean over the picture ("k23", RAEv2's main setting): the early layers keep the
pixels' detail, so the decoder copies where RAE's redraws (PSNR 27.0 on ImageNet against 18.9 for
the last layer alone, their numbers). "k7" takes every other block from 11 to 23 instead: the deeper,
more semantic half, closer to what DINOv2's last layer gives. The decoders are RAEv2's ViT-XL for each,
trained beyond ImageNet (`nyu-visionx/RAEv2-models`, stage1/general/dinov3l-k23 and -k7, CC BY-NC
4.0), the same architecture as RAE's: rae.py's Decoder reads them, with `weights_only`; no code from
the release runs. Which layers a decoder wants is read off its path.

A token is 16 pixels on both sides of it: the encoder reads a picture at 16 pixels a token (its
rotary positions take any size and shape), and the decoder, trained at 256, draws 16 x 16 tokens at a
time, in rae.py's overlapping windows for anything larger.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F

import rae as R

ENCODER = "facebook/dinov3-vitl16-pretrain-lvd1689m"
LAYERS = {"k23": range(1, 24),  # the outputs of blocks 1 to 23 (block 0 left out), as RAEv2 sums them
          "k7": range(11, 24, 2)}  # blocks 11, 13 ... 23


class RAEv2(R.RAE):
    NORMALISED = False  # its decoder draws pixels in 0..1

    def __init__(self, decoder, device="cuda", half=True, encoder=ENCODER, untrained=False):
        """`untrained`: DINOv3 ViT-L's shape with random weights, which times exactly like the real one,
        for benchmarks before access to the weights is granted (its tokens mean nothing)."""
        self.source, self.untrained = encoder, untrained
        found = re.search(r"dinov3l-(k\d+)", str(decoder).replace("\\", "/"))
        self.layers = LAYERS[found.group(1) if found else "k23"]
        self.encoder_dtype = torch.bfloat16 if half and device != "cpu" else torch.float32
        super().__init__(decoder, device, half)

    def load_encoder(self):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        if self.untrained:
            return DINOv3ViTModel(DINOv3ViTConfig(
                hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
                num_register_tokens=4, layerscale_value=1e-5, key_bias=False, patch_size=16, image_size=224))
        return DINOv3ViTModel.from_pretrained(self.source, local_files_only=True, attn_implementation="sdpa")

    @torch.no_grad()
    def encode(self, images, grid=None):
        """[b, h, w, 3] or [h, w, 3] in 0..1 -> [b, rows*cols, 1024] tokens (the picture fitted to the
        grid at 16 pixels a token, square unless one is given)."""
        rows, cols = grid or (self.side, self.side)
        x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
        x = (x[None] if x.ndim == 3 else x).permute(0, 3, 1, 2).to(self.device)
        x = F.interpolate(x, size=(rows * 16, cols * 16), mode="bicubic", align_corners=False)
        x = ((x - R.MEAN.to(x.device)) / R.STD.to(x.device)).to(self.encoder_dtype)
        out = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        first = len(out) - self.encoder.config.num_hidden_layers  # 1 when the embeddings come first
        skip = 1 + self.encoder.config.num_register_tokens  # the class token and the registers
        layers = [F.layer_norm(out[first + k][:, skip:].float(), (out[0].shape[-1],), eps=1e-5) for k in self.layers]
        z = torch.stack(layers).mean(0) + layers[-1].mean(1, keepdim=True)
        return z.to(self.dtype)

    def shell(self, z):
        """DINOv3's mean of normalised layers has no fixed length to put a mix back on: tokens go as they are."""
        return z
