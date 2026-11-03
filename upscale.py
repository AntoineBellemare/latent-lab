"""Enlarge pictures four times with Real-ESRGAN, the network written out here so only its weights are read.

    python upscale.py --weights RealESRGAN_x4plus.pth --out big/ frame-*.png

The weights are xinntao's `RealESRGAN_x4plus.pth` (BSD-3), read with `weights_only`: no code from
the release runs. One pass is a few tenths of a second for a 512 picture on a 3090 in half precision,
so it is for films and stills, not for the live lab (`hd.py` redraws, slower and inventing more).
The same input always gives the same output, so a film that moves smoothly at 512 moves smoothly
enlarged, which a diffusion redraw does not. On the space GAN's materials it draws filaments, rims
and cracks where a plain Lanczos enlargement leaves a blur; putting the plain enlargement's grain
back on top (tried 10-30) only softened it again.
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class Dense(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.conv1 = nn.Conv2d(feat, grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(feat + grow, grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(feat + 2 * grow, grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(feat + 3 * grow, grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(feat + 4 * grow, feat, 3, 1, 1)

    def forward(self, x):
        seen = [x]
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            seen.append(F.leaky_relu(conv(torch.cat(seen, 1)), 0.2))
        return self.conv5(torch.cat(seen, 1)) * 0.2 + x


class Block(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = Dense(feat, grow), Dense(feat, grow), Dense(feat, grow)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class Net(nn.Module):
    """basicsr's RRDBNet at scale 4, under the names its checkpoints use."""

    def __init__(self, feat=64, blocks=23, grow=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, feat, 3, 1, 1)
        self.body = nn.Sequential(*[Block(feat, grow) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(feat, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))


def load(weights, device="cuda"):
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net = Net()
    net.load_state_dict(state.get("params_ema", state.get("params", state)))
    net = net.eval().to(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # a film is thousands of frames of one size
        return net.half().to(memory_format=torch.channels_last)  # a fifth faster on a 3090
    return net


@torch.no_grad()
def enlarge(net, image):
    """[h, w, 3] floats in 0..1 to [4h, 4w, 3]."""
    p = next(net.parameters())
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(p.device, p.dtype)
    # A few pixels of mirror round the edge, so the border is drawn with neighbours like the rest.
    pad = 8
    big = net(F.pad(x, (pad, pad, pad, pad), mode="reflect"))[..., 4 * pad : -4 * pad, 4 * pad : -4 * pad]
    return big.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", required=True, help="RealESRGAN_x4plus.pth")
    ap.add_argument("--out", required=True, help="A folder.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    net = load(args.weights, args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.images:
        image = np.asarray(Image.open(name).convert("RGB"), np.float32) / 255
        big = enlarge(net, image)
        Image.fromarray((big * 255).round().astype(np.uint8)).save(out / (pathlib.Path(name).stem + ".png"))
        print(out / (pathlib.Path(name).stem + ".png"))


if __name__ == "__main__":
    main()
