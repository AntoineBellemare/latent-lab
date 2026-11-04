"""The archive's neural cellular automata, grown over a picture.

style_ca.py trains a tiny rule (twelve channels, one hidden layer, about six thousand numbers) on
the style of a folder of the archive, one per category, children of one parent: started from
nothing, it grows that material, and keeps growing it. Here it grows on the GPU at the frame's
size, and spreads through a mask from its edge inward, along a ragged front — so a slider can let
lichen take a figure, or draw it back.

A rule trained at 128 px makes grain of one size; `reach` runs it on a smaller grid and enlarges
the result, a coarser material from the same rule. `steps` is how long it has grown: early on the
material is still assembling itself; by about 64 ticks (what it was trained at) it is formed.
"""

import collections
import pathlib
import threading

import numpy as np
import torch

import style_ca as ca

_lock = threading.Lock()
_rules = {}
_grown = collections.OrderedDict()


def rules(root):
    """The trained rules in <models>/nca, by name."""
    return sorted(p.stem for p in (pathlib.Path(root) / "nca").glob("*.npy"))


def rule(root, name, device):
    key = (str(root), name)
    if key not in _rules:
        packed = np.load(pathlib.Path(root) / "nca" / f"{name}.npy")
        hidden = (packed.reshape(-1, 4).shape[0] - ca.GROUPS) // ca.BLOCK
        _rules[key] = ca.unpack(packed, ca.Rule(hidden)).to(device).eval()
    return _rules[key]


def grow(root, name, width, height, steps, reach=1.0, seed=0, device="cuda"):
    """The material the rule grows from nothing in `steps` ticks, as (height, width, 3) in [0, 1]."""
    key = (str(root), name, width, height, int(steps), round(float(reach), 3), int(seed))
    with _lock:
        if key in _grown:
            _grown.move_to_end(key)
            return _grown[key]
        net = rule(root, name, device)
        gw, gh = max(16, round(width / reach)), max(16, round(height / reach))
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device)] if "cuda" in str(device) else []):
            torch.manual_seed(int(seed))
            x = torch.zeros(1, ca.CH, gh, gw, device=device)
            for _ in range(int(steps)):
                x = net(x)
            rgb = ca.to_rgb(x)
            if (gw, gh) != (width, height):
                rgb = torch.nn.functional.interpolate(rgb, size=(height, width), mode="bicubic").clamp(0, 1)
        out = rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        _grown[key] = out
        while len(_grown) > 16:
            _grown.popitem(last=False)
        return out


def front(mask, spread, ragged, seed=0):
    """How far the growth has reached through `mask`, from its edge inward: 0 nothing, 1 all of
    it. A grown-noise field roughens the front so it advances in lobes, not as a smooth line."""
    import masks
    from scipy.ndimage import distance_transform_edt

    if spread >= 1.0:
        return mask
    inside = mask > 0.5
    if not inside.any():
        return mask
    depth = distance_transform_edt(inside).astype(np.float32)
    depth /= max(float(depth.max()), 1e-6)
    h, w = mask.shape
    if ragged > 0:
        depth = depth + ragged * 0.5 * (masks.rank(masks.growth(h, w, max(h, w) / 24.0, seed)) - 0.5)
    reached = masks.smoothstep((spread * 1.1 - depth) / 0.06 + 0.5)
    return mask * reached
