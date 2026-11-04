"""The picture's surfaces, read by a geometry model: which way each point faces, and how far away it is.

MoGe-2 (Microsoft, MIT licence; `Ruicheng/moge-2-vitl-normal`) reads one photograph into a point
map in metres, a normal map and the camera's focal length, in one pass. Where a depth model gives
a relative "nearer / further" that sees a person as a flat cut-out, this gives the surface itself:
the turn of a cheek, the folds of a dress, a poster lying flat on its wall. surface.flatten takes
its slopes from the normals, and its perspective from the depth.

Everything is in the camera's frame, OpenCV's way: x to the right, y down, z away from the camera;
a surface facing the camera has a normal with negative z. What the model calls no surface at all —
the sky — is given a surface facing the camera at the furthest distance it found.

The model loads on first use and reads each picture once, at 768 px on the long side at most; any
other frame of the picture is resized from that answer.
"""

import collections
import threading

import numpy as np
import torch

MODEL = "Ruicheng/moge-2-vitl-normal"
SIDE = 768  # the long side the model reads a picture at
KEEP = 32


class Geometry:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.lock = threading.Lock()
        self.memo = collections.OrderedDict()

    def __call__(self, key, picture):
        """{normal (h, w, 3), depth (h, w) in metres, sky (h, w) bool, focal (fx, fy) as fractions
        of the frame's width and height} for the picture `picture()` makes, kept under `key`."""
        with self.lock:
            if key in self.memo:
                self.memo.move_to_end(key)
                return self.memo[key]
            if self.model is None:
                from moge.model.v2 import MoGeModel

                # .float(): a half-precision load in another thread (the body-part model's) can leak
                # its default dtype into this one's construction, leaving the model half float16 —
                # and every later read failing. infer() runs its own fp16 autocast.
                self.model = MoGeModel.from_pretrained(MODEL).float().to(self.device).eval()
                print("  geometry: MoGe-2 up")
            image = picture()
            x = torch.from_numpy(np.asarray(image.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)
            with torch.no_grad():
                got = self.model.infer(x.to(self.device), resolution_level=9)
            normal = got["normal"].float().cpu().numpy()
            depth = got["depth"].float().cpu().numpy()
            sky = ~got["mask"].cpu().numpy().astype(bool) | ~np.isfinite(depth)
            far = float(depth[~sky].max()) if (~sky).any() else 1.0
            depth[sky] = far
            normal[sky] = (0.0, 0.0, -1.0)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
            k = got["intrinsics"].float().cpu().numpy()
            out = dict(normal=normal.astype(np.float32), depth=depth.astype(np.float32), sky=sky,
                       focal=(float(k[0, 0]), float(k[1, 1])))
            self.memo[key] = out
            while len(self.memo) > KEEP:
                self.memo.popitem(last=False)
            return out


def resized(geo, width, height):
    """The same geometry in another frame: normals bilinear and renormalised, depth bicubic. The
    focal length is a fraction of the frame, so it carries over as it is."""
    from PIL import Image

    from composite import _resize

    h, w = geo["depth"].shape
    if (w, h) == (width, height):
        return geo
    n = np.stack([_resize(geo["normal"][..., i], width, height, Image.BILINEAR) for i in range(3)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    return dict(normal=n.astype(np.float32), depth=_resize(geo["depth"], width, height, Image.BICUBIC),
                sky=_resize(geo["sky"].astype(np.float32), width, height, Image.BILINEAR) > 0.5,
                focal=geo["focal"])


def slopes(geo):
    """How steeply each surface turns from the line of sight, as (gx, gy): the rise of a height
    field facing the camera, tan of the angle between the normal and the ray through the pixel,
    along the direction it turns. A flat wall square to the camera is (0, 0) everywhere; a sphere's
    rim tends to infinity."""
    n = geo["normal"]
    h, w = n.shape[:2]
    fx, fy = geo["focal"]
    rx = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) / fx
    ry = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) / fy
    ray = np.stack(np.broadcast_arrays(rx[None, :], ry[:, None], np.ones((h, w), np.float32)), -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    c = -(n * ray).sum(-1)                         # cos of the turn: 1 facing the camera
    across = n + c[..., None] * ray                # the normal's part across the line of sight
    s = np.linalg.norm(across[..., :2], axis=-1) + 1e-8
    tan = np.sqrt(np.clip(1.0 - c * c, 0.0, 1.0)) / np.clip(c, 1e-3, None)
    # A height field facing the camera rises AGAINST the way its normal leans.
    return -tan * across[..., 0] / s, -tan * across[..., 1] / s
