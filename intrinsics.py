"""The picture's light, apart from its colours: an intrinsic decomposition.

Brightness alone cannot tell a shadow from a dark stripe: read as light, a striped shawl lays its
stripes over the material as if they were folds. Marigold-IID-Lighting (ETH Zürich, OpenRAIL++-M;
`prs-eth/marigold-iid-lighting-v1-1`, through diffusers) splits a photograph, in linear light,
into

- albedo     the surfaces' own colours, as if evenly lit;
- shading    the diffuse light falling on them, colour and all (known only up to a scale);
- residual   what neither explains: highlights, shine, glow (on the shading's scale).

surface.shade takes the picture's light from the shading, and can add the residual back as shine.

The model loads on first use and reads each picture once, at 768 px on the long side at most (four
denoising steps, half a second); any other frame of the picture is resized from that answer.
"""

import collections
import threading

import numpy as np
import torch

MODEL = "prs-eth/marigold-iid-lighting-v1-1"
SIDE = 768
KEEP = 32


class Intrinsics:
    def __init__(self, device):
        self.device = device
        self.pipe = None
        self.lock = threading.Lock()
        self.memo = collections.OrderedDict()

    def __call__(self, key, picture):
        """{albedo, shading, residual}, each (h, w, 3) in linear light, for the picture
        `picture()` makes, kept under `key`. The shading's median is scaled to 1, the residual with
        it, so the shading reads as 'how much light, against the picture's usual'."""
        with self.lock:
            if key in self.memo:
                self.memo.move_to_end(key)
                return self.memo[key]
            if self.pipe is None:
                from diffusers import MarigoldIntrinsicsPipeline

                self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(
                    MODEL, variant="fp16", torch_dtype=torch.float16).to(self.device)
                self.pipe.set_progress_bar_config(disable=True)
                print("  intrinsics: Marigold-IID up")
            image = picture().convert("RGB")
            with torch.no_grad():
                got = self.pipe(image, generator=torch.Generator(self.device).manual_seed(0)).prediction
            albedo, shading, residual = (np.ascontiguousarray(p, np.float32) for p in got)
            # The picture in linear light, to put the residual on the picture's own scale.
            seen = (np.asarray(image, np.float32) / 255.0) ** 2.2
            both = albedo * shading + residual
            k = float((seen * both).sum() / max(float((both * both).sum()), 1e-8))
            norm = float(np.median(shading @ np.array([0.2126, 0.7152, 0.0722], np.float32))) + 1e-4
            # `scale` brings albedo × shading back to the picture's own linear light.
            out = dict(albedo=albedo, shading=shading / norm, residual=residual * k, scale=k * norm)
            self.memo[key] = out
            while len(self.memo) > KEEP:
                self.memo.popitem(last=False)
            return out


def resized(iid, width, height):
    from PIL import Image

    from composite import _resize

    h, w = iid["shading"].shape[:2]
    if (w, h) == (width, height):
        return iid
    return {k: np.stack([_resize(v[..., i], width, height, Image.BILINEAR) for i in range(3)], -1)
            if isinstance(v, np.ndarray) else v for k, v in iid.items()}
