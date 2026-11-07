"""A material made large: from a drawing at 1024 px to 2K or 4K, two ways.

`rebuild` remakes the drawing out of real photographs: the archive photographs nearest it in the
space (or any photographs given) lend their patches, placed where their colour and broad light match
the drawing's (ebsynth). The arrangement is solved at the drawing's size, then the photographs are
read through it at full resolution — they are 6000 × 4000 originals — so every pixel of a 4K result
is one of yours, and sharp. Nothing is invented; the grain is the archive's own.

`redraw` invents the detail instead: the drawing is enlarged in steps of two, and at each step SDXL
(RealVisXL, the transfer method's own pipeline) redraws it tile by tile, the tile ControlNet holding
each tile to what is there and the ViT-H adapter — which reads the space's embeddings — holding it to
the material. Tiles overlap and are feathered together, and the broad colours are taken back from the
enlarged drawing, so tiles cannot drift apart.

`tileable`: the result repeats without a seam. The drawing is then drawn with wrapping convolutions
(explore.wrapped), and here the tiles are laid on a torus — a tile crossing the right edge carries on
at the left — so no seam is ever drawn. A rebuild is solved over a wrapped margin, which keeps the
edges continuous to within a patch.
"""

import math
import pathlib
import threading

import numpy as np
import torch
from PIL import Image

import composite as cx
from loadlock import LOAD

TILE = 1024
OVERLAP = 256
SOLVE = 2048
TILE_NET = "xinsir/controlnet-tile-sdxl-1.0"
PROMPT = "a close-up photograph of a natural material surface, richly detailed, sharp focus, high resolution"
NEGATIVE = "blurry, lowres, painting, illustration, text, watermark, frame"


def _starts(n, tile, wrap):
    """Where tiles start along a side of n pixels: evenly spread, each overlapping the next by about
    OVERLAP; on a torus they go round, otherwise the last one ends at the edge."""
    if n <= tile:
        return [0]
    if wrap:
        k = math.ceil(n / (tile - OVERLAP))
        return [round(i * n / k) for i in range(k)]
    k = math.ceil((n - OVERLAP) / (tile - OVERLAP))
    return [round(i * (n - tile) / (k - 1)) for i in range(k)]


def _window(t):
    """A tile's weight along one side: 1 in the middle, falling to almost 0 over its overlap."""
    i = np.arange(t, dtype=np.float32)
    return np.clip((np.minimum(i, t - 1 - i) + 1) / (OVERLAP / 2), 0.02, 1.0)


class Redraw:
    """The tiled redraw, on the transfer pipeline's SDXL: its modules are shared (they move on and off
    the card as that pipeline moves them), only the tile ControlNet is this one's own."""

    def __init__(self, transfer):
        self.transfer = transfer
        self.net = None
        self.pipe = None
        self.lock = threading.Lock()

    def load(self):
        if self.pipe is None:
            from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionXLControlNetImg2ImgPipeline

            base = self.transfer.load()
            with LOAD:  # loadlock.py
                self.net = ControlNetModel.from_pretrained(TILE_NET, torch_dtype=self.transfer.dtype, use_safetensors=True)
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline(
                vae=base.vae, text_encoder=base.text_encoder, text_encoder_2=base.text_encoder_2,
                tokenizer=base.tokenizer, tokenizer_2=base.tokenizer_2, unet=base.unet, controlnet=self.net,
                scheduler=DPMSolverMultistepScheduler.from_config(base.scheduler.config, use_karras_sigmas=True))
            self.pipe.set_progress_bar_config(disable=True)
            print("  hd redraw up (SDXL tile ControlNet)")
        return self.pipe

    def __call__(self, image, vec, long_side, detail=0.4, hold=0.7, scale=0.8, tileable=False, seed=0,
                 prompt="", steps=24, guidance=5.0, watch=None):
        """`image` (PIL) enlarged to `long_side` in steps of two, redrawn at each step. `detail` is how
        much each tile is redrawn, `hold` how firmly the tile ControlNet keeps it to what is there,
        `scale` how hard the material (`vec`, a ViT-H embedding) pulls."""
        with self.transfer.lock, self.lock, torch.no_grad():
            pipe = self.load()
            base = self.transfer.pipe
            device = self.transfer.device
            # The shared pipeline moves each model on and off the card per call (CPU offload); for a
            # few dozen tiles in a row that is most of the time, so its models stay on the card for
            # this job and the offloading is put back after.
            base.remove_all_hooks()
            try:
                for m in (base.text_encoder, base.text_encoder_2):
                    m.to(device)
                words = pipe.encode_prompt(prompt or PROMPT, device=device, num_images_per_prompt=1,
                                           do_classifier_free_guidance=True, negative_prompt=NEGATIVE)
                for m in (base.text_encoder, base.text_encoder_2):
                    m.to("cpu")
                for m in (base.unet, base.vae, self.net):
                    m.to(device)
                v = torch.tensor(np.asarray(vec, np.float32)[None], dtype=self.transfer.dtype, device=device)
                embeds = torch.cat([torch.zeros_like(v), v])[:, None]
                pipe.set_ip_adapter_scale(float(scale))
                out = image.convert("RGB")
                stage = 0
                while max(out.size) < long_side:
                    k = min(2.0, long_side / max(out.size))
                    w, h = (int(round(s * k / 8)) * 8 for s in out.size)
                    out = self._pass(pipe, out.resize((w, h), Image.LANCZOS), embeds, words, detail, hold,
                                     tileable, seed + 1000 * stage, steps, guidance, watch, stage)
                    stage += 1
                return out
            finally:
                for m in (base.unet, base.vae, self.net, base.text_encoder, base.text_encoder_2):
                    m.to("cpu")
                base.enable_model_cpu_offload(device=device)
                torch.cuda.empty_cache()

    def _pass(self, pipe, big, embeds, words, detail, hold, tileable, seed, steps, guidance, watch, stage):
        a = np.asarray(big, np.float32) / 255.0
        # A plain enlargement is soft, and the tile model keeps what it is shown: an unsharp mask
        # gives it edges to build on (the broad colours are restored from `a` afterwards).
        soft = a
        a = np.clip(a + 0.9 * (a - _blur(a, 1.6, "wrap" if tileable else "reflect")), 0.0, 1.0)
        H, W = a.shape[:2]
        th, tw = min(TILE, H), min(TILE, W)
        acc = np.zeros_like(a, np.float64)
        wsum = np.zeros((H, W, 1), np.float64)
        win = (_window(th)[:, None] * _window(tw)[None, :])[..., None]
        pe, ne, pp, npool = words
        spots = [(y, x) for y in _starts(H, th, tileable) for x in _starts(W, tw, tileable)]
        for n, (y, x) in enumerate(spots):
            rows, cols = (np.arange(y, y + th) % H), (np.arange(x, x + tw) % W)
            tile = Image.fromarray((a[np.ix_(rows, cols)] * 255.0 + 0.5).astype(np.uint8))
            got = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, pooled_prompt_embeds=pp,
                       negative_pooled_prompt_embeds=npool, image=tile, control_image=tile, width=tw, height=th,
                       strength=float(detail), controlnet_conditioning_scale=float(hold),
                       ip_adapter_image_embeds=[embeds], num_inference_steps=int(steps), guidance_scale=float(guidance),
                       generator=torch.Generator(self.transfer.device).manual_seed(seed + n)).images[0]
            g = np.asarray(got, np.float32) / 255.0
            acc[np.ix_(rows, cols)] += g * win
            wsum[np.ix_(rows, cols)] += win
            if watch:
                watch(f"{W}×{H}: tile {n + 1} of {len(spots)}", got)
        out = (acc / np.maximum(wsum, 1e-6)).astype(np.float32)
        # The broad colours from the enlarged drawing, the fine detail from the redraw: tiles that
        # drifted a shade apart are brought back to one picture.
        s = max(W, H) / 128.0
        mode = "wrap" if tileable else "reflect"
        out = out - _blur(out, s, mode) + _blur(soft, s, mode)
        return cx.img(np.clip(out, 0.0, 1.0))


def _blur(a, sigma, mode):
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(a[..., c], sigma, mode=mode) for c in range(a.shape[2])], -1)


def original(path):
    """The full-size original of an archive photograph. The space's files are the training copies
    (2400 × 1600); the originals (6000 × 4000) are hard-linked under the same names in the star
    folders beside them."""
    path = pathlib.Path(path)
    root = path.parents[2] if len(path.parents) > 2 else path.parent
    for folder in ("goofi-yellow-4star", "goofi-yellow-3star"):
        candidate = root / folder / "texture" / path.name
        if candidate.is_file():
            return candidate
    return path


def rebuild(engine, image, paths, long_side, likeness=2.0, patch=5, tileable=False):
    """`image` (PIL) remade from the photographs at `paths`, `long_side` px on its long side, every
    pixel read from the photographs at full size."""
    import ebsynth
    from composite import remap

    paths = [original(x) for x in paths]
    # Which photograph's patch goes where is decided at SOLVE px on the long side: decided at the
    # drawing's 1024 and read at 4096, every 4 × 4 block would come from a different place (mush).
    solve = min(SOLVE, long_side)
    if max(image.size) < solve:
        k0 = solve / max(image.size)
        image = image.resize((round(image.width * k0), round(image.height * k0)), Image.LANCZOS)
    d = cx.arr(image.convert("RGB"))
    H, W = d.shape[:2]
    pad = int(0.08 * max(H, W)) if tileable else 0
    guide = np.pad(d, ((pad, pad), (pad, pad), (0, 0)), mode="wrap") if pad else d

    def sources(height):
        got = []
        for path in paths:
            ow, oh = engine.original_size(path)
            got.append(engine.load_frame(path, max(16, round(height * ow / oh)), height))
        return got

    (my, mx), _ = ebsynth.rebuild(guide, sources(guide.shape[0]), likeness=float(likeness), patch=int(patch),
                                   mapping=True)
    k = long_side / max(W, H)
    gh, gw = round(guide.shape[0] * k), round(guide.shape[1] * k)
    full = sources(gh)
    hs = max(x.shape[0] for x in full)
    style = np.concatenate([np.pad(x, ((0, hs - x.shape[0]), (0, 0), (0, 0)), mode="reflect") for x in full], 1)
    big = remap(my, mx, style, gh, gw)
    if pad:
        p = round(pad * k)
        big = big[p:p + round(H * k), p:p + round(W * k)]
    return cx.img(np.clip(big, 0.0, 1.0))
