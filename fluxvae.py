"""FLUX.2's VAE as a token space beside RAE's: the same grid, other tokens.

The VAE (black-forest-labs/FLUX.2-klein-4B, `vae`, Apache 2.0) writes 32 channels per 8 x 8 pixels.
Each 2 x 2 of those is packed into one 128-channel token and normalised by the VAE's own batch-norm
statistics, which is the space FLUX.2's transformer reads, and one token stands for 16 x 16 pixels as
in RAE's grid. So every token method of the lab runs on it unchanged, with two differences.

- It draws back almost exactly what it was given (a round trip moves a photograph by about 1 level
  in 255, where RAE redraws it), so a material keeps its own grain.
- A straight mix of two unrelated latents cancels much of the spread that carries a VAE latent's
  detail and contrast, and draws a grey double exposure. `settle` puts each channel's mean and spread
  over the grid back where its inputs had them: the VAE's counterpart of RAE's `shell`.

Its principal axes over a set of photographs (`axes`) are photographic controls: on the four-star
archive the first four are, in some order, crispness, colour temperature, exposure and contrast.
`controls` names them by measuring what each one does to a picture.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

REPO = "black-forest-labs/FLUX.2-klein-4B"
KEY = "flux2-vae"  # what the lab's model table calls it, in place of a decoder's path


def ready():
    """The VAE's weights in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(REPO, "vae/config.json"), str)
    except Exception:
        return False


class FluxVAE:
    path = KEY
    side = 48  # tokens on the long side of a frame when no grid is asked for

    def __init__(self, device="cuda"):
        from diffusers import AutoencoderKLFlux2

        self.device = device
        self.dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.vae = AutoencoderKLFlux2.from_pretrained(REPO, subfolder="vae", torch_dtype=self.dtype).to(device).eval()
        bn = self.vae.bn
        self.mean = bn.running_mean.float().to(device).view(1, -1, 1, 1)
        self.std = torch.sqrt(bn.running_var.float().to(device) + bn.eps).view(1, -1, 1, 1)
        self._axes = None

    def grid(self, width, height, long=None):
        long = long or self.side
        if width >= height:
            return max(1, round(long * height / width)), long
        return long, max(1, round(long * width / height))

    def shell(self, z):
        """Tokens go to the decoder as they are: a mix is settled where it is made (`settle`)."""
        return z

    @torch.no_grad()
    def encode(self, images, grid=None):
        """[b, h, w, 3] or [h, w, 3] in 0..1 -> [b, rows*cols, 128] tokens, the picture fitted to the grid."""
        rows, cols = grid or (self.side, self.side)
        x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
        x = (x[None] if x.ndim == 3 else x).permute(0, 3, 1, 2).to(self.device)
        x = F.interpolate(x, size=(rows * 16, cols * 16), mode="bicubic", align_corners=False).clamp(0, 1)
        z = self.vae.encode((x * 2 - 1).to(self.dtype)).latent_dist.mode().float()
        z = (F.pixel_unshuffle(z, 2) - self.mean) / self.std
        return z.permute(0, 2, 3, 1).reshape(len(z), rows * cols, -1)

    @torch.no_grad()
    def decode(self, tokens, grid=None):
        """[b, rows*cols, 128] -> [b, rows*16, cols*16, 3] in 0..1 (float32, on the CPU)."""
        rows, cols = grid or (self.side, self.side)
        z = tokens.to(self.device).float().reshape(len(tokens), rows, cols, -1).permute(0, 3, 1, 2)
        z = F.pixel_shuffle(z * self.std + self.mean, 2).to(self.dtype)
        x = (self.vae.decode(z).sample.float() + 1) / 2
        return x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    @staticmethod
    def settle(z, sources, weights):
        """A mix `z` ([tokens, 128]) of the grids `sources`, weighed by `weights` (one share each), with
        each channel's mean and spread over the grid put back where its inputs had theirs, mixed by
        the same shares."""
        w = np.asarray(weights, np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        want_sd = sum(float(k) * s.std(0) for k, s in zip(w, sources))
        want_mu = sum(float(k) * s.mean(0) for k, s in zip(w, sources))
        return (z - z.mean(0)) / (z.std(0) + 1e-6) * want_sd + want_mu


def axes(space, pictures, cache=None):
    """The token space's principal axes over a set of photographs, strongest first: (mean [D], basis
    [D, D] with one axis a row, spread of each axis). Kept in `cache` (an .npz) once worked out."""
    if cache and pathlib.Path(cache).is_file():
        d = np.load(cache)
        return tuple(torch.from_numpy(d[k]).to(space.device) for k in ("mean", "basis", "spread"))
    bank = torch.cat([space.encode(p, space.grid(p.shape[1], p.shape[0], 32))[0].float() for p in pictures])
    mu = bank.mean(0)
    _, sv, v = torch.linalg.svd(bank - mu, full_matrices=False)
    spread = sv / np.sqrt(len(bank) - 1)
    if cache:
        np.savez(cache, mean=mu.cpu().numpy(), basis=v.cpu().numpy(), spread=spread.cpu().numpy())
    return mu, v, spread


CONTROLS = ("crispness", "warmth", "exposure", "contrast")


def controls(space, ax, probe, first=6):
    """Which of the first axes is which photographic control, and its sign: each control is the axis
    (among the first few) that moves its measure most when the probe picture is pushed along it,
    signed so that pushing raises the measure. {control: (axis, sign)}."""
    mu, basis, spread = ax
    grid = space.grid(probe.shape[1], probe.shape[0], 32)
    z = space.encode(probe, grid)[0].float()

    def measures(img):
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        fine = np.abs(lum - _blur(lum, 2.0)).mean()
        return {"crispness": fine, "warmth": float((img[..., 0] - img[..., 2]).mean()),
                "exposure": float(lum.mean()), "contrast": float(lum.std())}

    base = measures(space.decode(z[None], grid)[0])
    moved = {}
    for k in range(first):
        push = space.decode((z + 2 * spread[k] * basis[k])[None], grid)[0]
        moved[k] = {n: v - base[n] for n, v in measures(push).items()}
    out, taken = {}, set()
    for n in CONTROLS:
        k = max((k for k in moved if k not in taken), key=lambda k: abs(moved[k][n]) / (abs(base[n]) + 1e-3))
        taken.add(k)
        out[n] = (k, 1.0 if moved[k][n] >= 0 else -1.0)
    return out


def _blur(a, s):
    k = int(s * 3) * 2 + 1
    x = np.arange(k) - k // 2
    g = np.exp(-x ** 2 / (2 * s * s))
    g /= g.sum()
    a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, k // 2, mode="reflect"), g, "valid"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(np.pad(c, k // 2, mode="reflect"), g, "valid"), 0, a)
