"""The material on the picture's forms: the picture's own light on it, and its curve under it.

A mapping mode lays the material over the picture as over a flat sheet: the grain keeps one size
and one light everywhere, so on a face or a dress it reads as a pattern printed on glass in front
of them. Two corrections, shared by every mode, put it ON the things in the picture:

- `shade`: the picture's light and shadow laid over the material. In log light, the picture's
  light between the shading scale (anything finer is the material's own grain, and stays) and the
  broad scale (an object's overall tone) replaces the same band of the result. `tone` lets the
  broad part in too, so a dark dress stays a dark dress. The picture's light is smoothed
  edge-aware, so a shadow stops at an outline instead of bleeding across it. It works in linear
  light, where shading multiplies. Read from brightness, a dark stripe is taken for a shadow; with
  an intrinsic decomposition (intrinsics.py) the light is the picture's shading alone, at every
  scale, and its highlights can be laid back on as shine.
- `flatten` + `wrap`: the material stretched over the picture's relief. Every pixel's slope comes
  from the geometry model's normals (geometry.py: which way the surface there faces), or from the
  depth model's surface, domed toward each object's outline with `roundness`; `folds` adds a
  relief read from the picture's shading. A
  surface that turns away from the camera shows more of itself per pixel, so the material
  advances faster there: it compresses toward the silhouettes, the way a pattern does on a
  cylinder. With `recede` it also shrinks with distance. The texture coordinates are the
  least-squares solution of that stretch (an FFT Poisson solve), so they are one smooth flattening
  of the surface, not a pixel-by-pixel warp.
- `relight`: the material's own relief lit by the picture's light. The scene's light is fitted as
  second-order spherical harmonics — the nine numbers that say how bright a surface facing any
  way would be — to the picture's intrinsic shading over the geometry model's normals. The
  material's fine detail is read as a height field, its slopes tilt the surface's normals, and the
  result is lit by the ratio of the light on the tilted normals to the light on the smooth ones: a
  crack is shadowed on the side away from the picture's sun, whatever light it was photographed in.
"""

import collections
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import composite as cx
import geometry
import masks
import tiling

FLOOR = 0.004  # linear light below which log light means nothing: the black of a JPEG
SOLVED = 1600  # the longest side a flattening is solved at; larger frames get it scaled up


_LUM = torch.from_numpy(cx.LUM)


def lin(a):
    """Linear light, as a torch tensor sharing nothing with `a` (cached frames are read-only).
    The elementwise work of a frame runs in torch: several times numpy's speed on these sizes."""
    return torch.from_numpy(np.array(a, np.float32)).clamp_(0.0, 1.0).pow_(2.2)


def log_light(a):
    return torch.log(lin(a) @ _LUM + FLOOR).numpy()


def light_of(content, scale):
    """The picture's light without its grain: log luminance smoothed by an edge-aware filter
    steered by itself, so shadows keep their edges and stop at outlines."""
    lc = log_light(content)
    return lc, masks.guided(lc, lc, max(scale, 0.75) * 2.0, eps=0.02)


def shade(out, content, strength, scale, tone=0.0, iid=None, shine=0.0):
    """The result relit by the picture: its light at scales above `scale` px swapped for the
    picture's, `strength` of the way (above 1 overshoots: deeper shadows, brighter lights). With
    `iid` (intrinsics.py's reading of the picture, in this frame) the picture's light is its
    shading, and `shine` lays its highlights over the result."""
    if strength <= 0 and not (iid is not None and shine > 0):
        return out
    h, w = content.shape[:2]
    broad = max(h, w) / 8.0
    o = lin(out)
    lo = torch.log(o @ _LUM + FLOOR).numpy()
    r = max(scale, 0.75) * 2.0
    if iid is None:
        lc, sc = light_of(content, scale)
        # The result's light is smoothed by the same filter, steered by the PICTURE's edges: the
        # two are taken alike, so a result relit by its own picture comes back unchanged instead
        # of with the picture's edges counted twice.
        delta = sc - masks.guided(lo, lc, r, eps=0.02)
        if tone < 1:
            # The broad light is the object's own tone, not its shading: it is kept from the result
            # unless `tone` asks for the picture's. Smoothed edge-aware too, or a dark coat beside a
            # bright sky would bleed a halo into both.
            delta -= (1.0 - tone) * (masks.guided(lc, lc, broad * 2.0, eps=0.08) - masks.guided(lo, lc, broad * 2.0, eps=0.08))
    else:
        lc = log_light(content)
        ls = np.log(iid["shading"] @ cx.LUM + FLOOR).astype(np.float32)
        # The shading is light at every scale — a whole side of the scene in shadow too — so all
        # of it above the grain replaces the result's light, about its median. Only the broad
        # TONE is left to choose: the result's own, or the picture's colours' (what the picture's
        # light is once its shading is taken out).
        sl = masks.guided(ls, ls, r, eps=0.02)
        m = float(np.median(sl))
        keep = masks.guided(lo, lc, broad * 2.0, eps=0.08)
        delta = sl - m - masks.guided(lo, lc, r, eps=0.02) + keep
        if tone > 0:
            delta += tone * (masks.guided(lc - ls, lc, broad * 2.0, eps=0.08) + m - keep)
    if strength > 0:
        gain = torch.from_numpy(np.clip(strength * delta, -4.0, 2.0).astype(np.float32)).exp_()
        o.mul_(gain[..., None])
    if iid is not None and shine > 0:
        o.add_(torch.from_numpy(np.asarray(iid["residual"], np.float32)), alpha=float(shine))
    return o.clamp_(0.0, 1.0).pow_(1.0 / 2.2).numpy()


_fields, _lock = collections.OrderedDict(), threading.Lock()


def flatten(depth, content, relief, roundness=0.0, recede=0.0, folds=0.0, scale=2.0, key=None, geo=None):
    """Texture coordinates (u, v), in pixels of the frame, that lay a flat material over the
    picture's relief. With `geo` (geometry.py's reading of the picture) the slopes are the
    surfaces' own, and `relief` scales them: 1 is the surface as the camera sees it, 2 twice as
    steep. Without it they are the depth map's: `relief` is how deep its range is, as a fraction of
    the frame, and `roundness` domes every object down toward its own outline (a depth model sees a
    person as a flat cut-out). `folds` adds the relief read from the shading; `recede` shrinks the
    material with distance. All zero is the identity. `key` names the picture and frame, to keep
    the answer."""
    k = key and (key, geo is not None, round(relief, 4), round(roundness, 4), round(recede, 4), round(folds, 4),
                 round(scale, 3))
    if k:
        with _lock:
            if k in _fields:
                _fields.move_to_end(k)
                return _fields[k]
    h, w = content.shape[:2]
    span = max(h, w)
    if span > SOLVED:
        # The flattening is a smooth field: an export's is solved at a working size and brought up,
        # its coordinates scaled with it. Every knob is a fraction of the frame, so it is the same
        # flattening the working frame showed.
        f = span / SOLVED
        sw, sh = max(8, round(w / f)), max(8, round(h / f))
        u, v = flatten(None if depth is None else cx._resize(depth.astype(np.float32), sw, sh, Image.BILINEAR),
                       cx._resize(np.asarray(content, np.float32), sw, sh, Image.BILINEAR),
                       relief, roundness, recede, folds, scale / f,
                       geo=None if geo is None else geometry.resized(geo, sw, sh))
        got = (cx._resize(u, w, h, Image.BICUBIC) * (w / sw), cx._resize(v, w, h, Image.BICUBIC) * (h / sh))
    else:
        got = _flatten(depth, content, relief, roundness, recede, folds, scale, geo)
    if k:
        with _lock:
            _fields[k] = got
            while len(_fields) > 8:
                _fields.popitem(last=False)
    return got


def _flatten(depth, content, relief, roundness, recede, folds, scale, geo=None):
    h, w = content.shape[:2]
    span = max(h, w)
    fine = max(span / 400.0, 0.75)
    if geo is not None:
        # The surfaces' own slopes, from normals smoothed as finely as the depth would be.
        n = np.stack([masks.blur(geo["normal"][..., i], fine) for i in range(3)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
        zx, zy = geometry.slopes({**geo, "normal": n})
        zx, zy = relief * zx, relief * zy
        if folds > 0:
            _, sc = light_of(content, scale)
            fx, fy = cx.grad(folds * span * 0.03 * (sc - masks.blur(sc, span / 40.0)))
            zx, zy = zx + fx, zy + fy
    else:
        d = masks.blur(depth.astype(np.float32), fine)
        if roundness > 0:
            # Where a blurred copy sits below the depth, the surface is near the edge of something
            # standing in front: it is pulled down toward that edge, so each object becomes a dome.
            # Below the depth only — the background behind it is left alone.
            d = d - roundness * np.maximum(d - masks.blur(d, span / 8.0), 0.0)
        z = relief * span * d
        if folds > 0:
            _, sc = light_of(content, scale)
            # Light read as height — brighter is nearer the light — between the shading scale and a
            # fold's width: a crude shape from shading, but its slopes sit where the folds turn.
            z = z + folds * span * 0.03 * (sc - masks.blur(sc, span / 40.0))
        zx, zy = cx.grad(z)
    g = np.hypot(zx, zy)
    # A surface at slope g shows sqrt(1 + g²) of itself per pixel, along its slope only. Steeper
    # than about 70° it is no surface but a jump from one thing to what is behind it — a
    # silhouette — and the material simply carries on across it.
    stretch = (np.sqrt(1.0 + g * g) - 1.0) * np.clip((4.0 - g) / 1.5, 0.0, 1.0)
    nx, ny = zx / (g + 1e-6), zy / (g + 1e-6)
    rate = np.ones_like(g)
    if recede > 0 and geo is not None:
        rate = _rate(geo["depth"], geo["sky"], recede, span)
    elif recede > 0:
        # Nearer is larger: depth here is disparity, which falls with distance as size does.
        ref = float(np.median(d))
        rate = ((ref + 0.1) / (d + 0.1)) ** (2.0 * recede)
    gux, guy = rate * (1.0 + stretch * nx * nx), rate * stretch * nx * ny
    gvx, gvy = rate * stretch * nx * ny, rate * (1.0 + stretch * ny * ny)
    # The least-squares solve reads the last column's x-step and the last row's y-step as absent.
    for a in (gux, gvx):
        a[:, -1] = 0.0
    for a in (guy, gvy):
        a[-1, :] = 0.0
    u, v = solve(gux, guy), solve(gvx, gvy)
    # A stretched surface holds more material than the frame: read past its edge, the texture
    # mirrors, and a mirrored blotch shows as a symmetric shape. The flattening is scaled back
    # inside the texture — the whole material a little finer — by at most 1.5; beyond that (a
    # steep recede) the mirrors stay.
    fit = max(min(1.0, (w - 1) / (np.ptp(u) + 1e-6), (h - 1) / (np.ptp(v) + 1e-6)), 1.0 / 1.5)
    u = (u - (u.max() + u.min()) / 2.0) * fit + (w - 1) / 2.0
    v = (v - (v.max() + v.min()) / 2.0) * fit + (h - 1) / 2.0
    return u.astype(np.float32), v.astype(np.float32)


def _rate(z, sky, recede, span):
    """Nearer is larger, in proportion to distance: at recede 1 the material keeps one size on
    the ground, as a real one would in this camera's perspective. The distance is taken in log and
    smoothed broadly, and the sky put at the horizon: a rate that jumps from a figure to the trees
    behind it, or to a sky a model calls endless, is no stretch of any surface, and the solve
    turns it into streaks radiating from the jump. Four times either way at most."""
    ground = z[~sky] if (~sky).any() else z
    lz = np.log(np.maximum(np.where(sky, float(np.percentile(ground, 95)), z), 1e-3)).astype(np.float32)
    lz = masks.blur(lz, span / 24.0)
    return np.exp(np.clip(recede * (lz - float(np.median(lz[~sky] if (~sky).any() else lz))), -np.log(4.0), np.log(4.0)))


def solve(gx, gy):
    """The field whose forward steps best match (gx, gy), least squares, mean zero: composite's
    FFT Poisson on the even extension, in torch. Double precision — in single, a coordinate field
    a thousand pixels wide drifts by a pixel — and still several times faster than numpy's."""
    h, w = gx.shape
    gx, gy = torch.from_numpy(np.asarray(gx, np.float64)), torch.from_numpy(np.asarray(gy, np.float64))
    div = (gx - torch.roll(gx, 1, 1)) + (gy - torch.roll(gy, 1, 0))
    big = torch.cat([torch.cat([div, div.flip(1)], 1), torch.cat([div.flip(0), div.flip(0).flip(1)], 1)], 0)
    H, W = big.shape
    ky = 2.0 * torch.cos(2.0 * np.pi * torch.arange(H, dtype=torch.float64) / H) - 2.0
    kx = 2.0 * torch.cos(2.0 * np.pi * torch.arange(W // 2 + 1, dtype=torch.float64) / W) - 2.0
    den = ky[:, None] + kx[None, :]
    den[0, 0] = 1.0
    hat = torch.fft.rfft2(big) / den
    hat[0, 0] = 0.0
    return torch.fft.irfft2(hat, s=(H, W))[:h, :w].numpy()


def wrap(texture, uv, scale=1.0, key=None):
    """The texture read through the flattening, bilinear, its grain `scale` times its own size.
    Where any reading falls past the texture's edge, the whole read is hex-tiled (tiling.py), so
    the material carries on without a seam or a mirrored blotch."""
    h, w = texture.shape[:2]
    u, v = uv
    if scale != 1.0:
        u = (np.asarray(u, np.float64) - (w - 1) / 2.0) / scale + (w - 1) / 2.0
        v = (np.asarray(v, np.float64) - (h - 1) / 2.0) / scale + (h - 1) / 2.0
    if min(u.min(), v.min()) < -0.5 or u.max() > w - 0.5 or v.max() > h - 0.5:
        return tiling.read(texture, u, v, key=key)
    grid = torch.stack([torch.from_numpy(np.asarray(u, np.float32)) * (2.0 / (w - 1)) - 1.0,
                        torch.from_numpy(np.asarray(v, np.float32)) * (2.0 / (h - 1)) - 1.0], -1)[None]
    t = torch.from_numpy(np.array(texture, np.float32)).permute(2, 0, 1)[None]
    with torch.no_grad():
        got = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    return got[0].permute(1, 2, 0).contiguous().numpy()


def _sh(n):
    """The nine second-order spherical-harmonic basis functions of unit normals (..., 3)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, x * z, x * x - y * y, 3.0 * z * z - 1.0], -1)


def light_field(geo, iid, sample=40_000):
    """The scene's light as nine spherical-harmonic coefficients: least squares of the intrinsic
    shading (luminance, median 1) over the normals, sky left out."""
    n = geo["normal"].reshape(-1, 3)
    s = (iid["shading"] @ cx.LUM).reshape(-1)
    keep = np.flatnonzero(~geo["sky"].reshape(-1))
    if keep.size > sample:
        keep = np.random.default_rng(0).choice(keep, sample, replace=False)
    if keep.size < 50:
        return None
    # Most of a photograph faces the camera, so the constant, z and z² terms are nearly one
    # column: unregularised, they cancel each other with huge coefficients. A ridge keeps every
    # term but the constant small unless the normals' spread asks for it — the second order more
    # firmly than the first.
    y = _sh(n[keep]).astype(np.float64)
    ridge = keep.size * np.array([0.0, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5])
    coef = np.linalg.solve(y.T @ y + np.diag(ridge), y.T @ s[keep].astype(np.float64))
    return coef.astype(np.float32)


def relight(out, geo, iid, strength, scale):
    """The result's fine relief (its detail below `scale` px, read as height) lit by the picture's
    light: multiplied, in linear light, by the light on the tilted normals over the light on the
    smooth ones. `strength` is how steep the relief is taken to be."""
    if strength <= 0:
        return out
    coef = light_field(geo, iid)
    if coef is None:
        return out
    o = lin(out)
    lo = torch.log(o @ _LUM + FLOOR).numpy()
    height = lo - masks.blur(lo, max(scale, 0.75) * 2.0)
    hy, hx = np.gradient(masks.blur(height, 0.7))
    n = geo["normal"]
    # The surface's own directions along the frame's x and y: the image axes projected onto it.
    tx = np.array([1.0, 0.0, 0.0], np.float32) - n * n[..., 0:1]
    ty = np.array([0.0, 1.0, 0.0], np.float32) - n[..., 1:2] * n
    tx /= np.linalg.norm(tx, axis=-1, keepdims=True) + 1e-6
    ty /= np.linalg.norm(ty, axis=-1, keepdims=True) + 1e-6
    k = 4.0 * strength  # log-light slope per pixel to surface slope: an even hand, then the knob
    # A height facing the camera: the normal leans away from the way the relief rises.
    tilted = n - k * (hx[..., None] * tx + hy[..., None] * ty)
    tilted /= np.linalg.norm(tilted, axis=-1, keepdims=True) + 1e-6
    flat = np.maximum(_sh(n) @ coef, 0.05)
    lit = np.maximum(_sh(tilted) @ coef, 0.0)
    gain = torch.from_numpy(np.clip(lit / flat, 0.2, 3.0).astype(np.float32))
    sky = torch.from_numpy(geo["sky"])
    gain[sky] = 1.0
    return o.mul_(gain[..., None]).clamp_(0.0, 1.0).pow_(1.0 / 2.2).numpy()
