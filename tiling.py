"""The material read anywhere, at any size, without seams: histogram-preserving hex tiling.

A texture is one photograph: read past its edge it has to repeat, and a repeat shows — as a seam,
or, mirrored, as a symmetric blotch. Here the plane is cut into a triangle grid; every vertex of
the grid reads the texture from its own random place, and inside each triangle the three readings
are blended by how near each vertex is (Heitz & Neyret 2018, "High-performance by-example noise
using a histogram-preserving blending operator"; Mikkelsen 2022, "Practical real-time hex-tiling").

A plain blend of three readings is a greyer, flatter material — averaging lowers contrast. So the
blend is done where it doesn't: each channel of the texture (decorrelated first) is mapped through
its own distribution onto a Gaussian, blended there with weights whose squares sum to one (the
variance survives), and mapped back. The weights are sharpened toward the nearest vertex, so most
of the plane is one clean reading and only narrow bands are blended.

`read(texture, U, V)` takes coordinates in pixels of the texture, unbounded. `resize(texture,
scale)` is the texture at another size in its own frame: below 1 a finer grain, repeated.
"""

import collections
import math

import numpy as np
import torch
import torch.nn.functional as F

CELL = 0.3      # the grid's step, as a fraction of the texture's short side
SHARP = 4.0     # how sharply the blend favours the nearest vertex
_SKEW, _UNSKEW = (math.sqrt(3.0) - 1.0) / 2.0, (3.0 - math.sqrt(3.0)) / 6.0
_LUT = 1024


def _gauss_space(texture):
    """The texture's colours decorrelated (PCA) and each component sent through its own
    distribution onto a standard Gaussian, with the tables to send it back."""
    t = torch.from_numpy(np.asarray(texture, np.float32)).reshape(-1, 3)
    mean = t.mean(0)
    _, vecs = torch.linalg.eigh(torch.cov((t - mean).T))
    p = (t - mean) @ vecs
    n = p.shape[0]
    order = p.argsort(0)
    g = torch.empty_like(p)
    # In double: at an export's 24 million pixels, (n - ½) / n rounds to 1 in single precision.
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    q = normal.icdf((torch.arange(n, dtype=torch.float64) + 0.5) / n).float()
    for c in range(3):
        g[order[:, c], c] = q
    # back: evenly spaced Gaussian values, each read off the sorted component at its quantile
    levels = torch.linspace(float(q[0]), float(q[-1]), _LUT)
    sorted_p = p.gather(0, order)
    idx = ((normal.cdf(levels.double()) * n).long()).clamp(0, n - 1)
    back = sorted_p[idx]  # (_LUT, 3)
    return g, mean, vecs, levels, back


def _inverse(g, mean, vecs, levels, back):
    """Gaussian values back to colours, by the tables (linear between levels)."""
    lo, hi = float(levels[0]), float(levels[-1])
    x = ((g.clamp(lo, hi) - lo) / (hi - lo) * (_LUT - 1))
    i0 = x.floor().long().clamp(0, _LUT - 2)
    f = x - i0
    p = torch.stack([back[i0[:, c], c] * (1 - f[:, c]) + back[i0[:, c] + 1, c] * f[:, c] for c in range(3)], -1)
    return p @ vecs.T + mean


def _hash(i, j, seed):
    """Two pseudo-random numbers in [0, 1) per integer vertex, the same on every call."""
    h = (i * 374761393 + j * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    a = (h & 0xFFFF) / 65536.0
    b = ((h >> 16) & 0xFFFF) / 65536.0
    return a, b


def read(texture, U, V, seed=0, cell=CELL, sharp=SHARP, key=None):
    """The texture read at (U, V), pixels of the texture, anywhere on the plane: (h, w, 3).
    `key` names the texture, to keep its Gaussian tables between calls."""
    th, tw = texture.shape[:2]
    c = cell * min(th, tw)
    space = _space(texture, key)
    U, V = np.asarray(U, np.float64), np.asarray(V, np.float64)
    h, w = U.shape
    out = np.empty((h, w, 3), np.float32)
    rows = max(1, 2_000_000 // max(w, 1))  # an export is read in blocks of rows
    for y in range(0, h, rows):
        out[y:y + rows] = _read(space, th, tw, c, U[y:y + rows], V[y:y + rows], seed, sharp)
    return out


_spaces = collections.OrderedDict()


def _space(texture, key):
    k = (key, texture.shape) if key else None
    if k in _spaces:
        _spaces.move_to_end(k)
        return _spaces[k]
    g, mean, vecs, levels, back = _gauss_space(texture)
    got = (g.reshape(*texture.shape[:2], 3).permute(2, 0, 1)[None].contiguous(), mean, vecs, levels, back)
    if k:
        _spaces[k] = got
        while len(_spaces) > 6:
            _spaces.popitem(last=False)
    return got


def _read(space, th, tw, c, U, V, seed, sharp):
    gimg, mean, vecs, levels, back = space
    U, V = torch.from_numpy(U) / c, torch.from_numpy(V) / c
    # The simplex (triangle) grid: skew, find the cell, unskew.
    s = (U + V) * _SKEW
    i, j = torch.floor(U + s), torch.floor(V + s)
    t = (i + j) * _UNSKEW
    x0, y0 = U - (i - t), V - (j - t)
    upper = (x0 > y0).double()
    corners = [(i, j, x0, y0),
               (i + upper, j + 1 - upper, x0 - upper + _UNSKEW, y0 - (1 - upper) + _UNSKEW),
               (i + 1, j + 1, x0 - 1 + 2 * _UNSKEW, y0 - 1 + 2 * _UNSKEW)]
    reads, weights = [], []
    # A vertex reads round a random point kept just further in from the texture's edge than its
    # weight reaches (√½ of a step), so no reading leaves the photograph — and no further, or
    # every vertex reads the same few rows and the tiling shows a rhythm.
    margin = min(c * 0.72, 0.45 * min(th, tw))
    for ci, cj, dx, dy in corners:
        a, b = _hash(ci.long().numpy(), cj.long().numpy(), seed)
        px = margin + torch.from_numpy(a) * (tw - 1 - 2 * margin)
        py = margin + torch.from_numpy(b) * (th - 1 - 2 * margin)
        # the place read: the vertex's own point, plus where this pixel sits from the vertex
        rx, ry = px + dx * c, py + dy * c
        grid = torch.stack([(rx / (tw - 1)) * 2 - 1, (ry / (th - 1)) * 2 - 1], -1).float()[None]
        with torch.no_grad():
            reads.append(F.grid_sample(gimg, grid, mode="bilinear", padding_mode="border", align_corners=True)[0])
        # the vertex's weight: falls to zero at the far side of its triangles
        weights.append(torch.clamp(0.5 - (dx * dx + dy * dy), min=0.0).float())
    w = torch.stack(weights) ** sharp
    w = w / (w.sum(0, keepdim=True) + 1e-12)
    blend = sum(w[k][None] * reads[k] for k in range(3)) / torch.sqrt((w * w).sum(0) + 1e-12)[None]
    h, wd = blend.shape[1:]
    out = _inverse(blend.permute(1, 2, 0).reshape(-1, 3), mean, vecs, levels, back)
    return out.reshape(h, wd, 3).clamp(0.0, 1.0).numpy()


def resize(texture, scale, seed=0, key=None):
    """The texture in its own frame at `scale`: 2 doubles the grain (a crop, read directly); 0.5
    halves it, the material carried on past the photograph's edge by the tiling."""
    h, w = texture.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    U = (xs - (w - 1) / 2.0) / scale + (w - 1) / 2.0
    V = (ys - (h - 1) / 2.0) / scale + (h - 1) / 2.0
    if scale >= 1.0:
        grid = torch.stack([torch.from_numpy(U / (w - 1) * 2 - 1), torch.from_numpy(V / (h - 1) * 2 - 1)], -1).float()[None]
        t = torch.from_numpy(np.array(texture, np.float32)).permute(2, 0, 1)[None]
        with torch.no_grad():
            got = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        return got[0].permute(1, 2, 0).contiguous().numpy()
    return read(texture, U, V, seed, key=key)
