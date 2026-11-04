"""Where the material goes: one adaptive mask for every mode that has a picture to protect.

Every mode lays its material over the whole frame, which is why a result can read as one picture
on top of another. A mask decides WHERE. It is built from up to three layers, each from a source:

- the picture's own fields — its light, its shadow, its edges, its flat places, its saturation,
  a colour picked on it, its depth (near, far);
- its light and its colours APART (intrinsics.py): where the light falls and where the shadow is,
  whatever colour things are; pale and dark surfaces, whatever light is on them; its shine; a
  colour picked with the light taken out;
- AI: the people in it, an object named in words, a quality named in words, an object clicked on
  (segment.py: a person cutout, a subject matte, Grounding DINO + SAM 2, CLIPSeg, SAM 2), sharp
  cut-outs optionally matted so hair and leaves come out soft;
- the texture's own pattern (its light or its dark);
- shapes: a gradient from one side, a spot placed by a click, grown fractal noise.

Choices that make it adaptive rather than a stencil:

- Coverage is a QUANTILE for every field source. The field is passed through its own distribution,
  so `coverage 0.3` means the 30% of this picture that is darkest, or nearest, or closest to the
  picked colour — the same request on a bright frame and a dark one, at 256px and at 6000. The AI
  object sources are regions already; they take no coverage.
- `follow` snaps the boundary to the picture's own contours with a guided filter, so the material
  occupies regions of the picture instead of floating over them as a soft blob.
- `noise` erodes the boundary with fractal noise, a ragged front like lichen taking a wall.
- Each layer can be inverted and grown or shrunk, then met with the layers before it — `and`,
  `or`, `but not` — and every size-like knob scales with the frame, so a drag draft, a release
  and a full-size export get the same mask.

Over all the layers, a GRADIENT can be laid as an add-on: the selection faded from one side, from
a clicked spot, from its edges inward, or from near to far, measured over the selection's own
extent rather than the frame's, so "from the bottom" takes the tips of the hair and leaves its
roots. A layer's gradient source cuts a side of the frame; this one grades what the layers chose.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import composite as cx
from segment import MATTE

SOURCE_GROUPS = [
    ("the picture", ["light", "shadow", "edges", "flat", "saturation", "colour (click)"]),
    ("light & colour, apart", ["lit", "in shadow", "shine", "pale surfaces", "dark surfaces", "true colour (click)"]),
    ("depth", ["near", "far"]),
    ("AI", ["person", "body part", "subject", "object (words)", "quality (words)", "object (click)",
            "material (click)"]),
    ("the texture", ["texture light", "texture dark"]),
    ("shapes", ["gradient", "spot (click)", "growth"]),
]
SOURCES = ["none"] + [s for _, group in SOURCE_GROUPS for s in group]
COMBINE = ["and", "or", "but not"]
FADES = ["off", "across", "from a spot", "from the edges", "with depth"]
LAYERS = ["", "_2", "_3"]                         # the parameter suffix of each layer
DEPTH_SOURCES = {"near", "far"}
TEXTURE_SOURCES = {"texture light", "texture dark"}
REGIONS = {"person", "body part", "subject", "object (words)", "object (click)"}  # found whole: no coverage
MATTABLE = {"person", "body part", "object (words)", "object (click)"}  # sharp cut-outs a matte can soften
MATTES = {"subject"}  # soft already: taken as they are
AI_SOURCES = REGIONS | {"quality (words)", "material (click)"}
WORDED = {"object (words)", "quality (words)"}
POINTED = {"colour (click)", "spot (click)", "object (click)", "material (click)", "true colour (click)"}
INTRINSIC = {"lit", "in shadow", "shine", "pale surfaces", "dark surfaces", "true colour (click)"}
GOLD = np.array([0.79, 0.64, 0.33], np.float32)


def blur(a, sigma):
    """A Gaussian for masks: separable convolution on the CPU (a few ms at working size), with
    wide blurs run on a shrunk copy. Masks are smooth fields; they don't need the FFT blur."""
    if sigma <= 0.3:
        return a
    h, w = a.shape
    if sigma > 6 and min(h, w) > 64:
        f = int(min(sigma // 3, 32))
        small = cx._resize(a.astype(np.float32), max(8, w // f), max(8, h // f), Image.BOX)
        return cx._resize(blur(small, sigma / f), w, h, Image.BICUBIC)
    r = min(int(math.ceil(3.0 * sigma)), h - 1, w - 1)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    with torch.no_grad():
        t = torch.from_numpy(np.array(a, np.float32))[None, None]  # a copy: cached frames are read-only
        t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return t[0, 0].numpy()


def rank(a, sample=20_000, seed=0):
    """A field passed through its own distribution: 0.3 means 'above 30% of this picture'. The
    distribution is read from a sample: coverage lands within half a percent, in a millisecond."""
    flat = a.ravel()
    if flat.size > sample:
        flat = flat[np.random.default_rng(seed).choice(flat.size, sample, replace=False)]
    ref = np.sort(flat)
    return np.interp(a, ref, np.linspace(0.0, 1.0, ref.size)).astype(np.float32)


def growth(h, w, scale, seed):
    """Fractal noise whose blobs are `scale`-sized. The grids are sized relative to the frame, so
    the same seed grows the same pattern at any resolution. It is summed at the coarsest size that
    still resolves its finest octave, then brought up once."""
    rng = np.random.default_rng(seed)
    cell = max(float(scale), 1.0) * 4.0
    f = int(max(1, min(16, (cell / 16.0) // 2)))  # the finest octave keeps cells of 2px or more
    sh, sw = max(8, h // f), max(8, w // f)
    out = np.zeros((sh, sw), np.float32)
    amp, total = 1.0, 0.0
    for _ in range(5):
        gh = max(2, int(round(h / cell)) + 1)
        gw = max(2, int(round(w / cell)) + 1)
        out += amp * cx._resize(rng.standard_normal((gh, gw)).astype(np.float32), sw, sh, Image.BICUBIC)
        total += amp
        amp *= 0.5
        cell /= 2.0
    out /= total
    return out if (sh, sw) == (h, w) else cx._resize(out, w, h, Image.BICUBIC)


def guided(p, guide, radius, eps=1e-3):
    """Edge-aware smoothing of `p`, steered by `guide`: the boundary moves to the guide's edges.

    The fast guided filter (He & Sun): the linear coefficients are smooth by construction, so they
    are solved on a shrunk copy and brought back up, then applied to the FULL-size guide — the
    edges the boundary snaps to are still the picture's real edges, at a fraction of the blurs.
    """
    h, w = p.shape
    f = int(max(1, min(8, radius // 3)))
    if f > 1 and min(h, w) // f >= 16:
        ps = cx._resize(p.astype(np.float32), w // f, h // f, Image.BOX)
        gs = cx._resize(guide.astype(np.float32), w // f, h // f, Image.BOX)
    else:
        f, ps, gs = 1, p, guide
    s = max(float(radius) / f, 1.0) / 2.0
    mi, mp = blur(gs, s), blur(ps, s)
    var = blur(gs * gs, s) - mi * mi
    cov = blur(gs * ps, s) - mi * mp
    a = cov / (var + eps)
    b = mp - a * mi
    a, b = blur(a, s), blur(b, s)
    if f > 1:
        a = cx._resize(a, w, h, Image.BILINEAR)
        b = cx._resize(b, w, h, Image.BILINEAR)
    return a * guide + b


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def points_of(p, s):
    """A layer's clicks, (x, y, include) in 0–1 of the frame."""
    return [tuple(pt) for pt in (p.get("points" + s) or [])]


def last_point(p, s):
    pts = points_of(p, s)
    return (pts[-1][0], pts[-1][1]) if pts else (0.5, 0.5)


def field(kind, content, texture, depth, scale, p, s="", seg=None):
    """The raw field a ranked source reads, higher where the material should go."""
    lum = content @ cx.LUM
    h, w = lum.shape
    sc = max(float(scale), 0.0)
    if kind == "growth":
        return growth(h, w, max(sc, 1.0), int(p.get("mask_seed", 0)))
    if kind == "gradient":
        # Higher toward the side the angle names: 0 the top, a quarter turn the right.
        a = float(p.get("angle" + s, 0.0))
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        return (x - w / 2) * math.sin(a) - (y - h / 2) * math.cos(a)
    if kind == "spot (click)":
        cx_, cy_ = last_point(p, s)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        return -np.hypot(x - cx_ * w, y - cy_ * h)
    if kind == "light":
        f = lum
    elif kind == "shadow":
        f = -lum
    elif kind in DEPTH_SOURCES:
        f = depth if kind == "near" else -depth
    elif kind == "edges":
        gy, gx = np.gradient(blur(lum, max(sc / 2.0, 0.8)))
        f = np.hypot(gx, gy)
    elif kind == "flat":
        r = max(sc, 1.5)
        m = blur(lum, r)
        f = -np.sqrt(np.clip(blur(lum * lum, r) - m * m, 0.0, None))
    elif kind == "saturation":
        f = content.max(-1) - content.min(-1)
    elif kind in ("colour (click)", "true colour (click)"):
        # Closeness to the colour under the click, in a rough opponent space (light, red–green,
        # yellow–blue) with light counting a little less, so a shade of the colour still counts.
        # The true colour is read off the picture's albedo: its colours with the light taken out,
        # so the same paint in sun and in shadow is one colour.
        pic = seg("intrinsic:albedo", "") ** (1.0 / 2.2) if kind == "true colour (click)" else content
        px, py = last_point(p, s)
        r = max(1, int(0.01 * max(h, w)))
        x0, y0 = int(np.clip(px * w, 0, w - 1)), int(np.clip(py * h, 0, h - 1))
        ref = pic[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1].reshape(-1, 3).mean(0)

        def opp(c):
            return np.stack([0.6 * (c @ cx.LUM), c[..., 0] - c[..., 1],
                             0.5 * (c[..., 0] + c[..., 1]) - c[..., 2]], -1)

        f = -np.linalg.norm(opp(pic) - opp(ref[None, None, :]), axis=-1)
    elif kind in ("lit", "in shadow"):
        f = np.log(seg("intrinsic:shading", "") @ cx.LUM + 1e-3)  # the light alone, whatever it falls on
        f = f if kind == "lit" else -f
    elif kind == "shine":
        f = seg("intrinsic:residual", "") @ cx.LUM
    elif kind in ("pale surfaces", "dark surfaces"):
        f = seg("intrinsic:albedo", "") @ cx.LUM  # the colours alone, whatever light is on them
        f = f if kind == "pale surfaces" else -f
    elif kind == "quality (words)":
        f = seg(kind, p.get("words" + s, ""))
    elif kind == "material (click)":
        f = seg(kind, tuple(map(tuple, points_of(p, s))))
    elif kind == "texture light":
        f = texture @ cx.LUM
    elif kind == "texture dark":
        f = -(texture @ cx.LUM)
    else:
        raise ValueError(f"no mask source {kind!r}")
    return blur(f, sc) if sc > 0 else f


def one(s, content, texture, depth, p, seg):
    """One layer: its source read, ranked (or found whole by a model), eroded by noise, thresholded."""
    kind = p["mask" + s]
    scale, noise, seed = float(p["mask_scale"]), float(p["mask_noise"]), int(p.get("mask_seed", 0))
    h, w = content.shape[:2]
    if kind in WORDED and not str(p.get("words" + s, "")).strip():
        return np.zeros((h, w), np.float32)  # nothing named yet, nothing taken
    if kind in REGIONS:
        arg = (tuple(map(tuple, points_of(p, s))) if kind == "object (click)"
               else p.get("part" + s, "clothing") if kind == "body part" else p.get("words" + s, ""))
        matted = kind in MATTES or (p.get("matte") and kind in MATTABLE)
        r = seg(kind + MATTE if kind in MATTABLE and matted else kind, arg)
        if not r.any():
            return np.zeros((h, w), np.float32)
        if matted and noise <= 0:
            return np.clip(r, 0.0, 1.0)  # a matte's edge is its own: hair-fine, not to be smoothed
        # A region found whole: its boundary is softened over the feature size, so noise has a
        # band to eat into and the threshold a slope to sit on.
        r = blur(r, max(scale, 1.0))
        t = 0.5
    else:
        r = rank(field(kind, content, texture, depth, scale, p, s, seg))
        t = 1.0 - float(p.get("coverage" + s, 0.5))
    if noise > 0 and kind != "growth":
        g = rank(growth(h, w, max(scale, 2.0) * 2.0, seed + 17))
        if kind in REGIONS:
            r = np.clip(r + noise * (g - 0.5), 0.0, 1.0)
        else:
            r = rank((1.0 - noise) * r + noise * g)
    soft = max(float(p["mask_soft"]), 1e-4)
    return smoothstep((r - (t - soft / 2.0)) / soft)


def grow(m, px):
    """The layer's region grown outward (px > 0) or shrunk inward (px < 0), corners rounding as
    they go: the region is blurred and re-thresholded where a straight edge has moved by px."""
    if abs(px) < 0.5:
        return m
    b = blur(m, abs(px) / 2.0)
    t = 0.023 if px > 0 else 0.977  # a step blurred by σ is 2.3% two σ out, 97.7% two σ in
    return smoothstep((b - t) / 0.04 + 0.5)


def ramp(p, m, depth_fn):
    """The add-on gradient over a built mask `m`: 1 on the side the material comes from, fading to
    0, measured over the mask's own extent. `fade_reach` is where it is half faded and
    `fade_length` how long the fade is, both as shares of that extent."""
    kind = p.get("fade", "off")
    inside = m > 0.02
    if kind == "off" or not inside.any():
        return np.ones_like(m)
    h, w = m.shape
    if kind == "from the edges":
        from scipy.ndimage import distance_transform_edt

        f = -distance_transform_edt(m > 0.5).astype(np.float32)  # higher toward the edge
    elif kind == "with depth":
        f = depth_fn()  # higher near
    else:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        if kind == "across":  # higher toward the side the angle names, as the gradient source
            a = float(p.get("fade_angle", 0.0))
            f = (x - w / 2) * math.sin(a) - (y - h / 2) * math.cos(a)
        else:
            px, py = last_point(p, "_fade")
            f = -np.hypot(x - px * w, y - py * h)
    # Read over the selection, a stray speck of it far off left out: 0 where the material comes
    # from, 1 at the far end of the selection.
    lo, hi = np.percentile(f[inside], [1, 99])
    d = np.clip((hi - f) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    if p.get("fade_reverse"):
        d = 1.0 - d
    length = max(float(p.get("fade_length", 1.0)), 1e-3)
    return 1.0 - smoothstep((d - (float(p.get("fade_reach", 0.5)) - length / 2.0)) / length)


def build(p, content, texture, depth_fn, seg_fn=None, faded=True):
    """The mask these settings ask for, in [0, 1], or None when there is none. `faded=False`
    leaves the add-on gradient off: the whole region, for a method that draws into it and grades
    the drawing itself."""
    layers = []
    for s in LAYERS:
        if p.get("mask" + s, "none") == "none":
            break  # a layer counts only when every layer before it is set
        layers.append(s)
    if not layers:
        return None
    depth = depth_fn() if any(p["mask" + s] in DEPTH_SOURCES for s in layers) else None
    m = None
    for s in layers:
        layer = one(s, content, texture, depth, p, seg_fn)
        if p.get("invert" + s):
            layer = 1.0 - layer
        layer = grow(layer, float(p.get("grow" + s, 0.0)))
        if m is None:
            m = layer
        else:
            how = p.get("combine" + s, "and")
            m = m * layer if how == "and" else 1.0 - (1.0 - m) * (1.0 - layer) if how == "or" else m * (1.0 - layer)
    # Snapping to contours would smooth a matte's hair back into an outline: mattes are left as
    # they are when every layer is one.
    matted = all(p["mask" + s] in MATTES or (p.get("matte") and p["mask" + s] in MATTABLE) for s in layers)
    if p["follow"] > 0 and not (matted and float(p["mask_noise"]) <= 0):
        snapped = guided(m, content @ cx.LUM, max(p["mask_scale"] * 2.0, 4.0))
        m = (1.0 - p["follow"]) * m + p["follow"] * snapped
    if faded:
        m = m * ramp(p, m, depth_fn)
    return np.clip(m * float(p["amount"]), 0.0, 1.0)


def apply(p, content, texture, mapped, depth_fn, seg_fn=None):
    """The mapped picture where the mask is, the original picture where it is not."""
    m = build(p, content, texture, depth_fn, seg_fn)
    if m is None:
        return mapped
    if p.get("show_mask"):
        grey = (content @ cx.LUM)[..., None] * 0.55
        return grey * (1.0 - m[..., None]) + (grey * 0.45 + GOLD * 0.75) * m[..., None]
    out = content * (1.0 - m[..., None]) + mapped * m[..., None]
    if float(p.get("harmonise", 0.0)) > 0:
        # The material's colour and tone settled into the rest of the picture (harmonise.py).
        import harmonise

        out = harmonise.harmonise(out, m, float(p["harmonise"]), "cuda" if torch.cuda.is_available() else "cpu")
    return out
