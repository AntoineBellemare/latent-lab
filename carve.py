"""A picture carved into a texture, from the picture's own surfaces: a true bas-relief.

The `relief` mapping reads the picture's brightness as height, so dark hair or a black hat becomes
a pit and a white shirt a bump. Here the height comes from the geometry model (geometry.py, MoGe-2):
which way each point of the picture really faces. The texture keeps every pixel of its own; the
picture enters only as the light those surfaces would catch, the hollows between them, and a small
push of the grain over the forms — a face cut into bark, lichen or stone.

The same carved surface, lit evenly and in grey, is what the illusion method shows its ControlNet
when it hides a picture by its form rather than by its photographed light (`control`).

Light directions read as the relief mapping's do: angle 0.9 is light from the upper left.
"""

import numpy as np
from PIL import Image

import composite as cx
import geometry
import masks

TAN_MAX = 3.0  # slopes beyond about 72 degrees (a silhouette's rim) are held there


def surface(geo, focus=None, depth=1.0, reach=0.0):
    """The picture as a height field facing the camera: its slopes (gx, gy) deepened by `depth`,
    softened by `reach` px, and laid flat where `focus` (a soft 0..1 map) is 0."""
    gx, gy = geometry.slopes(geo)
    k = np.minimum(1.0, TAN_MAX / (np.hypot(gx, gy) + 1e-8))
    gx, gy = (gx * k).astype(np.float32), (gy * k).astype(np.float32)
    if reach > 0:
        gx, gy = masks.blur(gx, reach), masks.blur(gy, reach)
    if focus is not None:
        gx, gy = gx * focus, gy * focus
    return gx * depth, gy * depth


def light(gx, gy, angle, elevation, ambient):
    """How much light each point of the height field catches, 1 where it is flat: a sun at `angle`
    around the frame and `elevation` above it (radians), plus light that falls everywhere."""
    e = float(np.clip(elevation, 0.05, 1.5))
    lx, ly, lz = -np.cos(angle) * np.cos(e), -np.sin(angle) * np.cos(e), np.sin(e)
    facing = (-gx * lx - gy * ly + lz) / np.sqrt(gx * gx + gy * gy + 1.0)
    return ambient + (1.0 - ambient) * np.clip(facing, 0.0, None) / lz


def hollows(gx, gy, focus=None, gain=1.0):
    """Where the carved surface sinks below its surroundings, 0..1: the height the slopes integrate
    to, against its own blur at a twentieth of the frame. Read against its own deepest hollows, then
    scaled by `gain`, so a shallow carving has shallow hollows."""
    h = cx.solve(gx, gy)
    cavity = masks.blur(h, max(h.shape) / 20.0) - h
    inside = cavity > 0 if focus is None else (cavity > 0) & (focus > 0.5)
    q = np.quantile(cavity[inside], 0.95) if inside.any() else 1.0
    out = np.clip(gain * cavity / (q + 1e-8), 0.0, 1.0)
    return out if focus is None else out * focus


def carve(texture, geo, angle=0.9, elevation=0.6, depth=1.5, ambient=0.35, hollow=0.6, push=0.01,
          reach=1.5, focus=None):
    """The texture with the picture cut into it: lit by the carved surfaces, darkened in their
    hollows, its grain pushed a little along their slopes."""
    gx, gy = surface(geo, focus, depth, reach)
    h, w = gx.shape
    if push > 0:
        rms = np.sqrt((gx * gx + gy * gy).mean()) + 1e-8
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        span = max(h, w)
        texture = cx.sample(texture, X + gx / rms * push * span, Y + gy / rms * push * span)
    lit = light(gx, gy, angle, elevation, ambient)
    if hollow > 0:
        lit = lit * (1.0 - hollow * hollows(gx, gy, focus, depth / 1.5))  # as set at the default depth
    return np.clip(texture * np.clip(lit, 0.0, 2.0)[..., None], 0.0, 1.0).astype(np.float32)


def control(content, geo, signal, focus=None, soften=0.0):
    """The grey picture the illusion's ControlNet is shown: the light and dark it bends the drawing
    toward, and mid-grey wherever it should leave the texture to itself (QR Code Monster v2 reads
    #808080 as free). `signal` is "form" — the carved surface lit evenly from the upper left, with
    its hollows, so hair colour and clothes count for nothing — or "light", the picture's own
    brightness as photographed."""
    if signal == "form":
        gx, gy = surface(geo, focus, 1.0, 1.0)
        g = light(gx, gy, 0.9, 0.7, 0.0) * (1.0 - 0.7 * hollows(gx, gy, focus))
    else:
        g = content @ cx.LUM
    g = g.astype(np.float32)
    if soften > 0:
        g = masks.blur(g, soften)
    seen = g[focus > 0.5] if focus is not None and (focus > 0.5).any() else g
    lo, hi = np.percentile(seen, [2, 98])
    g = np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    if focus is not None:
        g = 0.5 + (g - 0.5) * focus
    return Image.fromarray((g * 255 + 0.5).astype(np.uint8)).convert("RGB")
