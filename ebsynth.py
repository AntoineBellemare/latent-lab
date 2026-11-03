"""Guided synthesis on the GPU: the picture rebuilt from the material's own patches (ebsynth).

Ebsynth (Jamriska et al., "Stylizing Video by Example", 2019; public domain) solves an image
analogy: given a material with GUIDES (images saying what each of its patches is — how bright,
which way its edges run) and the same guides drawn from the picture, it rebuilds the picture
from the material's own 5×5 patches, each placed where its guides match, with every pixel the
vote of the patches that cover it and a pull toward using all of the material evenly. The result
is made of nothing but the material — its cracks, grain and colour — arranged as the picture.

It is the analogy mode's search (composite.analogy, numpy PatchMatch) done on the GPU, at a
pyramid of sizes, with any number of guides. Its search runs in parallel and the threads race, so
the same settings give a different, equally good arrangement each run (measured: about 8 levels of
255 on average), as stylize's optimisation does; seeding its random start does not change that. The DLL is built from the published source by
<models>/ebsynth/build.bat (CUDA 12.6, sm_86); without it this module says so and the lab carries on.
"""

import ctypes
import os
import pathlib

import numpy as np

DLL = pathlib.Path(os.environ.get("LATENT_LAB_MODELS", pathlib.Path.home() / "latent-lab-models")) / "ebsynth" / "ebsynth.dll"
CUDA, WEIGHTED = 0x0002, 0x0002
_lib = None


def lib():
    global _lib
    if _lib is None:
        if not DLL.is_file():
            raise RuntimeError(f"ebsynth is not built: run {DLL.parent / 'build.bat'}")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(DLL.parent))
        _lib = ctypes.CDLL(str(DLL))
        p = ctypes.c_void_p
        i, f = ctypes.c_int, ctypes.c_float
        _lib.ebsynthRun.argtypes = [i, i, i, i, i, p, p, i, i, p, p, p, p, f, i, i, i, p, p, p, i, p, p]
        _lib.ebsynthRun.restype = None
    return _lib


def _u8(a):
    return np.ascontiguousarray(np.clip(np.asarray(a, np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8))


def synth(style, source_guides, target_guides, weights, uniformity=3500.0, patch=5, iters=6, match_iters=4,
          stop=5, polish=True):
    """The target rebuilt from `style`'s patches. `style` (sh, sw, 3) in [0, 1]; each guide a pair
    of (sh, sw) and (th, tw) maps in [0, 1], `weights` one per guide. Returns (th, tw, 3) floats."""
    sh, sw = style.shape[:2]
    th, tw = target_guides[0].shape[:2]
    sg = _u8(np.stack(source_guides, -1))
    tg = _u8(np.stack(target_guides, -1))
    st = _u8(style)
    ng = sg.shape[-1]
    levels = 1
    for lv in range(32, -1, -1):
        if min(sw, tw, sh, th) * 2.0 ** -lv >= 2 * patch + 1:
            levels = lv + 1
            break
    style_w = np.full(3, 1.0 / 3.0, np.float32)
    guide_w = np.asarray(weights, np.float32)
    per = lambda v: np.full(levels, v, np.int32)  # noqa: E731
    sv, pm, sp = per(iters), per(match_iters), per(stop)
    out = np.zeros((th, tw, 3), np.uint8)
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    lib().ebsynthRun(CUDA, 3, ng, sw, sh, ptr(st), ptr(sg), tw, th, ptr(tg), None, ptr(style_w), ptr(guide_w),
                     float(uniformity), int(patch), WEIGHTED, levels, ptr(sv), ptr(pm), ptr(sp), int(polish),
                     None, ptr(out))
    return out.astype(np.float32) / 255.0


def _matched(a, ref):
    """`a`'s values sent through `ref`'s distribution: the picture's tones in the material's range."""
    order = np.argsort(a.ravel(), kind="stable")
    out = np.empty(a.size, np.float32)
    out[order] = np.interp(np.linspace(0, 1, a.size), np.linspace(0, 1, ref.size), np.sort(ref.ravel()))
    return out.reshape(a.shape)


def analogy(content, texture, follow=2.0, forms=1.0, even=3500.0, patch=5, mapping=False):
    """The picture rebuilt from the texture's patches, guided by light at two scales: its fine
    light (`follow`) and its broad forms (`forms`), each in the texture's own range of tones.
    With `mapping`, the texture position each pixel copies, (map_y, map_x), instead of the image."""
    from composite import LUM, blur

    lt, lc = texture @ LUM, content @ LUM
    span = max(content.shape[:2])
    broad = max(span / 60.0, 2.0)
    guides_t = [blur(lt, 1.0), blur(lt, broad)]
    guides_c = [_matched(blur(lc, 1.0), guides_t[0]), _matched(blur(lc, broad), guides_t[1])]
    if not mapping:
        return synth(texture, guides_t, guides_c, [follow, forms], even, patch)
    return nnf(texture, guides_t, guides_c, [follow, forms], even, patch)


def nnf(style, source_guides, target_guides, weights, uniformity=3500.0, patch=5):
    """The correspondence alone: for each target pixel, the (y, x) of the style pixel it copies."""
    sh, sw = style.shape[:2]
    th, tw = target_guides[0].shape[:2]
    sg, tg, st = _u8(np.stack(source_guides, -1)), _u8(np.stack(target_guides, -1)), _u8(style)
    levels = next((lv + 1 for lv in range(32, -1, -1) if min(sw, tw, sh, th) * 2.0 ** -lv >= 2 * patch + 1), 1)
    per = lambda v: np.full(levels, v, np.int32)  # noqa: E731
    sv, pm, sp = per(6), per(4), per(5)
    field = np.zeros((th, tw, 2), np.int32)
    out = np.zeros((th, tw, 3), np.uint8)
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    lib().ebsynthRun(CUDA, 3, sg.shape[-1], sw, sh, ptr(st), ptr(sg), tw, th, ptr(tg), None,
                     ptr(np.full(3, 1 / 3, np.float32)), ptr(np.asarray(weights, np.float32)), float(uniformity),
                     int(patch), WEIGHTED, levels, ptr(sv), ptr(pm), ptr(sp), 1, ptr(field), ptr(out))
    return field[..., 1].astype(np.float32), field[..., 0].astype(np.float32)


def rebuild(drawn, sources, likeness=2.0, forms=1.0, even=3500.0, patch=5, mapping=False):
    """A drawing remade from real photographs: `sources` (a list of (h, w, 3) arrays, laid side by
    side as one) lend their patches, each placed where its colour and broad light match the
    drawing's. The drawing decides the arrangement; every pixel of the result is a photograph's.
    With `mapping`, the (map_y, map_x) into the side-by-side sources instead."""
    from composite import LUM, blur

    h = max(s.shape[0] for s in sources)
    style = np.concatenate([np.pad(s, ((0, h - s.shape[0]), (0, 0), (0, 0)), mode="reflect") for s in sources], 1)
    span = max(drawn.shape[:2])
    broad = max(span / 60.0, 2.0)
    colour_s = np.stack([blur(style[..., c], 1.0) for c in range(3)], -1)
    colour_t = np.stack([_matched(blur(drawn[..., c], 1.0), colour_s[..., c]) for c in range(3)], -1)
    form_s = blur(style @ LUM, broad)
    form_t = _matched(blur(drawn @ LUM, broad), form_s)
    gs = [colour_s[..., 0], colour_s[..., 1], colour_s[..., 2], form_s]
    gt = [colour_t[..., 0], colour_t[..., 1], colour_t[..., 2], form_t]
    w = [likeness / 3.0] * 3 + [forms]
    if mapping:
        return nnf(style, gs, gt, w, even, patch), style
    return synth(style, gs, gt, w, even, patch)
