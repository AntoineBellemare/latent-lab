"""Every method, declared once: a node signature the whole app is generated from.

A method here is a record — typed inputs, typed parameters with ranges and one-line docs, a cost
class, and a run function — and nothing else in the app repeats that information. The panels, the
job API and any future graph view all read this file, which is the cure for the disease the old
explorer had: a knob that exists in code but was never hand-copied into an interface.

Kinds a parameter can be: float, int, log (the slider moves an exponent of ten), choice, bool,
seed, text. Inputs a method can take: content (a picture), texture (a texture image), material
(a place in the IP space: preset, terms, archive photo, upload, centre or antipode, with concept
offsets and an against). Cost classes: live (milliseconds), seconds, minutes — the client uses
them to decide what renders on every slider move and what waits for a button.
"""

import collections
import json
import pathlib
import threading

import numpy as np

import composite as cx
import masks
import segment
import surface
import tiling
from ip_space import PRESETS


def P(key, kind, default, lo=None, hi=None, step=None, options=None, doc="", group=None, unit=None,
      shared=False, label=None, choices=None, when=None, optgroups=None, more=False, lead=False):
    """`shared` marks a setting that means the same thing across its family — the working size, the
    quality — so it follows the player from method to method instead of resetting on each.

    `label` is what the panel calls it (the key stays the API's name), `doc` is the line shown under
    it, and `choices` explains each option of a choice, so the panel explains itself.

    `when` says when the setting applies, and the panel shows it only then: a list of clauses that
    must all hold, each `[key, op, value]` with op one of == != > in set (non-empty), or
    `["any", [clause, ...]]`. `group` is the panel section it sits in. `optgroups` sorts a long
    choice into labelled groups: [(title, [option, ...]), ...]. `more` puts it behind its section's
    "more", for the knobs that are seldom touched. `lead` draws a choice with the inputs, as a
    switch, when it decides which input is used (the editor's swatch: the space, or a photograph)."""
    return dict(key=key, kind=kind, default=default, lo=lo, hi=hi, step=step, options=options, doc=doc,
                group=group, unit=unit, shared=shared, label=label or key.replace("_", " "),
                choices=choices or {}, when=when or [], optgroups=optgroups or [], more=more, lead=lead)


def G(p, group, **changes):
    """A shared declaration placed in a section (a copy: the original stays where it is)."""
    return {**p, "group": group, **changes}


# Pixel knobs are tuned in a frame of this long side and scale with the frame they run in, so a
# setting looks the same in a 256px drag, a 512px render and a 6000px export.
REF = 384
EXPORT_CAP = 8192  # the longest side an export will be made at
ASPECTS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"]


def out_sizes(preset):
    """The long sides a diffusion frame can be drawn at, the model's native size first, each in one
    pass: an img2img frame is anchored by its picture and a texture by its own repetition, so SD 1.5
    holds together up to twice its training size. Measured on the 3:2 field picture: one pass at
    1024 beat composing at 512 and refining an enlarged copy (that left a soft, dithered screen),
    and at 1536 SD 1.5 began to lose the picture (sky turned to bark) for 3x the time."""
    native = PRESETS[preset]["size"]
    return [native, native * 3 // 2, native * 2] if native < 1024 else [native, native * 5 // 4, native * 3 // 2]


def I(key, type, doc="", optional=False, when=None):  # noqa: E743 - reads as "input"
    return dict(key=key, type=type, doc=doc, optional=optional, when=when or [])


OUT = "output"
SIZE = P("size", "choice", 384, options=[256, 384, 512, 768, 1024], label="working size", shared=True, group=OUT,
         doc="pixels on the long side while you work; the picture keeps its shape, and export full size "
             "renders at the original resolution")
SEED = P("seed", "seed", 0, doc="a different number gives a different random draw; ⚄ picks one")

MAPPING = {
    "bands": ([P("split", "float", 6, 0.5, 24, 0.25, unit="px", label="split scale",
                 doc="the size where the picture hands over to the texture: larger keeps fewer of the "
                     "picture's details"),
               P("grain", "float", 1.0, 0, 2, 0.05, label="texture strength",
                 doc="how strongly the texture's fine grain comes in")],
              lambda c, t, p: cx.bands(c, t, p["split"], p["grain"])),
    "displace": ([P("amp", "float", 0.05, 0, 0.12, 0.002, label="push distance",
                    doc="how far the texture's pixels are pushed along the picture's slopes"),
                  P("reach", "float", 3, 1, 10, 0.25, unit="px", label="smoothing",
                    doc="how much the picture is softened before its slopes are read: higher follows "
                        "big forms, lower follows fine detail"),
                  P("shade", "float", 0.7, 0, 1.5, 0.05, label="shading",
                    doc="how strongly the picture's light and dark shade the result")],
                 lambda c, t, p: cx.displace(c, t, p["amp"], p["reach"], p["shade"])),
    "blend": ([P("wmix", "float", 0.65, 0, 1, 0.02, label="picture edges",
                 doc="0 lets the texture's edges win everywhere, 1 lets the picture's edges win")],
              lambda c, t, p: cx.blend(c, t, p["wmix"])),
    "chroma": ([P("share", "float", 0.2, 0, 1, 0.02, label="texture light",
                  doc="how much of the texture's own light and shade comes back (0 = only its colours)")],
               lambda c, t, p: cx.chroma(c, t, p["share"])),
    "grain": ([P("share", "float", 0.25, 0, 1, 0.02, label="picture light",
                 doc="how much of the picture's own light and shade comes back over the texture's")],
              lambda c, t, p: cx.chroma(t, c, p["share"])),
    "mask": ([P("level", "float", 0.55, 0, 1, 0.02, label="waterline",
                doc="how high the texture rises, from the picture's darkest tones up"),
              P("feather", "float", 0.08, 0.01, 0.3, 0.01, label="edge softness",
                doc="how gradually the texture fades out at the waterline")],
             lambda c, t, p: cx.mask(c, t, p["level"], p["feather"])),
    "relief": ([P("angle", "angle", 0.9, 0, 6.283, 0.05, label="light direction",
                  doc="where the light comes from, all the way round the frame"),
                P("ambient", "float", 0.55, 0.15, 1.1, 0.05, label="ambient light",
                  doc="light that falls everywhere: higher flattens the relief"),
                P("reach", "float", 2.5, 1, 8, 0.25, unit="px", label="smoothing",
                  doc="how much the picture is softened before its slopes are read")],
               lambda c, t, p: cx.relief(c, t, p["angle"], p["ambient"], p["reach"])),
    "carve": ([P("angle", "angle", 0.9, 0, 6.283, 0.05, label="light direction",
                 doc="where the light comes from, all the way round the frame"),
               P("elevation", "angle", 0.6, 0.1, 1.45, 0.05, label="light height",
                 doc="how high the light stands: low rakes across the carving and deepens it, high flattens it"),
               P("depth", "float", 1.5, 0, 4, 0.05, label="carving depth",
                 doc="how deep the picture is cut: 0 flat, 1 its surfaces as they are, more exaggerates them"),
               P("ambient", "float", 0.35, 0, 1, 0.02, label="ambient light",
                 doc="light that falls everywhere: higher softens the carving"),
               P("hollow", "float", 0.6, 0, 1.5, 0.05, label="hollows",
                 doc="how dark the carving's hollows go: eye sockets, the corners of the mouth, folds"),
               P("push", "float", 0.01, 0, 0.05, 0.002, label="grain over the forms",
                 doc="how far the texture's grain is pushed along the carved slopes, so it bends over them"),
               P("reach", "float", 1.5, 0, 10, 0.25, unit="px", label="smoothing",
                 doc="how much the surfaces are softened before they are cut: higher keeps only the big forms"),
               P("carve_on", "choice", "subject", options=["subject", "person", "whole picture"], label="carve",
                 choices={"subject": "only the main subject: the rest stays plain texture",
                          "person": "only the people: the rest stays plain texture",
                          "whole picture": "everything the geometry model sees, background too"},
                 doc="what of the picture is cut into the texture")],
              None),
    "veins": ([P("depth", "float", 1.1, 0, 1.8, 0.05, label="engraving depth",
                 doc="how deeply the texture's cracks and lines are cut into the picture"),
               P("reach", "float", 3, 1, 8, 0.25, unit="px", label="crack width",
                 doc="how wide a line in the texture has to be to count as a crack")],
              lambda c, t, p: cx.veins(c, t, p["depth"], p["reach"])),
    "flow": ([P("length", "float", 0.06, 0, 0.12, 0.002, label="stroke length",
                doc="how far the texture is combed along the picture's contours"),
              P("reach", "float", 3, 1, 8, 0.25, unit="px", label="smoothing",
                doc="how much the picture is softened before its contours are read"),
              P("smear", "float", 0.25, 0, 1, 0.02, label="smear",
                doc="0 combs sharply and keeps the grain, 1 smears softly"),
              P("steps", "int", 8, 3, 16, 1, label="steps",
                doc="how many small moves each stroke is made of (more = smoother curves, slower)")],
             lambda c, t, p: cx.flow(c, t, p["length"], p["reach"], p["smear"], p["steps"])),
    "quilt": ([P("alpha", "float", 0.85, 0.2, 0.98, 0.02, label="match picture",
                 doc="how strictly each patch matches the picture's light, versus fitting its neighbours"),
               P("block", "int", 36, 16, 96, 4, unit="px", label="patch size",
                 doc="the size of the texture pieces the picture is rebuilt from"),
               P("candidates", "int", 400, 100, 1000, 50, label="candidates",
                 doc="how many texture pieces are tried for each spot (more = closer match, slower)"),
               SEED],
              lambda c, t, p, mapping=False: cx.quilt(c, t, p["alpha"], p["block"], p["candidates"], p["seed"],
                                                      mapping=mapping)),
    "analogy": ([P("patch", "int", 7, 3, 15, 2, unit="px", label="patch size",
                   doc="how big a neighbourhood each pixel compares when it looks for its match"),
                 P("melt", "float", 1.0, 0, 3, 0.25, unit="px", label="melt",
                   doc="0 copies the texture's pixels verbatim, higher softens them into a warp"),
                 P("rounds", "int", 8, 2, 12, 1, label="search rounds",
                   doc="how long the match search runs (more = better matches, slower)"),
                 SEED],
                lambda c, t, p, mapping=False: cx.analogy(c, t, p["patch"], 0.0 if mapping else p["melt"],
                                                          p["rounds"], p["seed"], mapping=mapping)),
    "relit": ([P("own_relief", "float", 0.5, 0, 1.5, 0.05, label="material's own relief",
                 doc="how much of the light the material was photographed in stays as its own bumps: 0 only its "
                     "colours, flat; 1 its relief as photographed"),
               P("light_hue", "float", 1.0, 0, 1, 0.05, label="colour of the light",
                 doc="1 the picture's light keeps its colour (a warm sun stays warm on the material); 0 white light"),
               P("keep_colour", "float", 0.0, 0, 1, 0.05, label="picture's own colours",
                 doc="mix the picture's own colours back into the material's"),
               P("glints", "float", 1.0, 0, 1.5, 0.05, label="picture's shine",
                 doc="the picture's highlights and glow laid back over the material")],
              None),
    "synth": ([P("follow", "float", 2.0, 0.2, 8, 0.1, label="follow the picture",
                 doc="how closely the patches must match the picture's fine light: higher a truer likeness, "
                     "lower freer material"),
               P("forms", "float", 1.0, 0, 4, 0.1, label="keep broad forms",
                 doc="how much the picture's broad light and shadow decide where each kind of patch goes"),
               P("even", "log", 3.5, 2.5, 4.5, 0.05, label="use the material evenly",
                 doc="how hard every part of the material is pulled into use, instead of its few best patches "
                     "repeated"),
               P("patch", "int", 5, 3, 9, 2, label="patch size",
                 doc="the side of the patches the picture is rebuilt from: larger keeps more of the material's "
                     "own structure")],
              lambda c, t, p, mapping=False: _ebsynth().analogy(c, t, p["follow"], p["forms"], 10 ** p["even"],
                                                               int(p["patch"]), mapping=mapping)),
    "guided": ([P("radius", "float", 14, 2, 60, 2, unit="px", label="radius",
                  doc="how big an area each local fit covers: larger pours the texture over bigger shapes"),
                P("eps", "float", 0.015, 0.002, 0.08, 0.002, label="edge threshold",
                  doc="how strong an edge must be to stop the texture: lower keeps more edges"),
                P("detail", "float", 0.35, 0, 1, 0.05, label="texture detail",
                  doc="how much of the texture's fine grain is added back on top")],
               lambda c, t, p: cx.guided(c, t, p["radius"], p["eps"], p["detail"])),
}

def _ebsynth():
    import ebsynth

    return ebsynth


_SURFACE = "surface"
_REBUILD = "rebuild"
SURFACE_PX = ["shade_grain"]


def surface_params(wrap=True, size=False):
    """The picture's light laid over the material, and (where the material is laid pixel for pixel,
    not rebuilt from pieces) the material curved over the picture's relief; for the mapping modes,
    which read the texture image itself, the material's size."""
    on = [["shading", ">", 0]]
    out = [P("tex_scale", "float", 1.0, 0.25, 3, 0.05, group="material", shared=True, label="material size",
             doc="how big the material's grain is: below 1 finer, carried on past the photograph's edge in "
                 "overlapping patches with no seam; above 1 coarser")] if size else []
    out += [
        P("shading", "float", 0.0, 0, 1.5, 0.05, group=_SURFACE, shared=True, label="picture's shading",
          doc="lay the picture's own light and shadow over the material so it sits in the scene's light: "
              "1 swaps the material's light for the picture's, above 1 deepens it"),
        P("light", "choice", "intrinsic", options=["intrinsic", "brightness"], group=_SURFACE, shared=True, more=True,
          label="read light from", when=on,
          choices={"intrinsic": "the picture's light apart from its colours (Marigold-IID): a striped shawl "
                                "is lit, not shadowed by its own stripes",
                   "brightness": "the picture's brightness, as before: quick, but dark patterns read as shadows"},
          doc="how the picture's light is told from its colours"),
        P("relief_light", "float", 0.0, 0, 2, 0.05, group=_SURFACE, shared=True, more=True, label="lit relief",
          when=on + [["light", "==", "intrinsic"]],
          doc="light the material's own bumps and cracks from the direction the picture is lit from (its "
              "light fitted over the geometry model's surfaces), not the light they were photographed in"),
        P("shine", "float", 0.0, 0, 1.5, 0.05, group=_SURFACE, shared=True, more=True, label="picture's shine",
          when=on + [["light", "==", "intrinsic"]],
          doc="lay the picture's highlights (gloss, wet glints, glow) over the material too"),
        P("shade_tone", "float", 0.5, 0, 1, 0.05, group=_SURFACE, shared=True, more=True, label="object tones", when=on,
          doc="0 keeps the material's own brightness and takes only the shape of the light; 1 also takes "
              "the picture's darks and lights, so a dark coat stays dark"),
        P("shade_grain", "float", 6, 1, 24, 0.5, unit="px", group=_SURFACE, shared=True, more=True, label="material detail",
          when=on, doc="the material's own detail up to this size is kept; anything broader takes the "
                       "picture's light"),
    ]
    if wrap:
        on = [["wrap", ">", 0]]
        out += [
            P("wrap", "float", 0.0, 0, 2, 0.05, group=_SURFACE, shared=True, label="wrap over forms",
              doc="curve the material over the picture's relief: it bunches up where a surface turns away, "
                  "as a pattern does round a cylinder"),
            P("forms", "choice", "geometry", options=["geometry", "depth"], group=_SURFACE, shared=True, more=True,
              label="read forms with", when=on,
              choices={"geometry": "a geometry model (MoGe-2): which way every surface faces — a cheek, a fold, "
                                   "a poster flat on its wall; wrap 1 is the surface as the camera sees it",
                       "depth": "the depth model (DPT), as before: a rough relief in which people are flat "
                                "cut-outs, domed by roundness"},
              doc="which model reads the picture's forms"),
            P("roundness", "float", 0.6, 0, 1, 0.05, group=_SURFACE, shared=True, more=True, label="roundness",
              when=on + [["forms", "==", "depth"]],
              doc="dome every object down toward its own outline; a depth model sees people as flat cut-outs"),
            P("folds", "float", 0.0, 0, 2, 0.05, group=_SURFACE, shared=True, more=True, label="folds", when=on,
              doc="also bend the material over the folds and creases read from the picture's shading"),
            P("recede", "float", 0.0, 0, 1, 0.05, group=_SURFACE, shared=True, more=True, label="shrink with distance",
              when=on, doc="the material gets finer as the picture goes into the distance (for landscapes); "
                           "read by geometry, 1 is the camera's own perspective"),
        ]
    return out


SURFACE_KEYS = {p["key"] for p in surface_params()}


_WHERE, _SECOND, _THIRD = "where", "second layer", "third layer"
_EDGES = "mask edges"
_FADE = "gradient"
SOURCE_CHOICES = {
    "light": "the brightest parts of the picture",
    "shadow": "the darkest parts of the picture",
    "edges": "along the picture's outlines and contours",
    "flat": "the smooth, untextured parts of the picture (sky, skin, walls)",
    "saturation": "the most colourful parts of the picture",
    "colour (click)": "a colour: click the picture to pick it; coverage takes its closest shades",
    "lit": "where the picture's light falls, whatever colour the things it falls on (intrinsic shading)",
    "in shadow": "where the picture is in shadow, whatever colour the things in it (intrinsic shading)",
    "shine": "the picture's highlights, glints and glow (what its colours and light don't explain)",
    "pale surfaces": "things pale in colour, in sun or in shadow alike (intrinsic albedo)",
    "dark surfaces": "things dark in colour, in sun or in shadow alike (intrinsic albedo)",
    "true colour (click)": "a colour with the light taken out: click it, and the same paint in sun and in shade "
                           "is taken together",
    "near": "what is closest to the camera (from a depth model)",
    "far": "what is furthest from the camera (from a depth model)",
    "person": "the people in the picture, cut out by a person model",
    "body part": "a part of the people in the picture — their clothing, hair, skin, face, hands — (Sapiens2)",
    "subject": "the picture's main subject — a person, a statue, an animal, a thing — cut out with soft "
               "edges (BiRefNet matting)",
    "object (words)": "things you name in words, found and cut out sharply (Grounding DINO + SAM 2)",
    "quality (words)": "a quality or thing you name, as a soft map (CLIPSeg): wet, shiny, rough, sky",
    "object (click)": "an object you click on, cut out sharply (SAM 2); shift-click to exclude a part",
    "material (click)": "the same material as where you click — moss, rust, a fabric, water — wherever it is in "
                        "the picture (DINOv2 features); shift-click a material to leave out",
    "texture light": "where the texture itself is bright",
    "texture dark": "where the texture itself is dark: its cracks, pores and hollows",
    "gradient": "a gradient from one side of the frame; the angle picks the side",
    "spot (click)": "a round spot: click the picture to place it; coverage sets its size",
    "growth": "organic patches of grown noise, whatever the picture shows",
}
MASK_PX = ["mask_scale"] + ["grow" + s for s in masks.LAYERS]  # pixel knobs, scaled with the frame


def mask_params(texture=True):
    """The mask's settings: three layers, each shown once the layers before it are set, each knob
    only for the sources that read it; then the boundary and strength, shared by all layers.
    Without a texture image (diffusion), the texture's own sources are left out."""
    sources = [x for x in masks.SOURCES if texture or x not in masks.TEXTURE_SOURCES]
    groups = [(title, [x for x in group if x in sources]) for title, group in masks.SOURCE_GROUPS
              if any(x in sources for x in group)]
    ranked = [x for x in sources if x != "none" and x not in masks.REGIONS]
    out = []
    for n, (s, group) in enumerate(zip(masks.LAYERS, (_WHERE, _SECOND, _THIRD))):
        before = [["mask" + t, "!=", "none"] for t in masks.LAYERS[:n]]
        on = before + [["mask" + s, "!=", "none"]]
        nth = ["", "second ", "third "][n]
        out.append(P("mask" + s, "choice", "none", options=sources, optgroups=groups, group=group, shared=True,
                     label="source", when=before,
                     choices={**SOURCE_CHOICES, "none": ["no mask: the material covers the whole picture",
                                                         "no second layer", "no third layer"][n]},
                     doc="what decides which parts of the picture take the material" if n == 0
                     else f"optional: a {nth}layer, met with the layers above"))
        if n:
            out.append(P("combine" + s, "choice", "and", options=masks.COMBINE, group=group, shared=True,
                         label="combine", when=on,
                         choices={"and": "only where this layer and the ones above agree",
                                  "or": "where this layer or the ones above are",
                                  "but not": "the layers above, with this one cut away"},
                         doc="how this layer meets the layers above"))
        out += [
            P("words" + s, "text", "", group=group, shared=True, label="words",
              when=on + [["mask" + s, "in", sorted(masks.WORDED)]],
              doc="what to look for, e.g. the dress, trees; commas separate several"),
            P("part" + s, "choice", "clothing", options=list(segment.PARTS), group=group, shared=True,
              label="part", when=on + [["mask" + s, "==", "body part"]],
              doc="which part of the people the material goes on"),
            P("points" + s, "points", [], group=group, shared=True, label="clicks",
              when=on + [["mask" + s, "in", sorted(masks.POINTED)]],
              doc="click the picture to place them; for objects and materials, shift-click marks one to leave out"),
            P("angle" + s, "angle", 0.0, 0, 6.283, 0.05, group=group, shared=True, label="side",
              when=on + [["mask" + s, "==", "gradient"]],
              doc="which side the material comes from: 0 the top, then round clockwise"),
            P("coverage" + s, "float", 0.5, 0, 1, 0.01, group=group, shared=True, label="coverage",
              when=on + [["mask" + s, "in", ranked]],
              doc="how much of the picture this layer takes: 0.3 = the 30% that matches its source best"),
            P("invert" + s, "bool", False, group=group, shared=True, label="invert", when=on, more=True,
              doc="take everything this layer does not"),
            P("grow" + s, "float", 0.0, -24, 24, 0.5, unit="px", group=group, shared=True, label="grow / shrink",
              when=on, more=True, doc="push this layer's edge outward (more) or inward (less); corners round as it goes"),
        ]
    on = [["mask", "!=", "none"]]
    fading = on + [["fade", "!=", "off"]]
    out += [
        P("fade", "choice", "off", options=masks.FADES, group=_FADE, shared=True, label="gradient", when=on,
          choices={"off": "no gradient: the selection taken evenly",
                   "across": "strongest on one side of the selection, fading across it",
                   "from a spot": "strongest at a spot you click, fading away from it",
                   "from the edges": "strongest at the selection's edges, fading toward its middle",
                   "with depth": "strongest on what is near, fading with distance"},
          doc="fade the selection, measured over its own extent rather than the frame's"),
        P("fade_angle", "angle", 0.0, 0, 6.283, 0.05, group=_FADE, shared=True, label="side",
          when=fading + [["fade", "==", "across"]],
          doc="which side of the selection the material is strongest on: 0 the top, then round clockwise"),
        P("points_fade", "points", [], group=_FADE, shared=True, label="spot",
          when=fading + [["fade", "==", "from a spot"]], doc="click the picture to place the spot"),
        P("fade_reach", "float", 0.5, 0, 1, 0.01, group=_FADE, shared=True, label="reach", when=fading,
          doc="how far into the selection the material reaches: where it is half faded"),
        P("fade_length", "float", 1.0, 0, 1, 0.01, group=_FADE, shared=True, label="fade length", when=fading,
          doc="how long the fade is, as a share of the selection: 1 fades across all of it, 0 is a clean line"),
        P("fade_reverse", "bool", False, group=_FADE, shared=True, label="reverse", when=fading,
          doc="the other way round: strongest at the far end (the middle, the far side, the distance)"),
    ]
    out += [
        P("show_mask", "bool", False, when=on, group=_WHERE, shared=True, label="show mask",
          doc="draw the mask itself in gold instead of the result, to tune it"),
        P("matte", "bool", False, group=_EDGES, shared=True, label="matte the cut-outs",
          when=on + [["any", [["mask" + t, "in", sorted(masks.MATTABLE)] for t in masks.LAYERS]]],
          doc="resolve the edge of a person or object cut-out pixel by pixel (ViTMatte), so hair, fur and "
              "leaves come out soft instead of as a smooth outline"),
        P("mask_soft", "float", 0.2, 0, 1, 0.01, when=on, group=_EDGES, shared=True, label="edge softness",
          doc="how gradually the material fades out at the mask's edge"),
        P("mask_scale", "float", 4, 0, 32, 0.5, when=on, group=_EDGES, shared=True, unit="px", label="feature size",
          more=True,
          doc="small follows fine grain, large follows big forms"),
        P("follow", "float", 0.6, 0, 1, 0.05, when=on, group=_EDGES, shared=True, label="follow contours",
          doc="snap the mask's edge onto the picture's own outlines"),
        P("mask_noise", "float", 0.0, 0, 1, 0.02, when=on, group=_EDGES, shared=True, label="ragged edge",
          doc="break the mask's edge up with grown noise, like lichen spreading"),
        P("mask_seed", "seed", 0, group=_EDGES, shared=True, label="noise seed", more=True,
          when=on + [["any", [["mask_noise", ">", 0]] + [["mask" + t, "==", "growth"] for t in masks.LAYERS]]],
          doc="a different number grows a different noise pattern (ragged edge and growth)"),
        P("harmonise", "float", 0.0, 0, 1, 0.05, when=on, group=_EDGES, shared=True, label="settle colours",
          more=True,
          doc="move the material's colour and tone toward the rest of the picture, so it sits in the scene's light "
              "and white balance instead of looking pasted on (PCT-Net)"),
        P("amount", "float", 1.0, 0, 1, 0.01, when=on, group=_EDGES, shared=True, label="strength",
          doc="how fully the material replaces the picture inside the mask"),
    ]
    return out


MASK = mask_params()
MASK_KEYS = {p["key"] for p in MASK}
FADE_KEYS = {p["key"] for p in MASK if p["group"] == _FADE}

GROUPS = {
    OUT: "How big and how finished the result is.",
    "structure": "How much of the picture survives the redraw.",
    "material": "How the chosen material is laid on.",
    "finishing pass": "Optional: a second, lighter pass in another material.",
    "illusion": "How the picture hides in the texture: what of it, how visibly, and for how long it steers.",
    "enlarge": "How the large picture finds its detail: real photographs' patches, or a tile-by-tile redraw.",
    "walk": "Where you are between the two materials.",
    "draw": "Which point of the GAN's latent is drawn, and how freely.",
    "optimisation": "How long and how hard the optimiser works.",
    "patch swap": "How the texture's neighbourhoods are matched and swapped in.",
    _REBUILD: "Optional: the drawing remade out of patches of real photographs — the archive's own, or a "
              "texture — so its arrangement is the generator's and its every pixel a photograph's.",
    _SURFACE: "Put the material ON the things in the picture: lit by the picture's own light, and "
              "curved over its forms, instead of printed flat across the frame.",
    _WHERE: "Choose which parts of the picture take the material; everywhere else the original "
            "picture stays. Tick show mask to see where it goes.",
    _SECOND: "Optional: a second layer, added to the first, intersected with it, or cut out of it.",
    _THIRD: "Optional: a third layer, met with the two above.",
    _FADE: "Optional: the selection faded instead of taken evenly — from one side, a spot, its edges or "
           "the near ground — so the material can take the tips of the hair and leave its roots.",
    _EDGES: "How the mask's boundary behaves, and how strongly the material fills it.",
    "words & LoRA": "Words that steer the drawing, and the archive-trained LoRA underneath them.",
    "latent axes": "The model's own main directions of variation, strongest first; axis 0 changes the most.",
    "categories": "How much of each material family the GAN draws; they mix.",
    "compose": "How the materials are laid in the token grid: in what order, how they meet, and how much of "
               "the picture stays under them.",
    "mosh": "How the tokens are moved out of their places, and whose tokens they are.",
    "hybrid": "How the materials make one new texture.",
    "drawn": "What the generator draws toward, and how hard.",
    "material properties": "MARBLE's material directions: how metallic, rough, transparent or glowing it reads.",
}

# How the panel lays the sections out. `place` orders them: the mask right under the inputs, then
# the method's own settings (any section not named here), then the optional add-ons, and the
# output last, beside the render button. `folded` closes a section until it is opened; its heading
# still says what in it is set, so nothing changes the picture unseen.
LAYOUT = {
    _WHERE: dict(place="where"),
    _SECOND: dict(place="where", folded=True),
    _THIRD: dict(place="where", folded=True),
    _FADE: dict(place="where", folded=True),
    _EDGES: dict(place="where", folded=True),
    _SURFACE: dict(place="addon", folded=True),
    "words & LoRA": dict(place="addon", folded=True),
    "finishing pass": dict(place="addon", folded=True),
    _REBUILD: dict(place="addon", folded=True),
    OUT: dict(place="output"),
}

MAPPING_DOCS = {
    "bands": "the picture's broad shapes and light, with the texture's fine grain laid in (a frequency split)",
    "displace": "the texture's pixels pushed along the picture's slopes and shaded by its light: the picture "
                "appears as a relief in the material",
    "blend": "the two images' edges merged: at each pixel the stronger edge wins (Poisson blending)",
    "chroma": "the picture's light and shade, recoloured with the texture's colours",
    "grain": "the texture's light and shade, recoloured with the picture's colours",
    "mask": "the texture rises through the picture from its darkest tones up to a waterline",
    "relief": "the texture lit as if the picture were carved into it; turn the light direction to move the sun",
    "carve": "the picture cut into the texture as a true relief: its real surfaces, read by a geometry model "
             "(MoGe-2), light the texture and shadow its hollows, so a face is carved in bark or stone",
    "veins": "the texture's cracks and lines found and engraved into the picture",
    "flow": "the texture combed along the picture's contours, like brushstrokes that keep the grain",
    "quilt": "the picture rebuilt from real pieces of the texture, stitched along hidden seams (image quilting)",
    "analogy": "each pixel of the picture replaced by its best-matching spot in the texture: quilting without "
               "blocks or seams (PatchMatch)",
    "guided": "the texture's tones poured into the picture's shapes and held at its edges (guided filter)",
    "relit": "the material's own colours (its albedo) under the picture's own light and shine (its intrinsic "
             "shading and residual): the picture's light, the material's paint",
    "synth": "the picture rebuilt from the texture's own patches on the GPU, each placed where its light matches "
             "the picture's at two scales (ebsynth): nothing but the material, arranged as the picture",
}


def scaled(params, px_keys, long_side):
    """Pixel knobs carried from the REF frame to this one."""
    k = long_side / REF
    p = dict(params)
    for key in px_keys:
        if key == "patch":
            p[key] = max(3, int(round(p[key] * k)) | 1)  # an odd side, at least 3
        elif key == "block":
            p[key] = max(8, int(round(p[key] * k)))
        else:
            p[key] = p[key] * k
    return p


def export_long(engine, path):
    return min(max(engine.original_size(path)), EXPORT_CAP)


CORRESPONDENCE = {"quilt", "analogy", "synth"}  # solved at the working size, carried up by remap


def _mapping_runner(key, fn, px_keys):
    def run(engine, inputs, params, watch=None):
        from composite import img, remap

        cpath = engine.content_path(inputs["content"])
        tpath = engine.texture_path(inputs["texture"])
        work = int(params["size"])
        out_long = export_long(engine, cpath) if params.get("_export") else work
        masked = params.get("mask", "none") != "none"
        shaded = params.get("shading", 0) > 0
        # Live frames round to multiples of 8, which keeps every FFT length smooth; an export keeps
        # the picture's exact size.
        W, H = engine.frame(cpath, out_long, 1 if params.get("_export") else 8)
        sp = scaled(params, SURFACE_PX, out_long)
        size = float(params.get("tex_scale", 1.0))

        def sized(t, w, h):  # the texture at the material size asked for
            return t if size == 1.0 else tiling.resize(t, size, key=f"{tpath}@{w}x{h}")

        if key in CORRESPONDENCE and out_long > work:
            # Which texture pixel goes where is decided at the working size — the search is what
            # costs, and its answer is the same at any size. Only the reading is done full size,
            # from the full-size texture, so the export holds grain the working frame never had.
            w, h = engine.frame(cpath, work, 8)
            p = scaled(params, px_keys, work)
            my, mx = fn(engine.load_frame(cpath, w, h), sized(engine.load_frame(tpath, w, h), w, h), p, mapping=True)
            t = sized(engine.load_frame(tpath, W, H), W, H)
            out = remap(my, mx, t, H, W, p.get("melt", 0.0))
            c = engine.load_frame(cpath, W, H) if masked or shaded else None
        else:
            c, t = engine.load_frame(cpath, W, H), engine.load_frame(tpath, W, H)
            uv = None
            if params.get("wrap", 0) > 0:
                # The texture is laid over the picture's relief before the mode reads it, so every
                # mode — and the mask's texture sources — sees the same curved material.
                by_geo = params.get("forms", "geometry") == "geometry"
                uv = surface.flatten(None if by_geo else engine.depth_arr(inputs["content"], W, H), c, sp["wrap"],
                                     sp["roundness"], sp["recede"], sp["folds"], sp["shade_grain"],
                                     key=f"{cpath}@{W}x{H}",
                                     geo=engine.geometry_arr(inputs["content"], W, H) if by_geo else None)

            def lay(a, tag=""):
                """Anything aligned with the texture, laid the way the texture is."""
                k = f"{tpath}@{W}x{H}{tag}"
                if uv is not None:
                    return surface.wrap(a, uv, size, key=k)
                return a if size == 1.0 else tiling.resize(a, size, key=k)

            t = lay(t)
            if key == "relit":
                out = relit(engine, inputs, tpath, lay, scaled(params, px_keys, out_long), W, H)
            elif key == "carve":
                out = carved(engine, inputs, t, scaled(params, px_keys, out_long), W, H)
            else:
                out = fn(c, t, scaled(params, px_keys, out_long))
        if shaded:
            out = shade(engine, inputs, sp, out, c)
        if masked:
            # The mask is built at the frame being drawn, its sizes scaled like every pixel knob,
            # so a drag draft, a release and an export get the same mask.
            out = masks.apply(scaled(params, MASK_PX, out_long), c, t, out,
                              lambda: engine.depth_arr(inputs["content"], W, H),
                              lambda kind, arg: engine.segment(inputs["content"], kind, arg, W, H))
        return img(out)

    return run


def relit(engine, inputs, tpath, lay, p, W, H):
    """The material's albedo under the picture's shading and shine: both photographs split into
    colour and light (intrinsics.py), and recombined across. In linear light."""
    pic = engine.intrinsics_arr(inputs["content"], W, H)
    mat = engine.intrinsics_of(tpath, W, H)
    albedo = lay(mat["albedo"], ":albedo")
    if p["keep_colour"] > 0:
        albedo = (1.0 - p["keep_colour"]) * albedo + p["keep_colour"] * pic["albedo"]
    light = pic["shading"]
    if p["light_hue"] < 1:
        grey = (light @ cx.LUM)[..., None]
        light = grey + p["light_hue"] * (light - grey)
    out = albedo * light * pic["scale"]
    if p["own_relief"] > 0:
        # The material's own light, kept only as its relief: its shading's fine part, about 1.
        own = np.log(lay(mat["shading"], ":shading") @ cx.LUM + 1e-3)
        own = own - masks.blur(own.astype(np.float32), max(W, H) / 40.0)
        out = out * np.exp(np.clip(p["own_relief"] * own, -3.0, 3.0))[..., None]
    out = out + p["glints"] * pic["residual"]
    return np.clip(out, 0.0, 1.0) ** (1.0 / 2.2)


def focus_of(engine, inputs, what, W, H):
    """What of the picture a carving or an illusion keeps, as a soft 0..1 map; None for all of it."""
    if what == "whole picture":
        return None
    return masks.blur(engine.segment(inputs["content"], what, None, W, H).astype(np.float32), max(W, H) / 200.0)


def carved(engine, inputs, t, p, W, H):
    """carve.py: the texture lit by the picture's real surfaces (MoGe-2, read once per picture)."""
    import carve

    return carve.carve(t, engine.geometry_arr(inputs["content"], W, H), p["angle"], p["elevation"], p["depth"],
                       p["ambient"], p["hollow"], p["push"], p["reach"],
                       focus_of(engine, inputs, p.get("carve_on", "subject"), W, H))


def material_of(name):
    """A texture's name as the material space's input: the place its photograph sits in the space."""
    if name.startswith("u:"):
        return {"upmat": name[2:]}
    if name.startswith("i:"):
        return {"photo": int(name[2:])}
    return {"texfile": name}


def _illusion_runner():
    """A picture hidden in a texture (explore.py draws; carve.py makes what the ControlNet is shown)."""
    def run(engine, inputs, params, watch=None):
        import carve

        if params.get("_export"):
            params = {**params, "quality": "full", "size": out_sizes(engine.args.preset)[-1]}
        cpath = engine.content_path(inputs["content"])
        W, H = engine.frame(cpath, int(params.get("_size") or params["size"]), 8)

        def draw():
            c = engine.load_frame(cpath, W, H)
            focus = focus_of(engine, inputs, params.get("hide", "subject"), W, H)
            geo = engine.geometry_arr(inputs["content"], W, H) if params.get("signal") == "form" else None
            hint = carve.control(c, geo, params.get("signal", "form"), focus, params["soften"] * max(W, H) / 768.0)
            if params.get("show_hint"):
                return hint
            source = cx.img(engine.load_frame(engine.texture_path(inputs["texture"]), W, H))
            engine.make_room()
            q = engine.query({"content": inputs["content"], "materialA": material_of(inputs["texture"])},
                             params, "illusion")
            return engine.draw_illusion(q, source, hint, params["illusion"], params["let_go"])

        out = drawn("illusion", inputs, params, draw)
        if params.get("_export"):
            out = cx.cover(out, *engine.frame(cpath, export_long(engine, cpath)))
        return out

    return run


MIXED = ("materialA", "materialB", "materialC", "materialD")


def placed(engine, spec):
    """One material's place in the space, worked out once: a preset's is a mean over every crop its
    terms name, and redone per frame it cost 850 ms of a frame the GAN draws in tens."""
    memo = engine.__dict__.setdefault("_placed", {})
    key = json.dumps(spec, sort_keys=True)
    if key not in memo:
        memo[key] = engine.material(engine.query({"materialA": spec}, {}, "space"))
    return memo[key]


def mixed(engine, inputs, params):
    """The materials chosen, weighted, as one place in the space at real-crop length."""
    vecs, weights = [], []
    for key, w in zip(MIXED, ("wA", "wB", "wC", "wD")):
        if inputs.get(key) and float(params.get(w, 0)) > 0:
            vecs.append(placed(engine, inputs[key]))
            weights.append(float(params[w]))
    if not vecs:
        raise RuntimeError("choose at least one material, with a weight above 0")
    v = sum(w * x for w, x in zip(weights, vecs)) / sum(weights)
    return (v * (engine.norm / np.linalg.norm(v))).astype(np.float32)


def _mix_runner():
    """Several places in the space, weighted, drawn as one material (explore.render_vec)."""
    def run(engine, inputs, params, watch=None):
        v = mixed(engine, inputs, params)
        q = engine.query({"materialA": inputs["materialA"]}, params, "space")
        return engine.render_vec(v, q, wrap=bool(params.get("tileable")))

    return run


def _enlarge_runner(root):
    """A picture made 2K or 4K (hd.py): rebuilt from the archive photographs nearest it, or
    redrawn tile by tile by SDXL; tileable if asked."""
    def run(engine, inputs, params, watch=None):
        import pathlib

        import hd
        from PIL import Image

        cpath = engine.content_path(inputs["content"])
        pic = Image.open(cpath).convert("RGB")
        vec = engine.file_vec(cpath)  # the picture's own place in the space
        long_side = int(params["size"])
        tileable = bool(params.get("tileable"))
        if params["way"] == "rebuild":
            if inputs.get("texture"):
                paths = [engine.texture_path(inputs["texture"])]
            else:
                paths = nearest_photos(engine, vec, int(params["sources"]))
                if not all(pathlib.Path(x).is_file() for x in paths):
                    raise RuntimeError("the archive photographs are not reachable (is the photo drive connected?)")
            return hd.rebuild(engine, pic, paths, long_side, params["likeness"], params["rb_patch"], tileable)
        engine.make_room()
        redraw = engine.__dict__.get("_redraw")
        if redraw is None:
            redraw = engine._redraw = hd.Redraw(engine.transfer())
        return redraw(pic, vec, long_side, params["detail"], params["hold"], params["scale"], tileable,
                      int(params["seed"]), params.get("prompt", ""), watch=watch)

    return run


def shade(engine, inputs, sp, out, c):
    """surface.shade with the light read as the settings ask: from the picture's brightness, or
    from its intrinsic shading (read once per picture, resized to this frame)."""
    h, w = c.shape[:2]
    iid = engine.intrinsics_arr(inputs["content"], w, h) if sp.get("light", "intrinsic") == "intrinsic" else None
    if iid is not None and sp.get("relief_light", 0) > 0:
        # The relief is lit before the picture's shading is laid on: the shading replaces the broad
        # light, the relief adds the fine light the picture's sun would cast on the material.
        out = surface.relight(out, engine.geometry_arr(inputs["content"], w, h), iid, sp["relief_light"],
                              sp["shade_grain"])
    return surface.shade(out, c, sp["shading"], sp["shade_grain"], sp["shade_tone"], iid=iid, shine=sp.get("shine", 0.0))


_drawn, _drawn_lock = collections.OrderedDict(), threading.Lock()


POST_KEYS = MASK_KEYS | SURFACE_KEYS  # settings applied to a drawing after it is drawn


def drawn(key, inputs, params, draw):
    """A method's own drawing, before its shading and mask, kept for the last few settings: a change
    to those alone lays the same drawing again instead of redrawing it — seconds for diffusion,
    minutes for stylize — and a stylize export reuses the result already on screen. Drawings are
    kept at their working size; only what is made from them is export-sized."""
    k = json.dumps([key, inputs, {n: v for n, v in params.items() if n not in POST_KEYS and n != "_export"}],
                   sort_keys=True, default=str)
    with _drawn_lock:
        if k in _drawn:
            _drawn.move_to_end(k)
            return _drawn[k]
    out = draw()
    with _drawn_lock:
        _drawn[k] = out
        while len(_drawn) > 6:
            _drawn.popitem(last=False)
    return out


def finish(engine, inputs, params, out, texture=None):
    """A drawn result relit by its own picture's shading, then laid into the picture through the
    mask, at the result's size: the material where the mask is, the untouched picture elsewhere (at
    an export, the picture's own full-resolution pixels)."""
    shaded = params.get("shading", 0) > 0
    if not inputs.get("content") or (params.get("mask", "none") == "none" and not shaded):
        return out
    W, H = out.size
    c = engine.load_frame(engine.content_path(inputs["content"]), W, H)
    got = cx.arr(out)
    if shaded:
        sp = scaled(params, SURFACE_PX, max(W, H))
        got = shade(engine, inputs, sp, got, c)
    if params.get("mask", "none") == "none":
        return cx.img(got)
    t = engine.load_frame(engine.texture_path(texture), W, H) if texture else None
    got = masks.apply(scaled(params, MASK_PX, max(W, H)), c, t, got,
                      lambda: engine.depth_arr(inputs["content"], W, H),
                      lambda kind, arg: engine.segment(inputs["content"], kind, arg, W, H))
    return cx.img(got)


def _stylize_runner(loss):
    def run(engine, inputs, params, watch=None):
        from composite import fit
        from stylize import image, target_of, tensor, transfer

        def draw():
            vgg = engine.vgg()
            size = int(params["size"])
            cpath = engine.content_path(inputs["content"])
            spath = engine.texture_path(inputs["texture"])
            content = tensor(engine.load_image(cpath, *engine.frame(cpath, size)), engine.device)
            style = tensor(engine.load_image(spath, *engine.frame(spath, size)), engine.device)
            tradeoff = 10 ** params["tradeoff"]
            if loss == "patch":
                out = transfer(vgg, content, style, "patch", tradeoff, int(params["steps"]),
                               swap_at=int(params["swap_at"]), patch=int(params["patch_size"]),
                               stages=int(params["stages"]), palette_w=10 ** params["palette"],
                               init=params["init"],
                               watch=(lambda stage, im: watch(f"stage {stage + 1}", im)) if watch else None)
            else:
                target = target_of(vgg, style, loss)
                out = transfer(vgg, content, target, loss, tradeoff, int(params["steps"]))
            return image(out)

        cpath = engine.content_path(inputs["content"])
        out = drawn("stylize " + loss, inputs, params, draw)
        if params.get("_export"):
            # An optimisation over 24 megapixels is not on offer; the working result is scaled up
            # to the picture's own size and shape.
            from PIL import Image

            out = out.resize(engine.frame(cpath, export_long(engine, cpath)), Image.LANCZOS)
        return finish(engine, inputs, params, out, inputs["texture"])

    return run


def _diffusion_runner(kind, silent_material=False):
    def run(engine, inputs, params, watch=None):
        if silent_material:
            params = {**params, "scale": 0.0}  # the words alone: the adapter is present but mute
        if params.get("_export"):
            # An export is drawn at the largest output size, at full quality, then scaled up.
            params = {**params, "quality": "full", "size": out_sizes(engine.args.preset)[-1]}
        q = engine.query(inputs, params, kind)
        draw = {"space": engine.draw_space, "walk": engine.draw_walk}.get(kind, engine.draw_transform)
        out = drawn(kind, inputs, params, lambda: draw(q))
        if params.get("_export") and inputs.get("content"):
            # Diffusion is drawn at its own native size, in the picture's shape, then scaled up.
            from PIL import Image

            cpath = engine.content_path(inputs["content"])
            out = cx.cover(out, *engine.frame(cpath, export_long(engine, cpath)))  # crop, never stretch
        return finish(engine, inputs, params, out)

    return run


def _transfer_runner():
    """ZeST on SDXL (zest.py): the mask is built first and drawn INTO, not laid on after."""
    def run(engine, inputs, params, watch=None):
        import zest

        if params.get("_export"):
            params = {**params, "quality": "full", "size": TRANSFER_SIZES[-1]}
        cpath = engine.content_path(inputs["content"])
        W, H = engine.frame(cpath, int(params["size"]), 8)
        c = engine.load_frame(cpath, W, H)
        # Drawn into the whole region; a gradient fades the drawing in afterwards (finish).
        m = masks.build(scaled(params, MASK_PX, max(W, H)), c, None,
                        lambda: engine.depth_arr(inputs["content"], W, H),
                        lambda kind, arg: engine.segment(inputs["content"], kind, arg, W, H), faded=False)
        if m is None:
            m = np.ones((H, W), np.float32)
        vec = engine.material(engine.query(inputs, params, "transform"))
        full = params.get("quality") == "full"

        def draw():
            engine.make_room()  # SDXL and the editor's image model do not fit on the card together
            return engine.transfer()(
                engine.load_image(cpath, W, H), m, zest.disparity_image(engine.geometry_arr(inputs["content"], W, H)), vec,
                properties={k: params.get(k, 0.0) for k in zest.PROPERTIES}, prompt=params.get("prompt", ""),
                steps=30 if full else 12, guidance=params.get("guidance", 5.0), strength=params["strength"],
                structure=params["structure"],
                scale=params["scale"], block=params["inject"], grey=params["grey"], seed=params["seed"],
                look=None if params.get("xl_lora", "none") == "none" else engine_root(engine) / params["xl_lora"],
                look_w=params.get("xl_lora_w", 1.0))

        # The mask is drawn into, so it belongs to the drawing's identity here; the rebuild and the
        # gradient are not, so changing only how it is rebuilt or faded keeps the drawing.
        mask_sig = json.dumps({k: params.get(k) for k in sorted(MASK_KEYS - FADE_KEYS)}, sort_keys=True, default=str)
        out = drawn("transfer " + mask_sig, {k: v for k, v in inputs.items() if k != "texture"},
                    {k: v for k, v in params.items() if k not in REBUILD_KEYS}, draw)
        if params.get("rebuild", "off") != "off":
            out = _rebuilt(engine, inputs, params, out, m, vec)
        elif params.get("_export"):
            out = cx.cover(out, *engine.frame(cpath, export_long(engine, cpath)))
        return finish(engine, inputs, params, out)

    return run


TRANSFER_SIZES = [768, 1024, 1280]


def engine_root(engine):
    import pathlib

    return pathlib.Path(engine.args.space).parent


def xl_loras(root):
    """Every numbered checkpoint of every SDXL LoRA run (lora.py on RealVisXL), latest run first."""
    import pathlib

    if not root:
        return []
    found = sorted((str(p.parent.relative_to(root)).replace("\\", "/")
                    for p in pathlib.Path(root).glob("sdxl-lora*/step-*/pytorch_lora_weights.safetensors")),
                   key=lambda x: (x.split("/")[0], -int(x.rsplit("-", 1)[-1])))
    finals = [str(p.parent.relative_to(root)).replace("\\", "/")
              for p in pathlib.Path(root).glob("sdxl-lora*/pytorch_lora_weights.safetensors")]
    return finals + found
REBUILD_KEYS = {"rebuild", "sources", "likeness", "rb_patch"}


def nearest_photos(engine, vec, k):
    """The archive photographs nearest a place in the space, by their crops' mean."""
    means = engine.__dict__.get("_photo_means")
    if means is None:
        emb, owner = engine.space["emb"], engine.space["owner"]
        means = np.zeros((len(engine.files), emb.shape[1]), np.float32)
        np.add.at(means, owner, emb)
        means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
        engine._photo_means = means
    order = np.argsort(-(means @ (np.asarray(vec, np.float32) / np.linalg.norm(vec))))
    return [engine.files[i] for i in order[:k]]


def _rebuilt(engine, inputs, params, out, m, vec):
    """The drawing remade from real photographs' patches (ebsynth.rebuild), inside the mask's
    box: the archive photographs nearest the material, or the texture photograph chosen. At an
    export the arrangement is solved at the drawing's size and the full-size photographs are read
    through it, so every exported pixel is one of theirs."""
    import pathlib

    import ebsynth
    from composite import remap

    cpath = engine.content_path(inputs["content"])
    d = cx.arr(out)
    H, W = d.shape[:2]
    if params["rebuild"] == "archive":
        paths = nearest_photos(engine, vec, int(params["sources"]))
        if not all(pathlib.Path(x).is_file() for x in paths):
            raise RuntimeError("the archive photographs are not reachable (is the photo drive connected?)")
    else:
        if not inputs.get("texture"):
            raise RuntimeError("choose a texture photograph to rebuild from")
        paths = [engine.texture_path(inputs["texture"])]
    ys, xs = np.nonzero(m > 0.05)
    if ys.size == 0:
        return out
    pad = int(0.03 * max(H, W))
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)

    def sources(height):
        got = []
        for path in paths:
            ow, oh = engine.original_size(path)
            got.append(engine.load_frame(path, max(16, round(height * ow / oh)), height))
        return got

    knobs = dict(likeness=float(params["likeness"]), patch=int(params["rb_patch"]))
    if not params.get("_export"):
        d = d.copy()
        d[y0:y1, x0:x1] = ebsynth.rebuild(d[y0:y1, x0:x1], sources(y1 - y0), **knobs)
        return cx.img(d)
    (my, mx), style = ebsynth.rebuild(d[y0:y1, x0:x1], sources(y1 - y0), mapping=True, **knobs)
    Wf, Hf = engine.frame(cpath, export_long(engine, cpath))
    k = Wf / W
    big = cx.arr(cx.cover(out, Wf, Hf))
    full = sources(round((y1 - y0) * k))
    hs = max(x.shape[0] for x in full)
    style_full = np.concatenate([np.pad(x, ((0, hs - x.shape[0]), (0, 0), (0, 0)), mode="reflect") for x in full], 1)
    Y0, X0 = round(y0 * k), round(x0 * k)
    rh, rw = min(round((y1 - y0) * k), Hf - Y0), min(round((x1 - x0) * k), Wf - X0)
    big[Y0:Y0 + rh, X0:X0 + rw] = remap(my, mx, style_full, rh, rw)
    return cx.img(big)


def _edit_runner():
    """FLUX.2 klein (editor.py): the masked part redrawn with a material swatch in view."""
    def run(engine, inputs, params, watch=None):
        if params.get("_export"):
            params = {**params, "quality": "full", "size": TRANSFER_SIZES[-1]}
        cpath = engine.content_path(inputs["content"])
        W, H = engine.frame(cpath, int(params["size"]), 16)
        c = engine.load_frame(cpath, W, H)
        sp = scaled(params, MASK_PX, max(W, H))

        def depth():
            return engine.depth_arr(inputs["content"], W, H)

        def seg(kind, arg):
            return engine.segment(inputs["content"], kind, arg, W, H)

        # The editor may redraw the whole region. A gradient over it either grades the redraw
        # itself — how much of each place is redrawn — or fades the redrawn result in (finish).
        m = masks.build(sp, c, None, depth, seg, faded=False)
        if m is None:
            m = np.ones((H, W), np.float32)
        graded = params.get("fade", "off") != "off" and params.get("fade_by") == "redraw"
        ramp = masks.build(sp, c, None, depth, seg) if graded else None

        def swatch():
            if params["swatch"] == "texture":
                if not inputs.get("texture"):
                    raise RuntimeError("choose a texture photograph for the swatch")
                tpath = engine.texture_path(inputs["texture"])
                return engine.load_image(tpath, *engine.frame(tpath, 512))  # as the space's swatches are
            # The space draws the material first, concept axes and all; the editor is shown that.
            q = engine.query({"materialA": inputs.get("materialA")},
                             {"quality": "full", "size": 512, "aspect": "1:1", "seed": params["seed"], "scale": 1.0},
                             "space")
            return engine.draw_space(q)

        def named():
            """What the first mask layer selects, in words the editor can act on."""
            import editor

            src = params.get("mask", "none")
            what = {"object (words)": params.get("words", "").strip() or None,
                    "body part": "the " + params.get("part", "clothing"),
                    "person": "the person's clothes", "subject": "the main subject",
                    "object (click)": "the selected object"}.get(src)
            return editor.instruction(what)

        def draw():
            return engine.editor()(engine.load_image(cpath, W, H), m, swatch(), prompt=params.get("prompt", "") or named(),
                                   steps=6 if params.get("quality") == "full" else 4, strength=params["strength"],
                                   seed=params["seed"], ramp=ramp, focus=params.get("focus") != "whole picture")

        mask_sig = json.dumps({k: params.get(k) for k in sorted(MASK_KEYS if graded else MASK_KEYS - FADE_KEYS)},
                              sort_keys=True, default=str)
        out = drawn("edit " + mask_sig, inputs, params, draw)
        if params.get("_export"):
            out = cx.cover(out, *engine.frame(cpath, export_long(engine, cpath)))
        # A graded redraw is laid in through the whole region: the gradient is in the drawing
        # already. Shown as a mask, it keeps the gradient, which is what is being tuned.
        if graded and not params.get("show_mask"):
            params = {**params, "fade": "off"}
        return finish(engine, inputs, params, out)

    return run


def _grow_runner(root):
    """An archive NCA grown at the frame's size, through the mask along a spreading front."""
    def run(engine, inputs, params, watch=None):
        import nca

        cpath = engine.content_path(inputs["content"])
        work = int(params["size"])
        out_long = export_long(engine, cpath) if params.get("_export") else work
        W, H = engine.frame(cpath, out_long, 1 if params.get("_export") else 8)
        c = engine.load_frame(cpath, W, H)
        # The rule is grown at the working size and enlarged for an export: its grain is the
        # working frame's, as the other modes' would be.
        w, h = engine.frame(cpath, work, 8)
        t = nca.grow(root, params["nca"], w, h, params["steps"], params["reach"] * work / REF, params["seed"],
                     engine.device)
        if (w, h) != (W, H):
            t = np.stack([cx._resize(t[..., i], W, H, cx.Image.BICUBIC) for i in range(3)], -1).clip(0.0, 1.0)
        m = masks.build(scaled(params, MASK_PX, out_long), c, t,
                        lambda: engine.depth_arr(inputs["content"], W, H),
                        lambda kind, arg: engine.segment(inputs["content"], kind, arg, W, H))
        if m is None:
            m = np.ones((H, W), np.float32)
        m = nca.front(m, params["spread"], params["ragged"], params["seed"])
        sp = scaled(params, SURFACE_PX, out_long)
        if params.get("shading", 0) > 0:
            t = shade(engine, inputs, sp, t, c)
        if params.get("show_mask"):
            grey = (c @ cx.LUM)[..., None] * 0.55
            return cx.img(grey * (1.0 - m[..., None]) + (grey * 0.45 + masks.GOLD * 0.75) * m[..., None])
        out = c * (1.0 - m[..., None]) + t * m[..., None]
        if params.get("harmonise", 0) > 0:
            import harmonise

            out = harmonise.harmonise(out, m, params["harmonise"], engine.device)
        return cx.img(out)

    return run


def _gan_runner(spec):
    def run(engine, inputs, params, watch=None):
        # A GAN trained on the space (fastgan.py --look) is given the mix itself, as `mix` would draw it.
        return engine.draw_gan(spec, params, mixed(engine, inputs, params) if spec.get("looks") else None)

    return run


COMPOSED = ("texture", "textureB", "textureC")  # the materials a composition lays, first to last
ORDERS = {"light": "the first material in the picture's darkest parts, the last in its lightest",
          "depth": "the first material on what is nearest, the last on what is furthest",
          "across": "the first material on one side of the frame, the last on the other"}
_tokens, _tokens_lock = collections.OrderedDict(), threading.Lock()


def tokens_of(key, rae, load, grid):
    """A picture's token grid, kept for the last few dozen asked: encoding is 10 ms, but the photograph
    behind it can be a 6000 px JPEG off the photo drive."""
    k = json.dumps([key, grid], default=str)
    with _tokens_lock:
        if k in _tokens:
            _tokens.move_to_end(k)
            return _tokens[k]
    z = rae.encode(load(), grid)
    with _tokens_lock:
        _tokens[k] = z
        while len(_tokens) > 32:
            _tokens.popitem(last=False)
    return z


def crop_of(engine, name, W, H, zoom, seed):
    """A W x H piece of a material photograph, from a copy of it `zoom` times the frame's size: the
    middle, or somewhere else for another seed."""
    big = engine.load_frame(engine.texture_path(name), max(W, round(W * zoom)), max(H, round(H * zoom)))
    rng = np.random.default_rng(int(seed))
    x = (big.shape[1] - W) // 2 if not seed else int(rng.integers(0, big.shape[1] - W + 1))
    y = (big.shape[0] - H) // 2 if not seed else int(rng.integers(0, big.shape[0] - H + 1))
    return big[y:y + H, x:x + W]


def lineup(f, count, balance, seam):
    """`count` weights over the frame from a field in 0..1: the first material where the field is
    low, the last where it is high. `balance` moves the boundaries, `seam` widens them from a clean
    line (0) to one long morph across each material's share (1)."""
    w = np.zeros((count,) + f.shape, np.float32)
    if count == 1:
        w[0] = 1
        return w
    pos = np.clip(f, 0, 1) ** (2.0 ** ((0.5 - balance) * 4)) * (count - 1)
    i = np.minimum(np.floor(pos), count - 2).astype(int)
    s = masks.smoothstep(np.clip((pos - i - 0.5) / max(seam, 1e-3) + 0.5, 0, 1))
    for k in range(count):
        w[k] = np.where(i == k, 1 - s, 0) + np.where(i == k - 1, s, 0)
    return w


def order_of(engine, inputs, params, c, inside):
    """The field the materials are laid along, ranked over the selection so the balance halfway gives
    each the same share of it."""
    H, W = c.shape[:2]
    how = params["order"]
    if how == "depth":
        f = 1.0 - engine.depth_arr(inputs["content"], W, H)
    elif how == "across":
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        a = float(params["side"])
        f = (y - H / 2) * np.cos(a) - (x - W / 2) * np.sin(a)
    else:
        f = masks.blur(c @ cx.LUM, max(W, H) / 24)  # two tokens: regions, not speckle
    ref = f[inside] if inside is not None and inside.sum() > 64 else f.ravel()
    if ref.size > 20_000:
        ref = ref[np.random.default_rng(0).choice(ref.size, 20_000, replace=False)]
    ref = np.sort(ref)
    return np.interp(f, ref, np.linspace(0.0, 1.0, ref.size)).astype(np.float32)


class TokenFrame:
    """What a token method works in: the decoder, the picture's frame at the working size (16 px a
    token, the picture's own shape), the picture itself, and the lab's mask for it."""

    def __init__(self, engine, inputs, params, files):
        self.engine, self.inputs, self.params, self.files = engine, inputs, params, files
        self.rae = engine.rae(files["rae"])
        self.cpath = engine.content_path(inputs["content"])
        self.rows, self.cols = self.rae.grid(*engine.frame(self.cpath, 1024), int(params["long"]) // 16)
        self.W, self.H = self.cols * 16, self.rows * 16
        self.c = engine.load_frame(self.cpath, self.W, self.H)
        self.sp = scaled(params, MASK_PX, max(self.W, self.H))

    def crop(self, name):
        return crop_of(self.engine, name, self.W, self.H, float(self.params["zoom"]), self.params["seed"])

    def depth(self):
        return self.engine.depth_arr(self.inputs["content"], self.W, self.H)

    def seg(self, kind, arg):
        return self.engine.segment(self.inputs["content"], kind, arg, self.W, self.H)

    def shown_mask(self, texture):
        """The lab's gold mask view, when it is asked for; else None."""
        if self.sp.get("show_mask") and self.sp.get("mask", "none") != "none":
            return cx.img(masks.apply(self.sp, self.c, texture(), self.c, self.depth, self.seg))
        return None

    def mask(self, texture):
        textured = any(self.sp.get("mask" + t) in masks.TEXTURE_SOURCES for t in masks.LAYERS)
        return masks.build(self.sp, self.c, texture() if textured else None, self.depth, self.seg)

    def pool(self, a):
        """A field in the frame as one number per token: the share of its 16 x 16 pixels, softened by
        about a token, so that a hard edge is a seam the decoder draws rather than a step it draws as a
        row of squares."""
        return masks.blur(a.reshape(self.rows, 16, self.cols, 16).mean((1, 3)), 0.7)

    def tokens(self, name=None):
        """The picture's token grid, or a material's (its piece at this zoom and crop), [rows*cols, 768]."""
        grid = (self.rows, self.cols)
        if name is None:
            return tokens_of(["picture", str(self.cpath)], self.rae, lambda: self.c, grid)[0].float()
        return tokens_of(["material", name, self.params["zoom"], self.params["seed"]], self.rae,
                         lambda: self.crop(name), grid)[0].float()

    def drawn(self, z, share):
        """The tokens decoded; enlarged four times on release and in exports; and laid over the picture's
        own pixels (at four times, the original's) wherever `share`, per token, says nothing was
        changed. The decoder's drawing is kept a token past the last change, where it is the picture's
        own tokens drawn back, so the change from one to the other does not show."""
        import torch
        import torch.nn.functional as F

        out = decoded(self.engine, self.files, self.params, self.rae, z, (self.rows, self.cols))
        if share.min() < 0.98:
            h, w = out.shape[:2]
            near = F.max_pool2d(torch.from_numpy((share > 0.02).astype(np.float32))[None, None], 3, 1, 1)
            zone = F.interpolate(near, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
            zone = np.clip(masks.blur(zone, 6 * h / self.H), 0, 1)[..., None]
            own = self.c if h == self.H else self.engine.load_frame(self.cpath, w, h)
            out = zone * out + (1 - zone) * own
        return cx.img(out)


def decoded(engine, files, params, rae, z, grid):
    """Tokens drawn back to a picture, enlarged four times by Real-ESRGAN on release and in exports."""
    import rae as R

    out = rae.decode(R.renorm(z.reshape(1, grid[0] * grid[1], -1)), grid)[0]
    if params.get("_export") or (params.get("enlarge") and not params.get("_drag")):
        out = engine.enlarged(files["upscale"], out)
    return out


def column(a, like):
    """A per-token field as a column to weigh a [rows*cols, 768] grid by."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(a.reshape(-1, 1), np.float32)).to(like.device)


def _compose_runner(files):
    """Materials laid into a picture in DINOv2's token grid and drawn back by RAE's decoder (rae.py)."""
    def run(engine, inputs, params, watch=None):
        fr = TokenFrame(engine, inputs, params, files)
        names = [inputs[k] for k in COMPOSED if inputs.get(k)]
        if not names:
            raise RuntimeError("choose at least one material")
        first = lambda: fr.crop(names[0])  # noqa: E731
        shown = fr.shown_mask(first)
        if shown is not None:
            return shown
        m = fr.mask(first)
        f = order_of(engine, inputs, params, fr.c, None if m is None else m > 0.5) if len(names) > 1 else None
        weights = lineup(f if f is not None else np.zeros((fr.H, fr.W), np.float32), len(names),
                         float(params["balance"]), float(params["seam"]))
        if m is not None:
            weights = weights * m
        tok = np.stack([fr.pool(w) for w in weights])
        share = tok.sum(0)
        with engine.rae_lock:
            z_pic = fr.tokens()
            z = z_pic * column(1.0 - share, z_pic)
            for n, w in zip(names, tok):
                z = z + fr.tokens(n) * column(w, z)
            u = float(params["picture"])
            return fr.drawn((1 - u) * z + u * z_pic, share)

    return run


MOSHES = {
    "rebuild": "the picture rebuilt out of other tokens, each the closest match to the one it replaces: every "
               "token drawn out of its own neighbourhood",
    "drag": "tokens pulled along one direction, each as far as its place is bright (or near): where neighbours are "
            "pulled apart the decoder tears, where many take the same token it streaks, as a datamosh smears",
    "sort": "along the rows or the columns, every run of the picture's brightest places has its tokens put in "
            "order of brightness: pixel sorting, a token at a time",
}


def pieces(grid, b, step):
    """A [rows, cols, 768] grid cut into b x b pieces every `step` tokens (padded at the far edges by
    repetition when the step is the piece): [n, 768, b, b], and how many down and across."""
    import torch.nn.functional as F

    R, C, D = grid.shape
    g = grid.permute(2, 0, 1)
    if step > 1:
        g = F.pad(g[None], (0, -C % step, 0, -R % step), mode="replicate")[0]
    u = g.unfold(1, b, step).unfold(2, b, step)  # [D, ny, nx, b, b]
    ny, nx = u.shape[1:3]
    return u.permute(1, 2, 0, 3, 4).reshape(ny * nx, D, b, b), ny, nx


def laid(blocks, ny, nx, R, C):
    """Pieces [ny*nx, 768, b, b] laid side by side, back into a [R, C, 768] grid."""
    D, b = blocks.shape[1], blocks.shape[2]
    return blocks.reshape(ny, nx, D, b, b).permute(2, 0, 3, 1, 4).reshape(D, ny * b, nx * b)[:, :R, :C].permute(1, 2, 0)


def rebuilt(target, source, b, pick, same, gen):
    """Each b x b block of the target grid ([rows, cols, 768]) replaced by the source's block, from
    anywhere in it, whose tokens are closest (cosine, token by token); with `pick` above 1, by one of
    that many closest, at random. `same`: the source is the target, so a block never takes itself."""
    import torch
    import torch.nn.functional as F

    R, C, D = target.shape
    t, ny, nx = pieces(target, b, b)
    s, sy, sx = pieces(source, b, 1)
    sim = F.normalize(t, dim=1).flatten(1) @ F.normalize(s, dim=1).flatten(1).T
    if same:
        ty, tx = torch.meshgrid(torch.arange(ny) * b, torch.arange(nx) * b, indexing="ij")
        cy, cx_ = torch.meshgrid(torch.arange(sy), torch.arange(sx), indexing="ij")
        near = ((ty.reshape(-1, 1) - cy.reshape(1, -1)).abs() < b) & ((tx.reshape(-1, 1) - cx_.reshape(1, -1)).abs() < b)
        sim = sim.masked_fill(near.to(sim.device), -9.0)
    k = max(1, min(int(pick), sim.shape[1]))
    top = sim.topk(k, dim=1).indices
    choice = torch.randint(0, k, (len(top),), generator=gen).to(top.device)
    return laid(s[top[torch.arange(len(top), device=top.device), choice]], ny, nx, R, C)


def scattered(grid, b, reach, gen):
    """Each b x b piece of the grid swapped for the piece up to `reach` tokens away, chosen at random:
    the material's own pieces, in a new layout."""
    import torch

    R, C, D = grid.shape
    s, sy, sx = pieces(grid, b, 1)
    ny, nx = -(-R // b), -(-C // b)
    y = torch.arange(ny).repeat_interleave(nx) * b
    x = torch.arange(nx).repeat(ny) * b
    r = int(round(float(reach)))
    dy = torch.randint(-r, r + 1, (ny * nx,), generator=gen)
    dx = torch.randint(-r, r + 1, (ny * nx,), generator=gen)
    py = (y + dy).clamp(0, sy - 1)
    px = (x + dx).clamp(0, sx - 1)
    return laid(s[(py * sx + px).to(s.device)], ny, nx, R, C)


def dragged(source, field, length, angle):
    """Each token takes the one `length * field` tokens behind it, against the direction `angle`
    points (0 the top, then round clockwise): the grid smeared toward that side, by how much the field
    says at each place."""
    import torch

    R, C, D = source.shape
    y, x = torch.meshgrid(torch.arange(R, dtype=torch.float32), torch.arange(C, dtype=torch.float32), indexing="ij")
    f = torch.from_numpy(np.asarray(field, np.float32)) * float(length)
    sy = (y + f * np.cos(angle)).round().clamp(0, R - 1).long()
    sx = (x - f * np.sin(angle)).round().clamp(0, C - 1).long()
    return source[sy.to(source.device), sx.to(source.device)]


def sorted_runs(source, key, span, columns):
    """Along each row (or column), every run of places among the `span` brightest by `key`, its tokens
    put in the order of their keys, darkest first."""
    import torch

    g, k = (source.transpose(0, 1), key.T) if columns else (source, key)
    out = g.clone()
    on = masks.rank(k) > 1.0 - float(span)
    for i in range(k.shape[0]):
        j = 0
        while j < k.shape[1]:
            if not on[i, j]:
                j += 1
                continue
            e = j
            while e < k.shape[1] and on[i, e]:
                e += 1
            order = torch.from_numpy(np.argsort(k[i, j:e], kind="stable") + j).to(g.device)
            out[i, j:e] = g[i, order]
            j = e
    return out.transpose(0, 1) if columns else out


HYBRIDS = {
    "blend": "the materials mixed token by token: one surface at every point, the walk going from the first "
             "through each to the last",
    "patchwork": "patches of each material in a grown pattern, every seam between them drawn by the decoder",
    "layout": "the first material's arrangement rebuilt out of the others' tokens: its layout, their grain",
    "remix": "the first material's own pieces scattered over the frame: its grain, a new layout",
}
FRAMES = {"3:2": (3, 2), "1:1": (1, 1), "2:3": (2, 3)}


def _hybrid_runner(files):
    """New textures from the materials alone, in the token grid, drawn back by RAE's decoder."""
    def run(engine, inputs, params, watch=None):
        import torch

        rae = engine.rae(files["rae"])
        rows, cols = rae.grid(*FRAMES[params["frame"]], int(params["long"]) // 16)
        W, H = cols * 16, rows * 16
        names = [inputs[k] for k in COMPOSED if inputs.get(k)]
        if not names:
            raise RuntimeError("choose at least one material")
        zoom, seed = float(params["zoom"]), params["seed"]
        gen = torch.Generator().manual_seed(int(seed))
        way = params["pattern"]
        with engine.rae_lock:
            grids = [tokens_of(["material", n, zoom, seed], rae, lambda n=n: crop_of(engine, n, W, H, zoom, seed),
                               (rows, cols))[0].float() for n in names]
            if way in ("blend", "patchwork"):
                if way == "blend":
                    f = np.full((rows, cols), float(params["share"]), np.float32)
                    w = lineup(f, len(grids), 0.5, 1.0)
                else:
                    f = masks.rank(masks.growth(rows, cols, float(params["patch"]) / 4, int(seed)))
                    w = lineup(f, len(grids), float(params["balance"]), float(params["seam"]))
                    w = np.stack([masks.blur(x, 0.7) for x in w])
                z = sum(g * column(x, g) for g, x in zip(grids, w))
            else:
                first = grids[0].reshape(rows, cols, -1)
                if way == "layout":
                    others = [g.reshape(rows, cols, -1) for g in grids[1:]] or [first]
                    moved = rebuilt(first, torch.cat(others, 1), int(params["blocks"]), params["pick"], len(grids) == 1, gen)
                else:
                    moved = scattered(first, int(params["blocks"]), params["reach"], gen)
                h = float(params["hold"])
                z = ((1 - h) * moved + h * first).reshape(rows * cols, -1)
            return cx.img(decoded(engine, files, params, rae, z, (rows, cols)))

    return run


def _draw_runner(files):
    """New token grids drawn by the generator (tokendit.py), steered by the materials' places in the
    space, in its 16 x 16 windows over a frame of any shape, and drawn back by RAE's 256 decoder. While
    a slider drags, ten steps from the same noise: rougher, the same drawing."""
    def run(engine, inputs, params, watch=None):
        import torch

        import tokendit

        _, model, mean, std, step = engine.tokendit(files["tokendit"])
        rae = engine.rae(files["rae256"])
        names = [inputs[k] for k in COMPOSED if inputs.get(k)]
        place = None
        if names:
            # The walk between the materials, as `blend` walks their tokens: each one's place, weighted.
            ends = [placed(engine, material_of(n)) for n in names]
            w = lineup(np.full((1, 1), float(params["share"]), np.float32), len(ends), 0.5, 1.0)[:, 0, 0]
            v = sum(float(k) * e / np.linalg.norm(e) for k, e in zip(w, ends))
            place = torch.from_numpy((v / np.linalg.norm(v)).astype(np.float32))[None].to(mean.device)
        grid = rae.grid(*FRAMES[params["frame"]], int(params["long"]) // 16)
        steps = min(int(params["steps"]), 10) if params.get("_drag") else int(params["steps"])
        gen = torch.Generator(device=mean.device).manual_seed(int(params["seed"]))
        with engine.rae_lock:
            z = tokendit.sample(model, place, steps, float(params["guide"]), gen, mean, std, grid=grid)
            return cx.img(decoded(engine, files, params, rae, z, grid))

    return run


def _mosh_runner(files):
    """Tokens moved out of their neighbourhoods and drawn back by RAE's decoder: datamoshing in its grid."""
    def run(engine, inputs, params, watch=None):
        import torch

        fr = TokenFrame(engine, inputs, params, files)
        material = params["source"] == "material" and inputs.get("texture")
        source_image = (lambda: fr.crop(inputs["texture"])) if material else (lambda: fr.c)  # noqa: E731
        shown = fr.shown_mask(source_image)
        if shown is not None:
            return shown
        m = fr.mask(source_image)
        share = np.ones((fr.rows, fr.cols), np.float32) if m is None else fr.pool(m)
        lum = lambda image: image.reshape(fr.rows, 16, fr.cols, 16, 3).mean((1, 3)) @ cx.LUM  # noqa: E731
        gen = torch.Generator().manual_seed(int(params["seed"]))
        with engine.rae_lock:
            z_pic = fr.tokens()
            src = fr.tokens(inputs["texture"]) if material else z_pic
            grid = lambda z: z.reshape(fr.rows, fr.cols, -1)  # noqa: E731
            way = params["way"]
            if way == "rebuild":
                moved = rebuilt(grid(z_pic), grid(src), int(params["blocks"]), params["pick"], not material, gen)
            elif way == "drag":
                by = params["by"]
                if by == "depth":
                    field = masks.rank(engine.depth_arr(inputs["content"], fr.W, fr.H).reshape(fr.rows, 16, fr.cols, 16).mean((1, 3)))
                elif by == "noise":
                    field = masks.rank(masks.growth(fr.rows, fr.cols, 3.0, int(params["seed"])))
                else:
                    field = masks.rank(lum(fr.c))
                moved = dragged(grid(src), field, params["length"], float(params["side"]))
            else:
                # The picture's light chooses the runs and their order, whoever's tokens are sorted.
                moved = sorted_runs(grid(src), lum(fr.c), params["span"], params["along"] == "columns")
            moved = moved.reshape(fr.rows * fr.cols, -1)
            h = float(params["hold"])
            # What changed, per token: where nothing moved, the picture keeps its own pixels.
            changed = ((moved - z_pic).norm(dim=1) > 1e-3).float().cpu().numpy().reshape(fr.rows, fr.cols)
            share = share * changed * (1.0 - h)
            z = (1 - h) * moved + h * z_pic
            s = column(np.ones_like(share) if m is None else fr.pool(m), z)
            return fr.drawn(s * z + (1 - s) * z_pic, share)

    return run


AXES = 12  # the export's z is whitened PCA of w, ordered by variance; the first dozen carry the look


def lora_options(root, preset, keepers):
    """Every numbered checkpoint of every SD 1.5 LoRA run, keepers first. Other bases get none."""
    if preset != "sd15":
        return ["none"]
    import pathlib

    found = sorted(str(p.parent.relative_to(root)).replace("\\", "/")
                   for p in pathlib.Path(root).glob("sd15-lora*/step-*/pytorch_lora_weights.safetensors"))
    return ["none"] + [k for k in keepers if k in found] + [f for f in found if f not in keepers]


def build(preset="sd15", root=None, manifest=None):
    manifest = manifest or {}
    keepers = manifest.get("lora_keepers", [])
    loras = lora_options(root, preset, keepers) if root else ["none"]
    keeper = next((x for x in loras if x != "none"), "none")
    lora = lambda default: [  # noqa: E731
        P("lora", "choice", default, options=loras,
          doc="the archive-trained LoRA underneath (folder/checkpoint, best first); none for the plain model",
          group="words & LoRA", shared=True),
        P("lora_w", "float", 1.4, 0, 2, 0.05,
          label="LoRA strength", doc="how strongly the LoRA's look applies; the best range is about 1.0–1.8",
          group="words & LoRA", shared=True, when=[["lora", "!=", "none"]])]
    prompt = lambda default: P("prompt", "text", default,  # noqa: E731
                               doc="words that steer the drawing; 'in goofi style' calls up the LoRA's look",
                               group="words & LoRA")
    methods = []
    for key, (params, fn) in MAPPING.items():
        px_keys = [p["key"] for p in params if p.get("unit") == "px"]
        methods.append(dict(
            key=key, family="mapping", doc=MAPPING_DOCS[key], cost="live", export=True,
            inputs=[I("content", "content", "the picture you are transforming"),
                    I("texture", "texture", "the photograph whose material is applied to it")],
            params=params + surface_params(wrap=key not in CORRESPONDENCE, size=True) + MASK + [SIZE],
            run=_mapping_runner(key, fn, px_keys)))

    sty_common = [P("size", "choice", 448, options=[256, 384, 448, 512, 640],
                    doc="pixels on the long side the optimisation runs at; the picture keeps its shape", shared=True,
                    label="working size", group="optimisation"),
                  P("steps", "int", 220, 60, 400, 20, doc="how many optimisation steps (more = closer, slower)",
                    shared=True, group="optimisation")]
    for loss, doc in [("gram", "Gatys: Gram statistics matched across five layers"),
                      ("adain", "per-layer mean and variance matched"),
                      ("swd", "the features' whole distribution matched along fixed random directions (sliced "
                              "Wasserstein): the texture's extremes survive — the black of a crack, a glint"),
                      ("efdm", "each feature channel's whole distribution matched on its own (exact feature "
                               "distribution matching): between adain and swd"),
                      ("patch", "neural patch swap, re-made per stage; texture elements survive")]:
        pressure = {"patch": 3.3, "swd": 3.0, "efdm": 4.0}.get(loss, 4.5)
        params = list(sty_common) + [P("tradeoff", "log", pressure, 1 if loss == "swd" else 2, 6, 0.1,
                                       label="style pressure", group="optimisation",
                                       doc="how hard the texture's style is pushed, as a power of ten")]
        if loss == "patch":
            params += [
                P("swap_at", "choice", 11, options=[6, 11, 20],
                  group="patch swap", label="swap layer", choices={6: "relu2_1: fine grain", 11: "relu3_1: texture elements",
                                               20: "relu4_1: larger motifs"},
                  doc="how deep in the network the patches are swapped: deeper swaps bigger motifs"),
                P("patch_size", "choice", 3, options=[3, 5], group="patch swap", label="patch size",
                  doc="the side of each swapped neighbourhood, in feature cells"),
                P("stages", "int", 4, 1, 8, 1, group="patch swap", label="re-swap rounds",
                  doc="how many times the patches are re-matched; the main pressure knob"),
                P("palette", "log", 4, 2, 5, 0.1, group="patch swap", label="colour pressure",
                  doc="how strongly the texture's colours are carried over"),
                P("init", "choice", "content", options=["content", "mixed"], group="patch swap", label="start from",
                  choices={"content": "the picture as it is", "mixed": "the picture shaken with noise"},
                  doc="what the optimisation starts from"),
            ]
        methods.append(dict(
            key=f"stylize {loss}", family="optimise", doc=doc, cost="minutes", export=True,
            inputs=[I("content", "content", "the picture"), I("texture", "texture", "the style photograph")],
            params=params + surface_params(wrap=False) + MASK, run=_stylize_runner(loss)))

    quality = P("quality", "choice", "draft", options=["draft", "full"], group=OUT,
                choices={"draft": "a fast sketch in a few steps", "full": "the real picture, slower"},
                doc="draft while exploring, full for the real thing", shared=True)
    sizes = out_sizes(preset)
    out_size = P("size", "choice", 1024, options=sizes, label="output size", group=OUT, shared=True,
                 choices={sizes[0]: "the model's native size, fastest", sizes[-1]: "the sharpest, slowest"},
                 doc="pixels on the long side of the result; the frame keeps the picture's shape")
    scale = P("scale", "float", 1.0, 0.2, 1.4, 0.05, label="material pull", group="material",
              doc="how strongly the chosen material steers the drawing")
    aspect = P("aspect", "choice", "1:1", options=ASPECTS, doc="the frame's shape (width:height); there is no picture to take it from",
               shared=True, group=OUT)
    seed = G(SEED, OUT)
    render = [quality, out_size, seed]
    methods.append(dict(
        key="space", family="diffusion", doc="the material drawn free: a place in the space, rendered",
        cost="seconds", inputs=[I("materialA", "material", "where in the space")],
        params=[scale, prompt("")] + lora("none") + render + [aspect], run=_diffusion_runner("space")))
    methods.append(dict(
        key="walk", family="diffusion", doc="the great circle between two places, the slider between",
        cost="seconds",
        inputs=[I("materialA", "material", "end A"), I("materialB", "material", "end B")],
        params=[P("t", "float", 0.5, 0, 1, 0.01, label="position", group="walk",
                  doc="where along the walk: 0 is end A, 1 is end B"), scale,
                prompt("")] + lora("none") + render + [aspect],
        run=_diffusion_runner("walk")))
    tileable = P("tileable", "bool", False, label="tileable", group=OUT, shared=True,
                 doc="drawn so its edges wrap round: the material repeats without a seam, at any size")
    weight = lambda k, d: P(k, "float", d, 0, 1, 0.01, label="weight " + k[1], group="material",  # noqa: E731
                            doc=f"how much of material {k[1]} is in the mix (the weights are read as shares)")
    methods.append(dict(
        key="mix", family="diffusion",
        doc="several materials blended in the space, each by its weight, and drawn as one: photographs, presets, "
            "archive photographs, concept axes, in any proportion",
        cost="seconds",
        inputs=[I("materialA", "material", "the first material"),
                I("materialB", "material", "the second material", optional=True),
                I("materialC", "material", "a third material", optional=True),
                I("materialD", "material", "a fourth material", optional=True)],
        params=[weight("wA", 0.5), weight("wB", 0.5), weight("wC", 0.0), weight("wD", 0.0), scale,
                prompt("")] + lora("none") + render + [aspect, tileable],
        run=_mix_runner()))
    methods.append(dict(
        key="enlarge", family="diffusion",
        doc="a picture made 2K or 4K: rebuilt from the archive photographs nearest it (every pixel a real one of "
            "yours, read at full size), or redrawn tile by tile by SDXL (new detail, invented in the material's "
            "character); tileable if asked",
        cost="minutes",
        inputs=[I("content", "content", "the picture to enlarge: a material drawn here, or any picture"),
                I("texture", "texture", "the one photograph to rebuild from, instead of the archive's nearest",
                  optional=True, when=[["way", "==", "rebuild"]])],
        params=[P("way", "choice", "redraw", options=["redraw", "rebuild"], label="how", group="enlarge", lead=True,
                  choices={"redraw": "SDXL redraws it tile by tile, inventing detail in the material's character",
                           "rebuild": "remade from real photographs' patches, read at their full resolution"},
                  doc="how the detail of the large picture is found"),
                P("size", "choice", 4096, options=[2048, 3072, 4096], label="long side", group="enlarge",
                  doc="pixels on the long side of the result"),
                P("detail", "float", 0.4, 0.15, 0.7, 0.01, label="new detail", group="enlarge",
                  when=[["way", "==", "redraw"]],
                  doc="how much each tile is redrawn: low sharpens what is there, high invents more"),
                P("hold", "float", 0.7, 0.2, 1.2, 0.05, label="hold to the picture", group="enlarge",
                  when=[["way", "==", "redraw"]], doc="how firmly the tile ControlNet keeps each tile to the picture"),
                P("scale", "float", 0.8, 0.0, 1.4, 0.05, label="material pull", group="enlarge",
                  when=[["way", "==", "redraw"]],
                  doc="how hard the picture's own place in the space steers the redraw"),
                P("sources", "int", 3, 1, 6, 1, label="photographs", group="enlarge", when=[["way", "==", "rebuild"]],
                  doc="how many of the archive photographs nearest the picture lend their patches"),
                P("likeness", "float", 2.0, 0.2, 8, 0.1, label="follow the picture", group="enlarge",
                  when=[["way", "==", "rebuild"]],
                  doc="how closely the patches must match the picture's colours: lower lets the photographs' own "
                      "structure through"),
                P("rb_patch", "int", 5, 3, 9, 2, label="patch size", group="enlarge", when=[["way", "==", "rebuild"]],
                  doc="the side of the patches: larger keeps more of each photograph's own structure"),
                G(SEED, "enlarge"), prompt(""), tileable],
        run=_enlarge_runner(root)))
    methods.append(dict(
        key="transform", family="diffusion",
        doc="the picture's structure held (depth on sd15, edges on xl) while the material takes the flesh",
        cost="seconds", export=True,
        inputs=[I("content", "content", "the picture being carried"),
                I("materialA", "material", "the material"),
                I("materialB", "material", "the far / dark side's material", optional=True,
                  when=[["region", "!=", "off"]])],
        params=[P("structure", "float", 1.0, 0, 1.6, 0.05, label="hold structure", group="structure",
                  doc="how firmly the picture's shapes are kept"),
                P("strength", "float", 0.75, 0.2, 0.98, 0.02, label="redraw amount", group="structure",
                  doc="how much of the picture is redrawn: low keeps it, high reinvents it"),
                scale,
                P("region", "choice", "off", options=["off", "depth", "lum"], label="two materials", group="material",
                  choices={"off": "one material everywhere", "depth": "split near / far: material B on the far side",
                           "lum": "split light / dark: material B on the dark side"},
                  doc="put a second material on part of the picture"),
                P("rthresh", "float", 0.5, 0.1, 0.9, 0.02, label="split point", group="material",
                  when=[["region", "!=", "off"]], doc="where the two materials divide"),
                P("then", "text", "", label="finish with", group="finishing pass",
                  doc="a material for a second, finishing pass, e.g. preset name or terms"),
                P("then_s", "float", 0.45, 0.2, 0.7, 0.05, label="finish strength", group="finishing pass",
                  when=[["then", "set"]], doc="how much the finishing pass redraws"),
                prompt("")] + lora("none") + surface_params(wrap=False) + mask_params(texture=False) + render,
        run=_diffusion_runner("transform")))

    if preset == "sd15":
        # The ViT-H SDXL adapter reads the SD 1.5 space's own embeddings: offered only on that space.
        import zest

        props = [P(k, "float", 0.0, lo, hi, (hi - lo) / 80, label=k, group="material properties",
                   doc={"metallic": "negative: toward polished, glittering metal; positive: away from it, duller "
                                    "(MARBLE's own sign; it acts mostly beyond ±10)",
                        "roughness": "toward matte and rough (positive) or glossy and smooth (negative)",
                        "transparency": "toward glass, ice, clear resin",
                        "glow": "toward a material lit from within"}[k])
                 for k, (lo, hi) in zest.PROPERTIES.items()]
        methods.append(dict(
            key="transfer", family="diffusion",
            doc="the material drawn ON the masked things in the picture (ZeST on SDXL): redrawn inside the mask "
                "from the picture's depth, the object greyed first so the scene's light on it survives",
            cost="seconds", export=True,
            inputs=[I("content", "content", "the picture"), I("materialA", "material", "the material"),
                    I("texture", "texture", "the photograph whose patches rebuild the drawing", optional=True,
                      when=[["rebuild", "==", "texture"]])],
            params=[P("strength", "float", 0.95, 0.5, 1.0, 0.01, label="redraw amount", group="structure",
                      doc="how much of the masked part is redrawn: below 1 its own shading shows through more"),
                    P("structure", "float", 0.9, 0, 1.5, 0.05, label="hold shape", group="structure",
                      doc="how firmly the depth holds the redrawn things' shape"),
                    P("grey", "float", 1.0, 0, 1, 0.05, label="grey first", group="structure",
                      doc="how far the masked part is turned grey before the redraw: 1 drops its own colours, "
                          "keeping only its light"),
                    P("scale", "float", 1.0, 0.2, 1.6, 0.05, label="material pull", group="material",
                      doc="how strongly the chosen material steers the redraw"),
                    P("inject", "choice", "style", options=["style", "all"], label="material enters", group="material",
                      choices={"all": "every block of the network: the strongest pull",
                               "style": "the one block that carries style and material (MARBLE, InstantStyle): "
                                        "the picture's own content kept firmer"},
                      doc="where in the network the material is heard"),
                    *props,
                    P("xl_lora", "choice", "none", options=["none"] + xl_loras(root), label="archive LoRA",
                      group="words & LoRA",
                      doc="the SDXL LoRA trained on the archive (lora.py): the generator taught your materials' own "
                          "grain, not only their gist through the embedding"),
                    P("xl_lora_w", "float", 0.7, 0, 2, 0.05, label="LoRA strength", group="words & LoRA",
                      when=[["xl_lora", "!=", "none"]],
                      doc="how strongly the LoRA's look applies; about 0.7 keeps it photographic"),
                    P("guidance", "float", 5.0, 1.5, 9, 0.25, label="guidance", group="words & LoRA",
                      doc="how hard the words and material are followed; lower is more natural and less saturated "
                          "(about 3.5 with the archive LoRA: measured closest to the archive's own colour and "
                          "contrast)"),
                    P("rebuild", "choice", "off", options=["off", "archive", "texture"], label="rebuild from",
                      group=_REBUILD,
                      choices={"off": "the drawing as SDXL made it",
                               "archive": "patches of the archive photographs nearest the material: every pixel a "
                                          "real pixel of yours, arranged as the drawing (ebsynth)",
                               "texture": "patches of the texture photograph chosen above"},
                      doc="remake the drawing out of real photographs, keeping its arrangement"),
                    P("sources", "int", 3, 1, 6, 1, label="photographs", group=_REBUILD,
                      when=[["rebuild", "==", "archive"]],
                      doc="how many of the archive photographs nearest the material lend their patches"),
                    P("likeness", "float", 2.0, 0.2, 8, 0.1, label="follow the drawing", group=_REBUILD,
                      when=[["rebuild", "!=", "off"]],
                      doc="how closely the patches must match the drawing's colours: lower lets the photographs' own "
                          "structure through"),
                    P("rb_patch", "int", 5, 3, 9, 2, label="patch size", group=_REBUILD, when=[["rebuild", "!=", "off"]],
                      doc="the side of the patches: larger keeps more of each photograph's own structure"),
                    prompt(""),
                    P("quality", "choice", "draft", options=["draft", "full"], group=OUT, shared=True,
                      choices={"draft": "12 steps", "full": "30 steps"}, doc="draft while exploring, full for the real thing"),
                    P("size", "choice", 1024, options=TRANSFER_SIZES, label="output size", group=OUT,
                      doc="pixels on the long side of the result; the frame keeps the picture's shape"),
                    seed] + surface_params(wrap=False) + mask_params(texture=False),
            run=_transfer_runner()))

        methods.append(dict(
            key="illusion", family="diffusion",
            doc="a picture hidden in a texture: the texture photograph redrawn in its own material (SD 1.5 and the "
                "space's adapter), while a ControlNet made for hidden images (QR Code Monster v2) bends its light and "
                "dark toward the picture — texture up close, the picture when you squint",
            cost="seconds", export=True,
            inputs=[I("content", "content", "the picture to hide"),
                    I("texture", "texture", "the texture photograph it hides in")],
            params=[P("illusion", "float", 1.1, 0, 2.5, 0.05, label="how visible", group="illusion", lead=True,
                      doc="how hard the hidden picture bends the drawing: low hides it, high shows it plainly"),
                    P("signal", "choice", "form", options=["form", "light"], label="hide its", group="illusion",
                      choices={"form": "shape: its real surfaces (MoGe-2), lit evenly, so hair colour, clothes and "
                                       "shadows count for nothing",
                               "light": "light and dark, as photographed"},
                      doc="what of the picture is hidden"),
                    P("hide", "choice", "subject", options=["subject", "person", "whole picture"], label="hide",
                      group="illusion",
                      choices={"subject": "only the main subject: the rest is texture, free",
                               "person": "only the people: the rest is texture, free",
                               "whole picture": "the whole frame, background too"},
                      doc="what of the picture is hidden in the texture"),
                    P("soften", "float", 3, 0, 16, 0.5, unit="px", label="soften", group="illusion",
                      doc="blur the hidden picture first: softer hides it deeper, sharper keeps its features"),
                    P("let_go", "float", 0.9, 0.3, 1.0, 0.02, label="let go at", group="illusion",
                      doc="how far through the drawing the hidden picture keeps steering: earlier leaves the last, "
                          "finest steps to the texture"),
                    P("show_hint", "bool", False, label="show what is hidden", group="illusion",
                      doc="show the grey picture the ControlNet is given — light and dark it bends toward, mid-grey "
                          "left free — instead of drawing"),
                    P("strength", "float", 0.9, 0.5, 1.0, 0.01, label="redraw amount", group="structure",
                      doc="how much of the texture photograph is redrawn: lower keeps its own pattern, 1 draws "
                          "freely in its colours"),
                    P("scale", "float", 1.0, 0.0, 1.4, 0.05, label="material pull", group="material",
                      doc="how strongly the texture photograph's place in the space steers the drawing"),
                    prompt("a close-up photograph of a weathered natural surface, richly detailed")]
                   + lora("none") + render,
            run=_illusion_runner()))

        import editor

        methods.append(dict(
            key="edit", family="diffusion", instruction=editor.INSTRUCTION,
            doc="the masked part redrawn by an editor model (FLUX.2 klein) shown a swatch of the material: it puts "
                "the material on the thing's own folds and light, the way an instruction-following editor does",
            cost="seconds", export=True,
            inputs=[I("content", "content", "the picture"),
                    I("materialA", "material", "the material the swatch is drawn from", optional=True,
                      when=[["swatch", "==", "space"]]),
                    I("texture", "texture", "the photograph shown as the swatch", optional=True,
                      when=[["swatch", "==", "texture"]])],
            params=[P("swatch", "choice", "space", options=["space", "texture"], label="swatch", group="material",
                      lead=True,
                      choices={"space": "a place in the material space, concept axes and all, drawn first (SD 1.5)",
                               "texture": "a photograph as it is: one of the textures, one you upload (⬆), "
                                          "or one from the archive (▦)"},
                      doc="what the editor is shown as the material"),
                    P("prompt", "text", "", group="material", label="instruction",
                      doc="what to do, in words; empty names what the mask selects and asks for its surface "
                          "replaced by the swatch's material (" + editor.instruction("the dress") + ")"),
                    P("strength", "float", 1.0, 0.5, 1.0, 0.01, label="redraw amount", group="structure",
                      doc="how much of the masked part is redrawn: below 1 the picture's own shows through more"),
                    P("focus", "choice", "around the mask", options=["around the mask", "whole picture"],
                      label="work on", group="structure", more=True,
                      choices={"around the mask": "when the mask holds only part of the picture, just its surroundings: "
                                                  "faster, and a small thing gets more pixels",
                               "whole picture": "the whole frame every time, as the atlas was made"},
                      doc="how much of the picture the editor draws"),
                    P("quality", "choice", "draft", options=["draft", "full"], group=OUT, shared=True,
                      choices={"draft": "4 steps", "full": "6 steps"}, doc="draft while exploring, full for the real thing"),
                    P("size", "choice", 1024, options=TRANSFER_SIZES, label="output size", group=OUT,
                      doc="pixels on the long side of the result; the frame keeps the picture's shape"),
                    seed] + surface_params(wrap=False) + mask_params(texture=False) + [
                    P("fade_by", "choice", "redraw", options=["redraw", "blend"], label="the gradient sets", group=_FADE,
                      when=[["mask", "!=", "none"], ["fade", "!=", "off"]],
                      choices={"redraw": "how much of each place is redrawn: where it is low, the picture's own forms "
                                         "and colours hold and the material only settles into their grain",
                               "blend": "how much of the redrawn material shows: the full redraw faded into the picture"},
                      doc="what the gradient does to the edit")],
            run=_edit_runner()))

    # ---- the LoRA, reached by words: its own front door, and the restyle it was trained for ----
    methods.append(dict(
        key="prompt", family="words",
        doc="draw from words through the archive's LoRA — subjects it never photographed, in its eye",
        cost="seconds", inputs=[],
        params=[prompt("a close-up photograph of lichen on stone, encrusted and weathered, in goofi style"),
                ] + lora(keeper) + render + [aspect],
        run=_diffusion_runner("space", silent_material=True)))
    methods.append(dict(
        key="restyle", family="words",
        doc="carry a picture into the LoRA's look by words; structure 0 is plain img2img",
        cost="seconds", export=True, inputs=[I("content", "content", "the picture being carried")],
        params=[prompt("a close-up photograph, encrusted and weathered, mineral and wet, in goofi style"),
                ] + lora(keeper) + [
                P("strength", "float", 0.55, 0.2, 0.95, 0.02, label="redraw amount", group="structure",
                  doc="how much of the picture is redrawn: low keeps it, high reinvents it"),
                P("structure", "float", 0.0, 0, 1.6, 0.05, label="hold structure", group="structure",
                  doc="how firmly the picture's shapes are kept while the words redraw (0 = plain img2img)"),
                ] + surface_params(wrap=False) + mask_params(texture=False) + render,
        run=_diffusion_runner("transform", silent_material=True)))

    # ---- the archive's NCAs: the material grown, live, through the mask ----
    if root:
        import nca

        grown = nca.rules(root)
        # The rules were trained on the four-star set's category folders, the same folders the
        # 512 GAN's categories were named from.
        named = next((g["categories"] for g in manifest.get("gans", []) if g["key"] == "gan 512"), [])
        names = {f"cat{i}": n for i, n in enumerate(named)}
        names["parent"] = "the whole set: every category at once, the rule the others grew from"
        if grown:
            methods.append(dict(
                key="grow", family="growth",
                doc="a neural cellular automaton trained on the archive (style_ca.py) grows its material over the "
                    "picture, spreading through the mask from its edge inward",
                cost="live", export=True, inputs=[I("content", "content", "the picture the material grows over")],
                params=[P("nca", "choice", "parent" if "parent" in grown else grown[0], options=grown, label="rule",
                          group="growth", choices={k: v for k, v in names.items() if k in grown},
                          doc="which trained rule: one per category of the archive, and their parent"),
                        P("steps", "int", 64, 1, 400, 1, label="time grown", group="growth",
                          doc="how many ticks the rule has run: early the material is still assembling; it was "
                              "trained to be formed by about 64"),
                        P("spread", "float", 1.0, 0, 1, 0.01, label="spread", group="growth",
                          doc="how far the growth has reached through the mask, from its edge inward: slide it to "
                              "let the material take the picture, or draw it back"),
                        P("ragged", "float", 0.5, 0, 1, 0.02, label="ragged front", group="growth",
                          doc="how unevenly the front advances, in lobes (while spread is below 1)"),
                        P("reach", "float", 1.0, 0.5, 4, 0.05, label="grain size", group="growth",
                          doc="above 1 a coarser material from the same rule (grown smaller, enlarged)"),
                        G(SEED, "growth")]
                       + surface_params(wrap=False) + MASK + [SIZE],
                run=_grow_runner(root)))

    # ---- the GANs, the realtime flagships: one method per published model ----
    for g in manifest.get("gans", []):
        params = [P("spread", "float", 1.0, 0, 2, 0.02, group="draw",
                    doc="how far from typical the GAN may go: below 1 safer and duller"),
                  P("seed", "seed", 0, group="draw", doc="the starting point the axes move from; ⚄ picks a new one"),
                  P("seedB", "seed", 1, label="second seed", group="draw", when=[["t", ">", 0]],
                    doc="a second starting point, to walk toward"),
                  P("t", "float", 0.0, 0, 1, 0.01, label="walk", group="draw",
                    doc="0 is the seed, 1 is the second seed; move it to walk between them")]
        params += [P(f"axis {k}", "float", 0.0, -3, 3, 0.05,
                     doc=f"principal axis {k} of the model's own latent" + (" — the strongest" if k == 0 else ""),
                     group="latent axes", more=k >= 4)
                   for k in range(AXES)]
        inputs = []
        if g.get("looks"):
            # Trained on the space rather than on categories: the same four materials and weights as `mix`.
            inputs = [I("materialA", "material", "the first material"),
                      I("materialB", "material", "the second material", optional=True),
                      I("materialC", "material", "a third material", optional=True),
                      I("materialD", "material", "a fourth material", optional=True)]
            # First, so the weights sit beside the materials they weigh rather than under the axes.
            params = [weight("wA", 1.0), weight("wB", 0.0), weight("wC", 0.0), weight("wD", 0.0)] + params
        else:
            params += [P(name, "float", 1.0 if c == 0 else 0.0, 0, 1, 0.02, doc=f"category {c}: pull toward {name}",
                         group="categories")
                       for c, name in enumerate(g["categories"])]
        methods.append(dict(key=g["key"], family="realtime", doc=g["doc"], cost="live", inputs=inputs,
                            params=params, run=_gan_runner(g)))

    # ---- composing in tokens: the RAE decoder, and Real-ESRGAN to enlarge what it draws ----
    files = {k: manifest.get(k) for k in ("rae", "upscale")}
    if root and all(files.values()) and all((pathlib.Path(root) / f).is_file() for f in files.values()):
        files = {k: str(pathlib.Path(root) / f) for k, f in files.items()}
        methods.append(dict(
            key="compose", family="tokens", cost="live", export=True,
            doc="your material photographs laid into a picture in DINOv2's token grid and drawn back by RAE's "
                "decoder, which draws every seam itself: the mask says where they go, the picture's light, depth "
                "or a side says which goes where, and Real-ESRGAN enlarges the result four times",
            inputs=[I("content", "content", "the picture: its mask, light or depth places the materials, and its own "
                                            "tokens fill the rest"),
                    I("texture", "texture", "the first material: a photograph"),
                    I("textureB", "texture", "a second material", optional=True),
                    I("textureC", "texture", "a third material", optional=True)],
            params=[P("order", "choice", "light", options=list(ORDERS), label="order", group="compose", choices=ORDERS,
                      doc="with two or three materials: what decides which one goes where"),
                    P("side", "angle", 0.0, 0, 6.283, 0.05, label="side", group="compose",
                      when=[["order", "==", "across"]],
                      doc="which side the first material comes from: 0 the top, then round clockwise"),
                    P("balance", "float", 0.5, 0, 1, 0.01, label="balance", group="compose",
                      doc="moves the boundaries between the materials: lower gives the first more of the frame, "
                          "higher the last; slide it to let one grow into the other"),
                    P("seam", "float", 0.35, 0, 1, 0.01, label="seam width", group="compose",
                      doc="how gradually one material passes into the next: 0 a clean line (the decoder still "
                          "draws it), 1 one long morph across the frame"),
                    P("picture", "float", 0.0, 0, 1, 0.01, label="picture underneath", group="compose",
                      doc="how much of the picture's own tokens stay under the materials: slide it to morph from "
                          "the composition back to the picture"),
                    P("zoom", "float", 2.0, 1, 6, 0.1, label="material zoom", group="compose",
                      doc="how close into each material photograph: 1 the whole photograph across the frame, 3 a "
                          "third of it (the GANs' crop); larger is coarser grain"),
                    P("seed", "seed", 0, label="crop", group="compose",
                      doc="where in each photograph its piece is taken: 0 the middle, ⚄ somewhere else")]
                   + [x for x in MASK if x["key"] != "harmonise"]
                   + [P("long", "choice", 768, options=[512, 768, 1024], label="working size", group=OUT,
                        choices={512: "fastest, about 20 ms a frame", 768: "3:2 at 768 x 512, about 45 ms",
                                 1024: "the finest grain, about 125 ms"},
                        doc="pixels on the long side the decoder draws (16 per token); the frame keeps the "
                            "picture's shape"),
                      P("enlarge", "bool", True, label="enlarge ×4", group=OUT,
                        doc="Real-ESRGAN draws the result four times larger when you let go of a slider, and in "
                            "every export: 768 becomes 3072")],
            run=_compose_runner(files)))
        on = lambda way: [["way", "==", way]]  # noqa: E731
        methods.append(dict(
            key="mosh", family="tokens", cost="live", export=True,
            doc="datamoshing in RAE's token grid: tokens moved out of their neighbourhoods (rebuilt from the closest "
                "matches, dragged, or sorted) and drawn back by a decoder that draws each one as if it belonged",
            inputs=[I("content", "content", "the picture to mosh"),
                    I("texture", "texture", "the material photograph whose tokens are moved in", optional=True,
                      when=[["source", "==", "material"]])],
            params=[P("way", "choice", "rebuild", options=list(MOSHES), label="way", group="mosh", choices=MOSHES,
                      doc="how the tokens are moved"),
                    P("source", "choice", "material", options=["material", "picture"], label="tokens from",
                      group="mosh", choices={"material": "a material photograph's tokens, moved into the picture",
                                             "picture": "the picture's own tokens, moved within it"},
                      doc="whose tokens are moved"),
                    P("blocks", "choice", 1, options=[1, 2, 4], label="pieces", group="mosh", when=on("rebuild"),
                      choices={1: "one token: the finest checkerboard", 2: "2 x 2 tokens",
                               4: "4 x 4 tokens: pieces big enough to keep their own drawing, like a codec's macroblocks"},
                      doc="how many tokens across each piece that is moved"),
                    P("pick", "int", 1, 1, 12, 1, label="wildness", group="mosh", when=on("rebuild"),
                      doc="choose among this many closest matches at random: 1 always the closest"),
                    P("by", "choice", "light", options=["light", "depth", "noise"], label="pulled by", group="mosh",
                      when=on("drag"), choices={"light": "the brightest places pulled furthest",
                                                "depth": "the nearest places pulled furthest",
                                                "noise": "a grown noise field: the pull wanders"},
                      doc="what decides how far each token is pulled"),
                    P("length", "float", 6.0, 0, 24, 0.25, label="length", unit="tokens", group="mosh",
                      when=on("drag"), doc="how far the furthest-pulled tokens go (a token is 16 px)"),
                    P("side", "angle", 1.571, 0, 6.283, 0.05, label="toward", group="mosh", when=on("drag"),
                      doc="which way the smear runs: 0 the top, then round clockwise"),
                    P("span", "float", 0.4, 0, 1, 0.01, label="span", group="mosh", when=on("sort"),
                      doc="the share of the tokens that are sorted, the brightest first"),
                    P("along", "choice", "rows", options=["rows", "columns"], label="along", group="mosh",
                      when=on("sort"), doc="sort along the rows or along the columns"),
                    P("hold", "float", 0.0, 0, 1, 0.01, label="hold the picture", group="mosh",
                      doc="how much of the picture's own tokens stay mixed into the mosh"),
                    P("zoom", "float", 2.0, 1, 6, 0.1, label="material zoom", group="mosh",
                      when=[["source", "==", "material"]],
                      doc="how close into the material photograph: 1 the whole of it across the frame"),
                    P("seed", "seed", 0, label="seed", group="mosh",
                      doc="another number: other random choices, and another piece of the material")]
                   + [x for x in MASK if x["key"] != "harmonise"]
                   + [G(next(x for x in methods[-1]["params"] if x["key"] == "long"), OUT),
                      G(next(x for x in methods[-1]["params"] if x["key"] == "enlarge"), OUT)],
            run=_mosh_runner(files)))
        out_params = [G(next(x for x in methods[-2]["params"] if x["key"] == k), OUT) for k in ("long", "enlarge")]
        way = lambda *ways: [["pattern", "in", list(ways)]]  # noqa: E731
        methods.append(dict(
            key="hybrid", family="tokens", cost="live", export=True,
            doc="new textures from your material photographs alone, made in RAE's token grid: blended token by token, "
                "laid in patches with decoder-drawn seams, one's layout rebuilt in another's grain, or one remixed",
            inputs=[I("texture", "texture", "the first material: a photograph"),
                    I("textureB", "texture", "a second material", optional=True),
                    I("textureC", "texture", "a third material", optional=True)],
            params=[P("pattern", "choice", "patchwork", options=list(HYBRIDS), label="pattern", group="hybrid",
                      choices=HYBRIDS, doc="how the materials make one texture"),
                    P("share", "float", 0.5, 0, 1, 0.01, label="walk", group="hybrid", when=way("blend"),
                      doc="where between the materials: 0 the first, 1 the last, each in between on the way"),
                    P("patch", "float", 8.0, 2, 32, 0.5, label="patch size", unit="tokens", group="hybrid",
                      when=way("patchwork"), doc="how large the patches grow (a token is 16 px)"),
                    P("balance", "float", 0.5, 0, 1, 0.01, label="balance", group="hybrid", when=way("patchwork"),
                      doc="lower gives the first material more of the frame, higher the last"),
                    P("seam", "float", 0.35, 0, 1, 0.01, label="seam width", group="hybrid", when=way("patchwork"),
                      doc="from a clean edge the decoder still draws to one long morph between patches"),
                    P("blocks", "choice", 2, options=[1, 2, 4], label="pieces", group="hybrid",
                      when=way("layout", "remix"),
                      choices={1: "one token", 2: "2 x 2 tokens", 4: "4 x 4 tokens: pieces that keep their own drawing"},
                      doc="how many tokens across each piece that is moved"),
                    P("pick", "int", 1, 1, 12, 1, label="wildness", group="hybrid", when=way("layout"),
                      doc="choose among this many closest pieces at random: 1 always the closest"),
                    P("reach", "float", 12.0, 0, 48, 0.5, label="reach", unit="tokens", group="hybrid",
                      when=way("remix"), doc="how far a piece may travel: 0 the material as it is, more scatters it wider"),
                    P("hold", "float", 0.0, 0, 1, 0.01, label="hold the first", group="hybrid", when=way("layout", "remix"),
                      doc="how much of the first material's own tokens stay mixed in"),
                    P("zoom", "float", 2.0, 1, 6, 0.1, label="material zoom", group="hybrid",
                      doc="how close into each material photograph: 1 the whole of it across the frame"),
                    P("seed", "seed", 0, label="seed", group="hybrid",
                      doc="another number: another pattern, other random choices, other pieces of the materials"),
                    P("frame", "choice", "3:2", options=list(FRAMES), label="frame", group=OUT,
                      doc="the texture's shape")] + out_params,
            run=_hybrid_runner(files)))
        drawn = {k: manifest.get(k) for k in ("tokendit", "rae256")}
        if all(drawn.values()) and (pathlib.Path(root) / drawn["rae256"]).is_file() and \
                any((pathlib.Path(root) / drawn["tokendit"]).glob("ema-*.pt")):
            files = {**files, **{k: str(pathlib.Path(root) / f) for k, f in drawn.items()}}
            methods.append(dict(
                key="draw", family="tokens", cost="seconds", export=True,
                doc="new material drawn straight in DINOv2's token space by a diffusion transformer trained on the "
                    "archive's crops (tokendit.py), steered by your materials' places in the space, drawn in its "
                    "16 x 16 windows over a frame of any shape, drawn back by RAE's decoder and enlarged four times",
                inputs=[I("texture", "texture", "the material to steer toward (its place in the space)", optional=True),
                        I("textureB", "texture", "a second material, to walk toward", optional=True),
                        I("textureC", "texture", "a third", optional=True)],
                params=[P("share", "float", 0.0, 0, 1, 0.01, label="walk", group="drawn",
                          doc="where between the materials' places: 0 the first, 1 the last"),
                        P("guide", "float", 3.0, 1, 8, 0.1, label="guidance", group="drawn",
                          doc="how hard the drawing is pushed toward the materials: 1 barely, higher more like them"),
                        P("steps", "int", 30, 8, 60, 1, label="steps", group="drawn",
                          doc="how many steps from noise: fewer is faster and rougher"),
                        P("seed", "seed", 0, label="seed", group="drawn", doc="another number, another drawing"),
                        P("frame", "choice", "3:2", options=list(FRAMES), label="frame", group=OUT,
                          doc="the drawing's shape"),
                        P("long", "choice", 768, options=[512, 768, 1024], label="working size", group=OUT,
                          choices={512: "21 x 32 tokens, 1.4 s (0.4 s dragging)", 768: "32 x 48 tokens, 2.7 s (0.8 s dragging)",
                                   1024: "43 x 64 tokens, 6 to 9 s (1.7 s dragging)"},
                          doc="pixels on the long side the decoder draws (16 per token): a bigger drawing holds more "
                              "of the material at the same grain, in more of the generator's 16 x 16 windows"),
                        G(next(x for x in methods[-1]["params"] if x["key"] == "enlarge"), OUT)],
                run=_draw_runner(files)))
    return {m["key"]: m for m in methods}


def complete(spec, params):
    """A method's settings with every one it declares present: whatever a page, a recipe or a
    sidecar from before a setting existed leaves out takes its declared default."""
    return {**{p["key"]: p["default"] for p in spec["params"]}, **params}


def public(methods):
    """The registry as the client sees it: everything but the run functions."""
    return {k: {kk: vv for kk, vv in m.items() if kk != "run"} for k, m in methods.items()}
