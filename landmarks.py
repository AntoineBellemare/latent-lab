"""The space's own places, drawn: its centre, its far side, its algebra, its lines between photographs.

    python landmarks.py --space texture-ip-sd15.npz --out landmarks.png \
        --algebra "rippled water + tree bark - still water" --between DSC01705 DSC07698

Four kinds of landmark, none of them trained:

- the CENTROID — the mean of every crop vector, rescaled to a real crop's length: the one
  photograph the archive keeps trying to take.
- the ANTIPODE — the same direction reversed: the far side of the sphere, everything the archive
  refuses. The adapter has never been pointed there by a photograph; what it draws is the
  archive's negative space.
- ALGEBRA — caption terms added and subtracted as vectors, `a + b - c`, the analogy made visible.
- BETWEEN — the great-circle walk from one named photograph to another, seed held: the space
  between two memories, possibly taken years apart.
"""

import argparse
import pathlib
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

from ip_space import PRESETS, pipeline, draw, slerp
from terms import index, term_vector


def rescale(v, norm):
    return v * (norm / np.linalg.norm(v))


def photo_vector(space, pattern):
    """One photograph's crops averaged — the picture as a place, found by filename."""
    hits = [i for i, f in enumerate(space["files"]) if pattern.lower() in str(f).lower()]
    if not hits:
        raise SystemExit(f"no file matches {pattern!r}")
    mine = np.isin(space["owner"], hits)
    return rescale(space["emb"][mine].mean(0), float(space["norm"])), str(space["files"][hits[0]])


def algebra_vector(space, found, phrase, push):
    """`a + b - c` over caption terms, through the same machinery presets use."""
    bits = re.split(r"\s*([+-])\s*", phrase.strip())
    mix, sign = {}, 1.0
    for bit in bits:
        if bit == "+":
            sign = 1.0
        elif bit == "-":
            sign = -1.0
        elif bit:
            mix[bit] = mix.get(bit, 0.0) + sign
    return term_vector(space, mix, found, push)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5, help="Centroid and antipode: one column per seed.")
    ap.add_argument("--algebra", nargs="*", default=[], help="Rows like 'rippled water + tree bark - still water'.")
    ap.add_argument("--between", nargs="*", default=[], help="Pairs of filename fragments; a morph row per pair.")
    ap.add_argument("--push", type=float, default=8.0, help="Algebra mixes are pushed like presets are.")
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if len(args.between) % 2:
        raise SystemExit("--between takes pairs")

    space = dict(np.load(args.space, allow_pickle=True))
    norm = float(space["norm"])
    found = index(space)
    device = torch.device(args.device)
    p = dict(PRESETS["sd15"], fast=None, steps=30, guidance=7.0)
    quiet = argparse.Namespace(scale=args.scale, lora=None, style_only=False, prompt="")
    pipe, dtype = pipeline(quiet, p, device)

    rows = []
    centre = rescale(space["centre"].astype(np.float32), norm)
    rows.append(("the centroid — the photograph the archive keeps trying to take",
                 [(centre, s) for s in range(args.seeds)]))
    rows.append(("the antipode — everything the archive refuses",
                 [(rescale(-space["centre"].astype(np.float32), norm), s) for s in range(args.seeds)]))
    for phrase in args.algebra:
        v = algebra_vector(space, found, phrase, args.push)
        rows.append((f"algebra: {phrase}", [(v, s) for s in range(args.seeds)]))
    for a, b in zip(args.between[::2], args.between[1::2]):
        va, na = photo_vector(space, a)
        vb, nb = photo_vector(space, b)
        walkers = [(slerp(va, vb, t), args.seed) for t in np.linspace(0, 1, args.cols)]
        rows.append((f"between {pathlib.Path(na).stem} and {pathlib.Path(nb).stem}, seed held", walkers))

    side, label = p["size"], 26
    cols = max(len(r[1]) for r in rows)
    sheet = Image.new("RGB", (cols * side, (side + label) * len(rows)), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    for r, (title, points) in enumerate(rows):
        y = r * (side + label)
        pen.text((6, y + 6), title, fill=(214, 210, 202))
        for c, (v, s) in enumerate(points):
            tile = draw(pipe, dtype, p, quiet, v.astype(np.float32), s, device)
            sheet.paste(tile.resize((side, side)), (c * side, y + label))
        print(f"  {title}", flush=True)
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
