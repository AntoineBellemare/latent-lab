"""A personal axis through the space: defined by example images, usable as both ruler and knob.

    python concepts.py --space texture-ip-sd15.npz --name humid \
        --pos wet glossy rippled --neg dry dusty powdery --rank humid-rank.png --sweep humid-sweep.png

A concept here is a DIRECTION in the IP space, and that one object does two jobs at once: project
onto it and the archive is ranked by the concept (the ruler); add it to any point and the adapter
draws the step (the knob). Scorer and edit are the same vector — no bridge model stands between a
definition and a control.

This seed version defines the exemplar sets through caption terms (`--pos wet glossy` means "the
images whose captions carry those words"), which is a proxy an artist's own picks will replace: the
mechanism is identical, only the choosing hand differs. The direction is the difference of the two
sets' mean embeddings — the simplest probe there is, and the one to beat.

`--rank` draws the archive's two ends of the axis, which is the sanity check that costs nothing: a
concept that cannot sort photographs will not steer a generator. `--sweep` renders a walk along the
axis, from a start point (`--on` a preset, or the centre of the space), in units of the archive's
own spread along it. The fitted direction is saved into `--book` (concepts.json) beside the presets,
so `terms.py`-style mixes can reach it later.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

from composite import fit
from ip_space import PRESETS, pipeline, draw, slerp
from terms import index, term_vector


def exemplars(space, found, words):
    files = sorted(set().union(*(found.get(w, []) for w in words)))
    missing = [w for w in words if w not in found]
    if missing:
        print(f"  (no caption carries {', '.join(repr(w) for w in missing)})")
    return files


def direction_of(space, pos, neg):
    """Difference of means over crop embeddings, unit length, with both ends' files kept apart."""
    both = set(pos) & set(neg)
    pos = [f for f in pos if f not in both]
    neg = [f for f in neg if f not in both]
    a = space["emb"][np.isin(space["owner"], pos)].mean(0)
    b = space["emb"][np.isin(space["owner"], neg)].mean(0)
    d = a - b
    return (d / np.linalg.norm(d)).astype(np.float32), pos, neg


def image_scores(space, d):
    """One score per photograph: its crops' mean embedding, projected."""
    count = len(space["files"])
    sums = np.zeros((count, space["emb"].shape[1]), np.float32)
    np.add.at(sums, space["owner"], space["emb"])
    per = sums / np.bincount(space["owner"], minlength=count)[:, None]
    return per @ d


def thumb(path, side=160):
    im = Image.open(path).convert("RGB")
    im.draft("RGB", (im.width // 4, im.height // 4))
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
    return im.resize((side, side), Image.LANCZOS)


def rank_sheet(space, scores, name, out, show=16, side=160):
    order = np.argsort(scores)
    rows = [(f"most {name}", order[::-1][:show]), (f"least {name}", order[:show])]
    cols = show // 2
    sheet = Image.new("RGB", (cols * side, 2 * (26 + 2 * side)), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    for b, (title, picks) in enumerate(rows):
        y0 = b * (26 + 2 * side)
        pen.text((6, y0 + 6), f"{title} — the archive sorted along the fitted direction", fill=(214, 210, 202))
        for n, i in enumerate(picks):
            r, c = divmod(n, cols)
            sheet.paste(thumb(str(space["files"][i]), side), (c * side, y0 + 26 + r * side))
    sheet.save(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pos", nargs="+", required=True, help="Caption terms whose images show the concept.")
    ap.add_argument("--neg", nargs="+", required=True, help="Caption terms whose images lack it.")
    ap.add_argument("--rank", help="Contact sheet of the axis's two ends.")
    ap.add_argument("--sweep", help="Rendered walk along the axis.")
    ap.add_argument("--on", help="Start the sweep at this preset from --presets, rather than the centre.")
    ap.add_argument("--presets", default="presets.json")
    ap.add_argument("--push", type=float, default=3.0, help="How far --on's preset is pushed, as in terms.py.")
    ap.add_argument("--steps", type=float, nargs="*", default=[-8, -6, -4, -2, 0, 2, 4, 6, 8],
                    help="Sweep positions, in standard deviations of the archive's own spread along the axis.")
    ap.add_argument("--quality", type=int, default=45, help="Denoising steps per tile.")
    ap.add_argument("--book", default="concepts.json", help="Where fitted concepts accumulate.")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols-seeds", type=int, default=2, help="Seeds per sweep position, one row each.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    space = dict(np.load(args.space, allow_pickle=True))
    found = index(space)
    pos = exemplars(space, found, args.pos)
    neg = exemplars(space, found, args.neg)
    d, pos, neg = direction_of(space, pos, neg)
    scores = image_scores(space, d)
    sigma = float(scores.std())
    gap = (scores[pos].mean() - scores[neg].mean()) / sigma
    print(f"{args.name}: {len(pos)} for, {len(neg)} against, ends {gap:.2f} spreads apart")

    book = pathlib.Path(args.book)
    kept = json.loads(book.read_text(encoding="utf-8")) if book.is_file() else {}
    kept[args.name] = dict(pos=args.pos, neg=args.neg, sigma=sigma, direction=[round(float(v), 6) for v in d])
    book.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    print(f"saved {args.name!r} to {book}")

    if args.rank:
        rank_sheet(space, scores, args.name, args.rank)

    if args.sweep:
        p = dict(PRESETS["sd15"], fast=None, steps=args.quality, guidance=7.0)
        quiet = argparse.Namespace(scale=args.scale, lora=None, style_only=False, prompt="")
        pipe, dtype = pipeline(quiet, p, torch.device(args.device))
        if args.on:
            mixes = json.loads(pathlib.Path(args.presets).read_text(encoding="utf-8"))
            start = term_vector(space, dict(mixes[args.on]), found, args.push)
            where = f"preset {args.on!r}"
        else:
            start = space["centre"] * (float(space["norm"]) / np.linalg.norm(space["centre"]))
            where = "the centre of the space"
        # The walk goes along the sphere between the two END points, in equal angles. Stepping the
        # offset linearly and rescaling bunches all the angular travel in the middle columns, which
        # is exactly where the earlier sweeps jumped; equal angles spread the transition evenly.
        norm = float(space["norm"])
        lo, hi = min(args.steps), max(args.steps)
        ends = [start + t * sigma * d for t in (lo, hi)]
        ends = [v * (norm / np.linalg.norm(v)) for v in ends]
        side, label = p["size"], 26
        rows = args.cols_seeds
        sheet = Image.new("RGB", (len(args.steps) * side, label + rows * side), (18, 18, 20))
        pen = ImageDraw.Draw(sheet)
        pen.text((6, 6), f"{args.name}, swept from {where}: {', '.join(f'{t:+g} sd' for t in args.steps)}", fill=(214, 210, 202))
        for c, t in enumerate(args.steps):
            v = slerp(ends[0], ends[1], (t - lo) / (hi - lo)).astype(np.float32)
            for r in range(rows):
                tile = draw(pipe, dtype, p, quiet, v, args.seed + r, torch.device(args.device))
                sheet.paste(tile.resize((side, side)), (c * side, label + r * side))
            print(f"  {t:+g} sd", flush=True)
        sheet.save(args.sweep)
        print(f"wrote {args.sweep}")


if __name__ == "__main__":
    main()
