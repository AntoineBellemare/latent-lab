"""Walk a conditional model from one category to another, and measure whether the walk is even.

    python morph.py --model conditional.onnx --from 0 --to 2 --out morph.png

The strip shows what sits between two materials. The number under it is the one that decides whether
a signal can drive the crossing: the spread of the distance between consecutive frames. A morph that
holds still for half its length and then jumps is not a control surface, whatever the ends look like.

`--all` walks every neighbouring pair in turn, which says whether the whole category space is even or
only parts of it.
"""

import argparse
import itertools
import pathlib

import numpy as np
from PIL import Image

from diversity import open_model, wants


def walk(session, width, first, last, classes, steps, z):
    """Frames along a straight line from one category to another, the latent held still."""
    name = session.get_inputs()[0].name
    tag = session.get_inputs()[1].name
    out = []
    for t in np.linspace(0.0, 1.0, steps):
        share = np.zeros((1, classes), np.float32)
        share[0, first], share[0, last] = 1.0 - t, t
        got = session.run(None, {name: z[None], tag: share})[0]
        a = np.asarray(got, np.float32)
        a = a[0] if a.ndim == 4 else a
        if a.shape[0] <= 4:
            a = a.transpose(1, 2, 0)
        out.append(np.clip(a * 0.5 + 0.5, 0, 1))
    return np.stack(out)


def evenness(frames):
    """How evenly the walk spends itself: the spread of consecutive steps, over their mean."""
    step = np.sqrt(((frames[1:] - frames[:-1]) ** 2).mean(axis=(1, 2, 3)))
    return float(step.std() / (step.mean() + 1e-9)), step


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="first", type=int, default=0)
    ap.add_argument("--to", dest="last", type=int, default=1)
    ap.add_argument("--all", action="store_true", help="Every neighbouring pair, one strip each.")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--side", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    session, width = open_model(args.model)
    extra = wants(session)[1:]
    if not extra:
        raise SystemExit(f"{pathlib.Path(args.model).name} takes no category: there is nothing to morph between")
    classes = extra[0][1]

    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal(width).astype(np.float32)
    z *= np.sqrt(width) / (np.linalg.norm(z) + 1e-6)

    pairs = list(itertools.pairwise(range(classes))) if args.all else [(args.first, args.last)]
    bands, spreads = [], []
    for first, last in pairs:
        frames = walk(session, width, first, last, classes, args.steps, z)
        spread, _ = evenness(frames)
        spreads.append(spread)
        print(f"  {first} to {last}: step spread {spread:.3f}")
        small = [np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((args.side, args.side), Image.LANCZOS)) for f in frames]
        bands.append(np.concatenate(small, axis=1))
        if (first, last) != pairs[-1]:
            bands.append(np.full((8, args.steps * args.side, 3), 24, np.uint8))

    Image.fromarray(np.concatenate(bands, axis=0)).save(args.out)
    print(f"  mean step spread {np.mean(spreads):.3f} over {len(pairs)} crossing(s) -> {pathlib.Path(args.out).name}")
    print("  lower is smoother; the linear baseline of this project sits near 0.25")


if __name__ == "__main__":
    main()
