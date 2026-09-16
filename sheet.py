"""A model's draws beside the photographs it learnt from, as one picture.

    python sheet.py --model textures.onnx --images ~/pictures/set --out sheet.png
    python sheet.py --model conditional.onnx --images ~/pictures/set-by-category --out sheet.png --each

Four preview tiles is how three faults in this project went unnoticed: a tiling the metrics called
diverse, a saturated drift the pixel statistics scored as healthy, and a generator reported working
that drew at a fifth of its data's spread. Eighteen draws over eighteen real crops catches all three
at a glance, which is why every claim here gets one of these under it.

`--each` gives a conditional model one row per category, which is the only way to see whether the
category input does anything at all.
"""

import argparse
import pathlib

import numpy as np
from PIL import Image

from diversity import drawn, open_model, real, wants


def tiles(frames, side):
    return [np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((side, side), Image.LANCZOS)) for f in frames]


def row(frames, side):
    return np.concatenate(tiles(frames, side), axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", help="The set to put beside it; omitted, the model stands alone.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--across", type=int, default=6)
    ap.add_argument("--rows", type=int, default=3, help="Rows of model draws, when not using --each.")
    ap.add_argument("--side", type=int, default=224)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--category", type=int, help="Hold one category, for a conditional model.")
    ap.add_argument("--each", action="store_true", help="One row per category, for a conditional model.")
    args = ap.parse_args()

    session, width = open_model(args.model)
    extra = wants(session)[1:]
    classes = extra[0][1] if extra else 0
    gap = None
    bands = []

    if args.each and classes:
        for c in range(classes):
            bands.append(row(drawn(session, width, args.across, seed=args.seed, category=c), args.side))
    else:
        got = drawn(session, width, args.across * args.rows, seed=args.seed, category=args.category)
        bands += [row(got[i * args.across : (i + 1) * args.across], args.side) for i in range(args.rows)]

    if args.images:
        gap = np.full((10, args.across * args.side, 3), 24, np.uint8)
        size = drawn(session, width, 1, seed=args.seed).shape[1]
        true = real(args.images, size, args.across, seed=args.seed)
        bands += [gap, row(true, args.side)]

    Image.fromarray(np.concatenate(bands, axis=0)).save(args.out)
    what = f"{classes} categories, one row each" if args.each and classes else f"{args.rows} rows of draws"
    print(f"  {what}" + (", real crops under the rule" if args.images else "") + f" -> {pathlib.Path(args.out).name}")


if __name__ == "__main__":
    main()
