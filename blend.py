"""Average two or more children of one parent into a generator that sits between them.

    python blend.py --models cat0.pt cat2.pt --weights 0.7 0.3 --out mix.onnx

This is the whole reason `fastgan.py --init` exists. Two runs started from nothing put their hidden
units in their own arbitrary order, so averaging them averages unrelated things and the midpoint is
mush. Children of ONE parent keep that order and stay in its basin, and their weights average into a
generator that draws something between what each of them draws — the same reason a model soup works
and two cold starts do not.

Keep the fine-tunes short for the same reason. The further a child travels from its parent, the
rougher the road back to its siblings.
"""

import argparse
import pathlib

import torch

from fastgan import Generator, export


def shape_of(state):
    """The generator this checkpoint holds, read off its own weights.

    A checkpoint carries no record of the flags it was trained with, and asking for them again is a
    way to rebuild the wrong model in silence.
    """
    latent = state["mapping.net.0.weight"].shape[0]
    # The stem is a ConvTranspose2d(latent, channels(4, ngf) * 2, ...), and channels(4, ngf) is 16 ngf.
    ngf = round(state["stem.0.weight"].shape[1] / 2 / 16)
    doublings = len({q.split(".")[1] for q in state if q.startswith("steps.")})
    classes = state["tag.weight"].shape[1] if "tag.weight" in state else 0
    return latent, 4 * 2**doublings, ngf, classes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True, help="Two or more `.pt` checkpoints.")
    ap.add_argument("--weights", nargs="+", type=float, help="One per model; equal shares by default.")
    ap.add_argument("--out", required=True, help="Where to write the blended `.onnx`.")
    ap.add_argument("--raw", action="store_true", help="Blend the unaveraged weights, not the EMA ones.")
    ap.add_argument("--keep", type=int, default=0, help="Axes to export; 0 keeps every one.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    share = args.weights or [1.0 / len(args.models)] * len(args.models)
    if len(share) != len(args.models):
        raise SystemExit(f"{len(args.models)} models but {len(share)} weights")
    total = sum(share)
    if total <= 0:
        raise SystemExit("the weights sum to nothing")
    share = [q / total for q in share]

    which = "gen" if args.raw else "smooth"
    held = [torch.load(q, map_location="cpu", weights_only=False)[which] for q in args.models]
    shapes = {shape_of(q) for q in held}
    if len(shapes) > 1:
        raise SystemExit(f"these were not built the same: {shapes}")
    latent, size, ngf, classes = shapes.pop()

    mixed = {}
    for key in held[0]:
        stack = [q[key] for q in held]
        if stack[0].is_floating_point():
            mixed[key] = sum(w * q for w, q in zip(share, stack))
        else:
            # Integer buffers — a batch norm's count of batches seen — have no meaningful average.
            mixed[key] = stack[0]

    gen = Generator(latent, size, ngf, classes=classes).to(args.device)
    gen.load_state_dict(mixed)
    gen.eval()
    where, keep, kept = export(gen, args.out, latent, size, args.device, args.keep or None, classes)
    named = ", ".join(f"{pathlib.Path(m).stem} {w:.0%}" for m, w in zip(args.models, share))
    print(f"  {named}")
    print(f"  wrote {where}: {size} square, {keep} axes holding {kept:.1%} of the variance in w")


if __name__ == "__main__":
    main()
