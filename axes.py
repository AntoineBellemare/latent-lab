"""How many knobs does a model actually have?

Two different questions, and they have different answers. How much of the latent's own variance sits
on each axis — the scale the export baked in — and how much the PICTURE moves when you turn that
axis. A model can have two hundred axes on paper and three you can feel.

    python axes.py --model noema.onnx

Reports both, and where the useful knobs run out.
"""

import argparse
import pathlib

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


def baked(path):
    """The per-axis standard deviations the exporter folded into the graph, if it did."""
    model = onnx.load(str(path))
    for init in model.graph.initializer:
        if init.name == "sigma":
            return numpy_helper.to_array(init).ravel()
    return None


def draw(session, name, z, span):
    got = session.run(None, {name: np.asarray(z, np.float32)[None]})[0]
    a = np.asarray(got, dtype=np.float32)
    a = a[0] if a.ndim == 4 else a
    if a.shape[0] <= 4:
        a = a.transpose(1, 2, 0)
    return np.clip(a * 0.5 + 0.5 if span == "signed" else a, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--range", default="signed", choices=["signed", "unit"])
    ap.add_argument("--axes", type=int, default=24, help="How many to measure.")
    ap.add_argument("--push", type=float, default=2.0, help="How far to turn each, in sigma.")
    ap.add_argument("--bases", type=int, default=4, help="Places in the space to average the effect over.")
    ap.add_argument("--floor", type=float, default=0.02, help="Mean pixel move that counts as visible.")
    args = ap.parse_args()

    ready = onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(args.model, providers=[q for q in PROVIDERS if q in ready])
    name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    width = next(int(d) for d in reversed(shape) if isinstance(d, int) and d > 1)
    keep = min(args.axes, width)

    sigma = baked(args.model)
    middle = draw(session, name, np.zeros(width), args.range)

    # Averaged over several places in the space, not only its centre. Measured at the mean alone the
    # count swung between six and twenty-four across consecutive snapshots of one run: that is one
    # point of a 256-dimensional space answering for all of it.
    rng = np.random.default_rng(0)
    bases = [np.zeros(width)] + [rng.standard_normal(width) * 0.7 for _ in range(args.bases - 1)]
    moves = []
    for k in range(keep):
        felt = []
        for base in bases:
            lo, hi = base.copy(), base.copy()
            lo[k], hi[k] = base[k] - args.push, base[k] + args.push
            felt.append(np.abs(draw(session, name, hi, args.range) - draw(session, name, lo, args.range)).mean())
        moves.append(float(np.mean(felt)))
    moves = np.array(moves)

    print(f"{pathlib.Path(args.model).name}: {width} axes, measuring {keep} at +/-{args.push} sigma")
    print(f"  {'axis':>5}{'baked sigma':>14}{'share of var':>14}{'picture moves':>15}{'vs axis 0':>11}")
    total = float((sigma**2).sum()) if sigma is not None else None
    for k in range(keep):
        s = f"{sigma[k]:.4f}" if sigma is not None else "-"
        share = f"{sigma[k] ** 2 / total:6.2%}" if sigma is not None else "-"
        print(f"  {k:>5}{s:>14}{share:>14}{moves[k]:>15.4f}{moves[k] / moves[0]:>11.2f}")

    strong = int((moves > moves[0] * 0.25).sum())
    print(f"\n  axes moving the picture at least a quarter as much as axis 0: {strong}")
    print(f"  axes moving it at least {args.floor:.3f}, which is visible side by side: {int((moves > args.floor).sum())}")
    if sigma is not None:
        half = int(np.searchsorted(np.cumsum(sigma**2) / total, 0.5)) + 1
        print(f"  axes holding half the latent's variance: {half}")
        print(f"  axes holding ninety percent: {int(np.searchsorted(np.cumsum(sigma**2) / total, 0.9)) + 1}")
    print(f"  a flat baseline would show every axis near 1.00 in the last column")


if __name__ == "__main__":
    main()
