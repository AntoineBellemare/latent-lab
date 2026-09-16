"""Does a conditional model's category input actually do anything, and for which pairs?

    python separation.py --model conditional.onnx

Holds the latent still and turns only the category. What comes back is how far the picture moves,
against how far the LATENT moves it within one category — the honest scale to read it on, because a
category that shifts the image a tenth as much as its own noise is not a control.

The pair table is the point. One run scored 0.136 on average and looked conditioned, while three
mineral categories sat at 0.02 to 0.07 of each other: merged in all but name, and invisible in the
average. Read the smallest pair, not the mean.
"""

import argparse
import itertools
import pathlib

import numpy as np

from diversity import feed, open_model, wants


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--draws", type=int, default=6, help="Latents to average each comparison over.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quiet", action="store_true", help="Two lines, for a watcher.")
    args = ap.parse_args()

    session, width = open_model(args.model)
    extra = wants(session)[1:]
    if not extra:
        raise SystemExit(f"{pathlib.Path(args.model).name} takes no category")
    k = extra[0][1]

    rng = np.random.default_rng(args.seed)
    latents = []
    for _ in range(args.draws):
        z = rng.standard_normal(width).astype(np.float32)
        latents.append(z * np.sqrt(width) / (np.linalg.norm(z) + 1e-6))

    def draw(z, c):
        got = np.asarray(session.run(None, feed(session, z, c))[0], np.float32)
        got = got[0] if got.ndim == 4 else got
        if got.shape[0] <= 4:
            got = got.transpose(1, 2, 0)
        return np.clip(got * 0.5 + 0.5, 0, 1)

    held = {(c, i): draw(z, c) for c in range(k) for i, z in enumerate(latents)}
    between = np.zeros((k, k))
    for a, b in itertools.combinations(range(k), 2):
        between[a, b] = between[b, a] = np.mean([np.abs(held[(a, i)] - held[(b, i)]).mean() for i in range(len(latents))])
    within = np.mean(
        [
            np.abs(held[(c, i)] - held[(c, j)]).mean()
            for c in range(k)
            for i, j in itertools.combinations(range(len(latents)), 2)
        ]
    )
    pairs = sorted((between[a, b], a, b) for a, b in itertools.combinations(range(k), 2))

    if args.quiet:
        print(f"  category {between[np.triu_indices(k, 1)].mean():.4f} vs latent {within:.4f}, "
              f"weakest pair {pairs[0][1]}-{pairs[0][2]} at {pairs[0][0]:.4f}")
        return

    print(f"  moving the LATENT within one category: {within:.4f}")
    print(f"  moving the CATEGORY, latent held still: {between[np.triu_indices(k, 1)].mean():.4f}")
    print(f"\n  {'':4}" + "".join(f"{c:>7}" for c in range(k)))
    for a in range(k):
        print(f"  {a:>4}" + "".join(f"{between[a, b]:>7.3f}" if a != b else f"{'-':>7}" for b in range(k)))
    print(f"\n  weakest pair:   {pairs[0][1]} and {pairs[0][2]} at {pairs[0][0]:.4f}")
    print(f"  strongest pair: {pairs[-1][1]} and {pairs[-1][2]} at {pairs[-1][0]:.4f}")
    merged = [(v, a, b) for v, a, b in pairs if v < within * 0.25]
    if merged:
        print(f"  merged in all but name, under a quarter of what the latent does:")
        for v, a, b in merged:
            print(f"    {a} and {b}: {v:.4f}")


if __name__ == "__main__":
    main()
