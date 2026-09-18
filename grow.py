"""Grow a trained generator to twice its resolution, keeping everything it already knows.

    python grow.py --from conditional-twophase/textures.pt --out grown1024.pt
    python fastgan.py --images set-by-category --classes --size 1024 --init grown1024.pt

A 1024 run from nothing collapsed before its first snapshot: every latent drew the same blurry blob,
the critic won outright, and 94,000 more steps never brought it back. Carrying over the generator
alone does not fix that — a trained generator against a fresh critic is its own way to collapse — so
BOTH networks come across and only what the new resolution adds starts from zero.

In the generator everything below the new size lines up by name. The critic's blocks line up once
shifted by the one block the new size puts in front of them. The genuinely new parts — the
generator's last upsample and output, the critic's first downsample — are all that learns from
scratch.
"""

import argparse
import re

import torch

from blend import shape_of
from fastgan import Discriminator, Generator, excited, resolutions


def carry(target, given):
    """Every weight `given` holds under a name and shape `target` also has; the rest stays fresh."""
    state = target.state_dict()
    fresh = []
    for key in state:
        if key in given and given[key].shape == state[key].shape:
            state[key] = given[key]
        else:
            fresh.append(key)
    target.load_state_dict(state)
    return len(state) - len(fresh), fresh


def shifted(critic, by):
    """The old critic's blocks renamed to where they sit in the grown one.

    Its first layer is dropped: it reads RGB at the OLD resolution, and nothing in the grown critic
    does, so what replaces it has to learn.
    """
    out = {}
    for key, value in critic.items():
        found = re.match(r"down\.(\d+)\.(.+)", key)
        if not found:
            out[key] = value
        elif int(found.group(1)) >= 2:
            out[f"down.{int(found.group(1)) + by}.{found.group(2)}"] = value
    return out


def below(gen, w, size):
    """The feature map a generator holds at `size`, just before it would draw."""
    x = gen.stem(w.reshape(-1, gen.latent, 1, 1))
    held = {4: x}
    for r, step in zip(resolutions(gen.size), gen.steps):
        if r >= size:
            break
        x = step(x)
        small = r * 2 // 16
        if small in held:
            x = gen.excites[excited(gen.size).index(small)](held[small], x)
        held[r * 2] = x
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="source", required=True, help="A trained `.pt`.")
    ap.add_argument("--out", required=True, help="The grown `.pt`, for `fastgan.py --init`.")
    ap.add_argument("--seed", type=int, default=0, help="For the weights that start fresh.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    was = torch.load(args.source, map_location="cpu", weights_only=False)
    latent, size, ngf, classes = shape_of(was["gen"])
    # The verdict reads channels(16, ndf), which is four times ndf.
    ndf = was["dis"]["verdict.weight"].shape[1] // 4
    grown = size * 2

    gen = Generator(latent, grown, ngf, classes=classes)
    smooth = Generator(latent, grown, ngf, classes=classes)
    dis = Discriminator(grown, ndf, classes=classes)
    by = len(dis.down) - len(Discriminator(size, ndf, classes=classes).down)

    kept_g, fresh_g = carry(gen, was["gen"])
    carry(smooth, was["smooth"])
    kept_d, fresh_d = carry(dis, shifted(was["dis"], by))

    # The proof the carry is exact: below the old resolution the grown generator must draw exactly
    # what the old one did. Anything but zero means a layer landed in the wrong place.
    old = Generator(latent, size, ngf, classes=classes).eval()
    old.load_state_dict(was["gen"])
    gen.eval()
    with torch.no_grad():
        w = torch.randn(3, latent)
        drift = float((below(old, w, size) - below(gen, w, size)).abs().max())
    gen.train()
    if drift > 1e-5:
        raise SystemExit(f"the grown generator does not reproduce the old one below {size}: off by {drift}")

    torch.save(
        {
            "step": 0,
            "gen": gen.state_dict(),
            "dis": dis.state_dict(),
            "smooth": smooth.state_dict(),
            "opt_g": None,
            "opt_d": None,
            "path_mean": torch.zeros(()),
            "taken": 0,
        },
        args.out,
    )
    print(f"  {size} -> {grown}, {classes} categories")
    print(f"  generator: {kept_g} carried, {len(fresh_g)} fresh ({', '.join(sorted({k.split('.')[0] + '.' + k.split('.')[1] for k in fresh_g}))})")
    print(f"  critic:    {kept_d} carried, {len(fresh_d)} fresh, blocks shifted by {by}")
    print(f"  below {size} the grown generator matches the old one to {drift:.1e}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
