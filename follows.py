"""Does a space-conditioned GAN draw the place it is asked for?

    python follows.py --model looks.onnx --crops ~/crops --sheet follows.png

`fastgan.py --look` takes a place in the lab's space and is meant to draw it. A preview cannot say
whether it does: a generator that ignored its input would still draw good textures, just not the
ones asked for. So this asks for places, embeds what comes back with the encoder the places came
from, and reads three numbers.

- **found**: asked for one of N photographs' places, how often the picture's own vector lies nearest
  the place asked for. Beside it, the same for a real crop of that photograph, which is the
  ceiling: no picture can be told apart better than the photographs themselves.
- **likeness**: how far the picture points the way its place does, once the set's shared direction
  is taken off (every texture shares most of it). Beside it, a real crop's.
- **between**: a walk from one place to another, the latent held still, mixed the way the lab mixes
  (a weighted sum, renormalised). How far along the line each frame's vector stands, whether that
  only ever moves forward, and how evenly — the spread of the step lengths, as `morph.py` reads it.
"""

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image

from ip_space import CLIP_MEAN, CLIP_STD, PRESETS


class Eye:
    """The adapter's own encoder, which is what put every place where it is."""

    def __init__(self, preset, device):
        from transformers import CLIPVisionModelWithProjection

        p = PRESETS[preset]
        self.net = CLIPVisionModelWithProjection.from_pretrained(
            p["repo"], subfolder=p["encoder"], torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
        self.side = self.net.config.image_size
        self.device = device
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, pictures):
        """[n, h, w, 3] in [0, 1] to [n, 1024]."""
        x = torch.tensor(np.stack(pictures), device=self.device).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=self.side, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        with torch.no_grad():
            return self.net(pixel_values=((x - self.mean) / self.std).to(self.net.dtype)).image_embeds.float().cpu().numpy()


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def drawer(model, device):
    import onnxruntime

    ready = onnxruntime.get_available_providers() if device.type == "cuda" else []
    s = onnxruntime.InferenceSession(str(model), providers=[p for p in ("CUDAExecutionProvider",) if p in ready] + ["CPUExecutionProvider"])
    z, look = (i.name for i in s.get_inputs())
    width = s.get_inputs()[0].shape[-1]

    def draw(place, seed, spread=1.0):
        zz = np.random.default_rng(seed).standard_normal((1, width)).astype(np.float32) * spread
        a = np.asarray(s.run(None, {z: zz, look: place[None].astype(np.float32)})[0], np.float32)[0]
        return np.clip(a.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)

    return draw


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="An .onnx from `fastgan.py --look`: inputs z and look.")
    ap.add_argument("--crops", required=True, help="The `embed_crops.py` folder it was trained on.")
    ap.add_argument("--places", type=int, default=64, help="Photographs asked for.")
    ap.add_argument("--draws", type=int, default=3, help="Pictures per place.")
    ap.add_argument("--pairs", type=int, default=6, help="Walks between two places.")
    ap.add_argument("--steps", type=int, default=9, help="Frames per walk.")
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--sheet", help="Where to write a sheet: per place, a real crop and then the draws.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = pathlib.Path(args.crops).expanduser()
    held = np.load(root / "crops.npz")
    emb, owner, names = held["emb"], held["owner"], held["crops"]
    eye = Eye(str(held["preset"]), torch.device(args.device))
    draw = drawer(args.model, torch.device(args.device))
    rng = np.random.default_rng(args.seed)
    photos = rng.choice(np.unique(owner), args.places, replace=False)
    centre = unit(emb).mean(0)

    # Each photograph's place is the mean of its crops but one; that one is the real crop it is read
    # against, so the ceiling is a crop the place was not made from.
    places, real, real_names = [], [], []
    for p in photos:
        mine = np.flatnonzero(owner == p)
        out = rng.choice(mine)
        places.append(emb[mine[mine != out]].mean(0))
        real.append(emb[out])
        real_names.append(names[out])
    places, real = unit(np.stack(places)), unit(np.stack(real))

    pictures = [[draw(places[i], args.seed * 1000 + k, args.spread) for k in range(args.draws)] for i in range(len(places))]
    seen = unit(np.stack([eye(row) for row in pictures]))  # [places, draws, 1024]

    def found(vecs):
        near = np.argmax((unit(vecs - centre)) @ unit(places - centre).T, axis=-1)
        return float((near == np.arange(len(places)).reshape(-1, *([1] * (near.ndim - 1)))).mean())

    def likeness(vecs):
        a = unit(vecs - centre)
        b = unit(places - centre).reshape(len(places), *([1] * (a.ndim - 2)), -1)
        return float((a * b).sum(-1).mean())

    others = unit(places - centre)[rng.permutation(len(places))]
    print(f"{len(places)} places, {args.draws} draws each, spread {args.spread}")
    print(f"  found     {found(seen):.2f}   real crops {found(real):.2f}   chance {1 / len(places):.2f}")
    print(f"  likeness  {likeness(seen):.2f}   real crops {likeness(real):.2f}   "
          f"another place {float((unit(real - centre) * others).sum(-1).mean()):.2f}")

    # Walks, the lab's way: a weighted sum of two places at real length, renormalised.
    walks = []
    for k in range(args.pairs):
        a, b = places[2 * k], places[2 * k + 1]
        frames = [draw(unit((1 - t) * a + t * b), 7 + k, args.spread) for t in np.linspace(0, 1, args.steps)]
        v = unit(eye(frames)) - centre
        line = (b - a) / np.square(b - a).sum()
        along = (v - (a - centre)) @ line
        gaps = np.abs(np.diff(along))
        walks.append((frames, along))
        print(f"  between {k}: along {' '.join(f'{x:+.2f}' for x in along)}   forward "
              f"{float((np.diff(along) > 0).mean()):.2f}   step spread {float(gaps.std() / max(gaps.mean(), 1e-6)):.2f}")

    if args.sheet:
        thumb = 160
        rows = []
        for i in range(min(16, len(places))):
            with Image.open(root / "crops" / real_names[i]) as im:
                im = im.convert("RGB")
                c = (im.width - 512) // 2
                r = im.crop((c, c, c + 512, c + 512)).resize((thumb, thumb), Image.LANCZOS)
            row = [np.asarray(r, np.float32) / 255] + [np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize((thumb, thumb), Image.LANCZOS), np.float32) / 255 for p in pictures[i]]
            rows.append(np.concatenate(row, 1))
        for frames, _ in walks[:4]:
            strip = [np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((thumb, thumb), Image.LANCZOS), np.float32) / 255 for f in frames]
            rows.append(np.concatenate(strip, 1))
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
        Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(args.sheet)
        print(f"  sheet {args.sheet}")


if __name__ == "__main__":
    main()
