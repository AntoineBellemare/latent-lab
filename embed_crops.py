"""Cut a set into the crops a GAN trains on, and place every crop in the space the lab mixes in.

`fastgan.py --look` conditions a generator on the same vectors the lab's IP-Adapter reads, so a mix
of materials made in the lab is something the GAN can draw live. That needs pairs: a crop at the
GAN's scale, and the adapter encoder's vector for exactly that crop. A crop's vector depends on where
the crop was taken, so they are cut once, written to disk, and embedded together.

    python embed_crops.py --images ~/pictures/set --out ~/crops --size 512 --detail 3 --crops 16

Each crop is written a little larger than the GAN's square (`--margin`) so the trainer can still
move inside it. The vector is taken over the whole of it, and a square of 0.89 of its side reads the
same to the encoder.

The vectors have to come from the encoder the adapter was trained with — the same rule as
`ip_space.py`, for the same reason.
"""

import argparse
import concurrent.futures
import pathlib
import random

import numpy as np
import torch
from PIL import Image

from ip_space import CLIP_MEAN, CLIP_STD, PRESETS
from lora import images


def cut(path, side, detail, count, margin, seed):
    """`count` squares of a `detail`-th of the frame's short side, a `margin` larger, at `side * margin`."""
    rng = random.Random(seed)
    big = round(side * margin)
    with Image.open(path) as im:
        im = im.convert("RGB")
    span = min(round(min(im.size) / detail * margin), min(im.size))
    got = []
    for _ in range(count):
        left, top = rng.randint(0, im.width - span), rng.randint(0, im.height - span)
        got.append(((left, top, span), im.crop((left, top, left + span, top + span)).resize((big, big), Image.LANCZOS)))
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of images, read recursively.")
    ap.add_argument("--out", required=True, help="A folder: crops/ and crops.npz are written into it.")
    ap.add_argument("--size", type=int, default=512, help="The GAN's square.")
    ap.add_argument("--detail", type=float, default=3.0, help="How much of a frame the GAN's square sees: 3 is a third.")
    ap.add_argument("--crops", type=int, default=16, help="Crops per image.")
    ap.add_argument("--margin", type=float, default=1.125, help="How much larger than the square a crop is kept.")
    ap.add_argument("--preset", default="sd15", choices=sorted(PRESETS), help="Whose encoder: the adapter the lab runs.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8, help="Decoding and writing are the slow half.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import CLIPVisionModelWithProjection

    p = PRESETS[args.preset]
    device = torch.device(args.device)
    encoder = CLIPVisionModelWithProjection.from_pretrained(p["repo"], subfolder=p["encoder"], torch_dtype=torch.float16)
    encoder = encoder.to(device).eval()
    side = encoder.config.image_size
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    files = images(args.images)
    out = pathlib.Path(args.out).expanduser()
    (out / "crops").mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} images, {args.crops} crops each of 1/{args.detail:g} of a frame, kept at "
          f"{round(args.size * args.margin)} for a {args.size} square", flush=True)

    def one(i):
        try:
            got = cut(files[i], args.size, args.detail, args.crops, args.margin, seed=i)
        except OSError as e:
            print(f"  skipped {files[i].name}: {e}", flush=True)
            return i, [], []
        names = []
        for k, (_, im) in enumerate(got):
            name = f"{i:05d}_{k:02d}.jpg"
            im.save(out / "crops" / name, quality=95)
            names.append(name)
        small = [np.asarray(im.resize((side, side), Image.LANCZOS), np.uint8) for _, im in got]
        return i, [(n, box) for n, (box, _) in zip(names, got)], small

    vecs, owner, names, boxes, waiting = [], [], [], [], []

    def embed(batch):
        x = torch.tensor(np.stack(batch), device=device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            vecs.append(encoder(pixel_values=((x - mean) / std).half()).image_embeds.float().cpu())

    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        for done, (i, made, small) in enumerate(pool.map(one, range(len(files)))):
            for (name, box), s in zip(made, small):
                owner.append(i)
                names.append(name)
                boxes.append(box)
                waiting.append(s)
            while len(waiting) >= args.batch:
                embed(waiting[: args.batch])
                waiting = waiting[args.batch :]
            if done % 50 == 0:
                print(f"  {done}/{len(files)}", flush=True)
        if waiting:
            embed(waiting)

    emb = torch.cat(vecs).numpy()
    owner = np.array(owner)
    norm = float(np.median(np.linalg.norm(emb, axis=1)))
    np.savez(
        out / "crops.npz",
        emb=emb.astype(np.float32),
        owner=owner,
        crops=np.array(names),
        boxes=np.array(boxes, dtype=np.int32),
        files=np.array([str(f) for f in files]),
        norm=np.float32(norm),
        size=np.int32(args.size),
        detail=np.float32(args.detail),
        preset=np.array(args.preset),
    )
    print(f"wrote {len(emb)} crops and their vectors to {out}, typical length {norm:.2f}")


if __name__ == "__main__":
    main()
