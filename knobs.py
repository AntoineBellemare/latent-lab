"""Turn a source picture toward the textures in the set, with one knob per category.

    python knobs.py --space texture-ip-sd15.npz --image photo.jpg --mix cat1=1 --out turned.png
    python knobs.py --space texture-ip-sd15.npz --image photo.jpg --sweep cat1 --out sweep.png

Prompting is a discrete interface: it works, but only if you know which of fifty-nine material
words the model was trained on, and "an old painted wall" is not one of them. A knob needs no
vocabulary, moves continuously, and can be driven by something that is not a keyboard.

The space already holds what a knob needs. Every category has a mean vector, so a mix of categories
is a weighted sum of them, and any mix in between is a real place rather than a switch position.
Three separate things are being set, and they are worth keeping apart:

  --mix       WHICH textures, and in what PROPORTION to each other
  --scale     HOW MUCH texture, the knob that actually turns
  --strength  how much of the source picture is redrawn at all

The weights in --mix set proportion and nothing else: the blend is rescaled to the length real crops
have, which divides any shared factor back out, so `cat1=0.4` alone and `cat1=1` alone are the same
place. That is correct — a direction in the space has no volume — but it means the knob a hand or a
signal turns is --scale, with --mix choosing which way it points.

A caution the sheet will show: a category mean is the most generic point that category has, because
the orange lichen in one crop and the teal paint in another cancel toward grey. `--from crop` uses a
single real crop's vector instead, which keeps what made it worth photographing but is no longer the
same place twice. Means to steer by, crops to look at.
"""

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

PRESET = dict(
    model="stable-diffusion-v1-5/stable-diffusion-v1-5",
    repo="h94/IP-Adapter", subfolder="models",
    weight="ip-adapter_sd15.safetensors",
)


def fit(path, size):
    got = Image.open(path).convert("RGB")
    side = min(got.size)
    left, top = (got.width - side) // 2, (got.height - side) // 2
    return got.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def vector(space, weights, kind="mean", seed=0, push=1.0):
    """A weighted blend of categories, at the length real crops have.

    Rescaled at the end because an average is shorter than what it averages, and the adapter reads a
    short vector as a faint one — a mix would otherwise fade as it balanced.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(space["means"].shape[1], np.float32)
    centre = space["centre"]
    for c, w in weights.items():
        if kind == "crop":
            mine = np.flatnonzero(space["tag"] == c)
            at = space["emb"][rng.choice(mine)]
        else:
            at = space["means"][c]
        # Measured on this set, the category means sit 0.43 to 0.66 apart in units of the spread
        # WITHIN a category: two centres are half as far from each other as a crop is from its own.
        # Raw means therefore make knobs that turn without pointing anywhere. Pushing away from the
        # overall centre keeps the direction that distinguishes a category and drops what every
        # category shares, which is most of the vector.
        total += w * (centre + push * (at - centre))
    size = np.linalg.norm(total)
    return total * (float(space["norm"]) / size) if size > 1e-6 else total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--image", required=True, help="The picture to turn.")
    ap.add_argument("--mix", nargs="*", default=[], help="`catN=weight`, repeatable.")
    ap.add_argument("--sweep", help="Draw one category from nothing to full instead.")
    ap.add_argument("--every", action="store_true", help="Draw a row for every category.")
    ap.add_argument("--from", dest="kind", choices=["mean", "crop"], default="mean")
    ap.add_argument("--push", type=float, default=1.0,
                    help="Exaggerate what separates a category from the rest. 1 is the raw mean.")
    ap.add_argument("--scale", type=float, default=1.0, help="How hard the texture pulls.")
    ap.add_argument("--strength", type=float, default=0.6, help="How much of the source is redrawn.")
    ap.add_argument("--lora", help="A LoRA folder to put the look underneath.")
    ap.add_argument("--style", type=float, default=1.0)
    ap.add_argument("--prompt", default="", help="Empty lets the space speak alone.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from diffusers import AutoPipelineForImage2Image

    space = np.load(args.space, allow_pickle=True)
    names = [str(n) for n in space["names"]]
    where = {n: i for i, n in enumerate(names)}

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        PRESET["model"], torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(args.device)
    pipe.load_ip_adapter(PRESET["repo"], subfolder=PRESET["subfolder"], weight_name=PRESET["weight"],
                         image_encoder_folder=None)
    pipe.set_ip_adapter_scale(args.scale)
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=args.style)
    pipe.set_progress_bar_config(disable=True)

    source = fit(args.image, args.size)

    def turned(weights, scale=None):
        if scale is not None:
            pipe.set_ip_adapter_scale(scale)
        vec = vector(space, weights, args.kind, args.seed, args.push)
        emb = torch.tensor(vec, dtype=dtype, device=args.device).view(1, 1, -1)
        if args.guidance > 1.0:
            emb = torch.cat([torch.zeros_like(emb), emb])
        with torch.no_grad():
            return pipe(
                prompt=args.prompt, image=source, strength=args.strength,
                ip_adapter_image_embeds=[emb], num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(args.device).manual_seed(args.seed),
            ).images[0]

    rows, labels, heads = [], [], ["the picture"]
    picks = [0.3, 0.6, 0.9, 1.2]
    if args.sweep:
        c = where[args.sweep]
        rows.append([source] + [turned({c: 1.0}, s) for s in picks])
        labels.append(f"{args.sweep}")
        heads += [f"scale {s}" for s in picks]
        print(f"  swept {args.sweep}")
    elif args.every:
        heads += [f"scale {s}" for s in picks]
        for n in names:
            rows.append([source] + [turned({where[n]: 1.0}, s) for s in picks])
            labels.append(n)
            print(f"  turned {n}")
    else:
        weights = {}
        for item in args.mix:
            name, _, w = item.partition("=")
            weights[where[name]] = float(w or 1.0)
        rows.append([source, turned(weights)])
        labels.append(", ".join(args.mix))
        heads.append("turned")
        print(f"  mixed {args.mix}")

    gut, cell = 190, args.size
    sheet = Image.new("RGB", (gut + len(heads) * cell, 30 + len(rows) * cell), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    for i, h in enumerate(heads):
        pen.text((gut + i * cell + 8, 10), h, fill=(214, 210, 202))
    for r, (row, name) in enumerate(zip(rows, labels)):
        pen.text((8, 30 + r * cell + cell // 2), name[:24], fill=(176, 174, 168))
        for c, img in enumerate(row):
            sheet.paste(img, (gut + c * cell, 30 + r * cell))
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
