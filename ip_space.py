"""A space to move through, built from your images and fed to a diffusion model as an IP-Adapter.

An IP-Adapter lets an image steer a diffusion model the way a prompt does: an image encoder turns
the picture into one vector, and the adapter feeds that vector into every cross-attention layer
beside the text. Nothing about that vector has to come from one picture. This embeds crops of every
image in a set, and treats the cloud of vectors as the space: a mean per category, and principal
axes across all of them. A point anywhere in it is a prompt, and moving the point is a walk.

    python ip_space.py --images ~/pictures/set-by-category --detail 3 --out set-ip.npz
    python ip_space.py --space set-ip.npz --sheet set-ip.png
    python ip_space.py --space set-ip.npz --sheet set-ip-lora.png --lora set-lora

No training: the adapter and the base are pretrained, and your images are only ever its INPUT.
`--lora` puts a LoRA from `lora.py` under it, so the space picks between your categories and the
LoRA holds the look.

The vectors have to come from the encoder the adapter was trained with. `categories.py` embeds with
a different CLIP, so its cache cannot stand in for this one — the numbers would be in the wrong
space and the adapter would read them as noise.

The build prints how far apart the category means sit against the spread inside each category,
pair by pair, which is the same honest scale `separation.py` reads a GAN on. A pair near zero is one
this encoder cannot tell apart, and no amount of steering between them will look like steering.
"""

import argparse
import itertools
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

from lora import images, square

# A base, the adapter made for it, and how to draw from it quickly. `encoder` is where the adapter's
# own image encoder lives, relative to `repo`.
PRESETS = {
    "sdxl-turbo": dict(
        model="stabilityai/sdxl-turbo",
        repo="h94/IP-Adapter",
        subfolder="sdxl_models",
        weight="ip-adapter_sdxl.safetensors",
        encoder="sdxl_models/image_encoder",
        fast=None,
        steps=2,
        guidance=0.0,
        size=512,
    ),
    "sd15": dict(
        model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        repo="h94/IP-Adapter",
        subfolder="models",
        weight="ip-adapter_sd15.safetensors",
        encoder="models/image_encoder",
        fast="latent-consistency/lcm-lora-sdv1-5",
        steps=4,
        guidance=1.0,
        size=512,
    ),
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preset(args):
    """The preset, with anything given on the command line in its place."""
    got = dict(PRESETS[args.preset])
    for key in got:
        if getattr(args, key, None) is not None:
            got[key] = getattr(args, key)
    if got["fast"] == "none":
        got["fast"] = None
    return got


def build(args, p, device):
    from transformers import CLIPVisionModelWithProjection

    root = pathlib.Path(p["repo"]).expanduser()
    where = (str(root / p["encoder"]), {}) if root.is_dir() else (p["repo"], {"subfolder": p["encoder"]})
    encoder = CLIPVisionModelWithProjection.from_pretrained(where[0], **where[1]).to(device).eval()
    side = encoder.config.image_size
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    files = images(args.images)
    names = sorted({f.parent.name for f in files})
    print(f"{len(files)} images in {len(names)} categories, {args.crops} crops each seeing 1/{args.detail:g} of a frame")
    vecs, owner, tag = [], [], []
    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        crops, idx = [], []
        for i, f in enumerate(chunk):
            try:
                crops += [square(f, side, args.detail) for _ in range(args.crops)]
                idx += [start + i] * args.crops
            except OSError as e:
                print(f"  skipped {f.name}: {e}")
        if not crops:
            continue
        x = (torch.stack(crops).to(device) + 1) / 2
        with torch.no_grad():
            vecs.append(encoder(pixel_values=(x - mean) / std).image_embeds.float().cpu())
        owner += idx
        tag += [names.index(files[i].parent.name) for i in idx]
        print(f"  {min(start + args.batch, len(files))}/{len(files)}", end="\r", flush=True)
    emb = torch.cat(vecs).numpy()
    owner, tag = np.array(owner), np.array(tag)

    centre = emb.mean(0)
    _, s, vt = np.linalg.svd(emb - centre, full_matrices=False)
    keep = min(args.axes, len(s))
    spread = s[:keep] / np.sqrt(len(emb) - 1)
    share = s**2 / (s**2).sum()
    means = np.stack([emb[tag == c].mean(0) for c in range(len(names))])
    norm = float(np.median(np.linalg.norm(emb, axis=1)))

    np.savez(
        args.out,
        emb=emb.astype(np.float32),
        owner=owner,
        tag=tag,
        files=np.array([str(f) for f in files]),
        names=np.array(names),
        centre=centre.astype(np.float32),
        axes=vt[:keep].astype(np.float32),
        spread=spread.astype(np.float32),
        means=means.astype(np.float32),
        norm=np.float32(norm),
        preset=np.array(args.preset),
    )
    print(f"\nwrote {args.out}: {len(emb)} vectors of {emb.shape[1]}, typical length {norm:.2f}")
    print(f"  variance on the first axes: {', '.join(f'{q:.2f}' for q in share[:8])}")
    print(f"  axes to 90% of it: {int(np.searchsorted(np.cumsum(share), 0.9)) + 1}")
    report(emb, tag, names, means)


def report(emb, tag, names, means):
    """How far apart the category means are, against how far a crop strays from its own mean."""
    if len(names) < 2:
        return
    within = np.mean([np.linalg.norm(emb[tag == c] - means[c], axis=1).mean() for c in range(len(names))])
    nearest = np.argmin(((emb[:, None] - means[None]) ** 2).sum(-1), 1)
    print(f"\n  {'category':<16}{'crops':>7}{'nearest own mean':>18}")
    for c, n in enumerate(names):
        mine = tag == c
        print(f"  {n:<16}{mine.sum():>7}{(nearest[mine] == c).mean():>18.0%}")
    pairs = sorted((np.linalg.norm(means[a] - means[b]) / within, names[a], names[b]) for a, b in itertools.combinations(range(len(names)), 2))
    print("\n  apart, in units of the spread within a category — read the smallest, not the mean:")
    for gap, a, b in pairs[: min(len(pairs), 8)]:
        print(f"    {gap:6.2f}  {a} / {b}")


def pipeline(args, p, device):
    from diffusers import AutoPipelineForText2Image, LCMScheduler

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(p["model"], torch_dtype=dtype, safety_checker=None).to(device)
    root = pathlib.Path(p["repo"]).expanduser()
    pipe.load_ip_adapter(
        str(root) if root.is_dir() else p["repo"],
        subfolder=p["subfolder"],
        weight_name=p["weight"],
        image_encoder_folder=None,  # the space arrives as vectors; the encoder is not needed to draw
    )
    if args.style_only:
        # InstantStyle: the adapter speaks only to the block that carries style in SDXL, so a
        # reference's layout does not come along with its surface.
        pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, args.scale, 0.0]}})
    else:
        pipe.set_ip_adapter_scale(args.scale)
    adapters, weights = [], []
    if p["fast"]:
        pipe.load_lora_weights(p["fast"], adapter_name="fast")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        adapters.append("fast")
        weights.append(1.0)
    if args.lora:
        pipe.load_lora_weights(args.lora, adapter_name="look")
        adapters.append("look")
        weights.append(args.style)
    if adapters:
        pipe.set_adapters(adapters, weights)
    pipe.set_progress_bar_config(disable=True)
    return pipe, dtype


def point(space, category=None, coords=()):
    """A category's mean (or the centre), moved along the principal axes, at the length real crops have.

    Rescaled because an average is shorter than what it averages: the mean of a category sits
    nearer the origin than any crop in it, and the adapter reads a short vector as a faint one.
    """
    v = space["centre"] if category is None else space["means"][category]
    v = v + sum(c * space["spread"][i] * space["axes"][i] for i, c in enumerate(coords))
    return v * (space["norm"] / np.linalg.norm(v))


def slerp(a, b, t):
    """Along the sphere rather than through it, as `play.py` morphs: a straight line dips towards the mean."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    ua, ub = a / na, b / nb
    angle = np.arccos(np.clip(ua @ ub, -1.0, 1.0))
    if angle < 1e-4:
        return a + t * (b - a)
    return (np.sin((1 - t) * angle) * ua + np.sin(t * angle) * ub) / np.sin(angle) * ((1 - t) * na + t * nb)


def draw(pipe, dtype, p, args, vec, seed, device):
    emb = torch.tensor(vec, dtype=dtype, device=device).view(1, 1, -1)
    if p["guidance"] > 1.0:
        emb = torch.cat([torch.zeros_like(emb), emb])  # the unconditional half, first, as the pipeline splits it
    return pipe(
        prompt=args.prompt,
        ip_adapter_image_embeds=[emb],
        num_inference_steps=p["steps"],
        guidance_scale=p["guidance"],
        height=p["size"],
        width=p["size"],
        generator=torch.Generator(device).manual_seed(seed),
    ).images[0]


def render(args, p, device):
    space = dict(np.load(args.space))
    names = [str(n) for n in space["names"]]
    pipe, dtype = pipeline(args, p, device)
    cols = args.cols
    rows = []
    for c, n in enumerate(names):
        rows.append((f"{n}: its mean, {cols} seeds", [point(space, c) for _ in range(cols)], list(range(cols))))
    for a, b in zip(range(len(names)), range(1, len(names))):
        ends = point(space, a), point(space, b)
        rows.append((f"{names[a]} to {names[b]}, one seed", [slerp(*ends, t) for t in np.linspace(0, 1, cols)], [0] * cols))
    for k in range(min(args.sweep, len(space["spread"]))):
        rows.append((f"axis {k}, -2 to +2 spreads", [point(space, None, [0] * k + [t]) for t in np.linspace(-2, 2, cols)], [0] * cols))

    side, label = p["size"], 22
    canvas = Image.new("RGB", (cols * side, len(rows) * (side + label)), (18, 18, 20))
    pen = ImageDraw.Draw(canvas)
    for r, (title, vecs, seeds) in enumerate(rows):
        y = r * (side + label)
        pen.text((6, y + 5), title, fill=(220, 220, 220))
        for i, (vec, seed) in enumerate(zip(vecs, seeds)):
            canvas.paste(draw(pipe, dtype, p, args, vec, seed, device).resize((side, side)), (i * side, y + label))
        print(f"  {title}", flush=True)
    canvas.save(args.sheet)
    print(f"wrote {args.sheet}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", help="Build the space from this folder; categories are its subfolders.")
    ap.add_argument("--out", default="ip-space.npz", help="Where the built space goes.")
    ap.add_argument("--space", help="A built space to draw from.")
    ap.add_argument("--sheet", help="Draw a sheet of the space here: category means, morphs, axis sweeps.")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Default: sdxl-turbo, or whatever the space was built for.")
    ap.add_argument("--model", help="Override the preset's base.")
    ap.add_argument("--repo", help="Override where the adapter lives: a hub id or a local folder.")
    ap.add_argument("--subfolder", help="Override the adapter's subfolder.")
    ap.add_argument("--weight", help="Override the adapter's weight file.")
    ap.add_argument("--encoder", help="Override the image encoder's folder, relative to the repo.")
    ap.add_argument("--fast", help="Override the preset's speed-up LoRA; 'none' for none.")
    ap.add_argument("--steps", type=int, help="Override the preset's denoising steps.")
    ap.add_argument("--guidance", type=float, help="Override the preset's guidance.")
    ap.add_argument("--size", type=int, help="Override the preset's drawing size.")
    ap.add_argument("--detail", type=float, default=3.0, help="How much of the frame a crop sees, as in lora.py.")
    ap.add_argument("--crops", type=int, default=8, help="Crops per image: the space is of textures, not of frames.")
    ap.add_argument("--axes", type=int, default=32, help="Principal axes to keep.")
    ap.add_argument("--batch", type=int, default=8, help="Images per encoder pass.")
    ap.add_argument("--prompt", default="", help="Text beside the image vector. Empty lets the space speak alone.")
    ap.add_argument("--scale", type=float, default=1.0, help="How hard the adapter pulls.")
    ap.add_argument("--style-only", action="store_true", help="SDXL only: the adapter steers style, not layout.")
    ap.add_argument("--lora", help="A LoRA folder from lora.py to draw under the space.")
    ap.add_argument("--style", type=float, default=1.0, help="How hard the LoRA pulls.")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--sweep", type=int, default=4, help="Principal axes to sweep on the sheet.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not (args.images or (args.space and args.sheet)):
        raise SystemExit("give --images to build a space, or --space and --sheet to draw one")

    if args.preset is None:
        # Drawn with the adapter it was built for: another encoder's vectors are another space.
        built = args.space and not args.images and pathlib.Path(args.space).is_file()
        args.preset = str(np.load(args.space)["preset"]) if built else "sdxl-turbo"
    device = torch.device(args.device)
    p = preset(args)
    if args.images:
        build(args, p, device)
        args.space = args.space or args.out
    if args.sheet:
        render(args, p, device)


if __name__ == "__main__":
    main()
