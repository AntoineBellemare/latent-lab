"""Knobs built from what the captions say, rather than from which folder an image sits in.

    python terms.py --space texture-ip-sd15.npz --list
    python terms.py --space texture-ip-sd15.npz --image photo.jpg --term encrusted glassy --out sheet.png

The seven categories were found by clustering, and the clustering says they are not really there:
across every count from 3 to 16 the silhouette stays near 0.16, which is what a continuum looks
like, not what clusters look like. Forcing seven boundaries onto that is a choice, and every knob
built on a category inherits it.

A caption does not need boundaries. Each image already says what it is, how its surface behaves,
what light it sits in and what colours it holds, so a term like `encrusted` or `teal` names a set of
images directly, and the mean of their crops is a direction in the same space the category means
live in. There are hundreds of such terms instead of seven, they overlap on purpose, and they are
words a person already has in mind before they start.

Same two cautions as `knobs.py`: the vector is a direction, so intensity is `--scale`, and a raw
mean points almost nowhere until it is pushed away from the centre of the space.
"""

import argparse
import collections
import pathlib
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

from knobs import PRESET, fit

FIELDS = {
    "material": r"photograph of ([^,]+),",
    "surface": r"photograph of [^,]+, ([^,]+), in ",
    "colour": r", in ([a-z ]+?) and ([a-z ]+?),",
    "condition": r", ([^,]+), in goofi style",
}


def captions(space):
    """Every image's caption, read from the `.txt` that `caption.py` wrote beside it."""
    out = []
    for f in space["files"]:
        beside = pathlib.Path(str(f)).with_suffix(".txt")
        out.append(beside.read_text(encoding="utf-8").strip() if beside.is_file() else "")
    return out


def index(space):
    """term -> the file indices whose caption carries it, for every field."""
    found = collections.defaultdict(list)
    for i, text in enumerate(captions(space)):
        for field, pattern in FIELDS.items():
            if m := re.search(pattern, text):
                for part in m.groups():
                    for word in (part.split(" and ") if field == "surface" else [part]):
                        found[word.strip()].append(i)
    return found


def term_vector(space, terms, found, push=3.0):
    """The mean of the crops of every image carrying these terms, pushed off the centre."""
    centre = space["centre"]
    # The deviations are averaged and the centre added once. Adding `centre + push * deviation` per
    # term and summing gives the same direction — the two differ by the factor the rescale divides
    # out — but this way says what is meant: a mix is one point, not a pile of points.
    #
    # It does NOT make a mix as distinct as a single term, and nothing here can. Averaging terms
    # lands between them, which is nearer the centre of the space than any of them: measured, the
    # ten presets sit 16.8 apart where single terms sit 21.5. A mix therefore wants a larger push
    # than a term does.
    total = np.zeros(space["emb"].shape[1], np.float32)
    carried = 0.0
    for word, weight in terms.items():
        files = found.get(word, [])
        if not files:
            # Dropped rather than fatal: a preset written against one set is still mostly right on
            # the next one, and losing a term should cost that term, not the whole mix.
            print(f"  (no image is captioned {word!r} — dropped)")
            continue
        keep = np.isin(space["owner"], files)
        at = space["emb"][keep].mean(0)
        total += weight * (at - centre)
        carried += weight
    if carried <= 0:
        return centre * (float(space["norm"]) / np.linalg.norm(centre))
    total = centre + push * (total / carried)
    size = np.linalg.norm(total)
    return total * (float(space["norm"]) / size) if size > 1e-6 else total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--list", action="store_true", help="Every term, with how many images carry it.")
    ap.add_argument("--least", type=int, default=25, help="With --list, the fewest images to report.")
    ap.add_argument("--image", help="The picture to turn.")
    ap.add_argument("--term", nargs="*", default=[], help="One row each; `word=weight` to mix.")
    ap.add_argument("--preset", nargs="*", default=[], help="Named mixes from --presets; `all` for every one.")
    ap.add_argument("--presets", default="presets.json")
    ap.add_argument("--push", type=float, default=3.0)
    ap.add_argument("--lora", help="A LoRA folder to put the look under the space.")
    ap.add_argument("--style", type=float, default=1.0, help="How hard the LoRA pulls.")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--scales", type=float, nargs="*", default=[0.4, 0.8, 1.2])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    space = np.load(args.space, allow_pickle=True)
    found = index(space)

    if args.list:
        for field, pattern in FIELDS.items():
            mine = {w: v for w, v in found.items() if re.search(pattern, f"a close-up photograph of x, y, in {w} and z, {w}, in goofi style") or True}
            print(f"\n{field}:")
        for word, files in sorted(found.items(), key=lambda kv: -len(kv[1])):
            if len(files) >= args.least:
                print(f"  {len(files):>5}  {word}")
        return

    if not (args.image and args.out and (args.term or args.preset)):
        raise SystemExit("give --image, --out and one of --term / --preset, or --list")

    from diffusers import AutoPipelineForImage2Image

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        PRESET["model"], torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(args.device)
    pipe.load_ip_adapter(PRESET["repo"], subfolder=PRESET["subfolder"], weight_name=PRESET["weight"],
                         image_encoder_folder=None)
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=args.style)
    pipe.set_progress_bar_config(disable=True)
    source = fit(args.image, args.size)

    book = {}
    if args.preset:
        import json
        book = {k: v for k, v in json.loads(pathlib.Path(args.presets).read_text(encoding="utf-8")).items()
                if not k.startswith("_")}
        wanted = list(book) if args.preset == ["all"] else args.preset
        for name in wanted:
            if name not in book:
                raise SystemExit(f"no preset named {name!r}; have: {', '.join(book)}")
    else:
        wanted = []

    rows, labels = [], []
    for item in wanted + args.term:
        if item in book:
            mix = dict(book[item])
        else:
            mix = {}
            for part in item.split("+"):
                word, _, w = part.partition("=")
                mix[word.strip()] = float(w or 1.0)
        vec = term_vector(space, mix, found, args.push)
        emb = torch.tensor(vec, dtype=dtype, device=args.device).view(1, 1, -1)
        if args.guidance > 1.0:
            emb = torch.cat([torch.zeros_like(emb), emb])
        row = [source]
        for scale in args.scales:
            pipe.set_ip_adapter_scale(scale)
            with torch.no_grad():
                row.append(pipe(prompt="", image=source, strength=args.strength,
                                ip_adapter_image_embeds=[emb], num_inference_steps=args.steps,
                                guidance_scale=args.guidance,
                                generator=torch.Generator(args.device).manual_seed(args.seed)).images[0])
        rows.append(row)
        n = sum(len(found.get(w, [])) for w in mix)
        labels.append(f"{item}  ({n} images)")
        print(f"  turned {item}")

    gut = 330
    sheet = Image.new("RGB", (gut + (1 + len(args.scales)) * args.size, 30 + len(rows) * args.size), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    pen.text((gut + 8, 10), "the picture", fill=(214, 210, 202))
    for i, s in enumerate(args.scales):
        pen.text((gut + (i + 1) * args.size + 8, 10), f"scale {s}", fill=(214, 210, 202))
    for r, (row, name) in enumerate(zip(rows, labels)):
        pen.text((10, 30 + r * args.size + args.size // 2), name, fill=(214, 210, 202))
        for c, img in enumerate(row):
            sheet.paste(img, (gut + c * args.size, 30 + r * args.size))
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
