"""Describe every image by what is in it, so a prompt can ask for it.

    python caption.py --images ~/pictures/set-by-category --detail 3 --sample 12
    python caption.py --images ~/pictures/set-by-category --detail 3 --write

A category word is a blunt caption. One archive's `rock` held both bare grey faces and stone
encrusted with orange lichen, and a LoRA trained on that word learnt the average of them: ask it for
rock and it draws the base model's idea of rock, because that is the strongest thing the word means.
Asked instead for "lichen and mineral stains on wet rock, orange and teal", the same weights drew
what the photographs actually hold. The caption was the bottleneck, not the training.

So each image gets its own. CLIP scores it against vocabularies kept apart — material, surface,
condition — so a material only ever competes with other materials. Colour does not go through CLIP
at all: naming a colour is the one thing pixels answer better, so the dominant colours are clustered
in CIELAB and matched to a named palette by perceptual distance.

Written as a `.txt` beside each image, which is what `lora.py` reads in preference to its template.
Nothing else in the folder is touched, and `--write` is required to write anything.
"""

import argparse
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
CLIP = "openai/clip-vit-large-patch14"

# Scored apart, so a material competes only with materials. Each is (template, terms).
AXES = {
    "material": (
        "a close-up photograph of {}",
        [
            "still water", "rippled water", "a breaking wave", "sea foam", "froth and bubbles",
            "a shallow stream over stones", "a reflection on water", "ice", "frost", "melting snow",
            "bare rock", "a cracked rock face", "granite", "limestone", "slate", "sandstone", "basalt",
            "marble", "quartz", "a mineral vein", "pebbles", "gravel", "shingle", "sand", "silt", "dried mud",
            "clay", "moss", "lichen on stone", "algae", "seaweed", "a barnacle-covered rock", "shells",
            "tree bark", "birch bark", "pine bark", "driftwood", "weathered timber", "a cut log",
            "fallen leaves", "foliage", "grass", "roots", "lace-like branches",
            "rust on iron", "corroded metal", "oxidised copper", "peeling paint", "flaking enamel",
            "cracked plaster", "weathered concrete", "brick", "ceramic tile", "frosted glass",
            "torn paper", "worn fabric", "rope", "fishing net", "tangled cable",
            "smoke", "mist", "cloud", "a sunlit surface", "a sheet of light",
        ],
    ),
    "surface": (
        "a {} surface",
        [
            "cracked", "deeply fissured", "finely veined", "layered", "striated", "banded",
            "mottled", "speckled", "freckled", "pitted", "pockmarked", "eroded", "weathered",
            "flaking", "peeling", "blistered", "encrusted", "crystalline", "granular", "grainy",
            "pebbled", "smooth", "polished", "glassy", "rippled", "undulating", "swirling",
            "marbled", "braided", "fibrous", "stringy", "dendritic", "branching", "web-like",
            "honeycombed", "cellular", "bubbled", "foamy", "furrowed", "ridged", "scaled",
        ],
    ),
    "condition": (
        "{}",
        [
            "wet and glistening", "soaked", "submerged under clear water", "bone dry", "dusty",
            "sunlit with hard shadows", "in flat overcast light", "backlit", "in deep shade",
            "glossy and reflective", "matte", "frosted", "sharply detailed", "softly blurred",
        ],
    ),
}

# Named by perceptual distance in CIELAB, because a colour word is the one thing pixels answer better.
PALETTE = {
    "black": (16, 16, 18), "charcoal": (54, 54, 58), "slate grey": (98, 104, 110),
    "ash grey": (150, 150, 148), "pale grey": (196, 197, 196), "bone white": (232, 228, 218),
    "cream": (238, 226, 198), "sand": (214, 192, 152), "khaki": (176, 160, 116),
    "ochre": (190, 142, 62), "mustard": (206, 166, 44), "gold": (212, 175, 55),
    "amber": (196, 128, 36), "copper": (168, 94, 52), "rust": (150, 76, 38),
    "burnt orange": (186, 86, 30), "terracotta": (178, 102, 76), "brick red": (142, 58, 44),
    "crimson": (156, 34, 48), "maroon": (104, 34, 44), "plum": (104, 56, 88),
    "mauve": (150, 118, 142), "rose": (198, 136, 142), "blush": (222, 180, 174),
    "lavender": (176, 168, 206), "violet": (112, 84, 156), "indigo": (62, 60, 124),
    "navy": (34, 46, 80), "cobalt": (44, 86, 158), "steel blue": (92, 124, 156),
    "pale blue": (168, 196, 214), "teal": (36, 118, 124), "turquoise": (56, 166, 165),
    "aquamarine": (134, 204, 196), "sea green": (72, 140, 118), "emerald": (44, 124, 84),
    "moss green": (96, 118, 68), "olive": (118, 118, 62), "sage": (158, 170, 138),
    "acid green": (160, 194, 48), "lime": (190, 210, 86), "umber": (96, 72, 52),
    "chocolate": (72, 52, 40), "warm brown": (122, 88, 60),
    # Weathered things are mostly MUTED. A palette of saturated hues around a grey axis sends every
    # dusty, chalky, sun-bleached surface to "ash grey" or "chocolate" — a tan stream read as ash
    # grey and beige paper as chocolate until these existed. This band is where the archive lives.
    "taupe": (138, 124, 110), "greige": (166, 158, 144), "stone": (182, 176, 162),
    "oatmeal": (206, 196, 176), "putty": (190, 178, 158), "fawn": (190, 164, 134),
    "driftwood": (156, 146, 132), "mushroom": (146, 132, 124), "pewter": (124, 124, 126),
    "dusty rose": (168, 128, 124), "clay": (158, 110, 90), "tobacco": (134, 104, 68),
    "bronze": (140, 110, 70), "pale olive": (160, 158, 120), "olive drab": (108, 106, 78),
    "celadon": (176, 192, 170), "eucalyptus": (140, 158, 142), "seafoam": (186, 206, 196),
    "dusty blue": (134, 156, 172), "ice blue": (198, 214, 220), "muted teal": (88, 128, 128),
    "charcoal blue": (62, 72, 84), "deep teal": (40, 72, 78),
}


def to_lab(rgb):
    """sRGB in [0,1] to CIELAB, so 'nearest colour' means what the eye means by it."""
    rgb = np.asarray(rgb, np.float64)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


NAMED = {name: to_lab(np.array(v) / 255.0) for name, v in PALETTE.items()}


def colours_of(tiles, keep=2, floor=0.14):
    """The named colours an image is actually made of, by clustering its pixels in CIELAB."""
    px = np.concatenate([np.asarray(t.resize((48, 48), Image.BICUBIC), np.float64).reshape(-1, 3) / 255.0 for t in tiles])
    lab = to_lab(px)
    # The darkest quarter is dropped before clustering. Shadow carries little colour and the eye
    # discounts it, but it is often the largest region in a photograph, and it was naming vivid
    # orange rock "charcoal".
    lab = lab[lab[:, 0] >= np.percentile(lab[:, 0], 25)]
    rng = np.random.default_rng(0)
    centres = lab[rng.choice(len(lab), 6, replace=False)]
    for _ in range(25):
        owner = np.argmin(((lab[:, None] - centres[None]) ** 2).sum(-1), 1)
        moved = np.stack([lab[owner == i].mean(0) if (owner == i).any() else centres[i] for i in range(len(centres))])
        if np.allclose(moved, centres, atol=1e-3):
            break
        centres = moved
    share = np.bincount(owner, minlength=len(centres)) / len(lab)
    chroma = np.linalg.norm(centres[:, 1:], axis=1)
    # Neither the biggest cluster nor the most colourful one is what a person names. The biggest is
    # usually shadow — it called bright green foliage "charcoal" — and the most colourful can be a
    # speck. Scored together, what wins is the colour that actually carries the picture.
    worth = share * (chroma + 6.0)
    name_of = lambda i: min(NAMED, key=lambda n: float(((centres[i] - NAMED[n]) ** 2).sum()))
    order = list(np.argsort(-worth))

    # The tone the picture sits in, and then its ACCENT: the most saturated colour that is present
    # at all. A dark rock with rust-orange lichen averages to charcoal, which is true and useless —
    # the orange is what makes it worth photographing and what a prompt has to be told.
    # Weighted by chroma, not by area alone: naming the largest cluster sent golden water to
    # "charcoal" and foliage to "olive drab", because a dull background outvotes the subject.
    out = [name_of(int(order[0]))]
    accent = [i for i in range(len(centres)) if share[i] >= 0.05 and name_of(int(i)) not in out]
    if accent:
        best = max(accent, key=lambda i: chroma[i])
        if chroma[best] >= 14.0:
            out.append(name_of(int(best)))
    for i in order:
        if len(out) >= keep:
            break
        name = name_of(int(i))
        if name not in out and share[i] >= floor / 2:
            out.append(name)
    return out


def tiles_of(path, detail, count, side=224, seed=0):
    """What the trainer sees: squares a `detail`-th of the frame, so the caption describes that."""
    rng = random.Random(seed)
    with Image.open(path) as im:
        im.draft("RGB", (im.width // 2, im.height // 2))
        im = im.convert("RGB")
    piece = max(1, round(min(im.size) / detail))
    out = []
    for _ in range(count):
        left, top = rng.randint(0, im.width - piece), rng.randint(0, im.height - piece)
        out.append(im.crop((left, top, left + piece, top + piece)).resize((side, side), Image.BICUBIC))
    return out


@torch.no_grad()
def text_banks(model, processor, device):
    """One unit matrix of text features per axis, each scored on its own."""
    banks = {}
    for axis, (template, terms) in AXES.items():
        tokens = processor.tokenizer([template.format(t) for t in terms], padding=True, return_tensors="pt").to(device)
        got = model.get_text_features(**tokens)
        feats = got if torch.is_tensor(got) else got.pooler_output
        banks[axis] = F.normalize(feats.float(), dim=1)
    return banks


def phrase(material, surfaces, colours, condition, trigger):
    """The caption, as a photographer would say it rather than as a list of tags."""
    bits = [f"a close-up photograph of {material}"]
    if surfaces:
        bits.append(" and ".join(surfaces))
    if colours:
        bits.append("in " + " and ".join(colours))
    if condition:
        bits.append(condition)
    if trigger:
        bits.append(trigger)
    return ", ".join(bits)


def contact(pairs, path, side=256, lines=5, across=4):
    """The sampled crops with their captions under them.

    A caption is a claim about a photograph, and the only way to know whether it is true is to
    read it next to the picture.
    """
    try:
        face = ImageFont.load_default(size=13)
    except TypeError:
        face = ImageFont.load_default()
    label = 14 * lines + 10
    down = -(-len(pairs) // across)
    canvas = Image.new("RGB", (across * side, down * (side + label)), (16, 16, 18))
    pen = ImageDraw.Draw(canvas)
    for i, (tile, text) in enumerate(pairs):
        x, y = (i % across) * side, (i // across) * (side + label)
        canvas.paste(tile.resize((side, side), Image.LANCZOS), (x, y))
        row, rows = "", []
        for w in text.split():
            if len(row) + len(w) + 1 > 38:
                rows.append(row); row = w
            else:
                row = f"{row} {w}".strip()
        rows.append(row)
        for r, line in enumerate(rows[:lines]):
            pen.text((x + 5, y + side + 4 + r * 14), line, fill=(222, 222, 222), font=face)
    canvas.save(path)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of images, read recursively.")
    ap.add_argument("--trigger", default="in goofi style", help="The word tying a prompt to this set.")
    ap.add_argument("--detail", type=float, default=3.0, help="The trainer's own, so a caption describes its crops.")
    ap.add_argument("--crops", type=int, default=4, help="Crops averaged per image.")
    ap.add_argument("--surfaces", type=int, default=2, help="Surface words to keep.")
    ap.add_argument("--colours", type=int, default=2, help="Colour words to keep.")
    ap.add_argument("--floor", type=float, default=0.18, help="A term is only used if it takes this much of its axis.")
    ap.add_argument("--sample", type=int, default=0, help="Show this many captions and write nothing.")
    ap.add_argument("--contact", help="With --sample: a sheet of those crops with their captions under them.")
    ap.add_argument("--write", action="store_true", help="Actually write the `.txt` files.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.sample and not args.write:
        raise SystemExit("nothing to do: pass --sample N to look first, or --write to write")

    from transformers import CLIPModel, CLIPProcessor

    files = sorted(p for p in pathlib.Path(args.images).expanduser().rglob("*") if p.suffix.lower() in SUFFIXES)
    if not files:
        raise SystemExit(f"no images under {args.images}")
    if args.sample:
        files = random.Random(0).sample(files, min(args.sample, len(files)))

    model = CLIPModel.from_pretrained(CLIP, local_files_only=True).to(args.device).half().eval()
    processor = CLIPProcessor.from_pretrained(CLIP, local_files_only=True)
    banks = text_banks(model, processor, args.device)
    mean = torch.tensor(processor.image_processor.image_mean, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device=args.device).view(1, 3, 1, 1)

    written, shown = 0, []
    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        groups = [tiles_of(p, args.detail, args.crops) for p in chunk]
        flat = [t for g in groups for t in g]
        x = torch.from_numpy(np.stack([np.asarray(t) for t in flat])).to(args.device)
        x = ((x.permute(0, 3, 1, 2).float() / 255.0) - mean) / std
        with torch.no_grad():
            got = model.get_image_features(pixel_values=x.half())
            feats = got if torch.is_tensor(got) else got.pooler_output
        feats = F.normalize(feats.float(), dim=1).view(len(chunk), args.crops, -1).mean(1)
        feats = F.normalize(feats, dim=1)

        for i, path in enumerate(chunk):
            said = {}
            for axis, bank in banks.items():
                share = (100.0 * feats[i] @ bank.T).softmax(0)
                order = torch.argsort(share, descending=True)
                terms = AXES[axis][1]
                want = args.surfaces if axis == "surface" else 1
                said[axis] = [terms[int(k)] for k in order[:want] if float(share[k]) >= args.floor] or [terms[int(order[0])]]
            text = phrase(
                said["material"][0],
                said["surface"],
                colours_of(groups[i], args.colours),
                said["condition"][0],
                args.trigger,
            )
            if args.write:
                path.with_suffix(".txt").write_text(text, encoding="utf-8")
                written += 1
            else:
                print(f"  {path.parent.name}/{path.name}\n    {text}")
                shown.append((groups[i][0], text))
        if args.write and (start // args.batch) % 10 == 0:
            print(f"  captioned {min(start + args.batch, len(files))}/{len(files)}", flush=True)

    if args.write:
        print(f"wrote {written} captions beside the images under {args.images}")
    else:
        if args.contact and shown:
            contact(shown, args.contact)
        print(f"\n  nothing written; pass --write when these read right")


if __name__ == "__main__":
    main()
