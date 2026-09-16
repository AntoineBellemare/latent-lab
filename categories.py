"""Find the categories a folder of images falls into, from the images rather than from a guess.

`cluster.py` groups on colour, which is right for a style target and wrong for a material: dark
water and dark rock are one colour. This embeds every image with two unrelated models — CLIP, which
learnt what a surface is CALLED, and DINOv2, which learnt what it LOOKS like and never saw a word —
and asks how many groups the data holds rather than taking a number on trust.

    python categories.py --images ~/pictures/set --out ~/pictures/set-categories

Three numbers decide it, for every count of groups tried. Silhouette: are the groups apart.
Stability: do runs from different starts find the same groups. Agreement: do the two models find
the SAME groups — a split that two representations with nothing in common both reproduce is in the
pictures, not in either model. Each group is then named by asking CLIP which of a list of materials
its members look like; the contact sheets are how to check it was right.

Nothing under --images is written. The result is a CSV of file to category, a contact sheet per
category, and one overview.
"""

import argparse
import concurrent.futures
import csv
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
CLIP = "openai/clip-vit-large-patch14"
DINO = "facebook/dinov2-small"
# Surfaces and materials, not scenes: what is asked of a set of textures.
MATERIALS = [
    "water", "ripples on water", "waves", "foam", "bubbles", "ice", "frost", "snow",
    "rock", "stone", "pebbles", "sand", "soil", "mud", "moss", "lichen", "algae",
    "tree bark", "wood", "leaves", "foliage", "grass", "flowers", "branches",
    "clouds", "sky", "fog", "smoke", "fire",
    "metal", "rust", "concrete", "brick", "tiles", "glass", "plastic", "paint", "paper",
    "fabric", "netting", "cables", "feathers", "fur", "shells",
    "reflections of light", "shadows", "sparkles",
]
PROMPT = "a close-up photograph of {}"


def views(path, detail, side=224):
    """The whole frame, and four pieces the size a trainer at `detail` crops out of it."""
    with Image.open(path) as im:
        im.draft("RGB", (im.width // 2, im.height // 2))
        im = im.convert("RGB")
    w, h = im.size
    short = min(w, h)
    piece = short / detail
    boxes = [(w / 2, h / 2, short)] + [
        (min(max(fx * w, piece / 2), w - piece / 2), min(max(fy * h, piece / 2), h - piece / 2), piece)
        for fx, fy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    ]
    return [
        im.crop((round(x - s / 2), round(y - s / 2), round(x + s / 2), round(y + s / 2))).resize((side, side), Image.BICUBIC)
        for x, y, s in boxes
    ]


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP, local_files_only=True).to(device).half().eval()
    return model, CLIPProcessor.from_pretrained(CLIP, local_files_only=True)


def projected(out):
    """What a CLIP feature call answered: a tensor, or an output holding one, by library version."""
    return out if torch.is_tensor(out) else out.pooler_output


@torch.no_grad()
def embed(files, detail, device, batch=24):
    """One unit vector per image in each model: the mean of its views, each view made unit first."""
    from transformers import AutoImageProcessor, AutoModel

    clip, clip_prep = load_clip(device)
    dino = AutoModel.from_pretrained(DINO, local_files_only=True).to(device).half().eval()
    dino_prep = AutoImageProcessor.from_pretrained(DINO, local_files_only=True)

    def standard(mean, std):
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)
        return lambda x: ((x - m) / s).half()

    for_clip = standard(clip_prep.image_processor.image_mean, clip_prep.image_processor.image_std)
    for_dino = standard(dino_prep.image_mean, dino_prep.image_std)

    thumbs, by_clip, by_dino = [], [], []
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(files), batch):
            loaded = list(pool.map(lambda p: views(p, detail), files[start : start + batch]))
            thumbs += [np.asarray(v[0].resize((128, 128), Image.BICUBIC)) for v in loaded]
            x = torch.from_numpy(np.stack([np.asarray(v) for vs in loaded for v in vs])).to(device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            c = F.normalize(projected(clip.get_image_features(pixel_values=for_clip(x))).float(), dim=1)
            d = F.normalize(dino(pixel_values=for_dino(x)).pooler_output.float(), dim=1)
            n = len(loaded)
            by_clip.append(F.normalize(c.view(n, -1, c.shape[1]).mean(1), dim=1))
            by_dino.append(F.normalize(d.view(n, -1, d.shape[1]).mean(1), dim=1))
            done = min(start + batch, len(files))
            if done % 240 < batch or done == len(files):
                print(f"  embedded {done}/{len(files)}", flush=True)
    return torch.cat(by_clip), torch.cat(by_dino), np.stack(thumbs)


def kmeans(x, k, seed, rounds=100):
    """Spherical k-means, seeded k-means++ style: cosine is the distance both models were trained on."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    centres = x[torch.randint(len(x), (1,), generator=g, device=x.device)]
    for _ in range(k - 1):
        far = (1.0 - x @ centres.T).min(1).values.clamp_min(0)
        centres = torch.cat([centres, x[torch.multinomial(far.square() + 1e-12, 1, generator=g)]])
    for _ in range(rounds):
        owner = (x @ centres.T).argmax(1)
        hot = F.one_hot(owner, k).to(x.dtype)
        empty = hot.sum(0) == 0
        moved = F.normalize(torch.where(empty[:, None], centres, hot.T @ x), dim=1)
        if torch.allclose(moved, centres, atol=1e-7):
            break
        centres = moved
    owner = (x @ centres.T).argmax(1)
    return owner, centres, float((1.0 - (x * centres[owner]).sum(1)).sum())


def silhouette(dist, owner, k):
    """How much nearer each image's own group is than the next nearest: 1 apart, 0 touching."""
    hot = F.one_hot(owner, k).to(dist.dtype)
    size = hot.sum(0)
    to = (dist @ hot) / size.clamp_min(1)
    to[:, size == 0] = float("inf")
    n = size[owner]
    # The own-group mean counted the image's zero distance to itself; take it back out.
    own = to.gather(1, owner[:, None]).squeeze(1) * n / (n - 1).clamp_min(1)
    other = to.scatter(1, owner[:, None], float("inf")).min(1).values
    s = (other - own) / torch.maximum(own, other).clamp_min(1e-12)
    return float(torch.where(n > 1, s, torch.zeros_like(s)).mean())


def ari(a, b):
    """Adjusted Rand index of two partitions: 1 is the same grouping, 0 is what chance gives."""
    a, b = np.asarray(a), np.asarray(b)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda n: n * (n - 1) / 2
    both = pairs(table).sum()
    rows, cols = pairs(table.sum(1)).sum(), pairs(table.sum(0)).sum()
    chance = rows * cols / pairs(len(a))
    return float((both - chance) / ((rows + cols) / 2 - chance + 1e-12))


def sweep(x, ks, seeds):
    """For every k: the tightest of `seeds` runs, its silhouette, and how well the runs agree."""
    dist = 1.0 - x @ x.T
    found = {}
    for k in ks:
        runs = [kmeans(x, k, s) for s in range(seeds)]
        owner, centres, _ = min(runs, key=lambda r: r[2])
        labels = [r[0].cpu().numpy() for r in runs]
        stable = np.mean([ari(p, q) for i, p in enumerate(labels) for q in labels[i + 1 :]])
        found[k] = {"owner": owner, "centres": centres, "silhouette": silhouette(dist, owner, k), "stable": float(stable)}
    return found


@torch.no_grad()
def resemblance(clip, prep, x, device):
    """Each image's likeness to every material, as CLIP's own softmax over the list."""
    tokens = prep.tokenizer([PROMPT.format(m) for m in MATERIALS], padding=True, return_tensors="pt").to(device)
    text = F.normalize(projected(clip.get_text_features(**tokens)).float(), dim=1)
    return (clip.logit_scale.exp().float() * x @ text.T).softmax(1)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def grid(tiles, across):
    tiles = list(tiles)
    while len(tiles) % across:
        tiles.append(np.full_like(tiles[0], 24))
    return np.concatenate([np.concatenate(tiles[i : i + across], 1) for i in range(0, len(tiles), across)], 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="A folder BESIDE the set, never inside it.")
    ap.add_argument("--least", type=int, default=3, help="Fewest categories to try.")
    ap.add_argument("--most", type=int, default=16, help="Most categories to try.")
    ap.add_argument("--min", type=int, default=150, help="Fewest images a category needs to train on.")
    ap.add_argument("--groups", type=int, default=0, help="Force this many categories; 0 lets the data choose.")
    ap.add_argument("--detail", type=float, default=3.0, help="The trainer's own, so what is embedded is what it sees.")
    ap.add_argument("--seeds", type=int, default=10, help="Runs per k, for stability.")
    ap.add_argument("--limit", type=int, default=0, help="Embed only this many, to try the pipeline.")
    args = ap.parse_args()

    root = pathlib.Path(args.images).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    # Every trainer here walks its root with `rglob`, so a sheet or a CSV left inside the set is read
    # back in as training data — the way a folder of duplicates once was, with no sign but a count.
    if out == root or root in out.parents:
        raise SystemExit(f"--out {out} is inside --images {root}; put it beside the set")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images under {root}")
    names = [f.relative_to(root).as_posix() for f in files]
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = out / (f"embeddings-{args.limit}.npz" if args.limit else "embeddings.npz")
    held = np.load(cache) if cache.is_file() else None
    if held is not None and held["files"].tolist() == names:
        print(f"{len(files)} images under {root}, embeddings from {cache.name}")
        c, d, thumbs = torch.from_numpy(held["clip"]).to(device), torch.from_numpy(held["dino"]).to(device), held["thumbs"]
    else:
        print(f"{len(files)} images under {root}, embedding with CLIP and DINOv2 on {device}")
        c, d, thumbs = embed(files, args.detail, device)
        np.savez(cache, files=np.array(names), clip=c.cpu().numpy(), dino=d.cpu().numpy(), thumbs=thumbs)

    ks = sorted(set(range(args.least, args.most + 1)) | ({args.groups} if args.groups else set()))
    by_clip, by_dino = sweep(c, ks, args.seeds), sweep(d, ks, args.seeds)
    agree = {k: ari(by_clip[k]["owner"].cpu().numpy(), by_dino[k]["owner"].cpu().numpy()) for k in ks}
    sizes = {k: np.bincount(by_clip[k]["owner"].cpu().numpy(), minlength=k) for k in ks}

    print(f"\n  {'k':>3}{'silhouette':>12}{'stable':>8}{'dino sil':>10}{'dino stable':>13}{'agree':>8}{'smallest':>10}{'largest':>9}")
    for k in ks:
        print(
            f"  {k:>3}{by_clip[k]['silhouette']:>12.3f}{by_clip[k]['stable']:>8.2f}{by_dino[k]['silhouette']:>10.3f}"
            f"{by_dino[k]['stable']:>13.2f}{agree[k]:>8.2f}{sizes[k].min():>10}{sizes[k].max():>9}"
        )

    viable = [k for k in ks if sizes[k].min() >= args.min]
    if args.groups:
        k = args.groups
        why = "asked for"
    elif viable:
        # Agreement across the two models, weighted by how reliably CLIP finds its own answer: a
        # split both reproduce, and one that does not depend on where k-means happened to start.
        k = max(viable, key=lambda q: agree[q] * by_clip[q]["stable"])
        why = f"the split CLIP and DINOv2 agree on best, of those with every category at {args.min}+ images"
    else:
        k = max(ks, key=lambda q: sizes[q].min())
        why = f"no k leaves every category {args.min}+ images; this one has the largest smallest"
    print(f"\n  chosen: {k} categories: {why}")

    owner = by_clip[k]["owner"]
    centres = by_clip[k]["centres"]
    typical = (c * centres[owner]).sum(1).cpu().numpy()
    owner = owner.cpu().numpy()
    dino_owner = by_dino[k]["owner"].cpu().numpy()
    order = np.argsort(-np.bincount(owner, minlength=k))
    rename = {int(old): new for new, old in enumerate(order)}

    clip, prep = load_clip(device)
    like = resemblance(clip, prep, c, device).cpu().numpy()

    rows = []
    with open(out / "categories.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "category", "typicality", "looks_like"])
        for i, name in enumerate(names):
            writer.writerow([name, rename[int(owner[i])], f"{typical[i]:.4f}", MATERIALS[int(like[i].argmax())]])

    print(f"\n  {'cat':>4}{'images':>8}{'dino agrees':>13}   looks like")
    for old in order:
        new = rename[int(old)]
        mine = np.flatnonzero(owner == old)
        purity = np.bincount(dino_owner[mine]).max() / len(mine)
        mean_like = like[mine].mean(0)
        top = np.argsort(-mean_like)[:3]
        words = ", ".join(f"{MATERIALS[t]} {mean_like[t]:.0%}" for t in top)
        print(f"  {new:>4}{len(mine):>8}{purity:>13.0%}   {words}")

        central = mine[np.argsort(-typical[mine])]
        edge = central[::-1][:6]
        gap = np.full((8, 6 * thumbs.shape[2], 3), 24, np.uint8)
        Image.fromarray(np.concatenate([grid(thumbs[central[:12]], 6), gap, grid(thumbs[edge], 6)], 0)).save(
            out / f"category-{new:02d}.png"
        )
        rows.append((f"{new}  {MATERIALS[top[0]]}", f"{len(mine)} images\n{MATERIALS[top[1]]}, {MATERIALS[top[2]]}\nDINOv2 agrees {purity:.0%}", thumbs[central[:10]]))

    tile, strip = thumbs.shape[1], 280
    canvas = Image.new("RGB", (strip + 10 * tile, len(rows) * (tile + 6)), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for r, (title, sub, tiles) in enumerate(rows):
        y = r * (tile + 6)
        draw.text((14, y + 16), title, fill=(236, 236, 236), font=font(22))
        draw.text((14, y + 50), sub, fill=(158, 160, 166), font=font(15), spacing=4)
        for i, t in enumerate(tiles):
            canvas.paste(Image.fromarray(t), (strip + i * tile, y))
    canvas.save(out / "overview.png")
    print(f"\n  wrote categories.csv, overview.png and a sheet per category to {out}")
    print("  each sheet: its twelve most typical images, then the six at its edge")


if __name__ == "__main__":
    main()
