"""Hang the archive by another eye: pairs and walks ordered by DINOv2 instead of by trip.

    python hang.py --embeddings categories/embeddings.npz --categories categories-7/categories.csv \
        --rhymes rhymes.png --ribbon ribbon.png

The clustering already said the consistency is in the eye, not in the trip. This takes the next
step: let the other eye curate. `--rhymes` finds the strongest cross-category pairs — two
photographs the archive's own folders keep apart that a representation trained on looks alone puts
side by side: ice beside wall-fissure, foam beside lichen, taken years apart. `--ribbon` orders
images in one greedy nearest-neighbour walk, the archive as a single continuum with no folder
boundaries at all.

Pure lookup over `categories.py`'s cached embeddings and thumbnails: no model runs, nothing leaves
the machine, seconds on CPU.
"""

import argparse
import csv
import pathlib

import numpy as np
from PIL import Image, ImageDraw


def load(args):
    d = np.load(args.embeddings, allow_pickle=True)
    files = [str(f) for f in d["files"]]
    emb = d["dino"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    thumbs = d["thumbs"]
    cat = {}
    with open(args.categories, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        for r in rows[1:]:
            cat[r[0]] = r[1]
    tags = np.array([cat.get(pathlib.Path(f).name, "?") for f in files])
    return files, emb, thumbs, tags


def shoot_of(files):
    """The frame counter in the camera's filename — near numbers are one burst, one afternoon."""
    out = []
    for f in files:
        name = pathlib.Path(f).stem
        digits = "".join(ch for ch in name.split("DSC")[-1] if ch.isdigit())
        out.append(int(digits) if digits else -(10**6))
    return np.array(out)


def rhymes(emb, tags, count, shots=None, apart=0):
    """The strongest pairs that sit in different categories, each image used once.

    The two eyes agree most about bursts: without `apart`, the top of the list is the same subject
    seconds later, filed across the category line. Requiring the frame counters to differ keeps
    only pairs the camera could not have made by standing still — the rhymes, not the echoes.
    """
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    sim[tags[:, None] == tags[None, :]] = -1  # only across the folder boundary
    if apart and shots is not None:
        sim[np.abs(shots[:, None] - shots[None, :]) < apart] = -1
    best = sim.argmax(1)
    order = np.argsort(-sim[np.arange(len(emb)), best])
    used, pairs = set(), []
    for i in order:
        j = int(best[i])
        if i in used or j in used:
            continue
        pairs.append((int(i), j, float(sim[i, j])))
        used.update((int(i), j))
        if len(pairs) >= count:
            break
    return pairs


def walk(emb, start=0):
    """One greedy nearest-neighbour path through every image."""
    left = np.ones(len(emb), bool)
    order = [start]
    left[start] = False
    at = start
    for _ in range(len(emb) - 1):
        sim = emb[at] @ emb.T
        sim[~left] = -np.inf
        at = int(sim.argmax())
        order.append(at)
        left[at] = False
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", required=True, help="categories.py's embeddings.npz (dino + thumbs).")
    ap.add_argument("--categories", required=True, help="categories.csv naming each file's folder.")
    ap.add_argument("--rhymes", help="Sheet of the strongest cross-category pairs.")
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--apart", type=int, default=0, help="Least frame-counter distance for a rhyme; 0 allows bursts.")
    ap.add_argument("--ribbon", help="Sheet of one nearest-neighbour walk through a sample of the set.")
    ap.add_argument("--sample", type=int, default=240, help="Walk length for the ribbon.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.rhymes or args.ribbon):
        raise SystemExit("give --rhymes and/or --ribbon")

    files, emb, thumbs, tags = load(args)
    side = thumbs.shape[1]

    if args.rhymes:
        got = rhymes(emb, tags, args.pairs, shoot_of(files), args.apart)
        cols, gap = 4, 14
        rows = (len(got) + cols - 1) // cols
        pw = 2 * side + 2
        sheet = Image.new("RGB", (cols * (pw + gap) + gap, 26 + rows * (side + gap)), (18, 18, 20))
        pen = ImageDraw.Draw(sheet)
        pen.text((6, 6), "cross-category nearest neighbours, strongest first — the folders say apart, the eye says together", fill=(214, 210, 202))
        for n, (i, j, s) in enumerate(got):
            x = gap + (n % cols) * (pw + gap)
            y = 26 + (n // cols) * (side + gap)
            sheet.paste(Image.fromarray(thumbs[i]), (x, y))
            sheet.paste(Image.fromarray(thumbs[j]), (x + side + 2, y))
            pen.text((x + 2, y + side - 12), f"{tags[i]}·{tags[j]}  {s:.2f}", fill=(240, 238, 230))
        sheet.save(args.rhymes)
        print(f"wrote {args.rhymes}: {len(got)} pairs, nearest at {got[0][2]:.2f}, farthest kept {got[-1][2]:.2f}")

    if args.ribbon:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(emb), size=min(args.sample, len(emb)), replace=False)
        order = walk(emb[pick], start=0)
        chosen = [int(pick[k]) for k in order]
        cols = 16
        rows = (len(chosen) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * side, 26 + rows * side), (18, 18, 20))
        pen = ImageDraw.Draw(sheet)
        pen.text((6, 6), f"one nearest-neighbour walk through {len(chosen)} images — no folders, one continuum, boustrophedon", fill=(214, 210, 202))
        for n, i in enumerate(chosen):
            r, c = divmod(n, cols)
            if r % 2 == 1:
                c = cols - 1 - c  # as the ox ploughs, so neighbours stay adjacent at row ends
            sheet.paste(Image.fromarray(thumbs[i]), (c * side, 26 + r * side))
        sheet.save(args.ribbon)
        print(f"wrote {args.ribbon}")


if __name__ == "__main__":
    main()
