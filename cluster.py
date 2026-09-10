"""Split a folder of images into groups that look alike, so a style trainer has a coherent target.

`style_ca.py` fits the MEAN Gram matrix of what it reads, so a set holding pale frost, dark amber
and green algae trains to the average of the three and draws mud. This groups by colour and
contrast first; then one child per group, each from the same parent, and the groups blend on a wire.

    python cluster.py --images ~/pictures/ice --out ~/pictures/ice-groups --groups 3

Each group is written as RESIZED copies, short side `--side`, which is all a texture trainer reads
and a twentieth of the disk. A contact sheet lands beside each so the grouping can be judged by
looking at it.
"""

import argparse
import pathlib
import shutil

import numpy as np
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def look(pixels):
    """What a group is formed on: where the colour sits, and how much it moves."""
    flat = pixels.reshape(-1, 3)
    return np.concatenate([flat.mean(0), flat.std(0)])


def kmeans(points, k, rounds=50, seed=0):
    """Lloyd's algorithm, seeded k-means++ style by the furthest point each time."""
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(len(points))]]
    for _ in range(k - 1):
        away = np.min([np.linalg.norm(points - c, axis=1) for c in centres], axis=0)
        centres.append(points[int(np.argmax(away))])
    centres = np.stack(centres)
    for _ in range(rounds):
        owner = np.argmin(np.linalg.norm(points[:, None] - centres[None], axis=2), axis=1)
        moved = np.stack([points[owner == i].mean(0) if (owner == i).any() else centres[i] for i in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    return owner


def sheet(images, path, across=6, tile=160):
    tiles = []
    for im in images[: across * 3]:
        scale = tile / min(im.size)
        small = im.resize((max(tile, round(im.width * scale)), max(tile, round(im.height * scale))), Image.LANCZOS)
        left, top = (small.width - tile) // 2, (small.height - tile) // 2
        tiles.append(np.asarray(small.crop((left, top, left + tile, top + tile))))
    while len(tiles) % across:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i : i + across], 1) for i in range(0, len(tiles), across)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="A folder to write one subfolder per group into.")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--side", type=int, default=1024, help="Short side of the copies written.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = pathlib.Path(args.images).expanduser()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not files:
        raise SystemExit(f"no images under {root}")
    print(f"{len(files)} images under {root}")

    held, points = [], []
    for path in files:
        im = Image.open(path).convert("RGB")
        scale = args.side / min(im.size)
        im = im.resize((max(args.side, round(im.width * scale)), max(args.side, round(im.height * scale))), Image.LANCZOS)
        held.append(im)
        points.append(look(np.asarray(im, dtype=np.float32) / 255.0))
    points = np.stack(points)
    points = (points - points.mean(0)) / (points.std(0) + 1e-6)

    owner = kmeans(points, args.groups, seed=args.seed)
    out = pathlib.Path(args.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    for i in range(args.groups):
        mine = [j for j in range(len(files)) if owner[j] == i]
        folder = out / f"group{i}"
        folder.mkdir(parents=True)
        for j in mine:
            held[j].save(folder / f"{files[j].stem}.png")
        sheet([held[j] for j in mine], out / f"group{i}.png")
        shade = np.stack([np.asarray(held[j].resize((64, 64)), dtype=np.float32) / 255.0 for j in mine]).mean((0, 1, 2))
        print(f"  group{i}: {len(mine):>3} images, mean colour {shade.round(3)} -> {folder}")
    print(f"contact sheets beside each group in {out}")


if __name__ == "__main__":
    main()
