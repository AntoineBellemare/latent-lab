"""Can anything tell these categories apart from what the TRAINER sees?

    python separable.py --images ~/pictures/set-by-category --detail 3

Categories are usually assigned by looking at whole frames. A generator never sees a whole frame: it
sees a crop, and at `--detail 3` that crop is a third of one. Two categories that are obvious from
across a room can be the same handful of pixels close up, and a conditional model that merges them
is then reading its data correctly rather than failing.

So this crops the way the trainer crops, embeds with DINOv2, and fits a linear probe to every pair.
The held-out accuracy is the ceiling: a pair a probe cannot separate is a pair no amount of training
will separate either, and the honest fix is to merge them.
"""

import argparse
import concurrent.futures
import itertools
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DINO = "facebook/dinov2-small"


def crops(path, size, detail, each, rng):
    """What `Folder` hands the trainer: a square of one frame, at the scale it trains on."""
    side = max(size, round(size * detail))
    with Image.open(path) as im:
        im.draft("RGB", (im.width // 2, im.height // 2))
        im = im.convert("RGB")
    scale = side / min(im.size)
    im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), Image.LANCZOS)
    out = []
    for _ in range(each):
        top = rng.randint(0, im.height - size)
        left = rng.randint(0, im.width - size)
        out.append(im.crop((left, top, left + size, top + size)).resize((224, 224), Image.BICUBIC))
    return out


@torch.no_grad()
def embed(folders, size, detail, each, device, batch=32):
    """One vector per crop, and which category it came from."""
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(DINO, local_files_only=True).to(device).half().eval()
    prep = AutoImageProcessor.from_pretrained(DINO, local_files_only=True)
    mean = torch.tensor(prep.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(prep.image_std, device=device).view(1, 3, 1, 1)

    seen, tags = [], []
    for tag, folder in enumerate(folders):
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES)
        rng = random.Random(tag)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(files), batch):
                loaded = list(pool.map(lambda p: crops(p, size, detail, each, rng), files[start : start + batch]))
                flat = [q for group in loaded for q in group]
                x = torch.from_numpy(np.stack([np.asarray(q) for q in flat])).to(device)
                x = ((x.permute(0, 3, 1, 2).float() / 255.0) - mean) / std
                got = model(pixel_values=x.half()).pooler_output.float()
                seen.append(F.normalize(got, dim=1))
                tags += [tag] * len(flat)
        print(f"  {folder.name}: {len(files)} frames, {len(files) * each} crops", flush=True)
    return torch.cat(seen), torch.tensor(tags, device=device)


def probe(x, y, rounds=400, seed=0):
    """Held-out accuracy of a linear classifier: the ceiling on telling these two apart."""
    torch.manual_seed(seed)
    cut = torch.randperm(len(x), device=x.device)
    train, test = cut[: int(len(x) * 0.7)], cut[int(len(x) * 0.7) :]
    net = torch.nn.Linear(x.shape[1], 2).to(x.device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(rounds):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(x[train]), y[train]).backward()
        opt.step()
    with torch.no_grad():
        return float((net(x[test]).argmax(1) == y[test]).float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of category folders.")
    ap.add_argument("--size", type=int, default=512, help="The trainer's crop size.")
    ap.add_argument("--detail", type=float, default=3.0, help="The trainer's own.")
    ap.add_argument("--each", type=int, default=4, help="Crops per frame.")
    ap.add_argument("--floor", type=float, default=0.75, help="Below this a pair is worth merging.")
    args = ap.parse_args()

    root = pathlib.Path(args.images).expanduser()
    folders = sorted(q for q in root.iterdir() if q.is_dir())
    if len(folders) < 2:
        raise SystemExit(f"{root} holds no category folders")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = embed(folders, args.size, args.detail, args.each, device)

    print(f"\n  {'':5}" + "".join(f"{q.name:>9}" for q in folders))
    scores = {}
    for a, b in itertools.combinations(range(len(folders)), 2):
        keep = (y == a) | (y == b)
        scores[(a, b)] = probe(x[keep], (y[keep] == b).long())
    for a in range(len(folders)):
        line = f"  {folders[a].name:>5}"
        for b in range(len(folders)):
            line += f"{'-':>9}" if a == b else f"{scores[tuple(sorted((a, b)))]:>9.2f}"
        print(line)

    weak = sorted((v, k) for k, v in scores.items() if v < args.floor)
    print(f"\n  a pair at 0.50 is a coin toss; at 1.00 a probe never confuses them")
    if weak:
        print(f"  below {args.floor:.2f}, which is where a conditional model will merge them anyway:")
        for v, (a, b) in weak:
            print(f"    {folders[a].name} and {folders[b].name}: {v:.2f}")
    else:
        print(f"  every pair is separable above {args.floor:.2f} from the crops alone")


if __name__ == "__main__":
    main()
