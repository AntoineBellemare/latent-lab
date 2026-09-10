"""Find near-duplicates in a training set, and move them aside.

Photographs come in bursts and brackets. A GAN weights its loss by how often it sees a thing, so
forty frames of one wave teach it that waves are forty times more of the world than they are — which
looks exactly like mode collapse and is not.

    python dedupe.py --images ~/pictures/set                 # report only
    python dedupe.py --images ~/pictures/set --apply         # move duplicates beside the folder

Nothing is deleted. Each group keeps its sharpest frame and the rest move to a folder BESIDE the
set — beside, because a trainer walking the set with `rglob` would read a subfolder of rejects back
in and the whole exercise would be silent and pointless. Reversible with a drag.
"""

import argparse
import pathlib
import shutil

import numpy as np
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def features(root, cache):
    """ResNet-50 features, reused from the cache when the count still matches."""
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if cache.is_file():
        held = np.load(cache)
        if len(held) == len(files):
            return files, held
        print(f"  cache holds {len(held)} rows for {len(files)} images; recomputing")
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Dataset

    class D(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            im = Image.open(files[i]).convert("RGB")
            s = 256 / min(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            l, t = (im.width - 224) // 2, (im.height - 224) // 2
            x = torch.tensor(np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255).permute(2, 0, 1)
            return (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net = net.eval().cuda()
    out = []
    with torch.no_grad():
        for b in DataLoader(D(), batch_size=64):
            out.append(net(b.cuda()).cpu())
    f = torch.cat(out).numpy()
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    np.save(cache, f)
    return files, f


def groups(dist, near):
    """Single-link clusters of anything closer than `near` — a burst is a chain, not a clique."""
    n = len(dist)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.where(np.triu(dist < near, 1))):
        ra, rb = find(int(i)), find(int(j))
        if ra != rb:
            parent[ra] = rb
    held = {}
    for i in range(n):
        held.setdefault(find(i), []).append(i)
    return [g for g in held.values() if len(g) > 1]


def sharpness(path):
    im = Image.open(path).convert("L").resize((256, 256), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(np.abs(lap).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True)
    ap.add_argument("--near", type=float, default=0.30, help="Distance below which two frames are the same shot.")
    ap.add_argument("--apply", action="store_true", help="Move duplicates aside. Without it, report only.")
    args = ap.parse_args()

    root = pathlib.Path(args.images).expanduser()
    files, f = features(root, root.parent / "features.npy")
    print(f"{len(files)} images")

    dist = np.sqrt(np.maximum(0, 2 - 2 * (f @ f.T)))
    found = groups(dist, args.near)
    extra = sum(len(g) - 1 for g in found)
    print(f"  {len(found)} groups of near-identical frames, holding {extra + len(found)} images")
    print(f"  {extra} would be moved aside, leaving {len(files) - extra}")

    for g in sorted(found, key=len, reverse=True)[:8]:
        span = max(dist[i][j] for i in g for j in g if i != j)
        print(f"    {len(g):>3} frames, widest gap {span:.2f}: {files[g[0]].name} …")

    if not args.apply:
        print("\n  nothing moved. add --apply to act on this.")
        return

    # BESIDE the set, never inside it: every trainer here walks its folder with rglob, so a
    # subfolder of rejects is a subfolder the next run reads straight back in.
    aside = root.parent / (root.name + "-duplicates")
    aside.mkdir(exist_ok=True)
    moved = 0
    for g in found:
        best = max(g, key=lambda i: sharpness(files[i]))
        for i in g:
            if i != best:
                shutil.move(str(files[i]), str(aside / files[i].name))
                moved += 1
    # The cache no longer matches the folder, so the next run recomputes rather than misreading it.
    (root.parent / "features.npy").unlink(missing_ok=True)
    print(f"\n  moved {moved} into {aside}")
    print(f"  {len(files) - moved} images remain for training")


if __name__ == "__main__":
    main()
