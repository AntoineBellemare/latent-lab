"""How fast, and how faithful: RAE on DINOv2 (rae.py) against RAEv2 on DINOv3 (raev2.py), on the same
frames, one model on the GPU at a time.

    python bench_rae.py --out bench.json                     # speed
    python bench_rae.py --out bench.json --quality 48        # and reconstruction on 48 archive crops
    python bench_rae.py --out bench.json --untrained         # DINOv3's shape with random weights (speed only)

Speed, per frame size (3:2, the decoder's pixels on the long side): encoding a picture (a new photograph
or material, once each), decoding one grid (a frame while a slider moves, or a live morph), decoding
eight at once (a film), and Real-ESRGAN x4 on the decoded frame (on release, and in films). Each is the
median of repeated runs after warming up, with the GPU synchronised. Quality: PSNR, SSIM, the fine
detail kept (Laplacian energy against the original's) and the colour shift, at 512.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

import rae as R


def timed(fn, runs):
    """The median of `runs` timings in ms, after three to warm up."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000)
    return float(np.median(out))


def crops(folder, count, seed=0):
    """`count` archive crops, the middle 512 of each 576."""
    from PIL import Image

    index = np.load(pathlib.Path(folder) / "crops.npz")
    pick = np.random.default_rng(seed).choice(len(index["crops"]), count, replace=False)
    out = []
    for i in pick:
        with Image.open(pathlib.Path(folder) / "crops" / index["crops"][i]) as im:
            im = im.convert("RGB")
        w, h = im.size
        out.append(np.asarray(im.crop(((w - 512) // 2, (h - 512) // 2, (w + 512) // 2, (h + 512) // 2)), np.float32) / 255)
    return out


def scores(orig, rec):
    import torch.nn.functional as F

    def lum(x):
        return torch.as_tensor(x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)[None, None].float()

    a, b = lum(orig), lum(rec)
    g = torch.exp(-((torch.arange(11) - 5) ** 2) / (2 * 1.5 ** 2))
    k = ((g / g.sum())[:, None] * (g / g.sum())[None]).view(1, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, k), F.conv2d(b, k)
    va, vb, cov = F.conv2d(a * a, k) - mu_a ** 2, F.conv2d(b * b, k) - mu_b ** 2, F.conv2d(a * b, k) - mu_a * mu_b
    ssim = (((2 * mu_a * mu_b + 1e-4) * (2 * cov + 9e-4)) / ((mu_a ** 2 + mu_b ** 2 + 1e-4) * (va + vb + 9e-4))).mean()
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    detail = F.conv2d(b, lap).pow(2).mean() / F.conv2d(a, lap).pow(2).mean()
    mse = float(((orig - rec) ** 2).mean())
    return dict(psnr=10 * np.log10(1 / max(mse, 1e-10)), ssim=float(ssim), detail=float(detail),
                colour=float(np.abs(orig.reshape(-1, 3).mean(0) - rec.reshape(-1, 3).mean(0)).mean()),
                colourful=colourfulness(rec) / max(colourfulness(orig), 1e-6))


def colourfulness(x):
    """Hasler and Suesstrunk's measure (2003): the spread and strength of the opponent colours."""
    rg, yb = x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=r"C:/Users/Antoine/latent-lab-models", help="The models folder.")
    ap.add_argument("--v2", default="rae/ViTXL_n08_i512/model.pt", help="RAE's decoder (DINOv2), in --models.")
    ap.add_argument("--v3", default="raev2/stage1/general/dinov3l-k23/decoder.pt,raev2/stage1/general/dinov3l-k7/decoder.pt",
                    help="RAEv2's decoders (DINOv3), comma-separated; those not on disk are skipped.")
    ap.add_argument("--upscale", default="esrgan/RealESRGAN_x4plus.pth")
    ap.add_argument("--sizes", default="512,768,1024", help="Pixels on the long side, 3:2; 'none' for quality alone.")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--untrained", action="store_true", help="DINOv3's shape with random weights, for speed alone.")
    ap.add_argument("--quality", type=int, default=0, help="Reconstruct this many archive crops at 512.")
    ap.add_argument("--crops", default="crops-yellow3", help="embed_crops.py's folder, in --models.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.models)
    import upscale

    net = upscale.load(str(root / args.upscale))
    sizes = [] if args.sizes == "none" else [int(s) for s in args.sizes.split(",")]
    tests = crops(root / args.crops, args.quality) if args.quality else []
    rows, quality = [], {}
    models = [("DINOv2 · RAE", args.v2)] + [(f"DINOv3 {pathlib.Path(p).parent.name.split('-')[-1]} · RAEv2", p)
                                           for p in args.v3.split(",") if (root / p).is_file()]
    for name, path in models:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if "dinov3" in path.lower():
            import raev2

            model = raev2.RAEv2(str(root / path), untrained=args.untrained)
        else:
            model = R.RAE(str(root / path))
        loaded = torch.cuda.max_memory_allocated() / 1e9
        for long in sizes:
            grid = model.grid(3, 2, long // 16)
            picture = np.random.default_rng(0).random((grid[0] * 16, grid[1] * 16, 3), dtype=np.float32)
            z = model.encode(picture, grid)
            z8 = z.expand(8, -1, -1).contiguous()
            frame = model.decode(z, grid)[0]
            row = dict(model=name, long=long, grid=list(grid), windows=len(model.spans(grid[0])) * len(model.spans(grid[1])),
                       encode=timed(lambda: model.encode(picture, grid), args.runs),
                       decode=timed(lambda: model.decode(z, grid), args.runs),
                       film=timed(lambda: model.decode(z8, grid), max(3, args.runs // 3)) / 8,
                       enlarge=timed(lambda: upscale.enlarge(net, frame), 3))
            row["live_fps"] = 1000 / row["decode"]
            rows.append(row)
            print(f"{name:16s} {long:5d} grid {grid[0]:2d}x{grid[1]:2d} ({row['windows']:2d} windows)  encode {row['encode']:6.1f} ms  "
                  f"decode {row['decode']:6.1f} ms ({row['live_fps']:5.1f} fps)  film {row['film']:6.1f} ms/frame  "
                  f"x4 {row['enlarge']:6.1f} ms", flush=True)
        if sizes:
            rows[-1]["memory"] = max(loaded, torch.cuda.max_memory_allocated() / 1e9)
        if tests and not (args.untrained and "dinov3" in path.lower()):
            got = [scores(x, model.decode(model.encode(x, (32, 32)), (32, 32))[0]) for x in tests]
            quality[name] = {k: float(np.mean([g[k] for g in got])) for k in got[0]}
            print(name, {k: round(v, 4) for k, v in quality[name].items()}, flush=True)
        del model
    pathlib.Path(args.out).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), untrained=args.untrained,
                                                      speed=rows, quality=quality), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
