"""taste.py — a regression model of your own star ratings, read off DINOv2's view of the whole picture.

    python taste.py gather --roots ~/Pictures/Photos2026 --out dataset/taste/ratings.csv
    python taste.py embed  --ratings dataset/taste/ratings.csv --out dataset/taste/features.npz
    python taste.py fit    --features dataset/taste/features.npz --out dataset/taste/critic.npz --sheets dataset/taste/
    python taste.py fit-clip --features dataset/taste/features.npz --clip dataset/taste/clip_vith.npy --out dataset/taste/critic-clip.npz

The stars come from ON1 sidecars (`.on1`, and Adobe `.xmp` where ON1 wrote one), so the model learns
decisions already made by eye rather than labels made for it. ON1 writes 0 for "no stars", which is
the absence of a decision: those photographs are left out, not read as the lowest score.

The picture is seen by the encoder rae.py uses (DINOv2-B with registers, its last layer norm without
the learned scale and shift), at 32 tokens on the long side. A photograph's feature is the mean and
the spread of its token grid. That choice is what lets breed.py score a token grid it has bred
without drawing it: a bred grid and an encoded photograph are the same kind of object.

A score is only worth what its evaluation is worth. A burst puts near-identical frames on both sides
of a random split, and the model then looks clever by recognising them, so folds are drawn by group:
frames within 30 s of each other in one folder, joined with any pair whose tokens nearly coincide.
Beside the photographs' own model sits a technical baseline (sharpness, brightness, contrast,
saturation): what the regression gains over it is the part of the stars that is taste rather than
culling.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import re
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PICTURES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
RAWS = {".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".orf", ".rw2"}
LONG = 32           # tokens on the long side: 448 px into the encoder, the RAE decoder's own grid
BURST_SECONDS = 30  # frames this close in time, in one folder, are one group for the folds
TWIN = 0.95         # token-mean cosine at which two frames are the same picture for the folds


# ─────────────────────────── ratings ───────────────────────────

def _on1_ratings(sidecar: pathlib.Path) -> dict:
    """{file name, lower-cased: stars} from one ON1 sidecar. A RAW+JPEG pair shares one sidecar, with
    one record per file, matched on the record's `name`."""
    try:
        doc = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    out = {}
    photos = doc.get("photos") if isinstance(doc, dict) else None
    for rec in (photos or {}).values():
        if not isinstance(rec, dict):
            continue
        meta = rec.get("metadata") or {}
        stars = meta.get("Rating")
        name = str(rec.get("name", "")).lower()
        if isinstance(stars, (int, float)) and name:
            out[name] = (int(stars), meta.get("CaptureDate"), str(meta.get("Label") or "").strip())
    return out


def _xmp_rating(sidecar: pathlib.Path):
    """(stars, colour label) from an Adobe sidecar."""
    try:
        text = sidecar.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, ""
    m = re.search(r'xmp:Rating(?:="|>)(-?\d+)', text)
    lab = re.search(r'xmp:Label(?:="|>)([^"<]*)', text)
    return (int(m.group(1)) if m else None), (lab.group(1).strip() if lab else "")


_DATE_FORMATS = ("%b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S", "%Y:%m:%d %H:%M:%S")


def _when(raw, path: pathlib.Path) -> float:
    """Capture time in seconds: ON1's date, else the file's EXIF, else its modification time."""
    if isinstance(raw, str):
        cleaned = re.sub(r"^[A-Za-z]{3,4}\s+(?=[A-Za-z]{3}\s)", "", " ".join(raw.split()))
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(cleaned[:24], fmt).timestamp()
            except ValueError:
                continue
    if path.suffix.lower() in PICTURES:
        try:
            with Image.open(path) as im:
                stamp = im.getexif().get_ifd(0x8769).get(0x9003) or im.getexif().get(0x0132)
            if stamp:
                return datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            pass
    return path.stat().st_mtime


def gather(roots) -> list[dict]:
    """Every rated photograph under `roots`: one row per frame, the JPEG preferred to its RAW twin
    because it decodes faster and shows the same picture."""
    rows = {}
    for root in roots:
        for sidecar in pathlib.Path(root).expanduser().rglob("*"):
            ext = sidecar.suffix.lower()
            if ext not in (".on1", ".xmp"):
                continue
            stem = sidecar.with_suffix("")
            twins = [p for p in sidecar.parent.glob(stem.name + ".*")
                     if p.suffix.lower() in PICTURES | RAWS]
            if not twins:
                continue
            twins.sort(key=lambda p: (p.suffix.lower() not in PICTURES, p.name))
            media = twins[0]
            if ext == ".on1":
                found = _on1_ratings(sidecar)
                hit = next((found[p.name.lower()] for p in twins if p.name.lower() in found), None)
                if hit is None and len(found) == 1:
                    hit = next(iter(found.values()))
                stars, date, label = hit if hit else (None, None, "")
            else:
                if str(stem).lower() in rows:
                    continue        # the .on1 beside it already spoke
                (stars, label), date = _xmp_rating(sidecar), None
            if stars is None or stars <= 0:
                continue            # 0 is "no stars", and a negative is Lightroom's reject
            # The colour tag rides along: the photographer's own sorting (Yellow, here, is the
            # textures), which breed.py can start a population from.
            rows[str(stem).lower()] = dict(path=str(media), stars=stars, when=_when(date, media),
                                           folder=str(sidecar.parent), label=label)
    return sorted(rows.values(), key=lambda r: (r["folder"], r["when"]))


# ─────────────────────────── pictures ───────────────────────────

def open_picture(path, long_side=1024) -> Image.Image:
    """A photograph upright and RGB, no larger than `long_side`. A RAW file gives up the JPEG preview
    its camera embedded, which is what ON1 showed while it was rated."""
    path = pathlib.Path(path)
    if path.suffix.lower() in RAWS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            flip = raw.sizes.flip
        im = Image.open(io.BytesIO(thumb.data)) if thumb.format == rawpy.ThumbFormat.JPEG \
            else Image.fromarray(thumb.data)
        im = im.rotate({3: 180, 5: 90, 6: -90}.get(flip, 0), expand=True)
    else:
        im = Image.open(path)
        im.draft("RGB", (long_side * 2, long_side * 2))
        im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    return im


def technical(im: Image.Image) -> np.ndarray:
    """Sharpness, brightness, contrast, saturation: what a cull judges before taste does."""
    g = np.asarray(im.convert("L").resize((512, round(512 * im.height / im.width))), np.float32) / 255
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    hsv = np.asarray(im.convert("HSV"), np.float32) / 255
    return np.array([np.log(lap.var() + 1e-8), g.mean(), g.std(), hsv[..., 1].mean()], np.float32)


# ─────────────────────────── the eye ───────────────────────────

class Eye:
    """rae.py's encoder, alone: no decoder is needed to read a photograph."""

    def __init__(self, device="cuda"):
        import torch
        import rae

        self.torch, self.rae, self.device = torch, rae, device
        self.encoder = rae.RAE.load_encoder(None).to(device, torch.float16).eval()

    def grid(self, width, height):
        return (max(1, round(LONG * height / width)), LONG) if width >= height \
            else (LONG, max(1, round(LONG * width / height)))

    def tokens(self, im: Image.Image):
        """[1, rows*cols, 768] tokens and the CLS token, as rae.RAE.encode computes them."""
        torch, F = self.torch, self.torch.nn.functional
        rows, cols = self.grid(*im.size)
        x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(self.device)
        x = F.interpolate(x, size=(rows * 14, cols * 14), mode="bicubic", align_corners=False)
        x = ((x - self.rae.MEAN.to(x.device)) / self.rae.STD.to(x.device)).half()
        with torch.no_grad():
            h = self.encoder(pixel_values=x).last_hidden_state
        return h[:, 5:], h[:, 0]


class ClipEye:
    """CLIP ViT-H/14, the image encoder of ip_space's sd15 preset, as `clip_vith.npy` was computed: a
    centre square at 224 px, its projected embedding, unit length. On the Yellow textures a ridge on
    these ranks held-out groups at Spearman 0.275, against 0.132 for the DINO token statistics."""

    def __init__(self, device="cuda"):
        import torch
        from transformers import CLIPVisionModelWithProjection

        import ip_space

        p = ip_space.PRESETS["sd15"]
        self.torch, self.device = torch, device
        # Loaded in half precision straight away: loading it in full precision after breed.py's RAE was in
        # memory crashed inside transformers' weight loading (an access violation, 2026-11-06).
        self.encoder = CLIPVisionModelWithProjection.from_pretrained(
            p["repo"], subfolder=p["encoder"], dtype=torch.float16).to(device).eval()
        self.mean = torch.tensor(ip_space.CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip_space.CLIP_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, images) -> np.ndarray:
        """[n, h, w, 3] squares in 0..1 -> [n, 1024] unit vectors."""
        torch, F = self.torch, self.torch.nn.functional
        x = torch.as_tensor(np.asarray(images, np.float32)).permute(0, 3, 1, 2).to(self.device)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        with torch.no_grad():
            e = self.encoder(pixel_values=((x - self.mean) / self.std).half()).image_embeds.float()
        return F.normalize(e, dim=1).cpu().numpy()


def features(tokens) -> np.ndarray:
    """A token grid as one vector: the mean and the spread of its tokens. Works the same on an encoded
    photograph and on a grid bred in breed.py, which is the point."""
    t = tokens.float()
    return np.concatenate([t.mean(1).cpu().numpy(), t.std(1).cpu().numpy()], axis=1)


def embed(rows, out):
    eye = Eye()
    feats, cls, tech, keep = [], [], [], []
    for i, r in enumerate(rows):
        try:
            im = open_picture(r["path"])
        except Exception as e:
            print(f"  skipped {r['path']}: {e}")
            continue
        tok, c = eye.tokens(im)
        feats.append(features(tok)[0])
        cls.append(c.float().cpu().numpy()[0])
        tech.append(technical(im))
        keep.append(i)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    kept = [rows[i] for i in keep]
    np.savez(out, features=np.array(feats), cls=np.array(cls), technical=np.array(tech),
             stars=np.array([r["stars"] for r in kept], np.float32),
             when=np.array([r["when"] for r in kept]), folder=np.array([r["folder"] for r in kept]),
             path=np.array([r["path"] for r in kept]))
    print(f"embedded {len(kept)} of {len(rows)} into {out}")


# ─────────────────────────── groups, folds and fit ───────────────────────────

def groups(when, folder, feats) -> np.ndarray:
    """Bursts and near-duplicates joined, so a fold never splits one picture across train and test."""
    n = len(when)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    order = np.lexsort((when, folder))
    for a, b in zip(order[:-1], order[1:]):
        if folder[a] == folder[b] and abs(when[b] - when[a]) <= BURST_SECONDS:
            union(a, b)
    m = feats[:, : feats.shape[1] // 2]
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    for a, b in zip(*np.nonzero(np.triu(sim, 1) >= TWIN)):
        union(a, b)
    return np.array([find(i) for i in range(n)])


def pair_accuracy(y, p, rng=np.random.default_rng(0), pairs=20000) -> float:
    """How often two held-out photographs with different stars come out in the right order (0.5 is
    chance)."""
    a, b = rng.integers(0, len(y), pairs), rng.integers(0, len(y), pairs)
    d = y[a] != y[b]
    return float(np.mean(np.sign(p[a][d] - p[b][d]) == np.sign(y[a][d] - y[b][d])))


def score(y, p) -> dict:
    from scipy.stats import spearmanr

    return dict(spearman=float(spearmanr(y, p).correlation), mae=float(np.abs(y - p).mean()),
                r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), pairs=pair_accuracy(y, p))


def models():
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 21))),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256,), alpha=1.0,
                                                                     early_stopping=True, max_iter=400, random_state=0)),
        "technical": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 11))),
    }


def held_out(X, y, g, make, folds=5) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    p = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(X, y, g):
        p[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return p


def fit(data, out, sheets_dir=None):
    X, T, stars = data["features"], data["technical"], data["stars"]
    folder = data["folder"]
    g = groups(data["when"], folder, X)
    print(f"{len(stars)} photographs, {len(np.unique(g))} groups after joining bursts and twins")
    print("stars:", {int(s): int((stars == s).sum()) for s in np.unique(stars)})
    # Measured on this archive (2026-11-05): 65% of the stars' variance lies between near-identical
    # frames of one burst, and 56% of near-identical pairs carry different stars. Those are choices
    # between frames a whole-picture embedding sees as one, so each frame is trained toward its
    # group's mean: the part of a rating that says which pictures, not which frame. It ranks groups
    # better (Spearman 0.236 against 0.177 on the same folds).
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    y = (np.bincount(inv, weights=stars) / cnt)[inv].astype(np.float32)
    report = {"n": int(len(y)), "groups": int(len(np.unique(g))), "target": "group mean stars"}

    preds = {}
    for name, make in models().items():
        Xi = T if name == "technical" else X
        preds[name] = held_out(Xi, y, g, make)
        report[name] = score(y, preds[name])
        print(f"{name:10s}", "  ".join(f"{k} {v:.3f}" for k, v in report[name].items()))

    # Leave one folder out: does it carry to a shoot it never saw, or has it learned the shoots?
    folders = np.unique(folder)
    if len(folders) > 1:
        report["by_folder"] = {}
        for f in folders:
            te = folder == f
            if te.sum() < 30:
                continue
            m = models()["ridge"]().fit(X[~te], y[~te])
            report["by_folder"][pathlib.Path(f).name] = score(y[te], m.predict(X[te]))
            print(f"  held-out folder {pathlib.Path(f).name:28s}",
                  "  ".join(f"{k} {v:.3f}" for k, v in report["by_folder"][pathlib.Path(f).name].items()))

    # The critic breed.py loads: ridge on everything, written out as plain numbers, plus five
    # bootstrap copies over groups whose disagreement says where the critic is guessing.
    final = models()["ridge"]().fit(X, y)
    scaler, ridge = final.steps[0][1], final.steps[1][1]
    w = ridge.coef_ / scaler.scale_
    b = float(ridge.intercept_ - (ridge.coef_ * scaler.mean_ / scaler.scale_).sum())
    rng = np.random.default_rng(0)
    ug = np.unique(g)
    boot_w, boot_b = [], []
    for _ in range(5):
        pick = np.isin(g, rng.choice(ug, len(ug), replace=True))
        m = models()["ridge"]().fit(X[pick], y[pick])
        s, r = m.steps[0][1], m.steps[1][1]
        boot_w.append(r.coef_ / s.scale_)
        boot_b.append(float(r.intercept_ - (r.coef_ * s.mean_ / s.scale_).sum()))
    np.savez(out, w=w, b=b, boot_w=np.array(boot_w), boot_b=np.array(boot_b), alpha=float(ridge.alpha_),
             stars_mean=float(y.mean()), stars_std=float(y.std()))
    report["alpha"] = float(ridge.alpha_)
    pathlib.Path(out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"critic written to {out} (ridge alpha {ridge.alpha_:.0f})")

    if sheets_dir:
        p = preds["ridge"]
        order = np.argsort(p)
        paths = data["path"]
        sheet([paths[i] for i in order[::-1][:24]], [f"{int(y[i])}* {p[i]:.2f}" for i in order[::-1][:24]],
              pathlib.Path(sheets_dir) / "critic-top.png", "held out: the critic's 24 highest")
        sheet([paths[i] for i in order[:24]], [f"{int(y[i])}* {p[i]:.2f}" for i in order[:24]],
              pathlib.Path(sheets_dir) / "critic-bottom.png", "held out: the critic's 24 lowest")
        miss = np.argsort(p - y)
        sheet([paths[i] for i in miss[::-1][:12]] + [paths[i] for i in miss[:12]],
              [f"{int(y[i])}* {p[i]:.2f}" for i in list(miss[::-1][:12]) + list(miss[:12])],
              pathlib.Path(sheets_dir) / "critic-disagrees.png",
              "top two rows: it rates higher than you; bottom two: lower")
    return report


def fit_clip(data, clip, out):
    """The same critic on CLIP ViT-H embeddings (`clip_vith.npy`, one row per photograph of `data`),
    written in critic.npz's form with kind 'clip' so breed.py knows to read children through ClipEye."""
    from scipy.stats import spearmanr

    X = clip / np.linalg.norm(clip, axis=1, keepdims=True)
    stars = data["stars"]
    g = groups(data["when"], data["folder"], data["features"])
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    y = (np.bincount(inv, weights=stars) / cnt)[inv].astype(np.float32)
    p = held_out(X, y, g, models()["ridge"])
    report = {"n": int(len(y)), "groups": int(len(np.unique(g))), "target": "group mean stars", "kind": "clip",
              "ridge": score(y, p)}
    ug = np.unique(g)
    gy = np.array([y[g == k].mean() for k in ug])
    gp = np.array([p[g == k].mean() for k in ug])
    report["group_spearman"] = float(spearmanr(gy, gp).correlation)
    print("held out:", "  ".join(f"{k} {v:.3f}" for k, v in report["ridge"].items()),
          f" group spearman {report['group_spearman']:.3f}")
    final = models()["ridge"]().fit(X, y)
    scaler, ridge = final.steps[0][1], final.steps[1][1]
    w = ridge.coef_ / scaler.scale_
    b = float(ridge.intercept_ - (ridge.coef_ * scaler.mean_ / scaler.scale_).sum())
    rng = np.random.default_rng(0)
    boot_w, boot_b = [], []
    for _ in range(5):
        pick = np.isin(g, rng.choice(ug, len(ug), replace=True))
        m = models()["ridge"]().fit(X[pick], y[pick])
        s, r = m.steps[0][1], m.steps[1][1]
        boot_w.append(r.coef_ / s.scale_)
        boot_b.append(float(r.intercept_ - (r.coef_ * s.mean_ / s.scale_).sum()))
    np.savez(out, w=w, b=b, boot_w=np.array(boot_w), boot_b=np.array(boot_b), alpha=float(ridge.alpha_),
             stars_mean=float(y.mean()), stars_std=float(y.std()), kind="clip")
    report["alpha"] = float(ridge.alpha_)
    pathlib.Path(out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"clip critic written to {out} (ridge alpha {ridge.alpha_:.0f})")
    return report


def sheet(paths, labels, out, title, tile=200, cols=6):
    rows = (len(paths) + cols - 1) // cols
    head = 28
    canvas = Image.new("RGB", (cols * tile, head + rows * tile), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(230, 230, 230), font=font)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        try:
            im = open_picture(p, tile)
        except Exception:
            continue
        x, y = (i % cols) * tile, head + (i // cols) * tile
        canvas.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.rectangle([x, y, x + 86, y + 20], fill=(0, 0, 0))
        draw.text((x + 4, y + 3), lab, fill=(255, 255, 255), font=font)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("gather")
    a.add_argument("--roots", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("embed")
    a.add_argument("--ratings", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("fit")
    a.add_argument("--features", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--sheets")
    a = sub.add_parser("fit-clip")
    a.add_argument("--features", required=True, help="features.npz, for the stars, times and folders.")
    a.add_argument("--clip", required=True, help="clip_vith.npy, one CLIP ViT-H embedding per row of --features.")
    a.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "gather":
        rows = gather(args.roots)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "stars", "when", "folder", "label"])
            w.writeheader()
            w.writerows(rows)
        counts = {s: sum(r["stars"] == s for r in rows) for s in range(1, 6)}
        print(f"{len(rows)} rated photographs -> {args.out}  stars {counts}")
    elif args.cmd == "embed":
        with open(args.ratings, encoding="utf-8") as fh:
            rows = [dict(r, stars=int(r["stars"]), when=float(r["when"])) for r in csv.DictReader(fh)]
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        embed(rows, args.out)
    elif args.cmd == "fit-clip":
        fit_clip(dict(np.load(args.features, allow_pickle=False)), np.load(args.clip), args.out)
    else:
        fit(dict(np.load(args.features, allow_pickle=False)), args.out, args.sheets)


if __name__ == "__main__":
    main()
