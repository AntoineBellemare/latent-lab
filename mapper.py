"""A learned chart of the IP space: concept coordinates and a residual, decodable back to a vector.

    python mapper.py --space texture-ip-sd15.npz --out mapper.pt
    python mapper.py --space texture-ip-sd15.npz --model mapper.pt --axis humid --sheet ae-humid.png
    python mapper.py --space texture-ip-sd15.npz --model mapper.pt --axis humid --photo DSC02722 --sheet more-humid.png

The linear concept axes move a point by adding a fixed direction, and far from the centre that
walks off the archive's manifold. This trains the smallest thing that can do better: an
autoencoder over the 17k crop embeddings whose bottleneck is split in two — one coordinate per
fitted concept axis, aligned to the archive's own projections, and a residual that holds
everything else a texture is. Encode any photograph, turn one named coordinate, decode: the same
texture, more humid, still a place the archive could reach. The decoder's output is an ordinary
conditioning vector, so the adapter consumes it unchanged.

Training is minutes: the data is embeddings, not images. The honest numbers are printed —
reconstruction error against the spread, and how much concept information LEAKS into the residual
(a probe's R²; zero is clean) — and the sheet draws the learned walk beside the linear one at the
same positions, same seeds, which is the only comparison that means anything.

MEASURED, 2026-10-12, and the linear axes won. At 17k crops a 6+64 bottleneck holds 0.68–0.85 of
the spread as error and the probe reads the concepts out of the residual at R² 0.6–0.9 however the
covariance is punished; rendered, the decode has already traded the photograph for a generic wet
cousin at +0 sd, while the plain offset keeps the photograph and still delivers the change. The
report's own warning (a model may hide attribute information in the residual; evaluate, don't
declare) arrived on schedule. Kept as the harness for the next attempt — more capacity wants more
data, which the three-star space now supplies — but the instrument ships on difference-of-means.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_space(args):
    space = dict(np.load(args.space, allow_pickle=True))
    book = json.loads(pathlib.Path(args.concepts).read_text(encoding="utf-8"))
    names = sorted(book)
    dirs = np.stack([np.asarray(book[n]["direction"], np.float32) for n in names])
    return space, names, dirs


class Chart(torch.nn.Module):
    def __init__(self, dim, axes, residual, hidden=512):
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, axes + residual),
        )
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(axes + residual, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, dim),
        )
        self.axes = axes

    def split(self, x):
        code = self.enc(x)
        return code[:, : self.axes], code[:, self.axes :]


def train(args):
    space, names, dirs = load_space(args)
    emb = space["emb"].astype(np.float32)
    centre = emb.mean(0)
    scale = float(np.linalg.norm(emb - centre, axis=1).mean())
    x = (emb - centre) / scale
    scores = (emb @ dirs.T)
    s_mean, s_std = scores.mean(0), scores.std(0)
    a_true = (scores - s_mean) / s_std

    # Held out by parent photograph, never by crop — a crop's siblings are nearly itself.
    owners = space["owner"]
    rng = np.random.default_rng(0)
    test_images = rng.choice(owners.max() + 1, size=(owners.max() + 1) // 10, replace=False)
    test = np.isin(owners, test_images)
    device = torch.device(args.device)
    xt = torch.tensor(x[~test], device=device)
    at = torch.tensor(a_true[~test], device=device)
    xv = torch.tensor(x[test], device=device)
    av = torch.tensor(a_true[test], device=device)

    net = Chart(x.shape[1], len(names), args.residual).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # The external ruler, as tensors: a decoded point's standardized projections onto the axes.
    dirs_t = torch.tensor(dirs, device=device)
    sm = torch.tensor(s_mean, device=device)
    ss = torch.tensor(s_std, device=device)
    centre_t = torch.tensor(centre, device=device)

    def measured(z):
        return ((z * scale + centre_t) @ dirs_t.T - sm) / ss

    for epoch in range(args.epochs):
        order = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 1024):
            batch = order[i : i + 1024]
            a, r = net.split(xt[batch])
            back = net.dec(torch.cat([a, r], 1))
            recon = torch.nn.functional.mse_loss(back, xt[batch])
            align = torch.nn.functional.mse_loss(a, at[batch])
            # The residual should not secretly carry the concepts: punish their covariance.
            rc = r - r.mean(0)
            ac = at[batch] - at[batch].mean(0)
            leak = ((rc.T @ ac) / len(batch)).pow(2).sum()
            # The counterfactual cycle, judged by the ruler and not by the encoder: turn one
            # coordinate, decode, and the decoded point must MEASURE as turned.
            which = torch.randint(0, net.axes, (len(batch),), device=device)
            turned = a.detach().clone()
            turned[torch.arange(len(batch), device=device), which] += torch.rand(len(batch), device=device) * 6 - 3
            x2 = net.dec(torch.cat([turned, r.detach()], 1))
            cycle = torch.nn.functional.mse_loss(measured(x2), turned)
            loss = recon + args.align * align + args.punish * leak + args.cycle * cycle
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                a, r = net.split(xv)
                back = net.dec(torch.cat([a, r], 1))
                rerr = (back - xv).pow(2).sum(1).mean().sqrt() / xv.pow(2).sum(1).mean().sqrt()
                aerr = (a - av).std(0).mean()
            print(f"  epoch {epoch + 1}: held-out recon {rerr:.3f} of spread, axis error {aerr:.2f} sd", flush=True)

    # Leakage, measured the blunt way: can a linear probe read the concepts back out of r?
    with torch.no_grad():
        _, rt = net.split(xt)
        _, rv = net.split(xv)
    w = torch.linalg.lstsq(
        torch.cat([rt, torch.ones(len(rt), 1, device=device)], 1), at
    ).solution
    pred = torch.cat([rv, torch.ones(len(rv), 1, device=device)], 1) @ w
    r2 = 1 - (pred - av).pow(2).mean(0) / av.pow(2).mean(0)
    for n, q in zip(names, r2.tolist()):
        print(f"  leakage {n}: R² {q:.2f}")

    torch.save(dict(net=net.state_dict(), names=names, centre=centre, scale=scale,
                    s_mean=s_mean, s_std=s_std, residual=args.residual, dim=x.shape[1]), args.out)
    print(f"wrote {args.out}")


def load_model(args, device):
    got = torch.load(args.model, map_location=device, weights_only=False)
    net = Chart(got["dim"], len(got["names"]), got["residual"]).to(device).eval()
    net.load_state_dict(got["net"])
    return net, got


def sheet(args):
    from ip_space import PRESETS, pipeline, draw
    from terms import index

    space, names, dirs = load_space(args)
    device = torch.device(args.device)
    net, got = load_model(args, device)
    norm = float(space["norm"])
    k = got["names"].index(args.axis)
    d = dirs[load_space(args)[1].index(args.axis)]
    sigma = float(got["s_std"][k])

    if args.photo:
        hits = [i for i, f in enumerate(space["files"]) if args.photo.lower() in str(f).lower()]
        if not hits:
            raise SystemExit(f"no file matches {args.photo!r}")
        start = space["emb"][np.isin(space["owner"], hits)].mean(0).astype(np.float32)
        where = pathlib.Path(str(space["files"][hits[0]])).stem
    else:
        start = space["centre"].astype(np.float32)
        where = "the centre"

    x0 = torch.tensor((start - got["centre"]) / got["scale"], device=device)[None]
    with torch.no_grad():
        a0, r0 = net.split(x0)

    p = dict(PRESETS["sd15"], fast=None, steps=45, guidance=7.0)
    quiet = argparse.Namespace(scale=1.0, lora=None, style_only=False, prompt="")
    pipe, dtype = pipeline(quiet, p, device)
    steps = args.steps
    side, label = p["size"], 26
    rows = [("learned: encode, turn the coordinate, decode", []), ("linear: add the direction", [])]
    for t in steps:
        with torch.no_grad():
            a = a0.clone()
            a[0, k] = a0[0, k] + t
            back = net.dec(torch.cat([a, r0], 1))[0].cpu().numpy() * got["scale"] + got["centre"]
        rows[0][1].append(back * (norm / np.linalg.norm(back)))
        lin = start + t * sigma * d
        rows[1][1].append(lin * (norm / np.linalg.norm(lin)))

    canvas = Image.new("RGB", (len(steps) * side, 2 * (side + label)), (18, 18, 20))
    pen = ImageDraw.Draw(canvas)
    for rix, (title, vecs) in enumerate(rows):
        y = rix * (side + label)
        pen.text((6, y + 6), f"{args.axis} from {where} — {title}: {', '.join(f'{t:+g} sd' for t in steps)}",
                 fill=(214, 210, 202))
        for c, v in enumerate(vecs):
            tile = draw(pipe, dtype, p, quiet, v.astype(np.float32), args.seed, device)
            canvas.paste(tile.resize((side, side)), (c * side, y + label))
        print(f"  {title}", flush=True)
    canvas.save(args.sheet)
    print(f"wrote {args.sheet}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--concepts", default="concepts.json")
    ap.add_argument("--out", default="mapper.pt")
    ap.add_argument("--model", help="A trained chart; with --axis and --sheet, draw the comparison.")
    ap.add_argument("--axis", help="Which concept coordinate to turn.")
    ap.add_argument("--photo", help="Start from this photograph instead of the centre.")
    ap.add_argument("--steps", type=float, nargs="*", default=[-4, -2, 0, 2, 4])
    ap.add_argument("--sheet")
    ap.add_argument("--residual", type=int, default=26)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--align", type=float, default=4.0)
    ap.add_argument("--punish", type=float, default=2.0)
    ap.add_argument("--cycle", type=float, default=2.0, help="Edited codes must decode to points that measure as edited.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.model and args.axis and args.sheet:
        sheet(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
