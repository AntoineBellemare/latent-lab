"""Look at what a trained latent space actually does, as one HTML page.

Any `.onnx` the `Decoder` node runs — a GAN from `fastgan.py`, the linear space from `pixel_pca.py`
— is a map from a whitened vector to a picture. This asks it the questions an instrument-builder
has: what does each axis DO, does a straight line between two points stay on the manifold, and does
equal movement in the latent buy equal movement in the image.

    python report.py --models textures.onnx pca100.onnx --out report.html

That last question is the one that decides whether a signal can drive it. A space where the same
step sometimes moves everything and sometimes nothing cannot be played.
"""

import argparse
import base64
import io
import pathlib

import numpy as np
import onnxruntime

from diversity import real
from keep_best import lattice, look
from PIL import Image

PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


def open_model(path):
    ready = onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(str(path), providers=[q for q in PROVIDERS if q in ready])
    shape = session.get_inputs()[0].shape
    width = next(int(d) for d in reversed(shape) if isinstance(d, int) and d > 1)
    return session, width


def draw(session, z):
    """One latent, or a batch of them, as `[N, H, W, 3]` in [0, 1]."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float32))
    out = []
    for row in z:
        got = session.run(None, {session.get_inputs()[0].name: row[None]})[0]
        a = np.asarray(got, dtype=np.float32)
        a = a[0] if a.ndim == 4 else a
        if a.shape[0] <= 4:
            a = a.transpose(1, 2, 0)
        out.append(np.clip(a * 0.5 + 0.5, 0, 1))
    return np.stack(out)


def tile(frames, across, scale=1.0):
    """A grid of frames as one PNG, base64 for an <img src>."""
    frames = [(f * 255).astype(np.uint8) for f in frames]
    if scale != 1.0:
        side = max(16, round(frames[0].shape[0] * scale))
        frames = [np.asarray(Image.fromarray(f).resize((side, side), Image.LANCZOS)) for f in frames]
    while len(frames) % across:
        frames.append(np.zeros_like(frames[0]))
    rows = [np.concatenate(frames[i : i + across], 1) for i in range(0, len(frames), across)]
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(rows, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def beside(session, width, where, n=8, seed=0):
    """The model's draws over real crops of the same size, and the colour it does or does not reach.

    The one figure that has caught every fault in this project: a tiling the metrics called diverse,
    and a saturated drift the pixel statistics scored as healthy.
    """
    fake = samples(session, width, n, seed)
    true = real(where, fake.shape[1], n, seed=seed)
    sheet = tile(list(fake) + list(true), n, 0.45)
    named = ("saturation", "colour spread", "brightness spread")
    rows = list(zip(named, look(fake), look(true)))
    rows.append(("repeat above its ring", lattice(fake), lattice(true)))
    return sheet, rows


def sphere(z, width, spread=1.0):
    return z * (spread * np.sqrt(width) / (np.linalg.norm(z) + 1e-6))


def samples(session, width, n, seed=0):
    rng = np.random.default_rng(seed)
    return draw(session, [sphere(rng.standard_normal(width), width) for _ in range(n)])


def sweep(session, width, axis, span=2.5, steps=7):
    z = np.zeros((steps, width), np.float32)
    z[:, axis] = np.linspace(-span, span, steps)
    return draw(session, z)


def walk(session, width, steps=9, seed=0, spread=1.0):
    """A straight line between two points, put back on the shell at every stop."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(width), rng.standard_normal(width)
    z = [sphere((1 - t) * a + t * b, width, spread) for t in np.linspace(0, 1, steps)]
    return draw(session, z)


def smoothness(session, width, steps=48, paths=6, seed=0):
    """How evenly a straight line spends itself.

    Consecutive frames along an evenly-sampled path, measured. A space that is a control surface
    moves the image by the same amount at every stop, so the spread of these steps — not their size
    — is the number that matters. Reported as the coefficient of variation: lower is smoother.
    """
    rng = np.random.default_rng(seed)
    spreads, curves = [], []
    for _ in range(paths):
        a, b = rng.standard_normal(width), rng.standard_normal(width)
        frames = draw(session, [sphere((1 - t) * a + t * b, width) for t in np.linspace(0, 1, steps)])
        d = np.sqrt(((frames[1:] - frames[:-1]) ** 2).mean(axis=(1, 2, 3)))
        curves.append(d)
        spreads.append(d.std() / (d.mean() + 1e-9))
    return float(np.mean(spreads)), np.stack(curves).mean(0)


def spark(curve, width=560, height=90):
    """The step sizes along a path, as an inline SVG."""
    top = float(curve.max()) * 1.15 + 1e-9
    pts = " ".join(
        f"{i / (len(curve) - 1) * width:.1f},{height - v / top * height:.1f}" for i, v in enumerate(curve)
    )
    flat = height - float(curve.mean()) / top * height
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="spark">'
        f'<line x1="0" y1="{flat:.1f}" x2="{width}" y2="{flat:.1f}" class="mean"/>'
        f'<polyline points="{pts}" class="curve"/></svg>'
    )


CSS = """
:root{--ink:#15171a;--dim:#5d6570;--line:#dfe3e8;--bg:#fbfbfc;--card:#fff;--accent:#3d6ee0;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--ink:#e8eaed;--dim:#98a1ad;
--line:#2a2f37;--bg:#111316;--card:#181b1f;--accent:#7aa2f7}}
:root[data-theme=dark]{--ink:#e8eaed;--dim:#98a1ad;--line:#2a2f37;--bg:#111316;--card:#181b1f;--accent:#7aa2f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:20px;letter-spacing:-.01em;margin:52px 0 6px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:26px 0 8px;color:var(--dim);font-weight:600}
p{max-width:74ch;color:var(--ink)}.lede{color:var(--dim);max-width:74ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
img{max-width:100%;display:block;border-radius:6px;image-rendering:auto}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600}td.num{font-family:var(--mono);text-align:right}
.spark{width:100%;height:90px}.spark .curve{fill:none;stroke:var(--accent);stroke-width:2}
.spark .mean{stroke:var(--dim);stroke-dasharray:3 4;stroke-width:1}
code{font-family:var(--mono);font-size:13px;background:var(--card);border:1px solid var(--line);
padding:1px 5px;border-radius:4px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.tag{display:inline-block;font-family:var(--mono);font-size:12px;color:var(--dim);
border:1px solid var(--line);border-radius:999px;padding:2px 9px;margin-right:6px}
"""


def page(sections, title):
    body = "".join(sections)
    return f"<title>{title}</title><style>{CSS}</style><div class=wrap>{body}</div>"


def img(b64, alt=""):
    return f'<div class=scroll><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'


def report(paths, out, axes, seed, images=None):
    sections = [
        "<h1>Latent spaces, measured</h1>",
        "<p class=lede>Every model here is an <code>.onnx</code> the goofi <code>Decoder</code> node runs: "
        "a whitened vector in, a picture out. The question is not whether the pictures are good — it is "
        "whether the space between them can be played.</p>",
    ]
    rows = []
    for path in paths:
        name = pathlib.Path(path).stem
        session, width = open_model(path)
        cv, curve = smoothness(session, width)
        rows.append((name, width, cv))
        sections.append(f"<h2>{name}</h2>")
        sections.append(
            f'<p><span class=tag>{width} axes</span>'
            f'<span class=tag>step spread {cv:.3f}</span>'
            f'<span class=tag>{pathlib.Path(path).stat().st_size / 2**20:.0f} MB</span></p>'
        )
        sections.append("<h3>Samples</h3>")
        sections.append(img(tile(samples(session, width, 16, seed), 8), f"{name} samples"))
        if images:
            sheet, stats = beside(session, width, images, seed=seed)
            sections.append(
                "<h3>Beside the photographs</h3><p class=lede>Model draws on the top row, real crops "
                "from the training set below, at the same size. Ratios near 1.00 mean the model "
                "reaches as far as the data does; the last row is a repeating pattern standing above "
                "its own spectral ring, where the data sets the honest floor.</p>"
            )
            sections.append(img(sheet, f"{name} beside the data"))
            sections.append(
                "<div class=card><table><tr><th></th><th>model</th><th>data</th><th>ratio</th></tr>"
                + "".join(
                    f"<tr><td>{q}</td><td class=num>{m:.4f}</td><td class=num>{d:.4f}</td>"
                    f"<td class=num>{m / d:.2f}</td></tr>"
                    for q, m, d in stats
                )
                + "</table></div>"
            )
        sections.append(
            "<h3>What each axis does</h3><p class=lede>Axis 0 downward, each swept from "
            "−2.5 to +2.5 standard deviations. These are the model's own principal directions, "
            "so they are ordered: the top row is what it varies along most.</p>"
        )
        strips = [sweep(session, width, k) for k in range(min(axes, width))]
        sections.append(img(tile([f for s in strips for f in s], 7, 0.55), f"{name} axes"))
        sections.append("<h3>A straight line between two points</h3>")
        sections.append(img(tile(walk(session, width, 9, seed), 9, 0.7), f"{name} walk"))
        sections.append(
            "<h3>How evenly it spends itself</h3><p class=lede>Distance between consecutive frames "
            "along an evenly sampled path, averaged over six paths. Flat is good: it means the same "
            "movement in a signal buys the same movement in the image wherever it happens to be. "
            f"Coefficient of variation <b>{cv:.3f}</b>.</p>"
        )
        sections.append(f"<div class=card>{spark(curve)}</div>")

    best = min(rows, key=lambda r: r[2])[0] if rows else None
    head = (
        "<h2>Side by side</h2><div class=card><table><tr><th>model</th><th>axes</th>"
        "<th>step spread (lower is smoother)</th></tr>"
        + "".join(
            f"<tr><td>{n}{' &larr; smoothest' if n == best else ''}</td>"
            f"<td class=num>{w}</td><td class=num>{c:.3f}</td></tr>"
            for n, w, c in rows
        )
        + "</table></div>"
    )
    sections.insert(2, head)
    pathlib.Path(out).write_text(page(sections, "Latent spaces, measured"), encoding="utf-8")
    return out, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axes", type=int, default=8, help="How many principal axes to sweep.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", help="The training set, to put the model beside what it learnt from.")
    args = ap.parse_args()
    where, rows = report(args.models, args.out, args.axes, args.seed, args.images)
    for name, width, cv in rows:
        print(f"  {name}: {width} axes, step spread {cv:.3f}")
    print(f"wrote {where}")


if __name__ == "__main__":
    main()
