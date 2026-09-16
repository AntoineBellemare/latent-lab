"""How varied is a generator, against the variety of the images it was trained on?

Four tiles side by side is a weak way to see mode collapse — a model can look varied in four draws
and be collapsed in forty. This samples both the model and the data and reports the same statistic
for each: the mean distance between two draws, in pixels and in a perceptual feature space.

    python diversity.py --model textures.onnx --images ~/pictures/set

A model at 1.0 is as varied as its data. Below about 0.5 it is repeating itself.
"""

import argparse
import pathlib

import numpy as np
import onnxruntime
from PIL import Image

PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def open_model(path):
    ready = onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(str(path), providers=[q for q in PROVIDERS if q in ready])
    shape = session.get_inputs()[0].shape
    return session, next(int(d) for d in reversed(shape) if isinstance(d, int) and d > 1)


def wants(session):
    """Every input a model asks for: the latent first, then a category vector if it is conditional."""
    return [
        (q.name, next((int(d) for d in reversed(q.shape) if isinstance(d, int) and d > 1), 1))
        for q in session.get_inputs()
    ]


def feed(session, z, category=None):
    """What to hand the model: the latent, and for a conditional one the category to draw it in.

    `None` is an even blend of every category, which is the honest answer for a model asked to draw
    without being told what — and is what the exported graph does with a patch that has wired nothing.
    """
    given = {session.get_inputs()[0].name: np.asarray(z, np.float32)[None]}
    for name, k in wants(session)[1:]:
        hot = np.full((1, k), 1.0 / k, np.float32)
        if category is not None:
            hot[:] = 0.0
            hot[0, int(category) % k] = 1.0
        given[name] = hot
    return given


def drawn(session, width, n, seed=0, category=None):
    rng = np.random.default_rng(seed)
    extra = wants(session)[1:]
    out = []
    for _ in range(n):
        z = rng.standard_normal(width).astype(np.float32)
        z *= np.sqrt(width) / (np.linalg.norm(z) + 1e-6)
        # A conditional model is measured the way it is played: one category at a time, never the
        # mush of all of them at once.
        pick = category if category is not None or not extra else rng.integers(extra[0][1])
        got = session.run(None, feed(session, z, pick))[0]
        a = np.asarray(got, dtype=np.float32)
        a = a[0] if a.ndim == 4 else a
        if a.shape[0] <= 4:
            a = a.transpose(1, 2, 0)
        out.append(np.clip(a * 0.5 + 0.5, 0, 1))
    return np.stack(out)


def real(where, size, n, detail=3.0, seed=0):
    rng = np.random.default_rng(seed)
    files = sorted(p for p in pathlib.Path(where).rglob("*") if p.suffix.lower() in SUFFIXES)
    side = max(size, round(size * detail))
    out = []
    for path in rng.choice(files, size=min(n, len(files)), replace=False):
        im = Image.open(path).convert("RGB")
        s = side / min(im.size)
        im = im.resize((max(side, round(im.width * s)), max(side, round(im.height * s))), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32) / 255.0
        t = rng.integers(0, a.shape[0] - size + 1)
        l = rng.integers(0, a.shape[1] - size + 1)
        out.append(a[t : t + size, l : l + size])
    return np.stack(out)


def apart(frames):
    """Mean distance between two draws, and how much of it is colour rather than structure."""
    flat = frames.reshape(len(frames), -1)
    d = [np.linalg.norm(flat[i] - flat[j]) / np.sqrt(flat.shape[1]) for i in range(len(flat)) for j in range(i + 1, len(flat))]
    tone = frames.mean(axis=(1, 2))
    t = [np.linalg.norm(tone[i] - tone[j]) for i in range(len(tone)) for j in range(i + 1, len(tone))]
    return float(np.mean(d)), float(np.mean(t))


def sharpness(frames):
    """Mean absolute Laplacian — a blur meter, so 'soft' stops being a matter of opinion.

    Read it beside `tiling`: a repeating grid is high-frequency too, and will raise this number
    while making the picture worse.
    """
    g = frames.mean(axis=3)
    lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
    return float(lap.mean())


def tiling(frames, least=4):
    """How much of one repeating pattern the frame carries: the tallest isolated spectral spike.

    An upsampling stack can stamp a regular grid across everything it draws. Nothing else measured
    here sees it — the grid is in every draw equally, so they still differ from each other, and it
    is high-frequency, so it reads as sharpness.

    A photograph's spectrum falls off smoothly with frequency, so each bin sits near the median of
    its own radial ring. A repeat puts one bin far above that ring, whatever the falloff.
    """
    out = []
    for f in frames:
        g = f.mean(axis=2)
        power = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
        mid = np.array(power.shape) // 2
        y, x = np.indices(power.shape)
        ring = np.hypot(y - mid[0], x - mid[1]).astype(int)
        tallest = 0.0
        for r in range(least, min(mid)):
            band = power[ring == r]
            if band.size < 8:
                continue
            middle = np.median(band)
            tallest = max(tallest, float(band.max() / (middle + 1e-12)))
        out.append(tallest)
    return float(np.median(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--reals", type=int, default=96, help="Reference crops, fixed: a reference resampled with the model swings every ratio on its own.")
    ap.add_argument("--detail", type=float, default=3.0)
    ap.add_argument("--category", type=int, help="Hold one category, for a conditional model.")
    args = ap.parse_args()

    session, width = open_model(args.model)
    fake = drawn(session, width, args.draws, category=args.category)
    size = fake.shape[1]
    true = real(args.images, size, args.reals, args.detail)

    fd, ft = apart(fake)
    rd, rt = apart(true)
    fs, rs = sharpness(fake), sharpness(true)
    ft2, rt2 = tiling(fake), tiling(true)
    print(f"{pathlib.Path(args.model).name} at {size} square, {args.draws} draws, {width} axes")
    print(f"  {'':16}{'model':>10}{'data':>10}{'ratio':>9}")
    print(f"  {'apart':16}{fd:>10.4f}{rd:>10.4f}{fd / rd:>9.2f}")
    print(f"  {'apart in tone':16}{ft:>10.4f}{rt:>10.4f}{ft / rt:>9.2f}")
    print(f"  {'sharpness':16}{fs:>10.4f}{rs:>10.4f}{fs / rs:>9.2f}")
    print(f"  {'tiling':16}{ft2:>10.4f}{rt2:>10.4f}{ft2 / rt2:>9.2f}")
    if fd / rd < 0.5:
        print("  the model repeats itself: draws are less than half as far apart as the data")
    if fs / rs < 0.5:
        print("  the model is soft: it carries less than half the detail the data does")
    if ft2 > 2 * rt2:
        print("  the model repeats a pattern the data does not: an upsampling grid, most likely")


if __name__ == "__main__":
    main()
