"""Fly a GAN's latent space along a chosen route, and film it.

    python flights.py --model conditional-512.onnx --route orbit --category 2 --out waves-orbit.mp4
    python flights.py --model looks-presets.onnx --route drift --weights 0.6,0,0,0.4 --out drift.mp4
    python flights.py --model looks-best.onnx --route places --space texture-ip-sd15.npz --stops 10 --even \\
        --frame 3:2 --upscale RealESRGAN_x4plus.pth --size 1920x1280 --fps 60 --seconds 40 --out archive.mp4

Every export here takes its latent as whitened axes, strongest first (`fastgan.py`), so a route
written in z is the same kind of motion on every model, and the same `--seed` flies the same route
through two of them. Each route is a closed loop, so the film repeats without a seam:

- **orbit**: a great circle at the latent's usual radius. Every point on it is an ordinary draw;
  a full turn passes through four unrelated ones.
- **knobs**: a Lissajous figure on the eight strongest axes, every other axis held still. What the
  knobs a hand would reach for first can do on their own.
- **drift**: every axis wanders on its own slow periodic noise, the way a patch's band powers move
  it: nothing ever returns to the same place until the loop does.
- **tour**: six places, gliding between them along the sphere and pausing at each.
- **breath**: out from the middle (z = 0, the model's average texture) to beyond the usual spread
  and back, turning as it goes. What `spread` does.
- **categories**: the latent on a small slow circle while the category crosses every one in turn.
  A conditional model only.

A space-conditioned model (`fastgan.py --look`, an input called `look`) is not held to a list: it
takes any place in the lab's material space, so two more routes move the material itself, the
latent on the same small slow circle meanwhile:

- **places**: through a set of places on a smooth closed curve (Catmull-Rom on the sphere, no
  stops), in the order that makes the shortest loop. The places are a folder of `.npy` (`--places`,
  as `bank.py --places` writes them), or photographs from the lab's space (`--space`), `--stops` of
  them, each the most typical photograph of one region of the archive.
- **wander**: no destinations: each of the space's twelve main directions drifts on its own slow
  periodic noise, so the material passes through mixtures no single photograph holds.

`--even` makes the picture, rather than the route, move at one speed: the route is first drawn
densely, finer wherever the picture jumps, and the frames are then spaced by how far the picture
has moved, not by how far the route has. A route that hurried through one stretch and lingered in
another becomes one steady glide. `--frame 3:2` crops the square to that shape, `--upscale`
enlarges every frame four times with Real-ESRGAN (`upscale.py`), and `--size` sets the film's own.

Beside the film, `<out>.json` holds the route flattened to two dimensions for drawing (its own two
main directions, the places' when the material moves; for `categories`, the weights on a polygon),
where it passes each place, and how far the picture moves each frame, which is where a route runs
smooth and where it lurches — `morph.py`'s question, every frame.
"""

import argparse
import json
import pathlib
import subprocess

import numpy as np

ROUTES = ["orbit", "knobs", "drift", "tour", "breath", "categories", "places", "wander"]


def unit(v):
    return v / np.linalg.norm(v)


def great(a, b):
    """Two directions at right angles, each at the latent's usual length."""
    b = b - (b @ unit(a)) * unit(a)
    return unit(a) * len(a) ** 0.5, unit(b) * len(b) ** 0.5


def ease(s):
    return s * s * (3 - 2 * s)


def slerp(p, q, s):
    """Along the sphere between two draws, the length moving linearly, so the middle is not a dull average."""
    u, v = unit(p), unit(q)
    angle = np.arccos(np.clip(u @ v, -1, 1))
    way = u if angle < 1e-6 else (np.sin((1 - s) * angle) * u + np.sin(s * angle) * v) / np.sin(angle)
    return way * ((1 - s) * np.linalg.norm(p) + s * np.linalg.norm(q))


def circling(t, width, rng):
    """A latent on a small slow circle: the picture keeps breathing while something else travels."""
    base = rng.standard_normal(width)
    a, b = great(rng.standard_normal(width), rng.standard_normal(width))
    return base + 0.35 * (np.cos(2 * np.pi * t)[:, None] * a + np.sin(2 * np.pi * t)[:, None] * b)


def wobble(t, count, cycles, rng):
    """`count` slow periodic noises, each with spread 1 over the whole loop."""
    total = np.zeros((len(t), count))
    power = np.zeros(count)
    for m in range(1, cycles + 1):
        size = rng.standard_normal(count) / m
        total += size * np.sin(2 * np.pi * m * t[:, None] + rng.uniform(0, 2 * np.pi, count))
        power += size**2 / 2
    # The spread is known in closed form, so any handful of moments drawn agree with the whole loop.
    return total / power**0.5


def route(name, t, width, rng, spread, classes, stops=None, space=None):
    """Latents at each moment `t` of the loop (0 to 1), and what the second input gets there:
    category weights, places, or None to hold the given ones."""
    t = np.asarray(t, np.float64)
    if name == "orbit":
        a, b = great(rng.standard_normal(width), rng.standard_normal(width))
        z = np.cos(2 * np.pi * t)[:, None] * a + np.sin(2 * np.pi * t)[:, None] * b
        return z * spread, None
    if name == "knobs":
        z = np.tile(rng.standard_normal(width) * 0.6, (len(t), 1))
        for k, f in enumerate([1, 1, 2, 2, 3, 3, 4, 4]):
            z[:, k] = 1.6 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        return z * spread, None
    if name == "drift":
        return wobble(t, width, 3, rng) * spread, None
    if name == "tour":
        places = rng.standard_normal((6, width))
        z = np.zeros((len(t), width))
        for i, s in enumerate(t * len(places)):
            k, s = int(s), s - int(s)
            # A quarter of each leg is spent standing at the place, the rest gliding to the next.
            z[i] = slerp(places[k], places[(k + 1) % len(places)], ease(max(0.0, s - 0.25) / 0.75))
        return z * spread, None
    if name == "breath":
        a, b = great(rng.standard_normal(width), rng.standard_normal(width))
        turn = 2 * 2 * np.pi * t
        way = np.cos(turn)[:, None] * unit(a) + np.sin(turn)[:, None] * unit(b)
        reach = 1.35 * (1 - np.cos(2 * np.pi * t)) / 2
        return way * reach[:, None] * width ** 0.5 * spread, None
    if name == "categories":
        if not classes:
            raise SystemExit("this model takes no category: there is nothing to cross")
        z = circling(t, width, rng)
        hot = np.eye(classes)
        w = np.zeros((len(t), classes))
        for i, s in enumerate(t * classes):
            k, s = int(s) % classes, s - int(s)
            s = ease(max(0.0, s - 0.4) / 0.6)
            w[i] = (1 - s) * hot[k] + s * hot[(k + 1) % classes]
        return z * spread, w
    if name == "places":
        if stops is None:
            raise SystemExit("places needs --places or --space")
        n = len(stops)
        w = np.zeros((len(t), stops.shape[1]))
        for i, s in enumerate(t * n):
            k, s = int(s) % n, s - int(s)
            p0, p1, p2, p3 = stops[(k - 1) % n], stops[k], stops[(k + 1) % n], stops[(k + 2) % n]
            # Catmull-Rom: through every place with no corner at any, then back onto the sphere.
            w[i] = unit(0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s**2 + (3 * p1 - p0 - 3 * p2 + p3) * s**3))
        return circling(t, width, rng) * spread, w
    if name == "wander":
        if space is None:
            raise SystemExit("wander needs --space")
        keep = 12
        lean = wobble(t, keep, 2, rng) * 1.25  # in each direction's own spread
        w = space["centre"] + (lean * space["spread"][:keep]) @ space["axes"][:keep]
        return circling(t, width, rng) * spread, w / np.linalg.norm(w, axis=1, keepdims=True)
    raise SystemExit(f"no route called {name}: {', '.join(ROUTES)}")


def loop_order(stops):
    """Near enough the order of the shortest closed loop through the places: nearest neighbour
    first, then any two crossing legs untangled."""
    far = 1 - stops @ stops.T
    order = [0]
    while len(order) < len(stops):
        order.append(min((i for i in range(len(stops)) if i not in order), key=lambda i: far[order[-1], i]))
    better = True
    while better:
        better = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                a, b, c, d = order[i - 1], order[i], order[j], order[(j + 1) % len(order)]
                if far[a, c] + far[b, d] < far[a, b] + far[c, d] - 1e-9:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    better = True
    return order


def photographs(space, count, rng):
    """`count` photographs from the lab's space, each the most typical of one region of the archive:
    every photograph's place (the mean of its crops) gathered into `count` groups, and the photograph
    nearest each group's middle."""
    owner = space["owner"]
    places = np.stack([space["emb"][owner == i].mean(0) for i in range(len(space["files"]))])
    places /= np.linalg.norm(places, axis=1, keepdims=True)
    middles = places[rng.choice(len(places), count, replace=False)]
    for _ in range(30):
        near = (places @ middles.T).argmax(1)
        middles = np.stack([places[near == k].mean(0) if (near == k).any() else middles[k] for k in range(count)])
        middles /= np.linalg.norm(middles, axis=1, keepdims=True)
    picked = [int((places @ m).argmax()) for m in middles]
    return places[picked], [str(space["files"][i]) for i in picked]


def steady(look, frames, dense=4, rounds=4):
    """Moments of the loop at which the picture has moved equally far from each frame to the next,
    and how many jumps stayed bigger than a frame's move however finely the route was drawn.
    `look(u)` is a small copy of the picture at moment u."""
    u = list(np.arange(frames * dense) / (frames * dense))
    seen = [look(x) for x in u]
    for pass_ in range(rounds + 1):
        apart = np.array([np.abs(seen[(i + 1) % len(u)] - seen[i]).mean() for i in range(len(u))])
        step = apart.sum() / frames
        coarse = np.flatnonzero(apart > step / 2)
        if not len(coarse) or pass_ == rounds:
            break
        # Wherever the picture jumped more than half a frame's move, draw the route finer there.
        for i in coarse[::-1]:
            mid = (u[i] + (u[i + 1] if i + 1 < len(u) else 1.0)) / 2
            u.insert(i + 1, mid)
            seen.insert(i + 1, look(mid))
    gone = np.concatenate([[0], np.cumsum(apart)])
    return np.interp(np.arange(frames) * gone[-1] / frames, gone, u + [1.0]), int((apart > step).sum())


def flat(z, w, places=False, stops=None):
    """Two numbers per frame to draw the route by: its own two main directions (the places' when the
    material moves, the stops put in the same plane), or the weights on a polygon."""
    if w is not None and not places:
        corner = np.exp(2j * np.pi * np.arange(w.shape[1]) / w.shape[1] - 0.5j * np.pi)
        p = w @ corner
        return np.stack([p.real, p.imag], 1), None
    path = w if places else z
    middle = path.mean(0)
    _, _, vt = np.linalg.svd(path - middle, full_matrices=False)
    p = (path - middle) @ vt[:2].T
    big = np.abs(p).max()
    return p / big, None if stops is None else (stops - middle) @ vt[:2].T / big


def shape_of(frame, side):
    """The largest crop of a square `side` across with the proportions `w:h` and even sides."""
    a, b = (int(x) for x in frame.split(":"))
    for h in range(side - side % 2, 0, -2):
        w = h * a / b
        if w <= side and w == int(w) and int(w) % 2 == 0:
            return int(w), h
    raise SystemExit(f"no crop of {side} with proportions {frame}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--route", required=True, choices=ROUTES)
    ap.add_argument("--out", required=True, help="An .mp4.")
    ap.add_argument("--category", type=int, help="For a conditional model: hold this one.")
    ap.add_argument("--weights", help="For a conditional model: hold this blend, comma-separated shares.")
    ap.add_argument("--places", help="For places: a folder of .npy places to pass through (bank.py --places).")
    ap.add_argument("--space", help="For places and wander: the lab's space (.npz), its photographs and main directions.")
    ap.add_argument("--stops", type=int, default=10, help="For places from --space: how many photographs.")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="The same seed flies the same route through any model of the same width.")
    ap.add_argument("--even", action="store_true", help="Space the frames by how far the picture moves.")
    ap.add_argument("--frame", help="Crop to these proportions, e.g. 3:2.")
    ap.add_argument("--upscale", help="Real-ESRGAN x4 weights (RealESRGAN_x4plus.pth) to enlarge every frame with.")
    ap.add_argument("--size", help="The film's own size, WxH, after any enlarging.")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "cuda":
        import torch  # noqa: F401  the CUDA provider finds its cuDNN through torch's DLLs
    import onnxruntime

    ready = onnxruntime.get_available_providers() if args.device == "cuda" else []
    session = onnxruntime.InferenceSession(args.model, providers=[p for p in ("CUDAExecutionProvider",) if p in ready] + ["CPUExecutionProvider"])
    inputs = session.get_inputs()
    width = inputs[0].shape[-1]
    looks = len(inputs) > 1 and inputs[1].name == "look"
    classes = inputs[1].shape[-1] if len(inputs) > 1 else 0
    held = np.zeros(classes, np.float32)
    if args.weights:
        held[: len(args.weights.split(","))] = [float(x) for x in args.weights.split(",")]
    elif args.category is not None:
        held[args.category] = 1.0
    if args.route in ("places", "wander") and not looks:
        raise SystemExit(f"{args.route} moves the material: it needs a space-conditioned model (an input called look)")

    space = dict(np.load(args.space)) if args.space else None
    stops, names = None, None
    if args.route == "places":
        if args.places:
            files = sorted(pathlib.Path(args.places).glob("*.npy"))
            stops, names = np.stack([np.load(f) for f in files]), [f.stem for f in files]
        elif space is not None:
            stops, names = photographs(space, args.stops, np.random.default_rng(args.seed))
        if stops is not None:
            stops = stops / np.linalg.norm(stops, axis=1, keepdims=True)
            order = loop_order(stops)
            stops, names = stops[order], [names[i] for i in order]

    def at(u):
        return route(args.route, u, width, np.random.default_rng(args.seed), args.spread, classes, stops, space)

    crop = None

    def draw(z1, w1):
        nonlocal crop
        feed = {inputs[0].name: z1[None].astype(np.float32)}
        if classes:
            feed[inputs[1].name] = (w1 if w1 is not None else held)[None].astype(np.float32)
        image = np.clip(np.asarray(session.run(None, feed)[0], np.float32)[0].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
        if crop is None:
            side = image.shape[0]
            cw, ch = shape_of(args.frame, side) if args.frame else (side, side)
            crop = ((side - ch) // 2, (side - cw) // 2, ch, cw)
        top, left, ch, cw = crop
        return image[top : top + ch, left : left + cw]

    frames = round(args.seconds * args.fps)
    lurches = 0
    if args.even:
        def small(u):
            z1, w1 = at([u])
            image = draw(z1[0], None if w1 is None else w1[0])
            h, wd = image.shape[0] // 4 * 4, image.shape[1] // 4 * 4
            return image[:h, :wd].reshape(h // 4, 4, wd // 4, 4, 3).mean((1, 3))
        u, lurches = steady(small, frames)
    else:
        u = np.arange(frames) / frames
    z, w = at(u)

    net = None
    if args.upscale:
        import upscale
        net = upscale.load(args.upscale, args.device)
    size = tuple(int(x) for x in args.size.split("x")) if args.size else None

    out = pathlib.Path(args.out)
    film = None
    moved, last, first = [], None, None
    for i in range(frames):
        image = draw(z[i], None if w is None else w[i])
        shown = (np.clip(upscale.enlarge(net, image) if net is not None else image, 0, 1) * 255).round().astype(np.uint8)
        if size and (shown.shape[1], shown.shape[0]) != size:
            from PIL import Image
            shown = np.asarray(Image.fromarray(shown).resize(size, Image.LANCZOS))
        if film is None:
            h, wd = shown.shape[:2]
            film = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wd}x{h}",
                 "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
            first, made = image, [wd, h]
        film.stdin.write(shown.tobytes())
        if last is not None:
            moved.append(float(np.abs(image - last).mean()))
        last = image
    moved.append(float(np.abs(first - last).mean()))  # the seam, back to the start
    film.stdin.close()
    film.wait()

    moved = np.array(moved)
    drawn, dots = flat(z, w, looks and w is not None, stops)
    record = {
        "model": pathlib.Path(args.model).name, "route": args.route, "seed": args.seed, "fps": args.fps,
        "frames": frames, "spread": args.spread, "held": held.tolist() if w is None else None, "size": made,
        "even": args.even, "flat": np.round(drawn, 4).tolist(), "moved": np.round(moved, 5).tolist(),
        "evenness": float(moved.std() / moved.mean()), "moments": np.round(u, 7).tolist(),
    }
    if stops is not None:
        # The loop passes place k at moment k / n: the first frame on or after it.
        record["stops"] = [{"name": n, "frame": int(np.searchsorted(u, k / len(stops)) % frames), "at": np.round(d, 4).tolist()}
                           for k, (n, d) in enumerate(zip(names, dots))]
    out.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
    print(f"{out.name}: {frames} frames, picture moves {moved.mean():.4f} a frame on average, "
          f"evenness {moved.std() / moved.mean():.2f} (lower is steadier), seam {moved[-1]:.4f}"
          + (f", {lurches} jumps the route could not be drawn finely enough to smooth" if lurches else ""))


if __name__ == "__main__":
    main()
