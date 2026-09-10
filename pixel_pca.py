"""The linear texture space of a folder of images, as an `.onnx` the `Decoder` node runs.

Principal components of the PIXELS, not of a feature space: features cannot be inverted, pixels can.
Keep the top `--components` of them and the whole space is a matrix product — `mean + z * sigma @ V`
— which is exactly invertible, exactly smooth, and needs no training at all. Every straight line in
it is a valid image, which is the one thing a GAN's latent space cannot promise.

    python pixel_pca.py --images ~/pictures/set --out pca.onnx --components 50

It is the baseline to judge a GAN against, and worth having on its own: the reconstructions are
low-frequency and ghostly rather than photographic, and they move without a seam.

The per-component standard deviations are folded into the model, so its input is WHITENED — unit
Gaussian per axis. That is what `Decoder` already assumes, so `spread` is a truncation and `seed`
picks a slice, both meaning what they mean everywhere else.
"""

import argparse
import pathlib

import numpy as np
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def pixels(where, size, limit, detail, crops, seed=0):
    """Every image as one row of `3 * size * size` numbers in [-1, 1]."""
    root = pathlib.Path(where).expanduser()
    found = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)[:limit]
    if not found:
        raise SystemExit(f"no images under {root}")
    rng = np.random.default_rng(seed)
    side = max(size, round(size * detail))
    rows = []
    for path in found:
        try:
            im = Image.open(path).convert("RGB")
        except OSError:
            continue
        scale = side / min(im.size)
        im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        for _ in range(crops):
            t = rng.integers(0, a.shape[0] - size + 1)
            l = rng.integers(0, a.shape[1] - size + 1)
            rows.append(a[t : t + size, l : l + size].transpose(2, 0, 1).ravel())
    print(f"{len(found)} images under {root}, {len(rows)} crops of {size} square")
    return np.stack(rows)


def components(x, keep):
    """The top `keep` principal directions, and the spread of each.

    There are far more pixels than pictures, so the eigenproblem is solved in SAMPLE space — a
    matrix the size of the set rather than the size of an image.
    """
    mean = x.mean(0)
    x = x - mean
    gram = x @ x.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:keep]
    values, vectors = np.maximum(values[order], 1e-8), vectors[:, order]
    basis = (x.T @ vectors) / np.sqrt(values)
    sigma = np.sqrt(values / (len(x) - 1))
    held = values.sum() / np.maximum(np.linalg.eigvalsh(gram), 0).sum()
    return mean, basis.T.astype(np.float32), sigma.astype(np.float32), held


def export(path, mean, basis, sigma, size):
    """`z -> mean + (z * sigma) @ basis`, reshaped to the frame `Decoder` reads."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    keep = basis.shape[0]
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["z", "sigma"], ["scaled"]),
            helper.make_node("MatMul", ["scaled", "basis"], ["flat"]),
            helper.make_node("Add", ["flat", "mean"], ["shifted"]),
            helper.make_node("Reshape", ["shifted", "shape"], ["image"]),
        ],
        "pixel_pca",
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, keep])],
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size])],
        [
            numpy_helper.from_array(sigma.reshape(1, keep), "sigma"),
            numpy_helper.from_array(basis.astype(np.float32), "basis"),
            numpy_helper.from_array(mean.astype(np.float32).reshape(1, -1), "mean"),
            numpy_helper.from_array(np.array([1, 3, size, size], dtype=np.int64), "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    pathlib.Path(path).write_bytes(model.SerializeToString())
    return path


def frame(mean, basis, sigma, z, size):
    got = mean + (z * sigma) @ basis
    return ((np.clip(got.reshape(3, size, size), -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)


def sheet(mean, basis, sigma, size, path, seed=0):
    """The mean, the first components either way, and a walk between two points."""
    rng = np.random.default_rng(seed)
    keep = basis.shape[0]
    rows = [np.concatenate([frame(mean, basis, sigma, np.zeros(keep, np.float32), size)] * 5, 1)]
    for k in range(3):
        tiles = []
        for t in (-2.5, -1.25, 0.0, 1.25, 2.5):
            z = np.zeros(keep, np.float32)
            z[k] = t
            tiles.append(frame(mean, basis, sigma, z, size))
        rows.append(np.concatenate(tiles, 1))
    a, b = rng.standard_normal(keep).astype(np.float32), rng.standard_normal(keep).astype(np.float32)
    walk = []
    for t in np.linspace(0, 1, 5):
        z = (1 - t) * a + t * b
        walk.append(frame(mean, basis, sigma, z * np.sqrt(keep) / np.linalg.norm(z), size))
    rows.append(np.concatenate(walk, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    return path


def grid(basis, size, path, across=10, tile=72):
    """Every component as its own picture, each stretched to its own range so the pattern shows."""
    tiles = []
    for v in basis:
        a = v.reshape(3, size, size).transpose(1, 2, 0)
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        im = Image.fromarray((a * 255).astype(np.uint8)).resize((tile, tile), Image.LANCZOS)
        tiles.append(np.asarray(im))
    while len(tiles) % across:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i : i + across], 1) for i in range(0, len(tiles), across)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="Where to write the .onnx the Decoder node runs.")
    ap.add_argument("--components", type=int, default=50)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--detail", type=float, default=1.0, help="How much of a frame one crop sees.")
    ap.add_argument("--crops", type=int, default=1)
    args = ap.parse_args()

    x = pixels(args.images, args.size, args.limit, args.detail, args.crops)
    print(f"{x.shape[1]} numbers per crop; solving in sample space")
    mean, basis, sigma, held = components(x, args.components)
    print(f"top {args.components} components hold {held:.1%} of the pixel variance")
    print(f"wrote {export(args.out, mean, basis, sigma, args.size)}")
    print(f"preview {sheet(mean, basis, sigma, args.size, str(pathlib.Path(args.out).with_suffix('.png')))}")
    print(f"components {grid(basis, args.size, str(pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + '-grid.png')))}")
    print("In goofi: Decoder with `file` set to it. Its input is whitened, so `spread` is a truncation.")


if __name__ == "__main__":
    main()
