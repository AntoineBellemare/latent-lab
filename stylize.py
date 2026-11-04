"""Carry a picture toward a texture's feature STATISTICS — the classical resemblance baseline.

    python stylize.py --image portrait.png --style bark.jpg --out sheet.png
    python stylize.py --image portrait.png --style bark.jpg water.jpg --between 5 --out morph.png

Where `composite.py` keeps the texture's pixels, this keeps only its statistics: Gram matrices (or
AdaIN's mean and variance) of VGG features, pushed onto the content by optimisation, Gatys's
original move. No decoder is trained and none is downloaded — the image itself is the variable.
That makes it slow (seconds to a minute a tile) and honest: this is what "resemblance without
semantics" looks like, the floor any diffusion transfer has to clear to be worth its weights.

Two more losses match the features' whole DISTRIBUTIONS, not just their moments: `swd`, the sliced
Wasserstein distance (Heitz et al. 2021) — the features projected on fixed random directions,
each projection sorted and matched to the style's, so cross-channel structure counts; and `efdm`,
exact feature distribution matching (Zhang et al. 2022) — each channel sorted and matched on its
own. Where Gram matching averages a texture's contrast away, these keep its extremes: the black of
a crack, the glint of a crystal.

`--tradeoff` sweeps how hard the style presses against the content, one column each. `--between`
draws one row morphing the style TARGET between two references — statistics interpolate cleanly
even when pixels cannot, which is the one trick this baseline owns outright.
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from composite import fit

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
STYLE_AT = [1, 6, 11, 20, 29]  # relu1_1 .. relu5_1 in torchvision's vgg19.features
CONTENT_AT = 22  # relu4_2
SWAP_AT = 11  # relu3_1, where neural patches live


def swap_target(vgg, content, style, layer=SWAP_AT, patch=3):
    """The style-swap target: the content's feature map rebuilt from the style's own patches.

    Every 3x3 neighbourhood of the content's relu3_1 features is replaced by the style's most
    correlated neighbourhood — nearest-neighbour matching inside the network, found with one big
    convolution whose filters ARE the style's patches. Optimising toward this keeps the style's
    texture ELEMENTS intact where Gram matching keeps only their statistics.
    """
    with torch.no_grad():
        fc = features(vgg, content, [layer])[layer]
        fs = features(vgg, style, [layer])[layer]
        pats = F.unfold(fs, patch)[0].T  # every style patch, flattened
        filt = F.normalize(pats, dim=1).view(-1, fs.shape[1], patch, patch)
        best = F.conv2d(fc, filt).argmax(1)[0]
        chosen = pats[best.flatten()].T[None]
        fold = torch.nn.Fold(output_size=fc.shape[2:], kernel_size=patch)
        return fold(chosen) / fold(torch.ones_like(chosen))


def features(vgg, x, wanted):
    out, seen = {}, x
    for i, layer in enumerate(vgg):
        seen = layer(seen)
        if i in wanted:
            out[i] = seen
        if i >= max(wanted):
            break
    return out


def gram(f):
    b, c, h, w = f.shape
    flat = f.view(c, h * w)
    return (flat @ flat.T) / (c * h * w)


def stats(f):
    return f.mean((2, 3)), f.std((2, 3))


def target_of(vgg, img, loss):
    with torch.no_grad():
        got = features(vgg, img, STYLE_AT)
    if loss == "gram":
        return {i: gram(got[i]) for i in STYLE_AT}
    if loss in DISTRIBUTIONS:
        out = {}
        for i in STYLE_AT:
            f = got[i][0].flatten(1)  # (c, n)
            # Fixed directions, seeded per layer: L-BFGS needs one objective, not a new one each
            # evaluation. Twice as many as there are channels.
            dirs = None
            if loss == "swd":
                gen = torch.Generator(device="cpu").manual_seed(i)
                dirs = F.normalize(torch.randn(2 * f.shape[0], f.shape[0], generator=gen), dim=1).to(f.device)
                f = dirs @ f
            out[i] = (f.sort(1).values, dirs)
        return out
    return {i: stats(got[i]) for i in STYLE_AT}


DISTRIBUTIONS = {"swd", "efdm"}


def _sorted_loss(f, target):
    """Each row of f sorted and matched to the target's sorted row, resampled to f's length."""
    want, dirs = target
    if dirs is not None:
        f = dirs @ f
    got = f.sort(1).values
    n, m = got.shape[1], want.shape[1]
    if n != m:
        pos = torch.linspace(0, m - 1, n, device=f.device)
        lo = pos.floor().long().clamp(max=m - 2)
        frac = pos - lo
        want = want[:, lo] * (1 - frac) + want[:, lo + 1] * frac
    return F.mse_loss(got, want)


def lerp_target(a, b, t, loss):
    if loss == "gram":
        return {i: (1 - t) * a[i] + t * b[i] for i in a}
    return {i: ((1 - t) * a[i][0] + t * b[i][0], (1 - t) * a[i][1] + t * b[i][1]) for i in a}


def transfer(vgg, content, target, loss, tradeoff, steps=250, *, swap_at=SWAP_AT, patch=3,
             stages=4, palette_w=1e4, init="content", watch=None):
    with torch.no_grad():
        want_content = features(vgg, content, [CONTENT_AT])[CONTENT_AT]
    x = content.clone().contiguous()
    if init == "mixed":
        x = x + 0.3 * torch.randn_like(x)
    x = x.requires_grad_(True)

    if loss == "patch":
        # The swap is re-made as the image changes (CNNMRF's move): a smooth portrait first matches
        # the style's flattest patches, but each stage of texture earns richer matches, and the
        # material compounds. `target` arrives here as the STYLE image, not a precomputed map.
        # Deep patches are nearly colour-blind, so the two shallow layers carry the palette by Gram.
        with torch.no_grad():
            shallow = features(vgg, target, [1, 6])
            palette = {i: gram(shallow[i]) for i in (1, 6)}
        for stage in range(max(stages, 1)):
            tgt = swap_target(vgg, x.detach(), target, layer=swap_at, patch=patch)
            opt = torch.optim.LBFGS([x], max_iter=max(steps // max(stages, 1), 20), line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                got = features(vgg, x, sorted({1, 6, swap_at, CONTENT_AT}))
                colour = F.mse_loss(gram(got[1]), palette[1]) + F.mse_loss(gram(got[6]), palette[6])
                total = (tradeoff * F.mse_loss(got[swap_at], tgt) + palette_w * colour
                         + F.mse_loss(got[CONTENT_AT], want_content))
                total.backward()
                return total

            opt.step(closure)
            if watch:
                watch(stage, image(x.detach()))
        return x.detach()

    opt = torch.optim.LBFGS([x], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        got = features(vgg, x, sorted(set(STYLE_AT + [CONTENT_AT])))
        style = torch.zeros((), device=x.device)
        for i in STYLE_AT:
            if loss == "gram":
                style = style + F.mse_loss(gram(got[i]), target[i])
            elif loss in DISTRIBUTIONS:
                style = style + _sorted_loss(got[i][0].flatten(1), target[i])
            else:
                m, s = stats(got[i])
                style = style + F.mse_loss(m, target[i][0]) + F.mse_loss(s, target[i][1])
        total = tradeoff * style + F.mse_loss(got[CONTENT_AT], want_content)
        total.backward()
        return total

    opt.step(closure)
    return x.detach()


def tensor(im, device):
    a = torch.tensor(np.asarray(im, np.float32) / 255.0, device=device).permute(2, 0, 1)[None]
    return (a - torch.tensor(MEAN, device=device).view(1, 3, 1, 1)) / torch.tensor(STD, device=device).view(1, 3, 1, 1)


def image(x):
    a = x[0].permute(1, 2, 0).cpu().numpy() * np.array(STD) + np.array(MEAN)
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--style", nargs="+", required=True, help="Texture photographs; one row each.")
    ap.add_argument("--loss", choices=["gram", "adain", "swd", "efdm", "patch"], default="gram",
                    help="gram/adain match statistics; patch style-swaps relu3_1 neighbourhoods (re-swapped in "
                         "stages, palette carried by the shallow layers), so the style's texture ELEMENTS "
                         "survive. Its pressure saturates — the swapped target is built from the image's own "
                         "matches, so it is always nearly reachable — and ~2000 is plenty; sweeping it is "
                         "not the knob it is for gram.")
    ap.add_argument("--tradeoff", type=float, nargs="*", default=[3e3, 3e4, 3e5], help="Style pressure, one column each.")
    ap.add_argument("--between", type=int, default=0, help="Also a row morphing the target from the first style to the second.")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from torchvision.models import vgg19, VGG19_Weights

    device = torch.device(args.device)
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    content = tensor(fit(args.image, args.size), device)
    styles = {s: tensor(fit(s, args.size), device) for s in args.style}
    if args.loss == "patch":
        targets = dict(styles)  # the swap is re-made per stage inside transfer
    else:
        targets = {s: target_of(vgg, styles[s], args.loss) for s in args.style}

    rows = []
    for s in args.style:
        tiles = [fit(s, args.size)]
        for t in args.tradeoff:
            tiles.append(image(transfer(vgg, content, targets[s], args.loss, t, args.steps)))
            print(f"  {pathlib.Path(s).stem} @ {t:g}", flush=True)
        rows.append((f"{args.loss} · {pathlib.Path(s).stem} · tradeoff {', '.join(f'{t:g}' for t in args.tradeoff)}", tiles))

    if args.between >= 2 and args.loss in DISTRIBUTIONS | {"patch"}:
        print("  (the morph row interpolates moments; a swapped or sorted target is skipped)")
    if args.between >= 2 and len(args.style) >= 2 and args.loss not in DISTRIBUTIONS | {"patch"}:
        a, b = targets[args.style[0]], targets[args.style[1]]
        mid = args.tradeoff[len(args.tradeoff) // 2]
        tiles = []
        for t in np.linspace(0, 1, args.between):
            tiles.append(image(transfer(vgg, content, lerp_target(a, b, float(t), args.loss), args.loss, mid, args.steps)))
            print(f"  between @ {t:.2f}", flush=True)
        rows.append((f"statistics morph, {pathlib.Path(args.style[0]).stem} to {pathlib.Path(args.style[1]).stem}", tiles))

    side, label = args.size, 24
    cols = max(len(t) for _, t in rows)
    sheet = Image.new("RGB", (cols * side, label + side + (side + label) * len(rows)), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    pen.text((6, 4), f"the picture: {pathlib.Path(args.image).stem}", fill=(214, 210, 202))
    sheet.paste(fit(args.image, args.size), (0, label))
    offset = label + side
    for r, (title, tiles) in enumerate(rows):
        y = offset + r * (side + label)
        pen.text((6, y + 4), title, fill=(214, 210, 202))
        for c, tile in enumerate(tiles):
            sheet.paste(tile, (c * side, y + label))
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
