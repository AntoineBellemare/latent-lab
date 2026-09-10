"""Train the rule `graphics:NeuralCA` runs, against the style of a folder of images.

The model here is the shader, written again in torch: the same four filters, the same order in the
perception vector, the same hidden layer, the same `tanh` on the step and the same clamp. Anything
that drifts between the two shows up as a texture that trains well and runs wrong, so the two are
meant to be read side by side.

    python style_ca.py --images ~/pictures/lichen --out lichen.npy

Wire the file it writes into `NeuralCA`'s `weights` through the `Weights` node, and leave
`alive` off, `start` at zero, `rate` at 1.0 and `fire` at 0.5 — the settings it was trained at.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CH = 12
GROUPS = CH // 4
FILTERS = 4
PERC = CH * FILTERS
BLOCK = CH + 1 + GROUPS
BOUND = 8.0
FIRE = 0.5
STYLE_LAYERS = (1, 6, 11, 18, 25)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
LAPLACE = [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]
IDENTITY = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
KERNELS = [(IDENTITY, 1.0), (SOBEL_X, 8.0), (SOBEL_Y, 8.0), (LAPLACE, 16.0)]


class Rule(torch.nn.Module):
    """`rule()` in NeuralCA.wgsl: perception, one hidden layer, `tanh` on the step."""

    def __init__(self, hidden):
        super().__init__()
        self.layer1 = torch.nn.Linear(PERC, hidden)
        self.layer2 = torch.nn.Linear(hidden, CH)
        torch.nn.init.zeros_(self.layer2.weight)
        torch.nn.init.zeros_(self.layer2.bias)
        stack = [torch.tensor(k, dtype=torch.float32) / d for k, d in KERNELS]
        self.register_buffer("taps", torch.stack(stack).unsqueeze(1).repeat(1, CH, 1, 1))

    def perceive(self, x):
        wrapped = F.pad(x, (1, 1, 1, 1), mode="circular")
        seen = [F.conv2d(wrapped, self.taps[f].unsqueeze(1), groups=CH) for f in range(FILTERS)]
        return torch.cat(seen, dim=1)

    def forward(self, x):
        seen = self.perceive(x).permute(0, 2, 3, 1)
        delta = torch.tanh(self.layer2(F.relu(self.layer1(seen)))).permute(0, 3, 1, 2)
        fires = (torch.rand_like(x[:, :1]) < FIRE).float()
        return torch.clamp(x + fires * delta, -BOUND, BOUND)


def to_rgb(x):
    """`shade()` in the shader."""
    return torch.clamp(x[:, :3] * 0.5 + 0.5, 0.0, 1.0)


def pack(rule):
    """The weights as the shader reads them: one contiguous block per hidden unit, four to a texel."""
    w1 = rule.layer1.weight.detach().cpu()
    b1 = rule.layer1.bias.detach().cpu()
    w2 = rule.layer2.weight.detach().cpu()
    b2 = rule.layer2.bias.detach().cpu()
    hidden = w1.shape[0]
    out = torch.zeros(BLOCK * hidden + GROUPS, 4)
    for u in range(hidden):
        block = u * BLOCK
        out[block : block + CH] = w1[u].reshape(CH, 4)
        out[block + CH, 0] = b1[u]
        out[block + CH + 1 : block + BLOCK] = w2[:, u].reshape(GROUPS, 4)
    out[BLOCK * hidden :] = b2.reshape(GROUPS, 4)
    return out.numpy().astype(np.float32).reshape(1, -1, 4)


def unpack(packed, rule):
    """`pack` backwards, so a round trip can be checked rather than believed."""
    flat = torch.tensor(packed.reshape(-1, 4))
    hidden = (flat.shape[0] - GROUPS) // BLOCK
    assert hidden == rule.layer1.weight.shape[0], "the file names a different hidden width"
    for u in range(hidden):
        block = u * BLOCK
        rule.layer1.weight.data[u] = flat[block : block + CH].reshape(PERC)
        rule.layer1.bias.data[u] = flat[block + CH, 0]
        rule.layer2.weight.data[:, u] = flat[block + CH + 1 : block + BLOCK].reshape(CH)
    rule.layer2.bias.data = flat[BLOCK * hidden :].reshape(CH).clone()
    return rule


def images(where, size, limit, detail, crops):
    """Every image under `where`, as `crops` random squares each.

    `detail` is how much of the frame one crop sees: 1 is the whole of it, 8 is an eighth. A macro
    photograph resized whole into a 128-pixel square is a thumbnail of a scene, and the grain the
    style lives in is the first thing that resizing throws away.
    """
    root = pathlib.Path(where).expanduser()
    found = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not found:
        raise SystemExit(f"no images under {root}")
    print(f"{len(found)} images under {root}, reading {min(len(found), limit)} at {crops} crops each")
    side = max(size, round(size * detail))
    out = []
    for path in found[:limit]:
        try:
            im = Image.open(path).convert("RGB")
        except OSError as e:
            print(f"  skipped {path.name}: {e}")
            continue
        scale = side / min(im.size)
        im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), Image.LANCZOS)
        pixels = np.asarray(im, dtype=np.float32) / 255.0
        for _ in range(crops):
            top = np.random.randint(0, pixels.shape[0] - size + 1)
            left = np.random.randint(0, pixels.shape[1] - size + 1)
            out.append(torch.tensor(pixels[top : top + size, left : left + size]).permute(2, 0, 1))
    return torch.stack(out)


class Style(torch.nn.Module):
    """The Gram matrices of a set of images, and the distance of anything else from them."""

    def __init__(self, target, device):
        super().__init__()
        import torchvision

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[: max(STYLE_LAYERS) + 1].eval().to(device)
        for q in self.features.parameters():
            q.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1).to(device))
        total = None
        with torch.no_grad():
            for i in range(0, len(target), 8):
                got = [g.sum(0) for g in self.grams(target[i : i + 8].to(device))]
                total = got if total is None else [a + b for a, b in zip(total, got)]
        self.want = [g / len(target) for g in total]

    def grams(self, rgb):
        x = (rgb - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in STYLE_LAYERS:
                b, c, h, w = x.shape
                flat = x.reshape(b, c, h * w)
                out.append(flat @ flat.transpose(1, 2) / (c * h * w))
        return out

    def forward(self, rgb):
        return sum(F.mse_loss(g, want.expand_as(g)) for g, want in zip(self.grams(rgb), self.want))


def train(args, device):
    target = images(args.images, args.size, args.limit, args.detail, args.crops)
    style = Style(target, device)
    # Two rules trained from nothing put their hidden units in their own arbitrary order, so the
    # average of the two is the average of unrelated things and draws mush. Children of ONE parent
    # keep that order, and then a blend of their weights is a blend of their textures.
    rule = Rule(args.hidden)
    if args.parent:
        unpack(np.load(args.parent), rule)
    rule = rule.to(device)
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimiser, [args.steps // 2, args.steps * 3 // 4], 0.3)
    pool = torch.zeros(args.pool, CH, args.size, args.size, device=device)

    for step in range(args.steps):
        pick = torch.randint(0, args.pool, (args.batch,), device=device)
        x = pool[pick]
        # One entry starts over every step, so the rule never stops being able to grow from nothing.
        x[0] = 0.0
        for _ in range(int(torch.randint(args.min_steps, args.max_steps, (1,)))):
            x = rule(x)
        # Values outside the drawable range cost nothing in a Gram matrix, so they are priced here.
        overflow = (x - x.clamp(-1.0, 1.0)).abs().mean()
        loss = style(to_rgb(x)) + args.overflow * overflow
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        # A per-parameter normalise: NCA gradients span orders of magnitude between steps, and Adam
        # alone rides that into a rule that diverges.
        for q in rule.parameters():
            q.grad /= q.grad.norm() + 1e-8
        optimiser.step()
        schedule.step()
        pool[pick] = x.detach()
        if step % args.report == 0 or step == args.steps - 1:
            print(f"step {step:>6}  style {loss.item():.5f}  overflow {overflow.item():.5f}", flush=True)

    return rule


def half_holds(rule, device, size, steps):
    """The shader stores state and weights as f16. A rule that only survives f32 is not trained."""
    packed = pack(rule)
    rounded = torch.tensor(packed.reshape(-1, 4)).half().float().numpy().reshape(packed.shape)
    checked = unpack(rounded, Rule(rule.layer1.weight.shape[0])).to(device)
    x = torch.zeros(1, CH, size, size, device=device)
    with torch.no_grad():
        for _ in range(steps):
            x = checked(x).half().float()
    rgb = to_rgb(x)
    return bool(torch.isfinite(x).all()), float(rgb.std())


def preview(rule, device, size, steps, path):
    x = torch.zeros(1, CH, size, size, device=device)
    with torch.no_grad():
        for _ in range(steps):
            x = rule(x)
    frame = (to_rgb(x)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(frame).save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of images, read recursively.")
    ap.add_argument("--out", required=True, help="Where to write the .npy the Weights node loads.")
    ap.add_argument("--hidden", type=int, default=96, help="Hidden units. The shader reads this back.")
    ap.add_argument("--parent", default="", help="A trained .npy to start from, so this and its siblings blend.")
    ap.add_argument("--size", type=int, default=128, help="The grid trained on, and the crop taken.")
    ap.add_argument("--limit", type=int, default=256, help="How many images the style target averages.")
    ap.add_argument("--detail", type=float, default=1.0, help="How much of a frame one crop sees: 1 the whole of it, 8 an eighth.")
    ap.add_argument("--crops", type=int, default=1, help="Random crops taken per image.")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--min-steps", type=int, default=32, help="Ticks run before the loss is taken.")
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--overflow", type=float, default=1.0)
    ap.add_argument("--report", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.parent:
        args.hidden = (np.load(args.parent).reshape(-1, 4).shape[0] - GROUPS) // BLOCK
        print(f"starting from {args.parent}, {args.hidden} hidden units")
    print(f"training on {device}")
    rule = train(args, device)

    packed = pack(rule)
    back = unpack(packed, Rule(args.hidden).to(device))
    for a, b in zip(rule.state_dict().values(), back.state_dict().values()):
        assert torch.allclose(a.cpu(), b.cpu()), "pack and unpack disagree"

    np.save(args.out, packed)
    print(f"wrote {args.out}: {packed.shape}, {args.hidden} hidden units")

    finite, spread = half_holds(rule, device, args.size, 200)
    print(f"f16 after 200 ticks: finite {finite}, spread {spread:.4f}")
    if not finite or spread < 0.01:
        print("  this rule does not survive half precision, and the shader stores f16", file=sys.stderr)

    print(f"preview: {preview(rule, device, 256, 300, str(pathlib.Path(args.out).with_suffix('.png')))}")


if __name__ == "__main__":
    main()
