"""Train a small generator on a folder of images, and write the `.onnx` the `Decoder` node runs.

FastGAN (Liu et al., ICLR 2021), which is the recipe for this shape of problem: a few thousand
images, one GPU, hours rather than weeks. Three things carry it and all three are here — skip-layer
excitation, so a 4x4 feature still steers a 256x256 one; a discriminator that must RECONSTRUCT what
it judges, which is what stops it memorising; and differentiable augmentation, without which a small
set is learnt by heart in an afternoon.

    python fastgan.py --images ~/pictures/plates --out plates.onnx

The generator alone is exported. It takes `[1, latent]` and answers `[1, 3, size, size]` through a
`tanh`, which is what `Decoder`'s `range` of `signed` expects.
"""

import argparse
import copy
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# The paper's channel schedule, as a multiple of `ngf`.
WIDTH = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}


def channels(res, ngf):
    return max(8, int(WIDTH[res] * ngf))


def up(inp, out):
    """Upsample, then a convolution whose gate halves the channels back down.

    Bilinear rather than the paper's nearest: nearest is a box filter and it aliases, which shows up
    as texture STICKING — fine detail glued to the screen while the structure morphs underneath it.
    Slow interpolation is the whole point here, and that is exactly where the artifact lives.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(inp, out * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out * 2),
        nn.GLU(dim=1),
    )


class Excite(nn.Module):
    """Skip-layer excitation: a small feature map scales a large one, channel by channel."""

    def __init__(self, small, big):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(small, big, 4, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(big, big, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, small, big):
        return big * self.gate(small)


class Mapping(nn.Module):
    """`z` to `w`. StyleGAN's cheapest idea and its most useful one: the generator is far easier to
    steer through a latent it can use directly than through one it must disentangle on the way in."""

    def __init__(self, latent, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(latent, latent), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*layers)
        # Torch's default linear init shrinks by 0.58 a layer and the leak by another 0.72, so four
        # layers hand the synthesis network a `w` some thirty times smaller than the `z` that made
        # it — and a path length inflated by exactly that, which is the number the regulariser then
        # chases. Init for unit gain instead.
        for layer in (m for m in self.net if isinstance(m, nn.Linear)):
            nn.init.normal_(layer.weight, 0.0, (2.0 / latent) ** 0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, z):
        return self.net(z * torch.rsqrt(z.pow(2).mean(1, keepdim=True) + 1e-8))


class Generator(nn.Module):
    def __init__(self, latent, size, ngf, depth=4, classes=0):
        super().__init__()
        self.latent = latent
        self.size = size
        self.classes = classes
        if classes:
            # The category shifts `z` before the mapping, so ONE latent serves every category and a
            # knob keeps its direction while the material changes under it. Small to start with, so a
            # run begun from an unconditional parent draws what the parent drew until this earns its say.
            self.tag = nn.Linear(classes, latent, bias=False)
            nn.init.normal_(self.tag.weight, 0.0, 0.02)
        self.mapping = Mapping(latent, depth)
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(latent, channels(4, ngf) * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels(4, ngf) * 2),
            nn.GLU(dim=1),
        )
        self.steps = nn.ModuleList(
            [up(channels(r, ngf), channels(r * 2, ngf)) for r in resolutions(size)]
        )
        # Each excitation reaches four doublings up, which is where the paper puts them.
        self.excites = nn.ModuleList(
            [Excite(channels(r, ngf), channels(r * 16, ngf)) for r in excited(size)]
        )
        self.out = nn.Conv2d(channels(size, ngf), 3, 3, 1, 1, bias=False)

    def synthesis(self, w):
        x = self.stem(w.reshape(-1, self.latent, 1, 1))
        held = {4: x}
        for r, step in zip(resolutions(self.size), self.steps):
            x = step(x)
            small = r * 2 // 16
            if small in held:
                x = self.excites[excited(self.size).index(small)](held[small], x)
            held[r * 2] = x
        return torch.tanh(self.out(x))

    def steered(self, z, hot):
        """`z` with its category folded in, which is what the mapping network is given."""
        return z + self.tag(hot) if self.classes else z

    def forward(self, z, hot=None):
        return self.synthesis(self.mapping(self.steered(z, hot)))


def resolutions(size):
    """The doublings from 4 up to `size`."""
    return [r for r in sorted(WIDTH) if 4 <= r < size]


def excited(size):
    """The small maps that steer a large one sixteen times their width."""
    return [r for r in sorted(WIDTH) if r >= 4 and r * 16 <= size]


class Discriminator(nn.Module):
    """Down to 8x8 for the verdict, and back up from it for the reconstruction that regularises it."""

    def __init__(self, size, ndf, classes=0):
        super().__init__()
        self.classes = classes
        steps = [nn.Conv2d(3, channels(size, ndf), 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)]
        for r in reversed([r for r in sorted(WIDTH) if 8 < r <= size // 2]):
            steps += [
                nn.Conv2d(channels(r * 2, ndf), channels(r, ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels(r, ndf)),
                nn.LeakyReLU(0.2, True),
            ]
        self.down = nn.Sequential(*steps)
        self.verdict = nn.Conv2d(channels(16, ndf), 1, 4, 1, 0, bias=False)
        if classes:
            # Projection conditioning. Without it the generator can ignore the category altogether:
            # a critic that cannot tell which category it is shown cannot punish the wrong one. Zero
            # to start with, so a warm start's verdict is its parent's until the class earns its say.
            self.tag = nn.Linear(classes, channels(16, ndf), bias=False)
            nn.init.zeros_(self.tag.weight)
        self.whole = decoder(channels(16, ndf), ndf)
        self.part = decoder(channels(16, ndf), ndf)

    def forward(self, x, hot=None):
        f = self.down(x)
        quarter = random.randint(0, 3)
        half = f[:, :, (quarter // 2) * 4 : (quarter // 2) * 4 + 4, (quarter % 2) * 4 : (quarter % 2) * 4 + 4]
        spoken = self.verdict(f).flatten(1).mean(1)
        if self.classes and hot is not None:
            spoken = spoken + (f.mean((2, 3)) * self.tag(hot)).sum(1)
        return spoken, self.whole(f), self.part(half), quarter


def decoder(inp, ndf):
    """8x8 (or 4x4) of features back to a 32x32 picture: small, because it is a regulariser."""
    return nn.Sequential(
        up(inp, channels(64, ndf)),
        up(channels(64, ndf), channels(128, ndf)),
        nn.Conv2d(channels(128, ndf), 3, 3, 1, 1, bias=False),
        nn.Tanh(),
    )


def augment(x):
    """DiffAugment: colour, translation and cutout, all differentiable, all on the GPU."""
    b = x.shape[0]
    x = x + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    grey = x.mean(dim=1, keepdim=True)
    x = (x - grey) * (torch.rand(b, 1, 1, 1, device=x.device) * 2) + grey
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=x.device) + 0.5) + mean
    x = translate(x)
    return cutout(x)


def translate(x, ratio=0.125):
    b, _, h, w = x.shape
    shift = (int(h * ratio + 0.5), int(w * ratio + 0.5))
    dy = torch.randint(-shift[0], shift[0] + 1, (b, 1, 1), device=x.device)
    dx = torch.randint(-shift[1], shift[1] + 1, (b, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).reshape(1, h, 1) + dy + 1
    xs = torch.arange(w, device=x.device).reshape(1, 1, w) + dx + 1
    padded = F.pad(x, (1, 1, 1, 1))
    rows = torch.arange(b, device=x.device).reshape(b, 1, 1)
    return padded.permute(0, 2, 3, 1)[rows, ys.clamp(0, h + 1), xs.clamp(0, w + 1)].permute(0, 3, 1, 2)


def cutout(x, ratio=0.5):
    b, _, h, w = x.shape
    box = (int(h * ratio + 0.5), int(w * ratio + 0.5))
    cy = torch.randint(0, h + (1 - box[0] % 2), (b, 1, 1), device=x.device)
    cx = torch.randint(0, w + (1 - box[1] % 2), (b, 1, 1), device=x.device)
    ys = torch.arange(box[0], device=x.device).reshape(1, -1, 1) + cy - box[0] // 2
    xs = torch.arange(box[1], device=x.device).reshape(1, 1, -1) + cx - box[1] // 2
    mask = torch.ones(b, h, w, device=x.device)
    rows = torch.arange(b, device=x.device).reshape(b, 1, 1)
    mask[rows, ys.clamp(0, h - 1), xs.clamp(0, w - 1)] = 0
    return x * mask.unsqueeze(1)


class Folder(torch.utils.data.Dataset):
    """Every image under a root, held resized in memory and cropped afresh each time it is asked for.

    `detail` is how much of a frame one crop sees: 1 the whole of it, 3 a third. On a photographic
    set it is both the scale the model learns and the whole of the augmentation — sixty-seven frames
    are sixty-seven pictures at 1, and thousands of distinct squares at 3.

    Held decoded, because a 24-megapixel JPEG takes longer to open than the step it feeds.
    """

    def __init__(self, where, size, detail, classes=False):
        files = sorted(p for p in pathlib.Path(where).expanduser().rglob("*") if p.suffix.lower() in SUFFIXES)
        if not files:
            raise SystemExit(f"no images under {where}")
        # A class is the folder an image sits in, which is what `by_category.py` writes.
        self.names = sorted({p.parent.name for p in files}) if classes else []
        self.tags = [self.names.index(p.parent.name) for p in files] if classes else [0] * len(files)
        self.size = size
        side = max(size, round(size * detail))
        self.held = []
        for path in files:
            im = Image.open(path).convert("RGB")
            scale = side / min(im.size)
            im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), Image.LANCZOS)
            self.held.append(np.asarray(im, dtype=np.uint8))

    def __len__(self):
        return len(self.held)

    def __getitem__(self, i):
        held = self.held[i]
        top = random.randint(0, held.shape[0] - self.size)
        left = random.randint(0, held.shape[1] - self.size)
        crop = held[top : top + self.size, left : left + self.size]
        if random.random() < 0.5:
            crop = crop[:, ::-1]
        got = torch.tensor(np.ascontiguousarray(crop), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        return got, self.tags[i]


def quadrant(x, which):
    half = x.shape[-1] // 2
    return x[:, :, (which // 2) * half : (which // 2) * half + half, (which % 2) * half : (which % 2) * half + half]


def r1(dis, real, hot, gamma, every):
    """The gradient of the verdict at a real image, penalised.

    Nothing else bounds the discriminator's scale — the hinge is scale-sensitive, the verdict head
    carries no normalisation, and Adam has no weight decay — so the cheapest way for it to clear the
    margin is to grow. Once it has, both hinge terms are zero, its gradient dies, and the generator
    is left chasing a critic that has stopped learning. That is the failure this run showed twice.
    """
    real = real.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(dis(real, hot)[0].sum(), real, create_graph=True)
    return (gamma / 2) * every * grad.square().sum([1, 2, 3]).mean()


def path_length(gen, w, image, mean):
    """StyleGAN2's path length: how far the picture moves per unit of `w`.

    Held constant, it makes the latent space a CONTROL SURFACE — the same step in a signal moves the
    image by the same amount wherever it is standing. For an instrument driven by a body that matters
    more than fidelity does, and there is no way to bolt it on afterwards.
    """
    noise = torch.randn_like(image) / (image.shape[2] * image.shape[3]) ** 0.5
    (grad,) = torch.autograd.grad((image * noise).sum(), w, create_graph=True)
    lengths = grad.square().sum(1).sqrt()
    return (lengths - mean).square().mean(), lengths.detach().mean()


class Steered(nn.Module):
    """The generator as `Decoder` reads it: whitened coefficients of `w`'s own principal axes.

    So the wired numbers arrive ORDERED — axis 0 is the direction the model varies along most — and
    a unit Gaussian is the right thing to feed it, which is what `spread` and `seed` already assume.
    """

    def __init__(self, gen, mean, basis, sigma):
        super().__init__()
        self.gen = gen
        self.register_buffer("mean", mean)
        self.register_buffer("basis", basis)
        self.register_buffer("sigma", sigma)

    def forward(self, z):
        return self.gen.synthesis(self.mean + (z * self.sigma) @ self.basis)


class Blended(nn.Module):
    """The conditional generator as a patch drives it: a blend of categories, and the axes WITHIN one.

    The mean of `w` moves with the blend while the axes stay shared, so a knob keeps its direction as
    the material changes under it, and the place between two categories is a place rather than a
    crossfade of two pictures.
    """

    def __init__(self, gen, means, basis, sigma):
        super().__init__()
        self.gen = gen
        self.register_buffer("means", means)
        self.register_buffer("basis", basis)
        self.register_buffer("sigma", sigma)

    def forward(self, z, category):
        share = category.clamp_min(0.0)
        total = share.sum(1, keepdim=True)
        # A patch that has wired nothing yet sends zeros, which is every category at once rather
        # than a division by nothing.
        share = torch.where(total > 0, share / total.clamp_min(1e-6), torch.full_like(share, 1.0 / share.shape[1]))
        return self.gen.synthesis(share @ self.means + (z * self.sigma) @ self.basis)


def axes_of(gen, latent, device, samples=8192, keep=None, classes=0):
    """The principal axes of `w`, by sampling the mapping network the space is reached through.

    With categories the mean is taken PER category, and the axes come from what is left once each
    category's own mean is removed. The directions are then within-category variation rather than
    the switch from one category to the next, which the category input already carries.
    """
    gen.eval()
    with torch.no_grad():
        if classes:
            rounds = max(1, samples // (512 * classes))
            means, centred = [], []
            for c in range(classes):
                hot = F.one_hot(torch.full((512,), c, device=device), classes).float()
                w = torch.cat(
                    [gen.mapping(gen.steered(torch.randn(512, latent, device=device), hot)) for _ in range(rounds)]
                )
                means.append(w.mean(0))
                centred.append(w - w.mean(0))
            middle, centred = torch.stack(means), torch.cat(centred)
        else:
            w = torch.cat([gen.mapping(torch.randn(512, latent, device=device)) for _ in range(samples // 512)])
            middle, centred = w.mean(0), w - w.mean(0)
        u, s, v = torch.linalg.svd(centred, full_matrices=False)
    gen.train()
    keep = keep or latent
    held = (s[:keep] ** 2).sum() / (s**2).sum()
    return middle, v[:keep].contiguous(), (s[:keep] / (len(centred) - 1) ** 0.5).contiguous(), float(held)


def export(gen, path, latent, size, device, keep=None, classes=0):
    """The generator alone, batch one, steered through `w`'s principal axes."""
    middle, basis, sigma, held = axes_of(gen, latent, device, keep=keep, classes=classes)
    if classes:
        model = Blended(gen, middle, basis, sigma).eval()
        given = (torch.zeros(1, basis.shape[0], device=device), torch.zeros(1, classes, device=device))
        names = ["z", "category"]
    else:
        model = Steered(gen, middle, basis, sigma).eval()
        given = (torch.zeros(1, basis.shape[0], device=device),)
        names = ["z"]
    torch.onnx.export(
        model,
        given,
        str(path),
        input_names=names,
        output_names=["image"],
        opset_version=17,
        dynamo=False,
    )
    gen.train()
    return path, basis.shape[0], held


def stash(path, step, gen, dis, smooth, opt_g, opt_d, path_mean, taken):
    """Everything a run needs to carry on. The `.onnx` is an export and cannot resume anything."""
    torch.save(
        {
            "step": step,
            "gen": gen.state_dict(),
            "dis": dis.state_dict(),
            "smooth": smooth.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "path_mean": path_mean,
            "taken": taken,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of images, read recursively.")
    ap.add_argument("--out", required=True, help="Where to write the .onnx the Decoder node runs.")
    ap.add_argument("--size", type=int, default=256, choices=[64, 128, 256, 512, 1024])
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--ngf", type=int, default=64, help="Generator width. Halve it for a smaller model.")
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="Carry on from the .pt beside --out.")
    ap.add_argument(
        "--init",
        help="Start from another run's weights, with a fresh optimizer and the step count back at "
        "zero. Children of one parent stay in its basin, so their weights still blend; two cold "
        "starts order their units differently and average to mush.",
    )
    ap.add_argument(
        "--classes",
        action="store_true",
        help="Take each image's category from the folder holding it and condition on it: ONE model "
        "with a category input a patch can blend, instead of one model per folder.",
    )
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r1", type=float, default=0.0, help="R1 gradient penalty on real images; 0 picks one from the resolution.")
    ap.add_argument("--r1-every", type=int, default=16, help="Steps between R1 penalties.")
    ap.add_argument("--mapping-lr", type=float, default=0.01, help="The mapping network's share of the learning rate.")
    ap.add_argument(
        "--tag-lr",
        type=float,
        default=1.0,
        help="The category embeddings' multiple of the learning rate. Both start near silent so a "
        "warm start draws what its parent drew, and at 1.0 they can stay that way: one run left "
        "three similar categories merged after 42,000 steps while a probe on the same crops told "
        "them apart nine times in ten.",
    )
    ap.add_argument("--ema", type=float, default=0.999, help="How much of the averaged generator to keep each step.")
    ap.add_argument("--path", type=float, default=0.5, help="Path length weight: 0 turns it off.")
    ap.add_argument("--path-every", type=int, default=8, help="Steps between path length penalties.")
    ap.add_argument("--path-start", type=int, default=3000, help="Steps before path length applies at all.")
    ap.add_argument("--keep", type=int, default=0, help="Axes to export; 0 keeps every one.")
    ap.add_argument("--snap", type=int, default=2000, help="Export and preview every this many steps.")
    ap.add_argument("--detail", type=float, default=1.0, help="How much of a frame one crop sees: 1 the whole of it, 3 a third.")
    ap.add_argument("--workers", type=int, default=0, help="Held decoded in memory, so a worker buys nothing and costs a copy.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.keep = args.keep or None
    # The usual scaling: the penalty covers a whole image, so it grows with the pixels in one.
    args.r1 = args.r1 or 0.0002 * args.size**2 / args.batch

    device = torch.device(args.device)
    data = Folder(args.images, args.size, args.detail, args.classes)
    classes = len(data.names)
    print(f"{len(data)} images under {args.images}, training at {args.size} on {device}, detail {args.detail}")
    if classes:
        counts = np.bincount(data.tags, minlength=classes)
        print("  " + ", ".join(f"{n} {c}" for n, c in zip(data.names, counts)), flush=True)
    # Drawn evenly, or the largest category is trained on three times as hard as the smallest and the
    # rare ones arrive underdrawn.
    sampler = (
        torch.utils.data.WeightedRandomSampler([1.0 / counts[t] for t in data.tags], len(data), replacement=True)
        if classes
        else None
    )
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0
    )

    gen = Generator(args.latent, args.size, args.ngf, classes=classes).to(device)
    dis = Discriminator(args.size, args.ndf, classes=classes).to(device)
    print(f"generator {sum(q.numel() for q in gen.parameters()) / 1e6:.1f}M parameters, r1 {args.r1:.2f}")
    # The mapping moves at a hundredth of the synthesis, which is StyleGAN's own remedy and is aimed
    # at exactly what this run measured: left at the same rate it runs away, and `w` ends up with one
    # direction holding most of its variance — a latent with three usable knobs instead of thirty.
    slow = [q for n, q in gen.named_parameters() if n.startswith("mapping.")]
    quick = [q for n, q in gen.named_parameters() if n.startswith("tag.")]
    rest = [q for n, q in gen.named_parameters() if not n.startswith(("mapping.", "tag."))]
    opt_g = torch.optim.Adam(
        [
            {"params": slow, "lr": args.lr * args.mapping_lr},
            {"params": quick, "lr": args.lr * args.tag_lr},
            {"params": rest},
        ],
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(
        [
            {"params": [q for n, q in dis.named_parameters() if n.startswith("tag.")], "lr": args.lr * args.tag_lr},
            {"params": [q for n, q in dis.named_parameters() if not n.startswith("tag.")]},
        ],
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    # Sampled and exported from an average of the recent weights, never the just-stepped ones: at
    # batch 8 a single adversarial step is noisy enough to make a good run look like a bad one.
    smooth = copy.deepcopy(gen).eval()
    for q in smooth.parameters():
        q.requires_grad_(False)
    path_mean = torch.zeros((), device=device)
    held_path, taken = 0.0, 0
    step, feed = 0, iter(loader)
    if args.init:
        was = torch.load(args.init, map_location=device, weights_only=False)
        for who, key in ((gen, "gen"), (dis, "dis"), (smooth, "smooth")):
            missing, unused = who.load_state_dict(was[key], strict=False)
            if missing or unused:
                print(f"  {key}: {len(missing)} weights are new here, {len(unused)} of its own unused", flush=True)
        print(f"started from {pathlib.Path(args.init).name}, which had run {was['step']} steps", flush=True)

    carry = pathlib.Path(args.out).with_suffix(".pt")
    if args.resume and carry.is_file():
        was = torch.load(carry, map_location=device, weights_only=False)
        gen.load_state_dict(was["gen"])
        dis.load_state_dict(was["dis"])
        smooth.load_state_dict(was["smooth"])
        try:
            opt_g.load_state_dict(was["opt_g"])
            opt_d.load_state_dict(was["opt_d"])
        except ValueError as e:
            # A flag that regroups the parameters — `--tag-lr` does — leaves the saved moments
            # unfittable. Rebuilding them costs a few hundred steps; refusing to resume costs the run.
            print(f"  the saved optimizer does not fit this run ({e}); its moments start fresh", flush=True)
        # Torch restores each group's rate from the checkpoint, so a resume would quietly train at the
        # rates the LAST run was given. The flags on THIS run decide them.
        for group, rate in zip(opt_g.param_groups, [args.lr * args.mapping_lr, args.lr * args.tag_lr, args.lr]):
            group["lr"] = rate
        for group, rate in zip(opt_d.param_groups, [args.lr * args.tag_lr, args.lr]):
            group["lr"] = rate
        path_mean, taken, step = was["path_mean"], was["taken"], was["step"]
        print(f"carrying on from {carry.name} at step {step}", flush=True)
    while step < args.steps:
        try:
            real, tags = next(feed)
            real = real.to(device)
        except StopIteration:
            feed = iter(loader)
            continue
        hot = F.one_hot(tags.to(device), classes).float() if classes else None
        # The generator is asked for categories the way the loader draws them: evenly.
        drawn = F.one_hot(torch.randint(classes, (args.batch,), device=device), classes).float() if classes else None
        w = gen.mapping(gen.steered(torch.randn(args.batch, args.latent, device=device), drawn))
        fake = gen.synthesis(w)

        shown, faked = augment(real), augment(fake.detach())
        verdict, whole, part, which = dis(shown, hot)
        # Clamped, because `augment` shifts brightness by up to a half and then stretches contrast,
        # so the target leaves the range the decoders' `tanh` can reach at all — an irreducible floor
        # on the one term that is supposed to keep the discriminator's trunk honest.
        small = F.interpolate(shown, size=whole.shape[-1], mode="area").clamp(-1.0, 1.0)
        rebuilt = F.mse_loss(whole, small) + F.mse_loss(part, quadrant(small, which))
        loss_d = F.relu(1.0 - verdict).mean() + F.relu(1.0 + dis(faked, drawn)[0]).mean() + rebuilt
        if args.r1 > 0 and step % args.r1_every == 0:
            loss_d = loss_d + r1(dis, shown, hot, args.r1, args.r1_every)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        loss_g = -dis(augment(fake), drawn)[0].mean()
        # Lazily, because the second-order graph is the expensive part and the target it holds is a
        # running average that a fraction of the steps estimates just as well.
        if args.path > 0 and step >= args.path_start and step % args.path_every == 0:
            w = gen.mapping(gen.steered(torch.randn(args.batch, args.latent, device=device), drawn)).requires_grad_(True)
            penalty, length = path_length(gen, w, gen.synthesis(w), path_mean)
            # An exact running mean until the exponential one would be slower.
            taken += 1
            path_mean = path_mean.lerp(length, max(0.01, 1.0 / taken))
            # The FIRST application only seeds the target. Against a cold mean of zero the penalty is
            # the whole squared length — thousands — and one Adam step of that flattens the generator
            # into the degenerate solution this regulariser has: ignore w, and every length is equal.
            if taken > 1:
                loss_g = loss_g + args.path * args.path_every * penalty
            held_path = float(length)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        # Ramped, and that ramp is not optional. At a flat 0.999 the average covers a thousand steps
        # of a generator that is still changing fast, and the blur and lost variance that produces
        # read exactly like mode collapse: measured 0.17 of the data's spread against 0.78 without it.
        beta = min(args.ema, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for a, b in zip(smooth.parameters(), gen.parameters()):
                a.lerp_(b.detach(), 1.0 - beta)
            for a, b in zip(smooth.buffers(), gen.buffers()):
                a.copy_(b)

        if step % 100 == 0:
            print(
                f"step {step:>7}  d {loss_d.item():.4f}  g {loss_g.item():.4f}  "
                f"rebuilt {rebuilt.item():.4f}  path {held_path:.4f}",
                flush=True,
            )
        if step and step % args.snap == 0:
            preview(smooth, args, device, step)
            where, keep, held = export(smooth, args.out, args.latent, args.size, device, args.keep, classes)
            # Both, always. Weight averaging can blur a generator into something that reads exactly
            # like mode collapse, and telling the two apart afterwards needs the unaveraged one too.
            raw = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + "-raw.onnx")
            export(gen, raw, args.latent, args.size, device, args.keep, classes)
            print(f"  exported {where} and {raw.name}: {keep} axes holding {held:.1%} of w", flush=True)
            stash(carry, step, gen, dis, smooth, opt_g, opt_d, path_mean, taken)
        step += 1

    preview(smooth, args, device, step)
    where, keep, held = export(smooth, args.out, args.latent, args.size, device, args.keep, classes)
    # The last snapshot's checkpoint is thousands of steps behind this export. Stash again, so the
    # `.pt` holds the weights the `.onnx` was written from and a blend of two runs blends what was measured.
    stash(carry, step, gen, dis, smooth, opt_g, opt_d, path_mean, taken)
    print(f"wrote {where}: {keep} axes holding {held:.1%} of the variance in w")


def preview(gen, args, device, step=0):
    gen.eval()
    with torch.no_grad():
        z = torch.randn(4, args.latent, device=device)
        hot = F.one_hot(torch.arange(4, device=device) % gen.classes, gen.classes).float() if gen.classes else None
        got = gen(z, hot)
    gen.train()
    tiled = got.permute(0, 2, 3, 1).reshape(-1, args.size, 3).cpu().numpy()
    # Numbered, so a run's trajectory stays readable instead of being overwritten by its own end.
    path = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + f"-{step:06d}.png")
    Image.fromarray(((tiled * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)).save(path)
    print(f"  preview {path}")


if __name__ == "__main__":
    main()
