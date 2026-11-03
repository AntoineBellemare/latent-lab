"""New pictures drawn straight in DINOv2's token space: a diffusion transformer trained on the archive's
crops, steered by the lab's material space, and drawn back to pictures by RAE's decoder (rae.py).

    python tokendit.py train --crops crops-yellow3 --decoder rae/ViTXL_n08/model.pt --out tokendit/ --hours 3
    python tokendit.py sample --model tokendit/last.pt --decoder rae/ViTXL_n08/model.pt --space texture-ip-sd15.npz \\
        --places 12 --per 4 --guide 3 --upscale esrgan/RealESRGAN_x4plus.pth --out tokendit/samples/

This is stage two of Zheng, Ma, Tong and Xie, "Diffusion Transformers with Representation Autoencoders"
(2025), at the size of their DiT^DH-S, written out here: no code from their release runs. A 16 x 16
grid of 768-number tokens (a 224 picture, drawn back at 256) is far wider per token than a VAE's
latent, so the model is a narrow transformer (12 blocks at 384) that reads the noisy grid and the
condition, under a wide, shallow head (2 blocks at 2048) that reads the noisy grid again, steered
token by token by what the narrow one made of it. Flow matching with velocity prediction; noise
levels drawn logit-normal and shifted toward noise by sqrt(16 * 16 * 768 / 4096), as the paper does
for a latent this size; 50 Euler steps to sample.

The condition is a place in the lab's material space (CLIP ViT-H, as `fastgan.py --look` takes it):
each crop's own, or half the time its photograph's mean, so a preset, a photograph or any mix of
them steers it. One in ten is dropped for a learned "no material", which guidance pushes away from.
"""

import argparse
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

SHIFT = math.sqrt(16 * 16 * 768 / 4096)  # noise shift for this latent's size, as the paper sets it


def rope(side, dim, theta=10000.0):
    """2D rotary angles for a side x side grid: half of each head's pairs turn with the row, half with the column."""
    q = dim // 4
    freqs = 1.0 / theta ** (torch.arange(q, dtype=torch.float32) / q)
    y, x = torch.meshgrid(torch.arange(side, dtype=torch.float32), torch.arange(side, dtype=torch.float32), indexing="ij")
    ang = torch.cat([y.reshape(-1, 1) * freqs, x.reshape(-1, 1) * freqs], 1)  # [N, dim/2]
    return ang.cos(), ang.sin()


def turned(x, cos, sin):
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


def sincos(side, width):
    """Fixed 2D sine-cosine positions, [1, side*side, width]."""
    q = width // 4
    freqs = 1.0 / 10000 ** (torch.arange(q, dtype=torch.float32) / q)
    y, x = torch.meshgrid(torch.arange(side, dtype=torch.float32), torch.arange(side, dtype=torch.float32), indexing="ij")
    ay, ax = y.reshape(-1, 1) * freqs, x.reshape(-1, 1) * freqs
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1)[None]


class Block(nn.Module):
    """Pre-norm attention and SwiGLU MLP, each modulated (shift, scale) and gated by the condition, which
    is one vector per picture or one per token."""

    def __init__(self, width, heads, cond):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(width, eps=1e-6, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.RMSNorm(width, eps=1e-6, elementwise_affine=False)
        inner = int(2 / 3 * 4 * width)
        self.w12 = nn.Linear(width, 2 * inner)
        self.w3 = nn.Linear(inner, width)
        self.ada = nn.Linear(cond, 6 * width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, cos, sin):
        b, n, w = x.shape
        mod = self.ada(F.silu(c))
        if mod.dim() == 2:
            mod = mod[:, None]
        sh1, sc1, g1, sh2, sc2, g2 = mod.chunk(6, -1)
        h = self.norm1(x) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        q, k = turned(q, cos, sin).to(v.dtype), turned(k, cos, sin).to(v.dtype)
        x = x + g1 * self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w))
        a, g = self.w12(self.norm2(x) * (1 + sc2) + sh2).chunk(2, -1)
        return x + g2 * self.w3(F.silu(a) * g)


class TokenDiT(nn.Module):
    """DiT^DH: a narrow body on the noisy grid and the condition, a wide shallow head on the noisy grid
    again, modulated token by token by the body's output. Predicts the velocity, noise minus tokens."""

    def __init__(self, side=16, channels=768, widths=(384, 2048), depths=(12, 2), heads=(6, 16), cond=1024):
        super().__init__()
        w0, w1 = widths
        self.side, self.channels = side, channels
        self.s_embed = nn.Linear(channels, w0)
        self.x_embed = nn.Linear(channels, w1)
        self.register_buffer("pos0", sincos(side, w0), persistent=False)
        self.register_buffer("pos1", sincos(side, w1), persistent=False)
        for k, (w, h) in enumerate(zip(widths, heads)):
            cos, sin = rope(side, w // h)
            self.register_buffer(f"cos{k}", cos, persistent=False)
            self.register_buffer(f"sin{k}", sin, persistent=False)
        self.t_embed = nn.Sequential(nn.Linear(256, w0), nn.SiLU(), nn.Linear(w0, w0))
        self.y_embed = nn.Sequential(nn.Linear(cond, w0), nn.SiLU(), nn.Linear(w0, w0))
        self.null = nn.Parameter(torch.zeros(w0))  # "no material", for guidance
        self.body = nn.ModuleList(Block(w0, heads[0], w0) for _ in range(depths[0]))
        self.lift = nn.Linear(w0, w1)
        self.head = nn.ModuleList(Block(w1, heads[1], w1) for _ in range(depths[1]))
        self.final_norm = nn.RMSNorm(w1, eps=1e-6, elementwise_affine=False)
        self.final_ada = nn.Linear(w1, 2 * w1)
        self.out = nn.Linear(w1, channels)
        for layer in (self.final_ada, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def timestep(self, sigma):
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=sigma.device, dtype=torch.float32) / half)
        a = sigma.float()[:, None] * 1000 * freqs
        return self.t_embed(torch.cat([a.cos(), a.sin()], 1))

    def forward(self, x, sigma, place=None, keep=None):
        """x: [b, 256, 768] noisy normalised tokens; sigma: [b] noise level; place: [b, 1024] unit places or
        None (no material); keep: [b] booleans, False where the place is dropped."""
        t = self.timestep(sigma)
        if place is None:
            y = self.null.expand(len(x), -1)
        else:
            y = self.y_embed(place)
            if keep is not None:
                y = torch.where(keep[:, None], y, self.null.expand_as(y))
        c = t + y
        s = self.s_embed(x) + self.pos0
        for block in self.body:
            s = block(s, c, self.cos0, self.sin0)
        s = self.lift(s)
        h = self.x_embed(x) + self.pos1
        for block in self.head:
            h = block(h, s, self.cos1, self.sin1)
        shift, scale = self.final_ada(F.silu(s)).chunk(2, -1)
        return self.out(self.final_norm(h) * (1 + scale) + shift)


def shifted(u):
    """A noise level on the latent's own scale: u in [0, 1] pushed toward noise by SHIFT."""
    return SHIFT * u / (1 + (SHIFT - 1) * u)


class Crops(torch.utils.data.Dataset):
    """Each crop as a 224 picture (a random 512 piece of the 576, flipped either way), with its index (the
    places stay in the main process: copied into every worker they would be half a gigabyte each)."""

    def __init__(self, folder, names):
        self.folder, self.names = folder, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(self.folder / self.names[i]) as im:
            im = im.convert("RGB")
        rng = np.random.default_rng()
        x, y = rng.integers(0, im.width - 512 + 1), rng.integers(0, im.height - 512 + 1)
        im = im.crop((x, y, x + 512, y + 512)).resize((224, 224), Image.BICUBIC)
        a = np.asarray(im)
        if rng.random() < 0.5:
            a = a[:, ::-1]
        if rng.random() < 0.5:
            a = a[::-1]
        return torch.from_numpy(np.array(a)), i  # a copy: a PIL array is read-only


@torch.no_grad()
def encode(rae, pixels):
    """[b, 224, 224, 3] uint8 -> [b, 256, 768] tokens."""
    import rae as R

    x = pixels.to(rae.device).permute(0, 3, 1, 2).float() / 255
    x = ((x - R.MEAN.to(x.device)) / R.STD.to(x.device)).to(rae.dtype)
    return rae.encoder(pixel_values=x).last_hidden_state[:, 5:].float()


@torch.no_grad()
def sample(model, places, steps=50, guide=3.0, generator=None, mean=None, std=None, count=1):
    """Token grids for these places ([b, 1024] unit vectors), or `count` of them with no material: Euler
    from noise, guided away from "no material" by `guide`."""
    p = next(model.parameters())
    b = len(places) if places is not None else count
    x = torch.randn(b, model.side ** 2, model.channels, generator=generator, device=p.device)
    levels = shifted(torch.linspace(1, 0, steps + 1, device=p.device))
    for i in range(steps):
        sigma = levels[i].expand(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=p.is_cuda):
            if places is None or guide == 1:
                v = model(x, sigma, places)
            else:
                both = model(torch.cat([x, x]), torch.cat([sigma, sigma]), torch.cat([places, places]),
                             torch.cat([torch.ones(b, dtype=torch.bool, device=p.device), torch.zeros(b, dtype=torch.bool, device=p.device)]))
                cond, free = both.float().chunk(2)
                v = free + guide * (cond - free)
        x = x + (levels[i + 1] - levels[i]) * v.float()
    return x * std + mean if mean is not None else x


def sheet(images, cols):
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    out = np.zeros((rows * h, cols * w, 3), np.float32)
    for k, im in enumerate(images):
        out[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = im
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))


def train(args):
    import rae as R

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crops = pathlib.Path(args.crops)
    index = np.load(crops / "crops.npz")
    emb = index["emb"].astype(np.float32)
    owner = index["owner"]
    own = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    means = np.zeros((owner.max() + 1, emb.shape[1]), np.float32)
    np.add.at(means, owner, own)
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    places = np.stack([own, means[owner]], 1)  # each crop's place, and its photograph's
    data = Crops(crops / "crops", list(index["crops"]))
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=True, pin_memory=True)
    rae = R.RAE(args.decoder)
    if rae.side != 16:
        raise SystemExit("train with the 256 decoder (16 x 16 tokens)")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = TokenDiT().cuda()
    ema = TokenDiT().cuda().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step, mean, std = 0, None, None
    last = out / "last.pt"
    if last.is_file():
        state = torch.load(last, map_location="cuda", weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step, mean, std = state["step"], state["mean"], state["std"]
        print(f"resumed at step {step}", flush=True)
    else:
        ema.load_state_dict(model.state_dict())
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, {len(data)} crops", flush=True)

    if mean is None:
        # Each of the 768 channels to zero mean and unit spread, over a few thousand crops' tokens.
        seen = []
        for pixels, _ in loader:
            seen.append(encode(rae, pixels).reshape(-1, 768))
            if len(seen) * args.batch >= 4096:
                break
        seen = torch.cat(seen)
        mean, std = seen.mean(0), seen.std(0) + 1e-4
        print(f"token stats from {len(seen) // 256} crops: spread {std.mean():.3f}", flush=True)
    mean, std = mean.cuda(), std.cuda()

    gen = torch.Generator(device="cuda").manual_seed(0)
    probe = torch.from_numpy(means[np.linspace(0, len(means) - 1, 8).astype(int)]).cuda()
    started, deadline = time.time(), time.time() + args.hours * 3600
    shown, losses = time.time(), []
    while time.time() < deadline:
        for pixels, idx in loader:
            z = (encode(rae, pixels) - mean) / std
            b = len(z)
            pick = np.random.random(b) < 0.5
            place = torch.from_numpy(places[idx.numpy(), np.where(pick, 0, 1)]).cuda(non_blocking=True)
            keep = torch.rand(b, device="cuda") > 0.1
            u = torch.sigmoid(torch.randn(b, device="cuda"))  # logit-normal(0, 1)
            sigma = shifted(u)
            noise = torch.randn_like(z)
            xt = (1 - sigma[:, None, None]) * z + sigma[:, None, None] * noise
            # Warm up over 2,000 steps; hold; then down to a tenth over the last third of the time.
            frac = (time.time() - started) / (deadline - started)
            lr = args.lr * min(1.0, (step + 1) / 2000) * (1.0 if frac < 2 / 3 else 1.0 - 0.9 * (frac - 2 / 3) * 3)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(xt, sigma, place, keep)
            loss = F.mse_loss(v.float(), noise - z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                d = min(0.9995, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
            step += 1
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"step {step}  loss {np.mean(losses):.4f}  lr {lr:.2e}  {(time.time() - started) / 60:.0f} min", flush=True)
                losses = []
            if time.time() - shown > args.every * 60 or time.time() >= deadline:
                shown = time.time()
                state = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                             mean=mean.cpu(), std=std.cpu(), shift=SHIFT)
                torch.save(state, last)
                torch.save(dict(ema=ema.state_dict(), mean=mean.cpu(), std=std.cpu(), step=step, shift=SHIFT),
                           out / f"ema-{step:06d}.pt")
                z = sample(ema, probe, steps=50, guide=3.0, generator=gen.manual_seed(0), mean=mean, std=std)
                sheet(list(rae.decode(z)), 4).save(out / f"probe-{step:06d}.jpg", quality=88)
                print(f"saved step {step}", flush=True)
            if time.time() >= deadline:
                break
    print("done", step, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", required=True, help="embed_crops.py's folder: crops/ and crops.npz.")
    t.add_argument("--decoder", required=True, help="The 256 RAE decoder (16 x 16 tokens), for the probe sheets.")
    t.add_argument("--out", required=True)
    t.add_argument("--hours", type=float, default=3.0)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--every", type=float, default=20.0, help="Minutes between checkpoints and probe sheets.")
    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--decoder", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--space", help="The lab's space (.npz): steer by its photographs' places.")
    s.add_argument("--places", type=int, default=8, help="How many photographs from the space, one per region.")
    s.add_argument("--per", type=int, default=4, help="Pictures per place.")
    s.add_argument("--guide", type=float, default=3.0)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--upscale", help="Real-ESRGAN x4 weights, to enlarge every picture.")
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)

    import rae as R

    state = torch.load(args.model, map_location="cuda", weights_only=True)
    model = TokenDiT().cuda().eval()
    model.load_state_dict(state["ema"])
    rae = R.RAE(args.decoder)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    places, names = None, ["no material"]
    if args.space:
        import flights

        places, names = flights.photographs(dict(np.load(args.space)), args.places, np.random.default_rng(args.seed))
        places = torch.from_numpy(places.astype(np.float32)).cuda()
    net = None
    if args.upscale:
        import upscale

        net = upscale.load(args.upscale)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for k in range(len(names)):
        p = None if places is None else places[k:k + 1].expand(args.per, -1)
        z = sample(model, p, args.steps, args.guide, gen, state["mean"].cuda(), state["std"].cuda(), args.per)
        for j, image in enumerate(rae.decode(z)):
            if net is not None:
                image = upscale.enlarge(net, image)
            Image.fromarray((image * 255).round().astype(np.uint8)).save(out / f"{k:02d}-{j}.jpg", quality=92)
        print(k, names[k], flush=True)


if __name__ == "__main__":
    main()
