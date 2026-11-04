"""New pictures drawn straight in DINOv2's token space: a diffusion transformer trained on the archive's
crops, steered by the lab's material space, and drawn back to pictures by RAE's decoder (rae.py).

    python tokendit.py train --crops crops-yellow3 --decoder rae/ViTXL_n08/model.pt --out tokendit/ --hours 3
    python tokendit.py sample --model tokendit/last.pt --decoder rae/ViTXL_n08/model.pt --space texture-ip-sd15.npz \\
        --places 12 --per 4 --guide 3 --grid 32x48 --upscale esrgan/RealESRGAN_x4plus.pth --out tokendit/samples/
    python tokendit.py walk --model tokendit/ema-036390.pt --decoder rae/ViTXL_n08/model.pt --space texture-ip-sd15.npz \\
        --stops 12 --seconds 90 --grid 16x24 --upscale esrgan/RealESRGAN_x4plus.pth --out exports/tokendit/walk.mp4

This is stage two of Zheng, Ma, Tong and Xie, "Diffusion Transformers with Representation Autoencoders"
(2025), at the size of their DiT^DH-S, written out here: no code from their release runs. A 16 x 16
grid of 768-number tokens (a 224 picture, drawn back at 256) is far wider per token than a VAE's
latent, so the model is a narrow transformer (12 blocks at 384) that reads the noisy grid and the
condition, under a wide, shallow head (2 blocks at 2048) that reads the noisy grid again, steered
token by token by what the narrow one made of it. Flow matching with velocity prediction; noise
levels drawn logit-normal and shifted toward noise by sqrt(16 * 16 * 768 / 4096), as the paper does
for a latent this size; 50 Euler steps to sample.

It is trained on 16 x 16 grids only. Any other grid is drawn in overlapping 16 x 16 windows whose
velocities are cross-faded at every step, so each window sees the positions it was trained on and
neighbours agree where they overlap: a 32 x 48 grid (512 x 768, 2048 x 3072 enlarged) comes out one
seamless texture with larger forms than a window holds. Stretching the positions over the bigger
grid instead holds at 16 x 24 and turns to brown mud at 32 x 48.

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

    def corner(self, rows, cols):
        """Which of the trained grid's positions a smaller grid takes: its top-left corner."""
        return (torch.arange(rows)[:, None] * self.side + torch.arange(cols)).flatten().to(self.pos0.device)

    def timestep(self, sigma):
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=sigma.device, dtype=torch.float32) / half)
        a = sigma.float()[:, None] * 1000 * freqs
        return self.t_embed(torch.cat([a.cos(), a.sin()], 1))

    def forward(self, x, sigma, place=None, keep=None, grid=None):
        """x: [b, 256, 768] noisy normalised tokens; sigma: [b] noise level; place: [b, 1024] unit places or
        None (no material); keep: [b] booleans, False where the place is dropped; grid: (rows, cols) when
        x is a grid smaller than the trained one."""
        pos0, pos1, cos0, sin0, cos1, sin1 = self.pos0, self.pos1, self.cos0, self.sin0, self.cos1, self.sin1
        if grid is not None and tuple(grid) != (self.side, self.side):
            i = self.corner(*grid)
            pos0, pos1, cos0, sin0, cos1, sin1 = pos0[:, i], pos1[:, i], cos0[i], sin0[i], cos1[i], sin1[i]
        t = self.timestep(sigma)
        if place is None:
            y = self.null.expand(len(x), -1)
        else:
            y = self.y_embed(place)
            if keep is not None:
                y = torch.where(keep[:, None], y, self.null.expand_as(y))
        c = t + y
        s = self.s_embed(x) + pos0
        for block in self.body:
            s = block(s, c, cos0, sin0)
        s = self.lift(s)
        h = self.x_embed(x) + pos1
        for block in self.head:
            h = block(h, s, cos1, sin1)
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


def windows(n, side, overlap=8):
    """Where the windows start along one side of a grid, as rae.py's decoder lays its own: one if the
    side fits the trained grid, else as few as cover it with at least `overlap` tokens shared."""
    if n <= side:
        return [0]
    k = math.ceil((n - side) / (side - overlap)) + 1
    return [round(i * (n - side) / (k - 1)) for i in range(k)]


def guided(model, x, sigma, places, guide, grid=None):
    """The velocity for grids no bigger than the trained one, pushed away from "no material" by `guide`."""
    b = len(x)
    sigma = sigma.expand(b)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        if places is None or guide == 1:
            return model(x, sigma, places, grid=grid).float()
        on = torch.ones(b, dtype=torch.bool, device=x.device)
        both = model(torch.cat([x, x]), torch.cat([sigma, sigma]), torch.cat([places, places]), torch.cat([on, ~on]), grid=grid)
    cond, free = both.float().chunk(2)
    return free + guide * (cond - free)


def velocity(model, x, sigma, places, guide, rows, cols, chunk=128):
    """The guided velocity over a grid of any size: the trained grid's windows over it, each weighed most
    in its middle and fading over the overlap, so neighbours cross-fade (a grid that fits is one window)."""
    side, b = model.side, len(x)
    if (rows, cols) == (side, side):
        return guided(model, x, sigma, places, guide)
    hh, ww = min(rows, side), min(cols, side)
    at = [(y, c) for y in windows(rows, side) for c in windows(cols, side)]
    g = x.view(b, rows, cols, -1)
    pieces = torch.stack([g[:, y:y + hh, c:c + ww] for y, c in at], 1).reshape(b * len(at), hh * ww, -1)
    near = None if places is None else places.repeat_interleave(len(at), 0)
    v = torch.cat([guided(model, pieces[j:j + chunk], sigma, None if near is None else near[j:j + chunk], guide, (hh, ww))
                   for j in range(0, len(pieces), chunk)]).view(b, len(at), hh, ww, -1)
    ramp = lambda n: torch.minimum(torch.arange(n) + 1.0, n - torch.arange(n).float()).clamp(max=8).to(x.device)  # noqa: E731
    weight = (ramp(hh)[:, None] * ramp(ww)[None])[..., None]
    out, total = torch.zeros_like(g), torch.zeros(rows, cols, 1, device=x.device)
    for k, (y, c) in enumerate(at):
        out[:, y:y + hh, c:c + ww] += v[:, k] * weight
        total[y:y + hh, c:c + ww] += weight
    return (out / total).reshape(b, rows * cols, -1)


@torch.no_grad()
def sample(model, places, steps=50, guide=3.0, generator=None, mean=None, std=None, count=1, grid=None, noise=None):
    """Token grids for these places ([b, 1024] unit vectors), or `count` of them with no material: Euler
    from noise, guided away from "no material" by `guide`. `grid` (rows, cols) is any size, drawn in
    windows of the trained one; `noise` holds the start fixed, as a walk wants."""
    p = next(model.parameters())
    rows, cols = grid or (model.side, model.side)
    b = len(places) if places is not None else count
    if noise is None:
        noise = torch.randn(b, rows * cols, model.channels, generator=generator, device=p.device)
    x = noise.to(p.device, torch.float32).expand(b, -1, -1).clone()
    levels = shifted(torch.linspace(1, 0, steps + 1, device=p.device))
    for i in range(steps):
        x = x + (levels[i + 1] - levels[i]) * velocity(model, x, levels[i], places, guide, rows, cols)
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


def walk(args, model, rae, mean, std):
    """A film along a closed loop through the most typical photographs of as many regions of the space,
    the noise held: one drawing whose material changes, paced so it moves as much from each frame to the
    next. Written to args.out, with its record beside it."""
    import json
    import subprocess

    import flights
    import rae as R

    places, files = flights.photographs(dict(np.load(args.space)), args.stops, np.random.default_rng(args.seed))
    places /= np.linalg.norm(places, axis=1, keepdims=True)
    order = flights.loop_order(places)
    places, files = places[order], [files[i] for i in order]
    stops = [torch.from_numpy(x.astype(np.float32)).cuda() for x in places]
    rows, cols = args.grid
    noise = torch.randn(1, rows * cols, model.channels, generator=torch.Generator(device="cuda").manual_seed(args.seed), device="cuda")

    def route(us):
        v = torch.stack([R.along(stops, u) for u in us])
        return v / v.norm(dim=1, keepdim=True)

    def drawn(us):
        z = sample(model, route(us), args.steps, args.guide, mean=mean, std=std, grid=(rows, cols), noise=noise)
        return rae.decode(R.renorm(z), (rows, cols))

    def looks(us):
        """Small copies of the route at these moments, drawn in batches."""
        out = []
        for j in range(0, len(us), args.batch):
            out += [image.reshape(rows * 4, 4, cols * 4, 4, 3).mean((1, 3)) for image in drawn(us[j:j + args.batch])]
        return out

    frames = round(args.seconds * args.fps)
    print(f"{len(files)} photographs, grid {rows} x {cols}", flush=True)
    moments, lurches = flights.steady(lambda u: looks([u])[0], frames, dense=2, rounds=2, many=looks)
    print(f"timed: {lurches} jumps still larger than a frame's move", flush=True)
    net = None
    if args.upscale:
        import upscale

        net = upscale.load(args.upscale)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, moved, last, start = None, [], None, None
    for j in range(0, frames, args.batch):
        for image in drawn(list(moments[j:j + args.batch])):
            shown = np.clip(upscale.enlarge(net, image) if net is not None else image, 0, 1)
            shown = Image.fromarray((shown * 255).round().astype(np.uint8))
            if writer is None:
                writer = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % shown.size,
                     "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
                start, made = image, list(shown.size)
            writer.stdin.write(shown.tobytes())
            if last is not None:
                moved.append(float(np.abs(image - last).mean()))
            last = image
        print(f"frame {min(j + args.batch, frames)} / {frames}", flush=True)
    moved.append(float(np.abs(start - last).mean()))  # the seam, back to the start
    writer.stdin.close()
    writer.wait()
    moved = np.array(moved)
    passes = [float(np.interp(k / len(files), moments, np.arange(frames))) / args.fps for k in range(len(files))]
    out.with_suffix(".json").write_text(json.dumps(dict(
        files=files, seconds=args.seconds, fps=args.fps, frames=frames, size=made, grid=[rows, cols], seed=args.seed,
        guide=args.guide, steps=args.steps, passes=np.round(passes, 2).tolist(), moved=np.round(moved, 5).tolist(),
        evenness=float(moved.std() / moved.mean()), lurches=lurches, moments=np.round(moments, 7).tolist())), encoding="utf-8")
    print(f"{out}: {frames} frames, evenness {moved.std() / moved.mean():.3f}", flush=True)


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
    w = sub.add_parser("walk")
    w.add_argument("--model", required=True)
    w.add_argument("--decoder", required=True)
    w.add_argument("--space", required=True, help="The lab's space (.npz): the walk goes through its photographs' places.")
    w.add_argument("--stops", type=int, default=12, help="How many photographs, one per region of the space.")
    w.add_argument("--seed", type=int, default=0, help="Other photographs, and other noise held through the walk.")
    w.add_argument("--seconds", type=float, default=60.0)
    w.add_argument("--fps", type=int, default=24)
    w.add_argument("--guide", type=float, default=3.0)
    w.add_argument("--steps", type=int, default=30)
    w.add_argument("--batch", type=int, default=16, help="Frames drawn at once.")
    w.add_argument("--upscale", help="Real-ESRGAN x4 weights, to enlarge every frame.")
    w.add_argument("--crf", type=int, default=18)
    w.add_argument("--out", required=True, help="An .mp4.")
    for p in (s, w):
        p.add_argument("--grid", default="16x16", type=lambda v: tuple(int(x) for x in v.split("x")),
                       help="Rows x columns of tokens, 16 pixels each: 16x16 is what it was trained on, 32x48 a "
                            "3:2 at 512 x 768, drawn in its windows.")
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)

    import rae as R

    state = torch.load(args.model, map_location="cuda", weights_only=True)
    model = TokenDiT().cuda().eval()
    model.load_state_dict(state["ema"])
    rae = R.RAE(args.decoder)
    if args.cmd == "walk":
        return walk(args, model, rae, state["mean"].cuda(), state["std"].cuda())
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
        z = sample(model, p, args.steps, args.guide, gen, state["mean"].cuda(), state["std"].cuda(), args.per, args.grid)
        for j, image in enumerate(rae.decode(R.renorm(z), args.grid)):
            if net is not None:
                image = upscale.enlarge(net, image)
            Image.fromarray((image * 255).round().astype(np.uint8)).save(out / f"{k:02d}-{j}.jpg", quality=92)
        print(k, names[k], flush=True)


if __name__ == "__main__":
    main()
