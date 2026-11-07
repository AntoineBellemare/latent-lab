"""breed.py — new pictures bred from your best-rated ones and selected by taste.py's critic.

    python breed.py tokens --critic dataset/taste/critic.npz --ratings dataset/taste/ratings.csv --out dataset/breed/tokens
    python breed.py tokens --critic dataset/taste/critic-clip.npz --ratings dataset/taste/ratings.csv --label Yellow \
        --ways blend caricature --apart --mutate 0 --out dataset/breed/tokens-fusion
    python breed.py nca    --critic dataset/taste/critic.npz --ratings dataset/taste/ratings.csv --out dataset/breed/nca
    python breed.py ip     --critic dataset/taste/critic.npz --ratings dataset/taste/ratings.csv --out dataset/breed/ip

Three routes, one selection. Each keeps a population, breeds children from pairs chosen by tournament,
mutates some, and keeps the best by the critic's predicted stars minus the spread of its bootstrap
copies. The spread is subtracted because a genetic algorithm climbs whatever it is given: where the
copies disagree the critic is guessing, and a child scoring high there has found a gap in the critic
rather than a picture.

  tokens  DINOv2 token grids (rae.py), crossed with the lab's own hybrid ways (blend, patchwork,
          layout) and mutated with remix. Scored on the grid itself, no drawing needed, because the
          critic reads exactly these tokens. The survivors are drawn by the RAE decoder, encoded
          again and scored again, which says whether the critic's praise survives the drawing.
  nca     style_ca.py's rule as the genome. One parent is trained on the best-rated photographs and
          its children stay in its basin, so a weighted average of two is a blend of two textures.
          Fitness is read off frames at three points of a long run, plus how much the late frames
          still differ: the texture is asked to keep developing.
  ip      Points in the IP-Adapter's CLIP space (ip_space.py's sd15 preset), crossed by slerp and
          drawn by SD 1.5 with the LCM LoRA in four steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import torch

import taste

RAE_DECODER = ("nyu-visionx/RAE-collections", "decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt")


# ─────────────────────────── shared ───────────────────────────

class Critic:
    """taste.py's ridge as plain numbers: stars = features · w + b, and five bootstrap copies."""

    def __init__(self, path, caution=1.0):
        d = np.load(path)
        self.w, self.b = d["w"].astype(np.float32), float(d["b"])
        self.bw, self.bb = d["boot_w"].astype(np.float32), d["boot_b"].astype(np.float32)
        self.caution = caution
        # What the weights read: 'dino' (taste.features of a token grid) or 'clip' (taste.ClipEye of a picture).
        self.kind = str(d["kind"]) if "kind" in d.files else "dino"

    def __call__(self, feats):
        feats = np.asarray(feats, np.float32)
        stars = feats @ self.w + self.b
        spread = (feats @ self.bw.T + self.bb).std(1)
        return stars, spread, stars - self.caution * spread


LABEL = None    # set from --label: the colour tag a population starts from (Yellow: the textures)


def best_rated(ratings, least=4, limit=None, seed=0):
    with open(ratings, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["stars"]) >= least
                and (LABEL is None or r.get("label", "").lower() == LABEL.lower())]
    if not rows:
        raise SystemExit(f"no photographs with {least}+ stars{' tagged ' + LABEL if LABEL else ''} in {ratings}")
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return [r["path"] for r in rows[:limit]] if limit else [r["path"] for r in rows]


def tournament(fitness, rng, k=3):
    pick = rng.choice(len(fitness), k, replace=False)
    return int(pick[np.argmax(fitness[pick])])


def distinct_elites(fitness, look, k, alike=0.97):
    """The fittest `k`, skipping any that look like one already kept (token-mean cosine). Without
    this the first tokens run (2026-11-05) converged by generation 13 on twelve copies of one grey
    grid: the fittest child bred with itself until nothing else was left."""
    look = look / np.linalg.norm(look, axis=1, keepdims=True)
    kept = []
    for i in np.argsort(-fitness):
        if all(look[i] @ look[j] < alike for j in kept):
            kept.append(int(i))
        if len(kept) == k:
            break
    return np.array(kept)


def square(im, side):
    """The middle square of a picture, `side` pixels across."""
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((side, side), taste.Image.LANCZOS)


def log_line(gen, stars, spread, fit):
    i = int(np.argmax(fit))
    return dict(gen=gen, best_stars=float(stars[i]), best_spread=float(spread[i]), best_fitness=float(fit[i]),
                mean_stars=float(stars.mean()), mean_spread=float(spread.mean()))


def photo_baseline(critic, data_path):
    """What the critic says about the real photographs, for a scale: a bred child's 3.6 means
    something only next to where the archive's own pictures fall."""
    if critic.kind == "clip":
        x = np.load(pathlib.Path(data_path).with_name("clip_vith.npy"))
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    else:
        x = np.load(data_path)["features"]
    stars, spread, _ = critic(x)
    return dict(photos_mean=float(stars.mean()), photos_p90=float(np.percentile(stars, 90)),
                photos_max=float(stars.max()), photos_spread_mean=float(spread.mean()))


def write_sheet(images, labels, out, title, cols=6, tile=220):
    """Pictures already in memory, as one labelled sheet."""
    from PIL import Image, ImageDraw

    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, 28 + rows * tile), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    font = taste.ImageFont.load_default(size=14)
    draw.text((8, 6), title, fill=(230, 230, 230), font=font)
    for i, (im, lab) in enumerate(zip(images, labels)):
        if not isinstance(im, Image.Image):
            im = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        im = im.copy()
        im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, 28 + (i // cols) * tile
        canvas.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.rectangle([x, y, x + 120, y + 20], fill=(0, 0, 0))
        draw.text((x + 4, y + 3), lab, fill=(255, 255, 255), font=font)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


# ─────────────────────────── 2. tokens ───────────────────────────

def run_tokens(args, critic):
    from huggingface_hub import hf_hub_download

    import masks
    import rae as R
    from app.registry import lineup, rebuilt, scattered

    out = pathlib.Path(args.out)
    clip_eye = taste.ClipEye() if critic.kind == "clip" else None  # before the RAE: see ClipEye
    model = R.RAE(hf_hub_download(*RAE_DECODER))
    rows = cols = model.side                                    # 32 x 32: the decoder's own grid
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def read(drawn):
        """Drawn pictures as the critic's features."""
        if clip_eye is not None:
            return clip_eye(drawn)
        return taste.features(model.encode(drawn, (rows, cols)))

    def score(z):
        """By default the drawing is scored, not the grid. The first run scored grids and the critic
        was gamed: twelve grids at 4.05, above every real photograph, read 3.2 once drawn, because
        token statistics no picture has are still numbers a linear critic will reward."""
        z = model.shell(z)
        if args.score == "grid" and clip_eye is None:
            feats = taste.features(z)
        else:
            feats = np.concatenate([read(model.decode(z[i:i + 12], (rows, cols))) for i in range(0, len(z), 12)])
        stars, spread, fit = critic(feats)
        return stars, spread, fit, (feats if clip_eye is not None else feats[:, :768])

    paths = best_rated(args.ratings, least=4, limit=args.pop, seed=args.seed)
    pics = [square(taste.open_picture(p, 768), 512) for p in paths]
    pop = model.encode(np.stack([np.asarray(p, np.float32) / 255 for p in pics]), (rows, cols)).float()
    stars, spread, fit, look = score(pop)
    history = [log_line(0, stars, spread, fit)]
    print(f"gen  0  best {stars[np.argmax(fit)]:.2f}  mean {stars.mean():.2f}", flush=True)
    start_best = np.argsort(-fit)[:12]
    write_sheet([pics[i] for i in start_best], [f"{stars[i]:.2f} +-{spread[i]:.2f}" for i in start_best],
                out / "parents.png", "the starting population: your 4-star photographs, centre squares")

    # `blend` averages tokens, and averaging again and again walks every grid to the archive's mean
    # texture (the first run's grey mush). It stays available, off by default.
    # `caricature` is blend's opposite: one parent pushed away from the starting population's mean
    # token, so what makes it itself grows. Blend contracts toward that mean and caricature expands from
    # it; run together they keep fusing textures without the population sliding to grey.
    centre = pop.mean((0, 1))
    ways = tuple(args.ways)
    bests = []
    for g in range(1, args.gens + 1):
        elite = distinct_elites(fit, look, args.elites)
        unit = look / np.linalg.norm(look, axis=1, keepdims=True)
        children, made = [], []
        while len(children) < args.pop - args.elites:
            ia = tournament(fit, rng)
            if args.apart:
                # The partner comes from the half of the population least like the first parent.
                far = np.flatnonzero(unit @ unit[ia] <= np.median(unit @ unit[ia]))
                ib = int(far[tournament(fit[far], rng)])
            else:
                ib = tournament(fit, rng)
            a, b = pop[ia], pop[ib]
            way = ways[rng.integers(len(ways))]
            if way == "blend":
                t = rng.uniform(0.3, 0.7)
                z = (1 - t) * a + t * b
            elif way == "caricature":
                z = centre + rng.uniform(1.3, 1.7) * (a - centre)
            elif way == "patchwork":
                f = masks.rank(masks.growth(rows, cols, rng.uniform(1.0, 3.0), int(rng.integers(1 << 30))))
                w = np.stack([masks.blur(x, 0.7) for x in lineup(f, 2, 0.5, 0.35)])
                wa = torch.from_numpy(w[0].reshape(-1, 1)).to(a.device)
                z = a * wa + b * (1 - wa)
            else:
                z = rebuilt(a.reshape(rows, cols, -1), b.reshape(rows, cols, -1), 2, 1, False, gen).reshape(rows * cols, -1)
            if rng.random() < args.mutate:
                moved = scattered(z.reshape(rows, cols, -1), 2, 3, gen).reshape(rows * cols, -1)
                z = 0.5 * z + 0.5 * moved
            children.append(model.shell(z[None])[0].float())
            made.append(way)
        pop = torch.cat([pop[elite], torch.stack(children)])
        stars, spread, fit, look = score(pop)
        history.append(log_line(g, stars, spread, fit) | {"ways": {w: made.count(w) for w in ways}})
        bests.append(pop[int(np.argmax(fit))].clone())
        print(f"gen {g:2d}  best {stars[np.argmax(fit)]:.2f} (+-{spread[np.argmax(fit)]:.2f})  mean {stars.mean():.2f}", flush=True)

    # Draw the survivors, then read the drawings back: the critic praised tokens, and this is whether
    # the praise survives the decoder.
    top = distinct_elites(fit, look, 12)
    drawn = model.decode(pop[top], (rows, cols))
    s2, sp2, _ = critic(read(drawn))
    write_sheet(list(drawn), [f"{stars[i]:.2f} | drawn {s2[k]:.2f}" for k, i in enumerate(top)],
                out / "final.png", f"after {args.gens} generations: grid score | score of the drawing, re-read")
    lineage = model.decode(torch.stack(bests[:: max(1, len(bests) // 12)][:12]), (rows, cols))
    write_sheet(list(lineage), [f"gen {k * max(1, len(bests) // 12) + 1}" for k in range(len(lineage))],
                out / "lineage.png", "the best of every few generations")
    return dict(history=history, final_grid_stars=[float(stars[i]) for i in top],
                final_drawn_stars=[float(x) for x in s2])


# ─────────────────────────── 1. nca ───────────────────────────

class Rules:
    """A population of style_ca rules stepped together: the same rule as style_ca.Rule, with one set
    of weights per member, so the whole population runs as one batch on the GPU."""

    def __init__(self, genomes, hidden, device):
        import style_ca as S

        self.S, self.hidden, self.device = S, hidden, device
        self.perceiver = S.Rule(hidden).to(device)
        self.set(genomes)

    def shapes(self):
        S, h = self.S, self.hidden
        return [(h, S.PERC), (h,), (S.CH, h), (S.CH,)]

    def set(self, genomes):
        g = torch.as_tensor(genomes, dtype=torch.float32, device=self.device)
        parts, i = [], 0
        for shape in self.shapes():
            n = int(np.prod(shape))
            parts.append(g[:, i:i + n].reshape(len(g), *shape))
            i += n
        self.w1, self.b1, self.w2, self.b2 = parts

    @torch.no_grad()
    def step(self, x):
        P, C, H, W = x.shape
        seen = self.perceiver.perceive(x).permute(0, 2, 3, 1).reshape(P, H * W, -1)
        h = torch.relu(torch.bmm(seen, self.w1.transpose(1, 2)) + self.b1[:, None])
        d = torch.tanh(torch.bmm(h, self.w2.transpose(1, 2)) + self.b2[:, None])
        d = d.reshape(P, H, W, C).permute(0, 3, 1, 2)
        fires = (torch.rand_like(x[:, :1]) < self.S.FIRE).float()
        return torch.clamp(x + fires * d, -self.S.BOUND, self.S.BOUND)


def genome_of(rule):
    return torch.cat([rule.layer1.weight.flatten(), rule.layer1.bias, rule.layer2.weight.flatten(),
                      rule.layer2.bias]).detach().cpu().numpy()


def rule_of(genome, hidden):
    import style_ca as S

    rule = S.Rule(hidden)
    shapes = [(hidden, S.PERC), (hidden,), (S.CH, hidden), (S.CH,)]
    i = 0
    for t, shape in zip((rule.layer1.weight, rule.layer1.bias, rule.layer2.weight, rule.layer2.bias), shapes):
        n = int(np.prod(shape))
        t.data = torch.tensor(genome[i:i + n], dtype=torch.float32).reshape(shape)
        i += n
    return rule


def run_nca(args, critic):
    import torch.nn.functional as F

    import style_ca as S

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # The parent: style_ca's own training loop, on crops of the best-rated photographs. Its loader
    # reads a folder; these photographs live in several and some are RAW, so the crops are made here
    # and handed to the same loop.
    paths = best_rated(args.ratings, least=4, limit=args.limit, seed=args.seed)
    size, detail, crops = 128, 3.0, 2
    side = round(size * detail)
    targets = []
    for p in paths:
        im = taste.open_picture(p, 1600)
        scale = side / min(im.size)
        im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), taste.Image.LANCZOS)
        px = np.asarray(im, np.float32) / 255
        for _ in range(crops):
            t, l = rng.integers(0, px.shape[0] - size + 1), rng.integers(0, px.shape[1] - size + 1)
            targets.append(torch.tensor(px[t:t + size, l:l + size]).permute(2, 0, 1))
    S.images = lambda *a, **k: torch.stack(targets)
    t0 = time.time()
    parent = S.train(SimpleNamespace(images="", size=size, limit=0, detail=detail, crops=crops, hidden=args.hidden,
                                     parent="", steps=args.parent_steps, batch=4, pool=128, min_steps=32, max_steps=64,
                                     lr=2e-3, overflow=1.0, report=500), device)
    print(f"parent trained in {time.time() - t0:.0f} s", flush=True)
    np.save(out / "parent.npy", S.pack(parent))
    base = genome_of(parent)
    scale_of = np.concatenate([np.full(int(np.prod(s)), float(np.std(v)) or 1e-3)
                               for s, v in zip([(args.hidden, S.PERC), (args.hidden,), (S.CH, args.hidden), (S.CH,)],
                                               np.split(base, np.cumsum([args.hidden * S.PERC, args.hidden, S.CH * args.hidden])))])

    eye = taste.Eye()
    marks = args.frames                     # ticks at which a run is looked at

    def run(genomes, keep=None):
        """Every member from an empty grid; the frames at `marks`, and optionally a whole film."""
        rules = Rules(genomes, args.hidden, device)
        x = torch.zeros(len(genomes), S.CH, args.grid, args.grid, device=device)
        frames, film = {}, []
        for tick in range(1, max(marks) + 1):
            x = rules.step(x)
            if tick in marks:
                frames[tick] = S.to_rgb(x).clone()
            if keep is not None and tick % keep == 0:
                film.append(S.to_rgb(x[:1]).clone())
        return frames, film

    def judge(genomes):
        frames, _ = run(genomes)
        feats, alive = [], []
        for tick in marks:
            rgb = frames[tick]
            alive.append(rgb.flatten(1).std(1).cpu().numpy())
            up = F.interpolate(rgb, size=(448, 448), mode="bicubic", align_corners=False).clamp(0, 1)
            f = []
            for k in range(len(up)):
                tok, _ = eye.tokens(taste.Image.fromarray((up[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)))
                f.append(taste.features(tok)[0])
            feats.append(np.stack(f))
        stars = np.mean([critic(f)[0] for f in feats], 0)
        spread = np.mean([critic(f)[1] for f in feats], 0)
        m1, m2 = feats[-2][:, :768], feats[-1][:, :768]
        change = 1 - (m1 * m2).sum(1) / (np.linalg.norm(m1, axis=1) * np.linalg.norm(m2, axis=1))
        dead = np.min(alive, 0) < 0.02
        fit = stars - critic.caution * spread + args.change * np.minimum(change, 0.2)
        fit[dead] = -np.inf
        return stars, spread, change, fit, frames

    pop = np.stack([base] + [base + rng.normal(0, args.sigma, base.shape) * scale_of for _ in range(args.pop - 1)])
    stars, spread, change, fit, frames = judge(pop)
    history = [log_line(0, stars, spread, fit) | {"best_change": float(change[np.argmax(fit)])}]
    print(f"gen  0  best {stars[np.argmax(fit)]:.2f} change {change[np.argmax(fit)]:.3f}  mean {stars.mean():.2f}", flush=True)
    parent_frames = [frames[t][0].permute(1, 2, 0).cpu().numpy() for t in marks]

    for g in range(1, args.gens + 1):
        elite = np.argsort(-fit)[: args.elites]
        kids = []
        while len(kids) < args.pop - args.elites:
            a, b = pop[tournament(fit, rng)], pop[tournament(fit, rng)]
            t = rng.uniform(0.3, 0.7)
            child = (1 - t) * a + t * b                      # one basin: the average is a blend
            child = child + rng.normal(0, args.sigma / 2, child.shape) * scale_of
            kids.append(child)
        pop = np.concatenate([pop[elite], np.stack(kids)])
        stars, spread, change, fit, frames = judge(pop)
        i = int(np.argmax(fit))
        history.append(log_line(g, stars, spread, fit) | {"best_change": float(change[i])})
        print(f"gen {g:2d}  best {stars[i]:.2f} (+-{spread[i]:.2f}) change {change[i]:.3f}  mean {stars.mean():.2f}", flush=True)

    order = np.argsort(-fit)[:6]
    sheet_imgs, labels = [], []
    for i in order:
        for t in marks:
            sheet_imgs.append(frames[t][i].permute(1, 2, 0).cpu().numpy())
            labels.append(f"{stars[i]:.2f} t{t}")
    write_sheet(sheet_imgs, labels, out / "final.png",
                f"the six fittest rules, each at ticks {marks}: stars | tick", cols=len(marks) * 2)
    write_sheet(parent_frames, [f"parent t{t}" for t in marks], out / "parent.png", "the parent, before breeding",
                cols=len(marks))
    best = rule_of(pop[order[0]], args.hidden)
    np.save(out / "best.npy", S.pack(best))
    _, film = run(pop[order[:1]], keep=8)
    return dict(history=history, final_stars=[float(stars[i]) for i in order],
                final_change=[float(change[i]) for i in order], parent_steps=args.parent_steps,
                film=[f.squeeze(0).permute(1, 2, 0).cpu().numpy() for f in film])


# ─────────────────────────── 3. ip ───────────────────────────

def run_ip(args, critic):
    import ip_space as IP
    from transformers import CLIPVisionModelWithProjection

    out = pathlib.Path(args.out)
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    p = dict(IP.PRESETS["sd15"])
    enc = CLIPVisionModelWithProjection.from_pretrained(p["repo"], subfolder=p["encoder"]).to(device, torch.float16).eval()
    mean = torch.tensor(IP.CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IP.CLIP_STD, device=device).view(1, 3, 1, 1)

    paths = best_rated(args.ratings, least=4, limit=args.pop, seed=args.seed)
    pics = [square(taste.open_picture(q, 768), 224) for q in paths]
    x = torch.from_numpy(np.stack([np.asarray(q, np.float32) / 255 for q in pics])).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        emb = enc(pixel_values=((x - mean) / std).half()).image_embeds.float().cpu().numpy()
    norm = float(np.median(np.linalg.norm(emb, axis=1)))
    del enc
    torch.cuda.empty_cache()

    pipe, dtype = IP.pipeline(SimpleNamespace(style_only=False, scale=args.scale, lora=None, style=1.0), p, device)
    draw_args = SimpleNamespace(prompt="")
    eye = taste.Eye()

    def render(vecs, seeds):
        ims = [IP.draw(pipe, dtype, p, draw_args, v * (norm / np.linalg.norm(v)), int(s), device) for v, s in zip(vecs, seeds)]
        feats = np.concatenate([taste.features(eye.tokens(im)[0]) for im in ims])
        return ims, critic(feats)

    pop = emb
    seeds = rng.integers(0, 1 << 30, len(pop))
    ims, (stars, spread, fit) = render(pop, seeds)
    history = [log_line(0, stars, spread, fit)]
    print(f"gen  0  best {stars[np.argmax(fit)]:.2f}  mean {stars.mean():.2f}", flush=True)
    first = np.argsort(-fit)[:12]
    write_sheet([ims[i] for i in first], [f"{stars[i]:.2f} +-{spread[i]:.2f}" for i in first], out / "parents.png",
                "generation 0: each 4-star photograph's own vector, drawn")
    for g in range(1, args.gens + 1):
        elite = np.argsort(-fit)[: args.elites]
        kids, kseeds = [], []
        while len(kids) < args.pop - args.elites:
            i, j = tournament(fit, rng), tournament(fit, rng)
            v = IP.slerp(pop[i], pop[j], rng.uniform(0.3, 0.7))
            if rng.random() < args.mutate:
                v = v + rng.normal(0, args.sigma, v.shape) * np.linalg.norm(v) / np.sqrt(len(v))
            kids.append(v)
            kseeds.append(seeds[i] if rng.random() < 0.5 else seeds[j])
        new_ims, (s_k, sp_k, f_k) = render(kids, kseeds)
        pop = np.concatenate([pop[elite], np.stack(kids)])
        seeds = np.concatenate([seeds[elite], kseeds])
        ims = [ims[i] for i in elite] + new_ims
        stars = np.concatenate([stars[elite], s_k])
        spread = np.concatenate([spread[elite], sp_k])
        fit = np.concatenate([fit[elite], f_k])
        history.append(log_line(g, stars, spread, fit))
        print(f"gen {g:2d}  best {stars[np.argmax(fit)]:.2f} (+-{spread[np.argmax(fit)]:.2f})  mean {stars.mean():.2f}", flush=True)
    top = np.argsort(-fit)[:12]
    write_sheet([ims[i] for i in top], [f"{stars[i]:.2f} +-{spread[i]:.2f}" for i in top], out / "final.png",
                f"after {args.gens} generations")
    return dict(history=history, final_stars=[float(stars[i]) for i in top])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("route", choices=["tokens", "nca", "ip"])
    ap.add_argument("--critic", required=True)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--features", default="dataset/taste/features.npz", help="For the photographs' own scores, as a scale.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caution", type=float, default=1.0, help="How much of the critic's doubt is subtracted.")
    ap.add_argument("--label", default=None, help="Start only from photographs with this ON1 colour tag (Yellow: textures).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--elites", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--mutate", type=float, default=0.3)
    # tokens
    ap.add_argument("--ways", nargs="+", default=["patchwork", "layout"], choices=["blend", "patchwork", "layout", "caricature"])
    ap.add_argument("--apart", action="store_true",
                    help="Draw each second parent from the half of the population least like the first.")
    ap.add_argument("--score", default="drawn", choices=["drawn", "grid"],
                    help="Score each child's drawing (decoded and read back), or its grid alone.")
    ap.add_argument("--sigma", type=float, default=None)
    # nca
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--parent-steps", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--frames", type=int, nargs=3, default=[200, 400, 800])
    ap.add_argument("--change", type=float, default=1.0, help="Weight on how much the late frames still differ.")
    # ip
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    defaults = {"tokens": dict(pop=48, elites=8, gens=25, sigma=0.0),
                "nca": dict(pop=24, elites=6, gens=12, sigma=0.05),
                "ip": dict(pop=16, elites=4, gens=8, sigma=0.15)}[args.route]
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    global LABEL
    LABEL = args.label
    critic = Critic(args.critic, args.caution)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = {"tokens": run_tokens, "nca": run_nca, "ip": run_ip}[args.route](args, critic)
    film = result.pop("film", None)
    if film:
        from PIL import Image

        frames = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((320, 320), Image.NEAREST) for f in film]
        frames[0].save(out / "best.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
    result.update(photo_baseline(critic, args.features), seconds=round(time.time() - t0),
                  settings={k: v for k, v in vars(args).items() if k not in ("critic", "ratings")})
    (out / "report.json").write_text(json.dumps(result, indent=1))
    print(f"done in {result['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
