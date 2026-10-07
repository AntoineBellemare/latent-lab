"""Draw the same prompts, with the same seeds, through several LoRAs at once.

    python compare.py --lora sd15-lora-captioned/step-* --out compare.png

Choosing a checkpoint by looking at each one's own sheet does not work: every sheet was drawn from
different noise, so the one that happens to land a good seed looks like the better model. Here the
seed and the prompt are held still and only the LoRA changes, which is the only way the difference
on screen is the thing being measured.

The base model draws the top row of every block. A LoRA that has learned nothing looks like it, and
a LoRA that has burned looks like neither it nor the photographs.
"""

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

ASKS = [
    "a close-up photograph of lichen on stone, encrusted and mottled, in rust and olive drab, in flat overcast light, in goofi style",
    "a close-up photograph of rippled water, glassy and swirling, in teal and bronze, submerged under clear water, in goofi style",
    "a close-up photograph of oxidised copper, veined and flaking, in turquoise and umber, wet and glistening, in goofi style",
    "a close-up photograph of a mineral vein, striated and fissured, in amber and charcoal, sunlit with hard shadows, in goofi style",
]


def strip(pipe, vae, text, seeds, steps, guidance, size, device):
    """One row: the same prompt at each seed."""
    out = []
    for seed in seeds:
        with torch.no_grad():
            lat = pipe(
                prompt=text,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=size,
                width=size,
                generator=torch.Generator(device).manual_seed(seed),
                output_type="latent",
            ).images
            rgb = vae.decode(lat.float() / vae.config.scaling_factor).sample
        out.append(((rgb[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy())
    return np.concatenate(out, 1)


def labelled(rows, names, asks, size, gutter=230):
    """The grid, with each variant named down the left and each prompt titled above its block."""
    width = gutter + rows[0][0].shape[1]
    tall = sum(40 + len(block) * size for block in rows)
    sheet = Image.new("RGB", (width, tall), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    y = 0
    for block, ask in zip(rows, asks):
        pen.text((8, y + 14), ask[:120], fill=(210, 206, 198))
        y += 40
        for drawn, name in zip(block, names):
            sheet.paste(Image.fromarray(drawn), (gutter, y))
            pen.text((8, y + size // 2), name, fill=(170, 168, 162))
            y += size
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lora", nargs="*", default=[], help="`label=folder`, optionally `folder@strength`.")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", nargs="*", default=ASKS)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--scale", type=float, default=1.0, help="How strongly to apply each LoRA.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from diffusers import AutoencoderKL, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(args.device).eval()
    seeds = list(range(args.seeds))

    # The base first, then each LoRA in turn, unloaded before the next so none of them stack.
    names, blocks = ["base"], [[]]
    for ask in args.prompt:
        blocks[0].append(strip(pipe, vae, ask, seeds, args.steps, args.guidance, args.size, args.device))
    for where in args.lora:
        # Labelled by hand where it matters: two runs both hold a `step-002000`, and a sheet whose
        # rows cannot be told apart measures nothing.
        label, _, given = where.rpartition("=")
        # A trailing `@1.4` sets how hard THIS one is applied, so one checkpoint can be drawn at
        # several strengths side by side without retraining anything.
        given, _, strength = given.partition("@")
        path = pathlib.Path(given)
        pipe.load_lora_weights(str(path), weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=float(strength) if strength else args.scale)
        names.append(label or path.name)
        blocks.append([strip(pipe, vae, a, seeds, args.steps, args.guidance, args.size, args.device) for a in args.prompt])
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        print(f"  drew {names[-1]}")

    # Transposed: one block per prompt, every variant stacked inside it for a straight comparison.
    rows = [[blocks[v][p] for v in range(len(names))] for p in range(len(args.prompt))]
    labelled(rows, names, args.prompt, args.size).save(args.out)
    print(f"wrote {args.out}: {len(names)} variants x {len(args.prompt)} prompts x {args.seeds} seeds")


if __name__ == "__main__":
    main()
