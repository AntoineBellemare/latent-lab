"""Put a photograph that is not in the archive into the archive's look.

    python restyle.py --image a.jpg b.jpg --lora sd15-lora-gentle/step-002000 --out restyled.png

The LoRA and the IP space both answer "draw me something new". This answers a different question:
take a picture that already exists and carry it toward the archive without losing what it is.

That is img2img, and the whole thing turns on one number. Denoising strength says how much of the
original is thrown away before redrawing: too little and the look never arrives, too much and the
subject is gone and you have merely generated a texture. The number is not guessable and depends on
the picture, so this draws the sweep and lets the eye pick.
"""

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

LOOK = "a close-up photograph, encrusted and weathered, mineral and wet, in goofi style"


def fit(path, size):
    """The picture, square and at the model's size, cropped from the middle rather than squashed."""
    got = Image.open(path).convert("RGB")
    side = min(got.size)
    left, top = (got.width - side) // 2, (got.height - side) // 2
    return got.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def labelled(rows, names, heads, size, gutter=170):
    sheet = Image.new("RGB", (gutter + len(heads) * size, 30 + len(rows) * size), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    for i, head in enumerate(heads):
        pen.text((gutter + i * size + 8, 10), head, fill=(210, 206, 198))
    for r, (row, name) in enumerate(zip(rows, names)):
        pen.text((8, 30 + r * size + size // 2), name, fill=(170, 168, 162))
        for c, drawn in enumerate(row):
            sheet.paste(drawn, (gutter + c * size, 30 + r * size))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", nargs="+", required=True, help="Pictures to carry over.")
    ap.add_argument("--lora", help="A LoRA folder from lora.py.")
    ap.add_argument("--style", type=float, default=1.4, help="How hard the LoRA pulls.")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--prompt", default=LOOK)
    ap.add_argument("--strength", type=float, nargs="*", default=[0.3, 0.45, 0.6, 0.75],
                    help="How much of the original to throw away. The sweep is the point.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=args.style)

    rows, names = [], []
    for where in args.image:
        source = fit(where, args.size)
        row = [source]
        for strength in args.strength:
            with torch.no_grad():
                row.append(
                    pipe(
                        prompt=args.prompt,
                        image=source,
                        strength=strength,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator(args.device).manual_seed(args.seed),
                    ).images[0]
                )
        rows.append(row)
        names.append(pathlib.Path(where).stem[:22])
        print(f"  carried over {pathlib.Path(where).name}")

    heads = ["the picture"] + [f"strength {s}" for s in args.strength]
    labelled(rows, names, heads, args.size).save(args.out)
    print(f"wrote {args.out}: {len(rows)} pictures across {len(args.strength)} strengths")


if __name__ == "__main__":
    main()
