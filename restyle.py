"""Put a photograph that is not in the archive into the archive's look.

    python restyle.py --image a.jpg --lora "gentle=sd15-lora-gentle/step-002000@1.6" --out sheet.png

The LoRA and the IP space both answer "draw me something new". This answers a different question:
take a picture that already exists and carry it toward the archive without losing what it is.

That is img2img, and it turns on two numbers. Denoising strength says how much of the original is
thrown away before redrawing: too little and the look never arrives, too much and the subject is
gone and you have merely generated a texture. How far a picture can travel depends on how hard its
structure is, so the number is not guessable and the sweep is drawn rather than chosen. The second
number is how hard the LoRA pulls, written `@1.6` after its folder.

Given several LoRAs it draws them as columns, which is the only way to see what each one actually
contributes: same picture, same seed, same strength, one thing changed.
"""

import argparse
import pathlib

import torch
from PIL import Image, ImageDraw

LOOK = "a close-up photograph, encrusted and weathered, mineral and wet, in goofi style"


def fit(path, size):
    """The picture, square and at the model's size, cropped from the middle rather than squashed."""
    got = Image.open(path).convert("RGB")
    side = min(got.size)
    left, top = (got.width - side) // 2, (got.height - side) // 2
    return got.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def labelled(rows, names, heads, size, gutter=150):
    sheet = Image.new("RGB", (gutter + len(heads) * size, 34 + len(rows) * size), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    for i, head in enumerate(heads):
        pen.text((gutter + i * size + 8, 12), head, fill=(214, 210, 202))
    for r, (row, name) in enumerate(zip(rows, names)):
        pen.text((8, 34 + r * size + size // 2), name, fill=(172, 170, 164))
        for c, drawn in enumerate(row):
            sheet.paste(drawn, (gutter + c * size, 34 + r * size))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", nargs="+", required=True, help="Pictures to carry over.")
    ap.add_argument("--lora", nargs="*", default=[], help="`label=folder`, optionally `folder@strength`.")
    ap.add_argument("--bare", action="store_true", help="Also draw a column with no LoRA at all.")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--prompt", default=LOOK)
    ap.add_argument("--strength", type=float, nargs="*", default=[0.5],
                    help="How much of the original to throw away.")
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

    sources = {where: fit(where, args.size) for where in args.image}
    drawn = {}
    heads = []

    # Grouped by LoRA so each one is loaded once, and unfused before the next so none of them stack.
    for where in ([None] if args.bare else []) + list(args.lora):
        if where is None:
            label = "no LoRA"
        else:
            label, _, given = where.rpartition("=")
            given, _, pull = given.partition("@")
            pipe.load_lora_weights(given, weight_name="pytorch_lora_weights.safetensors")
            pipe.fuse_lora(lora_scale=float(pull) if pull else 1.0)
            label = label or pathlib.Path(given).name
        for strength in args.strength:
            head = label if len(args.strength) == 1 else f"{label} · {strength}"
            heads.append(head)
            for src in args.image:
                with torch.no_grad():
                    drawn[(head, src)] = pipe(
                        prompt=args.prompt,
                        image=sources[src],
                        strength=strength,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator(args.device).manual_seed(args.seed),
                    ).images[0]
        if where is not None:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        print(f"  drew {label}")

    rows = [[sources[src]] + [drawn[(h, src)] for h in heads] for src in args.image]
    names = [pathlib.Path(src).stem[:20] for src in args.image]
    labelled(rows, names, ["the picture"] + heads, args.size).save(args.out)
    print(f"wrote {args.out}: {len(rows)} pictures across {len(heads)} columns")


if __name__ == "__main__":
    main()
