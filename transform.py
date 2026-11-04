"""Carry a picture into a material while a second dial holds its structure — content apart from look.

    python transform.py --image portrait.png --space texture-ip-sd15.npz \
        --preset "water reflections" --out grid.png

`restyle.py` and `terms.py` have one dial: denoising strength, which trades the subject away to buy
the material. This adds the missing second dial. A depth ControlNet reads the source's geometry and
holds the generation to it, so `--structure` says how firmly the subject keeps its volume while
`--strength` and `--scale` say how far the material goes. High structure with high strength is the
place neither tool could reach: a face fully constituted of the material and still unmistakably
that face.

The grid is structure × strength, because the right pair is not guessable per picture: a portrait,
a facade and a landscape each give way somewhere else. The material comes from the IP space — a
preset, or terms mixed with `+` — and `--lora` can put the archive's look underneath.

Three further moves, all on the same vector:

- `--against` puts a second material in the NEGATIVE half of the guidance, so every step pushes
  toward one and away from the other: "crust, away from water" is how a correlated axis is told no.
- `--region depth --material2 <spec>` splits the image by its own geometry (or `--region lum`, by
  its light) and gives each side its own material through the adapter's attention masks: the figure
  in crust while the ground turns to water, in one pass, no collage.
- `--then "<spec>@0.4"` hands the finished image to a second material for a lighter second pass:
  the first material builds the forms, the second redraws the surface. Scheduling by handover.

Everything pretrained: SD 1.5, its depth ControlNet, the DPT depth estimator, the IP-Adapter.
Nothing is trained and nothing leaves the machine.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw

from composite import fit
from terms import index, term_vector

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROL = "lllyasviel/control_v11f1p_sd15_depth"
DEPTH = "Intel/dpt-hybrid-midas"


def depth_map(image, device):
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    proc = DPTImageProcessor.from_pretrained(DEPTH)
    net = DPTForDepthEstimation.from_pretrained(DEPTH).to(device).eval()
    with torch.no_grad():
        out = net(**proc(images=image, return_tensors="pt").to(device)).predicted_depth
    d = torch.nn.functional.interpolate(out[None], size=image.size[::-1], mode="bicubic")[0, 0]
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    del net
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return Image.fromarray((d.cpu().numpy() * 255).astype(np.uint8)).convert("RGB")


def material_vec(space, found, presets, spec, push):
    """A material from one spec: a preset's name as-is, or terms mixed with `+`."""
    spec = spec.strip()
    if spec in presets:
        mix = dict(presets[spec])
    else:
        mix = {}
        for part in spec.split("+"):
            word, _, w = part.partition("=")
            if word.strip():
                mix[word.strip()] = float(w or 1.0)
    return term_vector(space, mix, found, push)


def embeds_of(vectors, against, dtype, device, guidance):
    """The adapter's conditioning: one positive per region, and the against-material as the negative."""
    pos = torch.tensor(np.stack(vectors), dtype=dtype, device=device)[None]
    if guidance <= 1.0:
        return pos
    if against is None:
        return torch.cat([torch.zeros_like(pos), pos])
    neg = torch.tensor(np.stack([against] * len(vectors)), dtype=dtype, device=device)[None]
    return torch.cat([neg, pos])


def region_masks(source, control, kind, thresh, width, height=None):
    """Two complementary masks from the picture itself: its depth, or its light."""
    height = height or width
    base = control if kind == "depth" else source
    g = np.asarray(base.convert("L"), np.float32) / 255.0
    near = Image.fromarray(((g >= thresh) * 255).astype(np.uint8))
    far = Image.fromarray(((g < thresh) * 255).astype(np.uint8))
    from diffusers.image_processor import IPAdapterMaskProcessor

    masks = IPAdapterMaskProcessor().preprocess([near, far], height=height, width=width)
    return [masks.reshape(1, 2, height, width)], near


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--preset", help="A named mix from --presets.")
    ap.add_argument("--term", help="Terms mixed with +, as in landmarks.py's algebra.")
    ap.add_argument("--presets", default="presets.json")
    ap.add_argument("--push", type=float, default=8.0)
    ap.add_argument("--structure", type=float, nargs="*", default=[1.2, 0.8, 0.4], help="Rows: how firmly depth holds.")
    ap.add_argument("--strength", type=float, nargs="*", default=[0.55, 0.75, 0.95], help="Columns: how much is redrawn.")
    ap.add_argument("--scale", type=float, default=1.0, help="How hard the material vector pulls.")
    ap.add_argument("--against", help="A material spec for the negative half: toward the material, away from this.")
    ap.add_argument("--region", choices=["depth", "lum"], help="Split the picture by geometry or by light.")
    ap.add_argument("--material2", help="With --region: the far/dark side's material spec.")
    ap.add_argument("--rthresh", type=float, default=0.5, help="Where the region split sits, 0 to 1.")
    ap.add_argument("--then", help="`spec@strength`: a second, lighter pass in another material.")
    ap.add_argument("--lora", help="A LoRA folder for the archive's look underneath.")
    ap.add_argument("--style", type=float, default=1.0)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not (args.preset or args.term):
        raise SystemExit("give --preset or --term for the material")

    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source = fit(args.image, args.size)
    control = depth_map(source, device)

    space = dict(np.load(args.space, allow_pickle=True))
    found = index(space)
    presets = {k: v for k, v in json.loads(pathlib.Path(args.presets).read_text(encoding="utf-8")).items()
               if not k.startswith("_")}
    label = args.preset or args.term
    vectors = [material_vec(space, found, presets, label, args.push)]
    against = material_vec(space, found, presets, args.against, args.push) if args.against else None
    masks, near = None, None
    if args.region:
        if not args.material2:
            raise SystemExit("--region needs --material2 for the far/dark side")
        vectors.append(material_vec(space, found, presets, args.material2, args.push))
        masks, near = region_masks(source, control, args.region, args.rthresh, args.size)
        label += f"  |  {args.material2} ({args.region} < {args.rthresh:g})"
    if args.against:
        label += f"  |  against {args.against}"
    emb = embeds_of(vectors, against, dtype, device, args.guidance)
    then_vec, then_s = None, 0.4
    if args.then:
        spec, _, s = args.then.partition("@")
        then_vec = material_vec(space, found, presets, spec, args.push)
        then_s = float(s or then_s)
        then_emb = embeds_of([then_vec], against, dtype, device, args.guidance)
        label += f"  |  then {spec}@{then_s:g}"

    net = ControlNetModel.from_pretrained(CONTROL, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE, controlnet=net, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors",
                         image_encoder_folder=None)
    first_scale = [[args.scale] * len(vectors)] if masks else args.scale
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=args.style)
    pipe.set_progress_bar_config(disable=True)

    side, gut, label_h = args.size, 170, 30
    sheet = Image.new("RGB", (gut + len(args.strength) * side, label_h + (len(args.structure) + 1) * side), (18, 18, 20))
    pen = ImageDraw.Draw(sheet)
    pen.text((gut + 8, 8), f"material: {label}  ·  columns: strength {', '.join(f'{s:g}' for s in args.strength)}", fill=(214, 210, 202))
    sheet.paste(source, (gut, label_h))
    if len(args.strength) > 1:
        sheet.paste(control.resize((side, side)), (gut + side, label_h))
    if near is not None and len(args.strength) > 2:
        sheet.paste(near.convert("RGB").resize((side, side)), (gut + 2 * side, label_h))
    pen.text((8, label_h + side // 2), "source · depth" + (" · split" if near is not None else ""), fill=(172, 170, 164))
    for r, structure in enumerate(args.structure):
        y = label_h + (r + 1) * side
        pen.text((8, y + side // 2), f"structure {structure:g}", fill=(172, 170, 164))
        for c, strength in enumerate(args.strength):
            pipe.set_ip_adapter_scale(first_scale)
            extra = {"cross_attention_kwargs": {"ip_adapter_masks": masks}} if masks else {}
            if then_vec is not None:
                # The handover travels as LATENTS. A drawn texture decoded to pixels and encoded
                # again is adversarial to img2img — the round trip confettis even with the adapter
                # silent — while the same picture handed over inside the VAE's space redraws clean.
                extra["output_type"] = "latent"
            with torch.no_grad():
                done = pipe(
                    prompt=args.prompt,
                    image=source,
                    control_image=control,
                    strength=strength,
                    controlnet_conditioning_scale=float(structure),
                    ip_adapter_image_embeds=[emb],
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=torch.Generator(args.device).manual_seed(args.seed),
                    **extra,
                ).images
            if then_vec is not None:
                pipe.set_ip_adapter_scale(args.scale)
                with torch.no_grad():
                    done = pipe(
                        prompt=args.prompt,
                        image=done,
                        control_image=control,
                        strength=then_s,
                        controlnet_conditioning_scale=float(structure),
                        ip_adapter_image_embeds=[then_emb],
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator(args.device).manual_seed(args.seed),
                    ).images
            sheet.paste(done[0], (gut + c * side, y))
        print(f"  structure {structure:g}", flush=True)
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
