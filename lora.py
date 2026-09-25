"""Teach a diffusion model your images, as a LoRA the `Diffusion` node loads.

A LoRA is a small set of extra numbers on the attention layers — tens of megabytes against the base
model's gigabytes — so one image database becomes one file, and several styles are several files
that swap without reloading the model.

    python lora.py --images ~/pictures/plates --out plates-lora --trigger "in plt style"

Then set `Diffusion`'s `lora` to the folder it wrote and put the trigger in the prompt. `style`
scales how hard it pulls.

For a set of big photographs, and one LoRA that knows every category of it:

    python lora.py --images ~/pictures/set-by-category --classes --detail 3 \\
        --name cat0=water cat1=rock cat2=waves --out set-lora

`--detail` is the same setting `style_ca.py` and `fastgan.py` take, and it matters as much here: at 1
a 6000-pixel macro is squeezed whole into 512 pixels and the grain is gone before the model sees it.
`--classes` takes each image's category from its folder, draws the categories evenly, and captions
each image with `--caption`, so the prompt picks the material afterwards.

Captions come from a `.txt` beside each image where there is one, and from `--caption` otherwise.
Every caption is encoded ONCE and reused, because a run with few captions would otherwise spend its
time in the text encoder.

SD 1.x/2.x (`sd-turbo`) and SDXL (`sdxl-turbo`) bases both train here. SDXL is the one the pretrained
IP-Adapters and most ControlNets were made for, which is what `ip_space.py` needs.
"""

import argparse
import json
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def images(where):
    found = sorted(p for p in pathlib.Path(where).expanduser().rglob("*") if p.suffix.lower() in SUFFIXES)
    if not found:
        raise SystemExit(f"no images under {where}")
    return found


def square(path, size, detail):
    """One crop as the VAE takes it: square, a `detail`-th of the frame's short side, in [-1, 1].

    Opened per step rather than held: a set of big photographs held at `size * detail` is tens of
    gigabytes. `draft` lets a JPEG decode at a fraction of its size when that still covers the crop,
    which is most of the cost of opening a 24-megapixel frame.
    """
    side = max(size, round(size * detail))
    with Image.open(path) as im:
        shrink = min(im.size) / side
        if shrink >= 2:
            im.draft("RGB", (round(im.width / shrink), round(im.height / shrink)))
        im = im.convert("RGB")
    scale = side / min(im.size)
    im = im.resize((max(side, round(im.width * scale)), max(side, round(im.height * scale))), Image.LANCZOS)
    left = random.randint(0, im.width - size)
    top = random.randint(0, im.height - size)
    im = im.crop((left, top, left + size, top + size))
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        # A texture has no up: a vertical flip is as honest as a horizontal one, and doubles the set again.
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return torch.tensor(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


class Crops(torch.utils.data.IterableDataset):
    """An endless stream of crops, categories drawn evenly when there are any.

    Evenly, as `fastgan.py --classes` does: drawn by file, the largest category is trained on three
    times as hard as the smallest, and the prompt for the smallest ends up the weakest.
    """

    def __init__(self, files, size, detail, classes, caption_of):
        self.groups = {}
        for p in files:
            self.groups.setdefault(p.parent.name if classes else "", []).append(p)
        self.keys = sorted(self.groups)
        self.size, self.detail, self.caption_of = size, detail, caption_of

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        random.seed(torch.initial_seed() + (info.id if info else 0))
        while True:
            path = random.choice(self.groups[random.choice(self.keys)])
            try:
                yield square(path, self.size, self.detail), self.caption_of(path)
            except OSError as e:
                print(f"  skipped {path.name}: {e}", flush=True)


def names_of(pairs):
    out = {}
    for pair in pairs:
        folder, _, name = pair.partition("=")
        if not name:
            raise SystemExit(f"--name takes folder=words, got {pair!r}")
        out[folder] = name
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of images, read recursively.")
    ap.add_argument("--out", required=True, help="A folder to write the LoRA into.")
    ap.add_argument("--trigger", default="in goofi style", help="The word that ties a prompt to this set.")
    ap.add_argument("--classes", action="store_true", help="Each image's category is the folder it sits in.")
    ap.add_argument("--name", nargs="*", default=[], help="folder=words, what to call a category in the caption.")
    ap.add_argument(
        "--caption",
        default=None,
        help="The caption for images with no `.txt` beside them. {name} is the category, {trigger} the trigger. "
        "Default: the trigger alone, or 'a close-up photograph of {name}, {trigger}' with --classes.",
    )
    ap.add_argument("--detail", type=float, default=1.0, help="How much of the frame a crop sees: 3 is a third of it.")
    ap.add_argument("--model", default="stabilityai/sd-turbo", help="Any SD 1.x/2.x or SDXL base, e.g. stabilityai/sdxl-turbo.")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank. Higher holds more, and overfits sooner.")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4, help="Processes decoding crops while the GPU trains.")
    ap.add_argument("--report", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=2, help="Denoising steps for the sheet it draws at the end.")
    ap.add_argument("--guidance", type=float, default=0.0, help="Guidance for the sheet: 0 for turbo bases, ~5 otherwise.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from diffusers import AutoPipelineForText2Image, DDPMScheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    device = torch.device(args.device)
    frozen = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=frozen, safety_checker=None)
    pipe.to(device)
    xl = getattr(pipe, "text_encoder_2", None) is not None
    unet, vae = pipe.unet, pipe.vae
    # The VAE in fp32: SDXL's overflows to NaN in half precision, and it only encodes, so the cost is small.
    vae.to(torch.float32)
    for part in (unet, vae, pipe.text_encoder, getattr(pipe, "text_encoder_2", None)):
        if part is not None:
            part.requires_grad_(False)
    # The adapters alone in fp32: a LoRA update is small against a half-precision weight and
    # would round away to nothing.
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=TARGETS))
    trained = [q for q in unet.parameters() if q.requires_grad]
    for q in trained:
        q.data = q.data.float()
    print(f"{sum(q.numel() for q in trained) / 1e6:.2f}M trained parameters, rank {args.rank}, {'SDXL' if xl else 'SD'} base")

    files = images(args.images)
    named = names_of(args.name)
    template = args.caption or ("a close-up photograph of {name}, {trigger}" if args.classes else "{trigger}")

    def caption_of(path):
        beside = path.with_suffix(".txt")
        if beside.is_file():
            return beside.read_text(encoding="utf-8").strip()
        folder = path.parent.name
        return template.format(name=named.get(folder, folder), trigger=args.trigger)

    stream = Crops(files, args.size, args.detail, args.classes, caption_of)
    print(f"{len(files)} images under {args.images}, training at {args.size} on {device}, a crop sees 1/{args.detail:g} of a frame")
    for key in stream.keys:
        print(f"  {key or '(all)'}: {len(stream.groups[key])} images, e.g. {caption_of(stream.groups[key][0])!r}")
    loader = torch.utils.data.DataLoader(stream, batch_size=args.batch, num_workers=args.workers)

    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    optimiser = torch.optim.AdamW(trained, lr=args.lr, weight_decay=1e-2)
    said = {}

    def embedding(text):
        """The text as the UNet reads it, and for SDXL the pooled vector and size tags it also wants."""
        if text not in said:
            with torch.no_grad():
                if xl:
                    told, _, pooled, _ = pipe.encode_prompt(
                        prompt=text, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                else:
                    told, pooled = pipe.encode_prompt(text, device, 1, False)[0], None
            said[text] = (told, pooled)
        return said[text]

    # SDXL is told the size and crop of what it is shown. Every crop here is a whole picture at `size`:
    # the model should draw one, not a piece cut from a bigger one.
    time_ids = torch.tensor([[args.size, args.size, 0, 0, args.size, args.size]], device=device, dtype=frozen)

    batches = iter(loader)
    for step in range(args.steps):
        pixels, texts = next(batches)
        with torch.no_grad():
            latents = vae.encode(pixels.to(device, torch.float32)).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.to(frozen)
        encoded = [embedding(t) for t in texts]
        told = torch.cat([e[0] for e in encoded])
        extra = {}
        if xl:
            extra["added_cond_kwargs"] = {
                "text_embeds": torch.cat([e[1] for e in encoded]),
                "time_ids": time_ids.repeat(len(texts), 1),
            }

        noise = torch.randn_like(latents)
        when = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, when)
        want = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, when)

        got = unet(noisy, when, encoder_hidden_states=told, **extra).sample
        loss = F.mse_loss(got.float(), want.float())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optimiser.step()
        if step % args.report == 0 or step == args.steps - 1:
            print(f"step {step:>6}  loss {loss.item():.5f}", flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe.__class__.save_lora_weights(
        save_directory=str(out),
        unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
    )
    prompts = {key or "(all)": caption_of(stream.groups[key][0]) for key in stream.keys}
    # What was trained and how to ask for it, so a sampler does not have to be told again.
    (out / "latent-lab.json").write_text(
        json.dumps({"base": args.model, "trigger": args.trigger, "detail": args.detail, "size": args.size, "prompts": prompts}, indent=2),
        encoding="utf-8",
    )
    sheet(pipe, vae, prompts, args, device).save(out / "sheet.png")
    print(f"wrote {out} and a sheet of every prompt — set Diffusion's `lora` to it, and put `{args.trigger}` in the prompt")


def sheet(pipe, vae, prompts, args, device, seeds=6):
    """One row per prompt, `seeds` draws each: whether each category's caption actually means something.

    Drawn to latents and decoded here, because the VAE is in fp32 and the pipeline would hand it fp16.
    """
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for text in prompts.values():
        row = []
        for seed in range(seeds):
            with torch.no_grad():
                lat = pipe(
                    prompt=text,
                    num_inference_steps=args.sample_steps,
                    guidance_scale=args.guidance,
                    height=args.size,
                    width=args.size,
                    generator=torch.Generator(device).manual_seed(seed),
                    output_type="latent",
                ).images
                rgb = vae.decode(lat.float() / vae.config.scaling_factor).sample
            row.append(((rgb[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy())
        rows.append(np.concatenate(row, 1))
    return Image.fromarray(np.concatenate(rows, 0))


if __name__ == "__main__":
    main()
