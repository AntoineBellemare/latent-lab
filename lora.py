"""Teach a diffusion model your images, as a LoRA the `Diffusion` node loads.

A LoRA is a small set of extra numbers on the attention layers — tens of megabytes against the base
model's gigabytes — so one image database becomes one file, and several styles are several files
that swap without reloading the model.

    python lora.py --images ~/pictures/plates --out plates-lora --trigger "in plt style"

Then set `Diffusion`'s `lora` to the folder it wrote and put the trigger in the prompt. `style`
scales how hard it pulls.

Captions come from a `.txt` beside each image where there is one, and from `--trigger` otherwise.
Every caption is encoded ONCE and reused, because a run with one trigger would otherwise spend its
time in the text encoder.
"""

import argparse
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


def square(path, size):
    """One image as the VAE takes it: square, and in [-1, 1]."""
    im = Image.open(path).convert("RGB")
    scale = size / min(im.size)
    im = im.resize((max(size, round(im.width * scale)), max(size, round(im.height * scale))), Image.LANCZOS)
    left = random.randint(0, im.width - size)
    top = random.randint(0, im.height - size)
    im = im.crop((left, top, left + size, top + size))
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.tensor(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def caption(path, trigger):
    beside = path.with_suffix(".txt")
    return beside.read_text(encoding="utf-8").strip() if beside.is_file() else trigger


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="A folder of images, read recursively.")
    ap.add_argument("--out", required=True, help="A folder to write the LoRA into.")
    ap.add_argument("--trigger", default="in goofi style", help="The caption for images with no `.txt` beside them.")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank. Higher holds more, and overfits sooner.")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--report", type=int, default=100)
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
    unet, vae, encoder = pipe.unet, pipe.vae, pipe.text_encoder
    for part in (unet, vae, encoder):
        part.requires_grad_(False)
    # The adapters alone in fp32: a LoRA update is small against a half-precision weight and
    # would round away to nothing.
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=TARGETS))
    trained = [q for q in unet.parameters() if q.requires_grad]
    for q in trained:
        q.data = q.data.float()
    print(f"{sum(q.numel() for q in trained) / 1e6:.2f}M trained parameters, rank {args.rank}")

    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    optimiser = torch.optim.AdamW(trained, lr=args.lr, weight_decay=1e-2)
    files = images(args.images)
    print(f"{len(files)} images under {args.images}, training at {args.size} on {device}")

    said = {}

    def embedding(text):
        if text not in said:
            tokens = pipe.tokenizer(
                text, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            with torch.no_grad():
                said[text] = encoder(tokens)[0]
        return said[text]

    for step in range(args.steps):
        picked = [random.choice(files) for _ in range(args.batch)]
        pixels = torch.stack([square(p, args.size) for p in picked]).to(device, dtype=frozen)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        told = torch.cat([embedding(caption(p, args.trigger)) for p in picked])

        noise = torch.randn_like(latents)
        when = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, when)
        want = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, when)

        got = unet(noisy, when, encoder_hidden_states=told).sample
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
    print(f"wrote {out} — set Diffusion's `lora` to it, and put `{args.trigger}` in the prompt")


if __name__ == "__main__":
    main()
