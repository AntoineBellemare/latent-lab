"""The material drawn ON the things in a picture: zero-shot material transfer on SDXL.

ZeST (Cheng et al. 2024, "ZeST: Zero-Shot Material Transfer from a Single Image") draws a material
onto an object by three moves, all in one SDXL pass:

- the object is inpainted INSIDE its mask (ControlNet inpainting, the mask honoured at every
  step), the rest of the picture held;
- a depth ControlNet holds the object's shape (here from the geometry model's distances);
- the object is first turned GREY: its own colours gone, its light and shade kept, so the redraw
  keeps how the scene lights it and takes its colours and grain from the material alone.

The material enters through an IP-Adapter. This one, `ip-adapter_sdxl_vit-h`, reads the same
CLIP ViT-H embedding as the SD 1.5 adapter the lab's material space is built in — so every place
in the space, every photograph, every concept axis (concepts.json) steers it unchanged.

MARBLE (Cheng et al. 2025, Stability AI Community Licence) adds two things: the material may be
injected into the one UNet block that carries style and material (up_blocks.0.attentions.1,
InstantStyle's), leaving the picture's content to the rest; and four small networks, trained in
the same ViT-H space, that move an embedding toward metallic, rough, transparent or glowing by
a slider. Their weights live in <models>/marble/.
"""

import pathlib
import threading

import numpy as np
import torch
from PIL import Image

BASE = "SG161222/RealVisXL_V4.0"
DEPTH = "xinsir/controlnet-depth-sdxl-1.0"
VAE = "madebyollin/sdxl-vae-fp16-fix"
ADAPTER = ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl_vit-h.safetensors")
MARBLE_URL = "https://raw.githubusercontent.com/Stability-AI/marble/main/model_weights/{}.pt"
PROPERTIES = {  # MARBLE's sliders and the ranges its demo offers
    "metallic": (-20.0, 20.0), "roughness": (-1.0, 1.0), "transparency": (0.0, 4.0), "glow": (0.0, 3.0)}
STYLE_BLOCK = {"up": {"block_0": [0.0, 1.0, 0.0]}}


class ControlMLP(torch.nn.Module):
    """MARBLE's attribute network: an embedding and a strength in, an edit direction out."""

    def __init__(self, features=1024):
        super().__init__()
        self.fc1, self.fc2 = torch.nn.Linear(features, 1024), torch.nn.Linear(1024, 2048)
        self.edit_strength_fc1, self.edit_strength_fc2 = torch.nn.Linear(1, 128), torch.nn.Linear(128, 2)

    def forward(self, x, strength):
        x = self.fc2(torch.relu(self.fc1(x)))
        s = self.edit_strength_fc2(torch.relu(self.edit_strength_fc1(strength[:, None])))
        return s[:, :1] * x[:, :1024] + s[:, 1:] * x[:, 1024:]


class Transfer:
    def __init__(self, device, models_dir):
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dir = pathlib.Path(models_dir) / "marble"
        self.pipe = None
        self.mlps = {}
        self.looks = {}  # LoRA folders loaded, by adapter name
        self.lock = threading.Lock()

    def load(self):
        if self.pipe is None:
            from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                                   StableDiffusionXLControlNetInpaintPipeline)

            from models import MODELS

            with MODELS.loading("transfer: SDXL", lambda: setattr(self, "pipe", None), self.lock, guess=12.0):
                net = ControlNetModel.from_pretrained(DEPTH, torch_dtype=self.dtype)
                vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=self.dtype)
                pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    BASE, controlnet=net, vae=vae, torch_dtype=self.dtype)
                repo, sub, weight = ADAPTER
                pipe.load_ip_adapter(repo, subfolder=sub, weight_name=weight, image_encoder_folder=None)
            # Ten gigabytes of SDXL beside the lab's other models would fill the card, and a full
            # card on Windows spills into shared memory and slows every job to a crawl. The
            # pipeline lives in RAM and each model visits the GPU only while it runs: a second or
            # so per draw, and the card stays free for the rest of the lab.
            pipe.enable_model_cpu_offload(device=self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
            print("  transfer pipeline up (SDXL + depth + ViT-H adapter)")
        from models import MODELS

        MODELS.used("transfer: SDXL")
        return self.pipe

    def look(self, folder, weight):
        """The archive's SDXL LoRA under the draw (lora.py, trained on RealVisXL), or none."""
        pipe = self.load()
        if not folder:
            if self.looks:
                pipe.disable_lora()
            return
        name = "look_" + "".join(ch if ch.isalnum() else "_" for ch in str(folder))
        if name not in self.looks:
            pipe.load_lora_weights(str(folder), weight_name="pytorch_lora_weights.safetensors", adapter_name=name)
            self.looks[name] = folder
        pipe.enable_lora()
        pipe.set_adapters([name], [float(weight)])

    def mlp(self, name):
        if name not in self.mlps:
            path = self.dir / f"{name}.pt"
            if not path.is_file():
                import urllib.request

                self.dir.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MARBLE_URL.format(name), path)
            net = ControlMLP()
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.mlps[name] = net.to(self.device, self.dtype).eval()
        return self.mlps[name]

    def edited(self, vec, properties):
        """The embedding moved by MARBLE's attribute networks, each by its slider."""
        v = torch.tensor(np.asarray(vec, np.float32)[None], dtype=self.dtype, device=self.device)
        moved = v.clone()
        for name, s in properties.items():
            if s:
                moved = moved + self.mlp(name)(v, torch.tensor([float(s)], dtype=self.dtype, device=self.device))
        return moved

    def __call__(self, picture, mask, depth, vec, *, properties=None, prompt="", steps=30, guidance=5.0,
                 strength=0.95, structure=0.9, scale=1.0, block="all", grey=1.0, seed=0, look=None, look_w=1.0,
                 trigger="in goofi style"):
        """`picture` (PIL), `mask` (h, w) in [0, 1], `depth` (PIL, near bright), `vec` the ViT-H
        material embedding. The picture greyed `grey` of the way inside the mask, then redrawn there."""
        w, h = picture.size
        a = np.asarray(picture.convert("RGB"), np.float32) / 255.0
        lum = (a @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        m = np.clip(mask, 0.0, 1.0)[..., None] * float(grey)
        init = Image.fromarray(((a * (1.0 - m) + lum * m) * 255.0 + 0.5).astype(np.uint8))
        mask_im = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L")
        with self.lock, torch.no_grad():
            pipe = self.load()  # under the lock: a live draft and a job asking at once load it once
            self.look(look, look_w)
            if look and trigger not in (prompt or ""):
                prompt = (prompt or "a close-up photograph") + ", " + trigger
            v = self.edited(vec, properties or {})
            embeds = torch.cat([torch.zeros_like(v), v])[:, None]  # (uncond, cond) × one image × 1024
            pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, float(scale), 0.0]}} if block == "style" else float(scale))
            return pipe(prompt=prompt or "a photograph, best quality",
                        negative_prompt="monochrome, lowres, worst quality, low quality, blurry",
                        image=init, mask_image=mask_im, control_image=depth, width=w, height=h,
                        controlnet_conditioning_scale=float(structure), strength=float(strength),
                        ip_adapter_image_embeds=[embeds], num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        generator=torch.Generator(self.device).manual_seed(int(seed))).images[0]


def disparity_image(geo):
    """The geometry model's distances as a depth ControlNet expects them: near bright, far dark."""
    inv = 1.0 / np.maximum(geo["depth"], 1e-3)
    inv[geo["sky"]] = float(inv.min())
    inv = (inv - inv.min()) / max(float(np.ptp(inv)), 1e-6)
    return Image.fromarray((inv * 255.0 + 0.5).astype(np.uint8)).convert("RGB")
