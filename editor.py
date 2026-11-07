"""An instruction-following editor: the material shown to it as a picture, the place as a mask.

FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0) edits by looking: given the picture, a mask and
a REFERENCE image, its inpainting pipeline redraws the masked part with the reference in view and
an instruction in words. It understands "make the dress out of this" the way an editor model
does — the reference's material put onto the thing's own folds, seams and light — where an
adapter only hears an embedding.

The reference is a SWATCH. It can be any texture photograph; or a place in the lab's material
space drawn by the SD 1.5 space pipeline first, so the concept axes (humid, fractured, ...) still
steer what the editor is shown: the space chooses the material, the editor puts it on.

The model is step-distilled: four steps, no guidance. Its image model (8 GB) and VAE live on the
card: moved in and out for every picture, as they first were, they cost half of each edit (21 s
against 10 at 1024 and 8 steps, 16 against 5 for a draft). The 8 GB text encoder stays in RAM and
visits the card only for an instruction it has not read yet; what it makes of one is kept. When
another big pipeline needs the card, `park` moves the image model aside until the next edit.

Each step reads the picture being drawn, the picture again as reference, the swatch and the words,
so the time goes with how many of those tokens there are (measured on five atlas cards, 2026-10-28):
the swatch is read at 512 px (768 cost 13% more for the same material), the words are padded to 128
tokens rather than 512 (an instruction is some forty), full quality is 6 steps (8 looked the same and
cost 22% more), and when the mask holds only part of the frame the editor works AROUND it — the
mask's extent with a margin of context, drawn at no less than 640 px and laid back — which is 40%
faster there and finer, a small face getting more pixels than it had. A full edit: 3–7 s, from 21.

A RAMP grades the redraw inside the mask: differential diffusion holds each place to the picture
until only its share of the steps is left, so a gradient reads as the thing turning into the
material — its own forms holding where the ramp is low — rather than one picture faded over another.
"""

import collections
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import MODELS

KLEIN = "black-forest-labs/FLUX.2-klein-4B"
# Said plainly, naming the thing: "make the masked surfaces out of ..." leaves a dress as it was;
# "replace the fabric of the dress with ..." changes it.
INSTRUCTION = ("Replace the surface of {what} with the material from the reference image: its texture, grain "
               "and colour, following the shape, folds and lighting of {what}.")


def instruction(what):
    return INSTRUCTION.format(what=what or "the masked area")


# What a step's callback is handed besides the latents: the picture's own latents and the noise
# they are brought back to at each step (the pipeline offers only the latents and the prompt).
HELD = ["latents", "image_latents", "noise", "timesteps"]


def graded(pipe, ramp, w, h):
    """Differential diffusion (Levin & Fried, 2023): a place whose ramp is r is held to the picture
    until the last r of the steps, then let go. At 1 it is redrawn from noise; at 0.3 only the last
    third of the steps reach it, so its forms and colours hold and the material only settles into
    its grain. The step it is let go at is blended, not cut, so four steps still give a smooth ramp."""
    f = pipe.vae_scale_factor * 2
    r = torch.from_numpy(np.clip(ramp, 0.0, 1.0).astype(np.float32))[None, None]
    r = pipe._pack_latents(F.interpolate(r, size=(h // f, w // f), mode="area"))

    def step_end(pipe, i, t, kw):
        latents, ts = kw["latents"], kw["timesteps"]
        n = len(ts)
        free = (r.to(latents.device) * n - (n - 1 - i)).clamp(0.0, 1.0).to(latents.dtype)
        held = kw["image_latents"]
        if i < n - 1:
            held = pipe.scheduler.scale_noise(held, torch.tensor([ts[i + 1]]), kw["noise"])
        return {"latents": free * latents + (1.0 - free) * held}

    return step_end


def around(mask, margin=0.3, most=0.6):
    """Where to work when the mask holds only part of the picture: its extent with a margin of
    context on every side, or None when that would be most of the frame anyway."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.02)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = max(48.0, (x1 - x0) * margin), max(48.0, (y1 - y0) * margin)
    x0, x1 = max(0, int(x0 - mx)), min(w, int(x1 + mx))
    y0, y1 = max(0, int(y0 - my)), min(h, int(y1 + my))
    return None if (x1 - x0) * (y1 - y0) > most * w * h else (x0, y0, x1, y1)


def worksize(box, frame, least=640):
    """The size the surroundings are drawn at: the frame's own density, raised so the long side is
    at least `least` (a small face gets more pixels than it had), never past the frame's long side."""
    cw, ch = box[2] - box[0], box[3] - box[1]
    k = min(max(1.0, least / max(cw, ch)), max(frame) / max(cw, ch))
    return max(16, round(cw * k / 16) * 16), max(16, round(ch * k / 16) * 16)


def resized(a, w, h):
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize((w, h), Image.BILINEAR))


class Editor:
    def __init__(self, device):
        self.device = torch.device(device)
        self.pipe = None
        self.lock = threading.Lock()
        self.read = collections.OrderedDict()  # instruction -> its embeddings, on the card
        self.parked = True

    def load(self):
        if self.pipe is None:
            from diffusers import Flux2KleinInpaintPipeline

            device = self.device

            class Pipe(Flux2KleinInpaintPipeline):
                # diffusers takes the device of the first component by name, the text encoder, which
                # stays in RAM here; the pictures must go where the VAE and the image model run.
                @property
                def _execution_device(self):
                    return device

            with MODELS.loading("editor: FLUX.2 klein", self.unload, self.lock, guess=16.0):
                pipe = Pipe.from_pretrained(KLEIN, torch_dtype=torch.bfloat16)
                pipe.vae.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            pipe._callback_tensor_inputs = HELD
            self.pipe = pipe
            print("  editor up (FLUX.2 klein 4B)")
        MODELS.used("editor: FLUX.2 klein")
        return self.pipe

    def unload(self):
        """The whole pipeline let go (models.py): the next edit loads it again."""
        self.pipe = None
        self.read.clear()
        self.parked = True

    def park(self):
        """The image model off the card, for another pipeline's room; the next edit brings it back."""
        with self.lock:
            if self.pipe is not None and not self.parked:
                self.pipe.transformer.to("cpu")
                self.parked = True
                torch.cuda.empty_cache()

    def words(self, pipe, prompt, length=128):
        """The instruction's embeddings: read once by the text encoder, which visits the card for it
        (with the image model parked first when the card has no room for both). `length` is the
        token budget it is padded to."""
        key = (prompt, int(length))
        if key not in self.read:
            if torch.cuda.mem_get_info(self.device)[0] < 10 * 2**30 and not self.parked:
                pipe.transformer.to("cpu")
                self.parked = True
                torch.cuda.empty_cache()
            pipe.text_encoder.to(self.device)
            try:
                self.read[key] = pipe.encode_prompt(prompt=prompt, device=self.device, max_sequence_length=int(length))[0]
            finally:
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            while len(self.read) > 32:
                self.read.popitem(last=False)
        self.read.move_to_end(key)
        return self.read[key]

    def __call__(self, picture, mask, swatch, prompt="", steps=4, strength=1.0, seed=0, ramp=None, focus=True):
        """`picture` (PIL) redrawn inside `mask` ((h, w) in [0, 1]) after the `swatch` (PIL); `ramp`
        ((h, w) in [0, 1]), when given, how much of each place is redrawn (`graded`). With `focus`,
        only the mask's surroundings are worked on (`around`) and laid back into the picture."""
        box = around(mask) if focus else None
        if box is not None:
            whole, (x0, y0, x1, y1), (ww, hh) = picture, box, worksize(box, picture.size)
            picture = picture.crop(box).resize((ww, hh), Image.LANCZOS)
            mask = resized(mask[y0:y1, x0:x1], ww, hh)
            ramp = None if ramp is None else resized(ramp[y0:y1, x0:x1], ww, hh)
        out = self.draw(picture, mask, swatch, prompt, steps, strength, seed, ramp)
        if box is None:
            return out
        whole = whole.copy()
        whole.paste(out.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
        return whole

    def draw(self, picture, mask, swatch, prompt, steps, strength, seed, ramp):
        w, h = picture.size
        if w * h > 1024 * 1024:
            # The pipeline shrinks anything past a megapixel itself, out of the ramp's sight (a
            # gradient at 1280 failed on it): shrunk here instead, the same way, for both.
            k = math.sqrt(1024 * 1024 / (w * h))
            w, h = int(w * k) // 16 * 16, int(h * k) // 16 * 16
            picture = picture.resize((w, h), Image.LANCZOS)
            mask = resized(mask, w, h)
            ramp = None if ramp is None else resized(ramp, w, h)
        mask_im = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L")
        with self.lock, torch.no_grad():
            pipe = self.load()  # under the lock: a live draft and a job asking at once load it once
            words = prompt or instruction(None)
            embeds = self.words(pipe, words, 128 if len(words.split()) <= 80 else 512)
            if self.parked:
                pipe.transformer.to(self.device)
                self.parked = False
            more = {} if ramp is None else dict(callback_on_step_end=graded(pipe, ramp, w, h),
                                                callback_on_step_end_tensor_inputs=HELD)
            return pipe(prompt_embeds=embeds, image=picture.convert("RGB"), mask_image=mask_im,
                        image_reference=swatch.convert("RGB"), width=w, height=h, strength=float(strength),
                        num_inference_steps=int(steps), guidance_scale=1.0,
                        generator=torch.Generator(self.device).manual_seed(int(seed)), **more).images[0]
