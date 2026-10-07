# Texture diffusion

Draw new textures in the archive's own look, steerable later from biosignals. Two instruments,
built to compose on one base rather than compete:

- **LoRA** — the look, baked into the weights. Reached by words.
- **IP space** — the archive's own pictures as vectors, with principal axes to move along.
  Training-free, and continuous, which is what a signal needs to drive.

Both must sit on SD 1.5. The SDXL space built earlier cannot be used with an SD 1.5 LoRA: their
adapters read different encoders (CLIP-ViT-bigG against CLIP-ViT-H) and the vectors are not the
same kind of thing.

## Settled

- **Per-image captions beat category words.** One word per folder taught the LoRA the average of
  every rock in the set; it drew lichen as beaded confetti, worse than the base model. `caption.py`
  describes material, surface, condition and colour separately: 1,931 distinct captions over 2,133
  images. The confetti is gone.
- **Checkpoints are chosen from drawn sheets, not from loss.** Loss held a normal band through
  3,000 steps while the pictures blurred to mush. `compare.py` holds prompt and seed and changes
  only the LoRA.
- **Keeper: `sd15-lora-gentle/step-002000`, applied at 1.0 to 1.8.** Training at lr 3e-5 instead of
  1e-4 moved the point where the LoRA blurs from step 1,500 out past 4,000, and it carries more of
  the archive: dense lichen crust where the 1e-4 run drew chalk.
- **Steps and strength trade as a product.** A LoRA trained further behaves like a less-trained one
  applied harder, so neither can be read with the other held fixed. Blur begins around
  steps x strength ~ 4,800: 2,000@1.6 and 3,000@1.0 are indistinguishable, 3,000@1.6 softens on
  copper, 4,000@1.6 is mush on everything. Keep the product under ~4,000 to be safe.
  step-002000 is the keeper because it has the widest usable range, 1.0 to 1.8, which is what a
  signal needs if strength is going to be a live dial.
- **Distilled bases cannot be fine-tuned this way.** sd-turbo and sdxl-turbo are ADD-trained; an
  ordinary DDPM objective destroys them, which cost a 14,000-step run. SD 1.5 is the base.

- **The space is continuous**, which is the property that matters for driving it from a signal:
  rock dissolves into water across six morph steps with no jumps.
- **`texture-ip-sd15.npz`** holds 17,064 vectors of 1,024, built from 8 crops of each image.

## Open question, running

The space draws far softer than the photographs beside it, and the LoRA under it at 1.6 softens it
further rather than sharpening it, so neither is the cause. The `sd15` preset carries an LCM LoRA at
4 steps and guidance 1.0 — a realtime setting — and a style LoRA then stacks on top of LCM's own.
`sharpen.sh` turns the speed-up off and denoises in 30 steps. If that sharpens the pictures, the
softness was never in the vectors, and quality against speed becomes a dial to set deliberately
rather than one inherited from a preset.

## Next

1. Read the sharpened sheets. Decide what the space is actually worth.
2. If the preset was the cause, find the cheapest settings that still hold detail: that number is
   what realtime has to afford later.
3. Compose the space with `sd15-lora-gentle/step-002000` at a product under 4,000.

## Open, needs a decision

- **Trigger word.** `in goofi style` tokenizes to `goo` + `fi`, so it starts with a mild pull
  toward slime that training has to overwrite. Not a defect. Swapping costs a string replace over
  the captions and one retrain; no re-captioning.
- **`restyle.py`** — img2img, to put an arbitrary photograph into the archive's look. This is the
  answer to "will this transform any image", and nothing tests it yet.
- **Realtime and goofi integration.** Deliberately not started.
