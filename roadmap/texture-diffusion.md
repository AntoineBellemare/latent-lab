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
- **Keeper: `sd15-lora-captioned/step-001000`**, applied at strength 1.0 to 1.8. It does not break
  at 1.8, and higher strength means more crust and mineral speckle.
- **Distilled bases cannot be fine-tuned this way.** sd-turbo and sdxl-turbo are ADD-trained; an
  ordinary DDPM objective destroys them, which cost a 14,000-step run. SD 1.5 is the base.

## Running

- `sd15-lora-gentle` — the same captions at lr 3e-5 for 6,000 steps. 1e-4 overcooked by step 1,500,
  so a third of it should absorb more of the archive before blurring.

## Next

1. Compare the gentle checkpoints against step-001000 on held seeds. Keep whichever holds detail
   at the higher strengths.
2. Build the IP space on the SD 1.5 preset, `--crops 8` over the set.
3. Draw the space alone, with real crops beside each row, to see what it holds.
4. Draw the space under the LoRA, at two style strengths. This is the pairing that matters: the
   LoRA supplies the look, the space supplies a place to move.

## Open, needs a decision

- **Trigger word.** `in goofi style` tokenizes to `goo` + `fi`, so it starts with a mild pull
  toward slime that training has to overwrite. Not a defect. Swapping costs a string replace over
  the captions and one retrain; no re-captioning.
- **`restyle.py`** — img2img, to put an arbitrary photograph into the archive's look. This is the
  answer to "will this transform any image", and nothing tests it yet.
- **Realtime and goofi integration.** Deliberately not started.
