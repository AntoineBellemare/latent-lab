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

- **The space was never soft; the preset was.** Drawn through `sd15`, which carries an LCM LoRA at
  4 steps and guidance 1.0, every category came out blurred. At 30 steps with the speed-up off the
  same vectors draw rock faces with mineral striations and ochre veins. Quality against speed is a
  dial to set deliberately, not one to inherit.
- **The two instruments do not need each other.** Composed under the space at 30 steps the LoRA
  softens it rather than sharpening it: the space is built from the photographs, so it already
  carries the look. The LoRA earns its place where there is no image vector — prompting from words.
- **Any picture can be carried into the archive's look.** `restyle.py` runs img2img and sweeps the
  one number that matters. A portrait at strength 0.3 becomes carved weathered stone and is still
  recognisably her; a building survives to 0.75 while a landscape has dissolved into rock by 0.6.
  Structure decides how far a picture can travel, so the sweep is drawn rather than guessed.

- **Knobs, not words.** Prompting works but is a discrete interface: it needs the right one of 59
  material words, and "an old painted wall" is not one of them — asking for it draws a room with a
  lamp in it. `knobs.py` turns a source picture with a weight per category instead, which needs no
  vocabulary and moves continuously.
- **Four controls, and they are not interchangeable.** `--mix` chooses WHICH textures and in what
  proportion; `--scale` is the intensity knob; `--strength` is how much of the source is redrawn;
  `--push` is how far a category is exaggerated from the centre of the space.
- **Raw category means make knobs that turn without pointing anywhere.** The means sit 0.43 to 0.66
  apart in units of the spread within a category — two centres are half as far from each other as a
  crop is from its own — so every knob drew the same grey crust. `--push 3` keeps the direction
  that distinguishes a category and drops what they all share. Past 3 it saturates.
- **A weight is a direction, not a volume.** The blend is rescaled to the length real crops have,
  which divides out any shared factor, so `cat1=0.4` alone equals `cat1=1` alone. Only ratios in a
  mix matter. Intensity is `--scale`.

- **Curating the set changed the space by nothing measurable.** Four-star Yellow spans seventeen
  folders where the old set was one trip, yet the two spaces are statistically the same: 182 axes
  to 90% of the variance against 185, spreads of 12.80 against 13.12, centres 0.22 of a spread
  apart, and a crop from one set as close to the other set as to its own (median 10.2 either way).
  The caption vocabularies match term for term. Every preset drew the same picture through both.
- **What that says about the work.** The embedding reads surface, not provenance, and these
  surfaces are the same wherever they were photographed. The consistency is in the eye, not in the
  trip — which is the thing a style model was supposed to capture, so finding it already there is
  the right answer arriving as a null result.
- **Curate for the foundation, not for the space.** The curated set is cleaner, traceable to marks
  the photographer already made, and free of the carved figure and the mountain sky. It will not
  change what the SPACE draws.
- **It does change what the LoRA draws.** Trained on the same recipe, the curated LoRA is more
  saturated and carries more of the lichen's colour than the one trained on the trip: gold and
  green and pink where the old draws chalk, and granular verdigris where the old is grainy. Keeper:
  `sd15-lora-yellow/step-002000`.
- **Why one moved and the other did not.** A term knob is the mean of hundreds of crops, and means
  converge however the sample was drawn — which is what the geometry already said. Training takes
  gradient steps on individual photographs, so which photograph matters, and four-star frames are
  the better photographs.
- **The peak moved earlier.** The curated LoRA is mush by step 4,000 where the trip's was still
  usable there and only broke at 6,000. A more consistent set regularises less. Numbered saves are
  not optional.

- **The original set was already Yellow at four stars.** `pick_on1.py`, run 2026-09-08, filtered
  the archive on exactly `label == "Yellow" and rating >= 4` and matched 2,560 files across
  eighteen trips, which `extract.py` then wrote out as 2,335 JPEGs. Tracing it by DSC basename
  said "one trip" and that was an artefact: basenames collide across trips and the first match
  wins. The old set and the curated set are very nearly the same photographs.
- **`extract.py` also resized every frame to 1600px on its shortest side, and that is the whole
  GAN result.** `--detail 3` takes a third of the shortest side and resizes it to 512: from 1600px
  that is a 533px crop at essentially 1:1, from a 4000px original it is a 1333px crop downscaled
  2.6x. The rebuild hardlinked originals, so the GAN was shown crops with 2.6x less detail and
  drew exactly that. Nothing to do with diversity.
- **`--detail` is a fraction of the frame, not a pixel count.** The same flag means a different
  magnification on a different source resolution, so two sets are not the same experiment until
  `min(im.size)` matches on both. A GAN learning fine texture pays for it at once; the space and
  the LoRA, which work at 512 regardless, barely noticed.

- **Resized to 1600 the flatness goes away.** Yellow at three stars and up, written out the way
  the original was, draws pebbled stone, a turquoise shoreline with foam, striated water and
  orange crust — as sharp as the old GAN and nothing like the haze the full-resolution run made.
  The resolution was the fault, and the fix proves it.
- **Three stars buys data and costs conditioning.** 4,345 frames after dedupe against 2,133, but
  the categories come out weaker: DINOv2 agrees with the split only 58 to 69 per cent on four of
  the seven, and the weakest category pair falls to 0.039 against 0.143 for four stars. More
  marginal frames blur the boundaries the conditioning is supposed to find.
- **So the old GAN was already the four-star Yellow GAN.** The criterion asked for this week is
  the criterion it was built on in September. For a conditional model four stars remains the bar;
  three stars is the set to reach for only when the categories do not matter.

## Next

1. Find the cheapest denoising that still holds detail. That number is what realtime has to afford.
2. Decide the trigger word, then recaption and retrain once if it changes.
3. Carry the keeper, the space and the knobs into goofi: seven weights, a scale and a strength is a
   small enough surface for a signal to drive.

## Open, needs a decision

- **Trigger word.** `in goofi style` tokenizes to `goo` + `fi`, so it starts with a mild pull
  toward slime that training has to overwrite. Not a defect. Swapping costs a string replace over
  the captions and one retrain; no re-captioning.
- **`restyle.py`** — img2img, to put an arbitrary photograph into the archive's look. This is the
  answer to "will this transform any image", and nothing tests it yet.
- **Realtime and goofi integration.** Deliberately not started.
