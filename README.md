# latent-lab

Small generative image models, and the tools to decide whether their latent space is worth
navigating. Train on a folder of images, export an `.onnx`, drive it from anything.

These were built to feed [goofi](https://github.com/KairosHive/goofi) from biosignals — an EEG band
power moves a picture — but nothing here imports goofi and the `.onnx` files are ordinary. Torch is
deliberately not in goofi's own venvs: a runtime runs models, it does not train them.

Four ways to build a latent space, cheapest first: a neural cellular automaton that runs inside a
shader, a FastGAN, a LoRA over a distilled diffusion model, and a linear PCA that is the honest
baseline the other three have to beat.

**Half of this repo is measurement, and that is the deliberate part.** Every genuine defect in this
code was caught by one number disagreeing with another, and none by looking at the output. The
preview tiles lied three separate times: a generator reported as working was drawing at 0.17 of its
data's spread; the apparent mode collapse turned out to be un-ramped weight averaging rather than
the model; a space advertising 256 axes had three you could feel. `diversity.py`, `axes.py`,
`keep_best.py` and `report.py` exist because of that.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu126 torch torchvision
uv pip install --python .venv/Scripts/python.exe pillow numpy onnx
```

Drop the `--index-url` for a CPU-only machine, and change `cu126` to whatever CUDA the card wants.

## Where the trained models are

The weights built with this repo live on HuggingFace, in two private repos under the
[Kairos-Hive](https://huggingface.co/Kairos-Hive) org. The code here is the tooling; those are the
results.

**[Kairos-Hive/latent-textures](https://huggingface.co/Kairos-Hive/latent-textures)** — the models

| path | what it is | driven by |
|---|---|---|
| `lora/sd15-lora-yellow-step002000/` | SD 1.5 LoRA, rank 16 | words, through `lora.py`'s trigger |
| `space/texture-ip-yellow.npz` | 16,856 CLIP-ViT-H crop vectors | vectors, through `terms.py` / `knobs.py` |
| `gan/conditional-512.onnx` | FastGAN, 512px, 7 categories | a latent and a category |
| `gan/grown-1024.onnx` | the same, grown by `grow.py` | a latent and a category |
| `gan/conditional-512-yellow3.onnx` | 512px on the wider three-star set | a latent and a category |
| `captions/` | 2,107 per-image descriptions | — |
| `presets.json` | ten named mixes of caption terms | — |

**[Kairos-Hive/latent-textures-sets](https://huggingface.co/Kairos-Hive/latent-textures-sets)** — the
photographs and how they were chosen: `training-set-1600/` (2,133 frames) and `yellow3-1600/`
(4,345), both at 1600px with bursts removed, plus `meta/` holding the ON1 marks the selection was
made from and each set's category split.

**Which to reach for.** The GAN is the cheapest to run and the only one that needs no diffusion
model, so it is what goofi loads. The LoRA is for drawing something new from a description. The
space is for *navigating* — it is continuous, so a signal can drive it, and it needs no training at
all. They compose: the space says which surface, the LoRA says rendered like these photographs.

The sets are 1600px on the shortest side and that is not incidental. `--detail 3` takes a crop of a
third of the shortest side and resizes it to 512, so from 1600px that is a 533px crop at essentially
1:1, and from a 4000px original it is a 1333px crop downscaled 2.6x. **The flag is a fraction of the
frame, not a pixel count**, so the same recipe means a different magnification on a different
source. A GAN learning fine texture draws that loss at once; an embedding or a diffusion LoRA, which
work at 512 anyway, barely notice.

## Paths

The shell scripts here take three values from the environment rather than carrying one machine's
drives: `LAB_PY` (the interpreter), `LAB_MODELS` (where models go), `LAB_SETS` (where image sets
live). Copy `paths.local.sh.example` to `paths.local.sh` to set them once; that file is not tracked.

## style_ca.py → `graphics:NeuralCA`

A neural cellular automaton trained on the style of a folder of images. Twelve channels, one hidden
layer, about 6k numbers. It runs in the shader at full rate and costs nothing to blend between,
which is the point of it.

```bash
python style_ca.py --images ~/pictures/lichen --out lichen.npy
```

Then in goofi: a `Weights` node with `file` set to `lichen.npy`, wired into `NeuralCA`'s `weights`.
Leave `alive` off, `start` at `zero`, `rate` at 1.0 and `fire` at 0.5 — what it was trained at.
`reach` is free: above 1 it coarsens the texture the rule learned.

The style target is the MEAN Gram matrix over the images it reads, so a folder that holds one look
gives a sharp result and a folder that holds twenty gives their average. Point it at a
style-coherent subfolder, and train one file per look — two weight files blend on a wire.

### A set of textures, and the space between them

One model per look, all of them children of one parent, and the blend is a wire:

```bash
python cluster.py --images ~/pictures/ice --out ~/pictures/ice-groups --groups 3
python style_ca.py --images ~/pictures/ice --out ice.npy --detail 3 --crops 6 --hidden 192 --steps 5000
for g in 0 1 2; do
  python style_ca.py --images ~/pictures/ice-groups/group$g --out ice$g.npy --parent ice.npy --steps 1200
done
```

In goofi: a `Weights` per file, a `Math` per file scaling it by that file's share, and one `Operation`
in `add` mode folding them into `NeuralCA`'s `weights`. Every share is a param expression, so N files
are an N-1 dimensional space a signal moves through.

**`--parent` is what makes that work.** Two rules trained from nothing put their hidden units in
their own arbitrary order, so averaging them averages unrelated things and the midpoint is mush.
Children of one parent keep the order and stay in one basin — the same reason a model soup blends
and two cold starts do not. Keep the fine-tunes short for the same reason: the further a child
travels, the rougher the road back to its siblings.

**`--detail` is the setting to get right, and its default is wrong for a big photograph.** At 1 a
frame is resized whole into the training square, so a 6000-pixel macro becomes a 128-pixel thumbnail
of a scene and the grain the style lives in is the first thing thrown away. At 3 a crop sees a third
of the frame; at 12 it sees a twelfth and starts finding the lens's own blur instead of the subject.
Raise `--crops` alongside it so the average is still taken over enough of the set. Dump what the
trainer will actually see before spending an hour on it:

```python
python -c "import numpy as np, style_ca as s; from PIL import Image; \
t = s.images('PATH', 128, 6, 3, 1); \
Image.fromarray(np.concatenate([(q.permute(1,2,0).numpy()*255).astype('uint8') for q in t],1)).save('crops.png')"
```

It mirrors goofi's `node-bundles/graphics/NeuralCA.wgsl` line for line, and the shader's half of that is
pinned by `a_learned_automaton_grows_from_its_seed_instead_of_flooding_the_grid`. Change one and
change the other, or a texture trains well and runs wrong.

## fastgan.py → `ml:Decoder` or `ml:Onnx`

A generator that makes whole images rather than texture: FastGAN, which is the recipe for a few
thousand images on one GPU in hours. About 30M parameters at 256 square.

```bash
python fastgan.py --images ~/pictures/plates --out plates.onnx
```

Then in goofi: a `Decoder` node with `file` set to `plates.onnx`. (`Onnx` runs it too, and any other
model besides — `Decoder` is the one that also steers the latent for you.) Wire anything into `drive` — band
powers, a Kuramoto order parameter, a Reduce of anything — and each wired number takes a direction
of its own through the latent space. `spread` is the truncation knob: below 1 is safer and duller.
`smooth` is what turns a jumpy signal into a walk.

### One model per category, or one model conditioned on it

Two ways to build a space over the categories, and neither is a separate GAN per folder: latent
spaces of independently trained models have nothing to do with one another, so there is no path
between them, only a crossfade.

```bash
# a child per category, every one from the SAME parent
python fastgan.py --images set-by-category/cat1 --out cat1.onnx --init parent.pt --steps 5500
python blend.py --models cat1.pt cat4.pt --weights 0.5 0.5 --out between.onnx

# or one model that takes the category as an input a patch can blend
python fastgan.py --images set-by-category --classes --out conditional.onnx --init parent.pt
```

`--init` takes another run's weights with a fresh optimizer and the step count back at zero. Children
of one parent stay in its basin and their weights still average, which is what `blend.py` relies on
and the same reason a model soup works and two cold starts do not. Keep the fine-tunes short: the
further a child travels, the rougher the road back to its siblings.

`--classes` takes each image's category from the folder that holds it and conditions one model on it.
The discriminator is conditioned too, by projection — a critic that cannot tell which category it is
being shown cannot punish the wrong one, and the generator will simply ignore the label. Categories
are drawn evenly whatever their sizes, or the largest is trained on three times as hard as the
smallest. The export then takes `z` AND `category`: per-category means that move with the blend, and
axes shared across all of them, so a knob keeps its direction while the material changes under it.
Both tags start near silent, so a run begun from an unconditional parent draws what the parent drew
until the category earns its say. In goofi it is the `Onnx` node, with a slow signal on `category`
and fast ones on `z`.

It exports every `--snap` steps, so the `.onnx` is loadable while training continues, and writes a
`.pt` beside it that `--resume` carries on from. Watch the `.png`, and run `keep_best.py` alongside:
**a GAN wanders, and the export is overwritten every snapshot.** One 120,000-step run here peaked at
step 42,500 and spent the next 77,500 steps getting worse. Without something holding on to the peak,
it is gone by morning.

Unlike `style_ca.py` this has no counterpart inside goofi to drift from: `Decoder` runs any `.onnx`
that takes one vector and answers one image, so a pretrained generator from anywhere else works
just as well.

**What it exports is not the raw generator.** A mapping network (StyleGAN's) turns `z` into a `w`
the synthesis network can use directly, path-length regularization (StyleGAN2's) holds
`|d image / d w|` constant, and the export bakes in a PCA of `w`. So the `.onnx` takes WHITENED
coefficients of the model's own principal axes, ordered by variance — the same contract
`pixel_pca.py` writes, which is why `Decoder`'s `spread`, `seed` and `axes: direct` mean the same
thing for both.

Path length is the one to keep. It is not about fidelity: it makes the latent a CONTROL SURFACE, so
the same change in a band power moves the image by the same amount wherever it stands. Without it
one region of the space is hair-trigger and another is dead, and nothing at runtime can compensate.
`--path 0` turns it off.

## pixel_pca.py → `ml:Decoder`

The linear space, for nothing: principal components of the pixels, exported as one matrix product.
Exactly invertible, exactly smooth, instant. Worth knowing its ceiling — after roughly fifteen
components the principal components of natural images become a Fourier basis, so you get about
fifteen real axes and then sinusoids. It is the honest baseline a GAN has to beat, and a usable
soft-colour-field instrument in its own right.

## categories.py — what is this set actually made of?

```bash
python categories.py --images ~/pictures/set --out ~/pictures/set-categories
python by_category.py --images ~/pictures/set --csv ~/pictures/set-categories/categories.csv --out ~/pictures/set-by-category
```

Embeds every image twice — CLIP, which learnt what a surface is CALLED, and DINOv2, which learnt
what one LOOKS like and never saw a word — then tries every count of groups and scores each on three
things: are the groups apart (silhouette), do runs from different starts find the same groups
(stability), and do the two models find the SAME groups (agreement). The third is the one that
matters. A split two representations with nothing in common both reproduce is in the pictures rather
than in either model.

Expect low silhouettes on photographs, around 0.15, and do not read that as failure: a set of
textures is a continuum, and these are regions of it rather than islands. Agreement and stability
still tell you where to cut it. On 2,133 of one archive the answer was seven — water, rock, waves,
encrusted rock, peeling paint, bark, abstract light — holding 155 to 482 images each.

Check two things before trusting a category. That it is a MATERIAL and not one afternoon's shooting:
count the distinct shoots behind it, because a category that is one session will be memorised rather
than learnt. And that DINOv2 agreed: a category CLIP names confidently but DINOv2 does not see is a
word, not a look.

`by_category.py` turns the CSV into one folder per category as hardlinks — no second copy of the
set — which is what `--init`, `--classes` and every measuring tool here read.

## cluster.py

Group a folder by look before training anything. `--groups 8` and the contact sheets tell you
whether a set is one domain or several.

## Reading a model, before believing it

Four tools, and every one of them exists because a picture lied first. A generator can look like it
is working in four preview tiles and be drawing the same thing four times; it can advertise 256 axes
and have three you can feel. Measure, then look.

### diversity.py — is it repeating itself?

```bash
python diversity.py --model textures.onnx --images ~/pictures/set
```

Samples the model and the data and reports the same statistic for both: how far apart two draws are,
how much of that is tone rather than structure, a Laplacian sharpness ratio, and a spectral peak that
catches a repeating grid nothing else here sees. A model at 1.0 is as varied as its data; below 0.5
it is repeating itself. This caught a run that had been reported as a success from its previews and
was in fact drawing at 0.17 of the data's spread.

`--reals` is fixed at 96 and deliberately independent of `--draws`. Resampled alongside the model it
moves every ratio on its own: the same model measured 2.68, 6.23 and 4.91 for tiling on three runs
that differed only in which photographs the reference happened to draw. A trend read off that is a
trend in the instrument.

**And read `apart` narrowly.** It is a pixel distance, so four grey speckles at different brightness
score as far apart as a canyon and a rainbow mesh. It says a model is not collapsed; it does not say
the model spans the data. For that, compare saturation, colour spread and brightness spread against
the real set directly — `keep_best.py` does, and `report.py` prints the table.

### axes.py — how many knobs does it really have?

```bash
python axes.py --model textures.onnx
```

Two different questions with different answers: how much of the latent's variance sits on each axis,
and how far the PICTURE moves when that axis is turned. The second is what a hand feels. A healthy
space decays gracefully — the linear model runs 1.00, 0.35, 0.30, 0.28, 0.23 over its first five. A
degenerate one falls off a cliff after three, which is a generator that has learned three modes.

`--bases` averages the effect over several points in the space, and it is not optional. Measured at
the mean alone the count swung between 6 and 24 across consecutive snapshots of one run — one point
of a 256-dimensional space answering for all of it. Four points is stable to the reading.

Two counts are printed and they answer different questions. "A quarter as much as axis 0" is a harsh
bar when axis 0 is twice the next one; the absolute `--floor` says how many axes move the picture
enough to see side by side. The finished 512 model reads 28 and 51.

### keep_best.py — hold on to the peak, because the trainer will not

```bash
python keep_best.py --watch textures-raw.onnx --images ~/pictures/set
```

Watches the export, scores every snapshot against the real set on saturation, colour spread,
brightness spread and sharpness, rejects anything whose repeat stands more than `--ceiling` times
above the data's own, and keeps a `-best` copy of the winner.

Run it whenever a GAN runs. Training does not end at its best point — one run here scored 0.080 at
step 42,500 and never came within eight times that again over the following 77,500 steps, while the
trainer overwrote its export forty times. `--beat` gives it a score to improve on, and `--tag` a name
of its own, so a second pass cannot clobber what a first one found.

**Score everything you care about.** The first version of this weighed colour alone, and duly picked
a snapshot that matched the palette almost exactly and carried less than half the data's detail.
Sharpness is in the score now. A selector is only as good as the worst thing it is blind to.

### sheet.py — eighteen draws over the real thing

```bash
python sheet.py --model textures.onnx --images ~/pictures/set --out sheet.png
python sheet.py --model conditional.onnx --out each.png --each
```

Four preview tiles is how three faults here went unnoticed. Eighteen draws above eighteen real crops
catches all of them at a glance, and `--each` gives a conditional model one row per category, which
is the only way to see whether the category input does anything at all.

### morph.py — is the crossing between two categories even?

```bash
python morph.py --model conditional.onnx --all --out morph.png
```

Walks the category vector from one to another with the latent held still, and reports the spread of
the distance between consecutive frames. A morph that holds still for half its length and then jumps
is not a control surface, whatever its two ends look like.

### play.py — a browser, some knobs, the live model

```bash
python play.py --model textures.onnx            # then open the printed localhost URL
python play.py --model pca100.onnx --port 8010
```

Runs the model per request on a small local server: an XY pad over any two axes, a slider per axis, a
damped random walk through all of them, and truncation. Pin two points as A and B and the morph
slider runs the line between them, which is the move a signal actually makes. Nothing precomputed,
nothing leaves the machine. 60–90 ms a frame on CPU and ~130 ms at 512 on CUDA, which is fine to
drag. Two of them on different ports is the fastest way to feel what a step-spread number means.

The morph slerps the direction and lerps the magnitude rather than blending straight. A plain lerp
between two latents passes close to the origin, and the origin is the model's mean, so the middle of
every morph washes out to mush.

### report.py — all of it, as one page

```bash
python report.py --models textures.onnx pca100.onnx --out report.html
```

Samples, a sweep per principal axis, a walk, and the step-spread figure: the coefficient of variation
of the distance between consecutive frames along an evenly sampled path. Flat means playable. The
linear models sit near 0.25, which is the floor the spherical parameterisation imposes rather than
anything about the model.

## dedupe.py

```bash
python dedupe.py --images ~/pictures/set            # report
python dedupe.py --images ~/pictures/set --apply    # move them aside
```

Photographs come in bursts, and a GAN weights its loss by how often it sees a thing. Groups anything
closer than `--near` in ResNet-50 feature space, keeps the sharpest frame of each group and moves the
rest to a folder BESIDE the set. Beside, not inside: every trainer here walks its root with `rglob`,
so a subfolder of rejects is one the next run silently reads back in — which is exactly what happened
the first time, and the only sign was an image count in a log header.

## lora.py → `Diffusion` (in `extra-nodes/diffusion/`)

The heavy option: a distilled diffusion model with a LoRA of your images on top. Best style
fidelity of the three, and the only one that takes a prompt.

```bash
python lora.py --images ~/pictures/plates --out plates-lora --trigger "in plt style"
cargo run -- --extra-nodes extra-nodes/diffusion
```

Set `Diffusion`'s `lora` to the folder it wrote and put the trigger in `text`. Wire `image` and it
restyles whatever the patch draws; leave it unwired and `walk` drifts the latent so the picture
moves rather than flickering between strangers.

Measured on an RTX 3090 with `sd-turbo`, fp16, no compilation: **4–6 fps** at 384–512 square with 1
to 2 steps. That is a pace to compose to, not a frame rate — `NeuralCA` runs at the viewer's own
rate and `Decoder` at a hundred times this. Reach for it when the style matters more than the
motion.

`--rank` is how much the LoRA can hold: 16 is a style, higher starts memorising the set. The loss
printed is noisy by construction — every step draws a random timestep — so read its trend over
hundreds of steps and judge the result by looking at it.

### Training on a set of big photographs, and on its categories

```bash
python lora.py --images ~/pictures/set-by-category --classes --detail 3 \
    --name cat0=water cat1=rock cat2=waves --model stabilityai/sdxl-turbo --out set-lora
```

**`--detail` matters here as much as it does for `style_ca.py`.** At 1 a 6000-pixel macro is squeezed
whole into 512 pixels, and the LoRA learns the palette of the set without its surface. At 3 a crop
sees a third of the frame. Crops are opened per step rather than held, because the set held at
`size * detail` is tens of gigabytes; `--workers` decode them while the GPU trains.

`--classes` reads each image's category from its folder, which is what `by_category.py` writes,
draws the categories evenly whatever their sizes, and captions each crop with `--caption` — by
default `a close-up photograph of {name}, {trigger}`, where `--name` says what a folder is called. A
real word for the material borrows what the base already knows about water or bark; the trigger ties
it to this set. The prompt then picks the material, and a prompt can name two.

It ends by drawing a sheet, one row per caption, into the output folder beside `latent-lab.json`,
which records what was trained and how to ask for it. A row that looks like every other row is a
caption the LoRA did not learn to mean anything.

SD 1.x/2.x and SDXL bases both train. `sdxl-turbo` is worth the extra memory: it is the base the
pretrained IP-Adapters and most ControlNets were made for, which `ip_space.py` needs.

## ip_space.py — the set itself as the space, through an IP-Adapter

```bash
python ip_space.py --images ~/pictures/set-by-category --detail 3 --out set-ip.npz
python ip_space.py --space set-ip.npz --sheet set-ip.png --lora set-lora
```

An IP-Adapter lets an image steer a diffusion model the way a prompt does: an image encoder turns a
picture into one vector, and the adapter feeds that vector into every cross-attention layer beside
the text. Nothing about that vector has to come from one picture. This embeds crops of every image,
and takes the cloud of vectors as the space: a mean per category, and principal axes across them.
Every point is a prompt, and moving the point is a walk. **Nothing is trained**: the base and the
adapter are pretrained, and the photographs are only ever the adapter's input.

The build prints what the space is worth before anything is drawn: the variance on each axis, and
how far apart the category means sit against the spread inside a category, pair by pair. Read the
smallest pair, as with `separation.py` — a pair near zero is one this encoder cannot tell apart, and
steering between them will not look like steering.

The sheet draws each category's mean over several seeds, a morph between neighbouring categories
with the seed held, and a sweep along the first principal axes. Every point is rescaled to the
length a real crop's vector has: a mean is shorter than what it averages, and the adapter reads a
short vector as a faint one. Morphs go along the sphere for the same reason `play.py`'s do.

The vectors must come from the encoder the adapter was trained with, which is why this embeds again
rather than reading `categories.py`'s cache: that is another CLIP, and its numbers are another space.
The space remembers its `--preset`, so a sheet is drawn with the adapter it was built for.
`--style-only` (SDXL) steers only the block that carries style, so a reference's layout does not
come with its surface — for textures that seldom matters, for anything with a composition it does.

## caption.py — a word per image, not a word per folder

```bash
python caption.py --images ~/pictures/set --detail 3 --sample 16 --contact check.png
python caption.py --images ~/pictures/set --detail 3 --write
```

A category word is a blunt caption. Every rock in a set trained on `rock` teaches the average of all
of them, and the LoRA that came out drew lichen as beaded confetti — worse than the base model. This
describes each image instead: CLIP scores three vocabularies separately, what the material IS, how
its surface behaves and what light it sits in, and the colours are named from the pixels by
clustering in CIELAB against a palette of 44.

Over 2,133 images that gives 1,931 distinct captions from 59 materials and 106 surface terms, and
the confetti goes away. `--sample` previews without writing; `--contact` draws the crops with their
captions beneath, which is the only way to see that a material is wrong.

Two things the colours needed. The crops really are darker and less saturated than a thumbnail
suggests, so a caption that reads too grey is usually right — check a contact sheet before tuning.
And the palette needed the muted middle: saturated hues around a grey axis sent a tan stream to
"ash grey" and beige paper to "chocolate" until `taupe`, `pewter`, `driftwood` and the rest existed.
The dominant colour stays weighted by chroma, because naming the largest cluster instead sent golden
water to "charcoal" and foliage to "olive drab".

## compare.py — choose a checkpoint by holding the seed still

```bash
python compare.py --lora "2000=run/step-002000@1.4" "4000=run/step-004000@1.4" --out compare.png
```

Picking a LoRA from its own sheet does not work: every sheet is drawn from different noise, so the
checkpoint that lands a good seed looks like the better model. This holds the prompt and the seed
and changes only the LoRA, which is the only arrangement where what differs on screen is the thing
being measured. A trailing `@1.4` sets how hard each one is applied.

**A LoRA's training loss does not say when to stop.** One run held a normal 0.02 to 0.24 band
through 3,000 steps while the pictures blurred to mush; the peak was step 1,000. Noise-prediction
error at a small batch is dominated by which timestep was drawn, not by whether the picture holds
together, so always save numbered checkpoints and choose between them here.

Steps and strength trade as a product, so neither can be read with the other held fixed. Blur begins
around `steps x strength ~ 4800`: 2,000 at 1.6 and 3,000 at 1.0 are indistinguishable, and 4,000 at
1.6 is mush. Prefer the earlier checkpoint at a higher strength — it has somewhere left to travel,
which matters if the strength is going to be driven by a signal.

## terms.py — knobs built from what the captions say

```bash
python terms.py --space set-ip.npz --list
python terms.py --space set-ip.npz --image photo.jpg --term encrusted glassy --out sheet.png
python terms.py --space set-ip.npz --image photo.jpg --preset "rock structure" --push 8 --out sheet.png
```

Prompting is a discrete interface: it works, but only if you reach for one of the material words the
captions actually used — asking for "an old painted wall" draws a room with a lamp in it, because
the trained phrase is "a close-up photograph of peeling paint". A knob needs no vocabulary, moves
continuously, and can be driven by something that is not a keyboard.

Categories make poor knobs here. The clustering says they are not really there — silhouette stays
near 0.16 at every count from 3 to 16 — and the seven means then sit 0.43 to 0.66 apart in units of
the spread *within* a category, so every knob drew the same grey. A caption term names a set of
images directly, there are about 150 of them carried by enough images to use, they overlap on
purpose, and they are words already in mind before anyone starts.

Two numbers decide whether a knob points anywhere:

- `--push` exaggerates a mean away from the centre of the space, keeping what distinguishes a term
  and dropping what every texture shares. A single term wants 3. **A mix wants 8**, because
  averaging terms lands between them and between is nearer the centre: at push 3 the ten presets sit
  16.8 apart where single terms sit 21.5, and push 8 brings them to 25.8.
- `--scale` is the adapter's own strength, and the knob a hand or a signal actually turns. A weight
  in a mix sets proportion and never intensity — the rescale divides any shared factor back out, so
  `cat1=0.4` alone and `cat1=1` alone are the same place.

`presets.json` holds named mixes, which is how a high-level idea gets an entry point without
imposing a boundary: a preset names a point you then slide away from. A term no caption carries is
dropped with a warning, so a preset written against one set stays mostly right on the next.

## knobs.py — the same, by category rather than by term

```bash
python knobs.py --space set-ip.npz --image photo.jpg --mix cat1=1 --push 3 --scale 0.9 --out turned.png
```

Older and blunter than `terms.py`, kept for a space whose folders mean something. `--mix` chooses
which categories and in what proportion, `--scale` how much, `--strength` how much of the source is
redrawn at all, `--push` how far a category is exaggerated. Prefer `terms.py` unless the categories
are real.

## restyle.py — carry a picture that is not in the set into its look

```bash
python restyle.py --image photo.jpg --lora "mine=run/step-002000@1.6" --bare --out restyled.png
```

The LoRA and the space both answer "draw me something new". This answers a different question: take
a picture that already exists and carry it toward the set without losing what it is. That is img2img,
and it turns on one number — denoising strength, how much of the original is thrown away before
redrawing.

How far a picture can travel depends on whether it has surface. A portrait at 0.3 becomes carved
weathered stone and is still recognisably the same person; by 0.6 it is a lichen-covered relief. A
building survives to 0.75 because its geometry is hard, while a landscape of soft gradients has
dissolved into rock by 0.6, and a room with depth and furniture loses itself first. The number is
not guessable per picture, so the sweep is drawn and the eye picks.

Given several LoRAs it draws them as columns, with `--bare` adding the prompt alone. That column
matters more than expected: at a gentle strength every LoRA looks like no LoRA at all, and the
differences only appear when pushed.

## grow.py — twice the resolution, keeping what is already known

```bash
python grow.py --from run/textures.pt --out grown1024.pt
python fastgan.py --images set --classes --size 1024 --init grown1024.pt --fade 20000
```

A 1024 run from nothing collapsed before its first snapshot and 94,000 more steps never brought it
back. Carrying the generator alone does not fix it either — a trained generator against a fresh
critic is its own way to collapse — so both networks come across and only what the new resolution
adds starts from zero. It refuses to write unless the grown generator reproduces the old one
upsampled, which is the proof that no layer landed in the wrong place.

## separable.py — is a category learnable at all?

```bash
python separable.py --images ~/pictures/set-by-category
```

Crops the way the trainer crops, embeds with DINOv2, and fits a linear probe per category pair.
Run it before blaming a model for merging two categories: if a probe cannot tell them apart on the
same crops, nothing downstream will either, and the fix is the set rather than the training.
