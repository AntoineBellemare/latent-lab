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

### Or conditioned on the lab's space, so any mix of materials is drawn live

```bash
python embed_crops.py --images set-by-category --out crops --size 512 --detail 3 --crops 16
python fastgan.py --images set --look crops --size 512 --init parent.pt --tag-lr 10 --steps 20000 --out looks.onnx
python fastgan.py --images set --look crops --size 512 --tag-lr 1 --steps 40000 --resume --out looks.onnx
python follows.py --model looks.onnx --crops crops --sheet follows.png
python bank.py --model looks.onnx --space texture-ip-sd15.npz --out looks-presets.onnx
```

Seven categories are seven places. The space `ip_space.py` builds — the one the lab mixes materials
in — has a place for every crop, preset and photograph, and any weighted sum of them is one more.
The diffusion model draws such a mix in a quarter of a second at best; the GAN draws in tens of
milliseconds but could only blend its categories. `--look` gives the GAN the space instead.

`embed_crops.py` cuts each photograph into crops at the GAN's scale and places every crop with the
adapter's own encoder, so each training square comes with the vector that says where it sits.
`--look` trains on those pairs, conditions the generator on the vector the way `--classes`
conditions it on a category, and conditions the critic by projection for the same reason. Most of
the time (`--whole`) a crop is labelled with its PHOTOGRAPH's place rather than its own, because
every material the lab makes is a mean like that. The export takes `z` and `look`, a 1024-wide
place at any length: only its direction is read, so a mix is simply a weighted sum.

**The place never meets `z` before `w`, and four runs show why.** A category is added to `z`
before the mapping, and the mapping normalises what it is given, so in that sum the larger term
speaks. Added the same way, the place grew to sixty times the size of `z` within 6,000 steps and
every draw of a place came out the same picture: a crop's own vector all but names one picture, so
a generator loses nothing the critic can see by ignoring `z`. (The category models had settled at a
fifth to a third of `z`'s size on their own.) Held at `z`'s size, `z` lived and the place stopped
steering: it reached the network through a noise exactly as loud as itself. Added after the
mapping's first layer, `z` drowned again, because the place's path learns at ten times the rate and
`z`'s, through the mapping, at a hundredth. So `z` goes through the mapping alone, as in the model
the run starts from, and the place moves the result by a fixed reach in `w` (`--look-reach`, in
units of `w`'s own spread; the three-star model's categories sit about half a spread from its
centre). That is the shape the category export already has: a mean per material in `w`, `z` along
shared axes.

**And the critic reads the place back.** With the structure right, the fourth run still drew
textures that matched nothing after 6,000 steps. Projection teaches the critic which place it is
shown only through the fakes that miss it. A head that reads the place off the picture
(`--ask`) is trained on real crops by plain supervision, and the generator is asked to be read
right — AC-GAN's auxiliary, for a place. A floor (`--diverse`) asks two draws of one place to differ
by at least half what two real crops do, against the auxiliary's pull toward one easy picture.

What `gan-looks.sh` made of the three-star set, after 40,000 steps from the three-star parent: asked
for one of 64 photographs' places it lands nearest the right one 16% of the time (real crops 97%,
chance 2%), with a likeness of 0.16 (real crops 0.79); among the 16 presets, 29% (chance 6%). The
latent moves the picture about one and a half times as far as the material does. It follows a mix's
palette and broad character, more loosely than the diffusion model drawing the same vector; white
materials — frost, foam — come out washed out. It draws a 512 frame in 3.2 ms on a 3090, the same as
the category model, and in 90 ms on a CPU. Phase two (tag-lr 1) is where the presets came apart:
19–25% through phase one, 29–35% after.

`follows.py` is the measurement: asked for one of 64 photographs' places, how often what comes back
is nearest the place asked for, beside the same for a real crop of that photograph; and along a
walk between two places, the lab's way, how far the picture travels and whether only forward. A GAN
that ignored its input would draw just as well and fail all three.

`bank.py` folds named materials — the presets, or a folder of photographs — into the front of the
model, so it takes `z` and `category` like the category models and everything that reads those reads
it unchanged. Or keep `look` and hold the places yourself: each is a vector, and the patch sums them.

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

### flights.py — the latent, filmed along a route

```bash
python flights.py --model conditional-512.onnx --route orbit --category 2 --out waves-orbit.mp4
python flights.py --model looks-presets.onnx --route drift --weights 0.6,0,0.4 --out mix.mp4
```

Six closed routes, so every film loops without a seam: `orbit` (a great circle at the usual radius),
`knobs` (a Lissajous figure on the eight strongest axes), `drift` (every axis on its own slow
periodic noise, as band powers would move it), `tour` (six places, pausing at each), `breath` (out
from z = 0 to past the usual spread and back) and `categories` (the category crossing all of them).
Every export takes whitened axes strongest first, so the same `--seed` flies the same route through
two models, which is how to compare them by eye. A `.json` beside each film holds the route and how
far the picture moves each frame: on the same six routes the flagship moved two to three times as
much as the space GAN, because its latent changes the material and the space GAN's only its light.

```bash
python flights.py --model gan-looks/looks-best.onnx --route places --space texture-ip-sd15.npz --stops 12 \
  --even --frame 3:2 --upscale RealESRGAN_x4plus.pth --size 1920x1280 --fps 60 --seconds 48 --out archive.mp4
```

The space GAN is not held to seven categories, so two more routes move the material itself:
`places` passes through a set of places on a smooth closed curve (a folder of `.npy` from `bank.py
--places`, or `--stops` photographs from a space, each the most typical of one region of the archive)
and `wander` lets the space's twelve main directions drift, through mixtures no photograph holds.
`--even` spaces the frames by how far the picture moves rather than the route, which took the
unevenness of the five loops filmed on 10-30 from 0.6–0.8 down to 0.06–0.24. `--frame 3:2`
crops, `--upscale` enlarges every frame four times through `upscale.py`, and `--size` sets the film.

### upscale.py — four times larger, the same every time

```bash
python upscale.py --weights RealESRGAN_x4plus.pth --out big/ frame-*.png
```

Real-ESRGAN's network written out here, so only the weights file is read (`weights_only`). About
0.22 s for a 510 × 340 frame on the 3090 in half precision, so it is for films and stills; `hd.py`
redraws for single 4K pictures. Being deterministic, a film that is smooth at 512 stays smooth
enlarged, which a diffusion redraw is not. On the space GAN's materials it draws filaments, rims and
cracks where Lanczos leaves a blur.

### rae.py — DINOv2's tokens, drawn back

```bash
python rae.py --decoder rae/ViTXL_n08_i512/model.pt --out recon/ crop-*.png
python rae.py --decoder rae/ViTXL_n08_i512/model.pt --mix a.png b.png --steps 25 --out sweep/
python rae.py --decoder rae/ViTXL_n08_i512/model.pt --film --space texture-ip-sd15.npz --stops 24   --frame 3:2 --upscale esrgan/RealESRGAN_x4plus.pth --size 1920x1280 --seconds 180 --out morph.mp4
```

A representation autoencoder (Zheng, Ma, Tong and Xie, 2025): the frozen DINOv2-B with registers
turns a picture into a grid of 768-number tokens (32 × 32 at 512) and a ViT decoder draws the
picture back from them. The network is written out here; the decoders (`nyu-visionx/RAE-collections`,
MIT) are read with `weights_only`. Tried on 49 archive crops (11-03): it redraws rather than copies
(PSNR 21 at 512 against SDXL's VAE at 31, colour pushed about ten times more) but keeps as much fine
detail, and an encode and decode take 33 ms at 512 against the VAE's 103. Two materials mixed in the
token grid stay one surface at every step, where a crossfade double-exposes and a VAE latent turns to
coloured noise; a boundary set in the grid becomes a seam the decoder draws. The change along a mix
is uneven (most of it near the middle), and laying one grid's statistics on another moves almost
nothing.

A grid can be any shape: the encoder reads any size, and a grid larger than the decoder's 32 × 32 is
drawn as native windows that overlap by a quarter and cross-fade. Stretching the decoder's position
table to 3:2 instead scored the same but drew a stripe at every 16 px column on smooth materials. A
picture rebuilt from a material's tokens by nearest match decodes to a checkerboard: a token is
drawn with its neighbours, so mixes keep every token in its place.

`--film` morphs through a loop of photographs in the tokens (the archive's most typical photograph
of each of `--stops` regions, as `flights.py --route places` picks them, or the pictures given): a
closed Catmull-Rom curve through their grids, its frames spaced by how far the picture moves
(`flights.steady`, read on a half-size grid), each enlarged by Real-ESRGAN if asked.

In the lab, `compose` (rail: tokens · RAE) lays up to three texture photographs into a picture in
the token grid. The lab's mask says where they go and the picture's own tokens fill the rest; with
two or three, the picture's light, its depth or a side orders them (`balance` moves the boundaries,
`seam width` runs from a clean line to one long morph); `picture underneath` mixes the picture's
tokens back in, `material zoom` and `crop` choose the piece of each photograph. 58 ms a 768 × 512
frame while dragging; on release Real-ESRGAN draws it at 3072 × 2048 in about 0.6 s, and an export
keeps the original photograph's own pixels outside the mask. The decoder and the enlarger come from
`models.json` (`rae`, `upscale`); the method is left out when either file is missing.

`mosh`, beside it, is datamoshing in the token grid: tokens moved out of their neighbourhoods and
drawn back by a decoder that draws each as if it belonged, as a codec draws blocks of data in the
wrong frame. `rebuild` replaces each token (or 2 × 2, 4 × 4 piece: macroblocks) with its closest
match from a material photograph or from elsewhere in the picture, among the `wildness` closest at
random; `drag` pulls tokens toward a side, each as far as its place is bright, near or a noise field
says, so the grid tears where neighbours part and streaks where many take the same token; `sort`
puts the tokens of every run of the picture's brightest places in order of brightness (pixel
sorting). The mask limits where, `hold the picture` mixes the picture's tokens back in, and where
nothing moved the picture keeps its own pixels.

`hybrid` makes new textures from the materials alone, no picture: `blend` mixes them token by token
(the walk goes from the first through each to the last), `patchwork` lays them in grown patches with
decoder-drawn seams, `layout` rebuilds the first's arrangement out of the others' tokens, and
`remix` scatters the first's own pieces up to a reach. The **tokens tab** (`/tokens.html`) puts all
of it on one page: a picture and a palette of three materials, nine ways as tiles drawn from them
(laid in, rebuilt, self-moshed, smeared, sorted; blended, patchwork, layout × grain, remixed), one
dial per way with the rest under fine-tune, the lab's mask card for the ways on the picture, and
"surprise", nine random recipes from the palette or from any texture, to go on from with a click.

### raev2.py and bench_rae.py — DINOv3's tokens, beside DINOv2's

```bash
python rae.py --decoder raev2/stage1/general/dinov3l-k23/decoder.pt --film --space texture-ip-sd15.npz   --stops 24 --frame 3:2 --long 768 --upscale esrgan/RealESRGAN_x4plus.pth --size 1920x1280 --seconds 180 --out morph-v3.mp4
python bench_rae.py --out bench.json --quality 48
```

The same autoencoder on DINOv3: Singh, Zheng, Wu, Zhang, Shechtman and Xie's RAEv2 (2026). The token
is the mean of DINOv3 ViT-L/16's blocks 1 to 23, each through the final layer norm without its
learned scale and shift, plus the last layer's mean over the picture ("k23"); the early layers keep
the pixels' detail, so the decoder copies where RAE's redraws (PSNR 27.0 on ImageNet against 18.9 for
the last layer alone, their numbers). The decoder is RAEv2's ViT-XL trained beyond ImageNet
(`nyu-visionx/RAEv2-models`, `stage1/general/dinov3l-k23`, CC BY-NC 4.0), the same architecture as
RAE's, so `rae.py`'s Decoder reads it; `rae.load` picks DINOv3 for a decoder whose path names it,
and everything else (`--film`, windows, the lab) is shared. DINOv3 itself is gated: each user asks
Meta for `facebook/dinov3-vitl16-pretrain-lvd1689m` on Hugging Face. DINOv2's RAE is unchanged and
stays the default; the lab's token methods offer "tokens: DINOv3" once both are on disk.

A token is 16 pixels either way, but RAEv2's decoder was trained at 256, so a 3:2 frame is drawn in
many 16 x 16 windows (15 at 768 against DINOv2's two of 32 x 32), and its decoder draws pixels in
0..1 where RAE's draws them ImageNet-normalised. On the 3090 (`bench_rae.py`, 3:2; the lab's round
trip is compose with two materials and a subject mask, caches warm):

| long side | decoder alone, DINOv2 / DINOv3 | film frame, batched | encode a picture | lab, dragging | lab, on release (×4) |
|---|---|---|---|---|---|
| 512 | 18 / 32 ms (56 / 31 fps) | 14 / 26 ms | 9 / 21 ms | 56 / 71 ms | 0.47 / 0.50 s |
| 768 | 49 / 71 ms (21 / 14 fps) | 43 / 66 ms | 11 / 35 ms | 129 / 157 ms | 1.12 / 1.16 s |
| 1024 | 130 / 161 ms (8 / 6 fps) | 124 / 145 ms | 19 / 66 ms | 274 / 299 ms | 1.96 / 2.02 s |

Real-ESRGAN ×4 takes 0.22, 0.49 and 0.86 s at the three sizes for either. On 48 archive crops at 512,
DINOv3 reconstructs at PSNR 27.7 (SSIM 0.91, colour shift 0.015) against DINOv2's 21.3 (0.58,
0.044). Halfway between two pictures DINOv2 draws one new surface and DINOv3 something closer to a
double exposure, so DINOv2 stays the morph's (the flagship) and DINOv3 is for keeping a photograph
itself; the same 24-stop morph in DINOv3 is `exports/rae/morph-24-dinov3.mp4`.

`dinov3l-k7` (every other block from 11 to 23, the deeper half; `raev2.py` reads the layers off the
decoder's path) costs exactly what k23 costs and sits between the two: PSNR 22.2, SSIM 0.75. Measured
as colourfulness against the photograph's own (Hasler and Süsstrunk's, in `bench_rae.py`), DINOv2
gives 144%, DINOv3 k23 113% (close to the photograph: its "grey" is relative) and k7 165%; halfway
through a mix, DINOv2 keeps 84% of its ends' colour, k7 68%, k23 55%. The lab offers all three as
`tokens`; its morph is `exports/rae/morph-24-dinov3-k7.mp4`, the least evenly paced of the three.

### tokendit.py — new token grids, drawn

```bash
python tokendit.py train --crops crops-yellow3 --decoder rae/ViTXL_n08/model.pt --out tokendit/ --hours 3
python tokendit.py sample --model tokendit/last.pt --decoder rae/ViTXL_n08/model.pt --space texture-ip-sd15.npz   --places 12 --per 4 --guide 3 --grid 32x48 --upscale esrgan/RealESRGAN_x4plus.pth --out tokendit/samples/
python tokendit.py walk --model tokendit/ema-036390.pt --decoder rae/ViTXL_n08/model.pt --space texture-ip-sd15.npz   --stops 12 --seconds 90 --grid 16x24 --upscale esrgan/RealESRGAN_x4plus.pth --out exports/tokendit/walk.mp4
```

The second stage of the RAE paper at the size of its DiT^DH-S, written out: a diffusion transformer
that draws 16 × 16 grids of DINOv2 tokens, which RAE's 256 decoder turns into pictures. A narrow
body (12 blocks at 384) reads the noisy grid and the condition; a wide, shallow head (2 blocks at
2048) reads the noisy grid again, steered token by token by the body, which is how the paper fits
768-number tokens with a small model. Flow matching on the velocity, noise levels logit-normal and
shifted toward noise by sqrt(16·16·768 / 4096), 50 Euler steps. It trains on `embed_crops.py`'s
crops (a random 512 piece of each 576, flipped, encoded on the fly) and is steered by the lab's
material space: each crop's own place, or half the time its photograph's, with one in ten dropped
for "no material" so guidance can push away from it. 196M parameters; a few hundred steps already
give recognisable materials, coloured by the place asked for. Three hours at batch 32 (36,390 steps,
loss 0.59 at the first checkpoint, 0.46 at the end) on one 3090.

It only ever saw 16 × 16 grids. Any other grid is drawn in overlapping 16 × 16 windows whose
velocities are cross-faded at every step (`--grid`, and the lab's frame and size): each window sees
positions it was trained on and neighbours agree on their overlaps, so a 32 × 48 grid comes out one
seamless 3:2 texture, 2048 × 3072 enlarged, with forms larger than a window. Extending the
positions past the trained 16 instead holds at 16 × 24 and turns to brown mud at 32 × 48. A batch
of 16 draws at 0.15 s each at 16 × 16, 0.28 s at 16 × 24, 2 s at 32 × 48 (30 steps, guidance 3).

`tokendit.load` reads a training checkpoint (`ema-*.pt`) or the published half-precision
`.safetensors` (`Kairos-Hive/latent-textures`, `tokens/`, with the channels' mean and spread inside),
and the lab's `draw` takes either from its `tokendit` folder.

`walk` films a closed loop through the most typical photographs of as many regions of the space,
with the noise held, so one drawing changes material; paced by `flights.steady` (which now takes a
batched `many`) to move as much from each frame to the next. The lab's `draw` method (tokens tab,
"drawn new") takes the same frames and sizes, and drags at ten steps from the same noise.

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

The `sdxl` preset is the high-resolution renderer behind the same vectors: a quality SDXL base
(RealVisXL 4.0) at 1024 with the sdxl adapter and no speed-up. It shares its encoder with
`sdxl-turbo`, so any ViT-bigG space draws through both — quality against speed stays a dial, and
this is the quality end. A space's concept axes live in its encoder's geometry, so fit them per
space (`concepts.py --book concepts-xl.json`); the same caption exemplars give the same effect
sizes in either encoder, because the concepts are in the photographs.

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

## transform.py — the structure dial, so content and material stop sharing one knob

```bash
python transform.py --image portrait.png --space texture-ip-sd15.npz --preset "water reflections" --out grid.png
```

`restyle.py` and `terms.py` trade the subject away to buy the material, because denoising strength
is their only dial. This adds a depth ControlNet reading the source's geometry, so `--structure`
holds the volume while `--strength` and `--scale` push the material. High structure with high
strength is the place the one-knob tools cannot reach: a figure fully constituted of water that
still holds its pose. The grid is structure × strength because the right pair is not guessable —
a portrait, a facade and a landscape each give way somewhere else. Depth holds geometry, not
likeness: at strength 0.95 a face is still that body in that light, but not that face. The edge
ControlNet is the known next dial for likeness, and it is a download, not a training run.

Three more moves ride the same vector. `--against` puts a second material in the negative half of
the guidance — toward crust, away from water — which is also how a correlated axis is told no.
`--region depth --material2 X` splits the picture by its own geometry (or `lum`, by its light) and
gives each side its own material through the adapter's attention masks: the figure in crust, the
ground in water, one pass. `--then "X@0.45"` hands the image to a second material for a lighter
finishing pass: the first builds the forms, the second redraws the surface. **The handover travels
as latents.** A drawn texture decoded to pixels and encoded again is adversarial to img2img — it
confettis even with the adapter silent — while the same picture handed over inside the VAE's space
redraws clean. Real photographs survive the round trip; generated texture does not.

## composite.py — the texture's own pixels, no model at all

```bash
python composite.py --image portrait.png --texture bark.jpg --mode bands displace blend --out sheet.png
```

The exact-pixel end of the reference-fidelity dial, and the honest baseline `transform.py` has to
beat. `bands` hands the content's coarse scales to the texture's fine ones — the split is a
reading-distance dial, a face from across the room and pure material at arm's length. `displace`
pushes the texture's pixels around by the content's gradients and lights them with its shading: the
subject drawn BY the texture. `blend` is mixed-gradient Poisson, the stronger gradient winning
pixel by pixel. `chroma` puts the content's light in the texture's colour and `grain` the reverse;
`mask` floods the texture in from the shadows, the knob a luminance waterline; `relief` lights the
texture by the picture's slopes, the knob a sun walking the frame; `veins` engraves the texture's
own crack graph across the picture; `flow` combs the texture along the picture's contours, grain
kept, with `--smear` the LIC softness as a choice. At the top of the classical family sit `quilt`
(Efros–Freeman texture transfer: verbatim patches sewn along minimum-error cuts), `analogy` (its
per-pixel sibling by PatchMatch — no blocks, no seams) and `guided` (a guided filter pouring the
texture's tones into the picture's edges). Thirteen moves, numpy and PIL, CPU, exactly
reproducible, provenance intact.

## stylize.py — the statistics baseline

```bash
python stylize.py --image portrait.png --style caustics.jpg frost.jpg --between 5 --out sheet.png
```

Gatys by optimisation over the cached VGG19 — no decoder trained, none downloaded. Identity always
survives (statistics do not touch geometry) and the material arrives only as grain and palette,
which is exactly what makes it the floor: anything that claims to CONSTITUTE had better beat this.
`--between` morphs the style target itself, the one trick statistics own outright. `--loss patch`
climbs one rung: every relu3_1 neighbourhood of the content is style-swapped for the style's most
correlated one, so the material's texture ELEMENTS survive where Gram keeps only their statistics —
neural patch matching, still nothing trained. It wants a far smaller `--tradeoff`, around 2 to 30.

## concepts.py — a personal axis: the same vector is the ruler and the knob

```bash
python concepts.py --space texture-ip-sd15.npz --name humid \
    --pos "submerged under clear water" "wet and glistening" --neg "bone dry" "eroded" \
    --rank humid-rank.png --sweep humid-sweep.png
```

A concept is a direction in the IP space, fitted as the difference between two exemplar sets'
means. Project the archive onto it and the images sort by the concept (`--rank`, the sanity check
that costs nothing); add it to any point and the adapter draws the step (`--sweep`, in units of the
archive's own spread along the axis). Scorer and edit are one object — no bridge model. This seed
version picks exemplars through caption terms; an artist's own picks will replace that and nothing
else changes. Fitted axes accumulate in `concepts.json`. First results: `humid` sorts water from
baked plaster and sweeps monotonically from dry brick to open water; `fractured` un-fractures rock
into still water at its negative end, which is the archive's own correlation showing — in this set,
unbroken IS water.

## landmarks.py — the space's own places

```bash
python landmarks.py --space texture-ip-sd15.npz --out landmarks.png \
    --algebra "rippled water + tree bark - still water" --between DSC01705 DSC07698
```

The centroid — the one photograph the archive keeps trying to take — and its antipode, the far
side of the sphere, everything the archive refuses. `--algebra` adds and subtracts caption terms as
vectors; `--between` walks the great circle from one named photograph to another with the seed
held, the space between two memories.

## hang.py — the archive hung by another eye

```bash
python hang.py --embeddings categories/embeddings.npz --categories categories-7/categories.csv \
    --rhymes rhymes.png --ribbon ribbon.png --apart 400
```

Cross-category nearest neighbours in DINOv2 — pairs the folders keep apart that the eye puts
together — and one greedy nearest-neighbour walk through the whole set, the archive as a single
continuum. `--apart` requires the frame counters to differ, because without it the top of the list
is bursts straddling the category line: echoes, not rhymes. Pure lookup over `categories.py`'s
cache, CPU, seconds.

## lab.py — the app: every flagship, every knob, one page

```bash
python lab.py --space texture-ip-sd15.npz
python lab.py --space texture-ip-yellow3-xl.npz --preset sdxl --concepts concepts-yellow3-xl.json --port 8021
```

Every method is declared once, in `app/registry.py`, as a node signature: typed inputs, typed
parameters with ranges and one-line docs, a cost class. The page at `/` is generated from that, so
a parameter added to the registry appears in the interface with nothing else to edit. That one rule
is the fix for the old explorer, where a knob existed in code but reached no screen unless someone
copied it there by hand.

Five families. **GANs**: the three published flagships named in `models.json` (checked byte-for-byte
against Kairos-Hive), with the first twelve latent axes, seven named categories, spread and a
seed-to-seed walk, about 0.1 s a frame on CPU. **The space**: space, walk and transform, as before.
**LoRA & words**: `prompt` draws from text through any checkpoint of any LoRA run; `restyle` carries
a picture into the look. Every diffusion method also takes a prompt and a LoRA, so the archive's
two diffusion instruments finally compose. **Texture mapping**: all thirteen modes. **Optimisation**:
gram, adain and the neural patch swap with every knob, its stages streamed as they finish.

**Frames, not jobs.** Sliders ride a WebSocket (`app/live.py`): settings up, JPEG bytes down,
rendered in memory and never written to disk. The newest request replaces any waiting one, and the
page keeps one frame in flight, so the render time becomes the frame rate and a fast hand never
builds a backlog. While dragging, frames are drafts: texture mapping at 256px, diffusion on the
consistency LoRA fused into the UNet (unfused it doubles every step), 3 steps, the tiny decoder
(TAESD). On release you get a full frame. Measured on the 3090: GAN 512 at 80 fps, GAN 1024 at
24, relief 65, displace 27, space/walk/transform drafts at about 6. The GANs run on the GPU through
`onnxruntime-gpu==1.22.0`, the last build on the CUDA 12 runtime that PyTorch ships (newer ones
want CUDA 13). torch must be imported first, so its DLLs are on the path.

**Pictures keep their shape, and export at their own size.** Every method that takes a picture
renders in that picture's ratio: 6000×4000 works at 384×256, a portrait at 256×384, and the
texture is cropped to cover the same frame. Free generation (space, walk, prompt) takes an
*aspect*. Pixel knobs (blur radii, reaches, quilt blocks, analogy patches) are tuned in a 384
frame and scale with the frame they run in, so a setting looks the same in a drag draft, a
release and an export. *export full size* renders the current settings at the picture's original
pixel size, up to 8192 on the long side, as a background job:

- the classical modes compute natively at full size; wide blurs on big frames run on a
  box-shrunk copy, which matches the exact blur to within 0.001;
- quilt and analogy decide which texture pixel goes where at the working size (the search is the
  expensive part, and its answer doesn't depend on size), then `composite.remap` carries that
  map up and reads the full-size texture through it, blending at the seams. The export holds
  grain the working frame never had: 6000×4000 in about 8 s;
- stylize and the diffusion methods can't run at 24 megapixels; they render at their working or
  native size in the picture's shape and are scaled up, cropped rather than stretched. Diffusion
  frames are multiples of 8 (SD 1.5's UNet accepts that), not 64, so a 3:2 picture stays 3:2.

**Masks: where the material goes** (`masks.py`, `segment.py`; on every mode with a picture: the
texture-mapping modes, transform, restyle and the three stylizes). Unmasked, a mode lays its
material over the whole frame, and the result reads as one picture on another. A mask picks the
regions, in up to three layers met by *and*, *or* or *but not*, each from a source: the picture's
light, shadow, edges, flat places, saturation or a colour clicked on it; its depth (near/far); AI —
the people in it (the u2net person cutout in `~/.u2net`), an object named in words (Grounding DINO
finds it, SAM 2 cuts it out), a quality named in words (CLIPSeg: "wet", "shiny", "sky"), an object
clicked on (SAM 2; shift-click leaves a part out); the texture's own light or dark; or a shape — a
gradient from one side, a spot placed by a click, grown fractal noise. For the field sources
*coverage* is a quantile, so 0.3 means the picture's darkest, nearest or most sky-like 30% whatever
its exposure; the AI objects are regions already. Each layer can be inverted and grown or shrunk.
*follow* snaps the boundary to the picture's contours with a fast guided filter, *noise* roughens
it like a grown front, *show mask* draws the mask itself. Mask settings scale with the frame, so a
draft, a release and an export agree, and an export keeps the picture's own full-resolution pixels
outside the mask. The AI models load on first use (SAM 2 + Grounding DINO about 10 s) and their
answers are kept per picture, so moving a mask knob afterwards costs 60–150 ms at 1024; a drawing
is kept too, so changing only the mask of a transform or a stylize re-lays it instead of redrawing.

**Surface: the material on the picture's forms** (`surface.py`; shading on every mode with a
picture, wrap on the texture-mapping modes that lay the texture pixel for pixel, so not quilt and
analogy). A mode lays its material as on a flat sheet: one grain size and one light across a face
or a dress, like a pattern printed on glass in front of it. *Picture's shading* lays the picture's
own light over the material: in linear light, the picture's log luminance above the *material
detail* scale replaces the result's, both smoothed by a guided filter steered by the picture, so
shadows keep their edges, stop at outlines, and a result relit by its own picture comes back
unchanged (no doubled shading in modes that already carry it). *Object tones* decides whether an
object's overall darkness comes too (1: a dark coat stays dark) or only the shape of its light (0).
*Wrap over forms* curves the material over the depth model's relief (DPT, now kept as floats):
where a surface turns away it shows more of itself per pixel, so the texture coordinates are the
least-squares flattening of that stretch (a double-precision FFT Poisson solve in torch, ~50 ms at
768, kept per picture and frame). Depth jumps steeper than ~70° are silhouettes and are crossed,
not stretched; *roundness* domes each object toward its own outline, since a depth model sees a
person as a flat cut-out; *folds* adds a relief read from the shading; *shrink with distance*
makes the material finer as the picture recedes. The flattening is scaled back inside the texture
(by at most 1.5) so a stretched surface doesn't read mirrored copies of it. Costs at 768: shading
~45 ms, wrap ~10 ms per frame once flattened. Both default to off.

**What the survey added** (October 28; plan and sources in `roadmap/mask-material-survey.md`).
Every earlier behaviour is still there, as the alternative choice; a regression run through the
job API renders every old case bit-identical with the new options at their defaults.

- *Read forms with: geometry* (`geometry.py`, MoGe-2 ViT-L, MIT): the wrap takes its slopes from
  real surface normals (the turn of a cheek, a poster flat on its wall) instead of DPT's relief;
  wrap 1 is the surface as the camera sees it, and *shrink with distance* uses metric depth.
  *depth* keeps the DPT wrap with *roundness*. DPT still drives near/far masks and transform.
- *Read light from: intrinsic* (`intrinsics.py`, Marigold-IID-Lighting, OpenRAIL++-M): the
  picture's light is its intrinsic shading, so a striped shawl is lit instead of shadowed by its
  own stripes. *Picture's shine* adds back the highlights; *lit relief* lights the material's own
  bumps from the direction the picture is lit from (second-order spherical harmonics fitted to the
  shading over the MoGe normals). *brightness* keeps the earlier reading.
- Mask sources: *subject* (BiRefNet matting, MIT) cuts out whatever the picture is of, with soft
  edges; *material (click)* takes everything made of the clicked stuff (DINOv2-with-registers patch
  features, Apache), shift-click to leave a material out; *matte the cut-outs* resolves person and
  object edges with ViTMatte (Apache) so hair and leaves come out soft.
- *Material size* (`tiling.py`): the material at any grain, carried past the photograph's edge by
  histogram-preserving hex tiling (Heitz & Neyret) instead of mirrors; a wrap that reads past the
  edge is tiled the same way.
- *synth* (`ebsynth.py`): a mapping mode that rebuilds the picture from the material's own patches
  on the GPU, guided by its light at two scales (ebsynth, public domain, built locally as a CUDA DLL
  by `<models>/ebsynth/build.bat`): ~0.7 s at 768 where the numpy analogy takes ~7.6 s.
- *stylize swd* and *stylize efdm*: whole feature distributions matched (sliced Wasserstein, exact
  feature distribution matching), keeping a texture's extremes.
- *transfer* (`zest.py`, SD 1.5 space only): ZeST on SDXL — the material redrawn INSIDE the mask
  from MoGe depth, the object greyed first so its light survives, steered by the same ViT-H
  material space and concept axes through `ip-adapter_sdxl_vit-h`; *material enters: style*
  injects into the one style block (InstantStyle/MARBLE) and keeps the picture's content firmer.
  MARBLE's metallic / roughness / transparency / glow sliders (Stability AI Community Licence)
  move the material embedding. The SDXL pipeline is offloaded to RAM between draws so the card
  stays free for the rest of the lab.
- *Rebuild from real photographs* (transfer): the drawing remade from patches of the archive
  photographs nearest the material (ranked in the IP space, read from the photo drive) or of a
  texture photograph, by ebsynth — the generator decides the arrangement along the forms, every
  pixel is one of yours. Changing only the rebuild keeps the drawing (1–2 s instead of 15).
- *Archive LoRA* (transfer): an SDXL LoRA trained on the 2,133 four-star photographs at 1024 with
  their captions (`lora.py --model SG161222/RealVisXL_V4.0 --size 1024 --detail 1.5 --checkpointing
  --precision bf16 --lr 5e-5`), in `<models>/sdxl-lora-archive-2/`; the trigger "in goofi style" is
  added to the prompt when it is on. The first run, in fp16 at 1e-4, drifted (over-saturated by step
  500, black by 3000); bf16 and non-finite-step skipping fixed that. Every checkpoint, scored against
  real archive crops (colourfulness, saturation, contrast, fine detail), is more colourful than the
  archive at guidance 5; at guidance 3.5 and strength 0.7 step 1750 comes closest and stays
  photographic, so transfer has a guidance knob and the LoRA defaults to 0.7.
- *Body part* mask source (Sapiens2-seg 0.4B, in transformers; Sapiens2 Licence — no surveillance or
  biometric use): clothing, upper or lower clothing, hair, face, skin, hands, arms, legs, shoes.
- *edit* (`editor.py`, FLUX.2 klein 4B, Apache): an editor model shown a swatch — a place in the
  material space drawn first (concept axes and all) or a texture photograph — redraws the masked
  part in four steps, following an instruction. Offloaded to RAM between draws.
- *grow* (`nca.py`): the archive's own NCAs (style_ca.py, one per category and their parent) grow
  their material over the picture, spreading through the mask from its edge inward along a ragged
  front; *spread* lets it take the picture or draws it back.
- *Settle colours* (mask edges; `harmonise.py`, PCT-Net ViT, MPL-2.0, weights loaded as tensors
  only): the material's colour and tone moved toward the rest of the picture.
- *Light and colour apart*, as masks and as a mode (intrinsics.py's albedo, shading and residual):
  mask sources *lit* and *in shadow* (the light alone, whatever colour it falls on), *pale* and
  *dark surfaces* (the colours alone, whatever light is on them), *shine*, and *true colour (click)*,
  a colour picked with the light taken out so the same paint in sun and shade goes together. The
  mapping mode *relit* splits the texture too and puts its albedo under the picture's shading
  (its colour of light optional), its own relief kept by a knob, the picture's shine laid back on.
- Not yet: SAM 3 and DINOv3 are gated on Hugging Face and need an access request from the account.

Nothing goes to the history unless asked: *keep* (K) saves the current frame with its settings,
and double-clicking a kept frame reopens those settings. *hold* (H) pins a frame beside the live
one, and recipes save named settings to `recipes.json`. The minutes-long optimisations still use
the job queue, which shows each stage as it finishes.

## explore.py — every training-free instrument in one live page

```bash
python explore.py --space texture-ip-sd15.npz
```

Four tabs on a local server, `play.py`-style: the composite layer at drag rate on CPU; the IP
space drawn per request, with a preset or term mix for the material and one slider per fitted
concept axis from `concepts.json`; the two-dial transformer; and walks — pin two places (a preset,
a term mix, a photograph, the centre, the antipode) and slide the great circle between them, with
the concept axes bending the whole path. The diffusion tabs have a draft
mode — the LCM LoRA at 5 steps, ~0.3 s, for searching — and a full mode that turns it off for the
real picture. Models load lazily on first use of their tab, so the page is up in a second.

Below the controls sits the whole archive the space was built from: searched by its own captions,
sorted along any fitted concept axis, thumbnailed and paged. A click hands the photograph to the
open tab — the texture for composite, the MATERIAL for space and transform, since every photograph
is a point in the space. The texture dropdown is just the starting palette; the archive is the
real one.

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
