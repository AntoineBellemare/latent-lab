# Creative mechanisms: a plan per approach, and the roadmap over them

Companion to `visual-poetics.md`, which argues priorities and gates. This file answers a different
question: taken one at a time, what is the plan for *each* approach in the research report, and
what kind of creative act does each one serve? The organizing idea is that the eleven approaches
are not eleven tools in a row — they are seven **mechanisms**, seven different relationships
between an artwork and the archive. Every approach belongs to one; every output type is a mechanism
made public.

## The seven mechanisms

| # | Mechanism | The relationship to the archive | Approaches (report §) | Primary output types |
|---|---|---|---|---|
| I | **Reassembly** | the work is *made of* the photographs | deterministic mapping (3.1) | provenance prints, distance-dependent prints |
| II | **Resemblance** | the work *looks the way the archive looks* | AdaIN (3.3), LoRA (3.4) | mirror installations, adjective works, training-time pieces |
| III | **Navigation** | the work is a *position or path* in a space built from the archive | IP space (3.5), GAN latent (3.9), PCA, embeddings (3.11) | scores/paths, centroid/antipode pieces, nonhuman curation, live organ |
| IV | **Metamorphosis** | a *foreign image* is carried into the archive's matter | ControlNet/img2img (3.6), dissolution, slider editors (3.7) | threshold portraits, dissolution arcs, survival maps |
| V | **Language** | *words* become coordinates; the artist's vocabulary is the instrument | concept learning (§6), captions/terms | private vocabulary, coined axes, the measured unconscious |
| VI | **Feedback** | a *body* closes the loop; the signal is the hand | the engine's realtime layer under a signal (III + V; goofi and TouchDesigner are hosts) | performances, neurofeedback loops |
| VII | **Materialization** | the texture *leaves the screen* | PBR/height (3.2) | reliefs, carvings, relit objects |

Structure-texture models (3.8) and autoencoders (3.10) are latent members of III and IV — they
earn a build only if a gate below fires.

## The overarching roadmap, in four waves

Mechanisms do not queue; they ripen. Each wave has a theme, and every wave ships something
showable, because a mechanism is only proven by a work.

**Wave 1 — Honest hands (I, II, and IV's foundation).** Build the cheap, exact instruments and the
structure dial. Everything here is classical or pretrained; nothing can fail silently. Ships: the
first provenance pieces and the first held portrait.

**Wave 2 — The space speaks (III and V).** The archive's geometry becomes playable and the
personal vocabulary becomes real. Ships: the centroid/antipode pair, the first recorded scores, the
four pilot concept axes with their contact sheets and matrix.

**Wave 3 — Arcs and loops (IV completed, VI).** Dissolution as a macro-knob, concepts transferred
into the realtime generator, the signal on the dials. Ships: a performable arc and the
threshold-portrait series.

**Wave 4 — Public forms (all, plus VII if triggered).** Exhibition-scale resolution, curation by
the nonhuman eye, installation builds, and the physical gate decision. Ships: the show.

Dependencies, compressed: Wave 1 needs only the content pool. Wave 2's concepts (V) want Wave 1's
`describe.py` sibling but not its transforms. Wave 3 needs V's directions (for VI) and IV's dial
(for dissolution). Wave 4 needs taste, which the sheets accumulate.

---

## Mechanism I — Reassembly

### Approach: deterministic mapping (report §3.1)

*Already in hand:* nothing — the one empty slot in the repo, and the honest baseline everything
generative must beat.

*Build (Wave 1):* `composite.py` with three classical moves sharing one interface — multiscale
frequency transfer (content's coarse bands under texture's fine bands, with a band-split knob),
gradient displacement (texture warped by the content's gradient field, amplitude knob), and
gradient-domain (Poisson) blending for seams. Inputs: one content image, one or more archive
photographs chosen by hand or by IP-space nearest-neighbour. Add `maps.py`: crack graph (ridge
detector + skeletonize), orientation field, local scale map, apparent height (shading prior) from
any archive photograph — usable as displacement fields here and as conditioning in Mechanism IV.

*Measure:* reproducibility is free (deterministic); the interesting number is the band-split sweep
sheet — at which scale the eye hands over from face to material.

*Make:* **Bill of materials** — portraits assembled from named negatives, each shipped with its
sources and shares. **Reading distance** — large prints where the coarse band is the sitter and
the fine band is lichen: the viewer walks the dissolution axis. **Skeleton transfusions** — paint's
crack graph displacing water's pixels.

*Go further when:* a piece needs the texture to follow the content's *volume* rather than its
gradients — that fires the Mechanism VII gate (height), not more compositing.

## Mechanism II — Resemblance

### Approach: AdaIN (report §3.3)

*Already in hand:* nothing, but it is an afternoon over pretrained VGG.

*Build (Wave 1):* `adain.py` — content + reference(s), style interpolation between two references,
optional per-region masks. Keep it fast enough for a camera feed later.

*Measure:* it is itself a measure — the statistics-only baseline that shows what §3.6's semantics
add.

*Make:* **The mirror** — a camera restyled live, today's material chosen daily; visitors as
content. **Crossfade** — two archive references interpolated as a live duet. Also the scouting
tool: which archive images make strong anchors, before anything is trained on them.

*Go further when:* never, alone — AdaIN's ceiling is statistical resemblance; its job is to be
passed.

### Approach: LoRA (report §3.4)

*Already in hand:* the keepers (`sd15-lora-gentle/step-002000`, `sd15-lora-yellow/step-002000`),
the steps×strength law (~4,800 product), `compare.py`, numbered checkpoints on disk.

*Build (Wave 2):* nothing new to train. `scrub.py` — render a fixed prompt+seed across every
checkpoint of a run, as strip and as video: training time as a playable axis. Decide the trigger
word (open item in `texture-diffusion.md`); one retrain only if it changes.

*Measure:* the strip *is* the measure — where learning becomes memorising, visible.

*Make:* **Adjectives** — subjects the archive never contained, rendered in its eye; prompts as
captions-vocabulary collage. **How much of my eye** — strength as a live dial (1.0–1.8 is the
played range) on a signal. **Apprenticeship** — the checkpoint scrub as a piece about a world
learning a sensibility.

*Go further when:* a wanted look needs what SD 1.5 cannot carry — then and only then revisit the
base model, with `compare.py` as judge.

## Mechanism III — Navigation

### Approach: IP space (report §3.5)

*Already in hand:* `texture-ip-sd15.npz` / `-yellow.npz` (17k vectors), `terms.py`, `presets.json`,
spherical morphs, push/scale semantics — the most finished instrument in the project.

*Build (Wave 2):* `landmarks.py` — compute and draw the centroid and its antipode (the far point
on the sphere, rescaled to real-crop length like every point); pairwise "between two photographs"
morphs by id. `path.py` — record, save, replay and transpose walks through the space (JSON of
timestamped vectors + seeds): the score format, which Mechanism VI will drive live.

*Measure:* already measured (axis variance, category separations); add held-out-by-parent checks
for anything fitted in the space (see `visual-poetics.md`).

*Make:* **The photograph the archive keeps trying to take** (centroid) and **everything it
refuses** (antipode) — a self-portrait of an archive in two images. **Scores** — walks composed,
saved, transposed across seeds like a melody across keys. **Image algebra** — analogy completions
(rock : ? :: wave : foam) as a series. **Reminiscence** — live camera k-NN: the world answered by
the archive's nearest memories.

*Go further when:* a path needs to pass through what the encoder cannot distinguish (the weakest
category pairs) — that is evidence for a structure-texture model (§3.8), currently dormant.

### Approach: GAN latent (report §3.9)

*Already in hand:* conditional 512 + grown 1024, PCA-whitened `w` export (GANSpace built in),
`axes.py` (28 strong / 51 visible axes), `play.py`, the full decay footage of run-100k.

*Build (Wave 3):* `transfer.py` — sample the GAN, embed outputs into the IP space, score along a
concept direction (Mechanism V), regress against the GAN's own whitened latent; write the result
as a named direction file any host of the engine can load — `play.py` first, goofi and
TouchDesigner after. Validate with `axes.py` methodology
(multiple bases) and a sweep sheet per transferred concept.

*Measure:* regression R² per concept (how much of "humidity" the GAN's space can even express),
plus monotonicity of the sweep.

*Make:* **The dream organ** — concept-labelled knobs at 60 fps. **Decay footage** — the 77,500
post-peak steps of run-100k cut as a film of a model forgetting; the snapshots already exist.

*Go further when:* a transferred concept reads R² near zero — that names precisely what the GAN's
space lacks, and is the only honest argument for retraining it.

### Approach: embeddings and archive navigation (report §3.11)

*Already in hand:* CLIP + DINOv2 embeddings (`categories/embeddings.npz`), category CSVs,
agreement/stability machinery, captions.

*Build (Wave 4):* `hang.py` — order the archive by embedding neighbourhood (a greedy walk or 1-D
UMAP) and emit contact sheets / an ordering file: the nonhuman curation. Optionally a `play.py`-
style browser page over precomputed neighbours (no new models, pure lookup).

*Measure:* none needed — curation is judged on the wall.

*Make:* **Found rhymes** — the exhibition hung by DINOv2 instead of by trip: ice beside wall-crack,
years apart. **The walkable memory** — the browser as installation, one person's attention
spatialized for strangers.

*Go further when:* navigation wants *reconstruction* (click a point between two photographs and see
it as an image at full fidelity) — that is the autoencoder gate (§3.10), currently dormant; note
the IP space already approximates this generatively.

## Mechanism IV — Metamorphosis

### Approach: ControlNet + img2img (report §3.6)

*Already in hand:* `restyle.py` (strength sweeps, the one-knob version), the keeper LoRA, the IP
space as material source, `restyle-sources/` as the content pool's seed.

*Build (Wave 1):* `transform.py` — SD 1.5 + depth and edge ControlNet beside IP-Adapter + LoRA:
`--structure` (conditioning scale), `--strength`, `--scale`, material from preset/terms/concepts.
Grid sheets with structure × strength axes, `restyle.py`-style. Assemble the content pool (30–50
fixed images: faces with consent, architecture, landscape, objects).

*Measure:* the Wave 1 gate in `visual-poetics.md` — recognizable at strength ≥ 0.7, beating
compositing and AdaIN on constitution. Plus the identity-survival sweep per content class.

*Make:* **Recognition thresholds** — per sitter and per material, the strength at which they stop
recognizing themselves; the threshold is the portrait. **Survival maps** — which regions resist
materialization longest, as a diptych with the source. **Material-side structure** (Wave 3, with
`maps.py`): the wall's crack graph fed to ControlNet in place of the sitter's edges — constituted
of, in the strongest sense.

### The dissolution arc (report's goal C)

*Build (Wave 3):* `dissolve.py` — one macro-knob coupling structure weight down, strength up,
conditioning map coarsening; certified even with the morph/step-spread tools; a deterministic twin
in `composite.py`'s band-split so the same arc has two renderers.

*Make:* **The arc** — camera-born form releasing to material and returning; breath on the knob is
Mechanism VI's first duty.

### Approach: slider/instruction editors (report §3.7) — dormant

*Trigger:* an edit the material stack cannot phrase — weathering, submergence, aging — wanted as a
continuous dial. *First experiment if triggered:* one SliderEdit-class model, one weekend, the
content pool, verbs like "more submerged / a century older"; judged against `transform.py` on the
same sources. *Payoff if it wins:* time and weather join the vocabulary as axes — a facade aging a
century in ten frames.

### Approach: Swapping Autoencoder / structure-texture nets (report §3.8) — dormant

*Trigger:* Mechanism III hits encoder limits (paths through indistinguishable pairs) or IV needs
texture transplants that statistics and attention both miss. *First experiment if triggered:* train
on texture crops + content pool mixed, smallest config, judged by whether (structure A, texture B)
recombinations beat `transform.py` sheets. *Payoff if it wins:* the archive squared — 17k × 17k
chimeras from a finite set of photographs.

## Mechanism V — Language

### Approach: concept learning, four modes (report §6)

*Already in hand:* the caption vocabulary (59 materials, 106 surface terms — a found poem already),
`terms.py` as the text-defined mode working today, the IP space where scorer and edit are the same
vector.

*Build (Wave 2):* `concepts.py` — exemplar mode (local picker page, difference-of-means then
probe), preference mode (pairwise "which is more X?", rank fit), discovered mode (principal axes
drawn as sweeps, awaiting names); `concepts.json` beside `presets.json` (name, direction, exemplars,
domain, calibration in just-noticeable steps). `describe.py` — the handcrafted descriptors
(wavelets, Gabor, fractal dimension, lacunarity, crack stats, entropy) as confound audit and
objective correlates. `concept_matrix.py` — turn axis *i*, measure all axes *j*.

*Measure:* the Wave 2 gate in `visual-poetics.md` (exemplar beats text-only on held-out-by-parent
picks; monotone sweeps). Pilot axes: fractality, colloidal, humidity, dissolution.

*Make:* **Coinage** — the discovered mode as collaborative baptism: the machine proposes a
direction that has no word; the artist names it; the vocabulary grows neologisms. **The estranged
concept** — the sweep along learned "oneirism" as a piece about what the machine thinks you mean.
**The measured unconscious** — the concept matrix's off-diagonals (humidity drags green, vastness
drags emptiness) printed large: the archive's involuntary associations. **The disagreement
direction** — two people define one word; the vector between their definitions is walkable.

*Go further when:* a concept refuses to fit in CLIP space even with descriptor features — that
names a genuinely new representation problem, and is the only trigger for custom encoders.

## Mechanism VI — Feedback

### The body on the dials (the engine's realtime layer under a signal)

The engine exports ordinary `.onnx`, `.npy` and direction files; goofi and TouchDesigner are hosts
that play them, not the thing being built.

*Already in hand:* the ONNX exports and the host nodes that run them (goofi's
`Decoder`/`Onnx`/`NeuralCA`), NCA weight
blending, the 4–6 fps diffusion measurement, the roadmap's standing items (cheapest denoising;
carry keeper + space + knobs in).

*Build (Wave 3):* wire `transfer.py`'s concept directions into the GAN patch (a `Weights`-style
named-direction file); breath/EEG onto `dissolve.py`'s macro-knob at diffusion pace (an arc is
slow by nature — 4 fps suffices); `path.py` playback as a host input so a recorded score can be
re-performed live.

*Measure:* latency budget per chain, and the concept monotonicity re-checked *through the signal*
(a knob that is monotone under a hand may not be under a noisy band power — smooth accordingly).

*Make:* **The loop** — EEG-calm steering image-calm steering EEG-calm: biofeedback where the
feedback is beauty. **The breathing arc** — inhale toward form, exhale toward matter, live.
**Re-performance** — yesterday's recorded walk as tonight's accompaniment.

## Mechanism VII — Materialization

### Approach: PBR / height estimation (report §3.2) — dormant, with a named dream

*Trigger:* a piece commits to relief, relighting, or 3D surfaces; or Mechanism I needs volume.
*First experiment if triggered:* single-image height + normal estimation (Material Palette-class,
pretrained) on twenty archive images; judged by a raking-light render against the photograph.
*Payoff if it wins:* **Lichen that never grew** — latent textures as carving instructions (CNC or
print), relit all day by the gallery's own moving light; the final slider handed to the sun.

---

## Wave 1 ran, 2026-10-12 — every training-free approach built and tested

Six tools in one day, nothing trained, nothing downloaded; evidence sheets in
`latent-lab-models/poetics/` with its own README. What the sheets settled:

- **Reassembly works and displace is its find**: the sitter modeled IN paint, caustics and golden
  waves (`composite.py`); hard geometry survives far more displacement than a face.
- **The statistics floor is drawn** (`stylize.py`): identity trivially held, material as grain and
  palette only — what "constitution" must beat, and now visibly does.
- **The structure dial exists** (`transform.py`): pose and volume held to strength 0.95 while the
  flesh becomes water or lichen crust. Depth holds geometry, NOT likeness — the face drifts at high
  strength, and the edge ControlNet (a download) is the named next dial. The living-crust grid is
  the strongest constitution result to date.
- **A concept axis is a working knob on day one** (`concepts.py`): `humid` ranks the archive
  cleanly and sweeps monotonically from dry brick to open water; `fractured`'s negative end leaves
  rock and becomes still water — the archive's own correlation (unbroken IS water), the measured
  unconscious arriving on the first sweep, exactly as Mechanism V predicted.
- **The nonhuman curator earns the wall** (`hang.py`): with a shoot guard (`--apart`), genuine
  cross-category rhymes; without it, bursts — echoes, not rhymes, a finding in itself.
- **The space's landmarks render** (`landmarks.py`): centroid, antipode, term algebra,
  photograph-to-photograph great circles.

**Upgrades that followed the same day:** regional materials (the adapter's attention masks, split
by the picture's own depth or light), negative materials (`--against`, a vector in the guidance's
negative half), scheduled handover (`--then`, latents between the passes — a drawn texture
re-encoded from pixels is adversarial to img2img and confettis; inside the VAE's space it redraws
clean). First training experiment ran the same day (`mapper.py`, a concept-aligned autoencoder of
the embedding cloud): **the linear axes won** — the learned chart loses a photograph's identity
before delivering the edit, so the instrument ships on difference-of-means and the harness waits
for more data. The high-resolution branch is the original SDXL flagship resumed: the `sdxl` preset
draws the same vectors through RealVisXL at 1024, and the concept axes re-fit per encoder with the
same effect sizes, because the concepts are in the photographs.

## What ships when, in one view

| Wave | Builds | First works |
|---|---|---|
| 1 | `composite.py`, `maps.py`, `adain.py`, `transform.py`, content pool | bill-of-materials prints, reading-distance print, first held portrait |
| 2 | `concepts.py`, `describe.py`, `concept_matrix.py`, `landmarks.py`, `path.py`, `scrub.py` | centroid/antipode pair, first scores, pilot-axis sheets, apprenticeship strip |
| 3 | `dissolve.py`, `transfer.py`, realtime-host wiring, material-side ControlNet | threshold portraits, the breathing arc, dream-organ set, decay film cut |
| 4 | `hang.py`, resolution path, installation builds, VII gate decision | the show: found-rhymes hang, the mirror, the measured unconscious |

Dormant throughout, each behind its named trigger: slider editors, structure-texture nets,
autoencoders, PBR. A trigger firing is a finding, not a failure — it means a mechanism found its
limit in public, which is the only place a limit is worth finding.
