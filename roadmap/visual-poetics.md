# Visual poetics: materializing arbitrary images, steered by personal concepts

The mission, as the research report states it: take an arbitrary image — a face, a landscape, an
object — reconstitute it through a material from the archive, and navigate it along continuous,
personally defined dimensions (colloidal, fractality, humidity, dissolution, oneirism...), taught
by examples and judgments rather than only by words. Three output types, and they want different
machinery: **still artworks** (quality first), **a live instrument** (the engine's realtime layer,
biosignal-driven, speed first — goofi and TouchDesigner are hosts, not the thing built), and **a
navigable vocabulary** (the concept axes themselves, which both of the others consume). What this
repo builds throughout is one thing: a **latent-texture engine** — spaces, models, concept
directions and transforms with ordinary file formats at every edge.

## Where the report is right, where it is already done, and where it is wrong

The report was written blind to this repo, and it shows in both directions.

**Already done here, which the report lists as Phase 0–1 work.** The "unresolved architectural
fact: which GAN" is resolved: FastGAN with a StyleGAN mapping network, path-length regularization,
and a PCA-whitened `w` export — the latent is already an ordered, calibrated control surface, which
is what GANSpace would have been run to approximate. The dataset audit exists (`dedupe.py`,
`categories.py`, `separable.py`, per-image captions over 2,133 images); the archive representation
exists (`texture-ip-sd15.npz`, 17,064 CLIP-ViT-H crop vectors with measured category separations);
the measurement culture the report asks for (monotonicity, interference, don't trust previews) is
this repo's founding principle. Phase 0 and most of Phase 1 of the report's roadmap are behind us,
not ahead.

**Right, and worth taking.** Three things in the report are genuinely missing here:

1. **Structural conditioning.** `restyle.py` has one knob — denoising strength — so content
   preservation and material intensity are the same dial. The report is right that depth/edge
   ControlNet gives a second, independent dial. This is the single cheapest high-value addition:
   pretrained SD 1.5 ControlNets compose with the keeper LoRA and the IP space as they are.
2. **Personal concept axes.** Caption terms (`terms.py`) are the archive's vocabulary; the mission
   wants the *artist's* vocabulary — "colloidal" defined by picked exemplars and "which is more
   colloidal?" judgments. The report's four learning modes (text, exemplar, preference, discovered)
   are the right menu.
3. **Dissolution as an explicit goal.** The report's "third transformation goal" — structure
   released gradually until only a trace remains — is exactly a performance arc, and no tool here
   names it yet.

**Wrong, or heavier than the evidence supports.** GAN inversion is not needed: navigating actual
photographs is what the IP space already does, training-free. A custom IP-Adapter, a Swapping
Autoencoder, a new VAE, PBR decomposition, a universal concept encoder — the report itself gates
these behind "only if justified by evidence", and nothing yet justifies them. The report defers
realtime to the end, but for this project the live instrument is a primary output, not an
optimization pass — it just runs on the *cheap* generators (GAN, NCA, PCA), not on diffusion, so it
is not blocked by diffusion speed.

**Right, and nearly dropped.** The report's deterministic layer (§3.1) — compositing, alpha masks,
displacement warping, frequency-band transfer, gradient-domain blending — is not a lesser version
of the diffusion path; it is the only occupant of one end of a dial the mission names. "Reference
fidelity" asks whether the output carries the *actual pattern of a selected photograph* or a new
texture sharing its statistics, and no generative model sits at the exact-pixel end. It is also the
provenance output type — a work made of the archive's own pixels is a different artistic object
from one that resembles them — and it is the honest baseline the diffusion path has to beat, which
is this repo's founding move (`pixel_pca.py` plays that role for generators; nothing plays it for
transformation yet). The same section's local maps — crack masks, orientation fields, apparent
height — point somewhere the report itself doesn't quite take them: extracted from a chosen
photograph they can condition the *material's* geometry onto the content, by displacement or as a
ControlNet hint, which is closer to "constituted of" than any statistics transfer.

**The key simplification the report missed.** It devotes a section (6.3) to bridging a concept
*scorer* to an actual *edit* — gradient optimization, trained mappers, reward-guided diffusion.
Here the bridge is free: the IP-Adapter consumes vectors from the very space a concept probe would
be fitted in. A direction in CLIP-ViT-H space is simultaneously a scorer (project and rank) and an
edit (add to the conditioning vector and draw). `terms.py --push` already does this for caption
terms; a learned concept is one more direction in the same space. No new model stands between a
definition and a knob.

## The plan

### Phase 1 — a second dial: structure apart from material

Add depth and edge ControlNet (SD 1.5) to the img2img path. New tool or an extension of
`restyle.py`: `transform.py --image face.jpg --preset "water reflections" --structure 0.8
--strength 0.6`. `--structure` is the ControlNet conditioning scale; strength and IP scale stay the
material dials. Draw the sweep grid the way `restyle.py` does — structure on one axis, strength on
the other — because the right pair is not guessable per picture.

*And build the thing it has to beat:* `composite.py`, the deterministic baseline — the source's
luminance structure carrying the texture's pixels. Multiscale frequency transfer (the texture's
high bands over the content's low bands), displacement of the texture by the content's gradients,
gradient-domain blending for the seams. One tool, three classical moves, exact pixels, exact
reproducibility. It is E4's option (1) and the report's first named baseline, and it doubles as the
exact-pixel end of the reference-fidelity dial, which diffusion cannot reach. AdaIN is the second
cheap baseline and costs an afternoon via any pretrained VGG. Run all three on the same content
pool; the diffusion path earns Phase 3 only where the sheets show the cheap pair failing — which
they will, on material *constitution*, but the comparison is the proof and the two baselines remain
instruments in their own right where provenance or reproducibility is the point.

*Gate:* a portrait stays recognizably itself at a strength where today's `restyle.py` has dissolved
it (≥ 0.7), while carrying the material — and visibly beats the compositing and AdaIN baselines on
the same sources for material constitution, not just plausibility.

*Also in this phase:* assemble the content image pool — faces (with consent), architecture,
landscapes, objects. Small is fine (30–50 images), but fixed, so every later comparison holds the
inputs still. This is the one dataset the mission needs that does not exist (`restyle-sources/` in
the models folder is its seed).

### Phase 2 — personal concept axes in the space that already exists

`concepts.py`, three ways in, one representation out:

- **Exemplar-defined:** pick high/low images from the archive (a `play.py`-style local picker
  page); fit a direction in the IP space — difference of means first, linear probe if that is not
  enough. Start with the report's four deliberately dissimilar pilots: fractality, colloidal,
  humidity — and dissolution rather than vastness, because vastness is a whole-image compositional
  property that 512px texture crops cannot carry, which the report itself concedes.
- **Preference-defined:** "which is more humid?" over chosen pairs; rank-fit in the same space.
- **Discovered-then-named:** principal axes of the space drawn as sweeps (`ip_space.py --sheet`
  shows them already); the artist names the ones that mean something.

A concept saves to `concepts.json` beside `presets.json`: name, direction, exemplar ids, the
material families it was defined on, and a calibration (how far along it one just-noticeable step
is). `terms.py` learns to accept a concept wherever it accepts a term, so concepts compose with
presets and mixes immediately.

*Ground it in numbers that are not learned:* `describe.py`, handcrafted multiscale descriptors per
image — Fourier/wavelet band energy, Gabor orientation and structure-tensor anisotropy, box-count
fractal dimension and lacunarity, crack density and branching from a ridge detector, region-size
distribution, local entropy, colour statistics. The report proposes these twice and both uses are
right. As a **confound audit** of the IP space: correlate each principal axis against the
descriptors, and an axis that is mostly exposure or camera distance is unmasked before anyone
builds a knob on it — E1's real question. As **objective correlates** for concepts: perceived
fractality is not fractal dimension, but a learned "fractality" direction that does not even
correlate with fractal dimension and lacunarity is reading something else, and the check is free.
Where a CLIP probe is weak — branching and crack morphology are things CLIP compresses away — the
descriptors join the probe's features rather than compete with it.

*Measure it, repo-style:* a `concept_matrix.py` that turns axis *i* and reports the measured change
in every axis *j* — the report's attribute-response matrix, which is this repo's kind of number,
with the descriptors supplying the measured side alongside the learned scorers. And the scorer
sanity check: rank the archive along the direction, draw top and bottom twenty as a contact sheet,
and let the eye veto. A concept that cannot rank photographs will not steer a generator.

*Gate:* on held-out picks, the exemplar-trained direction beats the text-only CLIP direction for
the same word, and a sweep sheet along each pilot axis reads monotone to the artist who defined it.
Held-out means held out **by parent photograph**, never by crop — eight crops of one image sit in
the space together, and a probe scored against its own image's other crops is grading its memory
(the report's §9A point, and the same burst problem `dedupe.py` already fought at training time).

### Phase 3 — dissolution, and the full compositional surface

One macro-knob from structure release: as dissolution rises, ControlNet weight falls, denoising
strength rises, and the conditioning map itself coarsens (blur the depth map — the geometry
literally loses resolution). At 0 the photograph; at 1 free texture; the middle is where the poetry
lives, and the middle is what the morph/step-spread tools here were built to certify as even.
Frequency-band transfer gives the same arc its deterministic voice — hand over one scale band at a
time, fine to coarse, and the image dissolves with its pixels' provenance intact — so dissolution
is one knob with two renderers behind it.

*Material-side structure, the report's local maps taken further than it took them:* extract a crack
graph, orientation field or apparent-height map from a chosen photograph and impose it — as a
displacement field in `composite.py`, or fed to ControlNet in place of (or blended with) the
content's own edges. The face's geometry drawn by the wall's fractures is "constituted of" in the
strongest available sense, and both halves are pretrained or classical.

The composed instrument at the end of this phase, all of it existing machinery plus Phases 1–2:

> source image + structure dial + material point (preset / term mix / concept position) + concept
> sliders + dissolution + seed walk → image

which is the report's `I_out = F(I_in, S, M, a, s)` with every symbol pointing at a file in this
repo.

### Phase 4 — split by output type

- **Stills:** batch sweeps over the content pool; a resolution path past 512 (tile/upscale pass at
  30 steps; quality is allowed to cost). Keep provenance per artwork — source id, model, seed,
  conditioning, concept settings — the report's reproducibility point, and cheap if done now.
- **Live:** the realtime layer stays the GAN/NCA/PCA, and the concept axes *transfer* to it rather
  than run diffusion fast: sample the GAN, embed its outputs into the IP space, score them along a
  concept direction, and regress that score against the GAN's own whitened latent. A concept
  learned once becomes a direction in any generator that can be sampled — measurable with the
  `axes.py` methodology, and it gives the poetic sliders at 60 fps where diffusion gives them at 4.
  The existing "Next" items (cheapest denoising that holds detail; trigger word; carry keeper +
  space + knobs into the realtime hosts) live here unchanged.
- **Vocabulary:** the concept picker/sheets from Phase 2 are themselves a deliverable — the
  interface where a word becomes a coordinate.

### Explicitly not building, until a gate fails

A new GAN, a VAE, GAN inversion, a custom IP-Adapter, a Swapping Autoencoder, PBR/material-map
estimation, DINOv3 migration (DINOv2 is in place and measured), SliderEdit-class editors, a
reflection-specialist module, and any annotation program beyond the artist's own picks. Each has a
trigger written next to it in the report; none has fired. The report's own best sentence applies:
none is a prerequisite to the first proof of concept.

## Creative horizons, by approach

What each approach uniquely makes possible — the reason to build it beyond the gates above.

- **Deterministic (§3.1):** provenance as form. Every region of the output traceable to a negative;
  the bill of materials is part of the work. Frequency-band chimeras make viewing distance the
  dissolution dial — a print that is a face across the room and pure lichen at arm's length.
  One material's crack skeleton animating another's flesh.
- **LoRA (§3.4):** the archive as adjective. Subjects that were never photographed, rendered in the
  trip's eye; strength as a live dial is "how much of my eye is on this image", performable. The
  checkpoint timeline is its own axis — a world mid-way through learning a sensibility.
- **IP space (§3.5):** image algebra as poetry. The centroid — the one photograph the archive keeps
  trying to take — and its antipode, everything it never photographs. Saved paths through the
  sphere are scores: replayable, shareable compositions. k-NN against a live camera is the world
  projected into one photographer's memory.
- **ControlNet/img2img (§3.6):** the recognition threshold as portrait — every face dissolves at a
  different strength, and that number is a likeness. What survives materialization longest (eyes
  last, backgrounds first) maps what the model believes is essential.
- **GAN latent (§3.9):** the dream organ. Concept directions transferred into it close the
  neurofeedback loop — the image of your calm, affecting your calm. The 77,500 post-peak steps are
  an archive of decay worth playing back as a film.
- **Embedding navigation (§3.11):** the archive re-curated by a nonhuman eye — hang the exhibition
  by DINOv2 neighbourhood instead of by trip, and the rhymes the photographer never noticed appear.
- **Concept learning (§6):** a private language made operational. Discovered-then-named directions
  are collaborative coinage — the machine proposes, the artist baptizes. The sweep sheet along
  "oneirism" is the artist's own concept estranged — what the machine misunderstood is the mirror.
  The concept matrix's off-diagonals are not failures to erase: in this archive humidity drags
  green and vastness drags emptiness, and those involuntary associations are the photographer's
  unconscious, measured. Two people's definitions of one word differ by a vector, and that
  disagreement direction can be walked.
- **PBR / height (§3.2, if ever justified):** the texture leaves the screen — apparent-height maps
  as carving instructions, lichen that never grew, relit by the gallery's own light.
- **Dissolution (goal C):** the performance arc — camera-born form releasing to material and
  returning, breath on the knob. Four kinds of time to compose with: geological (the material),
  biographical (the archive's dates), machinic (training steps), performative (the signal).

## Order of operations, and why

Phase 1 before Phase 2 because structure control is the mission's opening image and a weekend of
plumbing, while concepts are the research contribution and deserve the instrument working end to
end first. Phase 4's GAN-transfer trick depends on Phase 2's directions existing. Every gate is a
drawn sheet plus a number, in that order — the sheets catch what the numbers miss and vice versa,
which is the lesson this repo was built on.
