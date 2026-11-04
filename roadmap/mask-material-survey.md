# Mask & material survey (27–28 Oct 2026)

Survey of 2024–2026 open-source work on masking, geometry/light, diffusion material transfer and
texture synthesis, for integration into the lab. The full report, with a catalogue of every
candidate, is at https://claude.ai/artifact/TRsDecoku1yMgUkSZEmFZa. Nothing was downloaded; every
download below needs an OK.

## Status (28 Oct)

Built and checked against a regression run of every earlier method (bit-identical at defaults):
1 MoGe-2 (`geometry.py`), 2 Marigold-IID (`intrinsics.py`), 4 BiRefNet + ViTMatte (`segment.py`),
5 material click on DINOv2-with-registers (DINOv3 gated), 6 hex tiling (`tiling.py`), 7 SWD + EFDM
(`stylize.py`), 8 lit relief (`surface.relight`), 9 ZeST + MARBLE (`zest.py`, method `transfer`;
differential diffusion not yet), 10 ebsynth (`ebsynth.py`, mode `synth`, DLL built locally).
Blocked: 3 SAM 3 (gated; the account has no access yet). Tested and dropped: MM Grounding DINO
large (worse masks than Grounding DINO tiny on these pictures). Then: the hybrid (transfer's drawing
rebuilt from archive patches), an SDXL LoRA trained on the archive, 11 Sapiens2 body parts, 12 the
FLUX.2 klein editor (Qwen-Image-Edit-2509 left out: 58 GB, 20B), 13 growth from the archive's own
NCAs (DyNCA/NoiseNCA not used: non-commercial, and trained on others' images), 14 PCT-Net
harmonisation (IC-Light not used).

## Build order

Phase 1: read the photo properly.

1. **MoGe-2 ViT-L normal** (MIT, 331M, ≈1.3 GB, `pip install git+https://github.com/microsoft/MoGe`):
   metric depth, normals and field of view. `surface.flatten` takes its slopes from the normals
   (g = −n_xy / n_z, clamped near grazing) and its recede from metric depth and the field of view.
   DPT stays as a fallback.
2. **Marigold-IID-Lighting v1-1** (OpenRAIL++-M, diffusers `MarigoldIntrinsicsPipeline`):
   albedo, diffuse shading and residual in linear light. `surface.light_of` reads the shading
   (stops albedo patterns leaking in as light), and the residual (shine) is added back. A/B it
   against Careaga & Aksoy's Intrinsic v2, which is academic use only.
3. **SAM 3** (`Sam3Model` is in transformers 5.16; HF repo gated; SAM License): replaces Grounding
   DINO + SAM 2.1, takes example boxes, and its soft concept map may replace CLIPSeg. MM Grounding
   DINO is the ungated stopgap.
4. **BiRefNet HR-matting** (MIT; remote code wants timm, kornia, einops, which are not installed)
   as a "subject (matte)" source, plus **ViTMatte** (MIT, in transformers) as a refine step: any
   mask goes to a trimap, then to a matte.
5. **Same material (click)**: DINOv3 patch-feature similarity (gated; DINOv2 Apache fallback),
   then MatSeg (CC0, Zenodo 11536561) as the learned version.
6. **Hex-tiling / histogram-preserving blending**, plus a material scale knob, so the material can
   be any size with no seams.
7. **Sliced Wasserstein + EFDM** losses in stylize.py.

Phase 2: new passes and modes.

8. **Relight the material's relief.** Fit second-order spherical-harmonic irradiance to the IID
   shading over the MoGe normals. Build a normal map from the material's high-pass, warp it with
   the wrap coordinates, rotate it into the surface frame, and apply SH(N') / SH(N).
   DiffusionLight-Turbo (MIT) is an option for hard light.
9. **ZeST recipe + MARBLE on SDXL.** Use the ControlNet inpaint pipeline with the mask inside
   sampling, start the object in grey, and pass soft masks through differential diffusion. MARBLE
   adds single-block injection plus metallic, roughness, transparency and glow MLPs (Stability
   Community License). Both use h94 `ip-adapter_sdxl_vit-h` (≈0.7 GB, not cached), which takes the
   same ViT-H embedding as SD 1.5, so concepts.json carries over unchanged. The current SDXL preset
   uses the bigG adapter.
10. **ebsynth (CUDA)**, public domain with a PatchMatch patent warning: GPU guided synthesis with
    light, normal and mask guides. It replaces the 34 s numpy analogy.
11. **Sapiens2** (ICLR 2026, Sapiens2 License): 29 body-part and clothing classes, plus human
    normals inside person masks.

Phase 3: heavy or experimental.

12. An **editor method**: FLUX.2 klein 4B (Apache), then Qwen-Image-Edit-2509 with the
    material-transfer LoRA (Apache, int4/fp8). The material enters as image tokens, so concept
    axes don't apply. Feed the editor a steered swatch rendered by the SD 1.5 stack.
13. **DyNCA / NoiseNCA** for live growth and dissolution (CC BY-NC-SA).
14. **Harmonising pass**: IC-Light fbc or PCT-Net.

Watch: MoGe-3 (Triton), Marigold V2 (Qwen-Edit LoRAs, Linux), Sa2VA / SAM 3 Agent, Cells2Pixels,
DealMaTe, Qwen-Image-2.1 (non-commercial).

## Unverified

- 3090 timings for most models.
- Whether MARBLE's MLPs work with the SD 1.5 adapter.
- SAM 3's soft map on material words.
- How well DINOv3 separates materials.
- The Depth Pro weight licence.
- Whether the PatchMatch patent is still in force.
- MatSwap on diffusers 0.40.
