#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# One sequential pass, run unattended. Deliberately no retry loop: a loop that relaunches a killed
# GPU job once put two trainers on one card writing to the same files for seven hours.
#
# Paths are Windows-style with forward slashes, which both bash and Python accept here. A POSIX
# `/c/Users/...` reaches Python as `\c\Users\...` and is not a path at all — that cost a run.

SET="$SETS/training-set-by-category"
SHARP="$OUT/sd15-lora-captioned/step-001000"
cd "$LAB" || exit 1

step() { echo; echo "=== $* ==="; date; }

# Every input checked before any GPU work starts, so a mistyped path costs a second.
for needed in "$SHARP/pytorch_lora_weights.safetensors" \
              "$OUT/sd15-lora-gentle/step-002000/pytorch_lora_weights.safetensors" \
              "$OUT/sd15-lora-gentle/step-004000/pytorch_lora_weights.safetensors" \
              "$OUT/sd15-lora-gentle/pytorch_lora_weights.safetensors"; do
    [ -f "$needed" ] || { echo "missing: $needed"; exit 1; }
done
[ -d "$SET" ] || { echo "missing the image set"; exit 1; }
echo "all inputs present"

# 1. Gentle against sharp, same prompts and seeds, drawn at the strength that exposes a weak LoRA.
step "comparing gentle against step-001000"
"$PY" -u compare.py --seeds 4 --out "$OUT/gentle-vs-sharp.png" \
    --lora "sharp 1000 @1.4=$SHARP@1.4" \
           "gentle 2000 @1.4=$OUT/sd15-lora-gentle/step-002000@1.4" \
           "gentle 4000 @1.4=$OUT/sd15-lora-gentle/step-004000@1.4" \
           "gentle 6000 @1.4=$OUT/sd15-lora-gentle@1.4" || echo "comparison failed"

# 2. The space itself, on the base the LoRA was trained against.
step "building the SD 1.5 IP space"
"$PY" -u ip_space.py --images "$SET" --preset sd15 --detail 3 --crops 8 --axes 32 \
    --out "$OUT/texture-ip-sd15.npz" || { echo "space build failed; nothing after it can run"; exit 1; }

# 3. What the space holds, with real crops beside each row.
step "drawing the space alone"
"$PY" -u ip_space.py --space "$OUT/texture-ip-sd15.npz" --beside "$SET" \
    --sheet "$OUT/ip-sd15-alone.png" || echo "space sheet failed"

# 4. The pairing that matters: the LoRA for the look, the space for somewhere to move.
step "drawing the space under the LoRA"
"$PY" -u ip_space.py --space "$OUT/texture-ip-sd15.npz" --beside "$SET" \
    --lora "$SHARP" --style 1.0 --sheet "$OUT/ip-sd15-lora-10.png" || echo "lora sheet 1.0 failed"
"$PY" -u ip_space.py --space "$OUT/texture-ip-sd15.npz" --beside "$SET" \
    --lora "$SHARP" --style 1.6 --sheet "$OUT/ip-sd15-lora-16.png" || echo "lora sheet 1.6 failed"

step "done"
ls -la "$OUT"/gentle-vs-sharp.png "$OUT"/ip-sd15-*.png "$OUT"/texture-ip-sd15.npz 2>/dev/null
