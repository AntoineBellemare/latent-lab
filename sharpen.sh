#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# The space drew soft, and the LoRA did not sharpen it, so neither is the cause. The sd15 preset
# runs an LCM LoRA at 4 steps and guidance 1.0, built for speed; a style LoRA then stacks on top of
# LCM's own. This turns the speed-up off and denoises properly to find out what the space actually
# holds. If it sharpens, the preset is a realtime setting and not the one to judge the space by.

SET="$SETS/training-set-by-category"
SPACE="$OUT/texture-ip-sd15.npz"
# Between the two readings: 2,000 was crisper on copper, 4,000 richer on lichen.
LORA="$OUT/sd15-lora-gentle/step-003000"
cd "$LAB" || exit 1

for _ in $(seq 60); do
    grep -aq "^=== refine done ===" "$OUT/refine.log" && break
    sleep 30
done
grep -aq "^=== refine done ===" "$OUT/refine.log" || { echo "refine never finished; not starting"; exit 1; }

echo; echo "=== the space, denoised properly ==="; date
"$PY" -u ip_space.py --space "$SPACE" --beside "$SET" --fast none --steps 30 --guidance 7.0 \
    --sheet "$OUT/ip-sd15-sharp.png" || echo "sharp sheet failed"

echo; echo "=== the same, under the LoRA ==="; date
"$PY" -u ip_space.py --space "$SPACE" --beside "$SET" --fast none --steps 30 --guidance 7.0 \
    --lora "$LORA" --style 1.0 --sheet "$OUT/ip-sd15-sharp-lora.png" || echo "sharp lora sheet failed"

echo "=== sharpen done ==="
