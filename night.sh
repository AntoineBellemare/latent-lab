#!/usr/bin/env bash
# One sequential pass, run unattended. Deliberately no retry loop: a loop that relaunches a killed
# GPU job once put two trainers on one card writing to the same files for seven hours.

PY="/c/Users/Antoine/github/goofi/training/.venv/Scripts/python.exe"
LAB="/c/Users/Antoine/github/latent-lab"
OUT="/c/Users/Antoine/goofi-models"
SET="G:/Antoine/PHOTOGRAPHIE/goofi-training-set-by-category"
SHARP="$OUT/sd15-lora-captioned/step-001000"
cd "$LAB" || exit 1

step() { echo; echo "=== $* ==="; date; }

# 1. Wait for the trainer already on the card. Bounded: if it has not finished in two hours
#    something is wrong, and starting more GPU work beside it would make that worse.
step "waiting for sd15-lora-gentle"
for _ in $(seq 120); do
    grep -aq "^wrote .*sd15-lora-gentle" "$OUT/sd15-lora-gentle.log" && break
    grep -aqE "Traceback" "$OUT/sd15-lora-gentle.log" && { echo "trainer failed; stopping"; exit 1; }
    sleep 60
done
grep -aq "^wrote .*sd15-lora-gentle" "$OUT/sd15-lora-gentle.log" || { echo "trainer still running after 2h; stopping"; exit 1; }

# 2. Gentle against sharp, same prompts and seeds.
step "comparing gentle against step-001000"
"$PY" -u compare.py --seeds 4 --out "$OUT/gentle-vs-sharp.png" \
    --lora "sharp 1000 @1.4=$SHARP@1.4" \
           "gentle 2000 @1.4=$OUT/sd15-lora-gentle/step-002000@1.4" \
           "gentle 4000 @1.4=$OUT/sd15-lora-gentle/step-004000@1.4" \
           "gentle 6000 @1.4=$OUT/sd15-lora-gentle@1.4"

# 3. The space itself, on the base the LoRA was trained against.
step "building the SD 1.5 IP space"
"$PY" -u ip_space.py --images "$SET" --preset sd15 --detail 3 --crops 8 --axes 32 \
    --out "$OUT/texture-ip-sd15.npz"

# 4. What the space holds, with real crops beside each row.
step "drawing the space alone"
"$PY" -u ip_space.py --space "$OUT/texture-ip-sd15.npz" --beside "$SET" \
    --sheet "$OUT/ip-sd15-alone.png"

# 5. The pairing that matters: the LoRA for the look, the space for somewhere to move.
step "drawing the space under the LoRA"
"$PY" -u ip_space.py --space "$OUT/texture-ip-sd15.npz" --beside "$SET" \
    --lora "$SHARP" --style 1.0 --sheet "$OUT/ip-sd15-lora-10.png"
"$PY" -u ip_space.py --space "$OUT/texture-ip-sd15.npz" --beside "$SET" \
    --lora "$SHARP" --style 1.6 --sheet "$OUT/ip-sd15-lora-16.png"

step "done"
ls -la "$OUT"/gentle-vs-sharp.png "$OUT"/ip-sd15-*.png "$OUT"/texture-ip-sd15.npz 2>/dev/null
