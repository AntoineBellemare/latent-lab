#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# The LoRA for the curated set, on the recipe that won on the old one: lr 3e-5, which moved the
# point where it blurs from step 1,500 out past 4,000. Numbered saves because the peak is chosen
# from drawn sheets and never from the loss — the loss held a normal band through 3,000 steps while
# the pictures blurred to mush.

SET="$SETS/yellow-4star/texture"
NEW="$OUT/sd15-lora-yellow"
cd "$LAB" || exit 1
step() { echo; echo "=== $* ==="; date; }

# Behind the space comparison, never beside it: one card.
for _ in $(seq 150); do
    grep -aq "^=== comparison done ===" "$OUT/two-spaces.log" && break
    sleep 60
done
grep -aq "^=== comparison done ===" "$OUT/two-spaces.log" || { echo "the comparison never finished; not starting"; exit 1; }

step "training the curated LoRA"
"$PY" -u lora.py --images "$SET" --detail 3 --model runwayml/stable-diffusion-v1-5 --size 512 \
    --batch 2 --rank 16 --lr 3e-5 --steps 6000 --report 500 --save-every 1000 --workers 4 \
    --sample-steps 20 --guidance 7.0 --out "$NEW" 2>&1 | tr '\r' '\n' | grep -aE "^step|trained param|images under|wrote|Traceback|Error" | tail -20

step "the curated LoRA against the old one"
for n in 1000 2000 3000 4000; do
    [ -f "$NEW/step-00${n}/pytorch_lora_weights.safetensors" ] || { echo "missing step-00$n"; exit 1; }
done
"$PY" -u compare.py --seeds 4 --out "$OUT/yellow-vs-asie.png" \
    --lora "old ASIE 2000=$OUT/sd15-lora-gentle/step-002000@1.4" \
           "curated 1000=$NEW/step-001000@1.4" \
           "curated 2000=$NEW/step-002000@1.4" \
           "curated 4000=$NEW/step-004000@1.4" 2>&1 | grep -aE "drew|wrote|Error"

step "yellow lora done"
