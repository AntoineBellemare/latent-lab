#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# A GAN conditioned on the lab's space instead of on seven categories, so a mix of materials made in
# the lab can be drawn live in goofi — the space mixes well and draws at a quarter of a second at best;
# the GAN draws in 36 ms and could only mix its categories.
#
# Warm-started from the three-star GAN, which already draws this set: the place starts silent and the
# run teaches it to steer. Two phases for the same reason the categories needed them — at the
# ordinary rate a new input can stay silent for good.

CROPS="$OUT/crops-yellow3"
GAN="$OUT/gan-looks"
cd "$LAB" || exit 1
step() { echo; echo "=== $* ==="; date; }
mkdir -p "$GAN"

if [ ! -f "$CROPS/crops.npz" ]; then
    step "cutting the crops and placing them"
    "$PY" -u embed_crops.py --images "$SETS/goofi-yellow3-1600-by-category" --out "$CROPS" \
        --size 512 --detail 3 --crops 16 2>&1 | grep -aE "images,|wrote|Traceback|Error"
fi

step "phase one (tag-lr 10)"
"$PY" -u fastgan.py --images yellow3 --look "$CROPS" --size 512 --batch 8 --workers 3 \
    --init "$OUT/gan-yellow3/textures.pt" --tag-lr 10 --steps 20000 --snap 2000 --out "$GAN/looks.onnx" \
    2>&1 | tr '\r' '\n' | grep -aE "crops of|started|^step +[0-9]+000 |exported|wrote|Traceback|Error"

step "phase two (tag-lr 1)"
"$PY" -u fastgan.py --images yellow3 --look "$CROPS" --size 512 --batch 8 --workers 3 \
    --tag-lr 1.0 --steps 40000 --snap 2000 --resume --out "$GAN/looks.onnx" \
    2>&1 | tr '\r' '\n' | grep -aE "carrying|^step +[0-9]+000 |exported|wrote|Traceback|Error"

step "does it draw what it is asked for"
"$PY" -u follows.py --model "$GAN/looks.onnx" --crops "$CROPS" --sheet "$GAN/follows.png" 2>&1 | grep -av "Loading"

step "gan looks done"
