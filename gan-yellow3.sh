#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# Yellow at three stars and up: 4,541 frames, still all texture, still seventeen trips, but 2.2x the
# data of the four-star set.
#
# The four-star GAN came out flat and that was read as diversity spreading a fixed capacity thin.
# The reading is not safe, because the four-star set is ALSO small: 2,107 frames covering seventeen
# trips against 2,133 covering one. This run holds the diversity and doubles the data, which is the
# only way to tell whether a GAN blurred because the world got wider or because it was shown too
# little of it.

SET="$SETS/yellow-3star/texture"
BYCAT="$SETS/yellow3-by-category"
GAN="$OUT/gan-yellow3"
cd "$LAB" || exit 1
step() { echo; echo "=== $* ==="; date; }
mkdir -p "$GAN"

step "dropping the bursts"
"$PY" -u dedupe.py --images "$SET" --apply 2>&1 | tr '\r' '\n' | grep -aE "groups|moved|leaving"

step "finding categories"
"$PY" -u categories.py --images "$SET" --out "$OUT/categories-yellow3" --groups 7 --detail 3 \
    2>&1 | tr '\r' '\n' | grep -aE "images under|chosen|^ +[0-9]" | tail -12
[ -f "$OUT/categories-yellow3/categories.csv" ] || { echo "no categories.csv; stopping"; exit 1; }

step "folding them into folders"
"$PY" -u by_category.py --images "$SET" --csv "$OUT/categories-yellow3/categories.csv" --out "$BYCAT" \
    2>&1 | tr '\r' '\n' | tail -10

step "phase one (tag-lr 10)"
"$PY" -u fastgan.py --images "$BYCAT" --classes --size 512 --detail 3 --batch 8 \
    --tag-lr 10 --steps 25000 --snap 2500 --out "$GAN/textures.onnx" \
    2>&1 | tr '\r' '\n' | grep -aE "images in|^step|wrote|Traceback|Error" | tail -8

step "phase two (tag-lr 1)"
"$PY" -u fastgan.py --images "$BYCAT" --classes --size 512 --detail 3 --batch 8 \
    --tag-lr 1.0 --steps 50000 --snap 2500 --resume --out "$GAN/textures.onnx" \
    2>&1 | tr '\r' '\n' | grep -aE "^step|wrote|Traceback|Error" | tail -8

step "what the categories are worth"
"$PY" -u separation.py --model "$GAN/textures.onnx" 2>&1 | tail -16

step "gan yellow3 done"
