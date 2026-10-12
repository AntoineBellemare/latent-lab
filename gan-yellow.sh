#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# The GAN recipe that won, on the curated set. Two phases, because one tag-lr does not do both
# jobs: at 1.0 the category embeddings stay near silent and three mineral categories came out
# merged after 42,000 steps, while a high tag-lr separates them but costs picture quality. So the
# categories are pulled apart first and the picture is settled afterwards.

SET="$SETS/yellow-4star/texture"
BYCAT="$SETS/yellow-by-category"
GAN="$OUT/gan-yellow"
cd "$LAB" || exit 1
step() { echo; echo "=== $* ==="; date; }

mkdir -p "$GAN"

step "finding categories in the curated set"
"$PY" -u categories.py --images "$SET" --out "$OUT/categories-yellow" --groups 7 --detail 3 \
    2>&1 | tr '\r' '\n' | grep -aE "images under|chosen|^ +[0-9]+ +[0-9]+|looks" | tail -14
[ -f "$OUT/categories-yellow/categories.csv" ] || { echo "no categories.csv; stopping"; exit 1; }

step "folding them into folders"
"$PY" -u by_category.py --images "$SET" --csv "$OUT/categories-yellow/categories.csv" --out "$BYCAT" \
    2>&1 | tr '\r' '\n' | tail -10

step "phase one: pulling the categories apart (tag-lr 10)"
"$PY" -u fastgan.py --images "$BYCAT" --classes --size 512 --detail 3 --batch 8 \
    --tag-lr 10 --steps 25000 --snap 2500 --out "$GAN/textures.onnx" \
    2>&1 | tr '\r' '\n' | grep -aE "images in|^step|wrote|Traceback|Error" | tail -12

step "what the categories are worth after phase one"
"$PY" -u separation.py --model "$GAN/textures.onnx" 2>&1 | tail -16

step "phase two: settling the picture (tag-lr 1)"
"$PY" -u fastgan.py --images "$BYCAT" --classes --size 512 --detail 3 --batch 8 \
    --tag-lr 1.0 --steps 50000 --snap 2500 --resume --out "$GAN/textures.onnx" \
    2>&1 | tr '\r' '\n' | grep -aE "^step|wrote|Traceback|Error" | tail -12

step "what they are worth at the end"
"$PY" -u separation.py --model "$GAN/textures.onnx" 2>&1 | tail -16

step "gan yellow done"
