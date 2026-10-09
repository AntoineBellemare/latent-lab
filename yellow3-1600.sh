#!/usr/bin/env bash
# Yellow at three stars and up, written out the way the original set was written: resized to 1600px
# on the shortest side, not hardlinked at full resolution.
#
# That resize is the whole of the earlier GAN result. `--detail 3` takes a third of the shortest
# side and resizes it to 512, so from 1600px it is a 533px crop at essentially 1:1 and from a 4000px
# original it is a 1333px crop downscaled 2.6x. The flag is a fraction of the frame, not a pixel
# count, so the same recipe means a different magnification on a different source.

PY="C:/Users/Antoine/github/goofi/training/.venv/Scripts/python.exe"
FROM="G:/Antoine/PHOTOGRAPHIE/goofi-yellow-3star/texture"
SET="G:/Antoine/PHOTOGRAPHIE/goofi-yellow3-1600/texture"
BYCAT="G:/Antoine/PHOTOGRAPHIE/goofi-yellow3-1600-by-category"
OUT="C:/Users/Antoine/goofi-models"
GAN="$OUT/gan-yellow3"
cd "C:/Users/Antoine/github/latent-lab" || exit 1
step() { echo; echo "=== $* ==="; date; }
mkdir -p "$GAN"

step "resizing to 1600 on the shortest side"
"$PY" -u - <<'PY'
import pathlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
SIDE = 1600
src = pathlib.Path("G:/Antoine/PHOTOGRAPHIE/goofi-yellow-3star/texture")
out = pathlib.Path("G:/Antoine/PHOTOGRAPHIE/goofi-yellow3-1600/texture")
out.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
def one(p):
    dest = out / (p.stem + ".jpg")
    if dest.is_file():
        return True
    try:
        im = Image.open(p).convert("RGB")
        s = SIDE / min(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dest, quality=95)
        return True
    except Exception as e:
        print(f"  skipped {p.name}: {e}", flush=True)
        return False
with ThreadPoolExecutor(max_workers=16) as pool:
    ok = sum(pool.map(one, files))
print(f"wrote {ok} of {len(files)} into {out}")
PY

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
