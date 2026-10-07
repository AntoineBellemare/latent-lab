#!/usr/bin/env bash
# The same presets, the same picture, the same seed, through both spaces. The only thing that
# differs is which photographs the space was built from, which is the whole question: does 2,107
# curated four-star frames beat 2,133 unfiltered ones.

PY="C:/Users/Antoine/github/goofi/training/.venv/Scripts/python.exe"
OUT="C:/Users/Antoine/goofi-models"
IMG="$OUT/restyle-sources/portrait.png"
cd "C:/Users/Antoine/github/latent-lab" || exit 1

for _ in $(seq 120); do
    grep -aq "^=== yellow done ===" "$OUT/yellow.log" && break
    sleep 60
done
grep -aq "^=== yellow done ===" "$OUT/yellow.log" || { echo "the set never finished; not starting"; exit 1; }

echo; echo "=== what each set is made of ==="; date
for pair in "old:$OUT/texture-ip-sd15.npz" "new:$OUT/texture-ip-yellow.npz"; do
    echo "--- ${pair%%:*}"
    "$PY" -u terms.py --space "${pair#*:}" --list --least 60 2>/dev/null | grep -aE "^ +[0-9]" | head -22
done

echo; echo "=== the same presets through both ==="
"$PY" -u terms.py --space "$OUT/texture-ip-sd15.npz" --image "$IMG" --preset all \
    --push 3 --strength 0.5 --scales 0.9 --out "$OUT/presets-old.png" 2>&1 | grep -aE "dropped|wrote"
"$PY" -u terms.py --space "$OUT/texture-ip-yellow.npz" --image "$IMG" --preset all \
    --push 3 --strength 0.5 --scales 0.9 --out "$OUT/presets-new.png" 2>&1 | grep -aE "dropped|wrote"

"$PY" - <<'PY'
from PIL import Image, ImageDraw
import json, pathlib
a = Image.open("C:/Users/Antoine/goofi-models/presets-old.png")
b = Image.open("C:/Users/Antoine/goofi-models/presets-new.png")
cell, gut = 512, 330
names = [k for k in json.loads(pathlib.Path("presets.json").read_text(encoding="utf-8")) if not k.startswith("_")]
sheet = Image.new("RGB", (gut + 3 * cell, 30 + len(names) * cell), (18, 18, 20))
pen = ImageDraw.Draw(sheet)
for i, h in enumerate(["the picture", "old: 2133 unfiltered", "new: 2107 curated"]):
    pen.text((gut + i * cell + 8, 10), h, fill=(214, 210, 202))
for r, name in enumerate(names):
    pen.text((10, 30 + r * cell + cell // 2), name, fill=(214, 210, 202))
    sheet.paste(a.crop((gut, 30 + r * cell, gut + cell, 30 + (r + 1) * cell)), (gut, 30 + r * cell))
    sheet.paste(a.crop((gut + cell, 30 + r * cell, gut + 2 * cell, 30 + (r + 1) * cell)), (gut + cell, 30 + r * cell))
    sheet.paste(b.crop((gut + cell, 30 + r * cell, gut + 2 * cell, 30 + (r + 1) * cell)), (gut + 2 * cell, 30 + r * cell))
sheet.save("C:/Users/Antoine/goofi-models/old-vs-curated.png")
sheet.resize((int(sheet.width * 0.42), int(sheet.height * 0.42)), Image.LANCZOS).save(
    "C:/Users/Antoine/goofi-models/old-vs-curated-small.png")
print("wrote old-vs-curated.png")
PY
echo "=== comparison done ==="
