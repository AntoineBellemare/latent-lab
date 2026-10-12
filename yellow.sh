#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# The curated set end to end: drop the bursts, describe every frame, build the space.
# Every path points at the `texture` folder itself, never its parent: dedupe puts rejects in a
# SIBLING folder, and a tool walking the parent would read them straight back in.

SET="$SETS/yellow-4star/texture"
cd "$LAB" || exit 1
step() { echo; echo "=== $* ==="; date; }

step "dropping the 46 near-duplicates"
"$PY" -u dedupe.py --images "$SET" --apply 2>&1 | tr '\r' '\n' | grep -aE "moved|leaving|groups" 

step "describing every frame"
"$PY" -u caption.py --images "$SET" --detail 3 --write 2>&1 | tr '\r' '\n' | grep -aE "captioned|wrote" | tail -3

step "building the space"
"$PY" -u ip_space.py --images "$SET" --preset sd15 --detail 3 --crops 8 --axes 32 \
    --out "$OUT/texture-ip-yellow.npz" 2>&1 | tr '\r' '\n' | grep -aE "images in|vectors of|axes to|wrote" 

step "yellow done"
