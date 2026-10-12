#!/usr/bin/env bash
. "$(dirname "$0")/paths.sh"
# Runs after night.sh. Step count and inference strength trade against each other: gentle-4000 was
# the richest on lichen but already soft on copper, while gentle-2000 was crispest on copper. That
# is the signature of one LoRA simply pushing harder than another, not of one being better, so the
# two are varied together here instead of one at a time.

G="$OUT/sd15-lora-gentle"
cd "$LAB" || exit 1

# Bounded wait: if night.sh has not finished in three hours it has failed, and starting GPU work
# beside whatever it is still doing would only confuse the result.
for _ in $(seq 180); do
    grep -aq "^=== done ===" "$OUT/night.log" && break
    sleep 60
done
grep -aq "^=== done ===" "$OUT/night.log" || { echo "night.sh never finished; not starting"; exit 1; }

echo; echo "=== pinning the step and strength together ==="; date
"$PY" -u compare.py --seeds 3 --out "$OUT/step-vs-strength.png" \
    --prompt "a close-up photograph of oxidised copper, veined and flaking, in turquoise and umber, wet and glistening, in goofi style" \
             "a close-up photograph of lichen on stone, encrusted and mottled, in rust and olive drab, in flat overcast light, in goofi style" \
    --lora "2000 @1.0=$G/step-002000@1.0" "2000 @1.6=$G/step-002000@1.6" \
           "3000 @1.0=$G/step-003000@1.0" "3000 @1.6=$G/step-003000@1.6" \
           "4000 @1.0=$G/step-004000@1.0" "4000 @1.6=$G/step-004000@1.6"
echo "=== refine done ==="
