#!/bin/bash
# Standing Z-Image visual gate: the Klein-gate analog whose absence let the
# 2026-07-18 MFA corruption live for 25 days. Renders both profiles at BOTH
# attention routes' resolutions (512 = SDPA, 1024 = MFA) across 3 prompts and
# content-checks every image. Run after ANY engine change; exit nonzero on
# any failure.
# Env: ZDRAW_WEIGHTS, ZDRAW_ZPACK, MFLUX_PY (for the checker's python),
# ZGATE_OUT, ZGATE_VISUAL_REVIEW, and ZGATE_REVIEWER. The gate remains failed
# until a named visual review passes.
set -u
cd "$(dirname "$0")/../.."
W=${ZDRAW_WEIGHTS:-$HOME/.zdraw/models/Z-Image-Turbo}
ZP=${ZDRAW_ZPACK:-$W/zdraw-w16.zpack}
PY=${MFLUX_PY:-python3}
OUT=${ZGATE_OUT:-runs/zimage-gate-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
echo "zimage gate: commit $(git rev-parse --short HEAD) host $(hostname -s) $(date)" | tee "$OUT/label.txt"
CASES="$OUT/cases.tsv"
printf 'case\tprofile\tsize\tprompt\tseed\timage\tlog\tgenerate_rc\tcontent_rc\n' > "$CASES"

fail=0
for profile in product strict; do
  for size in 512 1024; do
    n=0
    for prompt in \
      "a red fox in deep snow, golden hour light, detailed fur" \
      "editorial portrait of a fashion designer in a bright studio" \
      "a crowded spice market stall, dozens of open sacks, fine texture"; do
      n=$((n + 1))
      name="${profile}_${size}_p${n}"
      ZDRAW_PROGRESS=quiet ZDRAW_ZPACK=$ZP ./zig-out/bin/zdraw generate \
        --profile "$profile" --model z-image-turbo --weights "$W" \
        --prompt "$prompt" --width "$size" --height "$size" --steps 4 \
        --seed 46 --out "$OUT/$name.png" > "$OUT/$name.log" 2>&1
      rc=$?
      content_rc=125
      if [ "$rc" -ne 0 ]; then
        echo "FAIL $name: generate rc=$rc"
        fail=1
      else
        "$PY" tools/quality/content_check.py "$OUT/$name.png"
        content_rc=$?
        if [ "$content_rc" -ne 0 ]; then
          fail=1
        fi
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$profile" "$size" "$prompt" 46 \
        "$OUT/$name.png" "$OUT/$name.log" "$rc" "$content_rc" >> "$CASES"
    done
  done
done

visual=${ZGATE_VISUAL_REVIEW:-not-reviewed}
reviewer=${ZGATE_REVIEWER:-}
python3 tools/quality/zimage_receipt.py \
  --output-dir "$OUT" \
  --cases "$CASES" \
  --weights "$W" \
  --pack "$ZP" \
  --binary ./zig-out/bin/zdraw \
  --visual-review "$visual" \
  --reviewer "$reviewer"
receipt_rc=$?

if [ "$fail" -ne 0 ]; then
  echo "ZIMAGE-GATE FAIL (artifacts in $OUT)"
  exit 1
fi
if [ "$receipt_rc" -ne 0 ]; then
  echo "ZIMAGE-GATE AUTOMATED PASS; visual review remains open ($OUT)"
  exit "$receipt_rc"
fi
echo "ZIMAGE-GATE PASS (12/12 automated and visually reviewed, $OUT)"
