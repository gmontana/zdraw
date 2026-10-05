#!/bin/bash
# Solo-render reproducibility census: N renders of one prompt and seed, one
# process at a time, then a sha256 histogram. The mechanism that produced the
# m4-baseline-census and m4-strict-census rows, committed so the next census
# is a command rather than a session.
#
# Usage:
#   repro_census.sh <weights_dir> <zpack> <model> <profile|-> <n> [ENV=value ...]
#     model    z-image-turbo | flux2-klein-4b
#     profile  product | strict for Z-Image; "-" for Klein (no --profile)
#     ENV=..   extra environment for every render (e.g. ZDRAW_KLEIN_ACT=f32)
# Env: PROMPT, SIZE (1024), STEPS (4), SEED (46), MFLUX_PY (content check).
# Output: runs/repro-census/<model>-<profile>-<stamp>/ with every PNG and log,
#   sha256.txt, census.txt (count per hash), counters.txt (first run's Metal
#   dispatch / command buffer / GPU readback lines), load-evidence.txt.
# Prints "CENSUS n=N completed=M distinct=K". Incomplete renders or failed
# content checks exit nonzero. Multiple hashes are reported, not rejected.
set -euo pipefail
cd "$(dirname "$0")/../.."
W="${1:?weights dir}"
ZP="${2:?zpack path}"
MODEL="${3:?model name}"
PROFILE="${4:?profile or -}"
N="${5:?n}"
shift 5

[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "n must be a positive decimal integer (no leading zeros)" >&2; exit 1; }
BIN=./zig-out/bin/zdraw
PROMPT=${PROMPT:-"a red fox sitting in deep snow, golden hour light"}
SIZE=${SIZE:-1024}
STEPS=${STEPS:-4}
SEED=${SEED:-46}
PY=${MFLUX_PY:-python3}
[ -x "$BIN" ] || { echo "build $BIN before running the census" >&2; exit 1; }
command -v "$PY" >/dev/null 2>&1 || { echo "content-check Python not found: $PY (set MFLUX_PY)" >&2; exit 1; }
[ -f tools/quality/content_check.py ] || { echo "tools/quality/content_check.py missing" >&2; exit 1; }

mkdir -p runs/repro-census
OUT=$(mktemp -d "runs/repro-census/${MODEL}-${PROFILE}-$(date +%Y%m%d-%H%M%S)-XXXXXX")

{
  date; uptime; hostname -s
  echo "commit $(git rev-parse --short HEAD)"
  shasum -a 256 "$BIN"
  echo "extra env: $*"
  ps aux | sort -k3 -rn | sed -n '1,6p'
} > "$OUT/load-evidence.txt"

case "$MODEL" in
  z-image-turbo) zpack_var=ZDRAW_ZPACK ;;
  *) zpack_var=ZDRAW_KLEIN_ZPACK ;;
esac
profile_args=()
if [ "$PROFILE" != "-" ]; then
  profile_args=(--profile "$PROFILE")
fi

fail=0
images=()
for ((i = 1; i <= N; i++)); do
  if env ZDRAW_PROGRESS=quiet ZDRAW_METRICS=1 "$zpack_var=$ZP" "$@" \
    "$BIN" generate --model "$MODEL" --weights "$W" ${profile_args[@]+"${profile_args[@]}"} \
    --prompt "$PROMPT" --width "$SIZE" --height "$SIZE" --steps "$STEPS" \
    --seed "$SEED" --out "$OUT/r$i.png" > "$OUT/r$i.log" 2>&1; then
    if [ -s "$OUT/r$i.png" ]; then
      images+=("$OUT/r$i.png")
    else
      echo "render $i produced no image (see $OUT/r$i.log)"
      fail=1
    fi
  else
    rc=$?
    echo "render $i failed rc=$rc (see $OUT/r$i.log)"
    fail=1
  fi
done

# Only successful renders belong in the histogram; failed commands may leave
# a partial image behind. Never glob files from failed or earlier runs.
: > "$OUT/sha256.txt"
if [ "${#images[@]}" -gt 0 ]; then
  shasum -a 256 "${images[@]}" > "$OUT/sha256.txt"
  if ! "$PY" tools/quality/content_check.py "${images[@]}" > "$OUT/content.txt" 2>&1; then
    echo "content_check failed (see $OUT/content.txt)"
    fail=1
  fi
fi
awk '{print $1}' "$OUT/sha256.txt" | sort | uniq -c | sort -rn > "$OUT/census.txt"
grep -hE "Metal dispatches|command buffers|GPU readbacks" "$OUT/r1.log" > "$OUT/counters.txt" 2>/dev/null || true

distinct=$(wc -l < "$OUT/census.txt" | tr -d ' ')
if [ "${#images[@]}" != "$N" ]; then
  fail=1
fi
cat "$OUT/census.txt"
cat "$OUT/counters.txt" 2>/dev/null
echo "CENSUS n=$N completed=${#images[@]} distinct=$distinct model=$MODEL profile=$PROFILE out=$OUT"
exit "$fail"
