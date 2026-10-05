#!/usr/bin/env bash
# Minimum-RAM sweep (ledger memory-ladder-w0-instrument-20261006).
# For each free-RAM target M (GiB), wire hw.memsize - M with tools/ballast,
# render the certified card (timing-clean), and record wall time,
# hash match, footprint, mapped-resident and total bytes. "Minimum RAM" is the
# smallest M whose card MATCHes within +10% of the unballasted median wall time.
#
#   tools/ballast_sweep.sh --model flux2-klein-4b --weights DIR --out runs/ballast 16 12 8 6 4 3 2
#
# One zdraw process at a time; run on a quiet box (tools/perf_when_quiet.sh).
set -euo pipefail
model=flux2-klein-4b; weights=""; out="runs/ballast"; extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --weights) weights="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ -n "$weights" ] || { echo "--weights DIR required" >&2; exit 2; }
targets=("$@")
[ ${#targets[@]} -gt 0 ] || { echo "give free-RAM targets in GiB" >&2; exit 2; }
mkdir -p "$out"
here="$(cd "$(dirname "$0")" && pwd)"
ballast="$here/ballast/ballast"
[ -x "$ballast" ] || cc -O2 -o "$ballast" "$here/ballast/ballast.c"
mem_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
zdraw="${ZDRAW_BIN:-./zig-out/bin/zdraw}"

run_card() {  # $1 = label; timing-clean card (the always-on strided sampler
    # fills mapped_resident_gb/total_gb); a watchdog kills a thrashing render.
    local log="$out/card-$1.txt"
    "$zdraw" bench --model "$model" --weights "$weights" --card >"$log" 2>&1 &  # the card is the fixed 1024 census case
    local cpid=$! waited=0
    while kill -0 "$cpid" 2>/dev/null && [ "$waited" -lt "${CARD_TIMEOUT:-900}" ]; do sleep 2; waited=$((waited + 2)); done
    if kill -0 "$cpid" 2>/dev/null; then kill "$cpid" 2>/dev/null; echo "card $1: killed after ${waited}s" >>"$log"; fi
    wait "$cpid" 2>/dev/null || true
    local wall match foot mapped total
    wall=$(grep -o '"wall_s": *[0-9.]*' "$log" | head -1 | grep -o '[0-9.]*$' || echo nan)
    match=$(grep -o 'MATCH (certified)\|MISMATCH' "$log" | head -1 || echo none)
    foot=$(grep -o '"footprint_gb": *[0-9.]*' "$log" | head -1 | grep -o '[0-9.]*$' || echo nan)
    mapped=$(grep -o '"mapped_resident_gb": *[0-9.]*' "$log" | head -1 | grep -o '[0-9.]*$' || echo nan)
    total=$(grep -o '"total_gb": *[0-9.]*' "$log" | head -1 | grep -o '[0-9.]*$' || echo nan)
    echo "$1 wall_s=$wall hash=$match footprint_gb=$foot mapped_resident_gb=$mapped total_gb=$total"
}

echo "host mem ${mem_gib} GiB; model $model; the certified 1024 card" | tee "$out/sweep.txt"
run_card unballasted | tee -a "$out/sweep.txt"
for m in "${targets[@]}"; do
    wire=$(( mem_gib - m ))
    [ "$wire" -gt 0 ] || { echo "free target $m >= host memory" | tee -a "$out/sweep.txt"; continue; }
    "$ballast" "$wire" >"$out/ballast-$m.txt" &
    bpid=$!
    waited=0  # wiring 100+ GiB takes a while; render only once the ballast reports
    until grep -q wired "$out/ballast-$m.txt" || [ "$waited" -ge 600 ] || ! kill -0 "$bpid" 2>/dev/null; do sleep 2; waited=$((waited + 2)); done
    cat "$out/ballast-$m.txt" | tee -a "$out/sweep.txt"
    # Two renders per point: wiring the ballast evicts the weight file from the
    # page cache, so the first render is a cold read whatever M is; the second
    # shows whether the weights can stay cached beside the ballast (steady state).
    run_card "free${m}g-cold" | tee -a "$out/sweep.txt"
    run_card "free${m}g-warm" | tee -a "$out/sweep.txt"
    kill "$bpid" 2>/dev/null || true
    wait "$bpid" 2>/dev/null || true
    sleep 3
done
echo "done; minimum RAM = smallest free target whose warm render MATCHes within 1.10 x the unballasted wall" | tee -a "$out/sweep.txt"
