# Testing zdraw

Run these checks on an Apple Silicon Mac. GPU tests and generation must run
separately on an idle device.

## CLI and unit tests

```sh
zig build
python3 tools/cli_smoke.py ./zig-out/bin/zdraw
zig build -Doptimize=Debug test
zig build qa
```

Development scripts require Python 3.11+; the CLI does not. The smoke test uses
synthetic images and needs no weights. QA compiles developer tools and requires
zlint v0.8.1.

## Real-image quality gates

The Klein gate decodes PNGs using the Python standard library. The Z-Image
content checker needs NumPy and Pillow; install them in a separate environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy pillow
```

After fetching the relevant models and building the engine:

```sh
python tools/quality/klein_gate.py \
  --weights "$HOME/.zdraw/models/FLUX.2-klein-4B" \
  --zpack "$HOME/.zdraw/models/FLUX.2-klein-4B/zdraw-klein-w16.zpack" \
  --include-1024 --no-build
MFLUX_PY=python bash tools/quality/zimage_gate.sh
```

Both gates write images, logs and a manifest below `runs/`. Review the images
for blank areas, checkerboard corruption and changes to the intended content.
The Z-Image gate intentionally returns a nonzero result until visual review is
recorded. After inspecting its images, use the printed output directory:

```sh
python tools/quality/zimage_receipt.py \
  --output-dir runs/<gate-directory> --cases runs/<gate-directory>/cases.tsv \
  --weights "$HOME/.zdraw/models/Z-Image-Turbo" \
  --pack "$HOME/.zdraw/models/Z-Image-Turbo/zdraw-w16.zpack" \
  --binary ./zig-out/bin/zdraw --visual-review pass --reviewer "your name"
```

For a ten-render census of the fixed Klein case:

```sh
MFLUX_PY=python bash tools/quality/repro_census.sh \
  "$HOME/.zdraw/models/FLUX.2-klein-4B" \
  "$HOME/.zdraw/models/FLUX.2-klein-4B/zdraw-klein-w16.zpack" \
  flux2-klein-4b - 10
```

The census fails for missing/failed renders, hashing errors or failed content
checks. Multiple hashes do not change its exit code: inspect the completed count
and histogram. Its regression tests run in QA. Performance changes also follow
the [measurement protocol](measurement-protocol.md).

## Progressive terminal display

After fetching the default Klein W16 model, run the real-model terminal check
on an idle Mac (Python with Pillow is required for pixel checks):

```sh
python3 tools/cli_progress_smoke.py ./zig-out/bin/zdraw \
  --out-dir runs/progressive-check
```

Use a new output directory; `--weights DIR` and `--zpack FILE` select existing
weights. Six real generations check iTerm/Kitty/ANSI, final-only/no-show/redirected
output, terminal restoration, final-image visibility and identical pixels.

## Resolution coverage

The [sweep](resolutions.md) uses three timed fox renders per shape, plus portrait
and market cases. It retains all images and logs, including failures.

After fetching both default W16 models, use the NumPy/Pillow environment above:

```sh
python tools/quality/test_resolution_sweep.py
python tools/quality/resolution_sweep.py --mode quality \
  --out-dir runs/resolution-quality
bash tools/perf_when_quiet.sh runs/resolution-quiet.log -- \
  python tools/quality/resolution_sweep.py --mode timing \
    --out-dir runs/resolution-timing
```

Use new output directories. Timing mode warms each model once, then
reverses/rotates case order over three rounds. It clears inherited experiment
flags and records the environment, binaries, weights and output hashes.
Quality-mode times are diagnostic only.

Both modes require visual review. Nonzero results preserve failures and content
flags; small sharp/textured images can trigger the corruption heuristic. Keep
those flags alongside the review rather than rewriting the run as a blanket pass.

## Comparisons

For a fixed case on your Mac:

```sh
./zig-out/bin/zdraw version
./zig-out/bin/zdraw doctor --json
./zig-out/bin/zdraw bench --model flux2-klein-4b
```

`tools/competitive_bench.py --help` lists options for comparisons with separately
installed engines. Follow the [measurement protocol](measurement-protocol.md):
match workloads, interleave runs and use the same external memory instrument.
Record the protocol and result in [experiments.jsonl](experiments.jsonl).

`python3 tools/competitive_bench.py --help` lists comparator settings. MFLUX and
diffusers need separate environments (`--mflux-python`, `--diffusers-python`);
other engines need their own installations. Missing engines remain unavailable.
Follow the [measurement protocol](measurement-protocol.md).
