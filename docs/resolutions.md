# Resolution, quality and runtime

M4 Max, 128 GiB RAM, macOS 26.5.2. Timing and memory were measured on 5 October 2026
with the released engine; the quality review and sample images are from the
1 October 2026 sweep of the same shapes. Both models use
W16 weights, four steps and seed 46, with the prompt “a red fox sitting in deep
snow, golden hour light”.

## Runtime and memory

Each row contains **three fresh processes with cached weight pages**. Time covers
loading, encoding, sampling, decoding, PNG saving and cleanup. Parentheses show
min–max; memory columns are maxima across the runs. GiB means 2³⁰ bytes.

### Klein 4B

| Dimensions | Median seconds (range) | Peak RSS (GiB) | Peak footprint (GiB) | Output review |
| --- | ---: | ---: | ---: | --- |
| 64×64 | — | — | — | Unsupported |
| 128×128 | 1.95 (1.94–1.97) | 0.53 | 1.14 | Thumbnail / smoke |
| 256×256 | 2.32 (2.31–2.37) | 0.71 | 1.27 | Reviewed; heuristic flag |
| 512×512 | 4.37 (4.29–4.51) | 1.03 | 1.61 | Reviewed |
| 768×768 | 7.13 (6.95–7.48) | 1.58 | 2.21 | Reviewed |
| 1024×1024 | 11.96 (11.95–13.20) | 2.33 | 3.02 | Reviewed |
| 1536×1536 | — | — | — | Unsupported |
| 2048×2048 | — | — | — | Unsupported |
| 1024×768 | 9.64 (8.67–9.77) | 1.90 | 2.55 | Reviewed |
| 768×1024 | 9.68 (8.69–9.90) | 1.90 | 2.55 | Reviewed |
| 544×800 | — | — | — | Unsupported |

### Z-Image-Turbo

| Dimensions | Median seconds (range) | Peak RSS (GiB) | Peak footprint (GiB) | Output review |
| --- | ---: | ---: | ---: | --- |
| 64×64 | 1.22 (1.20–1.22) | 0.27 | 0.91 | Thumbnail / smoke |
| 128×128 | 1.46 (1.45–1.47) | 0.41 | 1.02 | Thumbnail / smoke |
| 256×256 | 2.16 (2.15–2.23) | 0.65 | 1.23 | Reviewed; heuristic flag |
| 512×512 | 5.73 (5.72–6.14) | 1.30 | 1.91 | Reviewed |
| 768×768 | 12.41 (12.15–13.44) | 2.41 | 3.10 | Reviewed |
| 1024×1024 | 24.52 (21.76–25.37) | 3.95 | 4.72 | Reviewed |
| 1536×1536 | 63.06 (55.16–64.54) | 8.32 | 9.36 | Reviewed |
| 2048×2048 | 127.16 (124.79–134.72) | 14.48 | 15.87 | Reviewed |
| 1024×768 | 18.07 (15.91–18.52) | 3.06 | 3.78 | Reviewed |
| 768×1024 | 18.06 (16.68–18.47) | 3.06 | 3.79 | Reviewed |
| 544×800 | 31.95 (31.68–32.16) | 2.26 | 2.48 | Reviewed |

These measurements apply to this prompt and regime. Shape matters: 544×800
Z-Image was slower than 1024×1024. Against the 1 October run of the same
protocol, the pool changes released since then lowered peak memory at every
resolution from 256×256 up (Klein 1024×1024 footprint 4.57 → 3.02 GiB, Z-Image
5.94 → 4.72 GiB, Z-Image 2048×2048 20.75 → 15.87 GiB); the 64 and 128 px rows
and Z-Image 544×800 are unchanged. The two runs are days apart, not an
interleaved A/B, so their times are reported side by side, not compared. RSS
and footprint are **not minimum Mac RAM**; clean mapped weight pages can occupy
additional memory. Smaller-memory Macs have not been tested.

### What RSS and footprint leave out

`ZDRAW_MEMTRACE=1` now also prints `mapped_res`, the resident pages of the
mapped weight files (via `mincore`), and `total = footprint + mapped_res`.
With the W16 Klein pack the whole 6.9 GiB file is resident from the first
denoise step (the read-ahead hint pulls it in), so a Klein 1024×1024 render
holds about **10.3 GiB** in total and a Z-Image product render about
**28.5 GiB**, against footprints of 3.0 and 4.7 GiB. The `bench --card`
output reports the peak of both as `mapped` and `total`. `footprint -p`
agrees with the footprint column within 3% and, like RSS, never counts the
mapped pages.

### Where the 128×128 floor goes (Klein, W16, 5 October 2026)

| Stage | Footprint (GiB) | Added | What it is |
| --- | ---: | ---: | --- |
| after runtime init | 0.07 | 0.07 | Metal device, pipelines, tables |
| after text encoding | 0.34 | 0.28 | resident text-encoder pools and work |
| denoise (first and last step) | 0.52 | 0.18 | DiT pool at 64 tokens plus the static attention scratch |
| VAE mid block before attention | 0.55 | 0.02 | |
| VAE mid attention | 1.05 | 0.50 | mid-attention scratch, sized independently of the output resolution |
| VAE up stages and finish | 1.14 | 0.09 | |

Toggles: `ZDRAW_VAE_WINO=0` and `ZDRAW_KLEIN_GEMM_MPP=0` leave the floor
unchanged (1.14); `ZDRAW_KLEIN_TE_RES=0` (the per-op text route) raises it
to 2.85 and `ZDRAW_KLEIN_RES=0` (the non-resident DiT) to 8.6, so the
resident routes are the lean ones. The half-gigabyte mid-attention scratch
and the text-encoder pools are the floor's two movable parts
(recorded in [experiments.jsonl](experiments.jsonl) as `memory-floor-20261005`).

## Quality and supported dimensions

The sweep also used portrait and market prompts on a second M4 Max / 128 GiB Mac
(macOS 26.6.2). Their times are excluded from the table.

- **Klein:** sides divisible by 32, area divisible by 8192, at most 1,048,576
  pixels. The four unsupported shapes above account for all 20 failed attempts.
  The CLI now rejects them before loading weights.
- **Z-Image:** all eleven tested shapes completed, including 2048×2048.
- **Images:** 92 outputs, 54 distinct pixel hashes, all visually inspected
  ([review record](evidence/20261001/resolutions/visual-review.json)). Nine small-image corruption warnings remain in the evidence;
  sharp edges and dense texture triggered the larger-image heuristic. Review
  found no obvious blank/checkerboard corruption. Detail at 64/128px is limited.
- **References:** every timed 1024px image, including warmups, matched its model's
  reference hash. Each successful timing configuration had one hash across
  three runs.

This is a three-prompt rendering screen, not a broad model-quality assessment.
It does not cover every legal shape, editing, strict profiles or other step counts.
The overall automated sweep remains nonzero because failures and flags are retained.

### Sample images

Sheets show the first timed fox and both additional prompts. Small images are
enlarged; larger images are reduced to fit. Failed cases are labelled.

| Model | 64–512 squares | 768–2048 squares | Rectangles |
| --- | --- | --- | --- |
| Klein 4B | [View](evidence/20261001/resolutions/flux2-klein-4b-small.png) | [View](evidence/20261001/resolutions/flux2-klein-4b-large.png) | [View](evidence/20261001/resolutions/flux2-klein-4b-rectangles.png) |
| Z-Image-Turbo | [View](evidence/20261001/resolutions/z-image-turbo-small.png) | [View](evidence/20261001/resolutions/z-image-turbo-large.png) | [View](evidence/20261001/resolutions/z-image-turbo-rectangles.png) |

## Method and evidence

Timing (5 October): the released engine source and the shipped
`tools/quality/resolution_sweep.py`; a [quiet-host gate](evidence/20261005/resolutions/quiet.txt)
passed six load readings below 2, 80 seconds apart; the [timing manifest](evidence/20261005/resolutions/timing.json)
retains every attempt with its wall time, memory, content checks and certified-hash
result (all 1024×1024 renders matched). Quality review (1 October): a
[quiet-host gate](evidence/20261001/resolutions/quiet.txt) passed the same test. One warmup per model preceded
three rounds with reversed/rotated case order; no other GPU work ran concurrently.
An external monotonic timer wrapped `/usr/bin/time -l` for wall time and memory.

The later input fixes change validation, not inference.
[Archive checks](evidence/20261001/resolutions/admission-validation.json) confirm
unchanged reference pixels. [Testing](testing.md#resolution-coverage) has commands.

The 1 October [summary](evidence/20261001/resolutions/summary.json),
[timing manifest](evidence/20261001/resolutions/m4max-a-timing.json) and
[quality manifest](evidence/20261001/resolutions/m4max-b-quality.json) retain every
attempt, log, image hash and review finding. Original PNGs are retained in the
run artifacts; the sheets above are the published image set. Protocol and results
are recorded in [experiments.jsonl](experiments.jsonl).
