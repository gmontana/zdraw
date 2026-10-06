# Comparisons with other engines

Four engines rendered the same request on one Mac in one session: zDraw,
[mflux](https://github.com/filipstrand/mflux) (MLX, Python),
[diffusers](https://github.com/huggingface/diffusers) on PyTorch's MPS
backend, and [iris.c](https://github.com/antirez/iris.c) (pure C with Metal).
Draw Things was installed but is not in the table: it had no models
downloaded and its HTTP API was off, and it cannot be driven unattended
without both.

## Protocol

M4 Max, 128 GiB, macOS 26.5.2; night of 5 to 6 October 2026, with the
[quiet-host gate](evidence/20261006/comparisons/quiet-klein.txt) passed before
the first render. Every engine rendered the same prompt ("a red fox in deep
snow, golden hour light") at 1024×1024, four steps, seed 7, with its own
default sampler. Two regimes are reported and never pooled:

- **cold process**: one render per fresh process, including model loading,
  three processes after one warmup;
- **warm session**: one process rendering three times after a warmup
  (zDraw's interactive session, iris.c's session, an in-process loop for
  mflux and diffusers).

Memory is the peak footprint of the engine's process from `/usr/bin/time -l`,
the same instrument for every row. It does not count clean file-backed pages,
which matters for zDraw alone: zDraw maps its weight file and never copies it,
so the 6.9 GiB Klein pack and the 22.1 GiB Z-Image pack live in the page cache
outside its footprint, while the other engines load weights into process
memory and show them in theirs. The last column adds the pack size to zDraw's
footprint for a like-for-like upper bound.

Each table was run twice with the engine order reversed; both medians are
shown as "first order / reversed". The engine that runs last in a session
tends to be a little slower, which is the spread between the two.

Versions: zDraw 0.1.0 (W16 pack, `product` profile); mflux 0.18.0 running its
unquantised bf16 weights, plus a Klein arm at its 8-bit load-time
quantisation; diffusers 0.39.0 on torch 2.9.1 (bf16, no quantisation); iris.c
at its 13 February 2026 head, built with `make mps`, on the weights its own
downloader fetches.

## FLUX.2 Klein 4B

| engine | cold process, median s | warm session, median s | peak footprint GiB | with weights counted |
| --- | ---: | ---: | ---: | ---: |
| zDraw | 11.81 / 12.89 | 11.00 / 12.00 | 3.02 | 9.9 |
| mflux (bf16) | 13.41 / 13.45 | 10.94 / 11.41 | 36.4 | 36.4 |
| mflux (8-bit) | 16.33 | 14.47 | 35.3 | 35.3 |
| iris.c | 13.96 / 14.58 | 12.31 / 12.80 | 29.7 | 29.7 |
| diffusers (MPS) | 20.40 / 20.43 | 14.88 / 15.76 | 26.7 | 26.7 |

mflux's 8-bit arm is slower than its bf16 arm and no smaller at the peak,
because it quantises at load time from weights it has already read into
memory; the single run shown is from the first order.

## Z-Image-Turbo

| engine | cold process, median s | warm session, median s | peak footprint GiB | with weights counted |
| --- | ---: | ---: | ---: | ---: |
| zDraw | 24.61 / 24.18 | 23.80 / 23.10 | 4.72 | 26.8 |
| mflux (bf16) | 27.22 / 26.77 | failed / 22.54 | 56.5 | 56.5 |
| diffusers (MPS) | 28.01 / 28.06 | 22.03 / 21.92 | 35.1 | 35.1 |
| iris.c (own pass) | 35.62 | 21.93 | 61.3 | 61.3 |

mflux's first warm-session attempt failed while Hugging Face closed the
connection mid-download of a 4 GB shard; with the weights cached, the reversed
order ran, and a later mflux-only rerun gave 23.46 s warm and 26.83 s cold.
iris.c failed to load the Z-Image VAE in both interleaved passes; an hour later
the same binary, weights and command ran, so its row comes from a pass of its
own after the others, and the cause of the earlier failure was not found. Its
cold time includes loading 61 GiB into process memory.

## What the numbers say, and do not say

On Klein, zDraw, mflux and iris.c are within about two seconds of each other
per image; diffusers is the slowest cold and competitive warm. On Z-Image,
zDraw's cold render is two to eleven seconds faster than the others and its
warm render is one to two seconds slower than mflux, diffusers and iris.c. Speed is
not the separation. Memory is: zDraw's process holds 3 GiB for Klein and
under 5 GiB for Z-Image against 27 to 61 GiB, and even with the whole weight
file counted it stays below every other engine. Two design decisions produce
that, both described in [How zDraw works](how-it-works.md): weights are
mapped and never copied, and activations live in reused pools.

The tables compare time and memory only. The engines do not produce the same
image from the same seed (their samplers and noise differ), so quality is not
compared here; every output passed the harness's blank and noise checks.
Settings were matched, not tuned: each engine ran four steps at the models'
default guidance. A session of this kind measures one machine on one night;
the [measurement protocol](measurement-protocol.md) explains why ratios are
quoted only from the same session.

## Reproduce

```sh
python3 tools/competitive_bench.py inventory
python3 tools/competitive_bench.py run --model flux2-klein-4b \
  --engines zdraw,mflux,diffusers,iris --mflux-quantize 0 \
  --klein-weights ~/.zdraw/models/FLUX.2-klein-4B \
  --zdraw-zpack ~/.zdraw/models/FLUX.2-klein-4B/zdraw-klein-w16.zpack \
  --iris-bin ~/iris.c/iris --iris-weights ~/iris.c/flux-klein-4b
```

Run it a second time with the engine order reversed. The harness writes
`summary.md`, `summary.csv` and `results.json` per run; the five runs behind
the tables are in
[evidence/20261006/comparisons](evidence/20261006/comparisons/) with their
per-run times, memory and commands.
