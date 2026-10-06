# Release validation

## Base-model defect, 6 October 2026

The 0.1.0 release recorded one specific wrong Klein base 4B image, seen twice
in about thirty renders, not reproducible on demand and detected by the bench
card. A day of diagnosis found the cause and it was not the engine's
arithmetic.

The wrong image appeared whenever the base model was rendered by a process
whose environment carried `ZDRAW_KLEIN_ZPACK` pointing at the distilled
model's pack, which the validation harness exports for the Klein card and
census. The engine accepted that sidecar for the base model because every
tensor shape matched, so `swapLoaded` replaced the base model's block
linears, embedders and output projection with the distilled model's while the
modulation, time and final-norm weights stayed the base model's own. The
result was deterministic: byte-identical wrong images on two different Macs,
diverging from the certified render at the very first GEMM by the small
distance between two closely related checkpoints. The evidence chain, with
per-block activation hashes of good and wrong forwards, is kept in the private
ledger; what mattered for the fix is that the text embeddings, noise, rope
table, time embedding and modulation vectors were identical in both states and
the embedder weights as bound by the GPU were not.

The fix verifies a sidecar against the checkpoint it is loaded with and
refuses a foreign one with a message instead of rendering. New packs carry a
checkpoint identity; older packs are checked through their embedder's f16
image. Validation of 0.1.1 on the development Mac: the distilled pack offered
to the base model and the base pack offered to the distilled model are both
refused with no image written; the matching pack through the override still
renders the certified hash; Klein, Z-Image product, Z-Image strict and Klein
base cards match; the Klein and Z-Image product censuses are 10/10 with the
certified hash, and three base renders of the certified 50-step case give it
too; the Debug test suite and the quality gate pass, and the shader
validation layer reports only the Winograd kernel noted below. On a
second Mac, the exact sequence that produced the
wrong image four times out of four now refuses, and the base card without the
override matches.

Findings from the same day are fixed alongside, all byte-identical on every
certified card and census: ten kernels (the row attention kernel, seven VAE
normalisation kernels, two Z-Image residual-norm kernels) reused a reduction
slot without a barrier after every thread had read it, and the row attention
kernel's static score array, which Metal's shader validation layer rejected,
is now host-sized threadgroup memory. The validation layer reports twice the
declared static threadgroup memory of several remaining kernels (the VAE
Winograd GEMM among them) against the 32 KiB limit; those kernels declare at
most the limit, render bit-identically, and are left as they are.

## Release source, 5 October 2026

The released source carries the memory pass and the Klein base 4B model on top
of the 1 October candidate. On one M4 Max (128 GiB, macOS 26.5.2):

| Check | Result |
| --- | --- |
| Native build, QA and Debug tests | Passed; style and architecture budgets unchanged or lower. |
| CLI smoke | Passed on the native build. |
| GitHub CI | The `hardware-gates` workflow passed on a virtual Apple M1 (macOS 14): build, tools, unit tests, qa, CLI smoke and the gates-only kernel checks, without weights. |
| Fixed benchmark cards | Klein 4B 4/4, Z-Image product 4/4 and Z-Image strict 1/1 matched the certified hashes. Klein base 4B matched in all but two of about thirty renders; the two wrong renders came from a harness environment that pointed `ZDRAW_KLEIN_ZPACK` at the distilled pack, which 0.1.0 accepted for the base model (see the 6 October section). |
| Fixed-case reproducibility | Ten sequential renders each for Klein 4B, Z-Image product and Klein base 4B, one PNG hash each; equal to the certified hash for Klein 4B and Z-Image product, while the base census renders the 4-step case and so checks reproducibility only. |
| Timing sweep | Repeated under the 1 October protocol; memory fell at every resolution from 256×256 up; times are reported, not compared (the runs are not an A/B). [Full results](resolutions.md). |

The image-quality gates, portable archive and second-Mac checks were not
repeated for this source; the pool changes are byte-identical by construction
and the cards and censuses above are the evidence for that.

## Release candidate, 30 September to 1 October 2026

Tested on two M4 Max Macs with 128 GiB RAM, running macOS 26.5.2 and 26.6.2,
on 30 September–1 October 2026.

| Check | Result |
| --- | --- |
| Native build, QA and Debug/GPU tests | Passed on both Macs. GPU tests ran separately from generation. |
| CLI and terminal behaviour | Help, errors, completion, saving, terminal restoration and progressive display passed. |
| Model workflows | Download/resume, packing, generation, instruction/masked edits, batches and session model switching passed. |
| Image-quality gates | Klein 8/8 and Z-Image 12/12 product/strict cases passed, including visual inspection and 1024px cases. |
| Fixed-case reproducibility | Ten sequential renders per model, one PNG hash each, matching the references. |
| Portable archive | Relocated CLI checks and both model reference images passed on both Macs. |
| Resolution sweep | 92 images from 112 attempts; 20 failures on unsupported Klein shapes. [Full results](resolutions.md). |

The resolution sweep's 54 distinct images were visually inspected. Nine
small-image heuristic warnings remain in the record; review found no obvious
blank/checkerboard corruption. This is a limited rendering check, not a general
measure of model quality.

The [progressive-display validation](evidence/20261001/progressive-validation.json)
and the [final archive checks](evidence/20261001/resolutions/admission-validation.json)
are the records behind this table.

## Coverage limits

Smaller-memory Macs, other chips and older macOS runtime versions are untested.
The macOS 14 / Apple M1 build target does not establish runtime compatibility.
Signing, notarization and hosted release CI have not been exercised.
Resolution timings cover default W16 text-to-image at four steps; they do not
cover every prompt, dimension, editing mode or precision profile.
