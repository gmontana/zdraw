# Release validation

## Release source, 5 October 2026

The released source carries the memory pass and the Klein base 4B model on top
of the 1 October candidate. On one M4 Max (128 GiB, macOS 26.5.2):

| Check | Result |
| --- | --- |
| Native build, QA and Debug tests | Passed; style and architecture budgets unchanged or lower. |
| CLI smoke | Passed on the native build. |
| GitHub CI | The `hardware-gates` workflow passed on a virtual Apple M1 (macOS 14): build, tools, unit tests, qa, CLI smoke and the gates-only kernel checks, without weights. |
| Fixed benchmark cards | Klein 4B 4/4, Z-Image product 4/4 and Z-Image strict 1/1 matched the certified hashes. Klein base 4B matched in all but two of about thirty renders over 5 and 6 October; the two failures produced the same wrong image (a fox over a collapsed, repeated texture, the look of the undistilled model without guidance) with identical routes and no fallbacks, each straight after other heavy GPU work. Sixteen instrumented renders afterwards (ten at 512², six bench cards at 1024², every step's velocity and latent hashed) all matched with identical intermediate hashes. The defect is real, reproducible in content but not on demand, and its cause is not yet known; `zdraw bench --model flux2-klein-base-4b` detects it. |
| Fixed-case reproducibility | Ten sequential renders each for Klein 4B, Z-Image product and Klein base 4B, one PNG hash each, equal to the certified hash. |
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
