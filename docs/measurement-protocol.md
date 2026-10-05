# Measurement protocol

A speed, memory or quality claim needs a recorded workload, exact engine and
compiler revisions, model/pack precision, machine, operating system and date.
Historical entries in [experiments.jsonl](experiments.jsonl) are not validation of
the current release candidate.

## Register and control the experiment

Before measuring, add the hypothesis, expected effect and its evidence,
procedure, success thresholds and rejection thresholds to
[experiments.jsonl](experiments.jsonl). Record negative results too.

Take timings only after six consecutive one-minute load-average readings below
2.0, sampled 80 seconds apart. `tools/perf_when_quiet.sh` records those readings
and runs a supplied command after the gate passes:

```sh
bash tools/perf_when_quiet.sh runs/quiet.log -- \
  ./zig-out/bin/zdraw bench --model flux2-klein-4b
```

The quiet check alone does not make a benchmark comparable or certify an image.
Record thermal state and avoid concurrent GPU work. If the machine cannot stay
quiet, report that limitation and do not promote timing claims from the run.

## Compare like workloads in the same session

Use A/B/B/A ordering across repeated rounds. Thermal drift can exceed the effect
being measured. Never compare a freshly measured candidate with an old baseline
or a number from another machine.

State the regime: cold process (one render per process), warm session (loaded
weights reused), or cold page cache (weights read from storage). Prompt caching
and startup overhead differ between engines. For encoder changes, use fresh
prompts; repeated generation of one prompt may skip encoding entirely.

Use `/usr/bin/time -l` for both arms and report both maximum resident set size
and peak physical footprint. File mappings and copied weights are charged
differently; these figures do not establish total RAM requirements. A claim
about running on a smaller-memory Mac needs a test on that hardware.

Record models, precision, dimensions, steps, seeds and prompts for every arm.
Matched dimensions and steps do not establish equivalent image quality.
Unavailable or failed comparator arms stay visible in the report.

## Prove execution and quality

Record dispatch, command-buffer and readback counters (`ZDRAW_METRICS=1`) and
verify that the intended route engaged. Identical hashes alone cannot prove
that a numerical experiment ran. Isolated kernel benchmarks do not establish
an end-to-end improvement; measure the operation inside the actual engine.

After engine changes, run the Z-Image and Klein quality gates, including 1024px
cases. Inspect images from both arms. Content checks catch common blank or
corrupt outputs but are not a substitute for visual review.

A determinism claim needs at least ten renders of the same case on a named
host, with the number of distinct hashes reported. Use
`tools/quality/repro_census.sh`; one matching render is insufficient.
The command fails for incomplete renders, hashing errors or failed content
checks. A successful exit does not require a single hash; inspect the reported
completed count and histogram before making a determinism claim.

An accepted numerical change may alter the certified hash. Use a CPU/tensor
oracle and the preregistered tolerances, compare product output with its strict
reference where appropriate, run the full image gates and census, then update
reference hashes explicitly. PSNR against an old full-image hash alone is not
a valid gate for accumulated denoising-rounding changes.
