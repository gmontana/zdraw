# Contributing

Bug reports, fixes, documentation and results from other Macs are welcome.
Keep pull requests focused and describe the change, tests and hardware used.

## Build and test

Use Zig 0.16.0, an Apple Silicon Mac and Xcode's Metal compiler.
QA also requires zlint v0.8.1.

```sh
zig build
python3 tools/cli_smoke.py ./zig-out/bin/zdraw
zig build -Doptimize=Debug test
zig build qa
```

Run GPU tests on an idle device, separately from generation. The CLI smoke test
needs no model weights. [Testing](docs/testing.md) covers real-image checks,
Python dependencies and reproduction commands. Do not raise QA budgets to hide
regressions.

## Working on the engine

`src/` contains the CLI, model runtimes, Metal bridge and unit tests; `tools/`
contains quality checks and developer scripts. Root Zig tools handle packing,
reference comparisons and benchmarks. Third-party kernels live in `vendor/`.

Read the [architecture contract](docs/architecture.md) before changing component
boundaries or resource ownership. Engine changes need both model quality gates
and visual review. Performance work follows the
[measurement protocol](docs/measurement-protocol.md): register the experiment
before running it and record negative results as well as improvements.

For bug reports, include the command, `zdraw version` and `zdraw doctor --json`.
Use the benchmark-card issue template to share results from your Mac.

## Licensing

Contributions use [Apache-2.0](LICENSE). You retain your copyright; no separate
contributor licence agreement is required. Use `git commit -s` to certify the
[Developer Certificate of Origin](https://developercertificate.org/).
Preserve third-party licences and add a NOTICE entry for new vendored code.

## Unsafe operations

Keep pointer casts, FFI buffers and allocator ownership inside the documented
component boundaries. A new unsafe builtin needs a nearby `// SAFETY:` comment
that states the concrete reason the operation is valid: allocation size,
alignment, lifetime or representation. `python3 tools/style/zig_guard.py
--check-diff --staged` checks new staged sites; the warnings the main guard
prints for existing sites are inherited debt, not evidence of a defect.
