---
name: Measurement report
about: Speed, memory, or reproducibility numbers from your machine
---

Reports from machines other than the M4 Max used for the published numbers
are the most useful thing you can send. Follow `docs/measurement-protocol.md`.

**Machine**: chip, GPU family, unified memory, macOS version

**zdraw commit and binary sha256**:

**Quiet**: six consecutive 1-minute load averages before the run (or
`tools/perf_when_quiet.sh` output)

**Commands** (exact) and per-render `/usr/bin/time -l` max RSS and
peak footprint:

**Hashes**: `tools/quality/repro_census.sh` output (n, distinct hashes)

**Content check**: `tools/quality/content_check.py` scores

**Comparison engine, if any**: version, same-session, interleaved order
