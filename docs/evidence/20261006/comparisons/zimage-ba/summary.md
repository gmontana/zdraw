# Competitive benchmark

- model: `z-image-turbo`
- tier: `product`
- settings: `matched`
- workload: 1024×1024, requested 4 steps, seed 7
- runs: 3; warmups: 1

| engine | protocol | steps | status | median s | peak footprint GiB | memory scope | output |
|---|---|---:|---|---:|---:|---|---|
| iris | warm-session | 4 | failed | — | 0.01 | front-end process | — |
| iris | cold-process | 4 | failed | 0.02 | 0.01 | front-end process | — |
| diffusers | warm-session | 4 | ok | 21.92 | 37.21 | front-end process | pass |
| diffusers | cold-process | 4 | ok | 28.06 | 35.15 | front-end process | pass |
| mflux | warm-session | 4 | ok | 22.54 | 57.35 | front-end process | pass |
| mflux | cold-process | 4 | ok | 26.77 | 56.46 | front-end process | pass |
| zdraw | warm-session | 4 | ok | 23.10 | 5.52 | front-end process | pass |
| zdraw | cold-process | 4 | ok | 24.18 | 4.72 | front-end process | pass |

## Decision

- `cold-process`: **INCOMPLETE**; missing required: iris, zdraw
- `warm-session`: **INCOMPLETE**; missing required: iris, zdraw

Cold-process and warm-session rows are never pooled. Memory is the macOS
`/usr/bin/time -l` peak footprint unless the row states a broader scope.
External XPC or daemon memory is excluded from front-end-only rows and must
be added before a public claim.
An unavailable required competitor makes the decision incomplete.
