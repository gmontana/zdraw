# Competitive benchmark

- model: `z-image-turbo`
- tier: `product`
- settings: `matched`
- workload: 1024×1024, requested 4 steps, seed 7
- runs: 3; warmups: 1

| engine | protocol | steps | status | median s | peak footprint GiB | memory scope | output |
|---|---|---:|---|---:|---:|---|---|
| zdraw | warm-session | 4 | ok | 23.80 | 5.51 | front-end process | pass |
| zdraw | cold-process | 4 | ok | 24.61 | 4.72 | front-end process | pass |
| mflux | warm-session | 4 | failed | — | 0.34 | front-end process | — |
| mflux | cold-process | 4 | ok | 27.22 | 56.46 | front-end process | pass |
| diffusers | warm-session | 4 | ok | 22.03 | 37.21 | front-end process | pass |
| diffusers | cold-process | 4 | ok | 28.01 | 35.13 | front-end process | pass |
| iris | warm-session | 4 | failed | — | 0.01 | front-end process | — |
| iris | cold-process | 4 | failed | 0.02 | 0.01 | front-end process | — |

## Decision

- `cold-process`: **INCOMPLETE**; missing required: iris, zdraw
- `warm-session`: **INCOMPLETE**; missing required: iris, zdraw

Cold-process and warm-session rows are never pooled. Memory is the macOS
`/usr/bin/time -l` peak footprint unless the row states a broader scope.
External XPC or daemon memory is excluded from front-end-only rows and must
be added before a public claim.
An unavailable required competitor makes the decision incomplete.
