# Competitive benchmark

- model: `flux2-klein-4b`
- tier: `product`
- settings: `matched`
- workload: 1024×1024, requested 4 steps, seed 7
- runs: 3; warmups: 1

| engine | protocol | steps | status | median s | peak footprint GiB | memory scope | output |
|---|---|---:|---|---:|---:|---|---|
| zdraw | warm-session | 4 | ok | 11.00 | 3.05 | front-end process | pass |
| zdraw | cold-process | 4 | ok | 11.81 | 3.02 | front-end process | pass |
| mflux | warm-session | 4 | ok | 10.94 | 36.74 | front-end process | pass |
| mflux | cold-process | 4 | ok | 13.41 | 36.39 | front-end process | pass |
| diffusers | warm-session | 4 | ok | 14.88 | 28.89 | front-end process | pass |
| diffusers | cold-process | 4 | ok | 20.40 | 26.74 | front-end process | pass |
| iris | warm-session | 4 | ok | 12.31 | 30.15 | front-end process | pass |
| iris | cold-process | 4 | ok | 13.96 | 29.67 | front-end process | pass |

## Decision

- `cold-process`: **INCOMPLETE**; missing required: zdraw
- `warm-session`: **INCOMPLETE**; missing required: zdraw

Cold-process and warm-session rows are never pooled. Memory is the macOS
`/usr/bin/time -l` peak footprint unless the row states a broader scope.
External XPC or daemon memory is excluded from front-end-only rows and must
be added before a public claim.
An unavailable required competitor makes the decision incomplete.
