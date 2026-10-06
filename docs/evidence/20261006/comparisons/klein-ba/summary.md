# Competitive benchmark

- model: `flux2-klein-4b`
- tier: `product`
- settings: `matched`
- workload: 1024×1024, requested 4 steps, seed 7
- runs: 3; warmups: 1

| engine | protocol | steps | status | median s | peak footprint GiB | memory scope | output |
|---|---|---:|---|---:|---:|---|---|
| iris | warm-session | 4 | ok | 12.80 | 29.69 | front-end process | pass |
| iris | cold-process | 4 | ok | 14.58 | 30.02 | front-end process | pass |
| diffusers | warm-session | 4 | ok | 15.76 | 28.84 | front-end process | pass |
| diffusers | cold-process | 4 | ok | 20.43 | 26.74 | front-end process | pass |
| mflux | warm-session | 4 | ok | 11.41 | 36.75 | front-end process | pass |
| mflux | cold-process | 4 | ok | 13.45 | 36.39 | front-end process | pass |
| zdraw | warm-session | 4 | ok | 12.00 | 3.05 | front-end process | pass |
| zdraw | cold-process | 4 | ok | 12.89 | 3.02 | front-end process | pass |

## Decision

- `cold-process`: **INCOMPLETE**; missing required: zdraw
- `warm-session`: **INCOMPLETE**; missing required: zdraw

Cold-process and warm-session rows are never pooled. Memory is the macOS
`/usr/bin/time -l` peak footprint unless the row states a broader scope.
External XPC or daemon memory is excluded from front-end-only rows and must
be added before a public claim.
An unavailable required competitor makes the decision incomplete.
