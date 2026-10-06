# Competitive benchmark

- model: `z-image-turbo`
- tier: `product`
- settings: `matched`
- workload: 1024×1024, requested 4 steps, seed 7
- runs: 3; warmups: 1

| engine | protocol | steps | status | median s | peak footprint GiB | memory scope | output |
|---|---|---:|---|---:|---:|---|---|
| mflux | warm-session | 4 | ok | 23.46 | 57.35 | front-end process | pass |
| mflux | cold-process | 4 | ok | 26.83 | 56.46 | front-end process | pass |

## Decision

- `cold-process`: **INCOMPLETE**
- `warm-session`: **INCOMPLETE**

Cold-process and warm-session rows are never pooled. Memory is the macOS
`/usr/bin/time -l` peak footprint unless the row states a broader scope.
External XPC or daemon memory is excluded from front-end-only rows and must
be added before a public claim.
An unavailable required competitor makes the decision incomplete.
