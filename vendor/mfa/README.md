# Vendored: metal-flash-attention forward kernel (D=128)

Source: github.com/philipturner/metal-flash-attention @ 8671cdd (MIT,
Copyright (c) 2024 Philip Turner — see LICENSE-MIT-philipturner).

mfa-fwd-d128.metal is the GENERATED kernel source for: forward pass,
D=128, FP16-mixed, Apple9 (M3/M4) parameters (16/128/32, cache-O),
produced by their own emitters via an additive wrapper target.
ABI + dispatch recipe: mfa-fwd-d128-abi.md.

Local patch required: Metal 4 toolchains reject the __asm
air.simdgroup_async_copy intrinsics in the embedded prologue (MFA's own
tests fail identically on Xcode 26); the tail-block async copies must be
swapped for synchronous cooperative copies (performance-equivalent at
our shapes: tails are 1 of 33 blocks). Recorded in provenance.md policy:
this is a deliberate licensing decision, attributed, not an accident.

## mfa-fwd-d128-m1m2.metal (patched, PSO-verified)

M1/M2 variant (forced non-apple9 row: PARALLEL=32/TRAVERSAL=128/HEAD=32,
cache Q; preferAsyncLoad=true so K/V stage via async copies in steady
state). Dispatch: TG 128 threads, grid ceil(R/32), 8192 B tg at index 0,
same buffers 0-4 + R/C constants. Prologue carries the same Metal 4
sync-copy patch as the apple9 file; PSO builds on the host compiler.
CAVEAT: here the sync copies run in the STEADY state (K/V staging + O
paging, 17 call sites), so the perf cost must be measured on REAL M1/M2
hardware before promotion — if it hurts, the fallback is our own v4b
kernel or restoring async copies under an older-toolchain build. Bridge
wiring: supportsFamily(apple9) selects variant + TG(64 vs 128) +
grid(R/16 vs R/32); blocked on CI-fleet hardware, not on code.
