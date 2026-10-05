//! Benchmark-only GEMM kernel for the substrate-decision comparison vs MPS.
//!
//! `gemm_f16` loads half A and half W directly from device memory (no staging),
//! which is the fairest head-to-head against Apple's MPS half GEMM. The runtime
//! never uses this — production kernels live in `mgemm_shader.zig`. Kept only so
//! `gemmbench` can still report how close our hand-written kernel lands to MPS.

pub const gemm_f16: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };
    \\
    \\kernel void gemm_f16(
    \\    const device half* A    [[buffer(0)]],
    \\    const device half* W    [[buffer(1)]],
    \\    device float* C         [[buffer(2)]],
    \\    constant GemmParams& p  [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tile_m = tg.y * 32;
    \\    const uint tile_n = tg.x * 32;
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint k0 = 0; k0 < p.k; k0 += 8) {
    \\        simdgroup_half8x8 a[4];
    \\        simdgroup_half8x8 b[4];
    \\        for (uint i = 0; i < 4; i++)
    \\            simdgroup_load(a[i], A + (tile_m + i * 8) * p.k + k0, p.k);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], W + (tile_n + j * 8) * p.k + k0, p.k, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\}
;

// MPP-informed v2 probe family (apple-gpu-notes section 7, pre-registered):
// the direct-access schedule of gemm_f16 re-geometried to the production
// 64x64 / 128-thread / 2x2-simdgroup occupancy (the old staged-vs-direct A/B
// compared against a 1-simdgroup direct baseline, which under-occupies).
// v2a = direct access at full occupancy; v2b = v2a + mem_none cadence
// barriers every 128 K; v2c = v2a on a 1D grid walking 8-wide column bands
// (the square-ish active-window goal of the Morton walk, exact coverage).
// Edge tiles clamp their origin and recompute the overlap - bench-only
// shapes all have m,n >= 64 and identical overlapping writes.
pub const v2: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };
    \\
    \\static inline void tile_body(
    \\    const device half* A,
    \\    const device half* W,
    \\    device float* C,
    \\    constant GemmParams& p,
    \\    uint tile_m0,
    \\    uint tile_n0,
    \\    uint sgid,
    \\    bool cadence
    \\) {
    \\    const uint sm = tile_m0 + (sgid / 2) * 32;
    \\    const uint sn = tile_n0 + (sgid % 2) * 32;
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    for (uint k0 = 0; k0 < p.k; k0 += 8) {
    \\        if (cadence && (k0 & 127u) == 0u) threadgroup_barrier(mem_flags::mem_none);
    \\        simdgroup_half8x8 a[4];
    \\        simdgroup_half8x8 b[4];
    \\        for (uint i = 0; i < 4; i++)
    \\            simdgroup_load(a[i], A + (sm + i * 8) * p.k + k0, p.k);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], W + (sn + j * 8) * p.k + k0, p.k, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\    }
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], C + (sm + i * 8) * p.n + (sn + j * 8), p.n);
    \\}
    \\
    \\kernel void gemm_f16_v2a(
    \\    const device half* A    [[buffer(0)]],
    \\    const device half* W    [[buffer(1)]],
    \\    device float* C         [[buffer(2)]],
    \\    constant GemmParams& p  [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint sgid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    tile_body(A, W, C, p, min(tg.y * 64, p.m - 64), min(tg.x * 64, p.n - 64), sgid, false);
    \\}
    \\
    \\kernel void gemm_f16_v2b(
    \\    const device half* A    [[buffer(0)]],
    \\    const device half* W    [[buffer(1)]],
    \\    device float* C         [[buffer(2)]],
    \\    constant GemmParams& p  [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint sgid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    tile_body(A, W, C, p, min(tg.y * 64, p.m - 64), min(tg.x * 64, p.n - 64), sgid, true);
    \\}
    \\
    \\kernel void gemm_f16_v2c(
    \\    const device half* A    [[buffer(0)]],
    \\    const device half* W    [[buffer(1)]],
    \\    device float* C         [[buffer(2)]],
    \\    constant GemmParams& p  [[buffer(3)]],
    \\    uint gid [[threadgroup_position_in_grid]],
    \\    uint sgid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    uint tiles_x = (p.n + 63) / 64;
    \\    uint tiles_y = (p.m + 63) / 64;
    \\    uint band = gid / (8 * tiles_y);
    \\    uint rem = gid - band * 8 * tiles_y;
    \\    uint by = rem / 8;
    \\    uint bx = band * 8 + (rem - by * 8);
    \\    if (bx >= tiles_x || by >= tiles_y) return;
    \\    tile_body(A, W, C, p, min(by * 64, p.m - 64), min(bx * 64, p.n - 64), sgid, false);
    \\}
;

// f32 twin of gemm_f16: the kill-criterion control for #44-v5 — measures what
// the proven direct-load schedule yields at FP32 on this GPU.
pub const gemm_f32: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };
    \\
    \\kernel void gemm_f32(
    \\    const device float* A   [[buffer(0)]],
    \\    const device float* W   [[buffer(1)]],
    \\    device float* C         [[buffer(2)]],
    \\    constant GemmParams& p  [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tile_m = tg.y * 32;
    \\    const uint tile_n = tg.x * 32;
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint k0 = 0; k0 < p.k; k0 += 8) {
    \\        simdgroup_float8x8 a[4];
    \\        simdgroup_float8x8 b[4];
    \\        for (uint i = 0; i < 4; i++)
    \\            simdgroup_load(a[i], A + (tile_m + i * 8) * p.k + k0, p.k);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], W + (tile_n + j * 8) * p.k + k0, p.k, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\}
;
