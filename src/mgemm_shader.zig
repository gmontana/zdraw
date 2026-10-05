//! Production tiled GEMM kernels using the simdgroup_matrix hardware MMA.
//!
//! Both compute C[M,N] = A[M,K] * Wᵀ with W stored row-major as [N,K] (the
//! model's linear weight layout). One simdgroup (32 threads) per threadgroup
//! owns a 32x32 output tile = a 4x4 grid of 8x8 fragments. f32 activations in,
//! f32 output, weight read in place at a byte offset (no converted-weight temp).
//!
//! `gemm_exact` keeps full f32 precision (float8x8 MMA) — bit-exact vs the naive
//! reference, gated by refcheck. `gemm_half` is the lossy fast path: it stages
//! both operands down to half for the ~2x-faster half8x8 MMA (f32 accumulate),
//! and is gated by the image-quality suite, never by refcheck. The benchmark-only
//! `gemm_f16` lives in mgemm_bench_shader.zig.

pub const gemm: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };
    \\
    \\static inline float gemm_weight(const device uchar* base, uint index, uint dtype) {
    \\    if (dtype == 3) {
    \\        const device float* w = (const device float*)base;
    \\        return w[index];
    \\    }
    \\    const device ushort* w = (const device ushort*)base;
    \\    ushort bits = w[index];
    \\    if (dtype == 2) return as_type<float>(uint(bits) << 16);
    \\    return float(as_type<half>(bits));
    \\}
    \\
    \\kernel void gemm_exact(
    \\    const device float* A    [[buffer(0)]],
    \\    const device uchar* Wbytes [[buffer(1)]],
    \\    device float* C          [[buffer(2)]],
    \\    constant GemmParams& p   [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint tile_m = tg.y * 32;
    \\    const uint tile_n = tg.x * 32;
    \\    const device uchar* W = Wbytes + p.weight_offset;
    \\    const device float* Wf = (const device float*)W;
    \\    threadgroup float w_stage[32 * 8];
    \\
    \\    if (tile_m + 32u > p.m || tile_n + 32u > p.n) {
    \\        const uint row = tile_m + tid;
    \\        if (row < p.m) {
    \\            for (uint cn = 0; cn < 32; cn++) {
    \\                const uint col = tile_n + cn;
    \\                if (col < p.n) {
    \\                    float sum = 0.0f;
    \\                    for (uint kk = 0; kk < p.k; kk++)
    \\                        sum += A[row * p.k + kk] * gemm_weight(W, col * p.k + kk, p.dtype);
    \\                    C[row * p.n + col] = sum;
    \\                }
    \\            }
    \\        }
    \\        return;
    \\    }
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
    \\        if (p.dtype == 3) {
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], Wf + (tile_n + j * 8) * p.k + k0,
    \\                               p.k, ulong2(0, 0), true);
    \\        } else {
    \\            for (uint kk = 0; kk < 8; kk++)
    \\                w_stage[tid * 8 + kk] =
    \\                    gemm_weight(W, (tile_n + tid) * p.k + k0 + kk, p.dtype);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], w_stage + j * 64, 8, ulong2(0, 0), true);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\}
    \\
    \\// Lossy fast path: stage f32 A and (any-dtype) W down to half, run the half8x8
    \\// MMA with f32 accumulate. ~half the bytes and ~2x the MMA rate of the f32 path.
    \\// Gated by the image-quality suite, never by refcheck.
    \\kernel void gemm_half(
    \\    const device float* A    [[buffer(0)]],
    \\    const device uchar* Wbytes [[buffer(1)]],
    \\    device float* C          [[buffer(2)]],
    \\    constant GemmParams& p   [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint tile_m = tg.y * 32;
    \\    const uint tile_n = tg.x * 32;
    \\    const device uchar* W = Wbytes + p.weight_offset;
    \\    threadgroup half a_stage[32 * 8];
    \\    threadgroup half w_stage[32 * 8];
    \\
    \\    if (tile_m + 32u > p.m || tile_n + 32u > p.n) {
    \\        const uint row = tile_m + tid;
    \\        if (row < p.m) {
    \\            for (uint cn = 0; cn < 32; cn++) {
    \\                const uint col = tile_n + cn;
    \\                if (col < p.n) {
    \\                    float sum = 0.0f;
    \\                    for (uint kk = 0; kk < p.k; kk++)
    \\                        sum += A[row * p.k + kk] * gemm_weight(W, col * p.k + kk, p.dtype);
    \\                    C[row * p.n + col] = sum;
    \\                }
    \\            }
    \\        }
    \\        return;
    \\    }
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    if (p.dtype == 3 && (p.k & 31u) == 0u) {
    \\        // f32-weight fast path: SAME values the gather produced (float ->
    \\        // half per element), staged 32 K wide with vectorized loads so the
    \\        // barrier pair amortizes over four MMA chunks. Same 8-chunk K
    \\        // order, same f32 accumulators: bit-identical, far less overhead.
    \\        const device float* Wf = reinterpret_cast<const device float*>(W);
    \\        threadgroup half a_wide[32 * 32];
    \\        threadgroup half w_wide[32 * 32];
    \\        for (uint k0 = 0; k0 < p.k; k0 += 32) {
    \\            const device float4* av =
    \\                reinterpret_cast<const device float4*>(A + (tile_m + tid) * p.k + k0);
    \\            const device float4* wv =
    \\                reinterpret_cast<const device float4*>(Wf + (tile_n + tid) * p.k + k0);
    \\            threadgroup half* arow = a_wide + tid * 32;
    \\            threadgroup half* wrow = w_wide + tid * 32;
    \\            for (uint q = 0; q < 8; q++) {
    \\                float4 va = av[q];
    \\                arow[q * 4 + 0] = half(clamp(va.x, -65504.0f, 65504.0f));
    \\                arow[q * 4 + 1] = half(clamp(va.y, -65504.0f, 65504.0f));
    \\                arow[q * 4 + 2] = half(clamp(va.z, -65504.0f, 65504.0f));
    \\                arow[q * 4 + 3] = half(clamp(va.w, -65504.0f, 65504.0f));
    \\                float4 vw = wv[q];
    \\                wrow[q * 4 + 0] = half(vw.x);
    \\                wrow[q * 4 + 1] = half(vw.y);
    \\                wrow[q * 4 + 2] = half(vw.z);
    \\                wrow[q * 4 + 3] = half(vw.w);
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            for (uint c8 = 0; c8 < 4; c8++) {
    \\                simdgroup_half8x8 a[4];
    \\                simdgroup_half8x8 b[4];
    \\                for (uint i = 0; i < 4; i++)
    \\                    simdgroup_load(a[i], a_wide + i * 8 * 32 + c8 * 8, 32);
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_load(b[j], w_wide + j * 8 * 32 + c8 * 8, 32, ulong2(0, 0), true);
    \\                for (uint i = 0; i < 4; i++)
    \\                    for (uint j = 0; j < 4; j++)
    \\                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\        return;
    \\    }
    \\
    \\    if (p.dtype == 1 && (p.k & 31u) == 0u) {
    \\        // Fast path for cached-f16 weights: B tiles simdgroup_load
    \\        // DIRECTLY from device (same values the gather produced), A
    \\        // stages 32 K at a time so the barrier pair amortizes over four
    \\        // MMA chunks. Same saturating cast, same 8-chunk K order, same
    \\        // f32 accumulators: bit-identical to the staged path, ~4-8x
    \\        // less barrier/gather overhead.
    \\        const device half* Wh = reinterpret_cast<const device half*>(W);
    \\        threadgroup half a_wide[32 * 32];
    \\        for (uint k0 = 0; k0 < p.k; k0 += 32) {
    \\            const device float4* av =
    \\                reinterpret_cast<const device float4*>(A + (tile_m + tid) * p.k + k0);
    \\            threadgroup half* arow = a_wide + tid * 32;
    \\            for (uint q = 0; q < 8; q++) {
    \\                float4 v = av[q];
    \\                arow[q * 4 + 0] = half(clamp(v.x, -65504.0f, 65504.0f));
    \\                arow[q * 4 + 1] = half(clamp(v.y, -65504.0f, 65504.0f));
    \\                arow[q * 4 + 2] = half(clamp(v.z, -65504.0f, 65504.0f));
    \\                arow[q * 4 + 3] = half(clamp(v.w, -65504.0f, 65504.0f));
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            for (uint c8 = 0; c8 < 4; c8++) {
    \\                simdgroup_half8x8 a[4];
    \\                simdgroup_half8x8 b[4];
    \\                for (uint i = 0; i < 4; i++)
    \\                    simdgroup_load(a[i], a_wide + i * 8 * 32 + c8 * 8, 32);
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_load(b[j], Wh + (tile_n + j * 8) * p.k + k0 + c8 * 8,
    \\                                   p.k, ulong2(0, 0), true);
    \\                for (uint i = 0; i < 4; i++)
    \\                    for (uint j = 0; j < 4; j++)
    \\                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\        return;
    \\    }
    \\
    \\    for (uint k0 = 0; k0 < p.k; k0 += 8) {
    \\        for (uint kk = 0; kk < 8; kk++) {
    \\            // Saturating cast: outlier activations clamp instead of inf.
    \\            a_stage[tid * 8 + kk] =
    \\                half(clamp(A[(tile_m + tid) * p.k + k0 + kk], -65504.0f, 65504.0f));
    \\            w_stage[tid * 8 + kk] =
    \\                half(gemm_weight(W, (tile_n + tid) * p.k + k0 + kk, p.dtype));
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        simdgroup_half8x8 a[4];
    \\        simdgroup_half8x8 b[4];
    \\        for (uint i = 0; i < 4; i++)
    \\            simdgroup_load(a[i], a_stage + i * 64, 8);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], w_stage + j * 64, 8, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], C + (tile_m + i * 8) * p.n + (tile_n + j * 8), p.n);
    \\}
    \\
    \\struct BiasParams { uint n; uint dtype; ulong bias_offset; };
    \\
    \\kernel void gemm_bias(
    \\    device float* C [[buffer(0)]],
    \\    const device uchar* Bbytes [[buffer(1)]],
    \\    constant BiasParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    C[gid] += gemm_weight(Bbytes + p.bias_offset, gid % p.n, p.dtype);
    \\}
;
