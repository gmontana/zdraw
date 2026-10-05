//! GEMM on the Metal 4 tensor path (mpp::tensor_ops::matmul2d over
//! cooperative_tensor fragments), the one documented route past the
//! simdgroup_matrix ceiling on M4-class GPUs (Rigel, arXiv 2606.12765: tiled
//! fp16 matmul2d 14.8 TFLOP/s at 2048^3, 1.05-1.21x simdgroup_matrix).
//! Metal Shading Language 4.0 (macOS 26); compiled at runtime by
//! zdraw_metal_compile_mpp, which returns NULL on older systems.
//!
//! Operands match gemm_f16a_direct: A [m][k] half row-major, W [n][k] half
//! row-major (the sidecar layout: NT), C [m][n] f32. Each threadgroup (4
//! simdgroups) builds dense tensor_inline views of its 64-row slices of A and
//! W from offset pointers (host-bound tensors need Metal 4 encoders), runs
//! matmul2d with the k loop inside the op, stages the f32 tile through
//! threadgroup memory and writes it out coalesced. Contract: m % 64, n % TN,
//! k % 32 == 0.
pub const src: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
    \\struct GemmParams { uint m; uint k; uint n; uint dtype; uint mode; ulong weight_offset; };
    \\
    \\// 64 x TN output tile, 4 simdgroups (128 threads), k looped inside the op.
    \\template <int TN>
    \\static inline void gemm_mpp_tile(
    \\    device half* A, device half* W, device float* C, constant GemmParams& p,
    \\    uint2 tg, uint tid, threadgroup float* cs
    \\) {
    \\    const uint m0 = tg.y * 64;
    \\    const uint n0 = tg.x * TN;
    \\    const int k = int(p.k);
    \\    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
    \\        A + m0 * p.k, dextents<int32_t, 2>(k, 64));
    \\    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
    \\        W + n0 * p.k, dextents<int32_t, 2>(k, TN));
    \\    constexpr auto desc = matmul2d_descriptor(64, TN, static_cast<int>(dynamic_extent),
    \\                                              false, true, false,
    \\                                              matmul2d_descriptor::mode::multiply);
    \\    matmul2d<desc, execution_simdgroups<4>> op;
    \\    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    \\    op.run(tA, tB, cT);
    \\    auto tCs = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
    \\        cs, dextents<int32_t, 2>(TN, 64));
    \\    cT.store(tCs);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    // 64 rows x TN floats; 128 threads: row = tid / (128/64) ...
    \\    const uint rows_per_pass = 128 / (TN / 32);   // threads cover TN/32 segments of 32
    \\    const uint seg = (tid % (TN / 32)) * 32;
    \\    for (uint row = tid / (TN / 32); row < 64; row += rows_per_pass) {
    \\        device float4* dst =
    \\            reinterpret_cast<device float4*>(C + (m0 + row) * p.n + n0 + seg);
    \\        const threadgroup float4* s4 =
    \\            reinterpret_cast<const threadgroup float4*>(cs + row * TN + seg);
    \\        for (uint q = 0; q < 8; q++) dst[q] = s4[q];
    \\    }
    \\}
    \\
    \\kernel void gemm_mpp64(
    \\    device half* A [[buffer(0)]],
    \\    device half* W [[buffer(1)]],
    \\    device float* C [[buffer(2)]],
    \\    constant GemmParams& p [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    threadgroup float cs[64 * 64];
    \\    gemm_mpp_tile<64>(A, W + (p.weight_offset >> 1), C, p, tg, tid, cs);
    \\}
    \\
    \\kernel void gemm_mpp32(
    \\    device half* A [[buffer(0)]],
    \\    device half* W [[buffer(1)]],
    \\    device float* C [[buffer(2)]],
    \\    constant GemmParams& p [[buffer(3)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    threadgroup float cs[64 * 32];
    \\    gemm_mpp_tile<32>(A, W + (p.weight_offset >> 1), C, p, tg, tid, cs);
    \\}
;
