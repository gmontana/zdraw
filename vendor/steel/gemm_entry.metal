// Vendored steel_gemm instantiations (MLX, MIT). f16 inputs, f32 accumulate,
// B transposed (our W is [N,K]); product uses f16 and aligned f32 outputs.
#include <metal_stdlib>
#include "steel_shim.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_w6.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_w4.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_w2.h"

template [[host_name("steel_gemm_h_32")]] [[kernel]] decltype(gemm<half, 32, 32, 16, 2, 2, false, true, float>)
    gemm<half, 32, 32, 16, 2, 2, false, true, float>;
template [[host_name("steel_gemm_h_64")]] [[kernel]] decltype(gemm<half, 64, 64, 16, 2, 2, false, true, float>)
    gemm<half, 64, 64, 16, 2, 2, false, true, float>;

template <
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm_f32out_aligned(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    device float* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using gemm_kernel =
      GEMMKernel<half, float, BM, BN, BK, WM, WN, false, true, true, true, float>;
  threadgroup half As[gemm_kernel::tgp_mem_size_a];
  threadgroup half Bs[gemm_kernel::tgp_mem_size_b];
  gemm_kernel::run(A, B, D, params, As, Bs, simd_lane_id, simd_group_id, tid, lid);
}

template [[host_name("steel_gemm_hf32_64")]] [[kernel]] decltype(gemm_f32out_aligned<64, 64, 16, 2, 2>)
    gemm_f32out_aligned<64, 64, 16, 2, 2>;

// W6 (6-bit grouped) weight variant: reads OUR W6 format, dequantizes in the
// B-loader, keeps steel's MMA scheduling. f16 A, f16 D out.
template [[host_name("steel_gemm_w6_64")]] [[kernel]] decltype(gemm_w6<half, 64, 64, 16, 2, 2, float>)
    gemm_w6<half, 64, 64, 16, 2, 2, float>;

// Tile-config variants for shape tuning (raced in gemmbench; the chain ships
// whichever wins on the production shapes).
template [[host_name("steel_gemm_w6_128x64")]] [[kernel]] decltype(gemm_w6<half, 128, 64, 16, 4, 2, float>)
    gemm_w6<half, 128, 64, 16, 4, 2, float>;
template [[host_name("steel_gemm_w6_64x128")]] [[kernel]] decltype(gemm_w6<half, 64, 128, 16, 2, 4, float>)
    gemm_w6<half, 64, 128, 16, 2, 4, float>;
template [[host_name("steel_gemm_w6_128x128")]] [[kernel]] decltype(gemm_w6<half, 128, 128, 16, 4, 4, float>)
    gemm_w6<half, 128, 128, 16, 4, 4, float>;
template [[host_name("steel_gemm_w6_bk32")]] [[kernel]] decltype(gemm_w6<half, 64, 64, 32, 2, 2, float>)
    gemm_w6<half, 64, 64, 32, 2, 2, float>;

// W4 (4-bit grouped) twin: OUR W4 format (2 codes per byte, low nibble first,
// group 64, f16 scales after the codes), same MMA scheduling. f16 A, f16 D out.
template [[host_name("steel_gemm_w4_64")]] [[kernel]] decltype(gemm_w4<half, 64, 64, 16, 2, 2, float>)
    gemm_w4<half, 64, 64, 16, 2, 2, float>;
template [[host_name("steel_gemm_w4_128x64")]] [[kernel]] decltype(gemm_w4<half, 128, 64, 16, 4, 2, float>)
    gemm_w4<half, 128, 64, 16, 4, 2, float>;

// W2 (2-bit grouped, ternary in practice) twin: OUR W2 format (4 codes per
// byte, lowest pair first, group 64, f16 scales after the codes).
template [[host_name("steel_gemm_w2_64")]] [[kernel]] decltype(gemm_w2<half, 64, 64, 16, 2, 2, float>)
    gemm_w2<half, 64, 64, 16, 2, 2, float>;

// Explicit-scales variants (S at buffer 2) for the Klein resident executor,
// which binds row sub-blocks whose scales the loader cannot derive.
template [[host_name("steel_gemm_w6s_64")]] [[kernel]] decltype(gemm_w6_s<half, 64, 64, 16, 2, 2, float>)
    gemm_w6_s<half, 64, 64, 16, 2, 2, float>;
template [[host_name("steel_gemm_w4s_64")]] [[kernel]] decltype(gemm_w4_s<half, 64, 64, 16, 2, 2, float>)
    gemm_w4_s<half, 64, 64, 16, 2, 2, float>;
template [[host_name("steel_gemm_w2s_64")]] [[kernel]] decltype(gemm_w2_s<half, 64, 64, 16, 2, 2, float>)
    gemm_w2_s<half, 64, 64, 16, 2, 2, float>;
