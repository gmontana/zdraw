// W6 quantized-weight GEMM kernel (zdraw). Reuses steel's BlockMMA scheduling
// and BlockLoader for A; swaps in BlockLoaderW2 for B (the [N,K] 6-bit weight,
// dequantized into the threadgroup tile on load). Specialized to the shipped
// config: transpose_a=false, transpose_b=true, T=half, AccumType=float.
//
// A is f16 [M,K]; B is the W6 weight buffer (codes-then-scales, see zw6.zig);
// D is f16 [M,N] (matching the steel f16 kernel's output dtype).

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader_w2.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace metal;
using namespace mlx::steel;

// Alignment function constants. These share indices 200/201/202 with the steel
// f16 kernel (steel_gemm_fused.h), so the host pipeline-builder is identical.
// gemm_entry.metal includes steel_gemm_fused.h before this header, which
// already declares align_M/align_N/align_K at those indices; reuse them.
#ifndef ZDRAW_W2_ALIGN_CONSTANTS
#define ZDRAW_W2_ALIGN_CONSTANTS
#define w2_align_M align_M
#define w2_align_N align_N
#define w2_align_K align_K
#endif

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm_w2(
    const device T* A [[buffer(0)]],
    const device uchar* W [[buffer(1)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on
  (void)lid;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  using gemm_kernel = GEMMKernel<
      T, T, BM, BN, BK, WM, WN, transpose_a, transpose_b, true, true, AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using mma_t = typename gemm_kernel::mma_t;

  // B threadgroup leading dim (transpose_b => BK + padding).
  constexpr short ldb_tgp = BK + gemm_kernel::tgp_padding_b;
  using loader_b_t = BlockLoaderW2<T, BN, BK, ldb_tgp, WM * WN * 32>;

  // Find block.
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Threadgroup memory.
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  threadgroup_barrier(mem_flags::mem_none);

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += c_row_long * params->lda;   // [M,K], not transposed
  D += c_row_long * params->ldd + c_col_long;

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(
      W, params->N, params->K, c_col, Bs, simd_group_id, simd_lane_id);

  const short tgp_bm = w2_align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = w2_align_N ? BN : short(min(BN, params->N - c_col));

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Unaligned-K tail first (mirrors steel_gemm_fused.h): handle the partial K
  // block at the end, then run the aligned body from k=0.
  if (!w2_align_K) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;

    loader_a.src += size_t(k_last);   // A col offset (transpose_a=false)
    loader_b.k0 += k_last;            // W2 K offset

    const short2 tile_dims_A = short2(k_remain, tgp_bm);
    const short2 tile_dims_B = short2(k_remain, tgp_bn);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    mma_op.mma(As, Bs);

    loader_a.src -= size_t(k_last);
    loader_b.k0 -= k_last;
  }

  ///////////////////////////////////////////////////////////////////////////
  // MN aligned loop
  if (w2_align_M && w2_align_N) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);
    return mma_op.store_result(D, params->ldd);
  }
  ///////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else {
    const short2 tile_dims_A = short2(BK, tgp_bm);
    const short2 tile_dims_B = short2(BK, tgp_bn);
    const bool m_full = (w2_align_M || tgp_bm == BM);
    const bool n_full = (w2_align_N || tgp_bn == BN);

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (m_full) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dims_A);
      }
      if (n_full) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dims_B);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    if (m_full && n_full) {
      return mma_op.store_result(D, params->ldd);
    } else {
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}

// Explicit-scales variant for the resident executor's row sub-block binds.
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm_w2_s(
    const device T* A [[buffer(0)]],
    const device uchar* W [[buffer(1)]],
    const device half* S [[buffer(2)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on
  (void)lid;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  using gemm_kernel = GEMMKernel<
      T, T, BM, BN, BK, WM, WN, transpose_a, transpose_b, true, true, AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using mma_t = typename gemm_kernel::mma_t;

  // B threadgroup leading dim (transpose_b => BK + padding).
  constexpr short ldb_tgp = BK + gemm_kernel::tgp_padding_b;
  using loader_b_t = BlockLoaderW2<T, BN, BK, ldb_tgp, WM * WN * 32>;

  // Find block.
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Threadgroup memory.
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  threadgroup_barrier(mem_flags::mem_none);

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += c_row_long * params->lda;   // [M,K], not transposed
  D += c_row_long * params->ldd + c_col_long;

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(
      W, S, params->N, params->K, c_col, Bs, simd_group_id, simd_lane_id);

  const short tgp_bm = w2_align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = w2_align_N ? BN : short(min(BN, params->N - c_col));

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Unaligned-K tail first (mirrors steel_gemm_fused.h): handle the partial K
  // block at the end, then run the aligned body from k=0.
  if (!w2_align_K) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;

    loader_a.src += size_t(k_last);   // A col offset (transpose_a=false)
    loader_b.k0 += k_last;            // W2 K offset

    const short2 tile_dims_A = short2(k_remain, tgp_bm);
    const short2 tile_dims_B = short2(k_remain, tgp_bn);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    mma_op.mma(As, Bs);

    loader_a.src -= size_t(k_last);
    loader_b.k0 -= k_last;
  }

  ///////////////////////////////////////////////////////////////////////////
  // MN aligned loop
  if (w2_align_M && w2_align_N) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);
    return mma_op.store_result(D, params->ldd);
  }
  ///////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else {
    const short2 tile_dims_A = short2(BK, tgp_bm);
    const short2 tile_dims_B = short2(BK, tgp_bn);
    const bool m_full = (w2_align_M || tgp_bm == BM);
    const bool n_full = (w2_align_N || tgp_bn == BN);

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (m_full) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dims_A);
      }
      if (n_full) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dims_B);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    if (m_full && n_full) {
      return mma_op.store_result(D, params->ldd);
    } else {
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}
