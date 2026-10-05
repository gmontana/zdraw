// zdraw entry for MLX's steel attention (MIT, see LICENSE): the full
// non-causal self-attention kernel at the FLUX.2 Klein shape, f16 Q/K/V/O in
// [B, H, L, D] (head-major) layout, f32 accumulate. Same template
// parameters MLX instantiates for D = 128 (BQ 32, BK 16, BD 128, WM 4, WN 1;
// 128 threads per threadgroup). Built into the shared metallib by
// tools/build_steel_lib.sh next to gemm_entry.metal.
#include <metal_stdlib>

// The kernel takes Limits<T> from MLX's top-level kernels/utils.h; only the
// numeric-limit specialisations it uses are reproduced here (same values as
// MLX's, MIT).
template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};
template <>
struct Limits<float> {
  static constexpr constant float max = metal::numeric_limits<float>::infinity();
  static constexpr constant float min = -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max = metal::numeric_limits<float>::max();
  static constexpr constant float finite_min = -metal::numeric_limits<float>::max();
};
template <>
struct Limits<half> {
  static constexpr constant half max = metal::numeric_limits<half>::infinity();
  static constexpr constant half min = -metal::numeric_limits<half>::infinity();
  static constexpr constant half finite_max = metal::numeric_limits<half>::max();
  static constexpr constant half finite_min = -metal::numeric_limits<half>::max();
};

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

using namespace metal;
using namespace mlx::steel;

template [[host_name("steel_attn_h128")]] [[kernel]] decltype(attention<half, 32, 16, 128, 4, 1, float, float>)
    attention<half, 32, 16, 128, 4, 1, float, float>;

// Tiling variant for the bench (ZDRAW_KLEIN_ATTN_VARIANT=bk32): a larger K
// block than MLX's default instantiation. The query block is fixed at
// WM * 8 rows by the kernel (static_assert TQ == 1), so BQ=64 needs WM=8.
template [[host_name("steel_attn_h128_bk32")]] [[kernel]] decltype(attention<half, 32, 32, 128, 4, 1, float, float>)
    attention<half, 32, 32, 128, 4, 1, float, float>;
