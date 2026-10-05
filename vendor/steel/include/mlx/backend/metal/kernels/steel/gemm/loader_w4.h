// W4 weight block loader (zdraw): the W6 loader's twin for 4-bit codes. Mirrors
// steel's BlockLoader threadgroup interface but reads 4-bit grouped-quantized
// weights and dequantizes into the threadgroup tile for the unchanged MMA path.
//
// Layout (src/zw4.zig): per [N,K] weight, packed codes (2 signed 4-bit codes
// per byte, low nibble first, row-major, each row padded so K rounds up to a
// whole group of W4G=64) followed by one f16 scale per (row, group).
//
// This loader is only valid for the transpose_b=true GEMM (our W is [N,K]):
// the B tile is (BROWS=BN rows, BCOLS=BK cols) and each tile column is a K
// position, each tile row an N (output) row — i.e. dst[r][c] = W[n0+r][k0+c].

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"

namespace mlx {
namespace steel {

// Group size for W4 (must match src/zw4.zig packW4 group).
STEEL_CONST short W4G = 64;

// T is the dequant/threadgroup type (half). BROWS = BN, BCOLS = BK.
// dst_ld is the threadgroup leading dim (BK + padding). tgp_size = WM*WN*32.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderW4 {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Per-row code bytes and groups-per-row (computed from K at construction).
  const int gpr;   // groups per row = ceil(K / W4G)
  const int cbpr;  // code bytes per row = gpr * W4G / 2

  // Thread location indices.
  const short thread_idx;
  const short bi;  // tile row offset (which N row within the BROWS strip)
  const short bj;  // tile col offset (which K position within BCOLS)

  // threadgroup destination and W6 source.
  threadgroup T* dst;
  const device uchar* codes;        // base of packed codes (== Wp)
  const device half* scales;        // base of f16 scales
  int n0;                           // global N row of this tile's row 0
  int k0;                           // global K col of this tile's col 0
  const int n_limit;                // total N (rows) for bounds
  const int k_limit;                // total K (cols) for bounds

  /* Constructor */
  METAL_FUNC BlockLoaderW4(
      const device uchar* w_base,   // weight buffer at weight_offset (== Wp)
      const int N_,
      const int K_,
      const int n_block_,           // tile's starting N row (c_col)
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : gpr((K_ + W4G - 1) / W4G),
        cbpr(((K_ + W4G - 1) / W4G) * W4G / 2),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        codes(w_base),
        scales((const device half*)(w_base + size_t(N_) * (((K_ + W4G - 1) / W4G) * W4G / 2))),
        n0(n_block_ + bi),
        k0(bj),
        n_limit(N_),
        k_limit(K_) {}

  /* Constructor with an explicit scales pointer: the executor binds row
     sub-blocks of a matrix, whose scales cannot be derived from the codes
     pointer (they live after ALL rows' codes). s_base already points at the
     sub-block's first row's scales. */
  METAL_FUNC BlockLoaderW4(
      const device uchar* w_base,
      const device half* s_base,
      const int N_,
      const int K_,
      const int n_block_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : gpr((K_ + W4G - 1) / W4G),
        cbpr(((K_ + W4G - 1) / W4G) * W4G / 2),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        codes(w_base),
        scales(s_base),
        n0(n_block_ + bi),
        k0(bj),
        n_limit(N_),
        k_limit(K_) {}

  // Dequantize `count` consecutive K values for global row `n`, starting at
  // global K col `kc`, into out[0..count). Two codes per byte, low nibble first;
  // the group scale is reused across the group (W4G is even so groups never
  // split a byte). Caller guarantees n < N.
  METAL_FUNC void dequant_run(
      threadgroup T* out, const int n, const int kc, const short count) const {
    const size_t row_base = size_t(n) * cbpr;
    const int srow = n * gpr;
    // Fast path for the aligned full run the tile geometry always produces in
    // load_unsafe: kc % 8 == 0 and count == 8. The run stays inside ONE scale
    // group and covers exactly four bytes: one scale fetch, one 32-bit load,
    // eight branch-free decodes. Identical per-element operations
    // (int code -> T -> * half scale) => bit-identical output.
    if ((count & 7) == 0 && (kc & 7) == 0) {
      for (short o = 0; o < count; o += 8) {
        const int kco = kc + o;
        const half s = scales[srow + (kco >> 6)];
        const size_t at = row_base + size_t(kco >> 1);
        const uint b0 = codes[at + 0], b1 = codes[at + 1];
        const uint b2 = codes[at + 2], b3 = codes[at + 3];
        const uint word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 8; j++) {
          const uint raw = (word >> (j * 4)) & 0xF;
          const int code = raw < 8 ? int(raw) : int(raw) - 16;
          out[o + j] = T(code) * s;
        }
      }
      return;
    }
    for (short j = 0; j < count; j++) {
      const int col = kc + j;
      const half s = scales[srow + col / W4G];
      const uint raw = (uint(codes[row_base + size_t(col >> 1)]) >> ((col & 1) * 4)) & 0xF;
      const int code = raw < 8 ? int(raw) : int(raw) - 16;
      out[j] = T(code) * s;
    }
  }

  /* Load + dequant without bound checking (N tile aligned, full K run). */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      dequant_run(&dst[i * dst_ld], n0 + i, k0, vec_size);
    }
  }

  /* Load + dequant with bound checking. src_tile_dim = (k_avail, n_avail). */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    // Remaining valid extent for this thread's (bi, bj) start.
    const short k_avail = src_tile_dim.x - bj;
    const short n_avail = src_tile_dim.y - bi;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      threadgroup T* d = &dst[i * dst_ld];
      const short nrow_ok = n_avail - i;
      if (nrow_ok <= 0 || k_avail <= 0) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) d[j] = T(0);
        continue;
      }
      const short kcount = k_avail < vec_size ? k_avail : vec_size;
      dequant_run(d, n0 + i, k0, kcount);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        if (j >= kcount) d[j] = T(0);
      }
    }
  }

  /* Iteration helper: advance K by one tile (BCOLS). */
  METAL_FUNC void next() {
    k0 += BCOLS;
  }
};

} // namespace steel
} // namespace mlx
