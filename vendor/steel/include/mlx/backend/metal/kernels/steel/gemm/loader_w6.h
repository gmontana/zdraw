// W6 weight block loader (zdraw). Mirrors steel's BlockLoader threadgroup
// interface but reads 6-bit grouped-quantized weights and dequantizes into the
// threadgroup tile, so the unchanged MMA path consumes half values.
//
// Layout (src/zw6.zig): per [N,K] weight, packed codes (4 signed 6-bit codes
// in 3 bytes, row-major, each row padded so K rounds up to a whole group of
// W6G=64) followed by one f16 scale per (row, group).
//
// This loader is only valid for the transpose_b=true GEMM (our W is [N,K]):
// the B tile is (BROWS=BN rows, BCOLS=BK cols) and each tile column is a K
// position, each tile row an N (output) row — i.e. dst[r][c] = W[n0+r][k0+c].

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"

namespace mlx {
namespace steel {

// Group size for W6 (must match src/zw6.zig packW6 group).
STEEL_CONST short W6G = 64;

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
struct BlockLoaderW6 {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Per-row code bytes and groups-per-row (computed from K at construction).
  const int gpr;   // groups per row = ceil(K / W6G)
  const int cbpr;  // code bytes per row = gpr * W6G / 4 * 3

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
  METAL_FUNC BlockLoaderW6(
      const device uchar* w_base,   // weight buffer at weight_offset (== Wp)
      const int N_,
      const int K_,
      const int n_block_,           // tile's starting N row (c_col)
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : gpr((K_ + W6G - 1) / W6G),
        cbpr(((K_ + W6G - 1) / W6G) * W6G / 4 * 3),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        codes(w_base),
        scales((const device half*)(w_base + size_t(N_) * (((K_ + W6G - 1) / W6G) * W6G / 4 * 3))),
        n0(n_block_ + bi),
        k0(bj),
        n_limit(N_),
        k_limit(K_) {}

  /* Constructor with an explicit scales pointer: the executor binds row
     sub-blocks of a matrix, whose scales cannot be derived from the codes
     pointer (they live after ALL rows' codes). s_base already points at the
     sub-block's first row's scales. */
  METAL_FUNC BlockLoaderW6(
      const device uchar* w_base,
      const device half* s_base,
      const int N_,
      const int K_,
      const int n_block_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : gpr((K_ + W6G - 1) / W6G),
        cbpr(((K_ + W6G - 1) / W6G) * W6G / 4 * 3),
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
  // global K col `kc`, into out[0..count). Reads one packed 24-bit word per 4
  // codes; reuses the group scale across the group (W6G is a multiple of 4 so
  // groups never split a word). Caller guarantees n < N.
  METAL_FUNC void dequant_run(
      threadgroup T* out, const int n, const int kc, const short count) const {
    const size_t row_base = size_t(n) * cbpr;
    const int srow = n * gpr;
    // Fast path for the aligned full run the tile geometry always produces in
    // load_unsafe: kc % 8 == 0 and count == 8. The run stays inside ONE scale
    // group (W6G % 8 == 0) and covers exactly TWO whole quad-words (6
    // consecutive code bytes), so everything straight-lines: one scale fetch,
    // six byte loads, eight branch-free decodes. Identical per-element
    // operations (int code -> T -> * half scale) => bit-identical output.
    if ((count & 7) == 0 && (kc & 7) == 0) {
      for (short o = 0; o < count; o += 8) {
        const int kco = kc + o;
        const half s = scales[srow + (kco >> 6)];
        const size_t at = row_base + size_t(kco >> 2) * 3;
        const uint b0 = codes[at + 0], b1 = codes[at + 1], b2 = codes[at + 2];
        const uint b3 = codes[at + 3], b4 = codes[at + 4], b5 = codes[at + 5];
        const uint w0 = b0 | (b1 << 8) | (b2 << 16);
        const uint w1 = b3 | (b4 << 8) | (b5 << 16);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const uint raw = (w0 >> (j * 6)) & 0x3F;
          const int code = raw < 32 ? int(raw) : int(raw) - 64;
          out[o + j] = T(code) * s;
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const uint raw = (w1 >> (j * 6)) & 0x3F;
          const int code = raw < 32 ? int(raw) : int(raw) - 64;
          out[o + 4 + j] = T(code) * s;
        }
      }
      return;
    }
    short j = 0;
    while (j < count) {
      const int col = kc + j;
      const int g = col / W6G;
      const half s = scales[srow + g];
      // Decode within the current quad-word; advance up to 4 codes at a time.
      const int quad = col >> 2;
      const size_t at = row_base + size_t(quad) * 3;
      const uint word =
          uint(codes[at]) | (uint(codes[at + 1]) << 8) | (uint(codes[at + 2]) << 16);
      short lane = col & 3;
      // Emit codes from this word until we cross a quad boundary or finish.
      do {
        const uint raw = (word >> (lane * 6)) & 0x3F;
        const int code = raw < 32 ? int(raw) : int(raw) - 64;
        out[j] = T(code) * s;
        j++;
        lane++;
      } while (lane < 4 && j < count);
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
