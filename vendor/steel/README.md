# Vendored MLX steel kernels (MIT)

`include/mlx/backend/metal/kernels/steel/` holds the GEMM and attention
subsets of Apple/ml-explore MLX's "steel" Metal kernel framework (`gemm/`,
`attn/`, `utils/`, `defines.h`), copied from the `mlx` package. MLX is
MIT-licensed; see LICENSE. The `*_nax.h` headers (MLX's Metal 4 paths) are
present but not used by either entry file.

Written for zdraw on top of those headers, not part of MLX:

- `gemm/kernels/steel_gemm_w6.h`, `steel_gemm_w4.h`, `steel_gemm_w2.h` and
  `gemm/loader_w6.h`, `loader_w4.h`, `loader_w2.h`: the quantised GEMMs. They
  reuse steel's `BlockMMA` and A-operand loader and replace the B loader with
  one that dequantises 6-, 4- or 2-bit codes (group 64, one f16 scale per
  group) into the threadgroup tile. The `*_s` variants take explicit scales
  for row sub-blocks.
- `steel_shim.h`, `gemm_entry.metal` and `attn_entry.metal`: the entry points
  compiled ahead of time into `lib/steel.metallib` by `build.zig`
  (`xcrun metal`). The metallib carries `steel_gemm_h_32/h_64/hf32_64`, the
  `w6_*`, `w4_*`, `w2_*` and `w6s/w4s/w2s_64` GEMMs, and the attention
  kernels `steel_attn_h128` and `steel_attn_h128_bk32`.

The engine loads the metallib at run time (`ZDRAW_STEEL_LIB`, then `lib/`
beside the binary); `zdraw doctor` reports whether it was found and which
routes will use it. The vendored metal-flash-attention kernel in `../mfa` is
the counted fallback for attention.
