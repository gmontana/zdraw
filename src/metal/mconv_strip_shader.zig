//! Strip-local conv kernels of the memory ladder's wall 1 (strip-memory VAE
//! decode): addressing-only variants of the base kernels, compiled on top of
//! the base conv source so they share `read_value`, `Params` and
//! `ConvWindowParams`. Their arithmetic is the base kernels' text.
const base = @import("mconv_shader.zig");

pub const conv: [:0]const u8 = base.conv ++ src;

const src: [:0]const u8 =
    \\// conv1x1_strip_h: conv1x1_h with the output stored strip-locally as
    \\// [out_ch][row1-row0][width] (memory-ladder wall 1). The input stays
    \\// whole-map and every arithmetic step is the same text as conv1x1_h, so
    \\// the values are bit-identical; only the store index differs.
    \\kernel void conv1x1_strip_h(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device half* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint row0_pos = params.row0 * params.width;
    \\    uint strip_len = row1_pos - row0_pos;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half w_stage[32 * 8];
    \\    threadgroup half x_stage[32 * 8];
    \\    threadgroup float c_stage[32 * 32];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        for (uint kk = 0; kk < 8; kk++) {
    \\            uint ic = ic0 + kk;
    \\            uint oc = tile_oc + tid;
    \\            uint pos = tile_pos + tid;
    \\            half wv = half(0.0f);
    \\            half xv = half(0.0f);
    \\            if (ic < params.in_ch) {
    \\                if (oc < params.out_ch) wv = wf[oc * params.in_ch + ic];
    \\                if (pos < row1_pos) xv = input[ic * hw + pos];
    \\            }
    \\            w_stage[tid * 8 + kk] = wv;
    \\            x_stage[tid * 8 + kk] = xv;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        simdgroup_half8x8 a[4];
    \\        simdgroup_half8x8 b[4];
    \\        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], w_stage + i * 64, 8);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], x_stage + j * 64, 8, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + tid;
    \\    if (oc >= params.out_ch) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos)
    \\            output[oc * strip_len + (pos - row0_pos)] = half(c_stage[tid * 32 + p] + bval);
    \\    }
    \\}
    \\// conv2d over the output rows [row0, row1) with the f32 input held
    \\// strip-locally as [in_ch][in_rows][width] for rows [in_row0, in_row0+in_rows)
    \\// (memory-ladder wall 1, tier 3: the strip finish). The halo rows the
    \\// conv reads must lie inside the strip; the frame-edge zero padding is
    \\// the global check as in conv2d. Same MMA, same accumulation order.
    \\struct ConvStripParams {
    \\    uint in_ch;
    \\    uint out_ch;
    \\    uint height;
    \\    uint width;
    \\    uint ksize;
    \\    uint pad;
    \\    uint dtype;
    \\    uint bias_dtype;
    \\    uint has_bias;
    \\    uint pad1;
    \\    ulong weight_offset;
    \\    ulong bias_offset;
    \\    uint row0;
    \\    uint row1;
    \\    uint in_row0;
    \\    uint in_rows;
    \\};
    \\
    \\kernel void conv2d_strip_in(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvStripParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * params.ksize * params.ksize;
    \\    const device uchar* wbase = weight + params.weight_offset;
    \\    threadgroup float w_stage[32 * 8];
    \\    threadgroup float x_stage[32 * 8];
    \\    threadgroup float c_stage[32 * 32];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint k0 = 0; k0 < k_total; k0 += 8) {
    \\        for (uint kk = 0; kk < 8; kk++) {
    \\            uint k = k0 + kk;
    \\            w_stage[tid * 8 + kk] = 0.0f;
    \\            x_stage[tid * 8 + kk] = 0.0f;
    \\            if (k < k_total) {
    \\                uint kr = (k / params.ksize) % params.ksize;
    \\                uint kc = k % params.ksize;
    \\                uint ic = k / (params.ksize * params.ksize);
    \\                uint oc = tile_oc + tid;
    \\                if (oc < params.out_ch) {
    \\                    uint w_i = ((oc * params.in_ch + ic) * params.ksize + kr) *
    \\                        params.ksize + kc;
    \\                    w_stage[tid * 8 + kk] = read_value(wbase, w_i, params.dtype);
    \\                }
    \\                uint pos = tile_pos + tid;
    \\                if (pos < row1_pos) {
    \\                    uint row = pos / params.width;
    \\                    uint col = pos - row * params.width;
    \\                    int rr = int(row) + int(kr) - int(params.pad);
    \\                    int cc = int(col) + int(kc) - int(params.pad);
    \\                    if (rr >= 0 && cc >= 0 &&
    \\                        rr < int(params.height) && cc < int(params.width) &&
    \\                        rr >= int(params.in_row0) &&
    \\                        rr < int(params.in_row0 + params.in_rows)) {
    \\                        uint in_i = (ic * params.in_rows + (uint(rr) - params.in_row0)) *
    \\                            params.width + uint(cc);
    \\                        x_stage[tid * 8 + kk] = input[in_i];
    \\                    }
    \\                }
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        simdgroup_float8x8 a[4];
    \\        simdgroup_float8x8 b[4];
    \\        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], w_stage + i * 64, 8);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], x_stage + j * 64, 8, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + tid;
    \\    if (oc >= params.out_ch) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[tid * 32 + p] + bval;
    \\    }
    \\}
;
