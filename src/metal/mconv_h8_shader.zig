//! W7 conv kernels (vae-conv-h8-20260827): the 8-simdgroup, staged-weight
//! versions of conv2d_prenorm_window_h4 / conv2d_upsample_window_h4. Appended
//! to mconv_shader.conv, so they share its params structs, read_value and
//! includes.
const base = @import("mconv_shader.zig");

/// The full windowed-conv library: mconv_shader.conv plus the W7 kernels.
pub const conv: [:0]const u8 = base.conv ++ src;

const src: [:0]const u8 =
    \\// W7: 8-simdgroup (256-thread) windowed conv. A threadgroup tiles 128 output
    \\// channels x 64 positions (simdgroup sgid: oc (sgid>>1)*32, pos (sgid&1)*32).
    \\// Per 8-channel block the 3x66 window is built once for 64 positions and the
    \\// 128 oc x 72 k weight chunk is staged in threadgroup memory through half4
    \\// loads, double-buffered in three 24-k stages (the gemm_f16a_direct pattern),
    \\// so the MMA A operand never comes from device memory (the _h4 kernels load A
    \\// straight from the weight buffer with a k_total stride, once per 32
    \\// positions). Same operand values, same k order and the same MMA sequence as
    \\// _h4, so the output is byte-identical. Threadgroup memory 21.2 KB; the
    \\// epilogue aliases the first 16 KB as the f32 C stage. Host contract: width %
    \\// 64 == 0, out_ch % 128 == 0, in_ch % 8 == 0, weight_offset % 8 == 0.
    \\constant ushort xoff_h8[72] = {
    \\    0, 1, 2, 66, 67, 68, 132, 133, 134,
    \\    198, 199, 200, 264, 265, 266, 330, 331, 332,
    \\    396, 397, 398, 462, 463, 464, 528, 529, 530,
    \\    594, 595, 596, 660, 661, 662, 726, 727, 728,
    \\    792, 793, 794, 858, 859, 860, 924, 925, 926,
    \\    990, 991, 992, 1056, 1057, 1058, 1122, 1123, 1124,
    \\    1188, 1189, 1190, 1254, 1255, 1256, 1320, 1321, 1322,
    \\    1386, 1387, 1388, 1452, 1453, 1454, 1518, 1519, 1520,
    \\};
    \\
    \\static inline void h8_stage_w(const device half* wf, threadgroup half* dst,
    \\                              uint tile_oc, uint k_total, uint kbase, uint tid) {
    \\    for (uint q = tid; q < 384; q += 256) {
    \\        uint r = q / 3;
    \\        uint seg = q - r * 3;
    \\        const device half4* s4 = reinterpret_cast<const device half4*>(
    \\            wf + (tile_oc + r) * k_total + kbase + seg * 8);
    \\        threadgroup half4* d4 =
    \\            reinterpret_cast<threadgroup half4*>(dst + r * 24 + seg * 8);
    \\        d4[0] = s4[0];
    \\        d4[1] = s4[1];
    \\    }
    \\}
    \\// One 24-k stage of the B operand: 3 taps x [64 pos][8 kk] gathered from the
    \\// window (dst index == q, so the writes are contiguous per thread).
    \\static inline void h8_stage_x(const threadgroup half* x_tile, threadgroup half* dst,
    \\                              uint s, uint tid) {
    \\    for (uint q = tid; q < 1536; q += 256) {
    \\        uint t = q >> 9;
    \\        uint rem = q & 511u;
    \\        uint p = rem >> 3;
    \\        uint kk = rem & 7u;
    \\        dst[q] = x_tile[uint(xoff_h8[(s * 3 + t) * 8 + kk]) + p];
    \\    }
    \\}
    \\static inline void h8_mma(const threadgroup half* ws, const threadgroup half* xs,
    \\                          uint sg_oc, uint sg_pos,
    \\                          thread simdgroup_float8x8 (&acc)[4][4]) {
    \\    for (uint t = 0; t < 3; t++) {
    \\        simdgroup_half8x8 a[4];
    \\        simdgroup_half8x8 b[4];
    \\        for (uint i = 0; i < 4; i++)
    \\            simdgroup_load(a[i], ws + (sg_oc + i * 8) * 24 + t * 8, 24);
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_load(b[j], xs + t * 512 + (sg_pos + j * 8) * 8, 8, ulong2(0, 0), true);
    \\        for (uint i = 0; i < 4; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\    }
    \\}
    \\// Epilogue: two 16-oc halves through the aliased f32 stage; each lane writes
    \\// one oc row x 16 positions as four half4.
    \\static inline void h8_store(threadgroup float* cst, device half* output,
    \\                            const device uchar* bias, ulong bias_offset,
    \\                            uint bias_dtype, uint has_bias, uint hw,
    \\                            uint oc0, uint pos0, uint row1_pos, uint sgid, uint lane,
    \\                            thread simdgroup_float8x8 (&acc)[4][4]) {
    \\    uint orow = lane & 15u;
    \\    uint pq = lane >> 4;
    \\    for (uint h = 0; h < 2; h++) {
    \\        for (uint i = 0; i < 2; i++)
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_store(acc[h * 2 + i][j], cst + sgid * 512 + i * 8 * 32 + j * 8, 32);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint oc = oc0 + h * 16 + orow;
    \\        float bval = 0.0f;
    \\        if (has_bias != 0) bval = read_value(bias + bias_offset, oc, bias_dtype);
    \\        uint pos = pos0 + pq * 16;
    \\        if (pos + 16 <= row1_pos) {
    \\            const threadgroup float4* src = reinterpret_cast<const threadgroup float4*>(
    \\                cst + sgid * 512 + orow * 32 + pq * 16);
    \\            device half4* dst = reinterpret_cast<device half4*>(output + oc * hw + pos);
    \\            for (uint q = 0; q < 4; q++) dst[q] = half4(src[q] + bval);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\}
    \\
    \\kernel void conv2d_prenorm_window_h8(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device half* output [[buffer(3)]],
    \\    constant ConvPrenormWindowParams& params [[buffer(4)]],
    \\    const device float* stats [[buffer(5)]],
    \\    const device uchar* norm_weight [[buffer(6)]],
    \\    const device uchar* norm_bias [[buffer(7)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint width = params.width;
    \\    uint height = params.height;
    \\    uint in_ch = params.in_ch;
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = tg.y * 128;
    \\    uint sg_oc = (sgid >> 1) * 32;
    \\    uint sg_pos = (sgid & 1u) * 32;
    \\    uint tile_pos = params.row0 * width + tg.x * 64;
    \\    uint k_total = in_ch * 9;
    \\    uint out_row = tile_pos / width;
    \\    uint col0 = tile_pos - out_row * width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup float4 smem4[1356];
    \\    threadgroup half* w_stage = reinterpret_cast<threadgroup half*>(smem4);
    \\    threadgroup half* x_stage = w_stage + 6144;
    \\    threadgroup half* x_tile = x_stage + 3072;
    \\    threadgroup float nparm[32];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    if (tid < 8) {
    \\        uint g = tid / group_ch;
    \\        nparm[tid * 4 + 0] = stats[g * 2 + 0];
    \\        nparm[tid * 4 + 1] = stats[g * 2 + 1];
    \\        nparm[tid * 4 + 2] = read_value(nwbase, tid, params.norm_dtype);
    \\        nparm[tid * 4 + 3] = read_value(nbbase, tid, params.norm_bias_dtype);
    \\    }
    \\    h8_stage_w(wf, w_stage, tile_oc, k_total, 0, tid);
    \\    uint cur = 0;
    \\    for (uint ic0 = 0; ic0 < in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 1584; idx += 256) {
    \\            uint ic_rel = idx / 198;
    \\            uint rem = idx - ic_rel * 198;
    \\            uint r = rem / 66;
    \\            uint cc = rem - r * 66;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(height) && gc >= 0 && gc < int(width)) {
    \\                uint xi = (ic0 + ic_rel) * hw + uint(rr) * width + uint(gc);
    \\                float x = (float(input[xi]) - nparm[ic_rel * 4 + 0]) * nparm[ic_rel * 4 + 1];
    \\                x = x * nparm[ic_rel * 4 + 2] + nparm[ic_rel * 4 + 3];
    \\                v = half(x / (1.0f + exp(-x)));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        h8_stage_x(x_tile, x_stage, 0, tid);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint s = 0; s < 3; s++) {
    \\            if (s < 2) {
    \\                h8_stage_w(wf, w_stage + (1 - cur) * 3072, tile_oc, k_total,
    \\                           ic0 * 9 + (s + 1) * 24, tid);
    \\                h8_stage_x(x_tile, x_stage + ((s + 1) & 1u) * 1536, s + 1, tid);
    \\            } else if (ic0 + 8 < in_ch) {
    \\                h8_stage_w(wf, w_stage + (1 - cur) * 3072, tile_oc, k_total,
    \\                           (ic0 + 8) * 9, tid);
    \\                if (tid < 8) {
    \\                    uint ic = ic0 + 8 + tid;
    \\                    uint g = ic / group_ch;
    \\                    nparm[tid * 4 + 0] = stats[g * 2 + 0];
    \\                    nparm[tid * 4 + 1] = stats[g * 2 + 1];
    \\                    nparm[tid * 4 + 2] = read_value(nwbase, ic, params.norm_dtype);
    \\                    nparm[tid * 4 + 3] = read_value(nbbase, ic, params.norm_bias_dtype);
    \\                }
    \\            }
    \\            h8_mma(w_stage + cur * 3072, x_stage + (s & 1u) * 1536, sg_oc, sg_pos, acc);
    \\            cur = 1 - cur;
    \\            if (s < 2) threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    h8_store(reinterpret_cast<threadgroup float*>(smem4), output, bias,
    \\             params.bias_offset, params.bias_dtype, params.has_bias, hw,
    \\             tile_oc + sg_oc, tile_pos + sg_pos, row1_pos, sgid, lane, acc);
    \\}
    \\
    \\kernel void conv2d_upsample_window_h8(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device half* output [[buffer(3)]],
    \\    constant ConvUpsampleWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.out_height * params.out_width;
    \\    uint row1_pos = params.row1 * params.out_width;
    \\    uint width = params.out_width;
    \\    uint height = params.out_height;
    \\    uint in_ch = params.channels;
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = tg.y * 128;
    \\    uint sg_oc = (sgid >> 1) * 32;
    \\    uint sg_pos = (sgid & 1u) * 32;
    \\    uint tile_pos = params.row0 * width + tg.x * 64;
    \\    uint k_total = in_ch * 9;
    \\    uint out_row = tile_pos / width;
    \\    uint col0 = tile_pos - out_row * width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup float4 smem4[1356];
    \\    threadgroup half* w_stage = reinterpret_cast<threadgroup half*>(smem4);
    \\    threadgroup half* x_stage = w_stage + 6144;
    \\    threadgroup half* x_tile = x_stage + 3072;
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    h8_stage_w(wf, w_stage, tile_oc, k_total, 0, tid);
    \\    uint cur = 0;
    \\    for (uint ic0 = 0; ic0 < in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 1584; idx += 256) {
    \\            uint ic_rel = idx / 198;
    \\            uint rem = idx - ic_rel * 198;
    \\            uint r = rem / 66;
    \\            uint cc = rem - r * 66;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(height) && gc >= 0 && gc < int(width)) {
    \\                uint sr = uint(rr) / 2u;
    \\                uint sc = uint(gc) / 2u;
    \\                v = input[((ic0 + ic_rel) * params.in_height + sr) * params.in_width + sc];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        h8_stage_x(x_tile, x_stage, 0, tid);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint s = 0; s < 3; s++) {
    \\            if (s < 2) {
    \\                h8_stage_w(wf, w_stage + (1 - cur) * 3072, tile_oc, k_total,
    \\                           ic0 * 9 + (s + 1) * 24, tid);
    \\                h8_stage_x(x_tile, x_stage + ((s + 1) & 1u) * 1536, s + 1, tid);
    \\            } else if (ic0 + 8 < in_ch) {
    \\                h8_stage_w(wf, w_stage + (1 - cur) * 3072, tile_oc, k_total,
    \\                           (ic0 + 8) * 9, tid);
    \\            }
    \\            h8_mma(w_stage + cur * 3072, x_stage + (s & 1u) * 1536, sg_oc, sg_pos, acc);
    \\            cur = 1 - cur;
    \\            if (s < 2) threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    h8_store(reinterpret_cast<threadgroup float*>(smem4), output, bias,
    \\             params.bias_offset, params.bias_dtype, params.has_bias, hw,
    \\             tile_oc + sg_oc, tile_pos + sg_pos, row1_pos, sgid, lane, acc);
    \\}
    \\
;
