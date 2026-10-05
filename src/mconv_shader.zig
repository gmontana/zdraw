//! Metal shader source for the VAE convolution fast path.

pub const conv: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct Params {
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
    \\};
    \\
    \\static inline float f16_value(const device uchar* data, uint index) {
    \\    const device ushort* words = reinterpret_cast<const device ushort*>(data);
    \\    return float(as_type<half>(words[index]));
    \\}
    \\
    \\static inline float bf16_value(const device uchar* data, uint index) {
    \\    const device ushort* words = reinterpret_cast<const device ushort*>(data);
    \\    return as_type<float>(uint(words[index]) << 16);
    \\}
    \\
    \\static inline float f32_value(const device uchar* data, uint index) {
    \\    uint offset = index * 4;
    \\    uint bits = uint(data[offset]) |
    \\        (uint(data[offset + 1]) << 8) |
    \\        (uint(data[offset + 2]) << 16) |
    \\        (uint(data[offset + 3]) << 24);
    \\    return as_type<float>(bits);
    \\}
    \\
    \\static inline float read_value(const device uchar* data, uint index, uint dtype) {
    \\    if (dtype == 1) return f16_value(data, index);
    \\    if (dtype == 2) return bf16_value(data, index);
    \\    if (dtype == 3) return f32_value(data, index);
    \\    return 0.0f;
    \\}
    \\
    \\constant ushort xoff_v2[72] = {
    \\    0, 1, 2, 34, 35, 36, 68, 69, 70,
    \\    102, 103, 104, 136, 137, 138, 170, 171, 172,
    \\    204, 205, 206, 238, 239, 240, 272, 273, 274,
    \\    306, 307, 308, 340, 341, 342, 374, 375, 376,
    \\    408, 409, 410, 442, 443, 444, 476, 477, 478,
    \\    510, 511, 512, 544, 545, 546, 578, 579, 580,
    \\    612, 613, 614, 646, 647, 648, 680, 681, 682,
    \\    714, 715, 716, 748, 749, 750, 782, 783, 784,
    \\};
    \\
    \\kernel void conv2d(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = tg.x * 32;
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
    \\                if (pos < hw) {
    \\                    uint row = pos / params.width;
    \\                    uint col = pos - row * params.width;
    \\                    int rr = int(row) + int(kr) - int(params.pad);
    \\                    int cc = int(col) + int(kc) - int(params.pad);
    \\                    if (rr >= 0 && cc >= 0 &&
    \\                        rr < int(params.height) && cc < int(params.width)) {
    \\                        uint in_i = (ic * params.height + uint(rr)) *
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
    \\        if (pos < hw) output[oc * hw + pos] = c_stage[tid * 32 + p] + bval;
    \\    }
    \\}
    \\
    \\struct ConvWindowParams {
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
    \\};
    \\
    \\// conv2d restricted to the contiguous output row-strip [row0, row1). tg.x
    \\// maps over the strip's flat positions [row0*width, row1*width) via
    \\// tile_pos = row0*width + tg.x*32; everything else - the (row,col) decode,
    \\// the input reads at GLOBAL (row+kr-pad, col+kc-pad) with the SAME zero-pad
    \\// bounds check against the full height/width, the MMA, and the write at the
    \\// GLOBAL output[oc*hw + pos] - is byte-for-byte identical to conv2d. The halo
    \\// (rows row0-1 and row1) is read from the full resident input by global
    \\// coords, so interior strip boundaries read real neighbor rows (NOT zero-pad)
    \\// and only true frame edges zero-pad, exactly as the full conv.
    \\kernel void conv2d_window(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
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
    \\                        rr < int(params.height) && cc < int(params.width)) {
    \\                        uint in_i = (ic * params.height + uint(rr)) *
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
    \\
    \\// conv2d_window_v7: register-tiled scalar-FMA conv. The f32 simdgroup MMA
    \\// path measures ~2 TF/s on M4 (kill-criterion control), but plain fma()
    \\// chains bit-match the MMA's per-element order (v6 probe) and run on the
    \\// full-rate FP32 ALUs. Each thread owns an 8-oc x 4-pos register tile
    \\// (32 accumulators): per k it loads 8 staged weights + 4 halo-tile taps
    \\// and issues 32 FMAs. Weights stage cooperatively as 72-float contiguous
    \\// per-oc rows (perfectly coalesced); outputs store directly from
    \\// registers - no c_stage. Same global K order per output -> bit-identical
    \\// to conv2d_window. Host contract as v2 plus dtype==3.
    \\// conv2d_window_h: f16-feature probe (the f16-VAE rate question). v3-class
    \\// MMA over half operands with f32 accumulators: input/weights f16, output
    \\// f32. Geometry identical to v7/v1 (32 threads, 32x32 tiles). Quality
    \\// class = F16SIM (input rounding only); exists to measure the conv rate
    \\// at f16 MMA before committing to the full f16-VAE plumbing.
    \\// conv2d_window_h4: the h probe with v3's 4-simdgroup operand sharing —
    \\// one x_tile/x_stage build serves four 32-oc tiles (128 threads/TG via
    \\// the oc_x4 dispatch flag). f16 MMA, f32 accumulators.
    \\kernel void conv2d_window_h4(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half x_stage[9][32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup half x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                v = input[(ic0 + ic_rel) * hw + uint(rr) * params.width + uint(gc)];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint c8 = sgid; c8 < 9; c8 += 4) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[c8][lane * 8 + kk] = x_tile[uint(xoff_v2[c8 * 8 + kk]) + lane];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint kbase = ic0 * 9;
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], wf + (tile_oc + i * 8) * k_total + kbase + c8 * 8, k_total);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage[c8] + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + lane;
    \\    if (oc >= params.out_ch) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[sgid][lane * 32 + p] + bval;
    \\    }
    \\}
    \\
    \\kernel void conv2d_window_h(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half x_stage[9][32 * 8];
    \\    threadgroup float c_stage[32 * 32];
    \\    threadgroup half x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                v = input[(ic0 + ic_rel) * hw + uint(rr) * params.width + uint(gc)];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[c8][tid * 8 + kk] = x_tile[uint(xoff_v2[c8 * 8 + kk]) + tid];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint kbase = ic0 * 9;
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], wf + (tile_oc + i * 8) * k_total + kbase + c8 * 8, k_total);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage[c8] + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
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
    \\
    \\kernel void conv2d_window_v7(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device float* wf =
    \\        reinterpret_cast<const device float*>(weight + params.weight_offset);
    \\    uint ocb = (tid & 3u) * 8;
    \\    uint pb = (tid >> 2) * 4;
    \\    threadgroup float w_stage[72 * 33];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    float acc[8][4];
    \\    for (uint o = 0; o < 8; o++)
    \\        for (uint p = 0; p < 4; p++) acc[o][p] = 0.0f;
    \\
    \\    bool has_w = (tile_oc + tid) < params.out_ch;
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                v = input[(ic0 + ic_rel) * hw + uint(rr) * params.width + uint(gc)];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        // Stage this block's 72 weights for each of the tile's 32 oc rows:
    \\        // thread tid streams ITS row's 72 consecutive floats (18 float4s).
    \\        {
    \\            uint kw = (tile_oc + tid) * k_total + ic0 * 9;
    \\            const device float4* wv4 = reinterpret_cast<const device float4*>(wf + kw);
    \\            for (uint q = 0; q < 18; q++) {
    \\                float4 w4 = has_w ? wv4[q] : float4(0.0f);
    \\                w_stage[(q * 4 + 0) * 33 + tid] = w4.x;
    \\                w_stage[(q * 4 + 1) * 33 + tid] = w4.y;
    \\                w_stage[(q * 4 + 2) * 33 + tid] = w4.z;
    \\                w_stage[(q * 4 + 3) * 33 + tid] = w4.w;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint k_rel = 0; k_rel < 72; k_rel++) {
    \\            uint off = uint(xoff_v2[k_rel]);
    \\            float xv[4];
    \\            for (uint p = 0; p < 4; p++) xv[p] = x_tile[off + pb + p];
    \\            float wv[8];
    \\            for (uint o = 0; o < 8; o++) wv[o] = w_stage[k_rel * 33 + ocb + o];
    \\            for (uint o = 0; o < 8; o++)
    \\                for (uint p = 0; p < 4; p++)
    \\                    acc[o][p] = fma(wv[o], xv[p], acc[o][p]);
    \\        }
    \\    }
    \\
    \\    for (uint o = 0; o < 8; o++) {
    \\        uint oc = tile_oc + ocb + o;
    \\        if (oc >= params.out_ch) continue;
    \\        float bval = 0.0f;
    \\        if (params.has_bias != 0) {
    \\            bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\        }
    \\        for (uint p = 0; p < 4; p++) {
    \\            uint pos = tile_pos + pb + p;
    \\            if (pos < row1_pos) output[oc * hw + pos] = acc[o][p] + bval;
    \\        }
    \\    }
    \\}
    \\
    \\
    \\
    \\// conv2d_window_v2: same math, same K order, same 8-wide MMA chunking as
    \\// conv2d_window -> BIT-IDENTICAL output; only the data paths change:
    \\//   - the input tile + 1px halo (8ch x 3rows x 34cols) is cooperatively
    \\//     loaded into threadgroup memory once per ic-block, so the 9 taps read
    \\//     shared memory instead of re-gathering device memory ~9x with
    \\//     per-element bounds/address math;
    \\//   - tap addresses come from a 72-entry constant table (ic*102+kr*34+kc);
    \\//   - the weight row is flat [oc][k_total], read as device float when
    \\//     dtype==3 instead of byte-assembled per element.
    \\// HOST CONTRACT (dispatch v1 otherwise): ksize==3, pad==1, width%32==0,
    \\// in_ch%32==0, and row strips start/end on whole rows. Out-of-frame halo
    \\// zero-fills exactly like v1's bounds check; columns of the MMA beyond
    \\// row1_pos or rows beyond out_ch are never stored, as in v1.
    \\
    \\kernel void conv2d_window_v2(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device uchar* wbase = weight + params.weight_offset;
    \\    const device float* wf = reinterpret_cast<const device float*>(wbase);
    \\    threadgroup float w_stage[32 * 8];
    \\    threadgroup float x_stage[32 * 8];
    \\    threadgroup float c_stage[32 * 32];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    uint oc = tile_oc + tid;
    \\    bool has_oc = oc < params.out_ch;
    \\    uint wrow = oc * k_total;
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                v = input[(ic0 + ic_rel) * hw + uint(rr) * params.width + uint(gc)];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        uint kbase = ic0 * 9;
    \\        for (uint k0 = 0; k0 < 72; k0 += 8) {
    \\            if (has_oc) {
    \\                uint kw = wrow + kbase + k0;
    \\                if (params.dtype == 3) {
    \\                    // 8 consecutive f32 weights: two float4 transactions.
    \\                    const device float4* wv = reinterpret_cast<const device float4*>(wf + kw);
    \\                    float4 w0 = wv[0];
    \\                    float4 w1 = wv[1];
    \\                    w_stage[tid * 8 + 0] = w0.x;
    \\                    w_stage[tid * 8 + 1] = w0.y;
    \\                    w_stage[tid * 8 + 2] = w0.z;
    \\                    w_stage[tid * 8 + 3] = w0.w;
    \\                    w_stage[tid * 8 + 4] = w1.x;
    \\                    w_stage[tid * 8 + 5] = w1.y;
    \\                    w_stage[tid * 8 + 6] = w1.z;
    \\                    w_stage[tid * 8 + 7] = w1.w;
    \\                } else {
    \\                    for (uint kk = 0; kk < 8; kk++)
    \\                        w_stage[tid * 8 + kk] = read_value(wbase, kw + kk, params.dtype);
    \\                }
    \\            } else {
    \\                for (uint kk = 0; kk < 8; kk++) w_stage[tid * 8 + kk] = 0.0f;
    \\            }
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[tid * 8 + kk] = x_tile[uint(xoff_v2[k0 + kk]) + tid];
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            simdgroup_float8x8 a[4];
    \\            simdgroup_float8x8 b[4];
    \\            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], w_stage + i * 64, 8);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (!has_oc) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[tid * 32 + p] + bval;
    \\    }
    \\}
    \\
    \\struct ConvPrenormWindowParams {
    \\    uint in_ch;
    \\    uint out_ch;
    \\    uint height;
    \\    uint width;
    \\    uint ksize;
    \\    uint pad;
    \\    uint dtype;
    \\    uint bias_dtype;
    \\    uint has_bias;
    \\    uint groups;
    \\    ulong weight_offset;
    \\    ulong bias_offset;
    \\    uint row0;
    \\    uint row1;
    \\    uint norm_dtype;
    \\    uint norm_bias_dtype;
    \\    ulong norm_weight_offset;
    \\    ulong norm_bias_offset;
    \\};
    \\
    \\// conv2d_window with GroupNorm+SiLU FUSED into the input read. Instead of
    \\// materializing norm1+SiLU into a full N buffer and convolving it, this loads
    \\// the RAW residual feature element and applies the identical norm transform
    \\// inline before the MAC. The transform is byte-for-byte the apply-then-conv
    \\// path: for an in-bounds input element x at channel ic the value fed to the
    \\// MMA is silu((x-mean[g])*scale[g]*wn[ic]+bn[ic]) with g=ic/(in_ch/groups) -
    \\// the SAME float op order (`(x-mean)*scale`, then `*wn+bn`, then silu) as
    \\// vae_norm_apply_silu_window, whose f32 store/load is lossless. Out-of-frame
    \\// halo elements stay 0.0 (zero-pad), exactly as the standalone conv reads a
    \\// zero there (N holds norm only in-frame). mean/scale come from the global
    \\// vae_norm_stats pass, so the result is bit-identical to conv2d_window over a
    \\// pre-normed N. Everything else (weight staging, MMA, global strip write) is
    \\// identical to conv2d_window.
    \\kernel void conv2d_prenorm_window(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvPrenormWindowParams& params [[buffer(4)]],
    \\    const device float* stats [[buffer(5)]],
    \\    const device uchar* norm_weight [[buffer(6)]],
    \\    const device uchar* norm_bias [[buffer(7)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * params.ksize * params.ksize;
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* wbase = weight + params.weight_offset;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
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
    \\                        rr < int(params.height) && cc < int(params.width)) {
    \\                        uint in_i = (ic * params.height + uint(rr)) *
    \\                            params.width + uint(cc);
    \\                        uint g = ic / group_ch;
    \\                        float mean = stats[g * 2 + 0];
    \\                        float scale = stats[g * 2 + 1];
    \\                        float v = (input[in_i] - mean) * scale;
    \\                        v = v * read_value(nwbase, ic, params.norm_dtype) +
    \\                            read_value(nbbase, ic, params.norm_bias_dtype);
    \\                        x_stage[tid * 8 + kk] = v / (1.0f + exp(-v));
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
    \\
    \\// conv2d_prenorm_window_v3: four simdgroups per threadgroup, each owning a
    \\// DIFFERENT 32-oc tile of the same 32 output positions, SHARING the halo
    \\// tile and the x_stage operand (the expensive loads amortize 4x). Each
    \\// simdgroup's accumulation is the same global K order and 8-chunking as
    \\// conv2d_prenorm_window -> bit-identical. Dispatch: threads (32,4,1),
    \\// grid y = ceil(out_ch/128). Host contract as conv2d_window_v2.
    \\kernel void conv2d_prenorm_window_v3(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvPrenormWindowParams& params [[buffer(4)]],
    \\    const device float* stats [[buffer(5)]],
    \\    const device uchar* norm_weight [[buffer(6)]],
    \\    const device uchar* norm_bias [[buffer(7)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint group_ch = params.in_ch / params.groups;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device uchar* wbase = weight + params.weight_offset;
    \\    const device float* wf = reinterpret_cast<const device float*>(wbase);
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    threadgroup float w_stage[4][32 * 8];
    \\    threadgroup float x_stage[32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    uint oc = tile_oc + lane;
    \\    bool has_oc = oc < params.out_ch;
    \\    uint wrow = oc * k_total;
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                uint ic = ic0 + ic_rel;
    \\                uint g = ic / group_ch;
    \\                float mean = stats[g * 2 + 0];
    \\                float scale = stats[g * 2 + 1];
    \\                uint xi = ic * hw + uint(rr) * params.width + uint(gc);
    \\                float x = (input[xi] - mean) * scale;
    \\                x = x * read_value(nwbase, ic, params.norm_dtype) +
    \\                    read_value(nbbase, ic, params.norm_bias_dtype);
    \\                v = x / (1.0f + exp(-x));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        uint kbase = ic0 * 9;
    \\        for (uint k0 = 0; k0 < 72; k0 += 8) {
    \\            if (has_oc) {
    \\                uint kw = wrow + kbase + k0;
    \\                if (params.dtype == 3) {
    \\                    const device float4* wv = reinterpret_cast<const device float4*>(wf + kw);
    \\                    float4 w0 = wv[0];
    \\                    float4 w1 = wv[1];
    \\                    w_stage[sgid][lane * 8 + 0] = w0.x;
    \\                    w_stage[sgid][lane * 8 + 1] = w0.y;
    \\                    w_stage[sgid][lane * 8 + 2] = w0.z;
    \\                    w_stage[sgid][lane * 8 + 3] = w0.w;
    \\                    w_stage[sgid][lane * 8 + 4] = w1.x;
    \\                    w_stage[sgid][lane * 8 + 5] = w1.y;
    \\                    w_stage[sgid][lane * 8 + 6] = w1.z;
    \\                    w_stage[sgid][lane * 8 + 7] = w1.w;
    \\                } else {
    \\                    for (uint kk = 0; kk < 8; kk++)
    \\                        w_stage[sgid][lane * 8 + kk] =
    \\                            read_value(wbase, kw + kk, params.dtype);
    \\                }
    \\            } else {
    \\                for (uint kk = 0; kk < 8; kk++) w_stage[sgid][lane * 8 + kk] = 0.0f;
    \\            }
    \\            if (sgid == 0) {
    \\                for (uint kk = 0; kk < 8; kk++) {
    \\                    x_stage[lane * 8 + kk] = x_tile[uint(xoff_v2[k0 + kk]) + lane];
    \\                }
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            simdgroup_float8x8 a[4];
    \\            simdgroup_float8x8 b[4];
    \\            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], w_stage[sgid] + i * 64, 8);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (!has_oc) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[sgid][lane * 32 + p] + bval;
    \\    }
    \\}
    \\
    \\struct ConvUpsampleWindowParams {
    \\    uint channels;     // in_ch == out_ch for the VAE upsample conv
    \\    uint out_height;   // 2 * in_height
    \\    uint out_width;    // 2 * in_width
    \\    uint in_height;
    \\    uint in_width;
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
    \\};
    \\
    \\// conv2d_upsample_window_v3: v2's upsample structure with four simdgroups per
    \\// threadgroup (different oc tiles, shared halo + x_stage); grid y =
    \\// ceil(channels/128), threads (32,4,1). Per-acc K order unchanged: the SAME 2x
    \\// nearest index map applied when the tile is filled: tile[ic][r][cc] holds
    \\// src[ic][(out_row+r-1)/2][(col0+cc-1)/2] (zero outside the 2x frame), so
    \\// every MAC input is bit-identical to conv2d_upsample_window. Host contract
    \\// as conv2d_window_v2 (on the OUT grid).
    \\kernel void conv2d_upsample_window_v3(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvUpsampleWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint hw = params.out_height * params.out_width;
    \\    uint row1_pos = params.row1 * params.out_width;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.out_width + tg.x * 32;
    \\    uint k_total = params.channels * 9;
    \\    uint out_row = tile_pos / params.out_width;
    \\    uint col0 = tile_pos - out_row * params.out_width;
    \\    const device uchar* wbase = weight + params.weight_offset;
    \\    const device float* wf = reinterpret_cast<const device float*>(wbase);
    \\    threadgroup float w_stage[4][32 * 8];
    \\    threadgroup float x_stage[32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    uint oc = tile_oc + lane;
    \\    bool has_oc = oc < params.channels;
    \\    uint wrow = oc * k_total;
    \\
    \\    for (uint ic0 = 0; ic0 < params.channels; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.out_height) &&
    \\                gc >= 0 && gc < int(params.out_width)) {
    \\                uint sr = uint(rr) / 2u;
    \\                uint sc = uint(gc) / 2u;
    \\                v = input[((ic0 + ic_rel) * params.in_height + sr) *
    \\                    params.in_width + sc];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        uint kbase = ic0 * 9;
    \\        for (uint k0 = 0; k0 < 72; k0 += 8) {
    \\            if (has_oc) {
    \\                uint kw = wrow + kbase + k0;
    \\                if (params.dtype == 3) {
    \\                    const device float4* wv = reinterpret_cast<const device float4*>(wf + kw);
    \\                    float4 w0 = wv[0];
    \\                    float4 w1 = wv[1];
    \\                    w_stage[sgid][lane * 8 + 0] = w0.x;
    \\                    w_stage[sgid][lane * 8 + 1] = w0.y;
    \\                    w_stage[sgid][lane * 8 + 2] = w0.z;
    \\                    w_stage[sgid][lane * 8 + 3] = w0.w;
    \\                    w_stage[sgid][lane * 8 + 4] = w1.x;
    \\                    w_stage[sgid][lane * 8 + 5] = w1.y;
    \\                    w_stage[sgid][lane * 8 + 6] = w1.z;
    \\                    w_stage[sgid][lane * 8 + 7] = w1.w;
    \\                } else {
    \\                    for (uint kk = 0; kk < 8; kk++)
    \\                        w_stage[sgid][lane * 8 + kk] =
    \\                            read_value(wbase, kw + kk, params.dtype);
    \\                }
    \\            } else {
    \\                for (uint kk = 0; kk < 8; kk++) w_stage[sgid][lane * 8 + kk] = 0.0f;
    \\            }
    \\            if (sgid == 0) {
    \\                for (uint kk = 0; kk < 8; kk++) {
    \\                    x_stage[lane * 8 + kk] = x_tile[uint(xoff_v2[k0 + kk]) + lane];
    \\                }
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            simdgroup_float8x8 a[4];
    \\            simdgroup_float8x8 b[4];
    \\            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], w_stage[sgid] + i * 64, 8);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (!has_oc) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[sgid][lane * 32 + p] + bval;
    \\    }
    \\}
    \\
    \\kernel void conv2d_upsample_window(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvUpsampleWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.out_height * params.out_width;
    \\    uint row1_pos = params.row1 * params.out_width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.out_width + tg.x * 32;
    \\    uint k_total = params.channels * params.ksize * params.ksize;
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
    \\                if (oc < params.channels) {
    \\                    uint w_i = ((oc * params.channels + ic) * params.ksize + kr) *
    \\                        params.ksize + kc;
    \\                    w_stage[tid * 8 + kk] = read_value(wbase, w_i, params.dtype);
    \\                }
    \\                uint pos = tile_pos + tid;
    \\                if (pos < row1_pos) {
    \\                    uint row = pos / params.out_width;
    \\                    uint col = pos - row * params.out_width;
    \\                    int rr = int(row) + int(kr) - int(params.pad);
    \\                    int cc = int(col) + int(kc) - int(params.pad);
    \\                    if (rr >= 0 && cc >= 0 &&
    \\                        rr < int(params.out_height) && cc < int(params.out_width)) {
    \\                        uint sr = uint(rr) / 2u;
    \\                        uint sc = uint(cc) / 2u;
    \\                        uint in_i = (ic * params.in_height + sr) *
    \\                            params.in_width + sc;
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
    \\    if (oc >= params.channels) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[tid * 32 + p] + bval;
    \\    }
    \\}
    \\
    \\kernel void conv2d_prenorm_window_v7(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvPrenormWindowParams& params [[buffer(4)]],
    \\    const device float* stats [[buffer(5)]],
    \\    const device uchar* norm_weight [[buffer(6)]],
    \\    const device uchar* norm_bias [[buffer(7)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device float* wf =
    \\        reinterpret_cast<const device float*>(weight + params.weight_offset);
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    uint ocb = (tid & 3u) * 8;
    \\    uint pb = (tid >> 2) * 4;
    \\    threadgroup float w_stage[72 * 33];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    float acc[8][4];
    \\    for (uint o = 0; o < 8; o++)
    \\        for (uint p = 0; p < 4; p++) acc[o][p] = 0.0f;
    \\
    \\    bool has_w = (tile_oc + tid) < params.out_ch;
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                uint ic = ic0 + ic_rel;
    \\                uint g = ic / group_ch;
    \\                float mean = stats[g * 2 + 0];
    \\                float scale = stats[g * 2 + 1];
    \\                uint xi = ic * hw + uint(rr) * params.width + uint(gc);
    \\                float x = (input[xi] - mean) * scale;
    \\                x = x * read_value(nwbase, ic, params.norm_dtype) +
    \\                    read_value(nbbase, ic, params.norm_bias_dtype);
    \\                v = x / (1.0f + exp(-x));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        // Stage this block's 72 weights for each of the tile's 32 oc rows:
    \\        // thread tid streams ITS row's 72 consecutive floats (18 float4s).
    \\        {
    \\            uint kw = (tile_oc + tid) * k_total + ic0 * 9;
    \\            const device float4* wv4 = reinterpret_cast<const device float4*>(wf + kw);
    \\            for (uint q = 0; q < 18; q++) {
    \\                float4 w4 = has_w ? wv4[q] : float4(0.0f);
    \\                w_stage[(q * 4 + 0) * 33 + tid] = w4.x;
    \\                w_stage[(q * 4 + 1) * 33 + tid] = w4.y;
    \\                w_stage[(q * 4 + 2) * 33 + tid] = w4.z;
    \\                w_stage[(q * 4 + 3) * 33 + tid] = w4.w;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint k_rel = 0; k_rel < 72; k_rel++) {
    \\            uint off = uint(xoff_v2[k_rel]);
    \\            float xv[4];
    \\            for (uint p = 0; p < 4; p++) xv[p] = x_tile[off + pb + p];
    \\            float wv[8];
    \\            for (uint o = 0; o < 8; o++) wv[o] = w_stage[k_rel * 33 + ocb + o];
    \\            for (uint o = 0; o < 8; o++)
    \\                for (uint p = 0; p < 4; p++)
    \\                    acc[o][p] = fma(wv[o], xv[p], acc[o][p]);
    \\        }
    \\    }
    \\
    \\    for (uint o = 0; o < 8; o++) {
    \\        uint oc = tile_oc + ocb + o;
    \\        if (oc >= params.out_ch) continue;
    \\        float bval = 0.0f;
    \\        if (params.has_bias != 0) {
    \\            bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\        }
    \\        for (uint p = 0; p < 4; p++) {
    \\            uint pos = tile_pos + pb + p;
    \\            if (pos < row1_pos) output[oc * hw + pos] = acc[o][p] + bval;
    \\        }
    \\    }
    \\}
    \\kernel void conv2d_upsample_window_v7(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvUpsampleWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.out_height * params.out_width;
    \\    uint row1_pos = params.row1 * params.out_width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.out_width + tg.x * 32;
    \\    uint k_total = params.channels * 9;
    \\    uint out_row = tile_pos / params.out_width;
    \\    uint col0 = tile_pos - out_row * params.out_width;
    \\    const device float* wf =
    \\        reinterpret_cast<const device float*>(weight + params.weight_offset);
    \\    uint ocb = (tid & 3u) * 8;
    \\    uint pb = (tid >> 2) * 4;
    \\    threadgroup float w_stage[72 * 33];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    float acc[8][4];
    \\    for (uint o = 0; o < 8; o++)
    \\        for (uint p = 0; p < 4; p++) acc[o][p] = 0.0f;
    \\
    \\    bool has_w = (tile_oc + tid) < params.channels;
    \\    for (uint ic0 = 0; ic0 < params.channels; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.out_height) &&
    \\                gc >= 0 && gc < int(params.out_width)) {
    \\                uint sr = uint(rr) / 2u;
    \\                uint sc = uint(gc) / 2u;
    \\                v = input[((ic0 + ic_rel) * params.in_height + sr) *
    \\                    params.in_width + sc];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        // Stage this block's 72 weights for each of the tile's 32 oc rows:
    \\        // thread tid streams ITS row's 72 consecutive floats (18 float4s).
    \\        {
    \\            uint kw = (tile_oc + tid) * k_total + ic0 * 9;
    \\            const device float4* wv4 = reinterpret_cast<const device float4*>(wf + kw);
    \\            for (uint q = 0; q < 18; q++) {
    \\                float4 w4 = has_w ? wv4[q] : float4(0.0f);
    \\                w_stage[(q * 4 + 0) * 33 + tid] = w4.x;
    \\                w_stage[(q * 4 + 1) * 33 + tid] = w4.y;
    \\                w_stage[(q * 4 + 2) * 33 + tid] = w4.z;
    \\                w_stage[(q * 4 + 3) * 33 + tid] = w4.w;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint k_rel = 0; k_rel < 72; k_rel++) {
    \\            uint off = uint(xoff_v2[k_rel]);
    \\            float xv[4];
    \\            for (uint p = 0; p < 4; p++) xv[p] = x_tile[off + pb + p];
    \\            float wv[8];
    \\            for (uint o = 0; o < 8; o++) wv[o] = w_stage[k_rel * 33 + ocb + o];
    \\            for (uint o = 0; o < 8; o++)
    \\                for (uint p = 0; p < 4; p++)
    \\                    acc[o][p] = fma(wv[o], xv[p], acc[o][p]);
    \\        }
    \\    }
    \\
    \\    for (uint o = 0; o < 8; o++) {
    \\        uint oc = tile_oc + ocb + o;
    \\        if (oc >= params.channels) continue;
    \\        float bval = 0.0f;
    \\        if (params.has_bias != 0) {
    \\            bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\        }
    \\        for (uint p = 0; p < 4; p++) {
    \\            uint pos = tile_pos + pb + p;
    \\            if (pos < row1_pos) output[oc * hw + pos] = acc[o][p] + bval;
    \\        }
    \\    }
    \\}
    \\
    \\kernel void conv2d_window_f16sim(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
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
    \\                        rr < int(params.height) && cc < int(params.width)) {
    \\                        uint in_i = (ic * params.height + uint(rr)) *
    \\                            params.width + uint(cc);
    \\                        x_stage[tid * 8 + kk] = float(half(input[in_i]));
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
    \\
    \\// conv2d_window_v7: register-tiled scalar-FMA conv. The f32 simdgroup MMA
    \\// path measures ~2 TF/s on M4 (kill-criterion control), but plain fma()
    \\// chains bit-match the MMA's per-element order (v6 probe) and run on the
    \\// full-rate FP32 ALUs. Each thread owns an 8-oc x 4-pos register tile
    \\// (32 accumulators): per k it loads 8 staged weights + 4 halo-tile taps
    \\// and issues 32 FMAs. Weights stage cooperatively as 72-float contiguous
    \\// per-oc rows (perfectly coalesced); outputs store directly from
    \\// registers - no c_stage. Same global K order per output -> bit-identical
    \\// to conv2d_window. Host contract as v2 plus dtype==3.
    \\kernel void conv2d_prenorm_window_f16sim(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvPrenormWindowParams& params [[buffer(4)]],
    \\    const device float* stats [[buffer(5)]],
    \\    const device uchar* norm_weight [[buffer(6)]],
    \\    const device uchar* norm_bias [[buffer(7)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device float* wf =
    \\        reinterpret_cast<const device float*>(weight + params.weight_offset);
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    uint ocb = (tid & 3u) * 8;
    \\    uint pb = (tid >> 2) * 4;
    \\    threadgroup float w_stage[72 * 33];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    float acc[8][4];
    \\    for (uint o = 0; o < 8; o++)
    \\        for (uint p = 0; p < 4; p++) acc[o][p] = 0.0f;
    \\
    \\    bool has_w = (tile_oc + tid) < params.out_ch;
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                uint ic = ic0 + ic_rel;
    \\                uint g = ic / group_ch;
    \\                float mean = stats[g * 2 + 0];
    \\                float scale = stats[g * 2 + 1];
    \\                uint xi = ic * hw + uint(rr) * params.width + uint(gc);
    \\                float x = (float(half(input[xi])) - mean) * scale;
    \\                x = x * read_value(nwbase, ic, params.norm_dtype) +
    \\                    read_value(nbbase, ic, params.norm_bias_dtype);
    \\                v = x / (1.0f + exp(-x));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        // Stage this block's 72 weights for each of the tile's 32 oc rows:
    \\        // thread tid streams ITS row's 72 consecutive floats (18 float4s).
    \\        {
    \\            uint kw = (tile_oc + tid) * k_total + ic0 * 9;
    \\            const device float4* wv4 = reinterpret_cast<const device float4*>(wf + kw);
    \\            for (uint q = 0; q < 18; q++) {
    \\                float4 w4 = has_w ? wv4[q] : float4(0.0f);
    \\                w_stage[(q * 4 + 0) * 33 + tid] = w4.x;
    \\                w_stage[(q * 4 + 1) * 33 + tid] = w4.y;
    \\                w_stage[(q * 4 + 2) * 33 + tid] = w4.z;
    \\                w_stage[(q * 4 + 3) * 33 + tid] = w4.w;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint k_rel = 0; k_rel < 72; k_rel++) {
    \\            uint off = uint(xoff_v2[k_rel]);
    \\            float xv[4];
    \\            for (uint p = 0; p < 4; p++) xv[p] = x_tile[off + pb + p];
    \\            float wv[8];
    \\            for (uint o = 0; o < 8; o++) wv[o] = w_stage[k_rel * 33 + ocb + o];
    \\            for (uint o = 0; o < 8; o++)
    \\                for (uint p = 0; p < 4; p++)
    \\                    acc[o][p] = fma(wv[o], xv[p], acc[o][p]);
    \\        }
    \\    }
    \\
    \\    for (uint o = 0; o < 8; o++) {
    \\        uint oc = tile_oc + ocb + o;
    \\        if (oc >= params.out_ch) continue;
    \\        float bval = 0.0f;
    \\        if (params.has_bias != 0) {
    \\            bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\        }
    \\        for (uint p = 0; p < 4; p++) {
    \\            uint pos = tile_pos + pb + p;
    \\            if (pos < row1_pos) output[oc * hw + pos] = acc[o][p] + bval;
    \\        }
    \\    }
    \\}
    \\
    \\kernel void conv2d_upsample_window_f16sim(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvUpsampleWindowParams& params [[buffer(4)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.out_height * params.out_width;
    \\    uint row1_pos = params.row1 * params.out_width;
    \\    uint tile_oc = tg.y * 32;
    \\    uint tile_pos = params.row0 * params.out_width + tg.x * 32;
    \\    uint k_total = params.channels * 9;
    \\    uint out_row = tile_pos / params.out_width;
    \\    uint col0 = tile_pos - out_row * params.out_width;
    \\    const device float* wf =
    \\        reinterpret_cast<const device float*>(weight + params.weight_offset);
    \\    uint ocb = (tid & 3u) * 8;
    \\    uint pb = (tid >> 2) * 4;
    \\    threadgroup float w_stage[72 * 33];
    \\    threadgroup float x_tile[8 * 3 * 34];
    \\
    \\    float acc[8][4];
    \\    for (uint o = 0; o < 8; o++)
    \\        for (uint p = 0; p < 4; p++) acc[o][p] = 0.0f;
    \\
    \\    bool has_w = (tile_oc + tid) < params.channels;
    \\    for (uint ic0 = 0; ic0 < params.channels; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 32) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            float v = 0.0f;
    \\            if (rr >= 0 && rr < int(params.out_height) &&
    \\                gc >= 0 && gc < int(params.out_width)) {
    \\                uint sr = uint(rr) / 2u;
    \\                uint sc = uint(gc) / 2u;
    \\                v = float(half(input[((ic0 + ic_rel) * params.in_height + sr) *
    \\                    params.in_width + sc]));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        // Stage this block's 72 weights for each of the tile's 32 oc rows:
    \\        // thread tid streams ITS row's 72 consecutive floats (18 float4s).
    \\        {
    \\            uint kw = (tile_oc + tid) * k_total + ic0 * 9;
    \\            const device float4* wv4 = reinterpret_cast<const device float4*>(wf + kw);
    \\            for (uint q = 0; q < 18; q++) {
    \\                float4 w4 = has_w ? wv4[q] : float4(0.0f);
    \\                w_stage[(q * 4 + 0) * 33 + tid] = w4.x;
    \\                w_stage[(q * 4 + 1) * 33 + tid] = w4.y;
    \\                w_stage[(q * 4 + 2) * 33 + tid] = w4.z;
    \\                w_stage[(q * 4 + 3) * 33 + tid] = w4.w;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint k_rel = 0; k_rel < 72; k_rel++) {
    \\            uint off = uint(xoff_v2[k_rel]);
    \\            float xv[4];
    \\            for (uint p = 0; p < 4; p++) xv[p] = x_tile[off + pb + p];
    \\            float wv[8];
    \\            for (uint o = 0; o < 8; o++) wv[o] = w_stage[k_rel * 33 + ocb + o];
    \\            for (uint o = 0; o < 8; o++)
    \\                for (uint p = 0; p < 4; p++)
    \\                    acc[o][p] = fma(wv[o], xv[p], acc[o][p]);
    \\        }
    \\    }
    \\
    \\    for (uint o = 0; o < 8; o++) {
    \\        uint oc = tile_oc + ocb + o;
    \\        if (oc >= params.channels) continue;
    \\        float bval = 0.0f;
    \\        if (params.has_bias != 0) {
    \\            bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\        }
    \\        for (uint p = 0; p < 4; p++) {
    \\            uint pos = tile_pos + pb + p;
    \\            if (pos < row1_pos) output[oc * hw + pos] = acc[o][p] + bval;
    \\        }
    \\    }
    \\}
    \\kernel void conv2d_prenorm_window_h4(
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
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half x_stage[9][32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup half x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                uint ic = ic0 + ic_rel;
    \\                uint g = ic / group_ch;
    \\                float mean = stats[g * 2 + 0];
    \\                float scale = stats[g * 2 + 1];
    \\                uint xi = ic * hw + uint(rr) * params.width + uint(gc);
    \\                float x = (float(input[xi]) - mean) * scale;
    \\                x = x * read_value(nwbase, ic, params.norm_dtype) +
    \\                    read_value(nbbase, ic, params.norm_bias_dtype);
    \\                v = half(x / (1.0f + exp(-x)));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint c8 = sgid; c8 < 9; c8 += 4) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[c8][lane * 8 + kk] = x_tile[uint(xoff_v2[c8 * 8 + kk]) + lane];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint kbase = ic0 * 9;
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], wf + (tile_oc + i * 8) * k_total + kbase + c8 * 8, k_total);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage[c8] + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + lane;
    \\    if (oc >= params.out_ch) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = half(c_stage[sgid][lane * 32 + p] + bval);
    \\    }
    \\}
    \\kernel void conv2d_upsample_window_h4(
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
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.out_width + tg.x * 32;
    \\    uint k_total = params.channels * 9;
    \\    uint out_row = tile_pos / params.out_width;
    \\    uint col0 = tile_pos - out_row * params.out_width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half x_stage[9][32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup half x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.channels; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(params.out_height) &&
    \\                gc >= 0 && gc < int(params.out_width)) {
    \\                uint sr = uint(rr) / 2u;
    \\                uint sc = uint(gc) / 2u;
    \\                v = input[((ic0 + ic_rel) * params.in_height + sr) *
    \\                    params.in_width + sc];
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint c8 = sgid; c8 < 9; c8 += 4) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[c8][lane * 8 + kk] = x_tile[uint(xoff_v2[c8 * 8 + kk]) + lane];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint kbase = ic0 * 9;
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], wf + (tile_oc + i * 8) * k_total + kbase + c8 * 8, k_total);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage[c8] + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + lane;
    \\    if (oc >= params.channels) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = half(c_stage[sgid][lane * 32 + p] + bval);
    \\    }
    \\}
    \\
    \\// conv1x1_h: f16 1x1 projection (the resblock skip) for the f16-VAE
    \\// path. Same 32oc x 32pos tile as the windowed convs, but with k=in_ch
    \\// and no halo/tap loop. This replaces the old scalar loop that made one
    \\// thread serially produce 32 output channels for one position.
    \\kernel void conv1x1_h(
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
    \\        if (pos < row1_pos) output[oc * hw + pos] = half(c_stage[tid * 32 + p] + bval);
    \\    }
    \\}
    \\
    \\
    \\kernel void conv2d_prenorm_window_h4c1(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant ConvPrenormWindowParams& params [[buffer(4)]],
    \\    const device float* stats [[buffer(5)]],
    \\    const device uchar* norm_weight [[buffer(6)]],
    \\    const device uchar* norm_bias [[buffer(7)]],
    \\    uint2 tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint hw = params.height * params.width;
    \\    uint row1_pos = params.row1 * params.width;
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half x_stage[9][32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup half x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                uint ic = ic0 + ic_rel;
    \\                uint g = ic / group_ch;
    \\                float mean = stats[g * 2 + 0];
    \\                float scale = stats[g * 2 + 1];
    \\                uint xi = ic * hw + uint(rr) * params.width + uint(gc);
    \\                float x = (float(input[xi]) - mean) * scale;
    \\                x = x * read_value(nwbase, ic, params.norm_dtype) +
    \\                    read_value(nbbase, ic, params.norm_bias_dtype);
    \\                v = half(x / (1.0f + exp(-x)));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint c8 = sgid; c8 < 9; c8 += 4) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[c8][lane * 8 + kk] = x_tile[uint(xoff_v2[c8 * 8 + kk]) + lane];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint kbase = ic0 * 9;
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], wf + (tile_oc + i * 8) * k_total + kbase + c8 * 8, k_total);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage[c8] + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + lane;
    \\    if (oc >= params.out_ch) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = c_stage[sgid][lane * 32 + p] + bval;
    \\    }
    \\}
    \\kernel void conv2d_prenorm_window_h4c2(
    \\    const device float* input [[buffer(0)]],
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
    \\    uint group_ch = params.in_ch / params.groups;
    \\    const device uchar* nwbase = norm_weight + params.norm_weight_offset;
    \\    const device uchar* nbbase = norm_bias + params.norm_bias_offset;
    \\    uint sgid = tid >> 5;
    \\    uint lane = tid & 31u;
    \\    uint tile_oc = (tg.y * 4 + sgid) * 32;
    \\    uint tile_pos = params.row0 * params.width + tg.x * 32;
    \\    uint k_total = params.in_ch * 9;
    \\    uint out_row = tile_pos / params.width;
    \\    uint col0 = tile_pos - out_row * params.width;
    \\    const device half* wf =
    \\        reinterpret_cast<const device half*>(weight + params.weight_offset);
    \\    threadgroup half x_stage[9][32 * 8];
    \\    threadgroup float c_stage[4][32 * 32];
    \\    threadgroup half x_tile[8 * 3 * 34];
    \\
    \\    simdgroup_float8x8 acc[4][4];
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\
    \\    for (uint ic0 = 0; ic0 < params.in_ch; ic0 += 8) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint idx = tid; idx < 8 * 3 * 34; idx += 128) {
    \\            uint ic_rel = idx / 102;
    \\            uint rem = idx - ic_rel * 102;
    \\            uint r = rem / 34;
    \\            uint cc = rem - r * 34;
    \\            int rr = int(out_row) + int(r) - 1;
    \\            int gc = int(col0) + int(cc) - 1;
    \\            half v = half(0.0f);
    \\            if (rr >= 0 && rr < int(params.height) &&
    \\                gc >= 0 && gc < int(params.width)) {
    \\                uint ic = ic0 + ic_rel;
    \\                uint g = ic / group_ch;
    \\                float mean = stats[g * 2 + 0];
    \\                float scale = stats[g * 2 + 1];
    \\                uint xi = ic * hw + uint(rr) * params.width + uint(gc);
    \\                float x = (input[xi] - mean) * scale;
    \\                x = x * read_value(nwbase, ic, params.norm_dtype) +
    \\                    read_value(nbbase, ic, params.norm_bias_dtype);
    \\                v = half(x / (1.0f + exp(-x)));
    \\            }
    \\            x_tile[idx] = v;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (uint c8 = sgid; c8 < 9; c8 += 4) {
    \\            for (uint kk = 0; kk < 8; kk++) {
    \\                x_stage[c8][lane * 8 + kk] = x_tile[uint(xoff_v2[c8 * 8 + kk]) + lane];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        uint kbase = ic0 * 9;
    \\        for (uint c8 = 0; c8 < 9; c8++) {
    \\            simdgroup_half8x8 a[4];
    \\            simdgroup_half8x8 b[4];
    \\            for (uint i = 0; i < 4; i++)
    \\                simdgroup_load(a[i], wf + (tile_oc + i * 8) * k_total + kbase + c8 * 8, k_total);
    \\            for (uint j = 0; j < 4; j++)
    \\                simdgroup_load(b[j], x_stage[c8] + j * 64, 8, ulong2(0, 0), true);
    \\            for (uint i = 0; i < 4; i++)
    \\                for (uint j = 0; j < 4; j++)
    \\                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++)
    \\        for (uint j = 0; j < 4; j++)
    \\            simdgroup_store(acc[i][j], c_stage[sgid] + i * 8 * 32 + j * 8, 32);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    uint oc = tile_oc + lane;
    \\    if (oc >= params.out_ch) return;
    \\    float bval = 0.0f;
    \\    if (params.has_bias != 0) {
    \\        bval = read_value(bias + params.bias_offset, oc, params.bias_dtype);
    \\    }
    \\    for (uint p = 0; p < 32; p++) {
    \\        uint pos = tile_pos + p;
    \\        if (pos < row1_pos) output[oc * hw + pos] = half(c_stage[sgid][lane * 32 + p] + bval);
    \\    }
    \\}
    \\
    \\
;
