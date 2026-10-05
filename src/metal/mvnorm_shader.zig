//! Metal kernels for resident VAE residual blocks.

pub const vnorm: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct NormParams {
    \\    uint channels;
    \\    uint height;
    \\    uint width;
    \\    uint groups;
    \\    uint dtype;
    \\    uint bias_dtype;
    \\    float eps;
    \\    ulong weight_offset;
    \\    ulong bias_offset;
    \\};
    \\
    \\static inline float read_value(const device uchar* data, uint index, uint dtype) {
    \\    if (dtype == 3) return ((const device float*)data)[index];
    \\    ushort bits = ((const device ushort*)data)[index];
    \\    if (dtype == 2) return as_type<float>(uint(bits) << 16);
    \\    return float(as_type<half>(bits));
    \\}
    \\
    \\kernel void vae_norm_silu(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight_bytes [[buffer(1)]],
    \\    const device uchar* bias_bytes [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant NormParams& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += input[ch * hw + pos];
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = input[ch * hw + pos] - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    const device uchar* bias = bias_bytes + p.bias_offset;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float v = (input[ch * hw + pos] - mean) * scale;
    \\        v = v * read_value(weight, ch, p.dtype) +
    \\            read_value(bias, ch, p.bias_dtype);
    \\        output[ch * hw + pos] = v / (1.0f + exp(-v));
    \\    }
    \\}
    \\
    \\kernel void vae_norm_silu_h(
    \\    const device half* input [[buffer(0)]],
    \\    const device uchar* weight_bytes [[buffer(1)]],
    \\    const device uchar* bias_bytes [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant NormParams& p [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += float(input[ch * hw + pos]);
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = float(input[ch * hw + pos]) - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    const device uchar* bias = bias_bytes + p.bias_offset;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float v = (float(input[ch * hw + pos]) - mean) * scale;
    \\        v = v * read_value(weight, ch, p.dtype) +
    \\            read_value(bias, ch, p.bias_dtype);
    \\        output[ch * hw + pos] = v / (1.0f + exp(-v));
    \\    }
    \\}
    \\
    \\// Exact-streaming split of vae_norm_silu, behind ZDRAW_VAE_STREAM. The two
    \\// reduction passes below are copied VERBATIM from vae_norm_silu (same loops,
    \\// same accumulation order) so mean/scale are bit-identical; only the final
    \\// write differs (stats buffer instead of the SiLU apply). One threadgroup
    \\// per group, same dispatch as encode_vae_norm.
    \\kernel void vae_norm_stats(
    \\    const device float* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += input[ch * hw + pos];
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = input[ch * hw + pos] - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    if (tid == 0) {
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = scale;
    \\    }
    \\}
    \\
    \\kernel void vae_norm_stats_fast(
    \\    const device float* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[512];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += input[ch * hw + pos];
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = input[ch * hw + pos] - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    if (tid == 0) {
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = scale;
    \\    }
    \\}
    \\
    \\kernel void vae_norm_stats_sq(
    \\    const device float* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce_sum[512];
    \\    threadgroup float reduce_sq[512];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    float sq = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float v = input[ch * hw + pos];
    \\        sum += v;
    \\        sq = fma(v, v, sq);
    \\    }
    \\    reduce_sum[tid] = sum;
    \\    reduce_sq[tid] = sq;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) {
    \\            reduce_sum[tid] += reduce_sum[tid + stride];
    \\            reduce_sq[tid] += reduce_sq[tid + stride];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) {
    \\        float mean = reduce_sum[0] / float(n);
    \\        float var = max(reduce_sq[0] / float(n) - mean * mean, 0.0f);
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = rsqrt(var + p.eps);
    \\    }
    \\}
    \\
    \\kernel void vae_norm_stats_f16sim(
    \\    const device float* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += float(half(input[ch * hw + pos]));
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = float(half(input[ch * hw + pos])) - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    if (tid == 0) {
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = scale;
    \\    }
    \\}
    \\
    \\struct NormWindowParams {
    \\    uint channels;
    \\    uint height;
    \\    uint width;
    \\    uint groups;
    \\    uint dtype;
    \\    uint bias_dtype;
    \\    float eps;
    \\    ulong weight_offset;
    \\    ulong bias_offset;
    \\    uint row0;
    \\    uint row1;
    \\    uint col0;
    \\    uint col1;
    \\};
    \\
    \\// Pure per-pixel apply of the precomputed (mean, scale) over a window
    \\// [row0,row1) x [col0,col1) in global H x W coordinates. The value
    \\// expression is identical to vae_norm_silu's apply loop, so it is trivially
    \\// bit-exact there; only written pixels are touched. One threadgroup per group.
    \\kernel void vae_norm_apply_silu_window(
    \\    const device float* input [[buffer(0)]],
    \\    const device float* stats [[buffer(1)]],
    \\    const device uchar* weight_bytes [[buffer(2)]],
    \\    const device uchar* bias_bytes [[buffer(3)]],
    \\    device float* output [[buffer(4)]],
    \\    constant NormWindowParams& p [[buffer(5)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint win_h = p.row1 - p.row0;
    \\    uint win_w = p.col1 - p.col0;
    \\    uint win = win_h * win_w;
    \\    uint n = group_ch * win;
    \\    float mean = stats[group * 2 + 0];
    \\    float scale = stats[group * 2 + 1];
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    const device uchar* bias = bias_bytes + p.bias_offset;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / win;
    \\        uint wpos = i - (i / win) * win;
    \\        uint row = p.row0 + wpos / win_w;
    \\        uint col = p.col0 + wpos - (wpos / win_w) * win_w;
    \\        uint pos = row * p.width + col;
    \\        float v = (input[ch * hw + pos] - mean) * scale;
    \\        v = v * read_value(weight, ch, p.dtype) +
    \\            read_value(bias, ch, p.bias_dtype);
    \\        output[ch * hw + pos] = v / (1.0f + exp(-v));
    \\    }
    \\}
    \\
    \\kernel void vae_norm_apply_silu_window_h(
    \\    const device half* input [[buffer(0)]],
    \\    const device float* stats [[buffer(1)]],
    \\    const device uchar* weight_bytes [[buffer(2)]],
    \\    const device uchar* bias_bytes [[buffer(3)]],
    \\    device half* output [[buffer(4)]],
    \\    constant NormWindowParams& p [[buffer(5)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint win_h = p.row1 - p.row0;
    \\    uint win_w = p.col1 - p.col0;
    \\    uint win = win_h * win_w;
    \\    uint n = group_ch * win;
    \\    float mean = stats[group * 2 + 0];
    \\    float scale = stats[group * 2 + 1];
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    const device uchar* bias = bias_bytes + p.bias_offset;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / win;
    \\        uint wpos = i - (i / win) * win;
    \\        uint row = p.row0 + wpos / win_w;
    \\        uint col = p.col0 + wpos - (wpos / win_w) * win_w;
    \\        uint pos = row * p.width + col;
    \\        float v = (float(input[ch * hw + pos]) - mean) * scale;
    \\        v = v * read_value(weight, ch, p.dtype) +
    \\            read_value(bias, ch, p.bias_dtype);
    \\        output[ch * hw + pos] = half(v / (1.0f + exp(-v)));
    \\    }
    \\}
    \\// vae_norm_apply_strip_hf: vae_norm_apply_silu_window_h with a float
    \\// output stored window-locally as [channels][row1-row0][col1-col0]: the
    \\// strip finish's norm scratch (memory-ladder wall 1, tier 3). The value
    \\// expression is vae_norm_silu_h's apply, so the floats are identical.
    \\kernel void vae_norm_apply_strip_hf(
    \\    const device half* input [[buffer(0)]],
    \\    const device float* stats [[buffer(1)]],
    \\    const device uchar* weight_bytes [[buffer(2)]],
    \\    const device uchar* bias_bytes [[buffer(3)]],
    \\    device float* output [[buffer(4)]],
    \\    constant NormWindowParams& p [[buffer(5)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint win_h = p.row1 - p.row0;
    \\    uint win_w = p.col1 - p.col0;
    \\    uint win = win_h * win_w;
    \\    uint n = group_ch * win;
    \\    float mean = stats[group * 2 + 0];
    \\    float scale = stats[group * 2 + 1];
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    const device uchar* bias = bias_bytes + p.bias_offset;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / win;
    \\        uint wpos = i - (i / win) * win;
    \\        uint row = p.row0 + wpos / win_w;
    \\        uint col = p.col0 + wpos - (wpos / win_w) * win_w;
    \\        uint pos = row * p.width + col;
    \\        float v = (float(input[ch * hw + pos]) - mean) * scale;
    \\        v = v * read_value(weight, ch, p.dtype) +
    \\            read_value(bias, ch, p.bias_dtype);
    \\        output[ch * win + wpos] = v / (1.0f + exp(-v));
    \\    }
    \\}
    \\
    \\kernel void vae_add(
    \\    device float* output [[buffer(0)]],
    \\    const device float* residual [[buffer(1)]],
    \\    constant uint& count [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid < count) output[gid] += residual[gid];
    \\}
    \\
    \\struct AddWindowParams {
    \\    uint channels;
    \\    uint height;
    \\    uint width;
    \\    uint row0;
    \\    uint row1;
    \\};
    \\
    \\// Exact-streaming residual add restricted to the contiguous output row-strip
    \\// [row0,row1) across all channels. The per-element op is the SAME `output +=
    \\// residual` as vae_add (addition is order-independent), and the global index
    \\// output[ch*hw + row*width + col] is identical, so it is bit-exact there;
    \\// only the strip's pixels are touched. One thread per strip element.
    \\kernel void vae_add_window(
    \\    device float* output [[buffer(0)]],
    \\    const device float* residual [[buffer(1)]],
    \\    constant AddWindowParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint strip = (p.row1 - p.row0) * p.width;
    \\    uint total = p.channels * strip;
    \\    if (gid >= total) return;
    \\    uint ch = gid / strip;
    \\    uint spos = gid - ch * strip;
    \\    uint pos = p.row0 * p.width + spos;
    \\    uint idx = ch * hw + pos;
    \\    output[idx] += residual[idx];
    \\}
    \\
    \\struct UpParams { uint channels; uint height; uint width; };
    \\
    \\kernel void vae_upsample2(
    \\    const device float* input [[buffer(0)]],
    \\    device float* output [[buffer(1)]],
    \\    constant UpParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint out_h = p.height * 2;
    \\    uint out_w = p.width * 2;
    \\    uint hw = out_h * out_w;
    \\    uint total = p.channels * hw;
    \\    if (gid >= total) return;
    \\    uint ch = gid / hw;
    \\    uint pos = gid - ch * hw;
    \\    uint row = pos / out_w;
    \\    uint col = pos - row * out_w;
    \\    uint in_i = (ch * p.height + row / 2) * p.width + col / 2;
    \\    output[gid] = input[in_i];
    \\}
    \\
    \\kernel void vae_upsample2_h(
    \\    const device half* input [[buffer(0)]],
    \\    device half* output [[buffer(1)]],
    \\    constant UpParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint out_h = p.height * 2;
    \\    uint out_w = p.width * 2;
    \\    uint hw = out_h * out_w;
    \\    uint total = p.channels * hw;
    \\    if (gid >= total) return;
    \\    uint ch = gid / hw;
    \\    uint pos = gid - ch * hw;
    \\    uint row = pos / out_w;
    \\    uint col = pos - row * out_w;
    \\    uint in_i = (ch * p.height + row / 2) * p.width + col / 2;
    \\    output[gid] = input[in_i];
    \\}
    \\
    \\kernel void vae_norm_stats_h(
    \\    const device half* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += float(input[ch * hw + pos]);
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = float(input[ch * hw + pos]) - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    if (tid == 0) {
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = scale;
    \\    }
    \\}
    \\kernel void vae_norm_stats_h_fast(
    \\    const device half* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[512];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        sum += float(input[ch * hw + pos]);
    \\    }
    \\    reduce[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = reduce[0] / float(n);
    \\    float var_sum = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float diff = float(input[ch * hw + pos]) - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    reduce[tid] = var_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(n) + p.eps);
    \\    if (tid == 0) {
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = scale;
    \\    }
    \\}
    \\kernel void vae_norm_stats_h_sq(
    \\    const device half* input [[buffer(0)]],
    \\    device float* stats [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce_sum[512];
    \\    threadgroup float reduce_sq[512];
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float sum = 0.0f;
    \\    float sq = 0.0f;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float v = float(input[ch * hw + pos]);
    \\        sum += v;
    \\        sq = fma(v, v, sq);
    \\    }
    \\    reduce_sum[tid] = sum;
    \\    reduce_sq[tid] = sq;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) {
    \\            reduce_sum[tid] += reduce_sum[tid + stride];
    \\            reduce_sq[tid] += reduce_sq[tid + stride];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) {
    \\        float mean = reduce_sum[0] / float(n);
    \\        float var = max(reduce_sq[0] / float(n) - mean * mean, 0.0f);
    \\        stats[group * 2 + 0] = mean;
    \\        stats[group * 2 + 1] = rsqrt(var + p.eps);
    \\    }
    \\}
    \\kernel void vae_add_window_h(
    \\    device half* output [[buffer(0)]],
    \\    const device half* residual [[buffer(1)]],
    \\    constant AddWindowParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint strip = (p.row1 - p.row0) * p.width;
    \\    uint total = p.channels * strip;
    \\    if (gid >= total) return;
    \\    uint ch = gid / strip;
    \\    uint spos = gid - ch * strip;
    \\    uint pos = p.row0 * p.width + spos;
    \\    uint idx = ch * hw + pos;
    \\    output[idx] = half(float(output[idx]) + float(residual[idx]));
    \\}
    \\// vae_add_strip_h: vae_add_window_h with the residual stored strip-locally
    \\// as [channels][row1-row0][width] (memory-ladder wall 1); same
    \\// arithmetic, different load index.
    \\kernel void vae_add_strip_h(
    \\    device half* output [[buffer(0)]],
    \\    const device half* residual [[buffer(1)]],
    \\    constant AddWindowParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint strip = (p.row1 - p.row0) * p.width;
    \\    uint total = p.channels * strip;
    \\    if (gid >= total) return;
    \\    uint ch = gid / strip;
    \\    uint spos = gid - ch * strip;
    \\    uint pos = p.row0 * p.width + spos;
    \\    uint idx = ch * hw + pos;
    \\    output[idx] = half(float(output[idx]) + float(residual[ch * strip + spos]));
    \\}
    \\// vae_add_strip_rev_h: the no-skip in-place add (tier 2): the conv2 result
    \\// sits strip-locally in `residual`, the input in `output`; the operands
    \\// are written in today's textual order (conv2 result + input).
    \\kernel void vae_add_strip_rev_h(
    \\    device half* output [[buffer(0)]],
    \\    const device half* residual [[buffer(1)]],
    \\    constant AddWindowParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint strip = (p.row1 - p.row0) * p.width;
    \\    uint total = p.channels * strip;
    \\    if (gid >= total) return;
    \\    uint ch = gid / strip;
    \\    uint spos = gid - ch * strip;
    \\    uint pos = p.row0 * p.width + spos;
    \\    uint idx = ch * hw + pos;
    \\    output[idx] = half(float(residual[ch * strip + spos]) + float(output[idx]));
    \\}
    \\// vae_rows_copy_h: the halo stash fill (tier 2): rows [row0, row1) of every
    \\// channel copied strip-locally into `stash` before the strip overwrites them.
    \\kernel void vae_rows_copy_h(
    \\    const device half* input [[buffer(0)]],
    \\    device half* stash [[buffer(1)]],
    \\    constant AddWindowParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint strip = (p.row1 - p.row0) * p.width;
    \\    uint total = p.channels * strip;
    \\    if (gid >= total) return;
    \\    uint ch = gid / strip;
    \\    uint spos = gid - ch * strip;
    \\    stash[ch * strip + spos] = input[ch * hw + p.row0 * p.width + spos];
    \\}
    \\
    \\// Mid-attention GroupNorm apply WITHOUT SiLU, fused with the NCHW-to-
    \\// token-major transpose (replaces the CPU gnorm + gatherAll pair). The
    \\// value expression matches vae_norm_silu's apply loop minus the SiLU
    \\// line; the transposed write is a pure permutation. One threadgroup per
    \\// group, stats from vae_norm_stats.
    \\kernel void vae_norm_apply_seq(
    \\    const device float* input [[buffer(0)]],
    \\    const device float* stats [[buffer(1)]],
    \\    const device uchar* weight_bytes [[buffer(2)]],
    \\    const device uchar* bias_bytes [[buffer(3)]],
    \\    device float* output [[buffer(4)]],
    \\    constant NormParams& p [[buffer(5)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint group_ch = p.channels / p.groups;
    \\    uint start = group * group_ch;
    \\    uint n = group_ch * hw;
    \\    float mean = stats[group * 2 + 0];
    \\    float scale = stats[group * 2 + 1];
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    const device uchar* bias = bias_bytes + p.bias_offset;
    \\    for (uint i = tid; i < n; i += tg_size) {
    \\        uint ch = start + i / hw;
    \\        uint pos = i - (i / hw) * hw;
    \\        float v = (input[ch * hw + pos] - mean) * scale;
    \\        v = v * read_value(weight, ch, p.dtype) +
    \\            read_value(bias, ch, p.bias_dtype);
    \\        output[(ulong)pos * p.channels + ch] = v;
    \\    }
    \\}
    \\
    \\// Plain RNE f32-to-f16 cast for the resident mid handoffs, matching the
    \\// CPU @floatCast the streamed set upload performs (deliberately NOT the
    \\// clamped/scaled probe kernel in metal_api.m).
    \\kernel void vae_f32_to_f16(
    \\    const device float* input [[buffer(0)]],
    \\    device half* output [[buffer(1)]],
    \\    constant uint& count [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid < count) output[gid] = half(input[gid]);
    \\}
    \\
    \\// Token-major attention output added back into the NCHW feature (replaces
    \\// the CPU copy + scatterAdd). One thread per element; each element
    \\// receives exactly one add, so the result is order-independent and exact.
    \\kernel void vae_seq_scatter_add(
    \\    device float* feature [[buffer(0)]],
    \\    const device float* seq [[buffer(1)]],
    \\    constant NormParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint hw = p.height * p.width;
    \\    uint total = p.channels * hw;
    \\    if (gid >= total) return;
    \\    uint ch = gid / hw;
    \\    uint pos = gid - ch * hw;
    \\    feature[gid] += seq[(ulong)pos * p.channels + ch];
    \\}
;
