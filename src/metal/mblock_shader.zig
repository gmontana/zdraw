//! Metal block-local RMSNorm and residual kernels.

pub const block: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint hidden;
    \\    uint dtype;
    \\    uint has_scale;
    \\    float eps;
    \\    ulong weight_offset;
    \\};
    \\
    \\static inline float weight_at(const device uchar* base, uint index, uint dtype) {
    \\    if (dtype == 3) return ((const device float*)base)[index];
    \\    ushort bits = ((const device ushort*)base)[index];
    \\    if (dtype == 2) return as_type<float>(uint(bits) << 16);
    \\    return float(as_type<half>(bits));
    \\}
    \\
    \\kernel void block_norm_scale(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight_bytes [[buffer(1)]],
    \\    const device float* scale [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint tok [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint base = tok * p.hidden;
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    float local = 0.0f;
    \\    for (uint dim = tid; dim < p.hidden; dim += tg_size) {
    \\        float value = input[base + dim];
    \\        local += value * value;
    \\    }
    \\    reduce[tid] = local;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float norm = rsqrt(reduce[0] / float(p.hidden) + p.eps);
    \\    for (uint dim = tid; dim < p.hidden; dim += tg_size) {
    \\        float value = input[base + dim] * norm * weight_at(weight, dim, p.dtype);
    \\        if (p.has_scale != 0) value *= scale[dim];
    \\        output[base + dim] = value;
    \\    }
    \\}
    \\
    \\kernel void block_residual_norm(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight_bytes [[buffer(1)]],
    \\    const device float* gate [[buffer(2)]],
    \\    device float* state [[buffer(3)]],
    \\    constant Params& p [[buffer(4)]],
    \\    uint tok [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint base = tok * p.hidden;
    \\    const device uchar* weight = weight_bytes + p.weight_offset;
    \\    float local = 0.0f;
    \\    for (uint dim = tid; dim < p.hidden; dim += tg_size) {
    \\        float value = input[base + dim];
    \\        local += value * value;
    \\    }
    \\    reduce[tid] = local;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float norm = rsqrt(reduce[0] / float(p.hidden) + p.eps);
    \\    for (uint dim = tid; dim < p.hidden; dim += tg_size) {
    \\        float value = input[base + dim] * norm * weight_at(weight, dim, p.dtype);
    \\        if (p.has_scale != 0) value *= gate[dim];
    \\        state[base + dim] += value;
    \\    }
    \\}
    \\
    \\kernel void block_residual_next_norm(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* resid_weight_bytes [[buffer(1)]],
    \\    const device float* gate [[buffer(2)]],
    \\    device float* state [[buffer(3)]],
    \\    const device uchar* norm_weight_bytes [[buffer(4)]],
    \\    const device float* scale [[buffer(5)]],
    \\    device float* output [[buffer(6)]],
    \\    constant Params& resid [[buffer(7)]],
    \\    constant Params& norm_p [[buffer(8)]],
    \\    uint tok [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float reduce[256];
    \\    uint base = tok * resid.hidden;
    \\    const device uchar* rw = resid_weight_bytes + resid.weight_offset;
    \\    float local = 0.0f;
    \\    for (uint dim = tid; dim < resid.hidden; dim += tg_size) {
    \\        float value = input[base + dim];
    \\        local += value * value;
    \\    }
    \\    reduce[tid] = local;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float resid_norm = rsqrt(reduce[0] / float(resid.hidden) + resid.eps);
    \\    float state_sum = 0.0f;
    \\    for (uint dim = tid; dim < resid.hidden; dim += tg_size) {
    \\        float inc = input[base + dim] * resid_norm * weight_at(rw, dim, resid.dtype);
    \\        if (resid.has_scale != 0) inc *= gate[dim];
    \\        float next = state[base + dim] + inc;
    \\        state[base + dim] = next;
    \\        state_sum += next * next;
    \\    }
    \\    reduce[tid] = state_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float norm = rsqrt(reduce[0] / float(norm_p.hidden) + norm_p.eps);
    \\    const device uchar* nw = norm_weight_bytes + norm_p.weight_offset;
    \\    for (uint dim = tid; dim < norm_p.hidden; dim += tg_size) {
    \\        float value = state[base + dim] * norm * weight_at(nw, dim, norm_p.dtype);
    \\        if (norm_p.has_scale != 0) value *= scale[dim];
    \\        output[base + dim] = value;
    \\    }
    \\}
;
