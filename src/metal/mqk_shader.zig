//! Metal Q/K head norm plus Z-Image RoPE.

pub const qk: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct Pair { float c; float s; };
    \\
    \\struct Params {
    \\    uint tokens;
    \\    uint heads;
    \\    uint kv_heads;
    \\    uint head_dim;
    \\    uint q_dtype;
    \\    uint k_dtype;
    \\    uint dim0;
    \\    uint dim1;
    \\    uint dim2;
    \\    float eps;
    \\    ulong q_offset;
    \\    ulong k_offset;
    \\    ulong base0;
    \\    ulong base1;
    \\    ulong base2;
    \\};
    \\
    \\static inline float weight_at(const device uchar* base, uint index, uint dtype) {
    \\    if (dtype == 3) return ((const device float*)base)[index];
    \\    ushort bits = ((const device ushort*)base)[index];
    \\    if (dtype == 2) return as_type<float>(uint(bits) << 16);
    \\    return float(as_type<half>(bits));
    \\}
    \\
    \\static inline ulong axis_base(constant Params& p, uint axis) {
    \\    if (axis == 0) return p.base0;
    \\    if (axis == 1) return p.base1;
    \\    return p.base2;
    \\}
    \\
    \\static inline uint axis_dim(constant Params& p, uint axis) {
    \\    if (axis == 0) return p.dim0;
    \\    if (axis == 1) return p.dim1;
    \\    return p.dim2;
    \\}
    \\
    \\kernel void qk_norm_rope(
    \\    device float* q [[buffer(0)]],
    \\    device float* k [[buffer(1)]],
    \\    const device uchar* q_weight [[buffer(2)]],
    \\    const device uchar* k_weight [[buffer(3)]],
    \\    const device ulong* pos [[buffer(4)]],
    \\    const device Pair* rope [[buffer(5)]],
    \\    constant Params& p [[buffer(6)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint q_groups = p.tokens * p.heads;
    \\    bool is_q = group < q_groups;
    \\    uint local = is_q ? group : group - q_groups;
    \\    uint heads = is_q ? p.heads : p.kv_heads;
    \\    uint tok = local / heads;
    \\    uint head = local - tok * heads;
    \\    device float* data = is_q ? q : k;
    \\    const device uchar* wb = is_q ? q_weight + p.q_offset : k_weight + p.k_offset;
    \\    uint dtype = is_q ? p.q_dtype : p.k_dtype;
    \\    uint base = (tok * heads + head) * p.head_dim;
    \\    threadgroup float reduce[256];
    \\    float local_sum = 0.0f;
    \\    for (uint dim = tid; dim < p.head_dim; dim += tg_size) {
    \\        float value = data[base + dim];
    \\        local_sum += value * value;
    \\    }
    \\    reduce[tid] = local_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) reduce[tid] += reduce[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float scale = rsqrt(reduce[0] / float(p.head_dim) + p.eps);
    \\    uint pair_count = p.head_dim / 2;
    \\    for (uint pair = tid; pair < pair_count; pair += tg_size) {
    \\        uint axis = pair * 2 < p.dim0 ? 0 : (pair * 2 < p.dim0 + p.dim1 ? 1 : 2);
    \\        uint axis_off = axis == 0 ? 0 : (axis == 1 ? p.dim0 : p.dim0 + p.dim1);
    \\        uint local_pair = pair - axis_off / 2;
    \\        uint axis_pairs = axis_dim(p, axis) / 2;
    \\        ulong rope_pos = pos[tok * 3 + axis];
    \\        Pair rot = rope[axis_base(p, axis) + rope_pos * axis_pairs + local_pair];
    \\        uint i = base + pair * 2;
    \\        float aw = scale * weight_at(wb, pair * 2, dtype);
    \\        float bw = scale * weight_at(wb, pair * 2 + 1, dtype);
    \\        float a = data[i] * aw;
    \\        float b = data[i + 1] * bw;
    \\        data[i] = a * rot.c - b * rot.s;
    \\        data[i + 1] = b * rot.c + a * rot.s;
    \\    }
    \\}
;
