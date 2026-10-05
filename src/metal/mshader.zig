//! Metal shader source used by the first GPU linear path.

pub const linear: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct Params {
    \\    uint rows;
    \\    uint cols;
    \\    uint batch;
    \\    uint dtype;
    \\    uint bias_dtype;
    \\    uint has_bias;
    \\    uint pad;
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
    \\kernel void linear_rows(
    \\    const device float* input [[buffer(0)]],
    \\    const device uchar* weight [[buffer(1)]],
    \\    const device uchar* bias [[buffer(2)]],
    \\    device float* output [[buffer(3)]],
    \\    constant Params& params [[buffer(4)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tg_size [[threads_per_threadgroup]]
    \\) {
    \\    uint row = group % params.rows;
    \\    uint first = (group / params.rows) * 4;
    \\    bool have0 = first < params.batch;
    \\    bool have1 = first + 1 < params.batch;
    \\    bool have2 = first + 2 < params.batch;
    \\    bool have3 = first + 3 < params.batch;
    \\    threadgroup float4 sums[256];
    \\    float4 sum = float4(0.0f);
    \\    for (uint col = tid; col < params.cols; col += tg_size) {
    \\        uint index = row * params.cols + col;
    \\        float w = read_value(weight + params.weight_offset, index, params.dtype);
    \\        if (have0) sum.x += input[(first + 0) * params.cols + col] * w;
    \\        if (have1) sum.y += input[(first + 1) * params.cols + col] * w;
    \\        if (have2) sum.z += input[(first + 2) * params.cols + col] * w;
    \\        if (have3) sum.w += input[(first + 3) * params.cols + col] * w;
    \\    }
    \\    sums[tid] = sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    \\        if (tid < stride) sums[tid] += sums[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) {
    \\        float4 value = sums[0];
    \\        float bias_value = 0.0f;
    \\        if (params.has_bias != 0) {
    \\            bias_value = read_value(bias + params.bias_offset, row, params.bias_dtype);
    \\        }
    \\        if (have0) output[(first + 0) * params.rows + row] = value.x + bias_value;
    \\        if (have1) output[(first + 1) * params.rows + row] = value.y + bias_value;
    \\        if (have2) output[(first + 2) * params.rows + row] = value.z + bias_value;
    \\        if (have3) output[(first + 3) * params.rows + row] = value.w + bias_value;
    \\    }
    \\}
;
