//! Final projection kernels for Z-Image.

pub const final: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct FinalNormParams { uint tokens; uint hidden; float eps; uint pad; };
    \\
    \\kernel void final_norm_scale(
    \\    const device float* input [[buffer(0)]],
    \\    const device float* scale [[buffer(1)]],
    \\    device float* output [[buffer(2)]],
    \\    constant FinalNormParams& p [[buffer(3)]],
    \\    uint token [[thread_position_in_grid]]
    \\) {
    \\    if (token >= p.tokens) return;
    \\    const uint base = token * p.hidden;
    \\    float mean = 0.0f;
    \\    for (uint i = 0; i < p.hidden; i++) mean += input[base + i];
    \\    mean /= float(p.hidden);
    \\
    \\    float var_sum = 0.0f;
    \\    for (uint i = 0; i < p.hidden; i++) {
    \\        const float diff = input[base + i] - mean;
    \\        var_sum += diff * diff;
    \\    }
    \\    const float inv = 1.0f / sqrt(var_sum / float(p.hidden) + p.eps);
    \\    for (uint i = 0; i < p.hidden; i++) {
    \\        output[base + i] = (input[base + i] - mean) * inv * scale[i];
    \\    }
    \\}
;
