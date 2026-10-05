//! Metal SwiGLU kernel for resident FFN execution.

pub const swiglu: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void swiglu_f32(
    \\    device float* gate [[buffer(0)]],
    \\    const device float* up [[buffer(1)]],
    \\    constant uint& count [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    if (id >= count) return;
    \\    float g = gate[id];
    \\    gate[id] = (g / (1.0f + exp(-g))) * up[id];
    \\}
    \\
    \\// Fused-GEMM layout: each row of `gateup` holds gate then up halves.
    \\kernel void swiglu_fused_f32(
    \\    const device float* gateup [[buffer(0)]],
    \\    device float* out [[buffer(1)]],
    \\    constant uint& count [[buffer(2)]],
    \\    constant uint& inner [[buffer(3)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    if (id >= count) return;
    \\    uint row = id / inner;
    \\    uint col = id - row * inner;
    \\    float g = gateup[row * 2 * inner + col];
    \\    float u = gateup[row * 2 * inner + inner + col];
    \\    out[id] = (g / (1.0f + exp(-g))) * u;
    \\}
;
