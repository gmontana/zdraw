//! Per-step adaLN modulation vectors on the GPU (klein-mods-gpu-20260827):
//! out[n] = sum_k act[k] * W[n][k] in f32, k ascending, no contraction - the
//! arithmetic of ops.linearView on the CPU, so the vectors are byte-identical
//! and the ~200 ms per step of host-side matvec leaves the denoise. W is read
//! raw from the mapped weight (f16, bf16 or f32; exact conversions).
pub const src: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#pragma METAL fp contract(off)
    \\using namespace metal;
    \\
    \\struct ModVecParams { uint n; uint k; uint dtype; uint pad; };
    \\
    \\kernel void kmodvec(
    \\    const device uchar* W [[buffer(0)]],
    \\    const device float* act [[buffer(1)]],
    \\    device float* out [[buffer(2)]],
    \\    constant ModVecParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n) return;
    \\    const uint base = gid * p.k;
    \\    float sum = 0.0f;
    \\    if (p.dtype == 2u) {
    \\        const device ushort* row = (const device ushort*)W + base;
    \\        for (uint c = 0; c < p.k; c++) {
    \\            const float w = as_type<float>(uint(row[c]) << 16);
    \\            sum = sum + act[c] * w;
    \\        }
    \\    } else if (p.dtype == 1u) {
    \\        const device half* row = (const device half*)W + base;
    \\        for (uint c = 0; c < p.k; c++) sum = sum + act[c] * float(row[c]);
    \\    } else {
    \\        const device float* row = (const device float*)W + base;
    \\        for (uint c = 0; c < p.k; c++) sum = sum + act[c] * row[c];
    \\    }
    \\    out[gid] = sum;
    \\}
;
