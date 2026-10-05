//! Metal kernel for the resident Qwen3 encoder: fused weighted per-head
//! RMSNorm + split-half rope. Neither existing rope kernel fits: krms_rope
//! pairs (i, i+1) FLUX-style, qk_norm_rope is Z-Image's 3-axis layout; Qwen
//! norms then rotates (i, i+half) against per-position tables.

pub const src: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct QRopeParams {
    \\    uint tokens;
    \\    uint heads;
    \\    uint head_dim;
    \\    uint w_dtype;
    \\    float eps;
    \\    uint pad;
    \\    ulong w_offset;
    \\};
    \\
    \\static inline float weight_at(const device uchar* base, uint index, uint dtype) {
    \\    if (dtype == 3) return ((const device float*)base)[index];
    \\    ushort bits = ((const device ushort*)base)[index];
    \\    if (dtype == 2) return as_type<float>(uint(bits) << 16);
    \\    return float(as_type<half>(bits));
    \\}
    \\
    \\// One thread per (token, head). The squared sum runs sequentially in
    \\// registers, matching the CPU reference's accumulation order; the cos/sin
    \\// tables are CPU-built (bit-identical transcendentals) and strided
    \\// head_dim/2 per token. Norm weight reads raw checkpoint bytes.
    \\kernel void qrms_rope(
    \\    device float* x [[buffer(0)]],
    \\    const device uchar* weight_bytes [[buffer(1)]],
    \\    const device float* cosb [[buffer(2)]],
    \\    const device float* sinb [[buffer(3)]],
    \\    constant QRopeParams& p [[buffer(4)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.tokens * p.heads) return;
    \\    uint t = gid / p.heads;
    \\    uint hd = p.head_dim / 2;
    \\    const device uchar* w = weight_bytes + p.w_offset;
    \\    device float* row = x + (ulong)gid * p.head_dim;
    \\    float ss = 0.0f;
    \\    for (uint i = 0; i < p.head_dim; i++) ss += row[i] * row[i];
    \\    float inv = rsqrt(ss / float(p.head_dim) + p.eps);
    \\    const device float* c = cosb + (ulong)t * hd;
    \\    const device float* s = sinb + (ulong)t * hd;
    \\    for (uint i = 0; i < hd; i++) {
    \\        float a = row[i] * inv * weight_at(w, i, p.w_dtype);
    \\        float b = row[i + hd] * inv * weight_at(w, i + hd, p.w_dtype);
    \\        row[i] = a * c[i] - b * s[i];
    \\        row[i + hd] = b * c[i] + a * s[i];
    \\    }
    \\}
;
