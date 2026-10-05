//! Metal glue kernels for the resident FLUX.2 Klein forward: the CPU-side
//! norm/mod/rope/swiglu/gate ops of the correctness path, moved on-GPU so
//! activations never leave the device between GEMMs.

pub const src: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct GlueParams {
    \\    uint tokens;
    \\    uint hidden;
    \\    uint heads;
    \\    uint head_dim;
    \\};
    \\
    \\// Rowwise LayerNorm (no affine, eps 1e-6) fused with adaLN modulation:
    \\// out = (1 + scale) * norm(x) + shift. One threadgroup per token row.
    \\kernel void kln_mod(
    \\    const device float* x [[buffer(0)]],
    \\    device float* out [[buffer(1)]],
    \\    const device float* shift [[buffer(2)]],
    \\    const device float* scale [[buffer(3)]],
    \\    constant GlueParams& p [[buffer(4)]],
    \\    uint tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tcount [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float red[256];
    \\    const device float* row = x + (ulong)tg * p.hidden;
    \\    float s = 0.0f;
    \\    for (uint i = tid; i < p.hidden; i += tcount) s += row[i];
    \\    red[tid] = s;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] += red[tid + st];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = red[0] / p.hidden;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float v = 0.0f;
    \\    for (uint i = tid; i < p.hidden; i += tcount) {
    \\        float d = row[i] - mean;
    \\        v += d * d;
    \\    }
    \\    red[tid] = v;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] += red[tid + st];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv = rsqrt(red[0] / p.hidden + 1e-6f);
    \\    device float* dst = out + (ulong)tg * p.hidden;
    \\    for (uint i = tid; i < p.hidden; i += tcount) {
    \\        float n = (row[i] - mean) * inv;
    \\        dst[i] = (1.0f + scale[i]) * n + shift[i];
    \\    }
    \\}
    \\
    \\// Per-(token, head) RMS norm (learned 128-dim weight) + interleaved-real
    \\// rope. One thread per (token, head); 128-dim loop in registers.
    \\kernel void krms_rope(
    \\    device float* x [[buffer(0)]],
    \\    const device float* w [[buffer(1)]],
    \\    const device float* cosb [[buffer(2)]],
    \\    const device float* sinb [[buffer(3)]],
    \\    constant GlueParams& p [[buffer(4)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.tokens * p.heads) return;
    \\    uint t = gid / p.heads;
    \\    device float* row = x + (ulong)gid * p.head_dim;
    \\    float ss = 0.0f;
    \\    for (uint i = 0; i < 128; i++) ss += row[i] * row[i];
    \\    float inv = rsqrt(ss / 128.0f + 1e-6f);
    \\    const device float* c = cosb + (ulong)t * p.head_dim;
    \\    const device float* s = sinb + (ulong)t * p.head_dim;
    \\    for (uint i = 0; i < 128; i += 2) {
    \\        float a0 = row[i] * inv * w[i];
    \\        float a1 = row[i + 1] * inv * w[i + 1];
    \\        row[i] = a0 * c[i] - a1 * s[i];
    \\        row[i + 1] = a1 * c[i + 1] + a0 * s[i + 1];
    \\    }
    \\}
    \\
    \\// krms_rope with the result written head-major half straight into the
    \\// MFA input scratch (what to_headmajor_f16 would have produced from the
    \\// f32 row): out[(h * total + tok_off + t) * 128 + i]. x is read only.
    \\struct RopeHmParams {
    \\    uint tokens;
    \\    uint heads;
    \\    uint head_dim;
    \\    uint tok_off;
    \\    uint total;
    \\};
    \\kernel void krms_rope_hm(
    \\    const device float* x [[buffer(0)]],
    \\    const device float* w [[buffer(1)]],
    \\    const device float* cosb [[buffer(2)]],
    \\    const device float* sinb [[buffer(3)]],
    \\    device half* out [[buffer(4)]],
    \\    constant RopeHmParams& p [[buffer(5)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.tokens * p.heads) return;
    \\    uint t = gid / p.heads;
    \\    uint h = gid - t * p.heads;
    \\    const device float* row = x + (ulong)gid * p.head_dim;
    \\    float ss = 0.0f;
    \\    for (uint i = 0; i < 128; i++) ss += row[i] * row[i];
    \\    float inv = rsqrt(ss / 128.0f + 1e-6f);
    \\    const device float* c = cosb + (ulong)t * p.head_dim;
    \\    const device float* s = sinb + (ulong)t * p.head_dim;
    \\    device half* o = out + ((ulong)h * p.total + p.tok_off + t) * p.head_dim;
    \\    for (uint i = 0; i < 128; i += 2) {
    \\        float a0 = row[i] * inv * w[i];
    \\        float a1 = row[i + 1] * inv * w[i + 1];
    \\        float r0 = a0 * c[i] - a1 * s[i];
    \\        float r1 = a1 * c[i + 1] + a0 * s[i + 1];
    \\        o[i] = half(clamp(r0, -65504.0f, 65504.0f));
    \\        o[i + 1] = half(clamp(r1, -65504.0f, 65504.0f));
    \\    }
    \\}
    \\
    \\// MFA output (head-major f32, the attention scratch) straight into the
    \\// half concat operand: dst[t * stride + h * dim + d]. Same cast as
    \\// kcat_rows_h, so the out GEMM sees identical halves without the f32
    \\// un-permute round trip. q = (tokens, heads, dim, stride).
    \\kernel void kunperm_hm_h(
    \\    const device float* in [[buffer(0)]],
    \\    device half* dst [[buffer(1)]],
    \\    constant uint4& q [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint dim = q.z;
    \\    uint heads = q.y;
    \\    if (gid >= q.x * heads * dim) return;
    \\    uint d = gid % dim;
    \\    uint h = (gid / dim) % heads;
    \\    uint t = gid / (dim * heads);
    \\    float v = in[((ulong)h * q.x + t) * dim + d];
    \\    dst[(ulong)t * q.w + h * dim + d] = half(clamp(v, -65504.0f, 65504.0f));
    \\}
    \\
    \\// kunperm_hm_h over a HALF head-major source (the steel attention O).
    \\kernel void kunperm_hm_hh(
    \\    const device half* in [[buffer(0)]],
    \\    device half* dst [[buffer(1)]],
    \\    constant uint4& q [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint dim = q.z;
    \\    uint heads = q.y;
    \\    if (gid >= q.x * heads * dim) return;
    \\    uint d = gid % dim;
    \\    uint h = (gid / dim) % heads;
    \\    uint t = gid / (dim * heads);
    \\    dst[(ulong)t * q.w + h * dim + d] = in[((ulong)h * q.x + t) * dim + d];
    \\}
    \\
    \\// kswiglu_h writing into a strided destination at a column offset (the
    \\// mlp half of the concat operand). q = (tokens, inner, stride, col).
    \\kernel void kswiglu_hs(
    \\    const device float* x [[buffer(0)]],
    \\    device half* out [[buffer(1)]],
    \\    constant uint4& q [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint inner = q.y;
    \\    if (gid >= q.x * inner) return;
    \\    uint t = gid / inner;
    \\    uint i = gid - t * inner;
    \\    float g = x[(ulong)t * inner * 2 + i];
    \\    float u = x[(ulong)t * inner * 2 + inner + i];
    \\    float v = g / (1.0f + exp(-g)) * u;
    \\    out[(ulong)t * q.z + q.w + i] = half(clamp(v, -65504.0f, 65504.0f));
    \\}
    \\
    \\// SwiGLU over the fused mlp buffer: out[t][i] = silu(x[t][i]) * x[t][inner+i].
    \\kernel void kswiglu(
    \\    const device float* x [[buffer(0)]],
    \\    device float* out [[buffer(1)]],
    \\    constant uint2& d [[buffer(2)]], // (tokens, inner)
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint inner = d.y;
    \\    if (gid >= d.x * inner) return;
    \\    uint t = gid / inner;
    \\    uint i = gid - t * inner;
    \\    float g = x[(ulong)t * inner * 2 + i];
    \\    float u = x[(ulong)t * inner * 2 + inner + i];
    \\    out[gid] = g / (1.0f + exp(-g)) * u;
    \\}
    \\
    \\// state += gate[j] * delta, rowwise vector gate.
    \\kernel void kgate_add(
    \\    device float* state [[buffer(0)]],
    \\    const device float* delta [[buffer(1)]],
    \\    const device float* gate [[buffer(2)]],
    \\    constant GlueParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.tokens * p.hidden) return;
    \\    uint j = gid % p.hidden;
    \\    state[gid] += gate[j] * delta[gid];
    \\}
    \\
    \\// Two-source rowwise concat: dst[t] = [a[t] (na wide) | b[t] (nb wide)].
    \\kernel void kcat_rows(
    \\    const device float* a [[buffer(0)]],
    \\    const device float* b [[buffer(1)]],
    \\    device float* dst [[buffer(2)]],
    \\    constant uint2& w2 [[buffer(3)]], // (na, nb)
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint na = w2.x;
    \\    uint nb = w2.y;
    \\    uint w = na + nb;
    \\    uint t = gid / w;
    \\    uint i = gid - t * w;
    \\    dst[gid] = (i < na) ? a[(ulong)t * na + i] : b[(ulong)t * nb + (i - na)];
    \\}
    \\
    \\// Plain row copy (stream concat/split).
    \\kernel void kcopy(
    \\    const device float* src_buf [[buffer(0)]],
    \\    device float* dst_buf [[buffer(1)]],
    \\    constant uint& count [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid < count) dst_buf[gid] = src_buf[gid];
    \\}
    \\
    \\// ---- f16 activation mode (ZDRAW_KLEIN_ACT=f16) variants: identical math
    \\// in f32, half store (saturating) so the big GEMMs read half A operands.
    \\kernel void kln_mod_h(
    \\    const device float* x [[buffer(0)]],
    \\    device half* out [[buffer(1)]],
    \\    const device float* shift [[buffer(2)]],
    \\    const device float* scale [[buffer(3)]],
    \\    constant GlueParams& p [[buffer(4)]],
    \\    uint tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tcount [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float red[256];
    \\    const device float* row = x + (ulong)tg * p.hidden;
    \\    float s = 0.0f;
    \\    for (uint i = tid; i < p.hidden; i += tcount) s += row[i];
    \\    red[tid] = s;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] += red[tid + st];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float mean = red[0] / p.hidden;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float v = 0.0f;
    \\    for (uint i = tid; i < p.hidden; i += tcount) {
    \\        float d = row[i] - mean;
    \\        v += d * d;
    \\    }
    \\    red[tid] = v;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] += red[tid + st];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv = rsqrt(red[0] / p.hidden + 1e-6f);
    \\    device half* dst = out + (ulong)tg * p.hidden;
    \\    for (uint i = tid; i < p.hidden; i += tcount) {
    \\        float n = (row[i] - mean) * inv;
    \\        dst[i] = half(clamp((1.0f + scale[i]) * n + shift[i], -65504.0f, 65504.0f));
    \\    }
    \\}
    \\
    \\kernel void kswiglu_h(
    \\    const device float* x [[buffer(0)]],
    \\    device half* out [[buffer(1)]],
    \\    constant uint2& d [[buffer(2)]], // (tokens, inner)
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint inner = d.y;
    \\    if (gid >= d.x * inner) return;
    \\    uint t = gid / inner;
    \\    uint i = gid - t * inner;
    \\    float g = x[(ulong)t * inner * 2 + i];
    \\    float u = x[(ulong)t * inner * 2 + inner + i];
    \\    out[gid] = half(clamp(g / (1.0f + exp(-g)) * u, -65504.0f, 65504.0f));
    \\}
    \\
    \\// Mixed ABI: a f32, b half, dst half. The o buffer arrives f32 from
    \\// attention; b is the half swiglu output.
    \\kernel void kcat_rows_h(
    \\    const device float* a [[buffer(0)]],
    \\    const device half* b [[buffer(1)]],
    \\    device half* dst [[buffer(2)]],
    \\    constant uint2& w2 [[buffer(3)]], // (na, nb)
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    uint na = w2.x;
    \\    uint nb = w2.y;
    \\    uint w = na + nb;
    \\    uint t = gid / w;
    \\    uint i = gid - t * w;
    \\    dst[gid] = (i < na)
    \\        ? half(clamp(a[(ulong)t * na + i], -65504.0f, 65504.0f))
    \\        : b[(ulong)t * nb + (i - na)];
    \\}
    \\
    \\struct AxpyParams { uint count; float dt; };
    \\
    \\// Resident Euler step: x += dt * v, so latents never round-trip to the
    \\// CPU between denoise steps (ZDRAW_KLEIN_XRES).
    \\kernel void kaxpy(
    \\    device float* x [[buffer(0)]],
    \\    const device float* v [[buffer(1)]],
    \\    constant AxpyParams& p [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid < p.count) x[gid] += p.dt * v[gid];
    \\}
;
