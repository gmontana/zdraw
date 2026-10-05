//! Owned VAE mid-block attention (heads 1, head_dim 512, 16384 tokens at
//! 1024px): row-blocked S = Q K^T, f32 softmax, P V on the exact f32 GEMM
//! kernel, in one command buffer. Replaces the MPSGraph SDPA on both tiers
//! (owner rule 2026-08-26: no Apple ML frameworks on a render path).
//! Fixed-order f32 arithmetic end to end, so it is deterministic where the
//! graph route was intermittently not.
//!
//! Scratch (pool slots unused during the VAE attention): `.gate` holds V^T
//! [ch, tokens] f32, `.ffn` holds one S/P block [rows, tokens] f32 (64 MB at
//! 1024px with 1024-row blocks).

const std = @import("std");

const attention = @import("attention.zig");
const metal_c = @import("metal_c.zig");
const mlend = @import("mlend.zig");
const mlinear = @import("mlinear.zig");
const mpipe = @import("mpipe.zig");

const block_rows: usize = 1024;

const src: [*:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// out[c][t] = in[t][c]; d = (tokens, ch)
    \\kernel void kvt_transpose_f32(
    \\    const device float* in [[buffer(0)]],
    \\    device float* out [[buffer(1)]],
    \\    constant uint2& d [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= d.x * d.y) return;
    \\    uint t = gid / d.y;
    \\    uint c = gid - t * d.y;
    \\    out[(ulong)c * d.x + t] = in[gid];
    \\}
    \\// In-place row softmax: x[row][0..n) = exp(scale*x - max) / sum. One
    \\// threadgroup (256) per row; three fixed-order passes, f32.
    \\struct SoftmaxParams { uint n; float scale; };
    \\kernel void krow_softmax_f32(
    \\    device float* x [[buffer(0)]],
    \\    constant SoftmaxParams& p [[buffer(1)]],
    \\    uint tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tcount [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float red[256];
    \\    device float* row = x + (ulong)tg * p.n;
    \\    float m = -INFINITY;
    \\    for (uint i = tid; i < p.n; i += tcount) m = max(m, row[i] * p.scale);
    \\    red[tid] = m;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] = max(red[tid], red[tid + st]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float rmax = red[0];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float s = 0.0f;
    \\    for (uint i = tid; i < p.n; i += tcount) {
    \\        float e = exp(row[i] * p.scale - rmax);
    \\        row[i] = e;
    \\        s += e;
    \\    }
    \\    red[tid] = s;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] += red[tid + st];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv = 1.0f / red[0];
    \\    for (uint i = tid; i < p.n; i += tcount) row[i] *= inv;
    \\}
    \\// Product tier (f16 operands, f32 accumulate in the GEMM): f32 -> half
    \\// copies, V^T in half, and a softmax that writes half P beside the f32 S.
    \\kernel void kf32_to_h(
    \\    const device float* in [[buffer(0)]],
    \\    device half* out [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid < n) out[gid] = half(clamp(in[gid], -65504.0f, 65504.0f));
    \\}
    \\kernel void kvt_transpose_h(
    \\    const device float* in [[buffer(0)]],
    \\    device half* out [[buffer(1)]],
    \\    constant uint2& d [[buffer(2)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= d.x * d.y) return;
    \\    uint t = gid / d.y;
    \\    uint c = gid - t * d.y;
    \\    out[(ulong)c * d.x + t] = half(clamp(in[gid], -65504.0f, 65504.0f));
    \\}
    \\kernel void krow_softmax_h(
    \\    const device float* x [[buffer(0)]],
    \\    device half* out [[buffer(1)]],
    \\    constant SoftmaxParams& p [[buffer(2)]],
    \\    uint tg [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint tcount [[threads_per_threadgroup]]
    \\) {
    \\    threadgroup float red[256];
    \\    const device float* row = x + (ulong)tg * p.n;
    \\    device half* orow = out + (ulong)tg * p.n;
    \\    float m = -INFINITY;
    \\    for (uint i = tid; i < p.n; i += tcount) m = max(m, row[i] * p.scale);
    \\    red[tid] = m;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] = max(red[tid], red[tid + st]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float rmax = red[0];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float s = 0.0f;
    \\    for (uint i = tid; i < p.n; i += tcount) s += exp(row[i] * p.scale - rmax);
    \\    red[tid] = s;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint st = tcount / 2; st > 0; st >>= 1) {
    \\        if (tid < st) red[tid] += red[tid + st];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv = 1.0f / red[0];
    \\    for (uint i = tid; i < p.n; i += tcount) {
    \\        orow[i] = half(exp(row[i] * p.scale - rmax) * inv);
    \\    }
    \\}
;

const SoftmaxParams = extern struct { n: u32, scale: f32 };

fn glue(
    bt: *anyopaque,
    pipe: *anyopaque,
    b0: ?*anyopaque,
    b1: ?*anyopaque,
    cb: *const anyopaque,
    cb_len: usize,
    cb_idx: u32,
    grid: usize,
    groups: u32,
) !void {
    const rc = metal_c.zdraw_metal_run_glue_enc(
        bt,
        pipe,
        b0,
        b1,
        null,
        null,
        null,
        null,
        cb,
        cb_len,
        cb_idx,
        grid,
        256,
        groups,
    );
    if (rc != 0) return error.MetalDispatchFailed;
}

fn gemmOff(
    bt: *anyopaque,
    pipe: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    gp: *const metal_c.GemmParams,
    a_off: u64,
    c_off: u64,
) !void {
    const rc = metal_c.zdraw_metal_run_gemm_off_enc(bt, pipe, a, w, c, gp, a_off, c_off);
    if (rc != 0) return error.MetalDispatchFailed;
}

fn gemmHalf(
    bt: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    gp: *const metal_c.GemmParams,
    a_off: u64,
    c_off: u64,
) !void {
    const rc = metal_c.zdraw_metal_run_gemm_f16a_enc(bt, a, w, c, gp, a_off, c_off);
    if (rc != 0) return error.MetalDispatchFailed;
}

fn params(m: usize, k: usize, n: usize, dtype: u32, mode: u32) metal_c.GemmParams {
    return .{ .m = @intCast(m), .k = @intCast(k), .n = @intCast(n), .dtype = dtype, .mode = mode };
}

const Pipes = struct {
    transpose: *anyopaque,
    softmax: *anyopaque,
    to_h: *anyopaque,
    transpose_h: *anyopaque,
    softmax_h: *anyopaque,
};
var pipes: ?Pipes = null;

fn pipesFor(device: *anyopaque) !Pipes {
    if (pipes) |p| return p;
    var err: [1024]u8 = undefined;
    const transpose = try mpipe.required(device, src, "kvt_transpose_f32", &err);
    errdefer metal_c.zdraw_metal_release_pipeline(transpose);
    const softmax = try mpipe.required(device, src, "krow_softmax_f32", &err);
    errdefer metal_c.zdraw_metal_release_pipeline(softmax);
    const to_h = try mpipe.required(device, src, "kf32_to_h", &err);
    errdefer metal_c.zdraw_metal_release_pipeline(to_h);
    const transpose_h = try mpipe.required(device, src, "kvt_transpose_h", &err);
    errdefer metal_c.zdraw_metal_release_pipeline(transpose_h);
    const softmax_h = try mpipe.required(device, src, "krow_softmax_h", &err);
    errdefer metal_c.zdraw_metal_release_pipeline(softmax_h);
    pipes = .{
        .transpose = transpose,
        .softmax = softmax,
        .to_h = to_h,
        .transpose_h = transpose_h,
        .softmax_h = softmax_h,
    };
    return pipes.?;
}

/// Release the module's pipelines (owned here, compiled on first use);
/// mlinear.Context.deinit calls it so the VAE contexts tear them down.
pub fn deinit() void {
    const p = pipes orelse return;
    metal_c.zdraw_metal_release_pipeline(p.transpose);
    metal_c.zdraw_metal_release_pipeline(p.softmax);
    metal_c.zdraw_metal_release_pipeline(p.to_h);
    metal_c.zdraw_metal_release_pipeline(p.transpose_h);
    metal_c.zdraw_metal_release_pipeline(p.softmax_h);
    pipes = null;
}

/// ZDRAW_VAE_ATTN_HALF=1 (the product profile): half Q/K/V^T/P through the
/// f16-A GEMM; unset (strict): the exact f32 path.
fn halfEnabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_ATTN_HALF") orelse return false;
    return raw[0] == '1';
}

/// ZDRAW_VAE_ATTN_OWNED=0 restores the graph route (A/B); default on.
pub fn enabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_ATTN_OWNED") orelse return true;
    return raw[0] != '0';
}

pub fn run(
    linear: *mlinear.Context,
    lender: ?mlend.Offer,
    q_h: *anyopaque,
    k_h: *anyopaque,
    v_h: *anyopaque,
    out_h: *anyopaque,
    cfg: attention.Config,
) !void {
    if (cfg.heads != 1 or cfg.kv_heads != 1 or cfg.causal) return error.UnsupportedShape;
    const tokens = cfg.tokens;
    const ch = cfg.head_dim;
    if (tokens % 32 != 0 or ch % 32 != 0) return error.UnsupportedShape;
    if (halfEnabled()) return runHalf(linear, lender, q_h, k_h, v_h, out_h, tokens, ch);
    const dev = linear.device;
    const gemm = linear.gemm_exact_pipeline orelse return error.GemmUnavailable;
    const p = try pipesFor(dev);
    const vt = try mlend.chainScratch(lender, &linear.pool, dev, .gate, ch * tokens * 4);
    const rows = @min(block_rows, tokens);
    const s_buf = try mlend.chainScratch(lender, &linear.pool, dev, .ffn, rows * tokens * 4);

    const bt = metal_c.zdraw_metal_batch_begin(linear.queue) orelse
        return error.MetalDispatchFailed;
    var ok = false;
    defer if (!ok) {
        _ = metal_c.zdraw_metal_batch_end(bt);
    };
    const td: [2]u32 = .{ @intCast(tokens), @intCast(ch) };
    try glue(bt, p.transpose, v_h, vt, &td, 8, 2, tokens * ch, 0);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
    const sp = SoftmaxParams{ .n = @intCast(tokens), .scale = scale };
    var r0: usize = 0;
    while (r0 < tokens) : (r0 += rows) {
        const m = @min(rows, tokens - r0);
        // S[m, tokens] = Q[r0..r0+m, ch] . K[tokens, ch]^T (K is already [N, K])
        const gp_s = params(m, ch, tokens, 3, 0);
        try gemmOff(bt, gemm, q_h, k_h, s_buf, &gp_s, r0 * ch * 4, 0);
        try glue(bt, p.softmax, s_buf, null, &sp, @sizeOf(SoftmaxParams), 1, m, 1);
        // out[r0..r0+m, ch] = P[m, tokens] . V[tokens, ch] = P . (V^T)^T
        const gp_o = params(m, tokens, ch, 3, 0);
        try gemmOff(bt, gemm, s_buf, vt, out_h, &gp_o, 0, r0 * ch * 4);
    }
    ok = true;
    if (metal_c.zdraw_metal_batch_end(bt) != 0) return error.MetalDispatchFailed;
}

/// Product tier: half Q/K/V^T/P through the f16-A GEMM (f32 accumulate,
/// f32 out), the same row-blocked sequence. Not bit-preserving (the tier
/// is not); gated by PSNR vs strict and the content check.
fn runHalf(
    linear: *mlinear.Context,
    lender: ?mlend.Offer,
    q_h: *anyopaque,
    k_h: *anyopaque,
    v_h: *anyopaque,
    out_h: *anyopaque,
    tokens: usize,
    ch: usize,
) !void {
    const dev = linear.device;
    const p = try pipesFor(dev);
    const n = tokens * ch;
    const pool = &linear.pool;
    const qh = try mlend.chainScratch(lender, pool, dev, .up, n * 2);
    const kh = try mlend.chainScratch(lender, pool, dev, .gateup, n * 2);
    const vt = try mlend.chainScratch(lender, pool, dev, .gate, n * 2);
    const rows = @min(block_rows, tokens);
    const s_buf = try mlend.chainScratch(lender, pool, dev, .ffn, rows * tokens * 4);
    const p_buf = try mlend.chainScratch(lender, pool, dev, .mod_in, rows * tokens * 2);

    const bt = metal_c.zdraw_metal_batch_begin(linear.queue) orelse
        return error.MetalDispatchFailed;
    var ok = false;
    defer if (!ok) {
        _ = metal_c.zdraw_metal_batch_end(bt);
    };
    const n32: u32 = @intCast(n);
    try glue(bt, p.to_h, q_h, qh, &n32, 4, 2, n, 0);
    try glue(bt, p.to_h, k_h, kh, &n32, 4, 2, n, 0);
    const td: [2]u32 = .{ @intCast(tokens), @intCast(ch) };
    try glue(bt, p.transpose_h, v_h, vt, &td, 8, 2, n, 0);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
    const sp = SoftmaxParams{ .n = @intCast(tokens), .scale = scale };
    var r0: usize = 0;
    while (r0 < tokens) : (r0 += rows) {
        const m = @min(rows, tokens - r0);
        const gp_s = params(m, ch, tokens, 1, 2);
        try gemmHalf(bt, qh, kh, s_buf, &gp_s, r0 * ch * 2, 0);
        try glue(bt, p.softmax_h, s_buf, p_buf, &sp, @sizeOf(SoftmaxParams), 2, m, 1);
        const gp_o = params(m, tokens, ch, 1, 2);
        try gemmHalf(bt, p_buf, vt, out_h, &gp_o, 0, r0 * ch * 4);
    }
    ok = true;
    if (metal_c.zdraw_metal_batch_end(bt) != 0) return error.MetalDispatchFailed;
}
