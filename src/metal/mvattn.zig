//! Resident VAE mid-block attention.
//!
//! Replaces vattn.run's CPU orchestration on the Metal decode: GroupNorm
//! stats/apply, the NCHW-to-token transpose, the q/k/v/out projections, and
//! the residual scatter-add all run on chain-pool GPU buffers (the blessed
//! borrow: the VAE decodes while the denoise chain is idle). The only
//! remaining CPU crossings are the feature upload/readback at this module's
//! boundary and the wide-SDPA hop: when the owned wide kernels are not
//! engaged, q/k/v round-trip through the UNCHANGED mattn.run fresh-buffer
//! route, keeping the vae-sdpa-nondeterminism surface byte-for-byte intact.
//!
//! The one value-changing element vs the CPU path is the GroupNorm stats
//! reduction (GPU tree order vs the scalar loops in gnorm.zig), the same
//! drift class as every previous stats migration; everything else is a
//! bit-exact relocation (the apply expression is verbatim, the transpose and
//! scatter-add are permutations, the projections and attention run the same
//! kernels on the same values).

const std = @import("std");

const attention = @import("../runtime/attention.zig");
const mattn = @import("mattn.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const metal_c = @import("metal_c.zig");
const mlend = @import("mlend.zig");
const mgemm_bias = @import("mgemm_bias.zig");
const mlinear = @import("mlinear.zig");
const mres_util = @import("mres_util.zig");
const mvattn_owned = @import("mvattn_owned.zig");
const mvres_param = @import("mvres_param.zig");
const vattn = @import("../vae/vattn.zig");

extern fn zdraw_metal_run_vae_attn_prep(
    queue: *anyopaque,
    stats_pipeline: *anyopaque,
    apply_pipeline: *anyopaque,
    feature: *anyopaque,
    stats: *anyopaque,
    norm_w: *anyopaque,
    norm_b: *anyopaque,
    seq_out: *anyopaque,
    params: *const mvres_param.NormParams,
    stats_threads: usize,
    apply_threads: usize,
) c_int;

extern fn zdraw_metal_run_vae_attn_finish(
    queue: *anyopaque,
    pipeline: *anyopaque,
    feature: *anyopaque,
    seq: *anyopaque,
    params: *const mvres_param.NormParams,
    thread_count: usize,
) c_int;

/// SDPA-boundary scratch, allocated by the caller (vdecode owns VAE CPU
/// allocation; this module allocates nothing).
pub const Boundary = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    mix: []f32,
};

/// True when the resident route can take this shape end to end (decided up
/// front: no mid-flight fallback with a stranded resident feature).
pub fn fits(channels: usize, tokens: usize) bool {
    return tokens % 32 == 0 and channels % 32 == 0 and channels > 0;
}

pub fn run(
    conv: *mconv.Context,
    linear: *mlinear.Context,
    attn: *mattn.Context,
    lender: ?mlend.Offer,
    state_data: []f32,
    views: vattn.Views,
    boundary: Boundary,
    cfg: vattn.Config,
) !void {
    const pool = &linear.pool;
    const state_bytes = std.mem.sliceAsBytes(state_data);
    const feat_h = try mlend.chainScratch(lender, pool, linear.device, .state, state_bytes.len);
    metal_c.zdraw_metal_write_buffer(feat_h, state_bytes.ptr, state_bytes.len);
    try runResident(conv, linear, attn, lender, feat_h, views, boundary, cfg);
    readSlice(feat_h, state_data);
}

/// The resident core on a caller-owned f32 NCHW feature buffer (the
/// streamMid route: the feature never crosses the CPU boundary here at all).
/// Scratch comes from `lender` when it has an idle buffer of the size (the
/// Klein DiT pool during decode), else from the chain pool as before.
pub fn runResident(
    conv: *mconv.Context,
    linear: *mlinear.Context,
    attn: *mattn.Context,
    lender: ?mlend.Offer,
    feat_h: *anyopaque,
    views: vattn.Views,
    boundary: Boundary,
    cfg: vattn.Config,
) !void {
    const tokens = cfg.height * cfg.width;
    const bytes = cfg.channels * tokens * 4;
    const dev = linear.device;
    const pool = &linear.pool;

    var wtmp: ?mbuffer.Buffer = null;
    defer if (wtmp) |*b| b.deinit();
    var btmp: ?mbuffer.Buffer = null;
    defer if (btmp) |*b| b.deinit();
    const wb = try linear.buffers.bindView(views.norm_w, &wtmp);
    const bb = try linear.buffers.bindView(views.norm_b, &btmp);
    const np = try normParams(views, cfg, wb.offset, bb.offset);
    const seq_h = try mlend.chainScratch(lender, pool, dev, .norm, bytes);
    try prep(conv, linear, feat_h, seq_h, wb.handle, bb.handle, &np);

    const q_h = try mlend.chainScratch(lender, pool, dev, .q, bytes);
    const k_h = try mlend.chainScratch(lender, pool, dev, .k, bytes);
    const v_h = try mlend.chainScratch(lender, pool, dev, .v, bytes);
    const mix_h = try mlend.chainScratch(lender, pool, dev, .mix, bytes);
    try mgemm_bias.batchOnHandles(linear, seq_h, q_h, views.q_w, views.q_b, tokens);
    try mgemm_bias.batchOnHandles(linear, seq_h, k_h, views.k_w, views.k_b, tokens);
    try mgemm_bias.batchOnHandles(linear, seq_h, v_h, views.v_w, views.v_b, tokens);

    const acfg = attention.Config{
        .tokens = tokens,
        .heads = cfg.channels / cfg.head_dim,
        .kv_heads = cfg.channels / cfg.head_dim,
        .head_dim = cfg.head_dim,
        .causal = false,
    };
    if (mvattn_owned.enabled()) {
        try mvattn_owned.run(linear, lender, q_h, k_h, v_h, mix_h, acfg);
    } else {
        attn.runOnHandles(q_h, k_h, v_h, mix_h, acfg) catch |err| switch (err) {
            error.UnsupportedShape => try crossSdpa(attn, q_h, k_h, v_h, mix_h, boundary, acfg),
            else => return err,
        };
    }

    const attn_h = try mlend.chainScratch(lender, pool, dev, .attn, bytes);
    try mgemm_bias.batchOnHandles(linear, mix_h, attn_h, views.out_w, views.out_b, tokens);
    if (zdraw_metal_run_vae_attn_finish(
        linear.queue,
        conv.attn_scatter_pipeline,
        feat_h,
        attn_h,
        &np,
        conv.attn_scatter_threads,
    ) != 0) return error.MetalDispatchFailed;
}

fn normParams(
    views: vattn.Views,
    cfg: vattn.Config,
    w_off: u64,
    b_off: u64,
) !mvres_param.NormParams {
    return .{
        .channels = try mres_util.toU32(cfg.channels),
        .height = try mres_util.toU32(cfg.height),
        .width = try mres_util.toU32(cfg.width),
        .groups = try mres_util.toU32(cfg.groups),
        .dtype = try mres_util.dtype(views.norm_w.dtype),
        .bias_dtype = try mres_util.dtype(views.norm_b.dtype),
        .eps = cfg.eps,
        .weight_offset = w_off,
        .bias_offset = b_off,
    };
}

/// Stats + no-SiLU apply/transpose in one synchronous command buffer. The
/// tiny stats buffer is transient (the dispatch waits before return).
fn prep(
    conv: *mconv.Context,
    linear: *mlinear.Context,
    feat_h: *anyopaque,
    seq_h: *anyopaque,
    norm_w: *anyopaque,
    norm_b: *anyopaque,
    np: *const mvres_param.NormParams,
) !void {
    var stats = try mbuffer.Buffer.empty(linear.device, np.groups * 2 * 4);
    defer stats.deinit();
    if (zdraw_metal_run_vae_attn_prep(
        linear.queue,
        conv.attn_stats_pipeline,
        conv.attn_apply_pipeline,
        feat_h,
        stats.handle,
        norm_w,
        norm_b,
        seq_h,
        np,
        conv.attn_stats_threads,
        conv.attn_apply_threads,
    ) != 0) return error.MetalDispatchFailed;
}

/// Wide kernels not engaged: hop the SDPA boundary through the UNCHANGED
/// fresh-buffer route, then return to the pool.
fn crossSdpa(
    attn: *mattn.Context,
    q_h: *anyopaque,
    k_h: *anyopaque,
    v_h: *anyopaque,
    mix_h: *anyopaque,
    boundary: Boundary,
    acfg: attention.Config,
) !void {
    readSlice(q_h, boundary.q);
    readSlice(k_h, boundary.k);
    readSlice(v_h, boundary.v);
    try attn.run(boundary.mix, boundary.q, boundary.k, boundary.v, acfg);
    metal_c.zdraw_metal_write_buffer(
        mix_h,
        std.mem.sliceAsBytes(boundary.mix).ptr,
        boundary.mix.len * 4,
    );
}

fn readSlice(handle: *anyopaque, out: []f32) void {
    metal_c.zdraw_metal_read_buffer(handle, std.mem.sliceAsBytes(out).ptr, out.len * 4);
}
