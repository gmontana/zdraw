//! Wall 1 of the memory ladder: the strip-memory VAE decode (ledger
//! memory-ladder-w1-strip-decode-20261006). The whole-map feature buffers of
//! the streamed decoder are replaced by channel-chunked statistics and
//! row-stripped convolutions. This module starts with the piece everything
//! else depends on: GroupNorm statistics computed over channel chunks must be
//! bit-identical to the whole-map pass, because the product route's
//! `vae_norm_stats_h_sq` accumulates per thread in channel-major order (row
//! strips cannot reproduce it, channel-group chunks can). The test below is
//! the preregistered go/no-go for the design.

const std = @import("std");
const builtin = @import("builtin");

const c = @import("metal_c.zig");
const chain = @import("mvres_stream_chain.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const mconv_strip = @import("mconv_strip_shader.zig");
const mpipe = @import("mpipe.zig");
const mvfinal = @import("mvfinal.zig");
const mvres_pool = @import("mvres_pool.zig");
const mvres_wino = @import("mvres_wino.zig");
const param = @import("mvres_param.zig");
const vnorm_shader = @import("mvnorm_shader.zig");
const vviews = @import("vviews.zig");

pub const NormParams = param.NormParams;

/// ZDRAW_VAE_STRIPMEM=1 turns the strip-memory decode on for the up blocks
/// (product route: f16 features, FULL_H, Winograd). Off by default until the
/// wall is certified.
pub fn enabled() bool {
    return tier() >= 1;
}

/// The strip-memory tier from ZDRAW_VAE_STRIPMEM: 0 off, 1 strip scratch
/// with whole-map input/output, 2 the group's later blocks in place.
pub fn tier() u8 {
    const raw = std.c.getenv("ZDRAW_VAE_STRIPMEM") orelse return 0;
    return switch (raw[0]) {
        '1' => 1,
        '2' => 2,
        '3' => 3,
        else => 0,
    };
}

/// The fixed plan: 64-channel chunks (a multiple of every GroupNorm group
/// width and of the Winograd output-channel tile) and 64-row strips
/// (rounded up per block to the tile-row alignment unit).
pub const plan_default = mvres_pool.StripPlan{ .chunk_ch = 64, .strip_rows = 64 };

/// The plan at the configured tier.
fn planAtTier() mvres_pool.StripPlan {
    var plan = plan_default;
    plan.tier = tier();
    return plan;
}

/// Mirror of wino_eligible (metal_api.m) on geometry: every block of the
/// group must be Winograd-eligible for both convs for the strip chain to run.
fn winoShapeOk(in_ch: usize, out_ch: usize, h: usize, w: usize) bool {
    if (in_ch % 32 != 0 or out_ch % 64 != 0 or h % 4 != 0 or w % 4 != 0) return false;
    return ((h / 4) * (w / 4)) % 64 == 0;
}

/// The strip plan for an up-block sequence, or null when the strip chain
/// does not apply (feature off, not the f16 FULL_H Winograd route, a block
/// norm reader, or a shape the Winograd contract rejects). Sizing and
/// dispatch both derive from this one decision.
pub fn planFor(
    stream: *chain.Ctx,
    blocks: []const chain.UpView,
    h16: bool,
    full_h: bool,
    block_norm: bool,
) ?mvres_pool.StripPlan {
    if (stream.strip == null or !h16 or !full_h or block_norm or !mvres_wino.enabled()) return null;
    for (blocks) |b| {
        const cfg = b.cfg;
        // Block 0 maps in_ch -> out_ch; later blocks out_ch -> out_ch.
        if (!winoShapeOk(cfg.in_ch, cfg.out_ch, cfg.height, cfg.width)) return null;
        if (!winoShapeOk(cfg.out_ch, cfg.out_ch, cfg.height, cfg.width)) return null;
        if (cfg.out_ch % plan_default.chunk_ch != 0) return null;
        // In place (tier 2) writes the output over the input's plane prefix.
        if (tier() >= 2 and cfg.in_ch < cfg.out_ch) return null;
    }
    return planAtTier();
}

/// The two strip-local kernels the chain adds (same arithmetic as their
/// whole-map originals; see the shader comments).
pub const Pipes = struct {
    add_strip: *anyopaque,
    add_rev: *anyopaque,
    skip_strip: *anyopaque,
    rows_copy: *anyopaque,
    apply_strip: *anyopaque,
    conv_strip_in: *anyopaque,

    pub fn init(dev: *anyopaque) !Pipes {
        var err: [1024]u8 = undefined;
        const vn = vnorm_shader.vnorm.ptr;
        const add_strip = try mpipe.required(dev, vn, "vae_add_strip_h", &err);
        errdefer c.zdraw_metal_release_pipeline(add_strip);
        const add_rev = try mpipe.required(dev, vn, "vae_add_strip_rev_h", &err);
        errdefer c.zdraw_metal_release_pipeline(add_rev);
        const skip_strip = try mpipe.required(dev, mconv_strip.conv.ptr, "conv1x1_strip_h", &err);
        errdefer c.zdraw_metal_release_pipeline(skip_strip);
        const rows_copy = try mpipe.required(dev, vn, "vae_rows_copy_h", &err);
        errdefer c.zdraw_metal_release_pipeline(rows_copy);
        const apply_strip = try mpipe.required(dev, vn, "vae_norm_apply_strip_hf", &err);
        errdefer c.zdraw_metal_release_pipeline(apply_strip);
        const conv_src = mconv_strip.conv.ptr;
        const conv_strip_in = try mpipe.required(dev, conv_src, "conv2d_strip_in", &err);
        errdefer c.zdraw_metal_release_pipeline(conv_strip_in);
        return .{
            .add_strip = add_strip,
            .add_rev = add_rev,
            .skip_strip = skip_strip,
            .rows_copy = rows_copy,
            .apply_strip = apply_strip,
            .conv_strip_in = conv_strip_in,
        };
    }

    pub fn deinit(self: Pipes) void {
        c.zdraw_metal_release_pipeline(self.conv_strip_in);
        c.zdraw_metal_release_pipeline(self.apply_strip);
        c.zdraw_metal_release_pipeline(self.rows_copy);
        c.zdraw_metal_release_pipeline(self.skip_strip);
        c.zdraw_metal_release_pipeline(self.add_rev);
        c.zdraw_metal_release_pipeline(self.add_strip);
    }
};

/// Mirrors ZdrawVaeStripScratch.
const Scratch = extern struct {
    conv1_chunk: *anyopaque,
    conv1_strip: *anyopaque,
    skip_strip: *anyopaque,
    stash: *anyopaque,
    chunk_ch: u32,
    strip_rows: u32,
    stash_rows: u32,
    pad_: u32 = 0,
};

extern fn zdraw_metal_run_vae_res_strip_chain(
    queue: *anyopaque,
    stats_pipeline: *anyopaque,
    stats2_pipeline: *anyopaque,
    add_pipeline: *anyopaque,
    add_strip_pipeline: *anyopaque,
    add_rev_pipeline: *anyopaque,
    skip_strip_pipeline: *anyopaque,
    rows_copy_pipeline: *anyopaque,
    buffers: [*]const chain.StreamBuffers,
    scratch: *const Scratch,
    params: [*]const param.ResParams,
    count: usize,
    stats_threads: usize,
    add_threads: usize,
    wino: *const mvres_wino.Set,
) c_int;

/// One resident group through the strip chain: the scratch roles come from
/// the pool (presized by applySizes, so no handle() call grows here), the
/// per-block buffers and params are the ones dispatchGroup already built.
// The group's strip scratch from the pool: the conv1 chunk, the conv1 strip
// with its halo, the skip/result strip and (tier 2) the halo stash.
fn scratchFor(stream: *chain.Ctx, cfg: chain.Config, plan: mvres_pool.StripPlan) !Scratch {
    const dev = stream.device;
    const unit = mvres_pool.stripUnitRows(cfg.width);
    const rows = mvres_pool.stripRowsFor(plan, cfg.width);
    const hw: usize = cfg.height * cfg.width;
    const chunk = try stream.pool.conv1_chunk.handle(dev, plan.chunk_ch * hw * @sizeOf(f16));
    const strip_bytes = cfg.out_ch * (rows + 2 * unit) * cfg.width * @sizeOf(f16);
    const strip = try stream.pool.conv1_strip.handle(dev, strip_bytes);
    // Tier 2 also lands conv2's strip result here for the in-place no-skip blocks.
    const skip: *anyopaque = if (cfg.in_ch != cfg.out_ch or plan.tier >= 2)
        try stream.pool.skip_strip.handle(dev, cfg.out_ch * rows * cfg.width * @sizeOf(f16))
    else
        chunk; // never read: no block of this group has a skip
    const stash_rows: usize = if (plan.tier >= 2) unit + 1 else 0;
    const stash: *anyopaque = if (plan.tier >= 2)
        try stream.pool.stash.handle(dev, cfg.in_ch * stash_rows * cfg.width * @sizeOf(f16))
    else
        chunk; // never read below tier 2
    const scratch = Scratch{
        .conv1_chunk = chunk,
        .conv1_strip = strip,
        .skip_strip = skip,
        .stash = stash,
        .chunk_ch = @intCast(plan.chunk_ch),
        .strip_rows = @intCast(rows),
        .stash_rows = @intCast(stash_rows),
    };
    return scratch;
}

pub fn dispatchGroup(
    ctx: *mconv.Context,
    stream: *chain.Ctx,
    sbufs: []const chain.StreamBuffers,
    ps: []const param.ResParams,
    cfg: chain.Config,
    stats_pipe: *anyopaque,
    stats2_pipe: *anyopaque,
    add_pipe: *anyopaque,
    pipes: Pipes,
    wino: *const mvres_wino.Set,
    stats_threads: usize,
) !void {
    const scratch = try scratchFor(stream, cfg, planAtTier());
    const rc = zdraw_metal_run_vae_res_strip_chain(
        ctx.queue,
        stats_pipe,
        stats2_pipe,
        add_pipe,
        pipes.add_strip,
        pipes.add_rev,
        pipes.skip_strip,
        pipes.rows_copy,
        sbufs.ptr,
        &scratch,
        ps.ptr,
        sbufs.len,
        stats_threads,
        stream.add_threads,
        wino,
    );
    // Distinct errors per C return code so a failure names its cause:
    // -4 a conv the Winograd contract rejects, -3 a plan/alignment violation,
    // -2 the command buffer did not complete (a GPU fault), -1 no encoder.
    return switch (rc) {
        0 => {},
        -4 => error.UnsupportedDType,
        -3 => error.InvalidShape,
        -2 => error.MetalCommandFailed,
        else => error.MetalDispatchFailed,
    };
}

// Single consumer of the offset-taking stats runner (see metal_api.m).
extern fn zdraw_metal_run_vae_norm_stats_at(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    input_off: usize,
    stats: *anyopaque,
    stats_off: usize,
    params: *const NormParams,
    thread_count: usize,
) c_int;

/// Bytes per element for the norm dtype codes (1 = half, otherwise f32).
fn elemBytes(dtype: u32) usize {
    return if (dtype == 1) 2 else 4;
}

/// GroupNorm statistics of `np` computed over channel chunks of `chunk_ch`
/// channels (a multiple of the group width). The kernel's group g reads the
/// channels [g*group_ch, (g+1)*group_ch) relative to its input base and
/// writes stats[g*2 .. g*2+2]; chunk k binds the input at channel k*chunk_ch
/// and the stats at group k*chunk_ch/group_ch, so every thread visits the
/// same elements in the same order as in the whole-map pass. One command
/// buffer per chunk (the standalone runner); the chain encoder will inline
/// the same binding into its single command buffer.
pub fn chunkedStats(
    queue: *anyopaque,
    pipe: *anyopaque,
    input: *anyopaque,
    stats: *anyopaque,
    np: NormParams,
    chunk_ch: u32,
    thread_count: usize,
) !void {
    if (np.groups == 0 or np.channels % np.groups != 0) return error.InvalidShape;
    const group_ch = np.channels / np.groups;
    if (chunk_ch == 0 or chunk_ch % group_ch != 0) return error.InvalidShape;
    if (np.channels % chunk_ch != 0) return error.InvalidShape;
    const hw: usize = @as(usize, np.height) * np.width;
    const elem = elemBytes(np.dtype);
    var cp = np;
    cp.channels = chunk_ch;
    cp.groups = chunk_ch / group_ch;
    var c0: u32 = 0;
    var g0: usize = 0;
    while (c0 < np.channels) : (c0 += chunk_ch) {
        const in_off = @as(usize, c0) * hw * elem;
        const st_off = g0 * 2 * @sizeOf(f32);
        const rc = zdraw_metal_run_vae_norm_stats_at(
            queue,
            pipe,
            input,
            in_off,
            stats,
            st_off,
            &cp,
            thread_count,
        );
        if (rc != 0) {
            return error.MetalDispatchFailed;
        }
        g0 += cp.groups;
    }
}

test "channel-chunked statistics are bit-identical to the whole-map pass" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const device = c.zdraw_metal_create_device() orelse return error.SkipZigTest;
    defer c.zdraw_metal_release_device(device);
    const queue = c.zdraw_metal_create_queue(device) orelse return error.SkipZigTest;
    defer c.zdraw_metal_release_queue(queue);
    var err: [1024]u8 = undefined;
    const pipe = try mpipe.required(device, vnorm_shader.vnorm.ptr, "vae_norm_stats_h_sq", &err);
    defer c.zdraw_metal_release_pipeline(pipe);

    // 128 channels in 32 groups over a 64x48 map of pseudo-random halves,
    // the top-stage shape family of the Klein decoder at a small size.
    const channels: u32 = 128;
    const groups: u32 = 32;
    const height: u32 = 64;
    const width: u32 = 48;
    const n: usize = @as(usize, channels) * height * width;
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(f16, n);
    defer alloc.free(data);
    var prng = std.Random.DefaultPrng.init(46);
    const r = prng.random();
    for (data) |*v| {
        const x: f32 = r.float(f32) * 4.0 - 2.0;
        v.* = @floatCast(x);
    }
    var input = try mbuffer.Buffer.fromBytes(device, std.mem.sliceAsBytes(data));
    defer input.deinit();
    const stats_bytes: usize = @as(usize, groups) * 2 * @sizeOf(f32);
    var whole = try mbuffer.Buffer.empty(device, stats_bytes);
    defer whole.deinit();
    var chunked = try mbuffer.Buffer.empty(device, stats_bytes);
    defer chunked.deinit();

    const np: NormParams = .{
        .channels = channels,
        .height = height,
        .width = width,
        .groups = groups,
        .dtype = 1,
        .bias_dtype = 1,
        .eps = 1e-6,
        .weight_offset = 0,
        .bias_offset = 0,
    };
    // The product chain runs this kernel with 512 threads per group.
    const threads: usize = 512;
    const whole_rc = zdraw_metal_run_vae_norm_stats_at(
        queue,
        pipe,
        input.handle,
        0,
        whole.handle,
        0,
        &np,
        threads,
    );
    if (whole_rc != 0) {
        return error.MetalDispatchFailed;
    }
    var expect: [256]u8 = undefined;
    c.zdraw_metal_read_buffer(whole.handle, &expect, stats_bytes);
    var nonzero = false;
    for (expect) |b| nonzero = nonzero or b != 0;
    try std.testing.expect(nonzero);

    // Chunks of 64, 16 and 4 channels (16, 4 and 1 groups per chunk).
    const zeros = [_]u8{0} ** 256;
    for ([_]u32{ 64, 16, 4 }) |chunk| {
        c.zdraw_metal_write_buffer(chunked.handle, &zeros, stats_bytes);
        try chunkedStats(queue, pipe, input.handle, chunked.handle, np, chunk, threads);
        var got: [256]u8 = undefined;
        c.zdraw_metal_read_buffer(chunked.handle, &got, stats_bytes);
        try std.testing.expectEqualSlices(u8, expect[0..stats_bytes], got[0..stats_bytes]);
    }
    // A chunk that is not a multiple of the group width is refused.
    try std.testing.expectError(
        error.InvalidShape,
        chunkedStats(queue, pipe, input.handle, chunked.handle, np, 6, threads),
    );
}

/// The strip-memory route for an up-block group: the chain's per-block
/// buffers and params, its statistics pipes, and the chunk/strip scratch.
pub fn dispatchStrips(
    ctx: *mconv.Context,
    stream: *chain.Ctx,
    sbufs: []const chain.StreamBuffers,
    ps: []const param.ResParams,
    cfg: chain.Config,
    sp: chain.StatsPipes,
    add_pipe: *anyopaque,
    wino_set: ?mvres_wino.Set,
) !void {
    const ws = wino_set orelse return error.InvalidShape;
    const strip_pipes = stream.strip orelse return error.InvalidShape;
    const threads: usize = if (sp.fast) 512 else stream.stats_threads;
    const stats2 = sp.stats2 orelse sp.stats;
    try dispatchGroup(
        ctx,
        stream,
        sbufs,
        ps,
        cfg,
        sp.stats,
        stats2,
        add_pipe,
        strip_pipes,
        &ws,
        threads,
    );
}

/// Tier 3: the finish runs in strips when the feature is a resident half
/// map and the strip pipes exist.
pub fn finishApplies(stream: *const chain.Ctx, final_f16: bool) bool {
    return tier() >= 3 and stream.strip != null and final_f16;
}

/// The strip finish over the last group's resident feature: statistics
/// once, then per strip the norm apply into the pool's strip-local float
/// scratch and the RGB conv. No whole-map f32 scratch is ever sized.
pub fn finishStrips(
    ctx: *mconv.Context,
    stream: *chain.Ctx,
    final: *anyopaque,
    views: vviews.Views,
    cfg: mvfinal.Config,
    out: []f32,
) !void {
    const pipes = stream.strip orelse return error.InvalidShape;
    const rows = mvres_pool.stripRowsFor(planAtTier(), cfg.width);
    const dev = stream.device;
    const norm_bytes = cfg.channels * (rows + 2) * cfg.width * @sizeOf(f32);
    const norm_strip = try stream.pool.norm_strip.handle(dev, norm_bytes);
    var stats = try mbuffer.Buffer.empty(dev, 32 * 2 * @sizeOf(f32));
    defer stats.deinit();
    const sf = mvfinal.StripFinish{
        .stats_pipe = stream.stats_h_pipe,
        .apply_pipe = pipes.apply_strip,
        .conv_pipe = pipes.conv_strip_in,
        .stats = stats.handle,
        .norm_strip = norm_strip,
        .strip_rows = @intCast(rows),
        .stats_threads = stream.stats_threads,
        .apply_threads = stream.add_threads,
    };
    const v = views;
    try mvfinal.runStrips(ctx, out, final, v.norm_w, v.norm_b, v.out_w, v.out_b, cfg, sf);
}
