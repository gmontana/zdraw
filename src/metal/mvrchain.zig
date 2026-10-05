//! Chain three VAE residual blocks without CPU-visible boundaries.

const std = @import("std");

const c = @import("metal_c.zig");
const chain_buf = @import("mvrchain_buf.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const bufs = @import("mvres_buf.zig");
const params = @import("mvres_param.zig");
const vres = @import("../vae/vres.zig");

pub const Config = chain_buf.Config;
const block_count = chain_buf.block_count;

extern fn zdraw_metal_run_vae_res_chain(
    queue: *anyopaque,
    norm_pipeline: *anyopaque,
    conv_pipeline: *anyopaque,
    add_pipeline: *anyopaque,
    buffers: [*]const bufs.ResBuffers,
    ps: [*]const params.ResParams,
    count: usize,
    norm_threads: usize,
    conv_threads: usize,
    add_threads: usize,
) c_int;

pub fn run(
    ctx: *mconv.Context,
    out: []f32,
    input: []const f32,
    views: [block_count]vres.Views,
    cfg: Config,
    recycle: ?mbuffer.Recycle,
) !void {
    try check(out, input, views, cfg);
    var set = try chain_buf.Set.make(ctx, input, views, cfg, recycle);
    defer set.deinit();
    var temps = [_]bufs.Temps{.{}} ** block_count;
    defer deinitTemps(&temps);
    var cbufs: [block_count]bufs.ResBuffers = undefined;
    var ps: [block_count]params.ResParams = undefined;
    try bindBlocks(ctx, &set, views, &temps, &cbufs, &ps, cfg);
    try dispatch(ctx, &cbufs, &ps);
    readBack(set.finalOutput(), out);
}

fn check(
    out: []const f32,
    input: []const f32,
    views: [block_count]vres.Views,
    cfg: Config,
) !void {
    if (out.len != chain_buf.elemCount(cfg.out_ch, cfg)) return error.InvalidShape;
    if (input.len != chain_buf.elemCount(cfg.in_ch, cfg)) return error.InvalidShape;
    for (views, 0..) |view, idx| {
        const in_slice = if (idx == 0) input else out;
        try params.check(out, in_slice, view, chain_buf.blockCfg(cfg, idx));
    }
}

fn bindBlocks(
    ctx: *mconv.Context,
    set: *const chain_buf.Set,
    views: [block_count]vres.Views,
    temps: *[block_count]bufs.Temps,
    cbufs: *[block_count]bufs.ResBuffers,
    ps: *[block_count]params.ResParams,
    cfg: Config,
) !void {
    for (views, 0..) |view, idx| {
        const binds = try bufs.bindAll(ctx, view, &temps[idx]);
        cbufs[idx] = set.resBuffers(idx, binds);
        ps[idx] = try params.make(view, binds, chain_buf.blockCfg(cfg, idx));
    }
}

fn dispatch(
    ctx: *mconv.Context,
    cbufs: *const [block_count]bufs.ResBuffers,
    ps: *const [block_count]params.ResParams,
) !void {
    const code = zdraw_metal_run_vae_res_chain(
        ctx.queue,
        ctx.norm_pipeline,
        ctx.pipeline,
        ctx.add_pipeline,
        cbufs.ptr,
        ps.ptr,
        block_count,
        ctx.norm_threads,
        ctx.threads,
        ctx.add_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(
        buf.handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

fn deinitTemps(temps: *[block_count]bufs.Temps) void {
    for (temps) |*temp| temp.deinit();
}
