//! Host wrappers for block-local Metal norm/residual kernels.

const std = @import("std");

const mblock_c = @import("mblock_c.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const util = @import("mres_util.zig");
const tensor = @import("tensor.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    norm_eps: f32,
};

pub fn normScale(
    ctx: *mlinear.Context,
    out: mbuffer.Buffer,
    input: mbuffer.Buffer,
    weight: tensor.View,
    scale: ?[]const f32,
    cfg: Config,
) !void {
    const pipe = ctx.norm_pipeline orelse return error.GemmUnavailable;
    try run(ctx, pipe, out, input, weight, scale, cfg);
}

pub fn residual(
    ctx: *mlinear.Context,
    state: mbuffer.Buffer,
    input: mbuffer.Buffer,
    weight: tensor.View,
    gate: ?[]const f32,
    cfg: Config,
) !void {
    const pipe = ctx.resid_pipeline orelse return error.GemmUnavailable;
    try run(ctx, pipe, state, input, weight, gate, cfg);
}

fn run(
    ctx: *mlinear.Context,
    pipe: *anyopaque,
    output: mbuffer.Buffer,
    input: mbuffer.Buffer,
    weight: tensor.View,
    scale: ?[]const f32,
    cfg: Config,
) !void {
    try check(weight, scale, cfg);
    var weight_tmp: ?mbuffer.Buffer = null;
    defer if (weight_tmp) |*buf| buf.deinit();
    var scale_buf = try scaleBuffer(ctx, scale);
    defer if (scale_buf) |*buf| buf.deinit();
    const weight_bind = try ctx.buffers.bindView(weight, &weight_tmp);
    const scale_handle = if (scale_buf) |buf| buf.handle else ctx.buffers.zero;
    const params = try makeParams(weight, weight_bind, scale, cfg);
    const code = mblock_c.zdraw_metal_run_block_kernel(
        ctx.queue,
        pipe,
        input.handle,
        weight_bind.handle,
        scale_handle,
        output.handle,
        &params,
        ctx.block_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn scaleBuffer(ctx: *mlinear.Context, scale: ?[]const f32) !?mbuffer.Buffer {
    if (scale) |s| {
        return try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(s));
    }
    return null;
}

fn makeParams(
    weight: tensor.View,
    bind: mbuffer.Bind,
    scale: ?[]const f32,
    cfg: Config,
) !mblock_c.Params {
    return .{
        .tokens = try util.toU32(cfg.tokens),
        .hidden = try util.toU32(cfg.hidden),
        .dtype = try util.dtype(weight.dtype),
        .has_scale = if (scale == null) 0 else 1,
        .eps = cfg.norm_eps,
        .weight_offset = try util.toU64(bind.offset),
    };
}

fn check(weight: tensor.View, scale: ?[]const f32, cfg: Config) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0) return error.InvalidShape;
    if (try weight.elems() != cfg.hidden) return error.InvalidShape;
    if (scale) |s| {
        if (s.len != cfg.hidden) return error.InvalidShape;
    }
    try weight.check();
}
