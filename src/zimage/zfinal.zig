//! Final Z-Image patch projection.
//!
//! After the transformer stack, Z-Image applies LayerNorm, a timestep-derived
//! scale, and a final linear layer back to latent patch values.

const std = @import("std");

const linear = @import("../runtime/linear_fast.zig");
const mfinal = @import("../metal/mfinal.zig");
const mlinear = @import("../metal/mlinear.zig");
const mfallback = @import("../metal/metal_fallback.zig");
const ops = @import("../runtime/ops.zig");
const tensor = @import("../pack/tensor.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    out_dim: usize,
    norm_eps: f32,
};

pub const Views = struct {
    mod_w: tensor.View,
    mod_b: tensor.View,
    linear_w: tensor.View,
    linear_b: tensor.View,
};

pub const Scratch = struct {
    cond: []f32,
    scale: []f32,
    norm: []f32,
    batch: []f32,
};

pub fn run(
    metal: ?*mlinear.Context,
    out: []f32,
    state: []const f32,
    adaln: []const f32,
    views: Views,
    scratch: Scratch,
    cfg: Config,
) !void {
    try check(out, state, scratch, cfg);
    if (scratch.cond.len != adaln.len) return error.InvalidShape;
    for (scratch.cond, adaln) |*dst, value| dst.* = ops.silu(value);
    try linear.run(metal, scratch.scale, scratch.cond, views.mod_w, views.mod_b);
    for (scratch.scale) |*value| value.* += 1.0;

    if (try projectMetal(metal, out, state, scratch.scale, views, cfg)) return;
    for (0..cfg.tokens) |tok| {
        const dst = batchTok(scratch.batch, cfg, tok);
        try layerNorm(dst, inTok(state, cfg, tok), cfg.norm_eps);
        for (dst, scratch.scale) |*value, scale| value.* *= scale;
    }
    try linear.runBatch(metal, out, scratch.batch, views.linear_w, views.linear_b, cfg.tokens);
}

fn projectMetal(
    metal: ?*mlinear.Context,
    out: []f32,
    state: []const f32,
    scale: []const f32,
    views: Views,
    cfg: Config,
) !bool {
    const ctx = metal orelse return false;
    mfinal.run(ctx, out, state, scale, views.linear_w, views.linear_b, .{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .out_dim = cfg.out_dim,
        .eps = cfg.norm_eps,
    }) catch |err| {
        if (mfallback.isGemmRefusal(err)) return false;
        return err;
    };
    return true;
}

pub fn layerNorm(out: []f32, input: []const f32, eps: f32) !void {
    if (out.len != input.len or input.len == 0) return error.InvalidShape;
    var mean: f32 = 0.0;
    for (input) |value| mean += value;
    mean /= @floatFromInt(input.len);

    var var_sum: f32 = 0.0;
    for (input) |value| {
        const diff = value - mean;
        var_sum += diff * diff;
    }
    const scale = 1.0 / @sqrt(var_sum / @as(f32, @floatFromInt(input.len)) + eps);
    for (out, input) |*dst, value| dst.* = (value - mean) * scale;
}

fn check(out: []const f32, state: []const f32, scratch: Scratch, cfg: Config) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0 or cfg.out_dim == 0) return error.InvalidShape;
    if (state.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (out.len != cfg.tokens * cfg.out_dim) return error.InvalidShape;
    if (scratch.scale.len != cfg.hidden or scratch.norm.len != cfg.hidden) {
        return error.InvalidShape;
    }
    if (scratch.batch.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
}

fn inTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn batchTok(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

test "layer norm centers and scales token" {
    var out = [_]f32{ 0.0, 0.0 };
    try layerNorm(&out, &.{ 1.0, 3.0 }, 0.0);

    try std.testing.expectApproxEqAbs(-1.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.0, out[1], 0.0001);
}

test "final projection uses timestep scale" {
    const cfg = Config{ .tokens = 1, .hidden = 2, .out_dim = 1, .norm_eps = 0.0 };
    const mod_w = tensor.View{ .dtype = .u8, .shape = &.{ 2, 1 }, .bytes = &.{ 0, 0 } };
    const mod_b = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 0, 0 } };
    const linear_w = tensor.View{ .dtype = .u8, .shape = &.{ 1, 2 }, .bytes = &.{ 0, 1 } };
    const linear_b = tensor.View{ .dtype = .u8, .shape = &.{1}, .bytes = &.{0} };
    var cond = [_]f32{0.0};
    var scale = [_]f32{ 0.0, 0.0 };
    var norm = [_]f32{ 0.0, 0.0 };
    var batch = [_]f32{ 0.0, 0.0 };
    var out = [_]f32{0.0};

    try run(null, &out, &.{ 1.0, 3.0 }, &.{0.0}, .{
        .mod_w = mod_w,
        .mod_b = mod_b,
        .linear_w = linear_w,
        .linear_b = linear_b,
    }, .{
        .cond = &cond,
        .scale = &scale,
        .norm = &norm,
        .batch = &batch,
    }, cfg);

    try std.testing.expectApproxEqAbs(1.0, out[0], 0.0001);
}
