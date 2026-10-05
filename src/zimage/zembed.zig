//! Z-Image token embedders.
//!
//! Image patches use a plain linear projection. Caption features first pass
//! through RMSNorm and then a linear projection.

const std = @import("std");

const linear = @import("../runtime/linear_fast.zig");
const mlinear = @import("../metal/mlinear.zig");
const ops = @import("../runtime/ops.zig");
const tensor = @import("../pack/tensor.zig");

pub const Config = struct {
    tokens: usize,
    in_dim: usize,
    hidden: usize,
    norm_eps: f32,
};

pub fn image(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    cfg: Config,
) !void {
    try check(out, input, cfg);
    try linear.runBatch(metal, out, input, weight, bias, cfg.tokens);
}

pub fn caption(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    norm_weight: tensor.View,
    weight: tensor.View,
    bias: tensor.View,
    scratch: []f32,
    cfg: Config,
) !void {
    try check(out, input, cfg);
    if (scratch.len != cfg.tokens * cfg.in_dim) return error.InvalidShape;
    // Norm every token first, then project once: linear() is linearBatch(1),
    // so the batched call runs the identical per-row kernel math while paying
    // one command-buffer round trip instead of one per caption token.
    for (0..cfg.tokens) |tok| {
        const row = scratch[tok * cfg.in_dim ..][0..cfg.in_dim];
        try ops.rmsNormView(row, inTok(input, cfg, tok), norm_weight, cfg.norm_eps);
    }
    try linear.runBatch(metal, out, scratch, weight, bias, cfg.tokens);
}

fn check(out: []const f32, input: []const f32, cfg: Config) !void {
    if (cfg.tokens == 0 or cfg.in_dim == 0 or cfg.hidden == 0) return error.InvalidShape;
    if (out.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (input.len != cfg.tokens * cfg.in_dim) return error.InvalidShape;
}

fn outTok(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn inTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    return data[tok * cfg.in_dim ..][0..cfg.in_dim];
}

test "image embed applies linear projection per token" {
    const cfg = Config{ .tokens = 1, .in_dim = 2, .hidden = 2, .norm_eps = 0.0 };
    const weight = tensor.View{ .dtype = .u8, .shape = &.{ 2, 2 }, .bytes = &.{ 1, 0, 0, 1 } };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 1, 2 } };
    var out = [_]f32{ 0.0, 0.0 };

    try image(null, &out, &.{ 3.0, 4.0 }, weight, bias, cfg);
    try std.testing.expectApproxEqAbs(4.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(6.0, out[1], 0.0001);
}

test "caption embed applies rms norm before linear" {
    const cfg = Config{ .tokens = 1, .in_dim = 2, .hidden = 2, .norm_eps = 0.0 };
    const norm = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 1, 1 } };
    const weight = tensor.View{ .dtype = .u8, .shape = &.{ 2, 2 }, .bytes = &.{ 1, 0, 0, 1 } };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 0, 0 } };
    var scratch = [_]f32{ 0.0, 0.0 };
    var out = [_]f32{ 0.0, 0.0 };

    try caption(null, &out, &.{ 1.0, 1.0 }, norm, weight, bias, &scratch, cfg);
    try std.testing.expectApproxEqAbs(1.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.0, out[1], 0.0001);
}
