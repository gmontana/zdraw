//! Z-Image feed-forward block.
//!
//! The transformer uses a SwiGLU MLP wrapped by two RMSNorms. AdaLN scale is
//! applied after the first norm when the caller supplies it.

const std = @import("std");

const linear = @import("linear_fast.zig");
const mlinear = @import("mlinear.zig");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    norm_eps: f32,
};

pub const Weights = struct {
    in_norm: tensor.View,
    out_norm: tensor.View,
    gate: tensor.View,
    down: tensor.View,
    up: tensor.View,
};

pub const Scratch = struct {
    norm: []f32,
    gate: []f32,
    up: []f32,
};

pub fn run(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    scale: ?[]const f32,
    weights: Weights,
    scratch: Scratch,
    cfg: Config,
) !void {
    try check(out, input, scale, scratch, cfg);
    try normalize(out, input, scale, weights.in_norm, cfg);
    try linear.gemmFfn(
        metal,
        out,
        out,
        weights.gate,
        weights.up,
        weights.down,
        scratch.gate,
        scratch.up,
        cfg.tokens,
    );
    try finish(out, weights.out_norm, scratch.norm, cfg);
}

fn normalize(
    out: []f32,
    input: []const f32,
    scale: ?[]const f32,
    weight: tensor.View,
    cfg: Config,
) !void {
    for (0..cfg.tokens) |tok| {
        const dst = outTok(out, cfg, tok);
        try ops.rmsNormView(dst, inTok(input, cfg, tok), weight, cfg.norm_eps);
        if (scale) |s| mul(dst, scaleTok(s, cfg, tok));
    }
}

fn finish(out: []f32, weight: tensor.View, tmp: []f32, cfg: Config) !void {
    for (0..cfg.tokens) |tok| {
        const dst = outTok(out, cfg, tok);
        try ops.rmsNormView(tmp, dst, weight, cfg.norm_eps);
        copy(dst, tmp);
    }
}

fn check(
    out: []const f32,
    input: []const f32,
    scale: ?[]const f32,
    scratch: Scratch,
    cfg: Config,
) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0) return error.InvalidShape;
    if (out.len != cfg.tokens * cfg.hidden or input.len != out.len) return error.InvalidShape;
    if (scale) |s| {
        if (s.len != cfg.hidden and s.len != out.len) return error.InvalidShape;
    }
    if (scratch.norm.len != cfg.hidden) return error.InvalidShape;
    if (scratch.gate.len == 0 or scratch.gate.len != scratch.up.len) return error.InvalidShape;
    if (scratch.gate.len % cfg.tokens != 0) return error.InvalidShape;
}

fn mul(dst: []f32, scale: []const f32) void {
    for (dst, scale) |*value, s| value.* *= s;
}

fn copy(dst: []f32, src: []const f32) void {
    for (dst, src) |*value, in| value.* = in;
}

fn outTok(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn inTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn scaleTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    if (data.len == cfg.hidden) return data;
    return inTok(data, cfg, tok);
}

test "feed-forward applies swiglu and output norm" {
    const cfg = Config{ .tokens = 1, .hidden = 2, .norm_eps = 0.0 };
    const norm = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 1, 1 } };
    const in_w = tensor.View{ .dtype = .u8, .shape = &.{ 1, 2 }, .bytes = &.{ 1, 1 } };
    const down = tensor.View{ .dtype = .u8, .shape = &.{ 2, 1 }, .bytes = &.{ 1, 0 } };
    var norm_buf = [_]f32{ 0.0, 0.0 };
    var gate = [_]f32{0.0};
    var up = [_]f32{0.0};
    var out = [_]f32{ 0.0, 0.0 };

    try run(null, &out, &.{ 1.0, 1.0 }, null, .{
        .in_norm = norm,
        .out_norm = norm,
        .gate = in_w,
        .down = down,
        .up = in_w,
    }, .{
        .norm = &norm_buf,
        .gate = &gate,
        .up = &up,
    }, cfg);

    try std.testing.expectApproxEqAbs(@sqrt(2.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(0.0, out[1], 0.0001);
}
