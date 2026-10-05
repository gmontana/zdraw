//! VAE residual block.
//!
//! This mirrors the Diffusers AutoencoderKL decoder block: two norm/SiLU/conv
//! stages and a residual shortcut. There is no time embedding in this VAE.

const std = @import("std");

const conv = @import("conv.zig");
const conv_fast = @import("conv_fast.zig");
const gnorm = @import("gnorm.zig");
const mconv = @import("mconv.zig");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

pub const Config = struct {
    in_ch: usize,
    out_ch: usize,
    height: usize,
    width: usize,
    groups: usize = 32,
    eps: f32 = 0.000001,
};

pub const Views = struct {
    norm1_w: tensor.View,
    norm1_b: tensor.View,
    conv1_w: tensor.View,
    conv1_b: tensor.View,
    norm2_w: tensor.View,
    norm2_b: tensor.View,
    conv2_w: tensor.View,
    conv2_b: tensor.View,
    skip_w: ?tensor.View = null,
    skip_b: ?tensor.View = null,
};

pub const Scratch = struct {
    norm: []f32,
    work: []f32,
    skip: []f32,
};

pub fn run(
    metal: ?*mconv.Context,
    out: []f32,
    input: []const f32,
    views: Views,
    scratch: Scratch,
    cfg: Config,
) !void {
    try check(out, input, views, scratch, cfg);
    const in_len = len(cfg.in_ch, cfg);
    const out_len = len(cfg.out_ch, cfg);
    const norm = scratch.norm[0..in_len];
    const work = scratch.work[0..out_len];

    try gnorm.run(norm, input, views.norm1_w, views.norm1_b, normCfg(cfg, cfg.in_ch));
    activate(norm);
    try conv_fast.run(metal, work, norm, views.conv1_w, views.conv1_b, convCfg(
        cfg.in_ch,
        cfg.out_ch,
        cfg,
        3,
        1,
    ));

    try gnorm.run(work, work, views.norm2_w, views.norm2_b, normCfg(cfg, cfg.out_ch));
    activate(work);
    try conv_fast.run(metal, out, work, views.conv2_w, views.conv2_b, convCfg(
        cfg.out_ch,
        cfg.out_ch,
        cfg,
        3,
        1,
    ));
    try addSkip(metal, out, input, views, scratch.skip[0..out_len], cfg);
}

fn addSkip(
    metal: ?*mconv.Context,
    out: []f32,
    input: []const f32,
    views: Views,
    skip: []f32,
    cfg: Config,
) !void {
    const src = if (views.skip_w) |weight| blk: {
        try conv_fast.run(
            metal,
            skip,
            input,
            weight,
            views.skip_b,
            convCfg(cfg.in_ch, cfg.out_ch, cfg, 1, 0),
        );
        break :blk skip;
    } else input;
    if (src.len != out.len) return error.InvalidShape;
    for (out, src) |*value, residual| value.* += residual;
}

fn check(
    out: []const f32,
    input: []const f32,
    views: Views,
    scratch: Scratch,
    cfg: Config,
) !void {
    if (out.len != len(cfg.out_ch, cfg) or input.len != len(cfg.in_ch, cfg)) {
        return error.InvalidShape;
    }
    if (views.skip_w == null and cfg.in_ch != cfg.out_ch) return error.InvalidShape;
    if (scratch.norm.len < input.len or scratch.work.len < out.len) return error.InvalidShape;
    if (views.skip_w != null and scratch.skip.len < out.len) return error.InvalidShape;
}

fn activate(values: []f32) void {
    for (values) |*value| value.* = ops.silu(value.*);
}

fn normCfg(cfg: Config, channels: usize) gnorm.Config {
    return .{
        .channels = channels,
        .height = cfg.height,
        .width = cfg.width,
        .groups = cfg.groups,
        .eps = cfg.eps,
    };
}

fn convCfg(in_ch: usize, out_ch: usize, cfg: Config, kernel: usize, pad: usize) conv.Config {
    return .{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = cfg.height,
        .width = cfg.width,
        .kernel = kernel,
        .pad = pad,
    };
}

fn len(channels: usize, cfg: Config) usize {
    return channels * cfg.height * cfg.width;
}

test "residual block keeps the shortcut" {
    const cfg = Config{ .in_ch = 1, .out_ch = 1, .height = 1, .width = 1, .groups = 1, .eps = 1.0 };
    const views = Views{
        .norm1_w = scalar(&one1),
        .norm1_b = scalar(&one1),
        .conv1_w = conv3(&center1),
        .conv1_b = scalar(&zero1),
        .norm2_w = scalar(&one1),
        .norm2_b = scalar(&one1),
        .conv2_w = conv3(&center2),
        .conv2_b = scalar(&zero1),
    };
    var norm = [_]f32{0};
    var work = [_]f32{0};
    var skip = [_]f32{0};
    var out = [_]f32{0};

    try run(null, &out, &.{2.0}, views, .{ .norm = &norm, .work = &work, .skip = &skip }, cfg);
    try std.testing.expectApproxEqAbs(2.0 + ops.silu(1.0) * 2.0, out[0], 0.0001);
}

test "residual block projects the shortcut" {
    const cfg = Config{ .in_ch = 1, .out_ch = 2, .height = 1, .width = 1, .groups = 1, .eps = 1.0 };
    const views = Views{
        .norm1_w = scalar(&one1),
        .norm1_b = scalar(&zero1),
        .conv1_w = convShape(&zero18, &.{ 2, 1, 3, 3 }),
        .conv1_b = vec2(&zero2),
        .norm2_w = vec2(&one2),
        .norm2_b = vec2(&zero2),
        .conv2_w = convShape(&zero36, &.{ 2, 2, 3, 3 }),
        .conv2_b = vec2(&zero2),
        .skip_w = convShape(&.{ 1, 2 }, &.{ 2, 1, 1, 1 }),
        .skip_b = vec2(&skip_b2),
    };
    var norm = [_]f32{0};
    var work = [_]f32{0} ** 2;
    var skip = [_]f32{0} ** 2;
    var out = [_]f32{0} ** 2;

    try run(null, &out, &.{3.0}, views, .{ .norm = &norm, .work = &work, .skip = &skip }, cfg);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 7.0 }, &out);
}

fn scalar(bytes: []const u8) tensor.View {
    return .{ .dtype = .u8, .shape = &shape1, .bytes = bytes };
}

fn vec2(bytes: []const u8) tensor.View {
    return .{ .dtype = .u8, .shape = &shape2, .bytes = bytes };
}

fn conv3(bytes: []const u8) tensor.View {
    return convShape(bytes, &.{ 1, 1, 3, 3 });
}

fn convShape(bytes: []const u8, shape: []const usize) tensor.View {
    return .{ .dtype = .u8, .shape = shape, .bytes = bytes };
}

const center1 = [_]u8{ 0, 0, 0, 0, 1, 0, 0, 0, 0 };
const center2 = [_]u8{ 0, 0, 0, 0, 2, 0, 0, 0, 0 };
const shape1 = [_]usize{1};
const shape2 = [_]usize{2};
const zero1 = [_]u8{0};
const one1 = [_]u8{1};
const zero2 = [_]u8{ 0, 0 };
const one2 = [_]u8{ 1, 1 };
const skip_b2 = [_]u8{ 0, 1 };
const zero18 = [_]u8{0} ** 18;
const zero36 = [_]u8{0} ** 36;
