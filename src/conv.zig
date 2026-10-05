//! Small NCHW convolution kernel for VAE decoder work.
//!
//! This is a direct CPU reference path. Faster backends can keep the same
//! shape contract later.

const std = @import("std");

const tensor = @import("tensor.zig");

pub const Config = struct {
    in_ch: usize,
    out_ch: usize,
    height: usize,
    width: usize,
    kernel: usize,
    pad: usize,
};

pub fn run(
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: Config,
) !void {
    try check(out, input, weight, bias, cfg);
    for (0..cfg.out_ch) |oc| {
        for (0..cfg.height) |row| {
            for (0..cfg.width) |col| {
                out[outIdx(cfg, oc, row, col)] = try value(input, weight, bias, cfg, oc, row, col);
            }
        }
    }
}

fn value(
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: Config,
    oc: usize,
    row: usize,
    col: usize,
) !f32 {
    var sum: f32 = if (bias) |b| b.atF32Unchecked(oc) else 0.0;
    for (0..cfg.in_ch) |ic| {
        for (0..cfg.kernel) |kr| {
            for (0..cfg.kernel) |kc| {
                if (source(row, col, kr, kc, cfg)) |src| {
                    sum += input[inIdx(cfg, ic, src[0], src[1])] *
                        weight.atF32Unchecked(wIdx(cfg, oc, ic, kr, kc));
                }
            }
        }
    }
    return sum;
}

fn source(row: usize, col: usize, kr: usize, kc: usize, cfg: Config) ?[2]usize {
    const r = @as(isize, @intCast(row)) + @as(isize, @intCast(kr)) -
        @as(isize, @intCast(cfg.pad));
    const c = @as(isize, @intCast(col)) + @as(isize, @intCast(kc)) -
        @as(isize, @intCast(cfg.pad));
    if (r < 0 or c < 0) return null;
    const rr: usize = @intCast(r);
    const cc: usize = @intCast(c);
    if (rr >= cfg.height or cc >= cfg.width) return null;
    return .{ rr, cc };
}

fn check(
    out: []const f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: Config,
) !void {
    if (cfg.in_ch == 0 or cfg.out_ch == 0) return error.InvalidShape;
    if (cfg.height == 0 or cfg.width == 0 or cfg.kernel == 0) return error.InvalidShape;
    if (out.len != cfg.out_ch * cfg.height * cfg.width) return error.InvalidShape;
    if (input.len != cfg.in_ch * cfg.height * cfg.width) return error.InvalidShape;
    try checkWeight(weight, cfg);
    if (bias) |b| {
        if (try b.elems() != cfg.out_ch) return error.InvalidShape;
        try b.check();
    }
}

fn checkWeight(weight: tensor.View, cfg: Config) !void {
    if (weight.shape.len != 4) return error.InvalidShape;
    if (weight.shape[0] != cfg.out_ch or weight.shape[1] != cfg.in_ch) {
        return error.InvalidShape;
    }
    if (weight.shape[2] != cfg.kernel or weight.shape[3] != cfg.kernel) {
        return error.InvalidShape;
    }
    try weight.check();
}

fn outIdx(cfg: Config, ch: usize, row: usize, col: usize) usize {
    return (ch * cfg.height + row) * cfg.width + col;
}

fn inIdx(cfg: Config, ch: usize, row: usize, col: usize) usize {
    return (ch * cfg.height + row) * cfg.width + col;
}

fn wIdx(cfg: Config, oc: usize, ic: usize, row: usize, col: usize) usize {
    return ((oc * cfg.in_ch + ic) * cfg.kernel + row) * cfg.kernel + col;
}

test "one by one convolution with bias" {
    const cfg = Config{ .in_ch = 1, .out_ch = 1, .height = 2, .width = 2, .kernel = 1, .pad = 0 };
    const weight = tensor.View{ .dtype = .u8, .shape = &.{ 1, 1, 1, 1 }, .bytes = &.{2} };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{1}, .bytes = &.{1} };
    var out = [_]f32{0} ** 4;

    try run(&out, &.{ 1, 2, 3, 4 }, weight, bias, cfg);
    try std.testing.expectEqualSlices(f32, &.{ 3, 5, 7, 9 }, &out);
}

test "padded convolution skips outside pixels" {
    const cfg = Config{ .in_ch = 1, .out_ch = 1, .height = 1, .width = 1, .kernel = 3, .pad = 1 };
    const bytes = [_]u8{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const weight = tensor.View{ .dtype = .u8, .shape = &.{ 1, 1, 3, 3 }, .bytes = &bytes };
    var out = [_]f32{0};

    try run(&out, &.{2}, weight, null, cfg);
    try std.testing.expectApproxEqAbs(2.0, out[0], 0.0001);
}
