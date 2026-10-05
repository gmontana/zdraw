//! GroupNorm for VAE decoder tensors.
//!
//! Input and output use NCHW without a batch dimension.

const std = @import("std");

const tensor = @import("../pack/tensor.zig");

pub const Config = struct {
    channels: usize,
    height: usize,
    width: usize,
    groups: usize,
    eps: f32,
};

pub fn run(
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    cfg: Config,
) !void {
    try check(out, input, weight, bias, cfg);
    const group_ch = cfg.channels / cfg.groups;
    for (0..cfg.groups) |group| {
        const start = group * group_ch;
        const stats = groupStats(input, cfg, start, group_ch);
        normGroup(out, input, weight, bias, cfg, start, group_ch, stats);
    }
}

const Stats = struct {
    mean: f32,
    scale: f32,
};

fn groupStats(input: []const f32, cfg: Config, start: usize, count: usize) Stats {
    const n = count * cfg.height * cfg.width;
    var mean: f32 = 0.0;
    for (start..start + count) |ch| {
        for (0..cfg.height) |row| {
            for (0..cfg.width) |col| mean += input[idx(cfg, ch, row, col)];
        }
    }
    mean /= @floatFromInt(n);

    var var_sum: f32 = 0.0;
    for (start..start + count) |ch| {
        for (0..cfg.height) |row| {
            for (0..cfg.width) |col| {
                const diff = input[idx(cfg, ch, row, col)] - mean;
                var_sum += diff * diff;
            }
        }
    }
    return .{ .mean = mean, .scale = 1.0 / @sqrt(var_sum / @as(f32, @floatFromInt(n)) + cfg.eps) };
}

fn normGroup(
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    cfg: Config,
    start: usize,
    count: usize,
    stats: Stats,
) void {
    for (start..start + count) |ch| {
        const gain = weight.atF32Unchecked(ch);
        const shift = bias.atF32Unchecked(ch);
        for (0..cfg.height) |row| {
            for (0..cfg.width) |col| {
                const i = idx(cfg, ch, row, col);
                out[i] = (input[i] - stats.mean) * stats.scale * gain + shift;
            }
        }
    }
}

fn check(
    out: []const f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    cfg: Config,
) !void {
    if (cfg.channels == 0 or cfg.groups == 0) return error.InvalidShape;
    if (cfg.height == 0 or cfg.width == 0) return error.InvalidShape;
    if (cfg.channels % cfg.groups != 0) return error.InvalidShape;
    if (out.len != input.len or input.len != cfg.channels * cfg.height * cfg.width) {
        return error.InvalidShape;
    }
    if (try weight.elems() != cfg.channels or try bias.elems() != cfg.channels) {
        return error.InvalidShape;
    }
    try weight.check();
    try bias.check();
}

fn idx(cfg: Config, ch: usize, row: usize, col: usize) usize {
    return (ch * cfg.height + row) * cfg.width + col;
}

test "single group normalizes all channels together" {
    const cfg = Config{ .channels = 1, .height = 1, .width = 2, .groups = 1, .eps = 0.0 };
    const weight = tensor.View{ .dtype = .u8, .shape = &.{1}, .bytes = &.{1} };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{1}, .bytes = &.{0} };
    var out = [_]f32{ 0.0, 0.0 };

    try run(&out, &.{ 1.0, 3.0 }, weight, bias, cfg);
    try std.testing.expectApproxEqAbs(-1.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.0, out[1], 0.0001);
}

test "separate groups normalize independently" {
    const cfg = Config{ .channels = 2, .height = 1, .width = 1, .groups = 2, .eps = 1.0 };
    const weight = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 2, 3 } };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 4, 5 } };
    var out = [_]f32{ 0.0, 0.0 };

    try run(&out, &.{ 7.0, 9.0 }, weight, bias, cfg);
    try std.testing.expectApproxEqAbs(4.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(5.0, out[1], 0.0001);
}
