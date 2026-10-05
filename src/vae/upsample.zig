//! Nearest-neighbor upsampling for VAE decoder tensors.

const std = @import("std");

pub const Config = struct {
    channels: usize,
    height: usize,
    width: usize,
    scale: usize = 2,
};

pub fn run(out: []f32, input: []const f32, cfg: Config) !void {
    try check(out, input, cfg);
    const out_h = cfg.height * cfg.scale;
    const out_w = cfg.width * cfg.scale;
    for (0..cfg.channels) |ch| {
        for (0..out_h) |row| {
            for (0..out_w) |col| {
                out[outIdx(ch, row, col, out_h, out_w)] =
                    input[inIdx(cfg, ch, row / cfg.scale, col / cfg.scale)];
            }
        }
    }
}

fn check(out: []const f32, input: []const f32, cfg: Config) !void {
    if (cfg.channels == 0 or cfg.height == 0 or cfg.width == 0) {
        return error.InvalidShape;
    }
    if (cfg.scale == 0) return error.InvalidShape;
    if (input.len != cfg.channels * cfg.height * cfg.width) {
        return error.InvalidShape;
    }
    if (out.len != cfg.channels * cfg.height * cfg.width * cfg.scale * cfg.scale) {
        return error.InvalidShape;
    }
}

fn inIdx(cfg: Config, ch: usize, row: usize, col: usize) usize {
    return (ch * cfg.height + row) * cfg.width + col;
}

fn outIdx(ch: usize, row: usize, col: usize, height: usize, width: usize) usize {
    return (ch * height + row) * width + col;
}

test "upsample one channel by two" {
    const cfg = Config{ .channels = 1, .height = 1, .width = 2 };
    var out = [_]f32{0} ** 8;

    try run(&out, &.{ 3.0, 4.0 }, cfg);
    try std.testing.expectEqualSlices(f32, &.{ 3, 3, 4, 4, 3, 3, 4, 4 }, &out);
}

test "reject wrong output shape" {
    const cfg = Config{ .channels = 1, .height = 1, .width = 1 };
    var out = [_]f32{0} ** 3;

    try std.testing.expectError(error.InvalidShape, run(&out, &.{1.0}, cfg));
}
