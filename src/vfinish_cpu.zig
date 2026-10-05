//! CPU fallback for the VAE decoder final norm + RGB conv.

const std = @import("std");

const conv_fast = @import("conv_fast.zig");
const gnorm = @import("gnorm.zig");
const ops = @import("ops.zig");
const vviews = @import("vviews.zig");

pub const Config = struct { channels: usize, height: usize, width: usize };

pub fn run(
    allocator: std.mem.Allocator,
    out: []f32,
    input: []const f32,
    views: vviews.Views,
    cfg: Config,
) !void {
    const norm = try allocator.alloc(f32, input.len);
    defer allocator.free(norm);
    try gnorm.run(norm, input, views.norm_w, views.norm_b, .{
        .channels = cfg.channels,
        .height = cfg.height,
        .width = cfg.width,
        .groups = 32,
        .eps = 0.000001,
    });
    for (norm) |*value| value.* = ops.silu(value.*);
    try conv_fast.run(null, out, norm, views.out_w, views.out_b, .{
        .in_ch = cfg.channels,
        .out_ch = 3,
        .height = cfg.height,
        .width = cfg.width,
        .kernel = 3,
        .pad = 1,
    });
}
