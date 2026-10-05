//! Final VAE norm plus RGB projection boundary.

const std = @import("std");

const mbuffer = @import("../metal/mbuffer.zig");
const mconv = @import("../metal/mconv.zig");
const mvfinal = @import("../metal/mvfinal.zig");
const vfinish = @import("vfinish_cpu.zig");
const vviews = @import("vviews.zig");

pub fn run(
    metal: ?*mconv.Context,
    allocator: std.mem.Allocator,
    out: []f32,
    input: []const f32,
    channels: usize,
    height: usize,
    width: usize,
    views: vviews.Views,
    recycle: ?mbuffer.Recycle,
    scratch: ?mvfinal.Scratch,
) !void {
    if (out.len != 3 * height * width or channels != 128) return error.InvalidShape;
    if (metal) |ctx| return mvfinal.run(
        ctx,
        out,
        input,
        views.norm_w,
        views.norm_b,
        views.out_w,
        views.out_b,
        .{ .channels = channels, .height = height, .width = width },
        recycle,
        scratch,
    );
    try vfinish.run(allocator, out, input, views, .{
        .channels = channels,
        .height = height,
        .width = width,
    });
}
