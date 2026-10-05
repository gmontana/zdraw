//! Tests for the Metal convolution fast path.

const std = @import("std");

const conv = @import("conv.zig");
const mconv = @import("mconv.zig");
const tensor = @import("tensor.zig");

const weight_bytes = [_]u8{
    0x00, 0x00, 0x80, 0x3f,
    0x00, 0x00, 0x00, 0x40,
    0x00, 0x00, 0x40, 0x40,
    0x00, 0x00, 0x80, 0x40,
};

fn weight() tensor.View {
    return .{ .dtype = .f32, .shape = &.{ 1, 1, 2, 2 }, .bytes = &weight_bytes };
}

test "Metal convolution matches padded CPU kernel" {
    var ctx = mconv.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();

    const cfg = conv.Config{
        .in_ch = 1,
        .out_ch = 1,
        .height = 2,
        .width = 2,
        .kernel = 2,
        .pad = 0,
    };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var got = [_]f32{0.0} ** 4;
    var want = [_]f32{0.0} ** 4;

    try ctx.run(&got, &input, weight(), null, cfg);
    try conv.run(&want, &input, weight(), null, cfg);
    try std.testing.expectEqualSlices(f32, &want, &got);
}
