//! Tests for the Z-Image attention block.

const std = @import("std");

const tensor = @import("../pack/tensor.zig");
const zattn = @import("zattn.zig");
const zrope = @import("zrope.zig");

test "single token attention step" {
    const cfg = zattn.Config{
        .tokens = 1,
        .hidden = 6,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 6,
        .norm_eps = 0.0,
    };
    const norm = tensor.View{ .dtype = .u8, .shape = &.{6}, .bytes = &.{ 1, 1, 1, 1, 1, 1 } };
    const mat = tensor.View{ .dtype = .u8, .shape = &.{ 6, 6 }, .bytes = &id6 };
    const rope = try zrope.Cache.init(std.testing.allocator, .{
        .dims = .{ 2, 2, 2 },
        .lens = .{ 1, 1, 1 },
        .theta = 256.0,
    });
    defer rope.deinit(std.testing.allocator);
    var norm_buf = [_]f32{0.0} ** 6;
    var q = [_]f32{0.0} ** 6;
    var k = [_]f32{0.0} ** 6;
    var v = [_]f32{0.0} ** 6;
    var mix = [_]f32{0.0} ** 6;
    var scores = [_]f32{0.0};
    var out = [_]f32{0.0} ** 6;

    try zattn.run(null, null, &out, &.{ 1, 1, 1, 1, 1, 1 }, null, &.{.{ 0, 0, 0 }}, .{
        .norm = norm,
        .q = mat,
        .k = mat,
        .v = mat,
        .o = mat,
        .q_norm = norm,
        .k_norm = norm,
    }, .{
        .norm = &norm_buf,
        .q = &q,
        .k = &k,
        .v = &v,
        .mix = &mix,
        .scores = &scores,
    }, cfg, rope);

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, &out);
}

const id6 = [_]u8{
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
};
