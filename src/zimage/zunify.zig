//! Build the standard Z-Image transformer sequence.
//!
//! Basic generation uses image tokens followed by caption tokens. The image
//! span is returned so the caller can unpatch only that part after the stack.

const std = @import("std");

const zseq = @import("zseq.zig");

pub const Span = struct {
    image_start: usize,
    image_len: usize,
    total: usize,
};

pub fn basic(
    out: []f32,
    pos_out: []zseq.Pos,
    image: []const f32,
    image_pos: []const zseq.Pos,
    cap: []const f32,
    cap_pos: []const zseq.Pos,
    hidden: usize,
) !Span {
    if (hidden == 0) return error.InvalidShape;
    if (image_pos.len * hidden != image.len) return error.InvalidShape;
    if (cap_pos.len * hidden != cap.len) return error.InvalidShape;
    const total = image_pos.len + cap_pos.len;
    if (out.len != total * hidden or pos_out.len != total) return error.InvalidShape;

    copy(out[0..image.len], image);
    copy(out[image.len..], cap);
    copyPos(pos_out[0..image_pos.len], image_pos);
    copyPos(pos_out[image_pos.len..], cap_pos);

    return .{ .image_start = 0, .image_len = image_pos.len, .total = total };
}

fn copy(out: []f32, input: []const f32) void {
    for (out, input) |*dst, value| dst.* = value;
}

fn copyPos(out: []zseq.Pos, input: []const zseq.Pos) void {
    for (out, input) |*dst, value| dst.* = value;
}

test "basic sequence places image before caption" {
    var out = [_]f32{0.0} ** 6;
    var pos = [_]zseq.Pos{.{ 0, 0, 0 }} ** 3;

    const span = try basic(
        &out,
        &pos,
        &.{ 1.0, 2.0, 3.0, 4.0 },
        &.{ .{ 10, 0, 0 }, .{ 11, 0, 0 } },
        &.{ 5.0, 6.0 },
        &.{.{ 1, 0, 0 }},
        2,
    );

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &out);
    try std.testing.expectEqual(zseq.Pos{ 10, 0, 0 }, pos[0]);
    try std.testing.expectEqual(zseq.Pos{ 1, 0, 0 }, pos[2]);
    try std.testing.expectEqual(@as(usize, 2), span.image_len);
    try std.testing.expectEqual(@as(usize, 3), span.total);
}
