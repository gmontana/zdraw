//! Sequence padding and position IDs for Z-Image.
//!
//! Features are padded to a multiple of 32 by repeating the last token. The
//! caller can later replace masked tokens with the learned pad token.

const std = @import("std");

pub const Pos = [3]usize;
pub const multi: usize = 32;

pub const Error = error{
    InvalidShape,
};

pub fn padLen(len: usize) usize {
    return (multi - len % multi) % multi;
}

pub fn paddedLen(len: usize) usize {
    return len + padLen(len);
}

pub fn grid(out: []Pos, size: Pos, start: Pos) !void {
    if (out.len != gridLen(size)) return error.InvalidShape;
    var pos: usize = 0;
    for (0..size[0]) |f| {
        for (0..size[1]) |h| {
            for (0..size[2]) |w| {
                out[pos] = .{ start[0] + f, start[1] + h, start[2] + w };
                pos += 1;
            }
        }
    }
}

pub fn padWithIds(
    out: []f32,
    pos_ids: []Pos,
    mask: []bool,
    feat: []const f32,
    dim: usize,
    size: Pos,
    start: Pos,
) !usize {
    if (dim == 0) return error.InvalidShape;
    const span = gridLen(size);
    const total = paddedLen(span);
    if (span == 0 or feat.len % dim != 0) return error.InvalidShape;
    const real = feat.len / dim;
    if (real == 0 or real > total) return error.InvalidShape;
    if (out.len != total * dim or pos_ids.len != total or mask.len != total) {
        return error.InvalidShape;
    }

    // Positions: the coordinate grid covers its full span; any slots past it
    // collapse to the origin. The image grid spans only its real patches (so its
    // pad tokens sit at the origin), while the caption grid spans the padded
    // length (so its pad tokens keep advancing the frame axis), matching the
    // reference position assignment.
    try grid(pos_ids[0..span], size, start);
    for (pos_ids[span..total]) |*value| value.* = .{ 0, 0, 0 };

    // Features: copy the real tokens, repeat the last token into the pad slots,
    // and mark them so the caller can swap in the learned pad token.
    copyFeat(out[0 .. real * dim], feat);
    const last = feat[(real - 1) * dim ..][0..dim];
    for (real..total) |tok| copyFeat(out[tok * dim ..][0..dim], last);
    for (mask[0..real]) |*value| value.* = false;
    for (mask[real..total]) |*value| value.* = true;
    return total;
}

pub fn applyPad(
    feat: []f32,
    mask: []const bool,
    pad: []const f32,
    dim: usize,
) !void {
    if (dim == 0 or pad.len != dim) return error.InvalidShape;
    if (feat.len != mask.len * dim) return error.InvalidShape;
    for (mask, 0..) |masked, tok| {
        if (masked) copyFeat(feat[tok * dim ..][0..dim], pad);
    }
}

fn gridLen(size: Pos) usize {
    return size[0] * size[1] * size[2];
}

fn copyFeat(out: []f32, input: []const f32) void {
    for (out, input) |*dst, value| dst.* = value;
}

test "pad length rounds to sequence multiple" {
    try std.testing.expectEqual(@as(usize, 0), padLen(32));
    try std.testing.expectEqual(@as(usize, 30), padLen(2));
    try std.testing.expectEqual(@as(usize, 64), paddedLen(33));
}

test "grid follows flattened frame row column order" {
    var ids = [_]Pos{.{ 0, 0, 0 }} ** 4;
    try grid(&ids, .{ 1, 2, 2 }, .{ 5, 0, 0 });

    try std.testing.expectEqualDeep([_]Pos{
        .{ 5, 0, 0 },
        .{ 5, 0, 1 },
        .{ 5, 1, 0 },
        .{ 5, 1, 1 },
    }, ids);
}

test "pad with ids repeats last token and marks padding" {
    var out = [_]f32{0.0} ** 64;
    var ids = [_]Pos{.{ 9, 9, 9 }} ** 32;
    var mask = [_]bool{false} ** 32;
    const feat = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const total = try padWithIds(&out, &ids, &mask, &feat, 2, .{ 1, 1, 2 }, .{ 7, 0, 0 });
    try std.testing.expectEqual(@as(usize, 32), total);
    try std.testing.expectEqualSlices(f32, &feat, out[0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, out[4..6]);
    try std.testing.expectEqual(Pos{ 7, 0, 1 }, ids[1]);
    try std.testing.expectEqual(Pos{ 0, 0, 0 }, ids[2]);
    try std.testing.expect(!mask[1]);
    try std.testing.expect(mask[2]);
}

test "apply pad token replaces masked features" {
    var feat = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try applyPad(&feat, &.{ false, true, false }, &.{ 9.0, 8.0 }, 2);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 9, 8, 5, 6 }, &feat);
}
