//! 4-bit grouped weight quantization (the W4 tier).
//!
//! Layout per matrix, mirroring zw6.zig: packed codes (2 signed 4-bit codes
//! per byte, low nibble first, row-major, each row padded to a whole group)
//! followed by one f16 scale per (row, group). group must be a multiple of 2.
//! The scale is absmax / 7 and codes are clamped to [-7, 7]: the same
//! symmetric round-to-nearest rule as W6, so a W4 pack is the W6 pack's
//! format at a different width, and the resident loader decodes it with the
//! same per-element operation (int code -> half -> * half scale).

const std = @import("std");

const tensor = @import("tensor.zig");

pub const W4 = struct {
    bytes: []u8,
    rows: usize,
    cols: usize,
    group: usize,

    pub fn deinit(self: *W4, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn groupsPerRow(self: W4) usize {
        return (self.cols + self.group - 1) / self.group;
    }

    fn codeBytesPerRow(self: W4) usize {
        return codesPerRow(self.cols, self.group);
    }

    pub fn scale(self: W4, row: usize, group_index: usize) f32 {
        const index = row * self.groupsPerRow() + group_index;
        const at = self.rows * self.codeBytesPerRow() + index * 2;
        var h: f16 = 0;
        @memcpy(std.mem.asBytes(&h), self.bytes[at..][0..2]);
        return @floatCast(h);
    }

    pub fn decode(self: W4, row: usize, col: usize) f32 {
        const at = row * self.codeBytesPerRow() + col / 2;
        const shift: u3 = @intCast((col % 2) * 4);
        const raw: u32 = (@as(u32, self.bytes[at]) >> shift) & 0x0F;
        const code: i32 = if (raw < 8) @intCast(raw) else @as(i32, @intCast(raw)) - 16;
        const s = self.scale(row, col / self.group);
        return @as(f32, @floatFromInt(code)) * s;
    }
};

/// Codec facade for callers generic over the packed widths (gemmbench).
pub const Packed = W4;
pub const pack = packW4;

/// Packed code bytes per row (cols padded to whole groups; 2 codes per byte).
pub fn codesPerRow(cols: usize, group: usize) usize {
    return (cols + group - 1) / group * group / 2;
}

/// Byte offset of the f16 scales region (it sits after ALL rows' codes).
pub fn scalesBase(rows: usize, cols: usize, group: usize) usize {
    return rows * codesPerRow(cols, group);
}

pub fn byteLen(rows: usize, cols: usize, group: usize) usize {
    const groups = (cols + group - 1) / group;
    return scalesBase(rows, cols, group) + rows * groups * 2;
}

pub fn packW4(allocator: std.mem.Allocator, view: tensor.View, group: usize) !W4 {
    return packW4Q(allocator, view, group, 7.0);
}

/// packW4 with the scale rule absmax / qmax. qmax 7 is the 4-bit grid; qmax 3
/// stores a trained 3-bit grid (codes -3..3) exactly, instead of rounding it
/// again onto the 4-bit grid.
pub fn packW4Q(allocator: std.mem.Allocator, view: tensor.View, group: usize, qmax: f32) !W4 {
    try view.check();
    if (view.shape.len != 2 or group == 0 or group % 2 != 0) return error.InvalidShape;
    const rows = view.shape[0];
    const cols = view.shape[1];
    const bytes = try allocator.alloc(u8, byteLen(rows, cols, group));
    @memset(bytes, 0);
    var out = W4{ .bytes = bytes, .rows = rows, .cols = cols, .group = group };
    for (0..rows) |row| {
        for (0..out.groupsPerRow()) |g| packGroup(&out, view, row, g, qmax);
    }
    return out;
}

fn packGroup(out: *W4, view: tensor.View, row: usize, group_index: usize, qmax: f32) void {
    const from = group_index * out.group;
    const to = @min(from + out.group, out.cols);
    var max_abs: f32 = 0.0;
    for (from..to) |col| {
        max_abs = @max(max_abs, @abs(view.atF32Unchecked(row * out.cols + col)));
    }
    const s: f32 = if (max_abs == 0.0) 0.0 else max_abs / qmax;
    writeScale(out, row, group_index, s);
    for (from..to) |col| {
        const value = view.atF32Unchecked(row * out.cols + col);
        writeCode(out, row, col, encode(value, s));
    }
}

fn writeScale(out: *W4, row: usize, group_index: usize, value: f32) void {
    const index = row * out.groupsPerRow() + group_index;
    const at = out.rows * out.codeBytesPerRow() + index * 2;
    const h: f16 = @floatCast(value);
    @memcpy(out.bytes[at..][0..2], std.mem.asBytes(&h));
}

fn writeCode(out: *W4, row: usize, col: usize, code: u4) void {
    const at = row * out.codeBytesPerRow() + col / 2;
    const shift: u3 = @intCast((col % 2) * 4);
    var byte: u8 = out.bytes[at];
    byte &= ~(@as(u8, 0x0F) << shift);
    byte |= @as(u8, code) << shift;
    out.bytes[at] = byte;
}

fn encode(value: f32, s: f32) u4 {
    if (s == 0.0) return 0;
    const qf = @round(value / s);
    const qi: i32 = @intFromFloat(@min(@max(qf, -7.0), 7.0));
    return @intCast(if (qi < 0) qi + 16 else qi);
}

test "w4 roundtrip stays within one scale step" {
    const data = [_]f32{ 0.5, -0.25, 0.125, -1.0, 0.75, 0.3, -0.6, 0.9 };
    const view = tensor.View{
        .bytes = std.mem.sliceAsBytes(&data),
        .shape = &.{ 2, 4 },
        .dtype = .f32,
    };
    var packed_w = try packW4(std.testing.allocator, view, 4);
    defer packed_w.deinit(std.testing.allocator);
    for (0..2) |r| for (0..4) |c| {
        const want = data[r * 4 + c];
        const got = packed_w.decode(r, c);
        const tol = packed_w.scale(r, c / 4) * 0.51 + 1e-6;
        try std.testing.expect(@abs(want - got) <= tol);
    };
}

test "w4 at qmax 3 stores a 3-bit grid exactly" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0 };
    const view = tensor.View{
        .bytes = std.mem.sliceAsBytes(&data),
        .shape = &.{ 1, 8 },
        .dtype = .f32,
    };
    var exact = try packW4Q(allocator, view, 8, 3.0);
    defer exact.deinit(allocator);
    var again = try packW4Q(allocator, view, 8, 7.0);
    defer again.deinit(allocator);
    var max_err3: f32 = 0;
    var max_err7: f32 = 0;
    for (0..8) |col| {
        max_err3 = @max(max_err3, @abs(exact.decode(0, col) - data[col]));
        max_err7 = @max(max_err7, @abs(again.decode(0, col) - data[col]));
    }
    try std.testing.expect(max_err3 < 1e-4);
    try std.testing.expect(max_err7 > 5e-3);
}

test "w4 codes pack two per byte and sign-extend" {
    const data = [_]f32{ 7.0, -7.0, 0.0, 1.0 };
    const view = tensor.View{
        .bytes = std.mem.sliceAsBytes(&data),
        .shape = &.{ 1, 4 },
        .dtype = .f32,
    };
    var packed_w = try packW4(std.testing.allocator, view, 4);
    defer packed_w.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2 + 2), packed_w.bytes.len);
    try std.testing.expectEqual(@as(f32, 7.0), packed_w.decode(0, 0));
    try std.testing.expectEqual(@as(f32, -7.0), packed_w.decode(0, 1));
    try std.testing.expectEqual(@as(f32, 0.0), packed_w.decode(0, 2));
    try std.testing.expectEqual(@as(f32, 1.0), packed_w.decode(0, 3));
}
