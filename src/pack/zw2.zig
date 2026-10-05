//! 2-bit grouped weight quantization (the W2 tier, ternary in practice).
//!
//! Layout per matrix, mirroring zw4.zig: packed codes (4 signed 2-bit codes
//! per byte, lowest pair first, row-major, each row padded to a whole group)
//! followed by one f16 scale per (row, group). group must be a multiple of 4.
//! The scale is absmax / 1 and codes are clamped to [-1, 1]: the packer's
//! symmetric rule at qmax 1, so a trained ternary grid (values in {-s, 0, s})
//! round-trips exactly, and the resident loader decodes it with the same
//! per-element operation (int code -> half -> * half scale).

const std = @import("std");

const tensor = @import("tensor.zig");

pub const W2 = struct {
    bytes: []u8,
    rows: usize,
    cols: usize,
    group: usize,

    pub fn deinit(self: *W2, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn groupsPerRow(self: W2) usize {
        return (self.cols + self.group - 1) / self.group;
    }

    fn codeBytesPerRow(self: W2) usize {
        return codesPerRow(self.cols, self.group);
    }

    pub fn scale(self: W2, row: usize, group_index: usize) f32 {
        const index = row * self.groupsPerRow() + group_index;
        const at = self.rows * self.codeBytesPerRow() + index * 2;
        var h: f16 = 0;
        @memcpy(std.mem.asBytes(&h), self.bytes[at..][0..2]);
        return @floatCast(h);
    }

    pub fn decode(self: W2, row: usize, col: usize) f32 {
        const at = row * self.codeBytesPerRow() + col / 4;
        const shift: u3 = @intCast((col % 4) * 2);
        const raw: u32 = (@as(u32, self.bytes[at]) >> shift) & 0x03;
        const code: i32 = if (raw < 2) @intCast(raw) else @as(i32, @intCast(raw)) - 4;
        const s = self.scale(row, col / self.group);
        return @as(f32, @floatFromInt(code)) * s;
    }
};

/// Codec facade for callers generic over the packed widths (gemmbench).
pub const Packed = W2;
pub const pack = packW2;

/// Packed code bytes per row (cols padded to whole groups; 4 codes per byte).
pub fn codesPerRow(cols: usize, group: usize) usize {
    return (cols + group - 1) / group * group / 4;
}

/// Byte offset of the f16 scales region (it sits after ALL rows' codes).
pub fn scalesBase(rows: usize, cols: usize, group: usize) usize {
    return rows * codesPerRow(cols, group);
}

pub fn byteLen(rows: usize, cols: usize, group: usize) usize {
    const groups = (cols + group - 1) / group;
    return scalesBase(rows, cols, group) + rows * groups * 2;
}

pub fn packW2(allocator: std.mem.Allocator, view: tensor.View, group: usize) !W2 {
    try view.check();
    if (view.shape.len != 2 or group == 0 or group % 4 != 0) return error.InvalidShape;
    const rows = view.shape[0];
    const cols = view.shape[1];
    const bytes = try allocator.alloc(u8, byteLen(rows, cols, group));
    @memset(bytes, 0);
    var out = W2{ .bytes = bytes, .rows = rows, .cols = cols, .group = group };
    for (0..rows) |row| {
        for (0..out.groupsPerRow()) |g| packGroup(&out, view, row, g);
    }
    return out;
}

fn packGroup(out: *W2, view: tensor.View, row: usize, group_index: usize) void {
    const from = group_index * out.group;
    const to = @min(from + out.group, out.cols);
    var max_abs: f32 = 0.0;
    for (from..to) |col| {
        max_abs = @max(max_abs, @abs(view.atF32Unchecked(row * out.cols + col)));
    }
    const s: f32 = if (max_abs == 0.0) 0.0 else max_abs;
    writeScale(out, row, group_index, s);
    for (from..to) |col| {
        const value = view.atF32Unchecked(row * out.cols + col);
        writeCode(out, row, col, encode(value, s));
    }
}

fn writeScale(out: *W2, row: usize, group_index: usize, value: f32) void {
    const index = row * out.groupsPerRow() + group_index;
    const at = out.rows * out.codeBytesPerRow() + index * 2;
    const h: f16 = @floatCast(value);
    @memcpy(out.bytes[at..][0..2], std.mem.asBytes(&h));
}

fn writeCode(out: *W2, row: usize, col: usize, code: u2) void {
    const at = row * out.codeBytesPerRow() + col / 4;
    const shift: u3 = @intCast((col % 4) * 2);
    var byte: u8 = out.bytes[at];
    byte &= ~(@as(u8, 0x03) << shift);
    byte |= @as(u8, code) << shift;
    out.bytes[at] = byte;
}

fn encode(value: f32, s: f32) u2 {
    if (s == 0.0) return 0;
    const qf = @round(value / s);
    const qi: i32 = @intFromFloat(@min(@max(qf, -1.0), 1.0));
    return @intCast(if (qi < 0) qi + 4 else qi);
}

test "w2 roundtrip stays within one scale step" {
    const data = [_]f32{ 0.5, -0.25, 0.125, -1.0, 0.75, 0.3, -0.6, 0.9 };
    const view = tensor.View{
        .bytes = std.mem.sliceAsBytes(&data),
        .shape = &.{ 2, 4 },
        .dtype = .f32,
    };
    var packed_w = try packW2(std.testing.allocator, view, 4);
    defer packed_w.deinit(std.testing.allocator);
    for (0..2) |r| for (0..4) |c| {
        const want = data[r * 4 + c];
        const got = packed_w.decode(r, c);
        const tol = packed_w.scale(r, c / 4) * 0.51 + 1e-6;
        try std.testing.expect(@abs(want - got) <= tol);
    };
}

test "w2 codes pack four per byte, sign-extend, and a ternary grid round-trips exactly" {
    const data = [_]f32{ 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5 };
    const view = tensor.View{
        .bytes = std.mem.sliceAsBytes(&data),
        .shape = &.{ 1, 8 },
        .dtype = .f32,
    };
    var packed_w = try packW2(std.testing.allocator, view, 8);
    defer packed_w.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2 + 2), packed_w.bytes.len);
    for (data, 0..) |want, c| try std.testing.expectEqual(want, packed_w.decode(0, c));
}
