//! 6-bit grouped weight quantization (the 16 GB-tier format).
//!
//! Layout per matrix: packed codes (4 signed 6-bit codes in 3 bytes,
//! row-major, each row padded to a whole group) followed by one f16 scale
//! per (row, group). group must be a multiple of 4.

const std = @import("std");

const tensor = @import("tensor.zig");

pub const W6 = struct {
    bytes: []u8,
    rows: usize,
    cols: usize,
    group: usize,

    pub fn deinit(self: *W6, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn groupsPerRow(self: W6) usize {
        return (self.cols + self.group - 1) / self.group;
    }

    fn codeBytesPerRow(self: W6) usize {
        return codesPerRow(self.cols, self.group);
    }

    pub fn scale(self: W6, row: usize, group_index: usize) f32 {
        const index = row * self.groupsPerRow() + group_index;
        const at = self.rows * self.codeBytesPerRow() + index * 2;
        var h: f16 = 0;
        @memcpy(std.mem.asBytes(&h), self.bytes[at..][0..2]);
        return @floatCast(h);
    }

    pub fn decode(self: W6, row: usize, col: usize) f32 {
        const quad = col / 4;
        const at = row * self.codeBytesPerRow() + quad * 3;
        const lane = col % 4;
        const word: u32 = @as(u32, self.bytes[at]) |
            (@as(u32, self.bytes[at + 1]) << 8) |
            (@as(u32, self.bytes[at + 2]) << 16);
        const raw: u32 = (word >> @intCast(lane * 6)) & 0x3F;
        const code: i32 = if (raw < 32) @intCast(raw) else @as(i32, @intCast(raw)) - 64;
        const s = self.scale(row, col / self.group);
        return @as(f32, @floatFromInt(code)) * s;
    }
};

/// Packed code bytes per row (cols padded to whole groups; 4 codes in 3 bytes).
/// Shared layout truth for packers and GEMM dispatch offset math.
pub fn codesPerRow(cols: usize, group: usize) usize {
    return (cols + group - 1) / group * group / 4 * 3;
}

/// Byte offset of the f16 scales region (it sits after ALL rows' codes).
pub fn scalesBase(rows: usize, cols: usize, group: usize) usize {
    return rows * codesPerRow(cols, group);
}

pub fn byteLen(rows: usize, cols: usize, group: usize) usize {
    const groups = (cols + group - 1) / group;
    return scalesBase(rows, cols, group) + rows * groups * 2;
}

pub fn packW6(allocator: std.mem.Allocator, view: tensor.View, group: usize) !W6 {
    try view.check();
    if (view.shape.len != 2 or group == 0 or group % 4 != 0) return error.InvalidShape;
    const rows = view.shape[0];
    const cols = view.shape[1];
    const bytes = try allocator.alloc(u8, byteLen(rows, cols, group));
    @memset(bytes, 0);
    var out = W6{ .bytes = bytes, .rows = rows, .cols = cols, .group = group };
    for (0..rows) |row| {
        for (0..out.groupsPerRow()) |g| packGroup(&out, view, row, g);
    }
    return out;
}

fn packGroup(out: *W6, view: tensor.View, row: usize, group_index: usize) void {
    const from = group_index * out.group;
    const to = @min(from + out.group, out.cols);
    var max_abs: f32 = 0.0;
    for (from..to) |col| {
        max_abs = @max(max_abs, @abs(view.atF32Unchecked(row * out.cols + col)));
    }
    const s: f32 = if (max_abs == 0.0) 0.0 else max_abs / 31.0;
    writeScale(out, row, group_index, s);
    for (from..to) |col| {
        const value = view.atF32Unchecked(row * out.cols + col);
        writeCode(out, row, col, encode(value, s));
    }
}

fn writeScale(out: *W6, row: usize, group_index: usize, value: f32) void {
    const index = row * out.groupsPerRow() + group_index;
    const at = out.rows * out.codeBytesPerRow() + index * 2;
    const h: f16 = @floatCast(value);
    @memcpy(out.bytes[at..][0..2], std.mem.asBytes(&h));
}

fn writeCode(out: *W6, row: usize, col: usize, code: u6) void {
    const quad = col / 4;
    const at = row * out.codeBytesPerRow() + quad * 3;
    const lane: u5 = @intCast((col % 4) * 6);
    var word: u32 = @as(u32, out.bytes[at]) |
        (@as(u32, out.bytes[at + 1]) << 8) |
        (@as(u32, out.bytes[at + 2]) << 16);
    word &= ~(@as(u32, 0x3F) << lane);
    word |= @as(u32, code) << lane;
    out.bytes[at] = @intCast(word & 0xFF);
    out.bytes[at + 1] = @intCast((word >> 8) & 0xFF);
    out.bytes[at + 2] = @intCast((word >> 16) & 0xFF);
}

fn encode(value: f32, s: f32) u6 {
    if (s == 0.0) return 0;
    const qf = @round(value / s);
    const qi: i32 = @intFromFloat(@min(@max(qf, -31.0), 31.0));
    return @intCast(if (qi < 0) qi + 64 else qi);
}

test "w6 roundtrip stays within one scale step" {
    const data = [_]f32{ 0.5, -0.25, 0.125, -1.0, 0.75, 0.3, -0.6, 0.9 };
    const view = tensor.View{
        .bytes = std.mem.sliceAsBytes(&data),
        .shape = &.{ 2, 4 },
        .dtype = .f32,
    };
    var packed_w = try packW6(std.testing.allocator, view, 4);
    defer packed_w.deinit(std.testing.allocator);
    for (0..2) |r| for (0..4) |c| {
        const want = data[r * 4 + c];
        const got = packed_w.decode(r, c);
        const tol = packed_w.scale(r, c / 4) * 0.51 + 1e-6;
        try std.testing.expect(@abs(want - got) <= tol);
    };
}
