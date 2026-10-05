//! Packed weight formats for zdraw runtime profiles.

const std = @import("std");

const tensor = @import("tensor.zig");

pub const W8 = struct {
    bytes: []u8,
    rows: usize,
    cols: usize,
    group: usize,

    pub fn deinit(self: *W8, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn qBytes(self: W8) usize {
        return self.rows * self.cols;
    }

    pub fn groupsPerRow(self: W8) usize {
        return ceilDiv(self.cols, self.group);
    }

    pub fn scale(self: W8, row: usize, group: usize) f32 {
        const index = row * self.groupsPerRow() + group;
        const at = self.qBytes() + index * @sizeOf(f32);
        return std.mem.bytesToValue(f32, self.bytes[at..][0..4]);
    }

    pub fn q(self: W8, row: usize, col: usize) i16 {
        return signed(self.bytes[row * self.cols + col]);
    }
};

pub fn packW8(allocator: std.mem.Allocator, view: tensor.View, group: usize) !W8 {
    try view.check();
    if (view.shape.len != 2 or group == 0) return error.InvalidShape;
    const rows = view.shape[0];
    const cols = view.shape[1];
    const bytes = try allocator.alloc(u8, byteLen(rows, cols, group));
    var out = W8{ .bytes = bytes, .rows = rows, .cols = cols, .group = group };
    for (0..rows) |row| {
        for (0..out.groupsPerRow()) |g| packGroup(&out, view, row, g);
    }
    return out;
}

pub fn byteLen(rows: usize, cols: usize, group: usize) usize {
    return rows * cols + rows * ceilDiv(cols, group) * @sizeOf(f32);
}

fn packGroup(out: *W8, view: tensor.View, row: usize, group_index: usize) void {
    const from = group_index * out.group;
    const to = @min(from + out.group, out.cols);
    const s = scaleFor(view, row, from, to);
    writeScale(out, row, group_index, s);
    for (from..to) |col| {
        const value = view.atF32Unchecked(row * out.cols + col);
        out.bytes[row * out.cols + col] = encode(value, s);
    }
}

fn scaleFor(view: tensor.View, row: usize, from: usize, to: usize) f32 {
    var max_abs: f32 = 0.0;
    for (from..to) |col| {
        max_abs = @max(max_abs, abs(view.atF32Unchecked(row * view.shape[1] + col)));
    }
    if (max_abs == 0.0) return 0.0;
    return max_abs / 127.0;
}

fn writeScale(out: *W8, row: usize, group_index: usize, value: f32) void {
    const index = row * out.groupsPerRow() + group_index;
    const at = out.qBytes() + index * @sizeOf(f32);
    @memcpy(out.bytes[at..][0..4], std.mem.asBytes(&value));
}

fn encode(value: f32, scale: f32) u8 {
    if (scale == 0.0) return 0;
    const qf = @round(value / scale);
    const qi: i32 = @intFromFloat(@min(@max(qf, -127.0), 127.0));
    return @intCast(if (qi < 0) qi + 256 else qi);
}

fn signed(byte: u8) i16 {
    return if (byte < 128) @intCast(byte) else @as(i16, byte) - 256;
}

// W16: a plain little-endian f16 image of the weight matrix. Sidecar entries
// carry it with group == 0 to distinguish from grouped W8 payloads.
pub const W16 = struct {
    bytes: []u8,
    rows: usize,
    cols: usize,

    pub fn deinit(self: *W16, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }
};

pub fn packW16(allocator: std.mem.Allocator, view: tensor.View) !W16 {
    try view.check();
    if (view.shape.len != 2) return error.InvalidShape;
    const rows = view.shape[0];
    const cols = view.shape[1];
    const bytes = try allocator.alloc(u8, rows * cols * 2);
    writeF16(bytes, view, 0);
    return .{ .bytes = bytes, .rows = rows, .cols = cols };
}

// Fused [gate; up] image: gate rows then up rows, one weight matrix that a
// single GEMM can multiply (n doubles, k unchanged).
pub fn packW16Pair(allocator: std.mem.Allocator, gate: tensor.View, up: tensor.View) !W16 {
    try gate.check();
    try up.check();
    if (gate.shape.len != 2 or up.shape.len != 2) return error.InvalidShape;
    if (gate.shape[0] != up.shape[0] or gate.shape[1] != up.shape[1]) return error.InvalidShape;
    const rows = gate.shape[0];
    const cols = gate.shape[1];
    const bytes = try allocator.alloc(u8, rows * cols * 4);
    writeF16(bytes, gate, 0);
    writeF16(bytes, up, rows * cols * 2);
    return .{ .bytes = bytes, .rows = rows * 2, .cols = cols };
}

fn writeF16(bytes: []u8, view: tensor.View, base: usize) void {
    const count = view.shape[0] * view.shape[1];
    for (0..count) |i| {
        const f = view.atF32Unchecked(i);
        const h: f16 = @floatCast(std.math.clamp(f, -65504.0, 65504.0));
        @memcpy(bytes[base + i * 2 ..][0..2], std.mem.asBytes(&h));
    }
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn abs(value: f32) f32 {
    return if (value < 0.0) -value else value;
}

test "pack W8 tensor layout" {
    const values = [_]f32{
        0.0,  1.0,   -1.0, 0.5,
        0.25, -0.25, 2.0,  -2.0,
    };
    const shape = [_]usize{ 2, 4 };
    const view = tensor.View{
        .dtype = .f32,
        .shape = &shape,
        .bytes = std.mem.sliceAsBytes(&values),
    };
    var w8 = try packW8(std.testing.allocator, view, 2);
    defer w8.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 24), w8.bytes.len);
    try std.testing.expectEqual(@as(i16, 0), w8.q(0, 0));
    try std.testing.expectEqual(@as(i16, 127), w8.q(0, 1));
    try std.testing.expectEqual(@as(i16, -127), w8.q(0, 2));
    try std.testing.expectEqual(@as(i16, 64), w8.q(0, 3));
    try std.testing.expectApproxEqAbs(1.0 / 127.0, w8.scale(0, 0), 0.000001);
}

test "pack W16 truncates to f16 bytes" {
    const values = [_]f32{ 0.0, 1.0, -1.5, 0.333984375 };
    const shape = [_]usize{ 2, 2 };
    const view = tensor.View{
        .dtype = .f32,
        .shape = &shape,
        .bytes = std.mem.sliceAsBytes(&values),
    };
    var w16 = try packW16(std.testing.allocator, view);
    defer w16.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 8), w16.bytes.len);
    const half = std.mem.bytesAsSlice(f16, w16.bytes);
    try std.testing.expectEqual(@as(f16, 1.0), half[1]);
    try std.testing.expectEqual(@as(f16, -1.5), half[2]);
}

test "pack W8 rejects non-matrix tensors" {
    const shape = [_]usize{4};
    const values = [_]f32{ 0.0, 1.0, -1.0, 0.5 };
    const view = tensor.View{
        .dtype = .f32,
        .shape = &shape,
        .bytes = std.mem.sliceAsBytes(&values),
    };
    try std.testing.expectError(error.InvalidShape, packW8(std.testing.allocator, view, 2));
}
