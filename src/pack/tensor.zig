//! Tiny tensor views over already-owned bytes.

const std = @import("std");

pub const DType = enum {
    f32,
    f16,
    bf16,
    u8,
};

pub const Layout = enum {
    flat,
};

pub const View = struct {
    dtype: DType,
    shape: []const usize,
    bytes: []const u8,
    layout: Layout = .flat,
    source: ?Source = null,
    /// Packed low-bit weights (the `.u8` marker views from a sidecar): the
    /// code width, 6 or 4, so the one consumer that binds them picks the
    /// matching codec and kernel. 0 for every ordinary view.
    packed_bits: u8 = 0,

    pub fn elems(self: View) !usize {
        var total: usize = 1;
        for (self.shape) |dim| total = try std.math.mul(usize, total, dim);
        return total;
    }

    pub fn byteLen(self: View) !usize {
        return try std.math.mul(usize, try self.elems(), byteSize(self.dtype));
    }

    pub fn check(self: View) !void {
        if (self.bytes.len != try self.byteLen()) return error.InvalidTensorBytes;
    }

    pub fn atF32(self: View, index: usize) !f32 {
        try self.check();
        const count = try self.elems();
        if (index >= count) return error.IndexOutOfBounds;
        return self.atF32Unchecked(index);
    }

    pub fn atF32Unchecked(self: View, index: usize) f32 {
        const size = byteSize(self.dtype);
        const start = index * size;
        const bytes = self.bytes[start..][0..size];
        return readF32(self.dtype, bytes);
    }

    pub fn copyF32(self: View, out: []f32) !void {
        try self.check();
        if (out.len != try self.elems()) return error.InvalidTensorBytes;
        const size = byteSize(self.dtype);
        for (out, 0..) |*value, i| {
            const start = i * size;
            value.* = readF32(self.dtype, self.bytes[start..][0..size]);
        }
    }
};

/// Promote a bf16 view to a freshly-allocated f32 slice. The single shared
/// conversion for model-load paths; asserts the dtype so format bugs fail
/// loudly instead of reinterpreting bytes.
pub fn promoteBf16(allocator: std.mem.Allocator, v: View) ![]f32 {
    if (v.dtype != .bf16) return error.UnsupportedDType;
    const raw = std.mem.bytesAsSlice(u16, v.bytes);
    const out = try allocator.alloc(f32, raw.len);
    for (out, raw) |*dst, bits16| {
        const bits: u32 = @as(u32, bits16) << 16;
        dst.* = @bitCast(bits);
    }
    return out;
}

pub const Source = struct {
    bytes: []const u8,
    offset: usize,
};

pub const Error = error{
    InvalidDType,
    InvalidTensorBytes,
    IndexOutOfBounds,
};

pub fn parseDType(name: []const u8) !DType {
    if (std.mem.eql(u8, name, "F32")) return .f32;
    if (std.mem.eql(u8, name, "F16")) return .f16;
    if (std.mem.eql(u8, name, "BF16")) return .bf16;
    if (std.mem.eql(u8, name, "U8")) return .u8;
    return error.InvalidDType;
}

pub fn byteSize(dtype: DType) usize {
    return switch (dtype) {
        .f32 => 4,
        .f16 => 2,
        .bf16 => 2,
        .u8 => 1,
    };
}

fn readF32(dtype: DType, bytes: []const u8) f32 {
    return switch (dtype) {
        .f32 => bitsToF32(std.mem.readInt(u32, bytes[0..4], .little)),
        .f16 => fromBits(@intCast(std.mem.readInt(u16, bytes[0..2], .little)), 5, 10, 15),
        .bf16 => bitsToF32(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16),
        .u8 => @floatFromInt(bytes[0]),
    };
}

fn bitsToF32(bits: u32) f32 {
    return std.mem.bytesToValue(f32, std.mem.asBytes(&bits));
}

fn fromBits(bits: u32, exp_bits: u5, mant_bits: u5, bias: i32) f32 {
    const one: u32 = 1;
    const sign_mask = one << (exp_bits + mant_bits);
    const exp_mask = (one << exp_bits) - 1;
    const mant_mask = (one << mant_bits) - 1;
    const exp_raw = (bits >> mant_bits) & exp_mask;
    const mant = bits & mant_mask;
    const sign: f32 = if ((bits & sign_mask) == 0) 1.0 else -1.0;

    if (exp_raw == exp_mask) {
        if (mant != 0) return std.math.nan(f32);
        return sign * std.math.inf(f32);
    }
    if (exp_raw == 0) {
        if (mant == 0) return sign * 0.0;
        return sign * std.math.ldexp(frac(mant, mant_bits), 1 - bias);
    }

    const exp: i32 = @intCast(exp_raw);
    return sign * std.math.ldexp(1.0 + frac(mant, mant_bits), exp - bias);
}

fn frac(mant: u32, mant_bits: u5) f32 {
    const top: f32 = @floatFromInt(mant);
    const bottom: f32 = @floatFromInt(@as(u32, 1) << mant_bits);
    return top / bottom;
}

test "validate tensor bytes" {
    const shape = [_]usize{ 2, 3 };
    const bytes = [_]u8{0} ** 12;
    const view = View{ .dtype = .bf16, .shape = &shape, .bytes = &bytes };

    const elem_count: usize = 6;
    const byte_count: usize = 12;

    try std.testing.expectEqual(elem_count, try view.elems());
    try std.testing.expectEqual(byte_count, try view.byteLen());
    try view.check();
}

test "reject wrong byte length" {
    const shape = [_]usize{4};
    const view = View{ .dtype = .f32, .shape = &shape, .bytes = &.{ 0, 1 } };
    try std.testing.expectError(error.InvalidTensorBytes, view.check());
}

test "read scalar values as f32" {
    const shape = [_]usize{1};
    const bytes = [_]u8{
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x3c, 0xc0, 0x3f,
        7,
    };
    const f32_view = View{ .dtype = .f32, .shape = shape[0..1], .bytes = bytes[0..4] };
    const f16_view = View{ .dtype = .f16, .shape = shape[0..1], .bytes = bytes[4..6] };
    const bf16_view = View{ .dtype = .bf16, .shape = shape[0..1], .bytes = bytes[6..8] };
    const u8_view = View{ .dtype = .u8, .shape = shape[0..1], .bytes = bytes[8..9] };

    try std.testing.expectApproxEqAbs(1.0, try f32_view.atF32(0), 0.0001);
    try std.testing.expectApproxEqAbs(1.0, try f16_view.atF32(0), 0.0001);
    try std.testing.expectApproxEqAbs(1.5, try bf16_view.atF32(0), 0.0001);
    try std.testing.expectApproxEqAbs(7.0, try u8_view.atF32(0), 0.0001);
}
