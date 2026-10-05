//! FLUX.2 Klein latent packing.
//!
//! The VAE emits 32-channel CHW latents. The transformer sees 2x2 spatial
//! patches packed into 128-channel sequence tokens. These helpers are pure
//! layout transforms and are intentionally allocation-free.

const std = @import("std");

pub const channels: usize = 32;
pub const packed_channels: usize = channels * 4;

pub const Shape = struct {
    height: usize,
    width: usize,

    pub fn tokens(self: Shape) !usize {
        try self.check();
        return (self.height / 2) * (self.width / 2);
    }

    fn check(self: Shape) !void {
        if (self.height == 0 or self.width == 0) return error.InvalidShape;
        if (self.height % 2 != 0 or self.width % 2 != 0) return error.InvalidShape;
    }
};

pub const Error = error{
    InvalidShape,
    InvalidBufferLength,
};

pub fn packedLen(shape: Shape) !usize {
    return try std.math.mul(usize, try shape.tokens(), packed_channels);
}

pub fn unpackedLen(shape: Shape) !usize {
    try shape.check();
    return try std.math.mul(usize, channels, shape.height * shape.width);
}

pub fn pack(out: []f32, input: []const f32, shape: Shape) !void {
    const need_out = try packedLen(shape);
    const need_in = try unpackedLen(shape);
    if (out.len != need_out or input.len != need_in) return error.InvalidBufferLength;

    const h2 = shape.height / 2;
    const w2 = shape.width / 2;
    for (0..h2) |oh| {
        for (0..w2) |ow| {
            const token = oh * w2 + ow;
            for (0..channels) |c| {
                inline for (0..2) |dy| {
                    inline for (0..2) |dx| {
                        out[token * packed_channels + packChan(c, dy, dx)] =
                            input[chw(c, oh * 2 + dy, ow * 2 + dx, shape)];
                    }
                }
            }
        }
    }
}

pub fn unpack(out: []f32, input: []const f32, shape: Shape) !void {
    const need_out = try unpackedLen(shape);
    const need_in = try packedLen(shape);
    if (out.len != need_out or input.len != need_in) return error.InvalidBufferLength;

    const h2 = shape.height / 2;
    const w2 = shape.width / 2;
    for (0..h2) |oh| {
        for (0..w2) |ow| {
            const token = oh * w2 + ow;
            for (0..channels) |c| {
                inline for (0..2) |dy| {
                    inline for (0..2) |dx| {
                        out[chw(c, oh * 2 + dy, ow * 2 + dx, shape)] =
                            input[token * packed_channels + packChan(c, dy, dx)];
                    }
                }
            }
        }
    }
}

fn packChan(c: usize, dy: usize, dx: usize) usize {
    return c * 4 + dy * 2 + dx;
}

fn chw(c: usize, y: usize, x: usize, shape: Shape) usize {
    return (c * shape.height + y) * shape.width + x;
}

test "1024px latent shape packs to the anchor token geometry" {
    const shape = Shape{ .height = 128, .width = 128 };
    try std.testing.expectEqual(@as(usize, 4096), try shape.tokens());
    try std.testing.expectEqual(@as(usize, 4096 * 128), try packedLen(shape));
}

test "pack follows the documented 2x2 permutation" {
    const shape = Shape{ .height = 2, .width = 2 };
    var input = [_]f32{0.0} ** (channels * 4);
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i);
    var seq = [_]f32{0.0} ** packed_channels;

    try pack(&seq, &input, shape);
    try std.testing.expectApproxEqAbs(input[0], seq[0], 0.0);
    try std.testing.expectApproxEqAbs(input[1], seq[1], 0.0);
    try std.testing.expectApproxEqAbs(input[2], seq[2], 0.0);
    try std.testing.expectApproxEqAbs(input[3], seq[3], 0.0);
    try std.testing.expectApproxEqAbs(input[4], seq[4], 0.0);
}

test "pack and unpack round-trip" {
    const shape = Shape{ .height = 4, .width = 6 };
    var input = [_]f32{0.0} ** (channels * 4 * 6);
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i);
    var seq = [_]f32{0.0} ** ((4 / 2) * (6 / 2) * packed_channels);
    var output = [_]f32{0.0} ** input.len;

    try pack(&seq, &input, shape);
    try unpack(&output, &seq, shape);
    for (input, output) |a, b| try std.testing.expectApproxEqAbs(a, b, 0.0);
}

test "invalid shape and buffer sizes are rejected" {
    try std.testing.expectError(error.InvalidShape, packedLen(.{ .height = 3, .width = 4 }));
    var bad = [_]f32{0.0} ** 4;
    try std.testing.expectError(
        error.InvalidBufferLength,
        pack(&bad, &bad, .{ .height = 2, .width = 2 }),
    );
}
