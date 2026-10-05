//! AdaLN modulation helpers for Z-Image blocks.
//!
//! A block gets four vectors from one linear layer: attention scale/gate and
//! MLP scale/gate. The scales are shifted by one; the gates pass through tanh.

const std = @import("std");

const linear = @import("linear_fast.zig");
const mlinear = @import("mlinear.zig");
const tensor = @import("tensor.zig");

pub const Error = error{
    InvalidShape,
};

pub const Parts = struct {
    attn_scale: []f32,
    attn_gate: []f32,
    mlp_scale: []f32,
    mlp_gate: []f32,
};

pub fn run(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
) !Parts {
    try linear.run(metal, out, input, weight, bias);
    const parts = try split(out);
    finish(parts);
    return parts;
}

pub fn split(values: []f32) !Parts {
    if (values.len == 0 or values.len % 4 != 0) return error.InvalidShape;
    const dim = values.len / 4;
    return .{
        .attn_scale = values[0..dim],
        .attn_gate = values[dim..][0..dim],
        .mlp_scale = values[2 * dim ..][0..dim],
        .mlp_gate = values[3 * dim ..][0..dim],
    };
}

pub fn finish(parts: Parts) void {
    for (parts.attn_scale) |*value| value.* += 1.0;
    for (parts.mlp_scale) |*value| value.* += 1.0;
    for (parts.attn_gate) |*value| value.* = std.math.tanh(value.*);
    for (parts.mlp_gate) |*value| value.* = std.math.tanh(value.*);
}

test "split returns four equal modulation chunks" {
    var values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const parts = try split(&values);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, parts.attn_scale);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, parts.attn_gate);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6 }, parts.mlp_scale);
    try std.testing.expectEqualSlices(f32, &.{ 7, 8 }, parts.mlp_gate);
}

test "finish applies scale shift and gate tanh" {
    var values = [_]f32{ 1.0, 0.5, 2.0, -1.0 };
    const parts = try split(&values);
    finish(parts);

    try std.testing.expectApproxEqAbs(2.0, parts.attn_scale[0], 0.0001);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 0.5)), parts.attn_gate[0], 0.0001);
    try std.testing.expectApproxEqAbs(3.0, parts.mlp_scale[0], 0.0001);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, -1.0)), parts.mlp_gate[0], 0.0001);
}

test "run modulation from tensor views" {
    const weight = tensor.View{ .dtype = .u8, .shape = &.{ 4, 2 }, .bytes = &.{
        1, 0,
        0, 1,
        2, 0,
        0, 3,
    } };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{4}, .bytes = &.{ 0, 1, 0, 0 } };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const parts = try run(null, &out, &.{ 2.0, 3.0 }, weight, bias);

    try std.testing.expectApproxEqAbs(3.0, parts.attn_scale[0], 0.0001);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 4.0)), parts.attn_gate[0], 0.0001);
    try std.testing.expectApproxEqAbs(5.0, parts.mlp_scale[0], 0.0001);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 9.0)), parts.mlp_gate[0], 0.0001);
}
