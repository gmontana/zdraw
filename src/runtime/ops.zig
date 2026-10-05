//! Small CPU reference kernels for model tests.

const std = @import("std");

const tensor = @import("../pack/tensor.zig");

pub const Error = error{
    InvalidShape,
};

pub fn dot(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.InvalidShape;
    var sum: f32 = 0.0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

pub fn linear(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
) !void {
    if (out.len == 0 or input.len == 0) return error.InvalidShape;
    if (weight.len != out.len * input.len) return error.InvalidShape;
    if (bias) |b| {
        if (b.len != out.len) return error.InvalidShape;
    }

    for (out, 0..) |*dst, row| {
        const offset = row * input.len;
        dst.* = try dot(input, weight[offset..][0..input.len]);
        if (bias) |b| dst.* += b[row];
    }
}

pub fn linearView(
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
) !void {
    const dims = try matrixShape(weight);
    if (dims.rows != out.len or dims.cols != input.len) return error.InvalidShape;
    try weight.check();
    if (bias) |b| {
        if (try b.elems() != out.len) return error.InvalidShape;
        try b.check();
    }

    for (out, 0..) |*dst, row| {
        var sum: f32 = 0.0;
        const offset = row * dims.cols;
        for (input, 0..) |value, col| {
            sum += value * weight.atF32Unchecked(offset + col);
        }
        dst.* = sum;
        if (bias) |b| dst.* += b.atF32Unchecked(row);
    }
}

pub fn rmsNorm(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) !void {
    if (out.len != input.len or weight.len != input.len) return error.InvalidShape;
    if (input.len == 0) return error.InvalidShape;

    var mean: f32 = 0.0;
    for (input) |value| mean += value * value;
    const count: f32 = @floatFromInt(input.len);
    const scale = 1.0 / @sqrt(mean / count + eps);

    for (out, input, weight) |*dst, value, gain| {
        dst.* = value * scale * gain;
    }
}

pub fn rmsNormView(
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    eps: f32,
) !void {
    if (out.len != input.len or try weight.elems() != input.len) {
        return error.InvalidShape;
    }
    if (input.len == 0) return error.InvalidShape;
    try weight.check();

    var mean: f32 = 0.0;
    for (input) |value| mean += value * value;
    const count: f32 = @floatFromInt(input.len);
    const scale = 1.0 / @sqrt(mean / count + eps);

    for (out, input, 0..) |*dst, value, i| {
        dst.* = value * scale * weight.atF32Unchecked(i);
    }
}

pub fn silu(value: f32) f32 {
    return value / (1.0 + @exp(-value));
}

pub fn swiglu(out: []f32, gate: []const f32, up: []const f32) !void {
    if (out.len != gate.len or up.len != gate.len) return error.InvalidShape;
    for (out, gate, up) |*dst, g, u| dst.* = silu(g) * u;
}

const Matrix = struct {
    rows: usize,
    cols: usize,
};

fn matrixShape(weight: tensor.View) !Matrix {
    if (weight.shape.len != 2) return error.InvalidShape;
    if (weight.shape[0] == 0 or weight.shape[1] == 0) return error.InvalidShape;
    return .{ .rows = weight.shape[0], .cols = weight.shape[1] };
}

test "dot and linear" {
    const input = [_]f32{ 1.0, 2.0 };
    const weight = [_]f32{ 3.0, 4.0, 5.0, 6.0 };
    const bias = [_]f32{ 1.0, -1.0 };
    var out = [_]f32{ 0.0, 0.0 };

    try linear(&out, &input, &weight, &bias);
    try std.testing.expectApproxEqAbs(12.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(16.0, out[1], 0.0001);
}

test "linear from tensor view" {
    const input = [_]f32{ 1.0, 2.0 };
    const shape = [_]usize{ 2, 2 };
    const weight_bytes = [_]u8{
        0x00, 0x42, 0x00, 0x44,
        0x00, 0x45, 0x00, 0x46,
    };
    const bias_bytes = [_]u8{ 1, 2 };
    const weight = tensor.View{ .dtype = .f16, .shape = &shape, .bytes = &weight_bytes };
    const bias = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &bias_bytes };
    var out = [_]f32{ 0.0, 0.0 };

    try linearView(&out, &input, weight, bias);
    try std.testing.expectApproxEqAbs(12.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(19.0, out[1], 0.0001);
}

test "rms norm and swiglu" {
    const input = [_]f32{ 3.0, 4.0 };
    const gain = [_]f32{ 1.0, 2.0 };
    var norm = [_]f32{ 0.0, 0.0 };
    try rmsNorm(&norm, &input, &gain, 0.0);

    try std.testing.expectApproxEqAbs(0.8485, norm[0], 0.0001);
    try std.testing.expectApproxEqAbs(2.2627, norm[1], 0.0001);

    var out = [_]f32{0.0};
    try swiglu(&out, &.{1.0}, &.{2.0});
    try std.testing.expectApproxEqAbs(1.4621, out[0], 0.0001);
}

test "rms norm from tensor view" {
    const input = [_]f32{ 3.0, 4.0 };
    const gain_bytes = [_]u8{ 0x00, 0x3c, 0x00, 0x40 };
    const gain = tensor.View{ .dtype = .f16, .shape = &.{2}, .bytes = &gain_bytes };
    var out = [_]f32{ 0.0, 0.0 };

    try rmsNormView(&out, &input, gain, 0.0);
    try std.testing.expectApproxEqAbs(0.8485, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(2.2627, out[1], 0.0001);
}
