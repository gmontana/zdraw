//! Timestep embedding for Z-Image denoising.
//!
//! This is the same small MLP used by the reference model: sinusoidal time
//! features, SiLU, then a second linear layer. The caller owns scratch memory.

const std = @import("std");

const linear = @import("../runtime/linear_fast.zig");
const mlinear = @import("../metal/mlinear.zig");
const ops = @import("../runtime/ops.zig");
const tensor = @import("../pack/tensor.zig");

pub const Error = error{
    InvalidShape,
};

pub const Views = struct {
    w0: tensor.View,
    b0: tensor.View,
    w1: tensor.View,
    b1: tensor.View,
};

const Shape = struct {
    rows: usize,
    cols: usize,
};

pub fn embed(out: []f32, t: f32) !void {
    try embedPeriod(out, t, 10000.0);
}

pub fn embedPeriod(out: []f32, t: f32, max_period: f32) !void {
    if (out.len == 0 or max_period <= 0.0) return error.InvalidShape;
    const half = out.len / 2;
    if (half == 0) return error.InvalidShape;

    const half_f: f32 = @floatFromInt(half);
    for (0..half) |i| {
        const idx: f32 = @floatFromInt(i);
        const freq = @exp(-@log(max_period) * idx / half_f);
        out[i] = @cos(t * freq);
        out[i + half] = @sin(t * freq);
    }
    if (out.len % 2 != 0) out[out.len - 1] = 0.0;
}

pub fn run(
    metal: ?*mlinear.Context,
    out: []f32,
    scratch: []f32,
    t: f32,
    t_scale: f32,
    views: Views,
) !void {
    const first = try matrix(views.w0);
    const second = try matrix(views.w1);
    if (second.cols != first.rows or out.len != second.rows) return error.InvalidShape;
    if (scratch.len < first.cols + first.rows) return error.InvalidShape;

    const freq = scratch[0..first.cols];
    const hidden = scratch[first.cols..][0..first.rows];
    try embed(freq, t * t_scale);
    try linear.run(metal, hidden, freq, views.w0, views.b0);
    for (hidden) |*value| value.* = ops.silu(value.*);
    try linear.run(metal, out, hidden, views.w1, views.b1);
}

fn matrix(view: tensor.View) !Shape {
    if (view.shape.len != 2) return error.InvalidShape;
    if (view.shape[0] == 0 or view.shape[1] == 0) return error.InvalidShape;
    return .{ .rows = view.shape[0], .cols = view.shape[1] };
}

test "sinusoidal embedding matches reference layout" {
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    try embedPeriod(&out, 1.0, 100.0);

    try std.testing.expectApproxEqAbs(@cos(1.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@cos(0.1), out[1], 0.0001);
    try std.testing.expectApproxEqAbs(@sin(1.0), out[2], 0.0001);
    try std.testing.expectApproxEqAbs(@sin(0.1), out[3], 0.0001);
}

test "odd embedding pads last value" {
    var out = [_]f32{ 5.0, 5.0, 5.0 };
    try embed(&out, 0.0);

    try std.testing.expectApproxEqAbs(1.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(0.0, out[1], 0.0001);
    try std.testing.expectApproxEqAbs(0.0, out[2], 0.0001);
}

test "run timestep MLP from tensor views" {
    const w0 = tensor.View{ .dtype = .u8, .shape = &.{ 2, 2 }, .bytes = &.{ 1, 0, 0, 1 } };
    const b0 = tensor.View{ .dtype = .u8, .shape = &.{2}, .bytes = &.{ 0, 1 } };
    const w1 = tensor.View{ .dtype = .u8, .shape = &.{ 1, 2 }, .bytes = &.{ 2, 3 } };
    const b1 = tensor.View{ .dtype = .u8, .shape = &.{1}, .bytes = &.{1} };
    var scratch = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var out = [_]f32{0.0};

    try run(null, &out, &scratch, 0.0, 1.0, .{ .w0 = w0, .b0 = b0, .w1 = w1, .b1 = b1 });
    try std.testing.expectApproxEqAbs(1.0 + 5.0 * ops.silu(1.0), out[0], 0.0001);
}
