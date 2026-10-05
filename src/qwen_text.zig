//! Small CPU pieces of the Qwen text encoder.

const std = @import("std");

const linear = @import("linear_fast.zig");
const mlinear = @import("mlinear.zig");
const ops = @import("ops.zig");
const shards = @import("shards.zig");
const tensor = @import("tensor.zig");
const weight_index = @import("weight_index.zig");

pub const Error = error{
    InvalidShape,
    InvalidToken,
};

pub const MlpScratch = struct {
    gate: []f32,
    up: []f32,
};

const embed_name = "model.embed_tokens.weight";

pub fn embedFromStore(
    out: []f32,
    ids: []const u32,
    store: *const shards.Store,
    index: weight_index.Index,
) !void {
    const table = try store.view(index, embed_name);
    try embed(out, ids, table);
}

pub fn embed(out: []f32, ids: []const u32, table: tensor.View) !void {
    const dims = try tableShape(table);
    if (out.len != ids.len * dims.hidden) return error.InvalidShape;
    try table.check();

    for (ids, 0..) |id, token_i| {
        if (id >= dims.vocab) return error.InvalidToken;
        const token: usize = @intCast(id);
        const row = token * dims.hidden;
        const dst = out[token_i * dims.hidden ..][0..dims.hidden];
        for (dst, 0..) |*value, col| {
            value.* = table.atF32Unchecked(row + col);
        }
    }
}

pub fn mlp(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    gate_w: tensor.View,
    up_w: tensor.View,
    down_w: tensor.View,
    scratch: MlpScratch,
) !void {
    try mlpBatch(metal, out, input, gate_w, up_w, down_w, scratch, 1);
}

pub fn mlpBatch(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    gate_w: tensor.View,
    up_w: tensor.View,
    down_w: tensor.View,
    scratch: MlpScratch,
    batch: usize,
) !void {
    if (batch == 0) return error.InvalidShape;
    if (scratch.gate.len != scratch.up.len) return error.InvalidShape;
    try linear.runBatch(metal, scratch.gate, input, gate_w, null, batch);
    try linear.runBatch(metal, scratch.up, input, up_w, null, batch);

    for (scratch.gate, scratch.up) |*gate, up| {
        gate.* = ops.silu(gate.*) * up;
    }
    try linear.runBatch(metal, out, scratch.gate, down_w, null, batch);
}

const Shape = struct {
    vocab: usize,
    hidden: usize,
};

fn tableShape(table: tensor.View) !Shape {
    if (table.shape.len != 2) return error.InvalidShape;
    if (table.shape[0] == 0 or table.shape[1] == 0) return error.InvalidShape;
    return .{ .vocab = table.shape[0], .hidden = table.shape[1] };
}

test "embed token rows" {
    const shape = [_]usize{ 3, 2 };
    const bytes = [_]u8{
        0x00, 0x3c, 0x00, 0x40,
        0x00, 0x42, 0x00, 0x44,
        0x00, 0x45, 0x00, 0x46,
    };
    const table = tensor.View{ .dtype = .f16, .shape = &shape, .bytes = &bytes };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    try embed(&out, &.{ 2, 0 }, table);

    try std.testing.expectApproxEqAbs(5.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(6.0, out[1], 0.0001);
    try std.testing.expectApproxEqAbs(1.0, out[2], 0.0001);
    try std.testing.expectApproxEqAbs(2.0, out[3], 0.0001);
}

test "reject token outside vocabulary" {
    const shape = [_]usize{ 1, 1 };
    const bytes = [_]u8{ 0x00, 0x3c };
    const table = tensor.View{ .dtype = .f16, .shape = &shape, .bytes = &bytes };
    var out = [_]f32{0.0};
    try std.testing.expectError(error.InvalidToken, embed(&out, &.{1}, table));
}

test "mlp applies swiglu and down projection" {
    const in_shape = [_]usize{ 2, 2 };
    const down_shape = [_]usize{ 1, 2 };
    const id = [_]u8{
        0x00, 0x3c, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x3c,
    };
    const sum = [_]u8{ 0x00, 0x3c, 0x00, 0x3c };
    const gate = tensor.View{ .dtype = .f16, .shape = &in_shape, .bytes = &id };
    const up = tensor.View{ .dtype = .f16, .shape = &in_shape, .bytes = &id };
    const down = tensor.View{ .dtype = .f16, .shape = &down_shape, .bytes = &sum };
    var gate_buf = [_]f32{ 0.0, 0.0 };
    var up_buf = [_]f32{ 0.0, 0.0 };
    var out = [_]f32{0.0};

    try mlp(null, &out, &.{ 1.0, 2.0 }, gate, up, down, .{
        .gate = &gate_buf,
        .up = &up_buf,
    });
    const expected = ops.silu(1.0) * 1.0 + ops.silu(2.0) * 2.0;
    try std.testing.expectApproxEqAbs(expected, out[0], 0.0001);
}

test "mlp batch keeps token rows independent" {
    const in_shape = [_]usize{ 2, 2 };
    const down_shape = [_]usize{ 1, 2 };
    const id = [_]u8{
        0x00, 0x3c, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x3c,
    };
    const sum = [_]u8{ 0x00, 0x3c, 0x00, 0x3c };
    const gate = tensor.View{ .dtype = .f16, .shape = &in_shape, .bytes = &id };
    const up = tensor.View{ .dtype = .f16, .shape = &in_shape, .bytes = &id };
    const down = tensor.View{ .dtype = .f16, .shape = &down_shape, .bytes = &sum };
    var gate_buf = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var up_buf = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var out = [_]f32{ 0.0, 0.0 };

    try mlpBatch(null, &out, &.{ 1.0, 2.0, 3.0, 4.0 }, gate, up, down, .{
        .gate = &gate_buf,
        .up = &up_buf,
    }, 2);

    const first = ops.silu(1.0) * 1.0 + ops.silu(2.0) * 2.0;
    const second = ops.silu(3.0) * 3.0 + ops.silu(4.0) * 4.0;
    try std.testing.expectApproxEqAbs(first, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(second, out[1], 0.0001);
}
