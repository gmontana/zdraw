//! Tests for the optional Metal linear path.

const std = @import("std");

const mgemm = @import("mgemm.zig");
const mlinear = @import("mlinear.zig");
const tensor = @import("tensor.zig");

const weight_bytes = [_]u8{
    0x00, 0x3c, 0x00, 0x40,
    0x00, 0x42, 0x00, 0x44,
};

const f32_weight_bytes = [_]u8{
    0x00, 0x00, 0x80, 0x3f,
    0x00, 0x00, 0x00, 0x40,
    0x00, 0x00, 0x40, 0x40,
    0x00, 0x00, 0x80, 0x40,
};

fn context() !mlinear.Context {
    return mlinear.Context.init();
}

fn weight() tensor.View {
    return .{ .dtype = .f16, .shape = &.{ 2, 2 }, .bytes = &weight_bytes };
}

fn f32Weight() tensor.View {
    return .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = &f32_weight_bytes };
}

test "Metal linear matches CPU for small F16 matrix" {
    var ctx = context() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();

    const input = [_]f32{ 1.0, 2.0 };
    var out = [_]f32{ 0.0, 0.0 };
    try ctx.linear(&out, &input, weight(), null);

    try std.testing.expectApproxEqAbs(5.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(11.0, out[1], 0.0001);
}

test "Metal batched linear projects tiled rows" {
    var ctx = context() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var out = [_]f32{0.0} ** 10;
    try ctx.linearBatch(&out, &input, weight(), null, 5);

    for (0..5) |item| {
        const x0 = input[item * 2];
        const x1 = input[item * 2 + 1];
        try std.testing.expectApproxEqAbs(x0 + 2.0 * x1, out[item * 2], 0.0001);
        try std.testing.expectApproxEqAbs(3.0 * x0 + 4.0 * x1, out[item * 2 + 1], 0.0001);
    }
}

test "Metal linear accepts F32 transformer weights" {
    var ctx = context() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();

    const input = [_]f32{ 2.0, 3.0 };
    var out = [_]f32{ 0.0, 0.0 };
    try ctx.linear(&out, &input, f32Weight(), null);

    try std.testing.expectApproxEqAbs(8.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(18.0, out[1], 0.0001);
}

test "Metal GEMM matches CPU for BF16 FFN-shaped matrix" {
    var ctx = context() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer ctx.deinit();
    if (ctx.gemm_exact_pipeline == null) return;
    ctx.gemm_mode = .exact;

    const m = 32;
    const k = 8;
    const n = 32;

    var input: [m * k]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;

    // BF16 = the high 16 bits of the f32; the CPU reference decodes the same bits
    // (low half zeroed) so it agrees with the kernel's widening exactly.
    var wbytes: [n * k * 2]u8 = undefined;
    var wvals: [n * k]f32 = undefined;
    for (0..n * k) |i| {
        const raw = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.25;
        const src = std.mem.asBytes(&raw);
        wbytes[i * 2] = src[2];
        wbytes[i * 2 + 1] = src[3];
        const dec = [4]u8{ 0, 0, src[2], src[3] };
        wvals[i] = std.mem.bytesToValue(f32, &dec);
    }
    const bf16_weight = tensor.View{ .dtype = .bf16, .shape = &.{ n, k }, .bytes = &wbytes };

    var out: [m * n]f32 = undefined;
    try mgemm.batch(&ctx, &out, &input, bf16_weight, m);

    for (0..m) |mi| {
        for (0..n) |ni| {
            var ref: f32 = 0;
            for (0..k) |kk| ref += input[mi * k + kk] * wvals[ni * k + kk];
            try std.testing.expectApproxEqAbs(ref, out[mi * n + ni], 0.005);
        }
    }
}
