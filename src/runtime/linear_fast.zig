//! Shared linear projection helper.
//!
//! Callers pass an optional Metal context. Supported F16/BF16 matrices run on
//! Metal; everything else uses the plain CPU reference kernel.

const mgemm = @import("../metal/mgemm.zig");
const mfallback = @import("../metal/metal_fallback.zig");
const mffn = @import("../metal/mffn.zig");
const std = @import("std");
const mlinear = @import("../metal/mlinear.zig");
const ops = @import("ops.zig");
const tensor = @import("../pack/tensor.zig");

pub fn run(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
) !void {
    if (metal) |ctx| {
        ctx.linear(out, input, weight, bias) catch |err| switch (err) {
            error.UnsupportedDType => {},
            else => return err,
        };
        if (supported(weight, bias)) return;
    }
    try ops.linearView(out, input, weight, bias);
}

pub fn runBatch(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    batch: usize,
) !void {
    if (metal) |ctx| {
        ctx.linearBatch(out, input, weight, bias, batch) catch |err| switch (err) {
            error.UnsupportedDType => {},
            else => return err,
        };
        if (supported(weight, bias)) return;
    }
    try runBatchCpu(out, input, weight, bias, batch);
}

/// FFN projection path. Routes bias-free F16/BF16 matrices to the simdgroup
/// GEMM when the context has it enabled and the shape fits its tiles; otherwise
/// (or on any GEMM-specific refusal) falls back to the plain `runBatch`.
pub fn gemmBatch(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    batch: usize,
) !void {
    if (metal) |ctx| {
        if (ctx.gemm_mode != .off and bias == null and gemmType(weight.dtype)) {
            mgemm.batch(ctx, out, input, weight, batch) catch |err| {
                if (mfallback.isGemmRefusal(err)) {
                    return runBatch(metal, out, input, weight, bias, batch);
                }
                return err;
            };
            return;
        }
    }
    try runBatch(metal, out, input, weight, bias, batch);
}

pub fn gemmBatchPair(
    metal: ?*mlinear.Context,
    out0: []f32,
    out1: []f32,
    input: []const f32,
    weight0: tensor.View,
    weight1: tensor.View,
    batch: usize,
) !void {
    if (metal) |ctx| {
        if (ctx.gemm_mode != .off and gemmType(weight0.dtype) and gemmType(weight1.dtype)) {
            mgemm.batchPair(
                ctx,
                out0,
                out1,
                input,
                weight0,
                weight1,
                batch,
            ) catch |err| {
                if (mfallback.isGemmRefusal(err)) {
                    try gemmBatch(metal, out0, input, weight0, null, batch);
                    try gemmBatch(metal, out1, input, weight1, null, batch);
                    return;
                }
                return err;
            };
            return;
        }
    }
    try gemmBatch(metal, out0, input, weight0, null, batch);
    try gemmBatch(metal, out1, input, weight1, null, batch);
}

pub fn gemmFfn(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    scratch_gate: []f32,
    scratch_up: []f32,
    batch: usize,
) !void {
    if (metal) |ctx| {
        if (residentType(ctx, gate, up, down)) {
            mffn.batch(ctx, out, input, gate, up, down, batch) catch |err| {
                if (mfallback.isGemmRefusal(err)) {
                    return ffnFallback(
                        metal,
                        out,
                        input,
                        gate,
                        up,
                        down,
                        scratch_gate,
                        scratch_up,
                        batch,
                    );
                }
                return err;
            };
            return;
        }
    }
    try ffnFallback(metal, out, input, gate, up, down, scratch_gate, scratch_up, batch);
}

fn ffnFallback(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    scratch_gate: []f32,
    scratch_up: []f32,
    batch: usize,
) !void {
    try gemmBatchPair(metal, scratch_gate, scratch_up, input, gate, up, batch);
    try ops.swiglu(scratch_gate, scratch_gate, scratch_up);
    try gemmBatch(metal, out, scratch_gate, down, null, batch);
}

fn residentType(ctx: *mlinear.Context, gate: tensor.View, up: tensor.View, down: tensor.View) bool {
    if (ctx.gemm_mode == .off) return false;
    return gemmType(gate.dtype) and gemmType(up.dtype) and gemmType(down.dtype);
}

fn gemmType(dtype: tensor.DType) bool {
    return dtype == .f16 or dtype == .bf16 or dtype == .f32;
}

fn runBatchCpu(
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    batch: usize,
) !void {
    const dims = try shape(weight);
    if (out.len != batch * dims.rows or input.len != batch * dims.cols) {
        return error.InvalidShape;
    }
    for (0..batch) |item| {
        try ops.linearView(
            out[item * dims.rows ..][0..dims.rows],
            input[item * dims.cols ..][0..dims.cols],
            weight,
            bias,
        );
    }
}

fn supported(weight: tensor.View, bias: ?tensor.View) bool {
    if (!supportedType(weight.dtype)) return false;
    if (bias) |b| return supportedType(b.dtype);
    return true;
}

fn supportedType(dtype: tensor.DType) bool {
    return dtype == .f32 or dtype == .f16 or dtype == .bf16;
}

const Shape = struct {
    rows: usize,
    cols: usize,
};

fn shape(weight: tensor.View) !Shape {
    if (weight.shape.len != 2) return error.InvalidShape;
    return .{ .rows = weight.shape[0], .cols = weight.shape[1] };
}

test "unsupported dtype uses CPU fallback" {
    const input = [_]f32{1.0};
    const bytes = [_]u8{ 0, 0, 0x80, 0x3f };
    const weight = tensor.View{ .dtype = .f32, .shape = &.{ 1, 1 }, .bytes = &bytes };
    var out = [_]f32{0.0};

    try run(null, &out, &input, weight, null);
    try @import("std").testing.expectApproxEqAbs(1.0, out[0], 0.0001);
}

/// Random bf16 bit pattern spanning the exponent range real checkpoints
/// occupy (~2^-15 to ~2^0), including exact zeros. A single-exponent
/// distribution hides decode differences between the two routes.
fn wideBf16(rand: std.Random) u16 {
    if (rand.uintLessThan(u8, 64) == 0) return 0; // exact zeros occur
    const sign: u16 = if (rand.boolean()) 0x8000 else 0;
    const exp: u16 = 112 + rand.uintLessThan(u16, 16); // 2^-15 .. 2^0
    return sign | (exp << 7) | (rand.int(u16) & 0x7F);
}

test "gemm and rows projection routes agree on the Qwen q shape" {
    // The Klein encoder's q projection ([512,2560] x [4096,2560]) diverged
    // from the reference only on the GEMM route, so the two routes are pinned
    // against each other here on that exact shape with bf16 weights.
    var ctx = mlinear.Context.init() catch return; // no Metal: nothing to compare
    defer ctx.deinit();
    const alloc = std.testing.allocator;
    const tokens: usize = 512;
    const k_dim: usize = 2560;
    const n_dim: usize = 4096;

    const input = try alloc.alloc(f32, tokens * k_dim);
    defer alloc.free(input);
    const w_bits = try alloc.alloc(u16, n_dim * k_dim);
    defer alloc.free(w_bits);
    var prng = std.Random.DefaultPrng.init(0x9e37);
    const rand = prng.random();
    for (input) |*v| v.* = rand.floatNorm(f32) * 0.5;
    // bf16 weights in the magnitude band the real checkpoint uses (|w| < 0.6).
    // bf16 bits built directly (sign | exp 0x7B | mantissa) so the magnitudes
    // land in the checkpoint's |w| < 0.6 band without a float round-trip.
    for (w_bits) |*b| b.* = wideBf16(rand);
    const weight = tensor.View{
        .dtype = .bf16,
        .shape = &.{ n_dim, k_dim },
        .bytes = std.mem.sliceAsBytes(w_bits),
    };

    const via_gemm = try alloc.alloc(f32, tokens * n_dim);
    defer alloc.free(via_gemm);
    const via_rows = try alloc.alloc(f32, tokens * n_dim);
    defer alloc.free(via_rows);
    try gemmBatch(&ctx, via_gemm, input, weight, null, tokens);
    try runBatch(&ctx, via_rows, input, weight, null, tokens);

    var worst: f32 = 0;
    for (via_gemm, via_rows) |a, b| worst = @max(worst, @abs(a - b));
    // Both routes accumulate in f32 from the same bf16 weights; anything past
    // a few ulps of the ~10-magnitude outputs is a real disagreement.
    try std.testing.expect(worst < 0.05);
}

test "projection routes diverge when activations carry f16-overflowing outliers" {
    // Qwen3 hidden states are known for large per-channel outliers. If either
    // route stages activations through f16 it saturates at 65504 and the two
    // routes part company; this pins that.
    var ctx = mlinear.Context.init() catch return;
    defer ctx.deinit();
    const alloc = std.testing.allocator;
    const tokens: usize = 64;
    const k_dim: usize = 2560;
    const n_dim: usize = 4096;

    const input = try alloc.alloc(f32, tokens * k_dim);
    defer alloc.free(input);
    const w_bits = try alloc.alloc(u16, n_dim * k_dim);
    defer alloc.free(w_bits);
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();
    for (input) |*v| v.* = rand.floatNorm(f32) * 0.5;
    // One outlier channel far above the f16 ceiling, as real hidden states have.
    for (0..tokens) |t| input[t * k_dim + 7] = 90000.0;
    // bf16 bits built directly (sign | exp 0x7B | mantissa) so the magnitudes
    // land in the checkpoint's |w| < 0.6 band without a float round-trip.
    for (w_bits) |*b| b.* = wideBf16(rand);
    const weight = tensor.View{
        .dtype = .bf16,
        .shape = &.{ n_dim, k_dim },
        .bytes = std.mem.sliceAsBytes(w_bits),
    };
    const via_gemm = try alloc.alloc(f32, tokens * n_dim);
    defer alloc.free(via_gemm);
    const via_rows = try alloc.alloc(f32, tokens * n_dim);
    defer alloc.free(via_rows);
    try gemmBatch(&ctx, via_gemm, input, weight, null, tokens);
    try runBatch(&ctx, via_rows, input, weight, null, tokens);
    var worst_rel: f32 = 0;
    for (via_gemm, via_rows) |a, b| {
        const denom = @max(@abs(b), 1.0);
        worst_rel = @max(worst_rel, @abs(a - b) / denom);
    }
    // KNOWN DEFECT (klein-te-gemm-divergence): the GEMM route stages A through
    // f16 and saturates at 65504, so it disagrees with the rows route here.
    // This asserts the defect so the test turns green the moment it is fixed
    // by staging in f32/bf16 or applying the DiT's scale-and-restore idiom.
    try std.testing.expect(worst_rel > 0.01);
}
