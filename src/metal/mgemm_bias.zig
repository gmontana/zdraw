//! Biased simdgroup GEMM wrapper for fixed-shape VAE projections.

const std = @import("std");

const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mfinal = @import("mfinal.zig");
const mlinear = @import("mlinear.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("../pack/tensor.zig");

const tile = 32;
const k_step = 8;

const Shape = struct {
    rows: usize,
    cols: usize,
};

// One Zig-side definition only (the ABI audit found this duplicated).
const BiasParams = mfinal.BiasParams;

extern fn zdraw_metal_run_gemm_bias(
    queue: *anyopaque,
    gemm_pipeline: *anyopaque,
    bias_pipeline: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    bias: *anyopaque,
    out: *anyopaque,
    params: *const c.GemmParams,
    bias_params: *const BiasParams,
) c_int;

pub fn batch(
    ctx: *mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    count: usize,
) !void {
    const dims = try shape(weight);
    try check(out, input, bias, count, dims);
    // Activation in/out borrow the chain pool (mres_buf's pattern): the VAE
    // decodes while the denoise chain is idle, and both slots are fully
    // written before they are read. Fresh per-call MTLBuffers never return
    // their GPU wiring on macOS, so this path accrued ~0.27 GB per 1024 decode.
    if (!fits(count, dims.cols, dims.rows)) return error.GemmShape;
    if (ctx.gemm_mode == .off) return error.GemmUnavailable;
    const in_h = try ctx.pool.filled(ctx.device, .norm, std.mem.sliceAsBytes(input));
    const out_h = try ctx.pool.handle(ctx.device, .attn, std.mem.sliceAsBytes(out).len);
    try batchOnHandles(ctx, in_h, out_h, weight, bias, count);
    readBack(out_h, out);
}

/// Dispatch-only variant on caller-owned GPU handles (the resident mid-
/// attention path): same shape/route checks, no pool fill and no readback.
/// The dispatch is synchronous, so the temp weight/bias binds may be released
/// on return.
pub fn batchOnHandles(
    ctx: *mlinear.Context,
    in_h: *anyopaque,
    out_h: *anyopaque,
    weight: tensor.View,
    bias: tensor.View,
    count: usize,
) !void {
    const dims = try shape(weight);
    if (try bias.elems() != dims.rows) return error.InvalidShape;
    try bias.check();
    if (!fits(count, dims.cols, dims.rows)) return error.GemmShape;
    if (ctx.gemm_mode == .off) return error.GemmUnavailable;
    const gemm_pipe = ctx.gemm_exact_pipeline orelse return error.GemmUnavailable;
    const bias_pipe = ctx.gemm_bias_pipeline orelse return error.GemmUnavailable;
    var weight_tmp: ?mbuffer.Buffer = null;
    defer if (weight_tmp) |*buf| buf.deinit();
    var bias_tmp: ?mbuffer.Buffer = null;
    defer if (bias_tmp) |*buf| buf.deinit();
    const weight_bind = try ctx.buffers.bindView(weight, &weight_tmp);
    const bias_bind = try ctx.buffers.bindView(bias, &bias_tmp);
    const params = try gemmParams(weight, weight_bind, count, dims);
    const bparams = try biasParams(bias, bias_bind, dims.rows);
    const code = zdraw_metal_run_gemm_bias(
        ctx.queue,
        gemm_pipe,
        bias_pipe,
        in_h,
        weight_bind.handle,
        bias_bind.handle,
        out_h,
        &params,
        &bparams,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn check(
    out: []const f32,
    input: []const f32,
    bias: tensor.View,
    count: usize,
    dims: Shape,
) !void {
    if (count == 0) return error.InvalidShape;
    if (out.len != dims.rows * count or input.len != dims.cols * count) {
        return error.InvalidShape;
    }
    if (try bias.elems() != dims.rows) return error.InvalidShape;
    try bias.check();
}

fn shape(weight: tensor.View) !Shape {
    if (weight.shape.len != 2) return error.InvalidShape;
    if (weight.shape[0] == 0 or weight.shape[1] == 0) return error.InvalidShape;
    try weight.check();
    return .{ .rows = weight.shape[0], .cols = weight.shape[1] };
}

fn gemmParams(
    weight: tensor.View,
    bind: mbuffer.Bind,
    count: usize,
    dims: Shape,
) !c.GemmParams {
    return .{
        .m = try mres_util.toU32(count),
        .k = try mres_util.toU32(dims.cols),
        .n = try mres_util.toU32(dims.rows),
        .dtype = try mres_util.dtype(weight.dtype),
        .mode = 1,
        .weight_offset = try mlinear.u64Fit(bind.offset),
    };
}

fn biasParams(bias: tensor.View, bind: mbuffer.Bind, rows: usize) !BiasParams {
    return .{
        .n = try mres_util.toU32(rows),
        .dtype = try mres_util.dtype(bias.dtype),
        .bias_offset = try mlinear.u64Fit(bind.offset),
    };
}

fn readBack(handle: *anyopaque, out: []f32) void {
    c.zdraw_metal_read_buffer(
        handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

fn fits(m: usize, k: usize, n: usize) bool {
    if (m == 0 or k == 0 or n == 0) return false;
    return m % tile == 0 and n % tile == 0 and k % k_step == 0;
}
