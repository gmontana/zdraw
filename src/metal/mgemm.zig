//! Host wrapper for the simdgroup GEMM kernels.

const std = @import("std");

const c = @import("metal_c.zig");
const gmode = @import("../runtime/gemm_mode.zig");
const mbuffer = @import("mbuffer.zig");
const mres_util = @import("mres_util.zig");
const mlinear = @import("mlinear.zig");
const tensor = @import("../pack/tensor.zig");

const tile = 32;
const k_step = 8;

const PairShape = struct {
    rows: usize,
    cols: usize,
};

pub fn run(
    ctx: *mlinear.Context,
    out: mbuffer.Buffer,
    input: mbuffer.Buffer,
    weight: tensor.View,
    m: usize,
    k: usize,
    n: usize,
) !void {
    const pipe = gmode.pipeline(
        ctx.gemm_mode,
        ctx.gemm_exact_pipeline,
        ctx.gemm_half_pipeline,
    ) orelse return error.GemmUnavailable;
    if (!fits(m, k, n)) return error.GemmShape;

    var weight_tmp: ?mbuffer.Buffer = null;
    defer if (weight_tmp) |*buf| buf.deinit();
    const weight_bind = try ctx.buffers.bindView(weight, &weight_tmp);

    const params = c.GemmParams{
        .m = try mres_util.toU32(m),
        .k = try mres_util.toU32(k),
        .n = try mres_util.toU32(n),
        .dtype = try mres_util.dtype(weight.dtype),
        .mode = modeCode(ctx.gemm_mode),
        .weight_offset = try mlinear.u64Fit(weight_bind.offset),
    };
    // The text encoder lands here, on gemm_half - measured at ~5.5 TFLOPS against
    // gemm_f16_direct's ~12.4. Its checkpoint is bf16, so the fast kernel is only
    // reachable with the bf16 staging variant behind ZDRAW_GEMM_BF16.
    if (bf16Enabled() and ours16Fits(&params)) {
        if (c.zdraw_metal_batch_begin(ctx.queue)) |bt| {
            const rc = c.zdraw_metal_run_gemm_ours16_enc(
                bt,
                input.handle,
                weight_bind.handle,
                out.handle,
                &params,
                0,
                0,
            );
            const done = c.zdraw_metal_batch_end(bt);
            // A non-zero rc means the kernel declined and encoded nothing, so
            // falling through to gemm_half below is both correct and required.
            if (rc == 0 and done == 0) return;
        }
    }
    const code = c.zdraw_metal_run_gemm(
        ctx.queue,
        pipe,
        input.handle,
        weight_bind.handle,
        out.handle,
        &params,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

/// Default ON; ZDRAW_GEMM_BF16=0 opts out. Byte-identical on Klein and
/// Z-Image, so the route is a pure throughput change.
fn bf16Enabled() bool {
    const raw = std.c.getenv("ZDRAW_GEMM_BF16") orelse return true;
    return raw[0] != '0';
}

/// Mirrors ours16_ok_w16 in metal_api.m: half mode, f16 or bf16 weights,
/// 32-aligned dims (shared shape math in mres_util).
fn ours16Fits(p: *const c.GemmParams) bool {
    return p.mode == 2 and (p.dtype == 1 or p.dtype == 2) and
        mres_util.ours16Dims(p.m, p.k, p.n);
}

pub fn runPair(
    ctx: *mlinear.Context,
    out0: mbuffer.Buffer,
    out1: mbuffer.Buffer,
    input: mbuffer.Buffer,
    weight0: tensor.View,
    weight1: tensor.View,
    m: usize,
    k: usize,
    n: usize,
) !void {
    const pipe = gmode.pipeline(
        ctx.gemm_mode,
        ctx.gemm_exact_pipeline,
        ctx.gemm_half_pipeline,
    ) orelse return error.GemmUnavailable;
    if (!fits(m, k, n)) return error.GemmShape;

    var weight0_tmp: ?mbuffer.Buffer = null;
    defer if (weight0_tmp) |*buf| buf.deinit();
    var weight1_tmp: ?mbuffer.Buffer = null;
    defer if (weight1_tmp) |*buf| buf.deinit();
    const weight0_bind = try ctx.buffers.bindView(weight0, &weight0_tmp);
    const weight1_bind = try ctx.buffers.bindView(weight1, &weight1_tmp);

    const params0 = try paramsFor(ctx.gemm_mode, weight0.dtype, weight0_bind, m, k, n);
    const params1 = try paramsFor(ctx.gemm_mode, weight1.dtype, weight1_bind, m, k, n);
    const code = c.zdraw_metal_run_gemm_pair(
        ctx.queue,
        pipe,
        input.handle,
        weight0_bind.handle,
        out0.handle,
        &params0,
        weight1_bind.handle,
        out1.handle,
        &params1,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

pub fn batch(
    ctx: *mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    count: usize,
) !void {
    const dims = try mlinear.shape(weight);
    if (count == 0) return error.InvalidShape;
    if (out.len != dims.rows * count or input.len != dims.cols * count) {
        return error.InvalidShape;
    }
    try weight.check();

    var in_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var out_buf = try mbuffer.Buffer.empty(ctx.device, std.mem.sliceAsBytes(out).len);
    defer out_buf.deinit();

    try run(ctx, out_buf, in_buf, weight, count, dims.cols, dims.rows);
    readBack(out_buf, out);
}

pub fn batchPair(
    ctx: *mlinear.Context,
    out0: []f32,
    out1: []f32,
    input: []const f32,
    weight0: tensor.View,
    weight1: tensor.View,
    count: usize,
) !void {
    const dims = try pairShape(out0, out1, input, weight0, weight1, count);

    var in_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var out0_buf = try mbuffer.Buffer.empty(ctx.device, std.mem.sliceAsBytes(out0).len);
    defer out0_buf.deinit();
    var out1_buf = try mbuffer.Buffer.empty(ctx.device, std.mem.sliceAsBytes(out1).len);
    defer out1_buf.deinit();

    try runPair(ctx, out0_buf, out1_buf, in_buf, weight0, weight1, count, dims.cols, dims.rows);
    readBack(out0_buf, out0);
    readBack(out1_buf, out1);
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(
        buf.handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

fn pairShape(
    out0: []const f32,
    out1: []const f32,
    input: []const f32,
    weight0: tensor.View,
    weight1: tensor.View,
    count: usize,
) !PairShape {
    const dims0 = try mlinear.shape(weight0);
    const dims1 = try mlinear.shape(weight1);
    if (count == 0) return error.InvalidShape;
    if (dims0.rows != dims1.rows or dims0.cols != dims1.cols) return error.InvalidShape;
    if (out0.len != dims0.rows * count or out1.len != dims1.rows * count) return error.InvalidShape;
    if (input.len != dims0.cols * count) return error.InvalidShape;
    try weight0.check();
    try weight1.check();
    return .{ .rows = dims0.rows, .cols = dims0.cols };
}

fn paramsFor(
    mode: gmode.Mode,
    dtype: tensor.DType,
    bind: mbuffer.Bind,
    m: usize,
    k: usize,
    n: usize,
) !c.GemmParams {
    return .{
        .m = try mres_util.toU32(m),
        .k = try mres_util.toU32(k),
        .n = try mres_util.toU32(n),
        .dtype = try mres_util.dtype(dtype),
        .mode = modeCode(mode),
        .weight_offset = try mlinear.u64Fit(bind.offset),
    };
}

fn modeCode(mode: gmode.Mode) u32 {
    return switch (mode) {
        .half => 2,
        .w8 => 3,
        .w6 => 4,
        .exact => 1,
        .off => 0,
    };
}

fn fits(m: usize, k: usize, n: usize) bool {
    if (m == 0 or k == 0 or n == 0) return false;
    return m % tile == 0 and n % tile == 0 and k % k_step == 0;
}
