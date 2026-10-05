//! Triple GEMM scheduler for same-input attention projections.
//!
//! It keeps Q/K/V projection dispatches in one command buffer. The outputs
//! still read back today because head norm, RoPE, and attention are not fully
//! resident yet.

const std = @import("std");

const c = @import("metal_c.zig");
const gmode = @import("gemm_mode.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("tensor.zig");

const tile = 32;
const k_step = 8;

const Shape = struct { rows: usize, cols: usize };
const Binds = struct { w0: mbuffer.Bind, w1: mbuffer.Bind, w2: mbuffer.Bind };
const Params = struct { p0: c.GemmParams, p1: c.GemmParams, p2: c.GemmParams };
const Outs = struct { b0: mbuffer.Buffer, b1: mbuffer.Buffer, b2: mbuffer.Buffer };

const Temps = struct {
    w0: ?mbuffer.Buffer = null,
    w1: ?mbuffer.Buffer = null,
    w2: ?mbuffer.Buffer = null,

    fn deinit(self: *Temps) void {
        if (self.w0) |*buf| buf.deinit();
        if (self.w1) |*buf| buf.deinit();
        if (self.w2) |*buf| buf.deinit();
    }
};

pub fn batch(
    ctx: *mlinear.Context,
    out0: []f32,
    out1: []f32,
    out2: []f32,
    input: []const f32,
    weight0: tensor.View,
    weight1: tensor.View,
    weight2: tensor.View,
    count: usize,
) !void {
    const dims = try check(out0, out1, out2, input, weight0, weight1, weight2, count);
    const pipe = gmode.pipeline(
        ctx.gemm_mode,
        ctx.gemm_exact_pipeline,
        ctx.gemm_half_pipeline,
    ) orelse return error.GemmUnavailable;

    var in_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var out0_buf = try empty(ctx, out0.len);
    defer out0_buf.deinit();
    var out1_buf = try empty(ctx, out1.len);
    defer out1_buf.deinit();
    var out2_buf = try empty(ctx, out2.len);
    defer out2_buf.deinit();

    var temps = Temps{};
    defer temps.deinit();
    const binds = try bind(ctx, weight0, weight1, weight2, &temps);
    const views = [3]tensor.View{ weight0, weight1, weight2 };
    const ps = try makeParams(ctx.gemm_mode, binds, views, count, dims);
    const outs = Outs{ .b0 = out0_buf, .b1 = out1_buf, .b2 = out2_buf };
    try dispatch(ctx, pipe, in_buf, outs, binds, ps);
    readBack(out0_buf, out0);
    readBack(out1_buf, out1);
    readBack(out2_buf, out2);
}

fn check(
    out0: []const f32,
    out1: []const f32,
    out2: []const f32,
    input: []const f32,
    weight0: tensor.View,
    weight1: tensor.View,
    weight2: tensor.View,
    count: usize,
) !Shape {
    const dims0 = try mlinear.shape(weight0);
    const dims1 = try mlinear.shape(weight1);
    const dims2 = try mlinear.shape(weight2);
    if (count == 0) return error.InvalidShape;
    if (dims0.rows != dims1.rows or dims0.cols != dims1.cols) return error.InvalidShape;
    if (dims0.rows != dims2.rows or dims0.cols != dims2.cols) return error.InvalidShape;
    if (out0.len != dims0.rows * count or out1.len != out0.len) return error.InvalidShape;
    if (out2.len != out0.len or input.len != dims0.cols * count) return error.InvalidShape;
    if (!fits(count, dims0.cols, dims0.rows)) return error.GemmShape;
    try weight0.check();
    try weight1.check();
    try weight2.check();
    return .{ .rows = dims0.rows, .cols = dims0.cols };
}

fn bind(
    ctx: *mlinear.Context,
    weight0: tensor.View,
    weight1: tensor.View,
    weight2: tensor.View,
    temps: *Temps,
) !Binds {
    return .{
        .w0 = try ctx.buffers.bindView(weight0, &temps.w0),
        .w1 = try ctx.buffers.bindView(weight1, &temps.w1),
        .w2 = try ctx.buffers.bindView(weight2, &temps.w2),
    };
}

fn makeParams(
    mode: gmode.Mode,
    binds: Binds,
    weights: [3]tensor.View,
    m: usize,
    dims: Shape,
) !Params {
    return .{
        .p0 = try params(mode, weights[0].dtype, binds.w0, m, dims.cols, dims.rows),
        .p1 = try params(mode, weights[1].dtype, binds.w1, m, dims.cols, dims.rows),
        .p2 = try params(mode, weights[2].dtype, binds.w2, m, dims.cols, dims.rows),
    };
}

fn dispatch(
    ctx: *mlinear.Context,
    pipe: *anyopaque,
    input: mbuffer.Buffer,
    outs: Outs,
    binds: Binds,
    ps: Params,
) !void {
    const code = c.zdraw_metal_run_gemm_triple(
        ctx.queue,
        pipe,
        input.handle,
        binds.w0.handle,
        outs.b0.handle,
        &ps.p0,
        binds.w1.handle,
        outs.b1.handle,
        &ps.p1,
        binds.w2.handle,
        outs.b2.handle,
        &ps.p2,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn params(
    mode: gmode.Mode,
    dtype: tensor.DType,
    bind_in: mbuffer.Bind,
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
        .weight_offset = try mlinear.u64Fit(bind_in.offset),
    };
}

fn empty(ctx: *mlinear.Context, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(ctx.device, count * @sizeOf(f32));
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(buf.handle, std.mem.sliceAsBytes(out).ptr, out.len * @sizeOf(f32));
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
