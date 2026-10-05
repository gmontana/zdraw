//! Resident Metal FFN projection.
//!
//! Encodes gate/up GEMMs, SwiGLU, and the down GEMM in one command buffer so
//! the intermediate activation never round-trips through the CPU.

const std = @import("std");

const c = @import("metal_c.zig");
const gmode = @import("../runtime/gemm_mode.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("../pack/tensor.zig");

const tile = 32;
const k_step = 8;

const Shape = struct { tokens: usize, hidden: usize, inner: usize };
const Binds = struct { gate: mbuffer.Bind, up: mbuffer.Bind, down: mbuffer.Bind };
const Params = struct { gate: c.GemmParams, up: c.GemmParams, down: c.GemmParams };
const Temps = struct {
    gate: ?mbuffer.Buffer = null,
    up: ?mbuffer.Buffer = null,
    down: ?mbuffer.Buffer = null,
};

pub fn batch(
    ctx: *mlinear.Context,
    out: []f32,
    input: []const f32,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    count: usize,
) !void {
    const shape = try check(out, input, gate, up, down, count);
    const gemm = gmode.pipeline(
        ctx.gemm_mode,
        ctx.gemm_exact_pipeline,
        ctx.gemm_half_pipeline,
    ) orelse return error.GemmUnavailable;
    const swiglu = ctx.swiglu_pipeline orelse return error.GemmUnavailable;

    var in_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(input));
    defer in_buf.deinit();
    var out_buf = try mbuffer.Buffer.empty(ctx.device, std.mem.sliceAsBytes(out).len);
    defer out_buf.deinit();
    var gate_buf = try mbuffer.Buffer.empty(ctx.device, bytesF32(count * shape.inner));
    defer gate_buf.deinit();
    var up_buf = try mbuffer.Buffer.empty(ctx.device, bytesF32(count * shape.inner));
    defer up_buf.deinit();

    var temps = Temps{};
    defer deinitTemps(&temps);
    const binds = try bind(ctx, gate, up, down, &temps);
    const ffn_params = try makeParams(ctx.gemm_mode, binds, gate, up, down, shape);
    const bufs = ActBufs{ .input = in_buf, .gate = gate_buf, .up = up_buf, .out = out_buf };
    try dispatch(ctx, gemm, swiglu, bufs, binds, ffn_params);
    readBack(out_buf, out);
}

fn check(
    out: []const f32,
    input: []const f32,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    count: usize,
) !Shape {
    const gate_shape = try mlinear.shape(gate);
    const up_shape = try mlinear.shape(up);
    const down_shape = try mlinear.shape(down);
    if (count == 0) return error.InvalidShape;
    if (gate_shape.rows != up_shape.rows or gate_shape.cols != up_shape.cols) {
        return error.InvalidShape;
    }
    if (down_shape.rows != gate_shape.cols or down_shape.cols != gate_shape.rows) {
        return error.InvalidShape;
    }
    if (out.len != count * down_shape.rows or input.len != count * gate_shape.cols) {
        return error.InvalidShape;
    }
    if (!fits(count, gate_shape.cols, gate_shape.rows)) return error.GemmShape;
    if (!fits(count, down_shape.cols, down_shape.rows)) return error.GemmShape;
    try gate.check();
    try up.check();
    try down.check();
    return .{ .tokens = count, .hidden = gate_shape.cols, .inner = gate_shape.rows };
}

fn bind(
    ctx: *mlinear.Context,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    temps: *Temps,
) !Binds {
    return .{
        .gate = try ctx.buffers.bindView(gate, &temps.gate),
        .up = try ctx.buffers.bindView(up, &temps.up),
        .down = try ctx.buffers.bindView(down, &temps.down),
    };
}

fn makeParams(
    mode: gmode.Mode,
    binds: Binds,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    shape: Shape,
) !Params {
    return .{
        .gate = try params(mode, gate.dtype, binds.gate, shape.tokens, shape.hidden, shape.inner),
        .up = try params(mode, up.dtype, binds.up, shape.tokens, shape.hidden, shape.inner),
        .down = try params(mode, down.dtype, binds.down, shape.tokens, shape.inner, shape.hidden),
    };
}

const ActBufs = struct {
    input: mbuffer.Buffer,
    gate: mbuffer.Buffer,
    up: mbuffer.Buffer,
    out: mbuffer.Buffer,
};

fn dispatch(
    ctx: *mlinear.Context,
    gemm: *anyopaque,
    swiglu: *anyopaque,
    bufs: ActBufs,
    binds: Binds,
    params_in: Params,
) !void {
    const code = c.zdraw_metal_run_ffn(
        ctx.queue,
        gemm,
        swiglu,
        bufs.input.handle,
        binds.gate.handle,
        binds.up.handle,
        binds.down.handle,
        bufs.gate.handle,
        bufs.up.handle,
        bufs.out.handle,
        &params_in.gate,
        &params_in.up,
        &params_in.down,
        params_in.gate.m * params_in.gate.n,
        ctx.swiglu_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn params(
    mode: gmode.Mode,
    dtype: tensor.DType,
    weight: mbuffer.Bind,
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
        .weight_offset = try mlinear.u64Fit(weight.offset),
    };
}

fn deinitTemps(temps: *Temps) void {
    if (temps.gate) |*buf| buf.deinit();
    if (temps.up) |*buf| buf.deinit();
    if (temps.down) |*buf| buf.deinit();
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(buf.handle, std.mem.sliceAsBytes(out).ptr, bytesF32(out.len));
}

fn bytesF32(count: usize) usize {
    return count * @sizeOf(f32);
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
