//! Buffer-level resident FFN execution.

const c = @import("metal_c.zig");
const gmode = @import("../runtime/gemm_mode.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const util = @import("mres_util.zig");
const tensor = @import("../pack/tensor.zig");

const Shape = struct { tokens: usize, hidden: usize, inner: usize };
const Binds = struct { gate: mbuffer.Bind, up: mbuffer.Bind, down: mbuffer.Bind };
const Params = struct { gate: c.GemmParams, up: c.GemmParams, down: c.GemmParams };

const Temps = struct {
    gate: ?mbuffer.Buffer = null,
    up: ?mbuffer.Buffer = null,
    down: ?mbuffer.Buffer = null,

    fn deinit(self: *Temps) void {
        if (self.gate) |*buf| buf.deinit();
        if (self.up) |*buf| buf.deinit();
        if (self.down) |*buf| buf.deinit();
    }
};

pub fn run(
    ctx: *mlinear.Context,
    out: mbuffer.Buffer,
    input: mbuffer.Buffer,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
    count: usize,
) !void {
    const shape = try check(gate, up, down, count);
    const gemm = gmode.pipeline(
        ctx.gemm_mode,
        ctx.gemm_exact_pipeline,
        ctx.gemm_half_pipeline,
    ) orelse return error.GemmUnavailable;
    const swiglu = ctx.swiglu_pipeline orelse return error.GemmUnavailable;
    var gate_buf = try empty(ctx, count * shape.inner);
    defer gate_buf.deinit();
    var up_buf = try empty(ctx, count * shape.inner);
    defer up_buf.deinit();
    var temps = Temps{};
    defer temps.deinit();
    const binds = try bind(ctx, gate, up, down, &temps);
    const ps = try makeParams(ctx.gemm_mode, binds, gate, up, down, shape);
    try dispatch(ctx, gemm, swiglu, input, gate_buf, up_buf, out, binds, ps);
}

fn check(gate: tensor.View, up: tensor.View, down: tensor.View, count: usize) !Shape {
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
    if (!util.fits(count, gate_shape.cols, gate_shape.rows)) return error.GemmShape;
    if (!util.fits(count, down_shape.cols, down_shape.rows)) return error.GemmShape;
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
        .gate = try param(mode, gate.dtype, binds.gate, shape.tokens, shape.hidden, shape.inner),
        .up = try param(mode, up.dtype, binds.up, shape.tokens, shape.hidden, shape.inner),
        .down = try param(mode, down.dtype, binds.down, shape.tokens, shape.inner, shape.hidden),
    };
}

fn dispatch(
    ctx: *mlinear.Context,
    gemm: *anyopaque,
    swiglu: *anyopaque,
    input: mbuffer.Buffer,
    gate: mbuffer.Buffer,
    up: mbuffer.Buffer,
    out: mbuffer.Buffer,
    binds: Binds,
    ps: Params,
) !void {
    const code = c.zdraw_metal_run_ffn(
        ctx.queue,
        gemm,
        swiglu,
        input.handle,
        binds.gate.handle,
        binds.up.handle,
        binds.down.handle,
        gate.handle,
        up.handle,
        out.handle,
        &ps.gate,
        &ps.up,
        &ps.down,
        ps.gate.m * ps.gate.n,
        ctx.swiglu_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn param(
    mode: gmode.Mode,
    dtype: tensor.DType,
    bind_in: mbuffer.Bind,
    m: usize,
    k: usize,
    n: usize,
) !c.GemmParams {
    return .{
        .m = try util.toU32(m),
        .k = try util.toU32(k),
        .n = try util.toU32(n),
        .dtype = try util.dtype(dtype),
        .mode = util.mode(mode),
        .weight_offset = try util.toU64(bind_in.offset),
    };
}

fn empty(ctx: *mlinear.Context, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(ctx.device, count * @sizeOf(f32));
}
