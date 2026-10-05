//! Metal final LayerNorm + projection chain.

const std = @import("std");

const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("../pack/tensor.zig");

const tile = 32;
const k_step = 8;

pub const NormParams = extern struct { tokens: u32, hidden: u32, eps: f32, pad: u32 = 0 };
pub const BiasParams = extern struct { n: u32, dtype: u32, bias_offset: u64 };

extern fn zdraw_metal_run_final_proj(
    queue: *anyopaque,
    norm_pipeline: *anyopaque,
    gemm_pipeline: *anyopaque,
    bias_pipeline: *anyopaque,
    state: *anyopaque,
    scale: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    batch: *anyopaque,
    output: *anyopaque,
    norm_params: *const NormParams,
    gemm_params: *const c.GemmParams,
    bias_params: *const BiasParams,
    norm_threads: usize,
) c_int;

pub const Config = struct { tokens: usize, hidden: usize, out_dim: usize, eps: f32 };

pub fn run(
    ctx: *mlinear.Context,
    out: []f32,
    state: []const f32,
    scale: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    cfg: Config,
) !void {
    try check(out, state, scale, bias, cfg);
    if (!fits(cfg.tokens, cfg.hidden, cfg.out_dim)) return error.GemmShape;
    if (ctx.gemm_mode == .off) return error.GemmUnavailable;
    var bufs = try makeBufs(ctx, out.len, state, scale);
    defer bufs.deinit();
    var temps = Temps{};
    defer temps.deinit();
    const wb = try ctx.buffers.bindView(weight, &temps.weight);
    const bb = try ctx.buffers.bindView(bias, &temps.bias);
    const params = try makeParams(weight, bias, wb, bb, cfg);
    try dispatch(ctx, bufs, .{ .weight = wb.handle, .bias = bb.handle }, params);
    readBack(bufs.output, out);
}

fn dispatch(ctx: *mlinear.Context, bufs: Bufs, binds: Binds, params: Params) !void {
    const code = zdraw_metal_run_final_proj(
        ctx.queue,
        ctx.final_pipeline orelse return error.GemmUnavailable,
        ctx.gemm_exact_pipeline orelse return error.GemmUnavailable,
        ctx.gemm_bias_pipeline orelse return error.GemmUnavailable,
        bufs.state.handle,
        bufs.scale.handle,
        binds.weight,
        binds.bias,
        bufs.batch.handle,
        bufs.output.handle,
        &params.norm,
        &params.gemm,
        &params.bias,
        ctx.final_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

pub fn makeParams(
    weight: tensor.View,
    bias: tensor.View,
    wb: mbuffer.Bind,
    bb: mbuffer.Bind,
    cfg: Config,
) !Params {
    if (!fits(cfg.tokens, cfg.hidden, cfg.out_dim)) return error.GemmShape;
    return .{
        .norm = .{
            .tokens = try mres_util.toU32(cfg.tokens),
            .hidden = try mres_util.toU32(cfg.hidden),
            .eps = cfg.eps,
        },
        .gemm = .{
            .m = try mres_util.toU32(cfg.tokens),
            .k = try mres_util.toU32(cfg.hidden),
            .n = try mres_util.toU32(cfg.out_dim),
            .dtype = try mres_util.dtype(weight.dtype),
            .mode = 1,
            .weight_offset = try mlinear.u64Fit(wb.offset),
        },
        .bias = .{
            .n = try mres_util.toU32(cfg.out_dim),
            .dtype = try mres_util.dtype(bias.dtype),
            .bias_offset = try mlinear.u64Fit(bb.offset),
        },
    };
}

fn check(
    out: []const f32,
    state: []const f32,
    scale: []const f32,
    bias: tensor.View,
    cfg: Config,
) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0 or cfg.out_dim == 0) return error.InvalidShape;
    if (state.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (out.len != cfg.tokens * cfg.out_dim) return error.InvalidShape;
    if (scale.len != cfg.hidden or try bias.elems() != cfg.out_dim) return error.InvalidShape;
    try bias.check();
}

fn makeBufs(
    ctx: *mlinear.Context,
    out_len: usize,
    state: []const f32,
    scale: []const f32,
) !Bufs {
    var state_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(state));
    errdefer state_buf.deinit();
    var scale_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(scale));
    errdefer scale_buf.deinit();
    var batch = try empty(ctx, state.len);
    errdefer batch.deinit();
    const output = try empty(ctx, out_len);
    return .{ .state = state_buf, .scale = scale_buf, .batch = batch, .output = output };
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(
        buf.handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

fn empty(ctx: *mlinear.Context, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(ctx.device, count * @sizeOf(f32));
}

fn fits(m: usize, k: usize, n: usize) bool {
    if (m == 0 or k == 0 or n == 0) return false;
    return m % tile == 0 and n % tile == 0 and k % k_step == 0;
}

pub const Params = struct { norm: NormParams, gemm: c.GemmParams, bias: BiasParams };

const Binds = struct { weight: *anyopaque, bias: *anyopaque };

const Bufs = struct {
    state: mbuffer.Buffer,
    scale: mbuffer.Buffer,
    batch: mbuffer.Buffer,
    output: mbuffer.Buffer,

    fn deinit(self: *Bufs) void {
        self.output.deinit();
        self.batch.deinit();
        self.scale.deinit();
        self.state.deinit();
    }
};

const Temps = struct {
    weight: ?mbuffer.Buffer = null,
    bias: ?mbuffer.Buffer = null,

    fn deinit(self: *Temps) void {
        if (self.bias) |*buf| buf.deinit();
        if (self.weight) |*buf| buf.deinit();
    }
};

test "rejects non-tiled final shape" {
    try std.testing.expect(!fits(1, 3840, 64));
    try std.testing.expect(fits(32, 3840, 64));
}
