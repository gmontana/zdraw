//! Resident Metal VAE upsample2 + conv chain.

const std = @import("std");

const c = @import("metal_c.zig");
const conv = @import("conv.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("tensor.zig");

pub const Config = struct { channels: usize, height: usize, width: usize };

const Params = extern struct { channels: u32, height: u32, width: u32 };
const Buffers = extern struct {
    input: *anyopaque,
    high: *anyopaque,
    weight: *anyopaque,
    bias: *anyopaque,
    output: *anyopaque,
};
const RunParams = extern struct { up: Params, conv: c.ConvParams };

extern fn zdraw_metal_run_vae_up_conv(
    queue: *anyopaque,
    up_pipeline: *anyopaque,
    conv_pipeline: *anyopaque,
    buffers: *const Buffers,
    params: *const RunParams,
    up_threads: usize,
    conv_threads: usize,
) c_int;

pub fn run(
    ctx: *mconv.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: Config,
    recycle: ?mbuffer.Recycle,
) !void {
    try check(out, input, weight, bias, cfg);
    var set = try makeBufs(ctx, input, out.len, recycle);
    defer set.deinit();
    var temps = Temps{};
    defer temps.deinit();
    const wb = try ctx.buffers.bindView(weight, &temps.weight);
    const bb = try ctx.buffers.bindBias(bias, &temps.bias);
    const params = RunParams{
        .up = try upParams(cfg),
        .conv = try mconv.paramsFor(weight, bias, convCfg(cfg), wb, bb),
    };
    try dispatch(ctx, bufferSet(set, wb, bb), params);
    readBack(set.output, out);
}

fn check(
    out: []const f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: Config,
) !void {
    const up_h = cfg.height * 2;
    const up_w = cfg.width * 2;
    if (input.len != cfg.channels * cfg.height * cfg.width) return error.InvalidShape;
    if (out.len != cfg.channels * up_h * up_w) return error.InvalidShape;
    try mconv.check(out, out, weight, bias, convCfg(cfg));
}

fn dispatch(ctx: *mconv.Context, bufs: Buffers, params: RunParams) !void {
    const code = zdraw_metal_run_vae_up_conv(
        ctx.queue,
        ctx.up_pipeline,
        ctx.pipeline,
        &bufs,
        &params,
        ctx.up_threads,
        ctx.threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn makeBufs(
    ctx: *mconv.Context,
    input: []const f32,
    out_len: usize,
    recycle: ?mbuffer.Recycle,
) !Bufs {
    var input_buf = try mbuffer.Buffer.fromInput(ctx.device, input, recycle);
    errdefer input_buf.deinit();
    var work_buf = try empty(ctx, out_len);
    errdefer work_buf.deinit();
    const out_buf = try empty(ctx, out_len);
    return .{ .input = input_buf, .work = work_buf, .output = out_buf };
}

fn bufferSet(set: Bufs, wb: mbuffer.Bind, bb: mbuffer.Bind) Buffers {
    return .{
        .input = set.input.handle,
        .high = set.work.handle,
        .weight = wb.handle,
        .bias = bb.handle,
        .output = set.output.handle,
    };
}

fn convCfg(cfg: Config) conv.Config {
    return .{
        .in_ch = cfg.channels,
        .out_ch = cfg.channels,
        .height = cfg.height * 2,
        .width = cfg.width * 2,
        .kernel = 3,
        .pad = 1,
    };
}

fn upParams(cfg: Config) !Params {
    return .{
        .channels = try mres_util.toU32(cfg.channels),
        .height = try mres_util.toU32(cfg.height),
        .width = try mres_util.toU32(cfg.width),
    };
}

fn readBack(buf: mbuffer.Buffer, out: []f32) void {
    c.zdraw_metal_read_buffer(
        buf.handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

fn empty(ctx: *mconv.Context, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(ctx.device, count * @sizeOf(f32));
}

const Bufs = struct {
    input: mbuffer.Buffer,
    work: mbuffer.Buffer,
    output: mbuffer.Buffer,

    fn deinit(self: *Bufs) void {
        self.output.deinit();
        self.work.deinit();
        self.input.deinit();
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
