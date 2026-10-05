//! Resident Metal VAE final norm + RGB conv chain.
const std = @import("std");
const c = @import("metal_c.zig");
const conv = @import("conv.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const mres_util = @import("mres_util.zig");
const pmod = @import("mvres_param.zig");
const tensor = @import("tensor.zig");
pub const Config = struct { channels: usize, height: usize, width: usize };
const Buffers = extern struct {
    input: *anyopaque,
    norm: *anyopaque,
    norm_w: *anyopaque,
    norm_b: *anyopaque,
    conv_w: *anyopaque,
    conv_b: *anyopaque,
    output: *anyopaque,
};
const Params = extern struct { norm: pmod.NormParams, conv: c.ConvParams };
extern fn zdraw_metal_run_vae_final(
    queue: *anyopaque,
    norm_pipeline: *anyopaque,
    conv_pipeline: *anyopaque,
    buffers: *const Buffers,
    params: *const Params,
    norm_threads: usize,
    conv_threads: usize,
) c_int;

// Mirrors ZdrawVaeFinalStripBuffers (the strip finish, memory-ladder wall 1).
const StripBuffers = extern struct {
    input: *anyopaque,
    stats: *anyopaque,
    norm_strip: *anyopaque,
    norm_w: *anyopaque,
    norm_b: *anyopaque,
    conv_w: *anyopaque,
    conv_b: *anyopaque,
    output: *anyopaque,
};

extern fn zdraw_metal_run_vae_final_strips(
    queue: *anyopaque,
    stats_pipeline: *anyopaque,
    apply_pipeline: *anyopaque,
    conv_pipeline: *anyopaque,
    buffers: *const StripBuffers,
    params: *const Params,
    strip_rows: u32,
    stats_threads: usize,
    apply_threads: usize,
) c_int;

/// The pipes and scratch the strip finish needs beyond the context's own.
pub const StripFinish = struct {
    stats_pipe: *anyopaque,
    apply_pipe: *anyopaque,
    conv_pipe: *anyopaque,
    stats: *anyopaque,
    norm_strip: *anyopaque,
    strip_rows: u32,
    stats_threads: usize,
    apply_threads: usize,
};

/// The finish over a resident half feature without the whole-map f32 norm
/// scratch: statistics once, then per strip the norm apply into a
/// strip-local float buffer and the RGB conv reading it. Same kernels'
/// arithmetic as `run` (the two-pass stats are vae_norm_silu_h's reductions
/// verbatim; the apply and the conv are addressing-only variants).
pub fn runStrips(
    ctx: *mconv.Context,
    out: []f32,
    input: *anyopaque,
    norm_w: tensor.View,
    norm_b: tensor.View,
    conv_w: tensor.View,
    conv_b: ?tensor.View,
    cfg: Config,
    sf: StripFinish,
) !void {
    try check(out, &.{}, norm_w, norm_b, conv_w, conv_b, cfg, true);
    var output = try empty(ctx, out.len);
    defer output.deinit();
    var temps = Temps{};
    defer temps.deinit();
    const binds = try bindAll(ctx, norm_w, norm_b, conv_w, conv_b, &temps);
    const params = try makeParams(norm_w, norm_b, conv_w, conv_b, binds, cfg);
    const bufs = StripBuffers{
        .input = input,
        .stats = sf.stats,
        .norm_strip = sf.norm_strip,
        .norm_w = binds.norm_w.handle,
        .norm_b = binds.norm_b.handle,
        .conv_w = binds.conv_w.handle,
        .conv_b = binds.conv_b.handle,
        .output = output.handle,
    };
    const code = zdraw_metal_run_vae_final_strips(
        ctx.queue,
        sf.stats_pipe,
        sf.apply_pipe,
        sf.conv_pipe,
        &bufs,
        &params,
        sf.strip_rows,
        sf.stats_threads,
        sf.apply_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
    readBack(output, out);
}

// Caller-owned GPU buffers for the input + norm scratch. When supplied (e.g.
// the streamed decode's already-resident pool buffers), the finish reuses them
// instead of allocating its own ~0.5 GB each, so no new top-resolution GPU
// buffer faults in at the decode's memory peak. Both must hold channels*h*w
// f32. With `resident`, `input` already holds the final feature on-GPU (the
// streamed chain's last group output), so the host upload is skipped entirely
// and the caller passes an empty host slice.
pub const Scratch = struct {
    input: *anyopaque,
    norm: *anyopaque,
    resident: bool = false,
    input_f16: bool = false,
};

pub fn run(
    ctx: *mconv.Context,
    out: []f32,
    input: []const f32,
    norm_w: tensor.View,
    norm_b: tensor.View,
    conv_w: tensor.View,
    conv_b: ?tensor.View,
    cfg: Config,
    recycle: ?mbuffer.Recycle,
    scratch: ?Scratch,
) !void {
    try check(out, input, norm_w, norm_b, conv_w, conv_b, cfg, residentIn(scratch));
    var set = try makeBufs(ctx, input, out.len, recycle, scratch);
    defer set.deinit();
    var temps = Temps{};
    defer temps.deinit();
    const binds = try bindAll(ctx, norm_w, norm_b, conv_w, conv_b, &temps);
    const params = try makeParams(norm_w, norm_b, conv_w, conv_b, binds, cfg);
    try dispatch(ctx, bufferSet(set, binds), params, if (scratch) |s| s.input_f16 else false);
    readBack(set.output, out);
}

fn check(
    out: []const f32,
    input: []const f32,
    norm_w: tensor.View,
    norm_b: tensor.View,
    conv_w: tensor.View,
    conv_b: ?tensor.View,
    cfg: Config,
    resident: bool,
) !void {
    // Resident input lives on the GPU (scratch.input); the host slice must be
    // empty and only the weight shapes can be validated against cfg.
    const expect_in: usize = if (resident) 0 else cfg.channels * cfg.height * cfg.width;
    if (input.len != expect_in) return error.InvalidShape;
    if (out.len != 3 * cfg.height * cfg.width) return error.InvalidShape;
    if (try norm_w.elems() != cfg.channels or try norm_b.elems() != cfg.channels) {
        return error.InvalidShape;
    }
    try norm_w.check();
    try norm_b.check();
    try mconv.checkWeight(conv_w, conv_b, convCfg(cfg));
}

fn residentIn(scratch: ?Scratch) bool {
    return if (scratch) |s| s.resident else false;
}

fn makeParams(
    norm_w: tensor.View,
    norm_b: tensor.View,
    conv_w: tensor.View,
    conv_b: ?tensor.View,
    binds: Binds,
    cfg: Config,
) !Params {
    return .{
        .norm = try normParams(norm_w, norm_b, binds.norm_w, binds.norm_b, cfg),
        .conv = try mconv.paramsFor(conv_w, conv_b, convCfg(cfg), binds.conv_w, binds.conv_b),
    };
}

fn normParams(
    weight: tensor.View,
    bias: tensor.View,
    wb: mbuffer.Bind,
    bb: mbuffer.Bind,
    cfg: Config,
) !pmod.NormParams {
    return .{
        .channels = try mres_util.toU32(cfg.channels),
        .height = try mres_util.toU32(cfg.height),
        .width = try mres_util.toU32(cfg.width),
        .groups = 32,
        .dtype = try mres_util.dtype(weight.dtype),
        .bias_dtype = try mres_util.dtype(bias.dtype),
        .eps = 0.000001,
        .weight_offset = wb.offset,
        .bias_offset = bb.offset,
    };
}

fn bindAll(
    ctx: *mconv.Context,
    nw: tensor.View,
    nb: tensor.View,
    cw: tensor.View,
    cb: ?tensor.View,
    temps: *Temps,
) !Binds {
    return .{
        .norm_w = try ctx.buffers.bindView(nw, &temps.norm_w),
        .norm_b = try ctx.buffers.bindView(nb, &temps.norm_b),
        .conv_w = try ctx.buffers.bindView(cw, &temps.conv_w),
        .conv_b = try ctx.buffers.bindBias(cb, &temps.conv_b),
    };
}

fn dispatch(ctx: *mconv.Context, bufs: Buffers, params: Params, input_f16: bool) !void {
    const code = zdraw_metal_run_vae_final(
        ctx.queue,
        if (input_f16) ctx.norm_h_pipeline else ctx.norm_pipeline,
        ctx.pipeline,
        &bufs,
        &params,
        ctx.norm_threads,
        ctx.threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn makeBufs(
    ctx: *mconv.Context,
    input: []const f32,
    out_len: usize,
    recycle: ?mbuffer.Recycle,
    scratch: ?Scratch,
) !Bufs {
    // Reuse caller-provided (pool) buffers for input + norm when given. Resident
    // scratch already holds the final feature in s.input (no host copy exists);
    // otherwise upload the host feature into the borrowed input buffer and drop
    // the host copy at once.
    if (scratch) |s| {
        if (!s.resident) {
            c.zdraw_metal_write_buffer(s.input, std.mem.sliceAsBytes(input).ptr, input.len * @sizeOf(f32));
            if (recycle) |r| r.allocator.free(r.input);
        }
        const out_buf = try empty(ctx, out_len);
        return .{
            .input = .{ .handle = s.input, .borrowed = true },
            .norm = .{ .handle = s.norm, .borrowed = true },
            .output = out_buf,
        };
    }
    var input_buf = try mbuffer.Buffer.fromInput(ctx.device, input, recycle);
    errdefer input_buf.deinit();
    var norm_buf = try empty(ctx, input.len);
    errdefer norm_buf.deinit();
    const out_buf = try empty(ctx, out_len);
    return .{
        .input = Slot.fromOwned(input_buf),
        .norm = Slot.fromOwned(norm_buf),
        .output = out_buf,
    };
}

fn bufferSet(set: Bufs, binds: Binds) Buffers {
    return .{
        .input = set.input.handle,
        .norm = set.norm.handle,
        .norm_w = binds.norm_w.handle,
        .norm_b = binds.norm_b.handle,
        .conv_w = binds.conv_w.handle,
        .conv_b = binds.conv_b.handle,
        .output = set.output.handle,
    };
}

fn convCfg(cfg: Config) conv.Config {
    return .{
        .in_ch = cfg.channels,
        .out_ch = 3,
        .height = cfg.height,
        .width = cfg.width,
        .kernel = 3,
        .pad = 1,
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

const Binds = struct {
    norm_w: mbuffer.Bind,
    norm_b: mbuffer.Bind,
    conv_w: mbuffer.Bind,
    conv_b: mbuffer.Bind,
};

// input/norm may be owned (allocated here) or borrowed (a caller pool buffer);
// borrowed slots are freed by their owner, so deinit skips them. output is always
// owned here (it is read back then dropped).
const Slot = struct {
    handle: *anyopaque,
    borrowed: bool = false,
    owned: ?mbuffer.Buffer = null,

    fn fromOwned(buf: mbuffer.Buffer) Slot {
        return .{ .handle = buf.handle, .owned = buf };
    }

    fn deinit(self: *Slot) void {
        if (!self.borrowed) if (self.owned) |*b| b.deinit();
    }
};

const Bufs = struct {
    input: Slot,
    norm: Slot,
    output: mbuffer.Buffer,

    fn deinit(self: *Bufs) void {
        self.output.deinit();
        self.norm.deinit();
        self.input.deinit();
    }
};

const Temps = struct {
    norm_w: ?mbuffer.Buffer = null,
    norm_b: ?mbuffer.Buffer = null,
    conv_w: ?mbuffer.Buffer = null,
    conv_b: ?mbuffer.Buffer = null,

    fn deinit(self: *Temps) void {
        if (self.conv_b) |*buf| buf.deinit();
        if (self.conv_w) |*buf| buf.deinit();
        if (self.norm_b) |*buf| buf.deinit();
        if (self.norm_w) |*buf| buf.deinit();
    }
};
