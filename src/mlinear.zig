//! Optional Metal linear projection.

const std = @import("std");
const mvattn_owned = @import("mvattn_owned.zig");

const c = @import("metal_c.zig");
const gmode = @import("gemm_mode.zig");
const mbuffer = @import("mbuffer.zig");
const mchain_pool = @import("mchain_pool.zig");
const mblock_shader = @import("mblock_shader.zig");
const mfinal_shader = @import("mfinal_shader.zig");
const mshader = @import("mshader.zig");
const mpipe = @import("mpipe.zig");
const mres_util = @import("mres_util.zig");
const mswiglu = @import("mswiglu_shader.zig");
const tensor = @import("tensor.zig");
const token_selection_runtime = @import("token_selection_runtime.zig");
const toma_runtime = @import("toma_runtime.zig");

const entry = "linear_rows";

pub const Context = struct {
    device: *anyopaque,
    queue: *anyopaque,
    pipeline: *anyopaque,
    gemm_exact_pipeline: ?*anyopaque,
    gemm_half_pipeline: ?*anyopaque,
    gemm_w8_pipeline: ?*anyopaque,
    gemm_bias_pipeline: ?*anyopaque,
    swiglu_pipeline: ?*anyopaque,
    swiglu_fused_pipeline: ?*anyopaque,
    norm_pipeline: ?*anyopaque,
    resid_pipeline: ?*anyopaque,
    resid_norm_pipeline: ?*anyopaque,
    final_pipeline: ?*anyopaque,
    gemm_mode: gmode.Mode,
    buffers: mbuffer.Cache,
    // Persistent activation scratch for the resident chains: reused every
    // step/run so freed-buffer GPU wiring cannot accrue (see mchain_pool).
    pool: mchain_pool.Pool = .{},
    token_selection: ?token_selection_runtime.Runtime = null,
    toma: ?toma_runtime.Runtime = null,
    threads: usize,
    swiglu_threads: usize,
    block_threads: usize,
    final_threads: usize,

    pub fn init() !Context {
        const device = c.zdraw_metal_create_device() orelse return error.MetalNotAvailable;
        errdefer c.zdraw_metal_release_device(device);
        const queue = c.zdraw_metal_create_queue(device) orelse return error.MetalQueueFailed;
        errdefer c.zdraw_metal_release_queue(queue);
        var err: [1024]u8 = undefined;
        const pipe = try mpipe.required(device, mshader.linear.ptr, entry, &err);
        errdefer c.zdraw_metal_release_pipeline(pipe);
        var buffers = try mbuffer.Cache.init(device);
        errdefer buffers.deinit();
        const gemm_exact_pipe = gmode.compile(device, "gemm_exact", &err);
        const gemm_half_pipe = gmode.compile(device, "gemm_half", &err);
        const gemm_w8_pipe = gmode.compileW8(device, &err);
        const sw = mswiglu.swiglu.ptr;
        const swiglu_pipe = mpipe.optional(device, sw, "swiglu_f32", &err);
        const norm_pipe = mpipe.optional(device, mblock_shader.block.ptr, "block_norm_scale", &err);
        const fp = mpipe.optional(device, mfinal_shader.final.ptr, "final_norm_scale", &err);
        const block = mblock_shader.block.ptr;
        return .{
            .device = device,
            .queue = queue,
            .pipeline = pipe,
            .gemm_exact_pipeline = gemm_exact_pipe,
            .gemm_half_pipeline = gemm_half_pipe,
            .gemm_w8_pipeline = gemm_w8_pipe,
            .gemm_bias_pipeline = gmode.compile(device, "gemm_bias", &err),
            .swiglu_pipeline = swiglu_pipe,
            .swiglu_fused_pipeline = mpipe.optional(device, sw, "swiglu_fused_f32", &err),
            .norm_pipeline = norm_pipe,
            .resid_pipeline = mpipe.optional(device, block, "block_residual_norm", &err),
            .resid_norm_pipeline = mpipe.optional(device, block, "block_residual_next_norm", &err),
            .final_pipeline = fp,
            .gemm_mode = gmode.resolve(gmode.fromEnv(), gemm_exact_pipe, gemm_half_pipe),
            .buffers = buffers,
            .threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(pipe)),
            .swiglu_threads = mpipe.optionalThreads(swiglu_pipe),
            .block_threads = mpipe.optionalThreads(norm_pipe),
            .final_threads = mpipe.optionalThreads(fp),
        };
    }

    pub fn deinit(self: *Context) void {
        mvattn_owned.deinit();
        if (self.token_selection) |*runtime| runtime.deinit();
        if (self.toma) |*runtime| runtime.deinit();
        self.pool.deinit();
        self.buffers.deinit();
        if (self.resid_norm_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.resid_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.final_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.norm_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.swiglu_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.swiglu_fused_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.gemm_bias_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.gemm_w8_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.gemm_exact_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        if (self.gemm_half_pipeline) |p| c.zdraw_metal_release_pipeline(p);
        c.zdraw_metal_release_pipeline(self.pipeline);
        c.zdraw_metal_release_queue(self.queue);
        c.zdraw_metal_release_device(self.device);
        self.* = undefined;
    }

    pub fn clearWCache(self: *Context) void {
        self.buffers.clearSources();
    }

    pub fn tomaRuntime(self: *Context) !*toma_runtime.Runtime {
        if (self.toma == null) {
            self.toma = try toma_runtime.Runtime.initBorrowed(self.device, self.queue);
        }
        return &self.toma.?;
    }

    pub fn selectionRt(self: *Context) !*token_selection_runtime.Runtime {
        if (self.token_selection == null) {
            self.token_selection = try token_selection_runtime.Runtime.initBorrowed(self.device);
        }
        return &self.token_selection.?;
    }

    pub fn linear(
        self: *Context,
        out: []f32,
        input: []const f32,
        weight: tensor.View,
        bias: ?tensor.View,
    ) !void {
        try self.linearBatch(out, input, weight, bias, 1);
    }

    pub fn linearBatch(
        self: *Context,
        out: []f32,
        input: []const f32,
        weight: tensor.View,
        bias: ?tensor.View,
        batch: usize,
    ) !void {
        const dims = try shape(weight);
        if (batch == 0) return error.InvalidShape;
        if (out.len != dims.rows * batch or input.len != dims.cols * batch) {
            return error.InvalidShape;
        }
        try weight.check();
        if (bias) |b| try checkBias(b, dims.rows);

        var in_buf = try mbuffer.Buffer.fromBytes(self.device, std.mem.sliceAsBytes(input));
        defer in_buf.deinit();
        var out_buf = try mbuffer.Buffer.empty(self.device, std.mem.sliceAsBytes(out).len);
        defer out_buf.deinit();

        var weight_tmp: ?mbuffer.Buffer = null;
        defer if (weight_tmp) |*buf| buf.deinit();
        var bias_tmp: ?mbuffer.Buffer = null;
        defer if (bias_tmp) |*buf| buf.deinit();

        const weight_bind = try self.buffers.bindView(weight, &weight_tmp);
        const bias_bind = try self.buffers.bindBias(bias, &bias_tmp);
        const params = try paramsFor(weight, bias, dims, batch, weight_bind, bias_bind);
        try self.dispatch(in_buf, out_buf, weight_bind, bias_bind, params);
        c.zdraw_metal_read_buffer(out_buf.handle, std.mem.sliceAsBytes(out).ptr, bytesF32(out.len));
    }

    fn dispatch(
        self: *Context,
        input: mbuffer.Buffer,
        output: mbuffer.Buffer,
        weight: mbuffer.Bind,
        bias: mbuffer.Bind,
        params: c.LinearParams,
    ) !void {
        const code = c.zdraw_metal_run_linear(
            self.queue,
            self.pipeline,
            input.handle,
            weight.handle,
            bias.handle,
            output.handle,
            &params,
            self.threads,
        );
        if (code != 0) return error.MetalDispatchFailed;
    }
};

const Shape = struct {
    rows: usize,
    cols: usize,
};

pub fn shape(weight: tensor.View) !Shape {
    if (weight.shape.len != 2) return error.InvalidShape;
    if (weight.shape[0] == 0 or weight.shape[1] == 0) return error.InvalidShape;
    return .{ .rows = weight.shape[0], .cols = weight.shape[1] };
}

fn checkBias(bias: tensor.View, rows: usize) !void {
    if (try bias.elems() != rows) return error.InvalidShape;
    try bias.check();
}

fn paramsFor(
    weight_view: tensor.View,
    bias_view: ?tensor.View,
    dims: Shape,
    batch: usize,
    weight_bind: mbuffer.Bind,
    bias_bind: mbuffer.Bind,
) !c.LinearParams {
    return .{
        .rows = try mres_util.toU32(dims.rows),
        .cols = try mres_util.toU32(dims.cols),
        .batch = try mres_util.toU32(batch),
        .dtype = try mres_util.dtype(weight_view.dtype),
        .bias_dtype = if (bias_view) |b| try mres_util.dtype(b.dtype) else 1,
        .has_bias = if (bias_view == null) 0 else 1,
        .pad = 0,
        .weight_offset = try u64Fit(weight_bind.offset),
        .bias_offset = try u64Fit(bias_bind.offset),
    };
}

fn bytesF32(count: usize) usize {
    return count * @sizeOf(f32);
}

pub fn u64Fit(value: usize) !u64 {
    return @intCast(value);
}
