//! Optional Metal 2D convolution for the VAE decoder.

const std = @import("std");

const conv = @import("conv.zig");
const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mpipe = @import("mpipe.zig");
const mres_util = @import("mres_util.zig");
const shader = @import("mconv_shader.zig");
const tensor = @import("tensor.zig");
const vshader = @import("mvnorm_shader.zig");

const entry = "conv2d";
const norm_entry = "vae_norm_silu";
const norm_h_entry = "vae_norm_silu_h";
const add_entry = "vae_add";
const up_entry = "vae_upsample2";
const attn_stats_entry = "vae_norm_stats";
const attn_apply_entry = "vae_norm_apply_seq";
const attn_scatter_entry = "vae_seq_scatter_add";

pub const Context = struct {
    device: *anyopaque,
    queue: *anyopaque,
    pipeline: *anyopaque,
    norm_pipeline: *anyopaque,
    norm_h_pipeline: *anyopaque,
    add_pipeline: *anyopaque,
    up_pipeline: *anyopaque,
    // Resident mid-attention (mvattn.zig): stats + no-SiLU apply/transpose +
    // scatter-add.
    attn_stats_pipeline: *anyopaque,
    attn_apply_pipeline: *anyopaque,
    attn_scatter_pipeline: *anyopaque,
    buffers: mbuffer.Cache,
    threads: usize,
    norm_threads: usize,
    add_threads: usize,
    up_threads: usize,
    attn_stats_threads: usize,
    attn_apply_threads: usize,
    attn_scatter_threads: usize,

    pub fn init() !Context {
        const device = c.zdraw_metal_create_device() orelse return error.MetalNotAvailable;
        errdefer c.zdraw_metal_release_device(device);
        const queue = c.zdraw_metal_create_queue(device) orelse return error.MetalQueueFailed;
        errdefer c.zdraw_metal_release_queue(queue);
        var err: [1024]u8 = undefined;
        const pipe = try mpipe.required(device, shader.conv.ptr, entry, &err);
        errdefer c.zdraw_metal_release_pipeline(pipe);
        const norm_pipe = try mpipe.required(device, vshader.vnorm.ptr, norm_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(norm_pipe);
        const norm_h_pipe = try mpipe.required(device, vshader.vnorm.ptr, norm_h_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(norm_h_pipe);
        const add_pipe = try mpipe.required(device, vshader.vnorm.ptr, add_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(add_pipe);
        const up_pipe = try mpipe.required(device, vshader.vnorm.ptr, up_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(up_pipe);
        const astats_pipe = try mpipe.required(device, vshader.vnorm.ptr, attn_stats_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(astats_pipe);
        const aapply_pipe = try mpipe.required(device, vshader.vnorm.ptr, attn_apply_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(aapply_pipe);
        const ascatter_pipe =
            try mpipe.required(device, vshader.vnorm.ptr, attn_scatter_entry, &err);
        errdefer c.zdraw_metal_release_pipeline(ascatter_pipe);
        var buffers = try mbuffer.Cache.init(device);
        errdefer buffers.deinit();
        return .{
            .device = device,
            .queue = queue,
            .pipeline = pipe,
            .norm_pipeline = norm_pipe,
            .norm_h_pipeline = norm_h_pipe,
            .add_pipeline = add_pipe,
            .up_pipeline = up_pipe,
            .attn_stats_pipeline = astats_pipe,
            .attn_apply_pipeline = aapply_pipe,
            .attn_scatter_pipeline = ascatter_pipe,
            .buffers = buffers,
            .threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(pipe)),
            .norm_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(norm_pipe)),
            .add_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(add_pipe)),
            .up_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(up_pipe)),
            .attn_stats_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(astats_pipe)),
            .attn_apply_threads = mpipe.threadCount(c.zdraw_metal_pipeline_threads(aapply_pipe)),
            .attn_scatter_threads = mpipe.threadCount(
                c.zdraw_metal_pipeline_threads(ascatter_pipe),
            ),
        };
    }

    pub fn deinit(self: *Context) void {
        self.buffers.deinit();
        c.zdraw_metal_clear_transient_caches();
        c.zdraw_metal_release_pipeline(self.attn_scatter_pipeline);
        c.zdraw_metal_release_pipeline(self.attn_apply_pipeline);
        c.zdraw_metal_release_pipeline(self.attn_stats_pipeline);
        c.zdraw_metal_release_pipeline(self.up_pipeline);
        c.zdraw_metal_release_pipeline(self.add_pipeline);
        c.zdraw_metal_release_pipeline(self.norm_h_pipeline);
        c.zdraw_metal_release_pipeline(self.norm_pipeline);
        c.zdraw_metal_release_pipeline(self.pipeline);
        c.zdraw_metal_release_queue(self.queue);
        c.zdraw_metal_release_device(self.device);
        self.* = undefined;
    }

    pub fn run(
        self: *Context,
        out: []f32,
        input: []const f32,
        weight: tensor.View,
        bias: ?tensor.View,
        cfg: conv.Config,
    ) !void {
        try check(out, input, weight, bias, cfg);
        // One pool per conv: the transient in/out/weight buffers (512 MB each
        // at 1024x1024) must not wait for the request's pool to drain.
        const pool = c.zdraw_metal_pool_push();
        defer c.zdraw_metal_pool_pop(pool);
        var in_buf = try mbuffer.Buffer.fromBytes(self.device, std.mem.sliceAsBytes(input));
        defer in_buf.deinit();
        var out_buf = try mbuffer.Buffer.empty(self.device, std.mem.sliceAsBytes(out).len);
        defer out_buf.deinit();
        try self.runBound(in_buf, out_buf, weight, bias, cfg);
        c.zdraw_metal_read_buffer(out_buf.handle, std.mem.sliceAsBytes(out).ptr, bytesF32(out.len));
    }

    pub fn runBound(
        self: *Context,
        input: mbuffer.Buffer,
        output: mbuffer.Buffer,
        weight: tensor.View,
        bias: ?tensor.View,
        cfg: conv.Config,
    ) !void {
        var weight_tmp: ?mbuffer.Buffer = null;
        defer if (weight_tmp) |*buf| buf.deinit();
        var bias_tmp: ?mbuffer.Buffer = null;
        defer if (bias_tmp) |*buf| buf.deinit();
        const weight_bind = try self.buffers.bindView(weight, &weight_tmp);
        const bias_bind = try self.buffers.bindBias(bias, &bias_tmp);
        const params = try paramsFor(weight, bias, cfg, weight_bind, bias_bind);
        const code = c.zdraw_metal_run_conv(
            self.queue,
            self.pipeline,
            input.handle,
            weight_bind.handle,
            bias_bind.handle,
            output.handle,
            &params,
            self.threads,
        );
        if (code != 0) return error.MetalDispatchFailed;
    }
};

pub fn check(
    out: []const f32,
    input: []const f32,
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: conv.Config,
) !void {
    if (out.len != cfg.out_ch * cfg.height * cfg.width) return error.InvalidShape;
    if (input.len != cfg.in_ch * cfg.height * cfg.width) return error.InvalidShape;
    try checkWeight(weight, bias, cfg);
}

// Weight/bias-vs-config validation alone, for callers whose buffer lengths are
// derived from cfg by construction (no host slices to compare).
pub fn checkWeight(weight: tensor.View, bias: ?tensor.View, cfg: conv.Config) !void {
    if (weight.shape.len != 4) return error.InvalidShape;
    if (weight.shape[0] != cfg.out_ch or weight.shape[1] != cfg.in_ch) {
        return error.InvalidShape;
    }
    if (weight.shape[2] != cfg.kernel or weight.shape[3] != cfg.kernel) {
        return error.InvalidShape;
    }
    try weight.check();
    if (bias) |b| try checkBias(b, cfg.out_ch);
}

fn checkBias(bias: tensor.View, out_ch: usize) !void {
    if (try bias.elems() != out_ch) return error.InvalidShape;
    try bias.check();
}

pub fn paramsFor(
    weight: tensor.View,
    bias: ?tensor.View,
    cfg: conv.Config,
    weight_bind: mbuffer.Bind,
    bias_bind: mbuffer.Bind,
) !c.ConvParams {
    return .{
        .in_ch = try mres_util.toU32(cfg.in_ch),
        .out_ch = try mres_util.toU32(cfg.out_ch),
        .height = try mres_util.toU32(cfg.height),
        .width = try mres_util.toU32(cfg.width),
        .ksize = try mres_util.toU32(cfg.kernel),
        .pad = try mres_util.toU32(cfg.pad),
        .dtype = try mres_util.dtype(weight.dtype),
        .bias_dtype = if (bias) |b| try mres_util.dtype(b.dtype) else 1,
        .has_bias = if (bias == null) 0 else 1,
        .weight_offset = try u64Fit(weight_bind.offset),
        .bias_offset = try u64Fit(bias_bind.offset),
    };
}

fn bytesF32(count: usize) usize {
    return count * @sizeOf(f32);
}

fn u64Fit(value: usize) !u64 {
    return @intCast(value);
}
