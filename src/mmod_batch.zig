//! Batched AdaLN modulation projections for resident stack execution.

const std = @import("std");

const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("tensor.zig");
const zblock = @import("zblock.zig");
const zmod = @import("zmod.zig");

extern fn zdraw_metal_run_mods(
    queue: *anyopaque,
    pipeline: *anyopaque,
    input: *anyopaque,
    weights: [*]const *anyopaque,
    biases: [*]const *anyopaque,
    output: *anyopaque,
    params: [*]const c.LinearParams,
    count: usize,
    output_stride: usize,
    thread_count: usize,
) c_int;

pub const Batch = struct {
    values: []f32,
    rows: usize,

    pub fn deinit(self: *Batch, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        self.* = undefined;
    }

    pub fn parts(self: Batch, layer: usize) !zmod.Parts {
        const start = layer * self.rows;
        return zmod.split(self.values[start..][0..self.rows]);
    }
};

pub fn run(
    allocator: std.mem.Allocator,
    ctx: *mlinear.Context,
    input: []const f32,
    views: []const zblock.Views,
) !Batch {
    if (views.len == 0) return error.InvalidShape;
    const dims = try checkFirst(input, views[0]);
    const values = try allocator.alloc(f32, views.len * dims.rows);
    errdefer allocator.free(values);

    // Pooled (per-step churn otherwise accrues wiring): the result is read
    // back before this returns, so the slots are free for the next step.
    const in_buf = try ctx.pool.filled(ctx.device, .mod_in, std.mem.sliceAsBytes(input));
    const out_buf = try ctx.pool.handle(ctx.device, .mod_out, values.len * @sizeOf(f32));

    var set = try makeSet(allocator, ctx, views, dims);
    defer set.deinit(allocator);
    const code = zdraw_metal_run_mods(
        ctx.queue,
        ctx.pipeline,
        in_buf,
        set.weights.ptr,
        set.biases.ptr,
        out_buf,
        set.params.ptr,
        views.len,
        dims.rows * @sizeOf(f32),
        ctx.threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
    c.zdraw_metal_read_buffer(
        out_buf,
        std.mem.sliceAsBytes(values).ptr,
        values.len * @sizeOf(f32),
    );
    const batch = Batch{ .values = values, .rows = dims.rows };
    for (0..views.len) |idx| zmod.finish(try batch.parts(idx));
    return .{ .values = values, .rows = dims.rows };
}

const Shape = struct {
    rows: usize,
    cols: usize,
};

const Set = struct {
    weights: []*anyopaque,
    biases: []*anyopaque,
    params: []c.LinearParams,
    weight_temps: []?mbuffer.Buffer,
    bias_temps: []?mbuffer.Buffer,

    fn deinit(self: *Set, allocator: std.mem.Allocator) void {
        for (self.bias_temps) |*buf| if (buf.*) |*value| value.deinit();
        for (self.weight_temps) |*buf| if (buf.*) |*value| value.deinit();
        allocator.free(self.bias_temps);
        allocator.free(self.weight_temps);
        allocator.free(self.params);
        allocator.free(self.biases);
        allocator.free(self.weights);
    }
};

fn makeSet(
    allocator: std.mem.Allocator,
    ctx: *mlinear.Context,
    views: []const zblock.Views,
    dims: Shape,
) !Set {
    var set = try allocSet(allocator, views.len);
    errdefer set.deinit(allocator);
    for (views, 0..) |view, idx| {
        const weight = view.ada_w orelse return error.InvalidShape;
        const bias = view.ada_b orelse return error.InvalidShape;
        try checkShape(dims, weight, bias);
        const wb = try ctx.buffers.bindView(weight, &set.weight_temps[idx]);
        const bb = try ctx.buffers.bindBias(bias, &set.bias_temps[idx]);
        set.weights[idx] = wb.handle;
        set.biases[idx] = bb.handle;
        set.params[idx] = try params(weight, bias, dims, wb, bb);
    }
    return set;
}

fn allocSet(allocator: std.mem.Allocator, count: usize) !Set {
    const weights = try allocator.alloc(*anyopaque, count);
    errdefer allocator.free(weights);
    const biases = try allocator.alloc(*anyopaque, count);
    errdefer allocator.free(biases);
    const ps = try allocator.alloc(c.LinearParams, count);
    errdefer allocator.free(ps);
    const weight_temps = try allocator.alloc(?mbuffer.Buffer, count);
    errdefer allocator.free(weight_temps);
    const bias_temps = try allocator.alloc(?mbuffer.Buffer, count);
    @memset(weight_temps, null);
    @memset(bias_temps, null);
    return .{
        .weights = weights,
        .biases = biases,
        .params = ps,
        .weight_temps = weight_temps,
        .bias_temps = bias_temps,
    };
}

fn checkFirst(input: []const f32, view: zblock.Views) !Shape {
    const weight = view.ada_w orelse return error.InvalidShape;
    const bias = view.ada_b orelse return error.InvalidShape;
    const dims = try shape(weight);
    if (dims.cols != input.len or dims.rows % 4 != 0) return error.InvalidShape;
    try checkShape(dims, weight, bias);
    return dims;
}

fn checkShape(dims: Shape, weight: tensor.View, bias: tensor.View) !void {
    const got = try shape(weight);
    if (got.rows != dims.rows or got.cols != dims.cols) return error.InvalidShape;
    if (try bias.elems() != dims.rows) return error.InvalidShape;
    try weight.check();
    try bias.check();
}

fn shape(weight: tensor.View) !Shape {
    const dims = try mlinear.shape(weight);
    return .{ .rows = dims.rows, .cols = dims.cols };
}

fn params(
    weight: tensor.View,
    bias: tensor.View,
    dims: Shape,
    wb: mbuffer.Bind,
    bb: mbuffer.Bind,
) !c.LinearParams {
    return .{
        .rows = try mres_util.toU32(dims.rows),
        .cols = try mres_util.toU32(dims.cols),
        .batch = 1,
        .dtype = try mres_util.dtype(weight.dtype),
        .bias_dtype = try mres_util.dtype(bias.dtype),
        .has_bias = 1,
        .pad = 0,
        .weight_offset = try mlinear.u64Fit(wb.offset),
        .bias_offset = try mlinear.u64Fit(bb.offset),
    };
}
