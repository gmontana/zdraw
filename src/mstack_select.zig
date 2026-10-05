//! Execute a searched gather/block/scatter program inside the resident stack.
//!
//! Full and compact states are pool-owned. The selected native blocks and the
//! final projection share one Metal command buffer; no activation crosses the
//! CPU boundary until the final image projection is read back.

const std = @import("std");

const bufs = @import("mblock_chain_buf.zig");
const chain_c = @import("mblock_chain_c.zig");
const final_c = @import("mstack_final_c.zig");
const mattn = @import("mattn.zig");
const metal_c = @import("metal_c.zig");
const mlinear = @import("mlinear.zig");
const mmod = @import("mmod_batch.zig");
const mstack = @import("mstack_chain.zig");
const mstack_final = @import("mstack_final.zig");
const policy = @import("mstack_policy.zig");
const token_selection = @import("token_selection_control.zig");
const token_runtime = @import("token_selection_runtime.zig");
const zblock = @import("zblock.zig");
const zrope = @import("zrope.zig");

pub fn run(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    out: []f32,
    state: []const f32,
    views: []const zblock.Views,
    adaln: []const f32,
    pos: []const zrope.Pos,
    cfg: mstack.Config,
    rope: zrope.Cache,
    final: mstack_final.Final,
    plan: token_selection.Plan,
) !void {
    try validate(state, views, pos, cfg, plan);
    const stack_policy = policy.fromEnv();
    const select = policy.selectFromEnv();
    const full_pipes =
        try mstack.getPipes(metal, attn, stack_policy, attn.pick(cfg.tokens, cfg.head_dim));
    var full_bufs = try bufs.make(metal, state, views[0], cfg);
    defer full_bufs.deinit();
    var modulations = try mmod.run(allocator, metal, adaln, views);
    defer modulations.deinit(allocator);
    var full_set = try mstack.makeSetFromMods(
        allocator,
        metal,
        views,
        &modulations,
        pos,
        cfg,
        rope,
        stack_policy,
        select,
        .main,
        .{ .total = views.len },
    );
    defer full_set.deinit(allocator);

    var selected = try selectViews(allocator, views, plan.layers);
    defer selected.deinit(allocator);
    const compact_state = try metal.pool.handle(
        metal.device,
        .token_selection_state,
        try activationBytes(plan.selected_tokens, plan.feature_width),
    );
    const compact_pos = try metal.pool.handle(
        metal.device,
        .token_selection_pos,
        try std.math.mul(usize, plan.selected_tokens, @sizeOf(zrope.Pos)),
    );
    var compact_cfg = cfg;
    compact_cfg.tokens = plan.selected_tokens;
    const compact_pipes = try mstack.getPipes(
        metal,
        attn,
        stack_policy,
        attn.pick(plan.selected_tokens, cfg.head_dim),
    );
    var compact_set = try mstack.makeSetGeoMods(
        allocator,
        metal,
        selected.views,
        &modulations,
        compact_cfg,
        rope,
        stack_policy,
        select,
        .main,
        .{ .total = views.len, .indices = selected.indices },
        .{ .pos = compact_pos, .rope = full_set.cweights[0].rope },
    );
    defer compact_set.deinit(allocator);
    var compact_bufs = full_bufs.c;
    compact_bufs.state = compact_state;

    const scores = try metal.pool.handle(
        metal.device,
        .token_selection_scores,
        try scoreBytes(plan),
    );
    const indices = try metal.pool.handle(
        metal.device,
        .token_selection_indices,
        try std.math.mul(usize, plan.selected_tokens, @sizeOf(u32)),
    );
    const reconstruction = try ReconstructionBuffers.init(metal, plan);
    const runtime = try metal.selectionRt();
    try runtime.validate(plan);
    var prepared_final =
        try mstack_final.prepareFinal(metal, out.len, state.len, final);
    defer prepared_final.deinit();
    const full_stack = EncodedStack{
        .pipes = full_pipes,
        .buffers = &full_bufs.c,
        .set = &full_set,
        .threads = threads(metal, attn, full_pipes),
    };
    const compact_stack = EncodedStack{
        .pipes = compact_pipes,
        .buffers = &compact_bufs,
        .set = &compact_set,
        .threads = threads(metal, attn, compact_pipes),
    };
    try dispatch(
        metal,
        runtime,
        full_stack,
        compact_stack,
        selected.indices,
        &prepared_final,
        scores,
        indices,
        compact_state,
        compact_pos,
        reconstruction,
        plan,
    );
    mstack_final.readOutput(prepared_final.output, out);
}

const EncodedStack = struct {
    pipes: mstack.Pipes,
    buffers: *chain_c.Buffers,
    set: *mstack.Set,
    threads: chain_c.Threads,
};

const ReconstructionBuffers = union(enum) {
    reference_scatter,
    cosine_residual: struct {
        normalized_source: *anyopaque,
        normalized_destination: *anyopaque,
        assignments: *anyopaque,
        inverse: *anyopaque,
    },

    fn init(
        metal: *mlinear.Context,
        plan: token_selection.Plan,
    ) !ReconstructionBuffers {
        return switch (plan.reconstruction) {
            .reference_scatter => .reference_scatter,
            .cosine_residual_interpolate => .{ .cosine_residual = .{
                .normalized_source = try metal.pool.handle(
                    metal.device,
                    .token_selection_norm_source,
                    try activationBytes(plan.source_tokens, plan.feature_width),
                ),
                .normalized_destination = try metal.pool.handle(
                    metal.device,
                    .token_selection_norm_destination,
                    try activationBytes(plan.selected_tokens, plan.feature_width),
                ),
                .assignments = try metal.pool.handle(
                    metal.device,
                    .token_selection_assignments,
                    try std.math.mul(usize, plan.source_tokens, @sizeOf(u32)),
                ),
                .inverse = try metal.pool.handle(
                    metal.device,
                    .token_selection_inverse,
                    try std.math.mul(usize, plan.source_tokens, @sizeOf(u32)),
                ),
            } },
        };
    }
};

fn dispatch(
    metal: *mlinear.Context,
    runtime: *token_runtime.Runtime,
    full: EncodedStack,
    compact: EncodedStack,
    selected_layers: []const usize,
    final: *mstack_final.FinalSet,
    scores: *anyopaque,
    indices: *anyopaque,
    compact_state: *anyopaque,
    compact_pos: *anyopaque,
    reconstruction: ReconstructionBuffers,
    plan: token_selection.Plan,
) !void {
    const batch = metal_c.zdraw_metal_batch_begin(metal.queue) orelse
        return error.MetalDispatchFailed;
    var batch_open = true;
    errdefer if (batch_open) {
        _ = metal_c.zdraw_metal_batch_end(batch);
    };
    var selected_index: usize = 0;
    for (0..full.set.filled) |layer| {
        if (selected_index < selected_layers.len and
            selected_layers[selected_index] == layer)
        {
            try runtime.encodeSelect(
                batch,
                full.buffers.state,
                scores,
                indices,
                compact_state,
                full.set.cweights[layer].pos,
                compact_pos,
                plan,
            );
            try encodeLayer(
                batch,
                compact.pipes,
                compact.buffers,
                &compact.set.cweights[selected_index],
                &compact.set.params[selected_index],
                &compact.threads,
                metal.swiglu_fused_pipeline,
            );
            switch (reconstruction) {
                .reference_scatter => try runtime.encodeScatter(
                    batch,
                    compact_state,
                    indices,
                    full.buffers.state,
                    plan,
                ),
                .cosine_residual => |buffers| {
                    try runtime.encodeCosResid(
                        batch,
                        compact_state,
                        indices,
                        full.buffers.state,
                        scores,
                        buffers.normalized_source,
                        buffers.normalized_destination,
                        buffers.assignments,
                        buffers.inverse,
                        plan,
                    );
                },
            }
            selected_index += 1;
        } else {
            try encodeLayer(
                batch,
                full.pipes,
                full.buffers,
                &full.set.cweights[layer],
                &full.set.params[layer],
                &full.threads,
                metal.swiglu_fused_pipeline,
            );
        }
    }
    if (selected_index != selected_layers.len) return error.SelectionLayerMismatch;
    if (final_c.zdraw_metal_encode_final(
        batch,
        metal.final_pipeline orelse return error.GemmUnavailable,
        full.pipes.gemm_exact,
        metal.gemm_bias_pipeline orelse return error.GemmUnavailable,
        full.buffers.state,
        &final.c,
        &final.params.norm,
        &final.params.gemm,
        &final.params.bias,
        metal.final_threads,
    ) != 0) return error.MetalDispatchFailed;
    const result = metal_c.zdraw_metal_batch_end(batch);
    batch_open = false;
    if (result != 0) return error.MetalDispatchFailed;
}

fn encodeLayer(
    batch: *anyopaque,
    pipes: mstack.Pipes,
    buffers: *const chain_c.Buffers,
    weights: *const chain_c.Weights,
    params: *const chain_c.Params,
    block_threads: *const chain_c.Threads,
    swiglu_fused: ?*anyopaque,
) !void {
    if (chain_c.zdraw_metal_encode_chain_layer(
        batch,
        pipes.norm,
        pipes.resid,
        pipes.gemm_exact,
        pipes.gemm_half,
        pipes.gemm_w8,
        pipes.qk,
        pipes.attn,
        pipes.swiglu,
        swiglu_fused,
        pipes.resid_norm,
        buffers,
        weights,
        params,
        block_threads,
    ) != 0) return error.MetalDispatchFailed;
}

fn threads(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    pipes: mstack.Pipes,
) chain_c.Threads {
    return .{
        .block = metal.block_threads,
        .qk = attn.qk_threads,
        .attn = pipes.attn_threads,
        .attn_kernel = pipes.attn_kernel,
        .swiglu = metal.swiglu_threads,
    };
}

const SelectedViews = struct {
    views: []zblock.Views,
    indices: []usize,

    fn deinit(self: *SelectedViews, allocator: std.mem.Allocator) void {
        allocator.free(self.indices);
        allocator.free(self.views);
        self.* = undefined;
    }
};

fn selectViews(
    allocator: std.mem.Allocator,
    views: []const zblock.Views,
    layers: []const u32,
) !SelectedViews {
    const selected_views = try allocator.alloc(zblock.Views, layers.len);
    errdefer allocator.free(selected_views);
    const indices = try allocator.alloc(usize, layers.len);
    for (layers, 0..) |layer, index| {
        if (layer >= views.len) return error.SelectionLayerMismatch;
        selected_views[index] = views[layer];
        indices[index] = layer;
    }
    return .{ .views = selected_views, .indices = indices };
}

fn validate(
    state: []const f32,
    views: []const zblock.Views,
    pos: []const zrope.Pos,
    cfg: mstack.Config,
    plan: token_selection.Plan,
) !void {
    if (views.len == 0 or
        cfg.tokens != plan.source_tokens or
        cfg.hidden != plan.feature_width or
        state.len != cfg.tokens * cfg.hidden or
        pos.len != cfg.tokens or
        plan.layers.len == 0)
    {
        return error.TokenSelectionShapeMismatch;
    }
    var prior: ?u32 = null;
    for (plan.layers) |layer| {
        if (layer >= views.len or (prior != null and prior.? >= layer)) {
            return error.SelectionLayerMismatch;
        }
        prior = layer;
    }
}

fn activationBytes(tokens: usize, width: usize) !usize {
    return std.math.mul(
        usize,
        try std.math.mul(usize, tokens, width),
        @sizeOf(f32),
    );
}

fn scoreBytes(plan: token_selection.Plan) !usize {
    const values = switch (plan.reconstruction) {
        .reference_scatter => plan.source_tokens,
        .cosine_residual_interpolate => try std.math.mul(
            usize,
            plan.source_tokens,
            plan.selected_tokens,
        ),
    };
    return std.math.mul(usize, values, @sizeOf(f32));
}

test "selection views preserve absolute layer identity" {
    const views = [_]zblock.Views{undefined} ** 4;
    var selected = try selectViews(std.testing.allocator, &views, &.{ 0, 3 });
    defer selected.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, selected.indices);
}
