//! Residual-safe GPU-resident ToMA execution for a single-stream DiT stack.
//!
//! The full-token residual stream never crosses a merge boundary. Attention
//! and FFN inputs are merged independently, their expensive cores run at the
//! reduced sequence length, and each module result is unmerged before its
//! full-token residual update. One pattern is built from the first normalized
//! layer input and reused only across the supplied compatible block family.

const std = @import("std");

const bufs = @import("mblock_chain_buf.zig");
const chain_c = @import("mblock_chain_c.zig");
const chain_params = @import("mblock_chain_param.zig");
const mattn = @import("mattn.zig");
const metal_c = @import("metal_c.zig");
const mlinear = @import("mlinear.zig");
const mstack = @import("mstack_chain.zig");
const policy = @import("mstack_policy.zig");
const toma_config = @import("toma_config.zig");
const toma_metal = @import("toma_metal.zig");
const toma_runtime = @import("toma_runtime.zig");
const zblock = @import("zblock.zig");
const zpack_file = @import("zpack_file.zig");
const zrope = @import("zrope.zig");

pub const ModalShape = struct {
    image_tokens: usize,
    caption_tokens: usize,
};

pub fn run(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    state: []f32,
    views: []const zblock.Views,
    adaln: []const f32,
    pos: []const zrope.Pos,
    cfg: mstack.Config,
    rope: zrope.Cache,
    family: zpack_file.Family,
    band: mstack.Band,
    modal: ModalShape,
    toma_cfg: toma_config.Config,
) !void {
    if (views.len == 0) return;
    try mstack.validateBand(band, views.len);
    try check(state, pos, cfg, modal, toma_cfg);
    const stack_policy = policy.fromEnv();
    const select = policy.selectFromEnv();
    const first_modes = policy.modes(
        metal.gemm_mode,
        stack_policy,
        select,
        band.index(0),
        band.total,
    );
    if (policy.allOff(first_modes)) return error.GemmUnavailable;

    var full_bufs = try bufs.make(metal, state, views[0], cfg);
    defer full_bufs.deinit();
    const full_pos = try metal.pool.filled(
        metal.device,
        .toma_full_pos,
        std.mem.sliceAsBytes(pos),
    );
    var reduced = try makeReduced(metal, full_bufs.c, cfg, modal, toma_cfg);
    const rope_handle = try metal.pool.filled(
        metal.device,
        .toma_rope,
        rope.pairBytes(),
    );
    const pipes = try mstack.getPipes(
        metal,
        attn,
        stack_policy,
        attn.pick(reduced.cfg.tokens, cfg.head_dim),
    );
    var set = try mstack.makeSetGeo(
        allocator,
        metal,
        views,
        adaln,
        reduced.cfg,
        rope,
        stack_policy,
        select,
        family,
        band,
        .{ .pos = reduced.pos, .rope = rope_handle },
    );
    defer set.deinit(allocator);
    const full_params = try allocator.alloc(chain_c.Params, set.filled);
    defer allocator.free(full_params);
    for (set.params[0..set.filled], full_params) |params, *full| {
        full.* = try chain_params.withTokens(params, cfg.tokens);
    }
    const threads = chain_c.Threads{
        .block = metal.block_threads,
        .qk = attn.qk_threads,
        .attn = pipes.attn_threads,
        .attn_kernel = pipes.attn_kernel,
        .swiglu = metal.swiglu_threads,
    };

    // ToMA selects destinations from the normalized input of the first block
    // in a compatible family, matching the released scheduler's reuse scope.
    try runFirstNorm(
        metal,
        pipes,
        &full_bufs.c,
        &set.cweights[0],
        &full_params[0],
        &threads,
    );
    const runtime = try metal.tomaRuntime();
    const pattern = try runtime.prepare(
        full_bufs.c.norm,
        modal.image_tokens,
        cfg.hidden,
        toma_cfg,
    );
    try encodeStack(.{
        .metal = metal,
        .pipes = pipes,
        .full = &full_bufs.c,
        .reduced = &reduced.buffers,
        .set = &set,
        .full_params = full_params,
        .threads = &threads,
        .runtime = runtime,
        .pattern = pattern,
        .full_pos = full_pos,
        .reduced_pos = reduced.pos,
        .caption_tokens = modal.caption_tokens,
        .hidden = cfg.hidden,
    });
    metal_c.zdraw_metal_read_buffer(
        full_bufs.c.state,
        std.mem.sliceAsBytes(state).ptr,
        std.mem.sliceAsBytes(state).len,
    );
}

const Reduced = struct {
    buffers: chain_c.Buffers,
    pos: *anyopaque,
    cfg: mstack.Config,
};

fn makeReduced(
    metal: *mlinear.Context,
    full: chain_c.Buffers,
    cfg: mstack.Config,
    modal: ModalShape,
    toma_cfg: toma_config.Config,
) !Reduced {
    const tokens = toma_cfg.destination_tokens + modal.caption_tokens;
    const bytes = tokens * cfg.hidden * @sizeOf(f32);
    var buffers = full;
    buffers.norm = try metal.pool.handle(metal.device, .toma_state, bytes);
    const module = try metal.pool.handle(metal.device, .toma_module, bytes);
    buffers.mix = module;
    buffers.ffn = module;
    var reduced_cfg = cfg;
    reduced_cfg.tokens = tokens;
    return .{
        .buffers = buffers,
        .pos = try metal.pool.handle(
            metal.device,
            .toma_reduced_pos,
            tokens * @sizeOf(zrope.Pos),
        ),
        .cfg = reduced_cfg,
    };
}

const Encoding = struct {
    metal: *mlinear.Context,
    pipes: mstack.Pipes,
    full: *chain_c.Buffers,
    reduced: *chain_c.Buffers,
    set: *mstack.Set,
    full_params: []const chain_c.Params,
    threads: *const chain_c.Threads,
    runtime: *toma_runtime.Runtime,
    pattern: *toma_metal.Pattern,
    full_pos: *anyopaque,
    reduced_pos: *anyopaque,
    caption_tokens: usize,
    hidden: usize,
};

fn encodeStack(encoding: Encoding) !void {
    const batch = metal_c.zdraw_metal_batch_begin(encoding.metal.queue) orelse
        return error.MetalDispatchFailed;
    var batch_open = true;
    errdefer if (batch_open) {
        _ = metal_c.zdraw_metal_batch_end(batch);
    };
    try encoding.runtime.context.encodePos(
        batch,
        encoding.pattern,
        encoding.full_pos,
        encoding.reduced_pos,
        encoding.caption_tokens,
    );
    for (0..encoding.set.filled) |layer| {
        try encodeAttention(encoding, batch, layer);
        try encodeFfn(encoding, batch, layer);
    }
    const result = metal_c.zdraw_metal_batch_end(batch);
    batch_open = false;
    if (result != 0) return error.MetalDispatchFailed;
}

fn encodeAttention(encoding: Encoding, batch: *anyopaque, layer: usize) !void {
    if (layer != 0 and chain_c.zdraw_metal_encode_toma_attn_norm(
        batch,
        encoding.pipes.norm,
        encoding.full,
        &encoding.set.cweights[layer],
        &encoding.full_params[layer],
        encoding.threads,
    ) != 0) return error.MetalDispatchFailed;
    try encoding.runtime.context.encodeMerge(
        batch,
        encoding.pattern,
        encoding.full.norm,
        encoding.reduced.norm,
        encoding.caption_tokens,
        encoding.hidden,
    );
    if (chain_c.zdraw_metal_encode_toma_attn_core(
        batch,
        encoding.pipes.gemm_exact,
        encoding.pipes.gemm_half,
        encoding.pipes.gemm_w8,
        encoding.pipes.qk,
        encoding.pipes.attn,
        encoding.reduced,
        &encoding.set.cweights[layer],
        &encoding.set.params[layer],
        encoding.threads,
    ) != 0) return error.MetalDispatchFailed;
    try encoding.runtime.context.encodeUnpack(
        batch,
        encoding.pattern,
        encoding.reduced.mix,
        encoding.full.mix,
        encoding.caption_tokens,
        encoding.hidden,
    );
    if (chain_c.zdraw_metal_encode_toma_attn_finish(
        batch,
        encoding.pipes.gemm_exact,
        encoding.pipes.gemm_half,
        encoding.pipes.gemm_w8,
        encoding.pipes.resid_norm,
        encoding.full,
        &encoding.set.cweights[layer],
        &encoding.full_params[layer],
        encoding.threads,
    ) != 0) return error.MetalDispatchFailed;
}

fn encodeFfn(encoding: Encoding, batch: *anyopaque, layer: usize) !void {
    try encoding.runtime.context.encodeMerge(
        batch,
        encoding.pattern,
        encoding.full.norm,
        encoding.reduced.norm,
        encoding.caption_tokens,
        encoding.hidden,
    );
    if (chain_c.zdraw_metal_encode_toma_ffn_core(
        batch,
        encoding.pipes.gemm_exact,
        encoding.pipes.gemm_half,
        encoding.pipes.gemm_w8,
        encoding.pipes.swiglu,
        encoding.metal.swiglu_fused_pipeline,
        encoding.reduced,
        &encoding.set.cweights[layer],
        &encoding.set.params[layer],
        encoding.threads,
    ) != 0) return error.MetalDispatchFailed;
    try encoding.runtime.context.encodeUnpack(
        batch,
        encoding.pattern,
        encoding.reduced.ffn,
        encoding.full.ffn,
        encoding.caption_tokens,
        encoding.hidden,
    );
    if (chain_c.zdraw_metal_encode_toma_ffn_finish(
        batch,
        encoding.pipes.resid,
        encoding.full,
        &encoding.set.cweights[layer],
        &encoding.full_params[layer],
        encoding.threads,
    ) != 0) return error.MetalDispatchFailed;
}

fn runFirstNorm(
    metal: *mlinear.Context,
    pipes: mstack.Pipes,
    buffers: *const chain_c.Buffers,
    weights: *const chain_c.Weights,
    params: *const chain_c.Params,
    threads: *const chain_c.Threads,
) !void {
    const batch = metal_c.zdraw_metal_batch_begin(metal.queue) orelse
        return error.MetalDispatchFailed;
    if (chain_c.zdraw_metal_encode_toma_attn_norm(
        batch,
        pipes.norm,
        buffers,
        weights,
        params,
        threads,
    ) != 0) {
        _ = metal_c.zdraw_metal_batch_end(batch);
        return error.MetalDispatchFailed;
    }
    if (metal_c.zdraw_metal_batch_end(batch) != 0) return error.MetalDispatchFailed;
}

fn check(
    state: []const f32,
    pos: []const zrope.Pos,
    cfg: mstack.Config,
    modal: ModalShape,
    toma_cfg: toma_config.Config,
) !void {
    if (modal.image_tokens == 0 or
        modal.image_tokens + modal.caption_tokens != cfg.tokens or
        state.len != cfg.tokens * cfg.hidden or
        pos.len != cfg.tokens or
        toma_cfg.destination_tokens >= modal.image_tokens)
    {
        return error.InvalidShape;
    }
    _ = try toma_config.shape(modal.image_tokens, cfg.hidden, toma_cfg);
}

test "modal shape must account for every token" {
    const cfg = mstack.Config{
        .tokens = 17,
        .hidden = 8,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 8,
        .norm_eps = 0.00001,
    };
    try std.testing.expectError(error.InvalidShape, check(
        &([_]f32{0} ** (17 * 8)),
        &([_]zrope.Pos{.{ 0, 0, 0 }} ** 17),
        cfg,
        .{ .image_tokens = 16, .caption_tokens = 0 },
        .{
            .mode = .paper_spec,
            .destination_tokens = 8,
            .region_count = 4,
            .selection_layout = .tile,
            .assignment_scope = .global,
            .unmerge = .paper_normalized_transpose,
            .assignment_scale = 1000,
        },
    ));
}
