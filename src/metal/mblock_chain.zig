//! One-command-buffer resident block execution.

const std = @import("std");

const c = @import("metal_c.zig");
const bufs = @import("mblock_chain_buf.zig");
const chain_c = @import("mblock_chain_c.zig");
const mattn = @import("mattn.zig");
const mlinear = @import("mlinear.zig");
const params = @import("mblock_chain_param.zig");
const types = @import("mblock_chain_types.zig");
const weights = @import("mblock_chain_weight.zig");
const zblock = @import("../zimage/zblock.zig");
const zmod = @import("../zimage/zmod.zig");
const zrope = @import("../zimage/zrope.zig");

pub const Config = types.Config;

const Pipes = struct {
    norm: *anyopaque,
    resid: *anyopaque,
    gemm_exact: *anyopaque,
    gemm_half: ?*anyopaque,
    gemm_w8: ?*anyopaque,
    qk: *anyopaque,
    attn: *anyopaque,
    attn_threads: usize,
    attn_kernel: usize,
    swiglu: *anyopaque,
    resid_norm: *anyopaque,
};

pub fn run(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    state: []f32,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
) !void {
    const pipes = try getPipes(metal, attn, attn.pick(cfg.tokens, cfg.head_dim));
    try params.check(state, pos, mods, cfg);
    var block_bufs = try bufs.make(metal, state, views, cfg);
    defer block_bufs.deinit();
    var block_weights = try weights.make(metal, views, mods, pos, rope);
    defer block_weights.deinit();
    const block_params = try params.make(
        metal.gemm_mode,
        block_weights.binds,
        views,
        mods,
        cfg,
        rope,
    );
    try dispatch(metal, attn, pipes, &block_bufs.c, &block_weights.c, &block_params);
    readBack(block_bufs.c.state, state);
}

fn dispatch(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    pipes: Pipes,
    buffers: *const chain_c.Buffers,
    block_weights: *const chain_c.Weights,
    block_params: *const chain_c.Params,
) !void {
    const threads = chain_c.Threads{
        .block = metal.block_threads,
        .qk = attn.qk_threads,
        .attn = pipes.attn_threads,
        .attn_kernel = pipes.attn_kernel,
        .swiglu = metal.swiglu_threads,
    };
    const code = chain_c.zdraw_metal_run_block_chain(
        metal.queue,
        pipes.norm,
        pipes.resid,
        pipes.gemm_exact,
        pipes.gemm_half,
        pipes.gemm_w8,
        pipes.qk,
        pipes.attn,
        pipes.swiglu,
        metal.swiglu_fused_pipeline,
        pipes.resid_norm,
        buffers,
        block_weights,
        block_params,
        &threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn getPipes(metal: *mlinear.Context, attn: *mattn.Context, pick: mattn.Context.Pick) !Pipes {
    if (metal.gemm_mode == .off) return error.GemmUnavailable;
    return .{
        .norm = metal.norm_pipeline orelse return error.GemmUnavailable,
        .resid = metal.resid_pipeline orelse return error.GemmUnavailable,
        .gemm_exact = metal.gemm_exact_pipeline orelse return error.GemmUnavailable,
        .gemm_half = metal.gemm_half_pipeline,
        .gemm_w8 = metal.gemm_w8_pipeline,
        .qk = attn.qk_pipeline orelse return error.GemmUnavailable,
        .attn = pick.pipeline,
        .attn_threads = pick.threads,
        .attn_kernel = @intFromEnum(pick.kernel),
        .swiglu = metal.swiglu_pipeline orelse return error.GemmUnavailable,
        .resid_norm = metal.resid_norm_pipeline orelse return error.GemmUnavailable,
    };
}

fn readBack(handle: *anyopaque, state: []f32) void {
    c.zdraw_metal_read_buffer(
        handle,
        std.mem.sliceAsBytes(state).ptr,
        state.len * @sizeOf(f32),
    );
}
