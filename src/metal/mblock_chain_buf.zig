//! Activation buffers for chained resident block execution.
//!
//! Drawn from the Metal context's persistent pool (mchain_pool), not allocated
//! per call: freed MTLBuffers never return their GPU wiring on macOS, so a
//! fresh set per step grew phys by ~2.3 GB/step at 1024px. The pool slot is
//! rewritten every use (state is uploaded, every other buffer is fully written
//! by its producing kernel before any read), so reuse is value-identical.

const std = @import("std");

const chain_c = @import("mblock_chain_c.zig");
const mlinear = @import("mlinear.zig");
const types = @import("mblock_chain_types.zig");
const zblock = @import("../zimage/zblock.zig");

pub const Bufs = struct {
    c: chain_c.Buffers,

    // Pool-owned buffers: nothing to release per run.
    pub fn deinit(self: *Bufs) void {
        _ = self;
    }
};

pub fn make(
    metal: *mlinear.Context,
    state: []const f32,
    views: zblock.Views,
    cfg: types.Config,
) !Bufs {
    const inner = (try mlinear.shape(views.ffn_gate)).rows;
    const dev = metal.device;
    const pool = &metal.pool;
    const state_bytes = state.len * @sizeOf(f32);
    const state_handle = try pool.filled(dev, .state, std.mem.sliceAsBytes(state));
    return makeWithState(metal, state_handle, cfg, state_bytes, inner);
}

fn makeWithState(
    metal: *mlinear.Context,
    state: *anyopaque,
    cfg: types.Config,
    state_bytes: usize,
    inner: usize,
) !Bufs {
    const dev = metal.device;
    const pool = &metal.pool;
    const inner_bytes = cfg.tokens * inner * @sizeOf(f32);
    return .{ .c = .{
        .state = state,
        .norm = try pool.handle(dev, .norm, state_bytes),
        .attn = try pool.handle(dev, .attn, state_bytes),
        .q = try pool.handle(dev, .q, state_bytes),
        .k = try pool.handle(dev, .k, state_bytes),
        .v = try pool.handle(dev, .v, state_bytes),
        .mix = try pool.handle(dev, .mix, state_bytes),
        .ffn = try pool.handle(dev, .ffn, state_bytes),
        .gate = try pool.handle(dev, .gate, inner_bytes),
        .up = try pool.handle(dev, .up, inner_bytes),
        .gateup = try pool.handle(dev, .gateup, inner_bytes * 2),
    } };
}
