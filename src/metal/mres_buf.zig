//! Activation buffers for resident Metal attention.
//!
//! Borrowed from the persistent chain pool (mchain_pool): the standalone
//! attention runs while the denoise chain is idle, every buffer is fully
//! written before it is read, and freed MTLBuffers never return their GPU
//! wiring on macOS — so per-call allocation here accrued ~0.2 GB per decode.

const std = @import("std");

const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");

pub const Set = struct {
    input: mbuffer.Buffer,
    q: mbuffer.Buffer,
    k: mbuffer.Buffer,
    v: mbuffer.Buffer,
    mix: mbuffer.Buffer,
    out: mbuffer.Buffer,

    // Pool-owned buffers: nothing to release per run.
    pub fn deinit(self: *Set) void {
        _ = self;
    }
};

pub fn init(
    metal: *mlinear.Context,
    input: []const f32,
    out_len: usize,
    tokens: usize,
    hidden: usize,
) !Set {
    const dev = metal.device;
    const pool = &metal.pool;
    const bytes = tokens * hidden * @sizeOf(f32);
    return .{
        .input = mbuffer.Buffer.borrow(
            try pool.filled(dev, .norm, std.mem.sliceAsBytes(input)),
        ),
        .q = mbuffer.Buffer.borrow(try pool.handle(dev, .q, bytes)),
        .k = mbuffer.Buffer.borrow(try pool.handle(dev, .k, bytes)),
        .v = mbuffer.Buffer.borrow(try pool.handle(dev, .v, bytes)),
        .mix = mbuffer.Buffer.borrow(try pool.handle(dev, .mix, bytes)),
        .out = mbuffer.Buffer.borrow(
            try pool.handle(dev, .attn, out_len * @sizeOf(f32)),
        ),
    };
}
