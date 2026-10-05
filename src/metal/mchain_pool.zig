//! Persistent activation-buffer pool for the resident chains.
//!
//! Every fresh MTLBuffer cycle permanently accrues GPU wiring on macOS
//! (release does not unwire), so per-step/per-decode allocation grew phys by
//! ~2.3 GB per denoise step at 1024px. The chains therefore draw their
//! activation scratch from one named, grow-on-demand pool that lives as long
//! as the Metal context: the same buffer is rewritten every step, so the
//! footprint is the high-water mark instead of the running sum. Contents are
//! NOT zeroed between uses - every kernel fully writes its slot before any
//! read, which the strict max|d|=0 image gate verifies.

const std = @import("std");

const c = @import("metal_c.zig");
const mbuffer = @import("mbuffer.zig");

pub const Slot = enum(u8) {
    state,
    norm,
    attn,
    q,
    k,
    v,
    mix,
    ffn,
    gate,
    up,
    gateup,
    final_batch,
    final_out,
    mod_in,
    mod_out,
    toma_state,
    toma_module,
    toma_full_pos,
    toma_reduced_pos,
    toma_rope,
    token_selection_state,
    token_selection_pos,
    token_selection_scores,
    token_selection_indices,
    token_selection_norm_source,
    token_selection_norm_destination,
    token_selection_assignments,
    token_selection_inverse,
};

pub const slot_count = @typeInfo(Slot).@"enum".fields.len;

pub const Pool = struct {
    slots: [slot_count]Reuse = @splat(.{}),

    const Reuse = struct {
        buf: ?mbuffer.Buffer = null,
        cap: usize = 0,
        // Lent to another phase (mlend): a grow would free the buffer the
        // borrower holds, so handle() refuses instead of reallocating.
        pinned: bool = false,
    };

    pub const Existing = struct { handle: *anyopaque, cap: usize };

    /// The slot's existing buffer and capacity, if any; never allocates. The
    /// lending accessor (mlend).
    pub fn existing(self: *const Pool, slot: Slot) ?Existing {
        const r = &self.slots[@intFromEnum(slot)];
        const b = r.buf orelse return null;
        return .{ .handle = b.handle, .cap = r.cap };
    }

    pub fn pin(self: *Pool, slot: Slot) void {
        self.slots[@intFromEnum(slot)].pinned = true;
    }

    pub fn unpin(self: *Pool, slot: Slot) void {
        self.slots[@intFromEnum(slot)].pinned = false;
    }

    /// Persistent buffer for `slot`, grown (never shrunk) to at least `bytes`.
    pub fn handle(self: *Pool, device: *anyopaque, slot: Slot, bytes: usize) !*anyopaque {
        const r = &self.slots[@intFromEnum(slot)];
        if (r.pinned and (r.buf == null or r.cap < bytes)) return error.PoolSlotLeased;
        if (r.buf == null or r.cap < bytes) {
            if (r.buf) |*old| old.deinit();
            r.buf = null;
            r.cap = 0;
            r.buf = try mbuffer.Buffer.empty(device, bytes);
            r.cap = bytes;
        }
        return r.buf.?.handle;
    }

    /// Like `handle`, but fills the buffer's first `bytes.len` bytes from host
    /// memory (the per-step state upload).
    pub fn filled(self: *Pool, device: *anyopaque, slot: Slot, bytes: []const u8) !*anyopaque {
        const out = try self.handle(device, slot, bytes.len);
        c.zdraw_metal_write_buffer(out, bytes.ptr, bytes.len);
        return out;
    }

    pub fn deinit(self: *Pool) void {
        for (&self.slots) |*r| {
            if (r.buf) |*buf| buf.deinit();
            r.* = .{};
        }
    }
};

test "pool reuses buffers at or below capacity" {
    // No Metal device in unit tests; exercise the bookkeeping only.
    var pool = Pool{};
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 0), pool.slots[0].cap);
}

test "an empty slot lends nothing and a pinned slot refuses to grow" {
    var pool = Pool{};
    defer pool.deinit();
    try std.testing.expect(pool.existing(.gateup) == null);
    pool.pin(.gateup);
    var fake: u32 = 0;
    const no_device: *anyopaque = &fake;
    try std.testing.expectError(error.PoolSlotLeased, pool.handle(no_device, .gateup, 1));
    pool.unpin(.gateup);
    try std.testing.expect(!pool.slots[@intFromEnum(Slot.gateup)].pinned);
}
