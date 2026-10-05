//! Cross-context scratch lending.
//!
//! An `Offer` is a value: the idle buffers one phase holds, each with its
//! capacity, that another phase may use for the duration of a borrow. It is
//! built from existing buffers and never allocates or grows, so lending can
//! only remove memory, never add behaviour. A borrower takes an offered
//! buffer only when its capacity covers the request and otherwise uses its
//! own pool exactly as before. The borrower fully rewrites every buffer it
//! receives before reading it (the contract every pooled slot already
//! carries, mchain_pool.zig), and the schedule proves the offered buffers
//! are dead for the whole borrow: the resident VAE decodes only after the
//! denoise has finished and waited on its last command buffer
//! (docs/architecture.md 5.3). The offering pool keeps ownership and
//! unpins or releases its buffers itself once the borrow ends.

const std = @import("std");

const mchain_pool = @import("mchain_pool.zig");

/// VAE pool roles an offer may serve for one decode: the whole-map conv1_out,
/// or the strip-memory chain's three scratch roles (memory-ladder wall 1).
pub const VaeRole = enum { conv1_out, conv1_chunk, conv1_strip, skip_strip };

pub const Request = union(enum) {
    /// mvattn's chain-pool scratch slot.
    chain: mchain_pool.Slot,
    /// A streamed-VAE pool buffer.
    vae: VaeRole,
};

/// An existing buffer and the bytes it holds.
pub const Buf = struct { handle: *anyopaque, cap: usize };

pub const Offer = struct {
    chain: [mchain_pool.slot_count]?Buf = @splat(null),
    conv1_out: ?Buf = null,
    conv1_chunk: ?Buf = null,
    conv1_strip: ?Buf = null,
    skip_strip: ?Buf = null,

    pub fn put(self: *Offer, slot: mchain_pool.Slot, buf: Buf) void {
        self.chain[@intFromEnum(slot)] = buf;
    }

    /// The offered buffer for `req` when its capacity covers `bytes`, else null.
    pub fn lend(self: Offer, req: Request, bytes: usize) ?*anyopaque {
        const buf = switch (req) {
            .chain => |slot| self.chain[@intFromEnum(slot)],
            .vae => |role| switch (role) {
                .conv1_out => self.conv1_out,
                .conv1_chunk => self.conv1_chunk,
                .conv1_strip => self.conv1_strip,
                .skip_strip => self.skip_strip,
            },
        } orelse return null;
        return if (buf.cap >= bytes) buf.handle else null;
    }
};

/// mvattn's one accessor for chain scratch: the offer first, then the
/// borrower's own chain pool (today's grow-on-demand path).
pub fn chainScratch(
    offer: ?Offer,
    pool: *mchain_pool.Pool,
    device: *anyopaque,
    slot: mchain_pool.Slot,
    bytes: usize,
) !*anyopaque {
    if (offer) |o| {
        if (o.lend(.{ .chain = slot }, bytes)) |h| return h;
    }
    return pool.handle(device, slot, bytes);
}

/// A chain pool's offer for the VAE decode: its idle `gateup` slot, the one
/// slot large enough for conv1_out. The caller pins the slot for the decode
/// so a grow cannot free the buffer under the borrower.
pub fn chainOffer(pool: *const mchain_pool.Pool) Offer {
    var offer = Offer{};
    if (pool.existing(.gateup)) |e| offer.conv1_out = .{ .handle = e.handle, .cap = e.cap };
    // The strip chain's chunk (half of conv1_out's bytes) rides in the same slot.
    offer.conv1_chunk = offer.conv1_out;
    return offer;
}

test "an offer serves only what it holds, within capacity" {
    var fake: u32 = 0;
    const fake_handle: *anyopaque = &fake;
    var offer = Offer{};
    offer.put(.q, .{ .handle = fake_handle, .cap = 64 });
    try std.testing.expectEqual(fake_handle, offer.lend(.{ .chain = .q }, 64).?);
    try std.testing.expectEqual(fake_handle, offer.lend(.{ .chain = .q }, 8).?);
    try std.testing.expect(offer.lend(.{ .chain = .q }, 65) == null);
    try std.testing.expect(offer.lend(.{ .chain = .k }, 8) == null);
    try std.testing.expect(offer.lend(.{ .vae = .conv1_out }, 8) == null);
    // The borrower's own pool is untouched when the offer serves the request
    // (no Metal device exists in unit tests, so a pool allocation would fail).
    var pool = mchain_pool.Pool{};
    defer pool.deinit();
    const h = try chainScratch(offer, &pool, fake_handle, .q, 64);
    try std.testing.expectEqual(fake_handle, h);
    try std.testing.expectEqual(@as(usize, 0), pool.slots[@intFromEnum(mchain_pool.Slot.q)].cap);
}

test "an empty chain pool offers nothing" {
    var pool = mchain_pool.Pool{};
    defer pool.deinit();
    const offer = chainOffer(&pool);
    try std.testing.expect(offer.lend(.{ .vae = .conv1_out }, 1) == null);
    try std.testing.expect(offer.lend(.{ .chain = .q }, 1) == null);
}
