//! The resident Klein DiT's pooled activation buffers: their identity
//! (PoolShape), the one size table every allocation and every lend decision
//! reads (poolBytes), and the offer it makes to the VAE decode (offerFrom). The dispatch
//! code in zflux2_resident.zig owns the buffers' lifetime and contents.

const std = @import("std");

const mbuffer = @import("mbuffer.zig");
const mchain_pool = @import("mchain_pool.zig");
const mlend = @import("mlend.zig");
const zflux2 = @import("zflux2.zig");

const head_dim = zflux2.head_dim;
const txt_len = zflux2.txt_len;

/// Resident pool identity: every pooled buffer is sized from one of these
/// dims (img/cat/q.. from img_len+hidden, wide/act/cat12 from inner, emb from
/// joint_dim), so the pool is valid to reuse iff ALL of them match. Dropping a
/// dim here would let a differently-sized request silently reuse a wrong-sized
/// buffer — emb especially, the lone joint_dim-sized buffer (txt_len*joint_dim).
pub const PoolShape = struct {
    img_len: usize = 0,
    hidden: usize = 0,
    inner: usize = 0,
    joint_dim: usize = 0,
    // Batched multi-seed image count; 1 = the classic single-image pool.
    batch: usize = 1,
};

pub const Bufs = struct {
    // The shape every buffer below was sized from (ensurePool's reuse key).
    shape: PoolShape = .{},
    img: mbuffer.Buffer, // [img_len, hidden] stream state
    txt: mbuffer.Buffer, // [txt_len, hidden]
    cat: mbuffer.Buffer, // [tokens, hidden] single-stream state
    nscratch: mbuffer.Buffer, // multi-stage GEMM scratch (written before read)
    norm_t: mbuffer.Buffer,
    norm_c: mbuffer.Buffer,
    q: mbuffer.Buffer, // [tokens, hidden] concat q/k/v
    k: mbuffer.Buffer,
    v: mbuffer.Buffer,
    o: mbuffer.Buffer,
    proj_t: mbuffer.Buffer,
    proj_i: mbuffer.Buffer,
    wide: mbuffer.Buffer, // [tokens, 2*inner]
    // f32-branch scratch, materialized on first dispatch via Ctx.actHandle/
    // cat12Handle: dead (never allocated, ~396 MB at 1024) under the default
    // half-activation mode, which uses ah/bh for the same roles.
    act: ?mbuffer.Buffer = null, // [tokens, inner]
    cat12: ?mbuffer.Buffer = null, // [tokens, hidden+inner]
    rope_cos: mbuffer.Buffer,
    rope_sin: mbuffer.Buffer,
    emb: mbuffer.Buffer,
    final_in: mbuffer.Buffer,
    out128: mbuffer.Buffer,
    // Half-activation mode only (ZDRAW_KLEIN_ACT, the default): half A-operand
    // scratch so the big GEMMs read half the activation bytes. ah doubles as
    // the ln-out and cat12 roles (serialized within a block, hazard-tracked);
    // bh holds the swiglu output and, in the double blocks, the txt-stream LN
    // output (txt_len*hidden <= tokens*inner keeps it in bounds). Null (no
    // cost) when f32 opts out.
    ah: ?mbuffer.Buffer = null, // [tokens, hidden+inner] half
    bh: ?mbuffer.Buffer = null, // [tokens, inner] half

    pub fn deinit(self: *Bufs) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| {
            // shape is plain identity data; only the buffers deinit.
            if (f.type == ?mbuffer.Buffer) {
                if (@field(self, f.name)) |*buf| buf.deinit();
            } else if (f.type == mbuffer.Buffer) {
                @field(self, f.name).deinit();
            }
        }
    }
};

/// The pooled roles, each sized by `poolBytes` from the PoolShape alone.
pub const PoolRole = enum {
    img,
    txt,
    cat,
    nscratch,
    norm_t,
    norm_c,
    q,
    k,
    v,
    o,
    proj_t,
    proj_i,
    wide,
    rope,
    emb,
    final_in,
    out128,
    ah,
    bh,
};

/// Byte size of a pooled role: the single source of truth for ensurePool's
/// allocations and for what the pool may lend.
pub fn poolBytes(s: PoolShape, role: PoolRole) usize {
    const tokens = txt_len + s.img_len;
    const nb = s.batch;
    return switch (role) {
        .img, .proj_i, .final_in => nb * s.img_len * s.hidden * 4,
        .txt, .norm_t, .proj_t => nb * txt_len * s.hidden * 4,
        .cat, .nscratch, .norm_c, .q, .k, .v, .o => nb * tokens * s.hidden * 4,
        .wide => nb * tokens * s.inner * 2 * 4,
        .rope => nb * tokens * head_dim * 4,
        .emb => txt_len * s.joint_dim * 4,
        .out128 => nb * s.img_len * 128 * 4,
        .ah => nb * tokens * (s.hidden + s.inner) * 2,
        .bh => nb * tokens * s.inner * 2,
    };
}

pub fn sized(dev: *anyopaque, s: PoolShape, role: PoolRole) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(dev, poolBytes(s, role));
}

/// The pool's buffer for a role; null for a half-scratch role that is not
/// materialized (f32 opt-out).
pub fn roleBuffer(b: *const Bufs, role: PoolRole) ?mbuffer.Buffer {
    return switch (role) {
        .img => b.img,
        .txt => b.txt,
        .cat => b.cat,
        .nscratch => b.nscratch,
        .norm_t => b.norm_t,
        .norm_c => b.norm_c,
        .q => b.q,
        .k => b.k,
        .v => b.v,
        .o => b.o,
        .proj_t => b.proj_t,
        .proj_i => b.proj_i,
        .wide => b.wide,
        .rope => b.rope_cos,
        .emb => b.emb,
        .final_in => b.final_in,
        .out128 => b.out128,
        .ah => b.ah,
        .bh => b.bh,
    };
}

/// The chain-pool slots the mid-attention asks for, and the pooled role
/// that serves each. f32 slots map onto f32 slots of the same or larger
/// size; the two largest requests (ffn's S/P block, mod_in's half P block)
/// take the half scratch, which exists only in the default half mode. The
/// VAE's conv1_out (out_ch * hw * f16 = 256 * px^2 bytes) rides in `wide`
/// (288 * px^2 + 36 MiB), the only slot that covers it.
const chain_map = [_]struct { slot: mchain_pool.Slot, role: PoolRole }{
    .{ .slot = .norm, .role = .norm_c },
    .{ .slot = .q, .role = .q },
    .{ .slot = .k, .role = .k },
    .{ .slot = .v, .role = .v },
    .{ .slot = .mix, .role = .o },
    .{ .slot = .attn, .role = .cat },
    .{ .slot = .state, .role = .nscratch },
    .{ .slot = .gate, .role = .img },
    .{ .slot = .up, .role = .proj_i },
    .{ .slot = .gateup, .role = .final_in },
    .{ .slot = .ffn, .role = .ah },
    .{ .slot = .mod_in, .role = .bh },
};

/// The idle pool as an offer: every mapped slot with its true capacity from
/// the size table, so the borrower's capacity check cannot drift from the
/// allocation. Pure: unit-tested below.
pub fn offerFrom(b: *const Bufs) mlend.Offer {
    var offer = mlend.Offer{};
    for (chain_map) |m| {
        const buf = roleBuffer(b, m.role) orelse continue;
        offer.put(m.slot, .{ .handle = buf.handle, .cap = poolBytes(b.shape, m.role) });
    }
    offer.conv1_out = .{ .handle = b.wide.handle, .cap = poolBytes(b.shape, .wide) };
    // The strip-memory chain's scratch (wall 1): the chunk rides in `wide`
    // like conv1_out did, the strip and the skip in the half scratch slots.
    offer.conv1_chunk = offer.conv1_out;
    if (b.ah) |ah| {
        offer.conv1_strip = .{ .handle = ah.handle, .cap = poolBytes(b.shape, .ah) };
    }
    if (b.bh) |bh| {
        offer.skip_strip = .{ .handle = bh.handle, .cap = poolBytes(b.shape, .bh) };
    }
    return offer;
}

test "pool roles are sized from the shape alone" {
    const mib: usize = 1024 * 1024;
    const s = PoolShape{ .img_len = 4096, .hidden = 3072, .inner = 9216, .joint_dim = 7680 };
    try std.testing.expectEqual(54 * mib, poolBytes(s, .q));
    try std.testing.expectEqual(48 * mib, poolBytes(s, .img));
    try std.testing.expectEqual(324 * mib, poolBytes(s, .wide));
    try std.testing.expectEqual(108 * mib, poolBytes(s, .ah));
    try std.testing.expectEqual(81 * mib, poolBytes(s, .bh));
    try std.testing.expectEqual(6 * mib, poolBytes(s, .txt));
    try std.testing.expectEqual(15 * mib, poolBytes(s, .emb));
}

/// A pool of borrowed fake handles (no Metal in unit tests), one distinct
/// address per role so a lend can be traced back to its slot.
const Fakes = struct {
    cells: [19]u32 = @splat(0),

    fn bufs(self: *Fakes, shape: PoolShape, half: bool) Bufs {
        const c = &self.cells;
        var b = Bufs{
            .img = mbuffer.Buffer.borrow(&c[0]),
            .txt = mbuffer.Buffer.borrow(&c[1]),
            .cat = mbuffer.Buffer.borrow(&c[2]),
            .nscratch = mbuffer.Buffer.borrow(&c[3]),
            .norm_t = mbuffer.Buffer.borrow(&c[4]),
            .norm_c = mbuffer.Buffer.borrow(&c[5]),
            .q = mbuffer.Buffer.borrow(&c[6]),
            .k = mbuffer.Buffer.borrow(&c[7]),
            .v = mbuffer.Buffer.borrow(&c[8]),
            .o = mbuffer.Buffer.borrow(&c[9]),
            .proj_t = mbuffer.Buffer.borrow(&c[10]),
            .proj_i = mbuffer.Buffer.borrow(&c[11]),
            .wide = mbuffer.Buffer.borrow(&c[12]),
            .rope_cos = mbuffer.Buffer.borrow(&c[13]),
            .rope_sin = mbuffer.Buffer.borrow(&c[14]),
            .emb = mbuffer.Buffer.borrow(&c[15]),
            .final_in = mbuffer.Buffer.borrow(&c[16]),
            .out128 = mbuffer.Buffer.borrow(&c[17]),
        };
        if (half) {
            b.ah = mbuffer.Buffer.borrow(&c[18]);
            b.bh = mbuffer.Buffer.borrow(&c[0]);
        }
        b.shape = shape;
        return b;
    }
};

test "every mid-attention request fits its offered slot at every Klein size" {
    // The VAE mid block runs at latent (px/8)^2 tokens with 512 channels:
    // f32 slots are 32*px^2 bytes, the half Q/K/V^T 16*px^2, the S/P block
    // 1024 rows x tokens f32 = 64*px^2 (half 32*px^2); the product conv1_out
    // is 128 ch at px^2 in f16 = 256*px^2. The DiT pool at the same output has
    // img_len = px^2/256 tokens of hidden 3072.
    var fakes = Fakes{};
    const sizes = [_]usize{ 256, 512, 768, 1024 };
    const batches = [_]usize{ 1, 2 };
    for (batches) |nb| for (sizes) |px| {
        const shape = PoolShape{
            .img_len = px * px / 256,
            .hidden = 3072,
            .inner = 9216,
            .joint_dim = 7680,
            .batch = nb,
        };
        const b = fakes.bufs(shape, true);
        const offer = offerFrom(&b);
        const f32_slot = 32 * px * px;
        const half_slot = 16 * px * px;
        const s_block = 64 * px * px;
        const p_block = 32 * px * px;
        const c1 = 256 * px * px;
        try std.testing.expectEqual(b.norm_c.handle, offer.lend(.{ .chain = .norm }, f32_slot).?);
        try std.testing.expectEqual(b.q.handle, offer.lend(.{ .chain = .q }, f32_slot).?);
        try std.testing.expectEqual(b.o.handle, offer.lend(.{ .chain = .mix }, f32_slot).?);
        try std.testing.expectEqual(b.cat.handle, offer.lend(.{ .chain = .attn }, f32_slot).?);
        const scratch = offer.lend(.{ .chain = .state }, f32_slot).?;
        try std.testing.expectEqual(b.nscratch.handle, scratch);
        try std.testing.expectEqual(b.img.handle, offer.lend(.{ .chain = .gate }, f32_slot).?);
        try std.testing.expectEqual(b.proj_i.handle, offer.lend(.{ .chain = .up }, half_slot).?);
        const gateup = offer.lend(.{ .chain = .gateup }, half_slot).?;
        try std.testing.expectEqual(b.final_in.handle, gateup);
        try std.testing.expectEqual(b.ah.?.handle, offer.lend(.{ .chain = .ffn }, s_block).?);
        try std.testing.expectEqual(b.bh.?.handle, offer.lend(.{ .chain = .mod_in }, p_block).?);
        try std.testing.expectEqual(b.wide.handle, offer.lend(.{ .vae = .conv1_out }, c1).?);
        // Wall 1 scratch at px^2 output: the chunk is 64 ch x px^2 f16 =
        // 128*px^2, the strip 128 ch x 72 rows x px f16 and the skip
        // 128 ch x 64 rows x px f16 (the top stage's shapes).
        const chunk = offer.lend(.{ .vae = .conv1_chunk }, 128 * px * px).?;
        try std.testing.expectEqual(b.wide.handle, chunk);
        const strip = offer.lend(.{ .vae = .conv1_strip }, 128 * 72 * px * 2).?;
        try std.testing.expectEqual(b.ah.?.handle, strip);
        const skip = offer.lend(.{ .vae = .skip_strip }, 128 * 64 * px * 2).?;
        try std.testing.expectEqual(b.bh.?.handle, skip);
    };
}

test "the offer declines what the pool cannot serve" {
    var fakes = Fakes{};
    const s = PoolShape{ .img_len = 4096, .hidden = 3072, .inner = 9216, .joint_dim = 7680 };
    // f32 opt-out: no half scratch, so the two large requests fall back.
    const plain = offerFrom(&fakes.bufs(s, false));
    try std.testing.expect(plain.lend(.{ .chain = .ffn }, 64 << 20) == null);
    try std.testing.expect(plain.lend(.{ .chain = .mod_in }, 32 << 20) == null);
    const half = offerFrom(&fakes.bufs(s, true));
    // Too large for the mapped slot.
    try std.testing.expect(half.lend(.{ .chain = .q }, 55 << 20) == null);
    // Slots the mid-attention never asks for are not offered.
    try std.testing.expect(half.lend(.{ .chain = .toma_state }, 8) == null);
    // Strict's f32 conv1_out (512 * px^2 at 1024) exceeds wide: own buffer.
    try std.testing.expect(half.lend(.{ .vae = .conv1_out }, 512 << 20) == null);
}
