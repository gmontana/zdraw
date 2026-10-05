//! Persistent GPU buffers of the streamed VAE decoder and the pure sizing
//! of their high-water marks (the chain in mvres_stream_chain.zig owns the
//! dataflow; this module owns the storage roles).

const std = @import("std");

const mbuffer = @import("mbuffer.zig");
const mlend = @import("mlend.zig");
const mvres_wino = @import("mvres_wino.zig");

// A grow-on-demand GPU buffer reused across every group. Allocating then freeing
// a fresh Metal buffer per group does NOT return its (already-faulted, zeroed)
// pages to the OS, so per-group allocation accumulates in phys_footprint as the
// VAE resolution climbs. Keeping ONE persistent buffer per role and only growing
// it when a larger group needs it caps the resident working set at the single
// worst group instead of the sum of every group. The chain fully overwrites each
// buffer it uses (convs write every strip pixel, stats every group), so a reused
// buffer needs no re-zeroing.
pub const ReuseBuf = struct {
    buf: ?mbuffer.Buffer = null,
    cap: usize = 0,
    // A buffer another phase lent for the current decode: served instead of
    // the own buffer while set. Decided once at presize, so a role never
    // flips between lease and own buffer inside a decode; cleared by the
    // chain's endLease. Never released here (the lender owns it).
    lent: ?Lease = null,

    pub const Lease = struct { handle: *anyopaque, cap: usize };

    pub fn handle(self: *ReuseBuf, device: *anyopaque, bytes: usize) !*anyopaque {
        if (self.lent) |l| {
            // A request above the lease is a presize-contract violation (the
            // sizes mirror Set.make exactly); fail loudly like the no-grow asserts.
            if (bytes <= l.cap) return l.handle;
            return error.InvalidShape;
        }
        return self.ownHandle(device, bytes);
    }

    /// The own buffer, bypassing any lease (grow-on-demand as before).
    pub fn ownHandle(self: *ReuseBuf, device: *anyopaque, bytes: usize) !*anyopaque {
        if (self.buf == null or self.cap < bytes) {
            if (self.buf) |*b| b.deinit();
            self.buf = try mbuffer.Buffer.empty(device, bytes);
            self.cap = bytes;
        }
        return self.buf.?.handle;
    }

    /// Presize-time decision: take `offer` (a lent buffer whose capacity the
    /// lender checked against `bytes`) for this decode, else size the own
    /// buffer as before.
    pub fn reserve(self: *ReuseBuf, device: *anyopaque, bytes: usize, offer: ?*anyopaque) !void {
        if (offer) |h| {
            self.lent = .{ .handle = h, .cap = bytes };
            return;
        }
        self.lent = null;
        _ = try self.ownHandle(device, bytes);
    }

    pub fn deinit(self: *ReuseBuf) void {
        if (self.buf) |*b| b.deinit();
        self.* = undefined;
    }
};

test "a leased role serves requests within its capacity and never grows" {
    var fake: u32 = 0;
    const fake_handle: *anyopaque = &fake;
    var r = ReuseBuf{};
    try r.reserve(fake_handle, 64, fake_handle);
    try std.testing.expectEqual(fake_handle, try r.handle(fake_handle, 64));
    try std.testing.expectEqual(fake_handle, try r.handle(fake_handle, 8));
    try std.testing.expectError(error.InvalidShape, r.handle(fake_handle, 65));
    try std.testing.expect(r.buf == null); // the own buffer was never created
    r.lent = null;
    try std.testing.expect(r.lent == null);
}

// Persistent GPU buffers reused for every resident group, sized to the largest
// user once so phys does not climb per group. Three full buffers carry the whole
// VAE:
//   feat1   - the uploaded residual input, output ping-pong slot 1, AND the
//             upsample's 2x destination. Block 0 is the only block that reads the
//             pool input buffer, and it never writes output slot 1 (the 3-block
//             ping-pong writes slot 0 then 1 then 0), so input is dead before slot
//             1 is first written; slot 1 is in turn dead once the group finishes
//             (the result lands in slot 0), so the following upsample writes its 2x
//             output here. feat1 already has to hold the next group's larger input,
//             so it absorbs the 2x output for free.
//   output0 - output ping-pong slot 0 (also the upsample's low-res source).
//   conv1_out - conv1 result (read globally by stats2 + conv2's halo) AND the 1x1
//             skip projection: conv2 fully consumes conv1_out before the skip conv
//             writes it, and conv1_out >= the skip's out_ch*hw, so the skip reuses
//             it free. It is never grown by the upsample, so it stays at the small
//             resnet size.
// The upsample needs no buffers of its own: it reads output0 and writes feat1.
pub const Pool = struct {
    feat1: ReuseBuf = .{},
    output0: ReuseBuf = .{},
    conv1_out: ReuseBuf = .{},
    // Strip-memory decode (memory-ladder wall 1, mvres_strip_chain.zig): conv1
    // by output-channel chunk for the statistics, conv1 by row strip (with its
    // halo) for conv2, and the strip-local 1x1 skip. The up blocks then never
    // size conv1_out; the mid blocks still use it whole at 128x128.
    conv1_chunk: ReuseBuf = .{},
    conv1_strip: ReuseBuf = .{},
    skip_strip: ReuseBuf = .{},
    // Tier 2 in-place strips: the halo rows the previous strip overwrote.
    stash: ReuseBuf = .{},
    // Tier 3 strip finish: the float norm scratch for one strip plus halo.
    norm_strip: ReuseBuf = .{},
    // Normed+SiLU scratch for the unfuse path (sized max(in,out)_ch * hw) and
    // the f32 home of the resident mid-attention / FINAL_H-off tail; sized
    // only by its readers (see blockNormNeeded).
    normbuf: ReuseBuf = .{},
    // Winograd scratch (product route, ZDRAW_VAE_WINO): U planes, V and M
    // tile batches; ~67 MB persistent at the largest group.
    wino: mvres_wino.Scratch = .{},
    // Another phase's idle buffers offered for the current decode (the
    // chain's beginLease); conv1_out is the one role that takes a lease.
    lender: ?mlend.Offer = null,

    pub fn deinit(self: *Pool) void {
        self.wino.deinit();
        self.norm_strip.deinit();
        self.stash.deinit();
        self.skip_strip.deinit();
        self.conv1_strip.deinit();
        self.conv1_chunk.deinit();
        self.normbuf.deinit();
        self.conv1_out.deinit();
        self.output0.deinit();
        self.feat1.deinit();
    }
};

/// High-water byte sizes of the pool roles over an up-block sequence.
pub const Sizes = struct {
    feat1: usize = 0,
    output0: usize = 0,
    /// 0 under a strip plan (the up blocks never touch conv1_out then).
    conv1_out: usize = 0,
    /// 0 when nothing in the sequence reads normbuf (the product default).
    normbuf: usize = 0,
    /// Strip-memory roles; 0 without a strip plan.
    conv1_chunk: usize = 0,
    conv1_strip: usize = 0,
    skip_strip: usize = 0,
    stash: usize = 0,
    norm_strip: usize = 0,
};

/// Strip-memory decode (wall 1): conv1 computed in `chunk_ch` output-channel
/// chunks for the statistics and in row strips of `strip_rows` rows (rounded
/// up to the block's tile-row alignment unit) for conv2.
pub const StripPlan = struct {
    chunk_ch: usize,
    strip_rows: usize,
    /// 1 = strip scratch with whole-map input/output (tier 1); 2 = the
    /// group's blocks after the first write in place (tier 2); 3 = tier 2
    /// plus the finish in strips without the whole-map f32 norm scratch.
    tier: u8 = 1,
};

/// Rows per alignment unit: the smallest run of tile rows whose tile count is
/// a multiple of 64 (the Winograd GEMM's n contract), times 4 pixel rows.
/// Mirrors strip_unit_rows in metal_api.m.
pub fn stripUnitRows(width: usize) usize {
    const tiles_x = width / 4;
    var u: usize = 1;
    while ((tiles_x * u) % 64 != 0) u += 1;
    return 4 * u;
}

/// The plan's strip height for a block of this width (a unit multiple).
pub fn stripRowsFor(plan: StripPlan, width: usize) usize {
    const unit = stripUnitRows(width);
    return ((plan.strip_rows + unit - 1) / unit) * unit;
}

/// One up block's geometry as the sizing sees it.
pub const BlockDims = struct {
    in_ch: usize,
    out_ch: usize,
    height: usize,
    width: usize,
    /// The block ends with the 2x nearest upsample (its output lands in feat1).
    upsample: bool,
};

/// Pure sizing for the chain's `presizeUpBlocks`: mirrors Set.make and
/// dispatchUp exactly.
/// `block_norm` is whether the per-block norm scratch has a reader (unfuse);
/// `final_h` whether the sequence hands over an f16 result (FINAL_H), which
/// otherwise needs the f32 tail-conversion home at the end.
pub fn upBlockSizes(
    blocks: []const BlockDims,
    h16: bool,
    full_h: bool,
    block_norm: bool,
    final_h: bool,
    strip: ?StripPlan,
) !Sizes {
    const el: usize = if (h16) @sizeOf(f16) else @sizeOf(f32);
    const c1_el: usize = if (h16 and full_h) @sizeOf(f16) else @sizeOf(f32);
    var s = Sizes{};
    for (blocks, 0..) |cfg, idx| {
        const hw = try checkedMul(cfg.height, cfg.width);
        const in_b = try checkedMul(try checkedMul(cfg.in_ch, hw), el);
        const out_b = try checkedMul(try checkedMul(cfg.out_ch, hw), el);
        s.feat1 = @max(s.feat1, @max(in_b, out_b));
        s.output0 = @max(s.output0, out_b);
        if (strip) |plan| {
            // Strip mode is half-feature by construction (h16 + FULL_H).
            const unit = stripUnitRows(cfg.width);
            const rows = stripRowsFor(plan, cfg.width);
            const chunk = try checkedMul(try checkedMul(plan.chunk_ch, hw), @sizeOf(f16));
            s.conv1_chunk = @max(s.conv1_chunk, chunk);
            const strip_elems = try checkedProduct3(cfg.out_ch, rows + 2 * unit, cfg.width);
            s.conv1_strip = @max(s.conv1_strip, try checkedMul(strip_elems, @sizeOf(f16)));
            // skip_strip also carries conv2's strip result for in-place no-skip
            // blocks (tier 2), so it is sized for every group there.
            if (cfg.in_ch != cfg.out_ch or plan.tier >= 2) {
                const skip_elems = try checkedProduct3(cfg.out_ch, rows, cfg.width);
                s.skip_strip = @max(s.skip_strip, try checkedMul(skip_elems, @sizeOf(f16)));
            }
            if (plan.tier >= 2) {
                const stash_elems = try checkedProduct3(cfg.in_ch, unit + 1, cfg.width);
                s.stash = @max(s.stash, try checkedMul(stash_elems, @sizeOf(f16)));
            }
            if (plan.tier >= 3 and idx == blocks.len - 1) {
                // The finish's strip-local float norm scratch: rows + a 1-row halo each side.
                const norm_elems = try checkedProduct3(cfg.out_ch, rows + 2, cfg.width);
                s.norm_strip = try checkedMul(norm_elems, @sizeOf(f32));
            }
        } else {
            s.conv1_out = @max(s.conv1_out, try checkedMul(try checkedMul(cfg.out_ch, hw), c1_el));
        }
        if (block_norm) {
            const wide = try checkedMul(@max(cfg.in_ch, cfg.out_ch), hw);
            s.normbuf = @max(s.normbuf, try checkedMul(wide, @sizeOf(f32)));
        }
        if (cfg.upsample) {
            const up_hw = try checkedMul(cfg.height * 2, cfg.width * 2);
            const up_elems = try checkedMul(cfg.out_ch, up_hw);
            s.feat1 = @max(s.feat1, try checkedMul(up_elems, el));
        }
        if (h16 and !final_h and idx == blocks.len - 1) {
            // f32 tail-conversion target when FINAL_H is off (runUpBlocks end).
            const last = try checkedProduct3(cfg.out_ch, cfg.height, cfg.width);
            s.normbuf = @max(s.normbuf, try checkedMul(last, @sizeOf(f32)));
        }
    }
    return s;
}

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidShape;
}

fn checkedProduct3(a: usize, b: usize, c: usize) !usize {
    return checkedMul(try checkedMul(a, b), c);
}

test "product up-block sizing wires no block norm scratch" {
    // The decoder at latent 128x128 (1024px output): 512->512 (2x), 512->512
    // (2x), 512->256 (2x), 256->128 at full resolution.
    const blocks = [_]BlockDims{
        .{ .in_ch = 512, .out_ch = 512, .height = 128, .width = 128, .upsample = true },
        .{ .in_ch = 512, .out_ch = 512, .height = 256, .width = 256, .upsample = true },
        .{ .in_ch = 512, .out_ch = 256, .height = 512, .width = 512, .upsample = true },
        .{ .in_ch = 256, .out_ch = 128, .height = 1024, .width = 1024, .upsample = false },
    };
    const mib: usize = 1024 * 1024;
    // Product: f16 features, FULL_H conv1, FINAL_H on, no unfuse.
    const product = try upBlockSizes(&blocks, true, true, false, true, null);
    try std.testing.expectEqual(@as(usize, 512 * mib), product.feat1); // 256ch at 1024^2 f16
    try std.testing.expectEqual(@as(usize, 256 * mib), product.output0); // 128ch at 1024^2 f16
    try std.testing.expectEqual(@as(usize, 256 * mib), product.conv1_out);
    try std.testing.expectEqual(@as(usize, 0), product.normbuf);
    // The unfuse debug route reads the block scratch: max(in,out) x hw x f32.
    const unfuse = try upBlockSizes(&blocks, true, true, true, true, null);
    try std.testing.expectEqual(@as(usize, 1024 * mib), unfuse.normbuf);
    try std.testing.expectEqual(product.feat1, unfuse.feat1);
    // FINAL_H off needs the f32 tail home for the last block's output.
    const tail = try upBlockSizes(&blocks, true, true, false, false, null);
    try std.testing.expectEqual(@as(usize, 512 * mib), tail.normbuf);
    // Strict (f32 features) never sizes the norm slot without unfuse.
    const strict = try upBlockSizes(&blocks, false, false, false, false, null);
    try std.testing.expectEqual(@as(usize, 1024 * mib), strict.feat1);
    try std.testing.expectEqual(@as(usize, 512 * mib), strict.conv1_out);
    try std.testing.expectEqual(@as(usize, 0), strict.normbuf);
}

test "strip alignment units follow the Winograd tile contract" {
    try std.testing.expectEqual(@as(usize, 4), stripUnitRows(1024));
    try std.testing.expectEqual(@as(usize, 4), stripUnitRows(256));
    try std.testing.expectEqual(@as(usize, 8), stripUnitRows(128));
    try std.testing.expectEqual(@as(usize, 16), stripUnitRows(64));
    const p64 = StripPlan{ .chunk_ch = 64, .strip_rows = 64 };
    const p60 = StripPlan{ .chunk_ch = 64, .strip_rows = 60 };
    try std.testing.expectEqual(@as(usize, 64), stripRowsFor(p64, 1024));
    try std.testing.expectEqual(@as(usize, 64), stripRowsFor(p60, 128));
}

test "strip sizing replaces the whole-map conv1_out with the chunk and strip roles" {
    const blocks = [_]BlockDims{
        .{ .in_ch = 512, .out_ch = 512, .height = 128, .width = 128, .upsample = true },
        .{ .in_ch = 512, .out_ch = 512, .height = 256, .width = 256, .upsample = true },
        .{ .in_ch = 512, .out_ch = 256, .height = 512, .width = 512, .upsample = true },
        .{ .in_ch = 256, .out_ch = 128, .height = 1024, .width = 1024, .upsample = false },
    };
    const mib: usize = 1024 * 1024;
    const plan = StripPlan{ .chunk_ch = 64, .strip_rows = 64 };
    const s = try upBlockSizes(&blocks, true, true, false, true, plan);
    try std.testing.expectEqual(@as(usize, 0), s.conv1_out);
    try std.testing.expectEqual(@as(usize, 512 * mib), s.feat1);
    try std.testing.expectEqual(@as(usize, 256 * mib), s.output0);
    try std.testing.expectEqual(@as(usize, 128 * mib), s.conv1_chunk); // 64ch at 1024^2 f16
    // The largest strip: 256ch at 512^2 and 512ch at 256^2 both give 64 + 2*4
    // rows x 18 MiB; the top stage's 128ch at 1024^2 is the same size.
    try std.testing.expectEqual(@as(usize, 256 * 72 * 512 * 2), s.conv1_strip);
    // The skip strips (512->256 at 512^2 and 256->128 at 1024^2) are 16 MiB.
    try std.testing.expectEqual(@as(usize, 128 * 64 * 1024 * 2), s.skip_strip);
}

test "tier 2 sizes the stash and a strip result for every group" {
    const blocks = [_]BlockDims{
        .{ .in_ch = 512, .out_ch = 512, .height = 128, .width = 128, .upsample = true },
        .{ .in_ch = 256, .out_ch = 128, .height = 1024, .width = 1024, .upsample = false },
    };
    const plan = StripPlan{ .chunk_ch = 64, .strip_rows = 64, .tier = 2 };
    const s = try upBlockSizes(&blocks, true, true, false, true, plan);
    // 128x128: unit 8 rows, stash 512ch x 9 rows; 1024: unit 4, stash 256ch x 5 rows.
    try std.testing.expectEqual(@as(usize, 256 * 5 * 1024 * 2), s.stash);
    // The 512-channel group has no skip but needs the strip result in tier 2.
    try std.testing.expectEqual(@as(usize, 128 * 64 * 1024 * 2), s.skip_strip);
}
