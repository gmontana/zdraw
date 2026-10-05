//! ToMA-style token merging for the interior of the stack (1024px lever).
//!
//! v1 pairs consecutive image tokens (row-major neighbors) 2:1 for layers
//! [from, to); caption tokens pass through unmerged. Unmerge restores each
//! pair by adding the band's delta to both members, preserving their
//! difference (the standard residual-preserving scheme).

const std = @import("std");

const env = @import("../runtime/env.zig");
const toma_control = @import("../control/toma_control.zig");
const zrope = @import("zrope.zig");
const zs = @import("zstep_shape.zig");

pub const Band = struct { from: usize, to: usize };

pub fn enabled(dims: zs.Dims) bool {
    if (toma_control.managed()) return toma_control.plan() != null;
    // Read once, not per step: fixed after startup, and this is consulted in
    // the denoise hot loop (the flag itself is killed, default off).
    const on = toma_flag orelse blk: {
        const v = env.flag("ZDRAW_TOMA", false);
        toma_flag = v;
        break :blk v;
    };
    return on and dims.total >= 2048 and dims.img_total % 2 == 0;
}

var toma_flag: ?bool = null;

pub fn bandFor(total: usize) Band {
    if (toma_control.plan()) |plan| {
        return .{
            .from = @min(plan.layer_from, total),
            .to = @min(plan.layer_to, total),
        };
    }
    const from = env.usizeVar("ZDRAW_TOMA_FROM", 6);
    const to = env.usizeVar("ZDRAW_TOMA_TO", 24);
    return .{ .from = @min(from, total), .to = @min(@max(to, from), total) };
}

pub const Merged = struct {
    state: []f32,
    pos: []zrope.Pos,
    snapshot: []f32, // merged-state values BEFORE the band, for unmerge
    total: usize,
    img_pairs: usize,
    img_keep: usize, // unmerged image tail (tile alignment)

    pub fn deinit(self: *Merged, allocator: std.mem.Allocator) void {
        allocator.free(self.state);
        allocator.free(self.pos);
        allocator.free(self.snapshot);
        self.* = undefined;
    }
};

/// Pair-average the image segment; copy caption tokens through. A tail of
/// image tokens stays unmerged so the merged total is 64-aligned (the fast
/// fused kernels require tile-aligned token counts).
pub fn merge(
    allocator: std.mem.Allocator,
    state: []const f32,
    pos: []const zrope.Pos,
    dims: zs.Dims,
) !Merged {
    const h = dims.hidden;
    const max_pairs = dims.img_total / 2;
    const raw_total = max_pairs + dims.cap_total;
    // Dropping d pairs grows the total by d (two tokens replace one), so
    // choose d to land on a 64 multiple.
    const aligned = if (raw_total < 64) raw_total else raw_total / 64 * 64;
    const drop = @min(raw_total - aligned, max_pairs);
    const use_pairs = max_pairs - drop;
    const keep = dims.img_total - use_pairs * 2;
    const total = use_pairs + keep + dims.cap_total;
    const out = try allocator.alloc(f32, total * h);
    errdefer allocator.free(out);
    const out_pos = try allocator.alloc(zrope.Pos, total);
    errdefer allocator.free(out_pos);
    pairAverage(out, out_pos, state, pos, use_pairs, h);
    const tail = keep + dims.cap_total;
    @memcpy(out[use_pairs * h ..][0 .. tail * h], state[(use_pairs * 2) * h ..][0 .. tail * h]);
    @memcpy(out_pos[use_pairs..], pos[use_pairs * 2 ..][0..tail]);
    const snap = try allocator.dupe(f32, out);
    return .{
        .state = out,
        .pos = out_pos,
        .snapshot = snap,
        .total = total,
        .img_pairs = use_pairs,
        .img_keep = keep,
    };
}

/// Residual-preserving unmerge: each pair member gains the band's delta;
/// caption tokens take the banded values directly.
pub fn unmerge(state: []f32, merged: Merged, dims: zs.Dims) void {
    const h = dims.hidden;
    for (0..merged.img_pairs) |i| {
        const now = merged.state[i * h ..][0..h];
        const before = merged.snapshot[i * h ..][0..h];
        const a = state[(2 * i) * h ..][0..h];
        const b = state[(2 * i + 1) * h ..][0..h];
        for (a, b, now, before) |*va, *vb, n, p| {
            const delta = n - p;
            va.* += delta;
            vb.* += delta;
        }
    }
    const tail = merged.img_keep + dims.cap_total;
    const dst = state[(merged.img_pairs * 2) * h ..][0 .. tail * h];
    @memcpy(dst, merged.state[merged.img_pairs * h ..][0 .. tail * h]);
}

fn pairAverage(
    out: []f32,
    out_pos: []zrope.Pos,
    state: []const f32,
    pos: []const zrope.Pos,
    pairs: usize,
    h: usize,
) void {
    for (0..pairs) |i| {
        const a = state[(2 * i) * h ..][0..h];
        const b = state[(2 * i + 1) * h ..][0..h];
        const dst = out[i * h ..][0..h];
        for (dst, a, b) |*d, va, vb| d.* = 0.5 * (va + vb);
        out_pos[i] = pos[2 * i];
    }
}

test "merge halves image tokens and unmerge preserves pair deltas" {
    const dims = zs.Dims{
        .hidden = 2,
        .patch_dim = 1,
        .img_raw = 4,
        .img_total = 4,
        .cap_tokens = 1,
        .cap_total = 1,
        .total = 5,
    };
    var state = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const pos = [_]zrope.Pos{
        .{ 0, 0, 0 }, .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 0, 1, 1 }, .{ 1, 0, 0 },
    };
    var m = try merge(std.testing.allocator, &state, &pos, dims);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 3), m.total);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), m.state[0], 1e-6);
    // band adds +1 to every merged value
    for (m.state[0 .. 2 * 2]) |*v| v.* += 1.0;
    unmerge(&state, m, dims);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), state[0], 1e-6); // 1 + 1
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), state[2], 1e-6); // 3 + 1
}
