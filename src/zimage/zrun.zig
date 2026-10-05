//! Z-Image transformer stack runner.
//!
//! This file owns only the boring loop over blocks. Loading weights, building
//! sequences, and denoising stay outside so the pipeline remains easy to read.

const std = @import("std");

const env = @import("../runtime/env.zig");
const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const mstack = @import("../metal/mstack_chain.zig");
const mfallback = @import("../metal/metal_fallback.zig");
const zblock = @import("zblock.zig");
const zlayer = @import("zlayer.zig");
const zpack_file = @import("../pack/zpack_file.zig");
const zprobe = @import("zprobe.zig");
const zrope = @import("zrope.zig");

pub fn layers(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    state: []f32,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    scratch: zlayer.Scratch,
    cfg: zlayer.Config,
    rope_cache: zrope.Cache,
    family: zpack_file.Family,
    probe: ?zprobe.LayerProbe,
) !void {
    if (probe) |p| {
        return layersProbe(metal, attn, state, views, adaln, pos, scratch, cfg, rope_cache, p);
    }
    // ZDRAW_ACT=f16 simulates f16 inter-layer activation storage (the resident
    // refactor's precision) on the CPU loop, so we can gate image quality before
    // rewriting the resident kernels. It forces the non-resident path.
    const act_f16 = env.equals("ZDRAW_ACT", "f16");
    if (!act_f16) {
        if (try stack(allocator, metal, attn, state, views, adaln, pos, cfg, rope_cache, family)) {
            return;
        }
    }
    for (views) |view| {
        try zlayer.run(metal, attn, state, view, adaln, pos, scratch, cfg, rope_cache);
        if (act_f16) roundF16(state);
    }
}

/// Round f32 activations through f16 to simulate f16 storage (f32 compute).
fn roundF16(state: []f32) void {
    for (state) |*v| {
        const h: f16 = @floatCast(v.*);
        v.* = @floatCast(h);
    }
}

fn layersProbe(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    state: []f32,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    scratch: zlayer.Scratch,
    cfg: zlayer.Config,
    rope_cache: zrope.Cache,
    probe: zprobe.LayerProbe,
) !void {
    try probe.captureBefore(state);
    for (views, 0..) |view, layer| {
        try zlayer.run(metal, attn, state, view, adaln, pos, scratch, cfg, rope_cache);
        try probe.capture(layer, state);
    }
}

fn stack(
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    state: []f32,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    cfg: zlayer.Config,
    rope: zrope.Cache,
    family: zpack_file.Family,
) !bool {
    if (adaln == null) return false;
    const m = metal orelse return false;
    const a = attn orelse return false;
    mstack.run(allocator, m, a, state, views, adaln, pos, stackCfg(cfg), rope, family, .{
        .total = views.len,
    }) catch |err| {
        if (mfallback.isFallback(err)) return false;
        return err;
    };
    return true;
}

fn stackCfg(cfg: zlayer.Config) mstack.Config {
    return .{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .norm_eps = cfg.norm_eps,
    };
}

test "empty stack leaves state unchanged" {
    var state = [_]f32{ 1.0, 2.0 };
    const cfg = zlayer.Config{
        .tokens = 1,
        .hidden = 2,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .norm_eps = 0.0,
    };
    const rope = try zrope.Cache.init(std.testing.allocator, .{
        .dims = .{ 2, 2, 2 },
        .lens = .{ 1, 1, 1 },
        .theta = 1.0,
    });
    defer rope.deinit(std.testing.allocator);

    try layers(
        std.testing.allocator,
        null,
        null,
        &state,
        &.{},
        null,
        &.{.{ 0, 0, 0 }},
        undefined,
        cfg,
        rope,
        .main,
        null,
    );
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, &state);
}
