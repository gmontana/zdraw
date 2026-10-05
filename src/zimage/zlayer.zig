//! One Z-Image transformer block.
//!
//! The block order follows the reference model directly:
//! attention residual first, then the SwiGLU feed-forward residual.

const std = @import("std");

const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const ops = @import("../runtime/ops.zig");
const tensor = @import("../pack/tensor.zig");
const zattn = @import("zattn.zig");
const zblock = @import("zblock.zig");
const zffn = @import("zffn.zig");
const zmod = @import("zmod.zig");
const zrope = @import("zrope.zig");
const zlayer_res = @import("zlayer_res.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
};

pub const Scratch = struct {
    norm: []f32,
    mod: []f32,
    attn: zattn.Scratch,
    attn_out: []f32,
    ffn: zffn.Scratch,
    ffn_out: []f32,
};

pub fn run(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    state: []f32,
    views: zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    scratch: Scratch,
    cfg: Config,
    rope_cache: zrope.Cache,
) !void {
    try check(state, scratch, cfg);
    const mods = if (adaln) |input| try modulate(metal, scratch.mod, input, views) else null;
    if (try zlayer_res.run(metal, attn, state, views, mods, pos, resConfig(cfg), rope_cache)) {
        return;
    }
    try zattn.run(
        metal,
        attn,
        scratch.attn_out,
        state,
        attnScale(mods),
        pos,
        attnWeights(views),
        scratch.attn,
        attnConfig(cfg),
        rope_cache,
    );
    try addNorm(state, scratch.attn_out, views.attn_out, attnGate(mods), scratch.norm, cfg);
    try zffn.run(
        metal,
        scratch.ffn_out,
        state,
        mlpScale(mods),
        ffnWeights(views),
        scratch.ffn,
        ffnConfig(cfg),
    );
    addGated(state, scratch.ffn_out, mlpGate(mods), cfg);
}

fn modulate(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    views: zblock.Views,
) !zmod.Parts {
    const weight = views.ada_w orelse return error.InvalidShape;
    const bias = views.ada_b orelse return error.InvalidShape;
    return zmod.run(metal, out, input, weight, bias);
}

fn addNorm(
    state: []f32,
    src: []const f32,
    weight: tensor.View,
    gate: ?[]const f32,
    norm: []f32,
    cfg: Config,
) !void {
    if (norm.len != cfg.hidden) return error.InvalidShape;
    for (0..cfg.tokens) |tok| {
        const dst = mutTok(state, cfg, tok);
        try ops.rmsNormView(norm, constTok(src, cfg, tok), weight, cfg.norm_eps);
        addTok(dst, norm, gate);
    }
}

fn addGated(state: []f32, src: []const f32, gate: ?[]const f32, cfg: Config) void {
    for (0..cfg.tokens) |tok| addTok(mutTok(state, cfg, tok), constTok(src, cfg, tok), gate);
}

fn addTok(dst: []f32, src: []const f32, gate: ?[]const f32) void {
    if (gate) |g| {
        for (dst, src, g) |*value, inc, mult| value.* += inc * mult;
    } else {
        for (dst, src) |*value, inc| value.* += inc;
    }
}

fn check(state: []const f32, scratch: Scratch, cfg: Config) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0) return error.InvalidShape;
    if (state.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (scratch.norm.len != cfg.hidden) return error.InvalidShape;
    if (scratch.attn_out.len != state.len) return error.InvalidShape;
    if (scratch.ffn_out.len != state.len) return error.InvalidShape;
}

fn attnWeights(views: zblock.Views) zattn.Weights {
    return .{
        .norm = views.attn_in,
        .q = views.q,
        .k = views.k,
        .v = views.v,
        .o = views.proj,
        .q_norm = views.q_norm,
        .k_norm = views.k_norm,
    };
}

fn ffnWeights(views: zblock.Views) zffn.Weights {
    return .{
        .in_norm = views.ffn_in,
        .out_norm = views.ffn_out,
        .gate = views.ffn_gate,
        .down = views.ffn_down,
        .up = views.ffn_up,
    };
}

fn attnConfig(cfg: Config) zattn.Config {
    return .{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .norm_eps = cfg.norm_eps,
    };
}

fn ffnConfig(cfg: Config) zffn.Config {
    return .{ .tokens = cfg.tokens, .hidden = cfg.hidden, .norm_eps = cfg.norm_eps };
}

fn resConfig(cfg: Config) zlayer_res.Config {
    return .{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .norm_eps = cfg.norm_eps,
    };
}

fn attnScale(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.attn_scale;
    return null;
}

fn attnGate(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.attn_gate;
    return null;
}

fn mlpScale(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.mlp_scale;
    return null;
}

fn mlpGate(parts: ?zmod.Parts) ?[]const f32 {
    if (parts) |p| return p.mlp_gate;
    return null;
}

fn mutTok(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn constTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

test "gated residual scales each hidden value" {
    var dst = [_]f32{ 1.0, 2.0 };
    addTok(&dst, &.{ 4.0, 5.0 }, &.{ 0.5, 2.0 });

    try std.testing.expectApproxEqAbs(3.0, dst[0], 0.0001);
    try std.testing.expectApproxEqAbs(12.0, dst[1], 0.0001);
}
