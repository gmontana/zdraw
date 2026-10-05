//! Parameter construction for chained resident block execution.

const std = @import("std");

const c = @import("metal_c.zig");
const chain_c = @import("mblock_chain_c.zig");
const gmode = @import("gemm_mode.zig");
const mblock_c = @import("mblock_c.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const types = @import("mblock_chain_types.zig");
const util = @import("mres_util.zig");
const weight = @import("mblock_chain_weight.zig");
const tensor = @import("tensor.zig");
const zblock = @import("zblock.zig");
const zmod = @import("zmod.zig");
const zrope = @import("zrope.zig");

pub const Modes = struct {
    q: gmode.Mode,
    k: gmode.Mode,
    v: gmode.Mode,
    proj: gmode.Mode,
    ffn_gate: gmode.Mode,
    ffn_up: gmode.Mode,
    ffn_down: gmode.Mode,
};

pub fn uniform(mode: gmode.Mode) Modes {
    return .{
        .q = mode,
        .k = mode,
        .v = mode,
        .proj = mode,
        .ffn_gate = mode,
        .ffn_up = mode,
        .ffn_down = mode,
    };
}

pub fn make(
    mode: gmode.Mode,
    binds: weight.Binds,
    views: zblock.Views,
    mods: ?zmod.Parts,
    cfg: types.Config,
    rope: zrope.Cache,
) !chain_c.Params {
    return makeModes(uniform(mode), binds, views, mods, cfg, rope);
}

pub fn makeModes(
    modes: Modes,
    binds: weight.Binds,
    views: zblock.Views,
    mods: ?zmod.Parts,
    cfg: types.Config,
    rope: zrope.Cache,
) !chain_c.Params {
    return .{
        .attn_norm = try blockParam(views.attn_in, binds.attn_in, types.attnScale(mods), cfg),
        .q = try gemmParam(modes.q, views.q, binds.q, cfg.tokens, cfg.hidden, cfg.hidden),
        .k = try gemmParam(modes.k, views.k, binds.k, cfg.tokens, cfg.hidden, cfg.hidden),
        .v = try gemmParam(modes.v, views.v, binds.v, cfg.tokens, cfg.hidden, cfg.hidden),
        .qk = try qkParam(views, binds, cfg, rope),
        .attn = try attnParam(cfg),
        .proj = try gemmParam(
            modes.proj,
            views.proj,
            binds.proj,
            cfg.tokens,
            cfg.hidden,
            cfg.hidden,
        ),
        .attn_resid = try blockParam(views.attn_out, binds.attn_out, types.attnGate(mods), cfg),
        .ffn_norm = try blockParam(views.ffn_in, binds.ffn_in, types.mlpScale(mods), cfg),
        .ffn_gate = try ffnParam(modes.ffn_gate, views.ffn_gate, binds.ffn_gate, cfg, true),
        .ffn_up = try ffnParam(modes.ffn_up, views.ffn_up, binds.ffn_up, cfg, true),
        .ffn_down = try ffnParam(modes.ffn_down, views.ffn_down, binds.ffn_down, cfg, false),
        .ffn_resid = try blockParam(views.ffn_out, binds.ffn_out, types.mlpGate(mods), cfg),
        .ffn_fused = try fusedParam(views.ffn_fused, binds.ffn_fused, cfg),
    };
}

/// Retarget only the token dimension of an already validated block shape.
///
/// ToMA uses the same layer weights and hidden geometry at two sequence
/// lengths: reduced tokens inside attention/FFN cores and full tokens at
/// residual boundaries.
pub fn withTokens(input: chain_c.Params, tokens: usize) !chain_c.Params {
    const value = try util.toU32(tokens);
    var output = input;
    output.attn_norm.tokens = value;
    output.q.m = value;
    output.k.m = value;
    output.v.m = value;
    output.qk.tokens = value;
    output.attn.tokens = value;
    output.proj.m = value;
    output.attn_resid.tokens = value;
    output.ffn_norm.tokens = value;
    output.ffn_gate.m = value;
    output.ffn_up.m = value;
    output.ffn_down.m = value;
    output.ffn_resid.tokens = value;
    if (output.ffn_fused.mode != 0) output.ffn_fused.m = value;
    return output;
}

// Fused gate+up GEMM (n doubles); mode 0 leaves the path disabled.
fn fusedParam(view: ?tensor.View, bind: mbuffer.Bind, cfg: types.Config) !c.GemmParams {
    const v = view orelse return .{ .m = 0, .k = 0, .n = 0, .dtype = 0, .mode = 0 };
    const shape = try mlinear.shape(v);
    return try gemmParam(.half, v, bind, cfg.tokens, shape.cols, shape.rows);
}

pub fn check(
    state: []const f32,
    pos: []const zrope.Pos,
    mods: ?zmod.Parts,
    cfg: types.Config,
) !void {
    if (state.len != cfg.tokens * cfg.hidden or pos.len != cfg.tokens) return error.InvalidShape;
    if (cfg.hidden != cfg.heads * cfg.head_dim) return error.InvalidShape;
    if (cfg.hidden != cfg.kv_heads * cfg.head_dim) return error.InvalidShape;
    if (!util.fits(cfg.tokens, cfg.hidden, cfg.hidden)) return error.GemmShape;
    try checkScale(types.attnScale(mods), cfg.hidden);
    try checkScale(types.attnGate(mods), cfg.hidden);
    try checkScale(types.mlpScale(mods), cfg.hidden);
    try checkScale(types.mlpGate(mods), cfg.hidden);
}

fn blockParam(
    view: tensor.View,
    bind: mbuffer.Bind,
    scale: ?[]const f32,
    cfg: types.Config,
) !mblock_c.Params {
    return .{
        .tokens = try util.toU32(cfg.tokens),
        .hidden = try util.toU32(cfg.hidden),
        .dtype = try util.dtype(view.dtype),
        .has_scale = if (scale == null) 0 else 1,
        .eps = cfg.norm_eps,
        .weight_offset = try util.toU64(bind.offset),
    };
}

fn qkParam(
    views: zblock.Views,
    binds: weight.Binds,
    cfg: types.Config,
    rope: zrope.Cache,
) !c.QkNormParams {
    const bases = rope.axisBases();
    return .{
        .tokens = try util.toU32(cfg.tokens),
        .heads = try util.toU32(cfg.heads),
        .kv_heads = try util.toU32(cfg.kv_heads),
        .head_dim = try util.toU32(cfg.head_dim),
        .q_dtype = try util.dtype(views.q_norm.dtype),
        .k_dtype = try util.dtype(views.k_norm.dtype),
        .dim0 = try util.toU32(rope.cfg.dims[0]),
        .dim1 = try util.toU32(rope.cfg.dims[1]),
        .dim2 = try util.toU32(rope.cfg.dims[2]),
        .eps = cfg.norm_eps,
        .q_offset = try util.toU64(binds.q_norm.offset),
        .k_offset = try util.toU64(binds.k_norm.offset),
        .base0 = try util.toU64(bases[0]),
        .base1 = try util.toU64(bases[1]),
        .base2 = try util.toU64(bases[2]),
    };
}

fn attnParam(cfg: types.Config) !c.AttnParams {
    return .{
        .tokens = try util.toU32(cfg.tokens),
        .heads = try util.toU32(cfg.heads),
        .kv_heads = try util.toU32(cfg.kv_heads),
        .head_dim = try util.toU32(cfg.head_dim),
        .causal = 0,
    };
}

fn ffnParam(
    mode: gmode.Mode,
    view: tensor.View,
    bind: mbuffer.Bind,
    cfg: types.Config,
    up: bool,
) !c.GemmParams {
    const shape = try mlinear.shape(view);
    const k = if (up) cfg.hidden else shape.cols;
    const n = if (up) shape.rows else cfg.hidden;
    return try gemmParam(mode, view, bind, cfg.tokens, k, n);
}

fn gemmParam(
    mode: gmode.Mode,
    view: tensor.View,
    bind: mbuffer.Bind,
    m: usize,
    k: usize,
    n: usize,
) !c.GemmParams {
    return .{
        .m = try util.toU32(m),
        .k = try util.toU32(k),
        .n = try util.toU32(n),
        .dtype = if (mode == .w6) 4 else try util.dtype(view.dtype),
        .mode = util.mode(mode),
        .weight_offset = try util.toU64(bind.offset),
    };
}

fn checkScale(scale: ?[]const f32, hidden: usize) !void {
    if (scale) |s| {
        if (s.len != hidden) return error.InvalidShape;
    }
}

test "withTokens preserves block geometry and retargets sequence fields" {
    var input = std.mem.zeroes(chain_c.Params);
    input.q = .{ .m = 8, .k = 16, .n = 24, .dtype = 2, .mode = 3, .weight_offset = 32 };
    input.ffn_fused = .{ .m = 8, .k = 16, .n = 48, .dtype = 1, .mode = 2, .weight_offset = 64 };
    const output = try withTokens(input, 5);

    try std.testing.expectEqual(@as(u32, 5), output.attn_norm.tokens);
    try std.testing.expectEqual(@as(u32, 5), output.q.m);
    try std.testing.expectEqual(@as(u32, 16), output.q.k);
    try std.testing.expectEqual(@as(u32, 24), output.q.n);
    try std.testing.expectEqual(@as(u32, 3), output.q.mode);
    try std.testing.expectEqual(@as(u64, 32), output.q.weight_offset);
    try std.testing.expectEqual(@as(u32, 5), output.k.m);
    try std.testing.expectEqual(@as(u32, 5), output.v.m);
    try std.testing.expectEqual(@as(u32, 5), output.qk.tokens);
    try std.testing.expectEqual(@as(u32, 5), output.attn.tokens);
    try std.testing.expectEqual(@as(u32, 5), output.proj.m);
    try std.testing.expectEqual(@as(u32, 5), output.attn_resid.tokens);
    try std.testing.expectEqual(@as(u32, 5), output.ffn_norm.tokens);
    try std.testing.expectEqual(@as(u32, 5), output.ffn_gate.m);
    try std.testing.expectEqual(@as(u32, 5), output.ffn_up.m);
    try std.testing.expectEqual(@as(u32, 5), output.ffn_down.m);
    try std.testing.expectEqual(@as(u32, 5), output.ffn_resid.tokens);
    try std.testing.expectEqual(@as(u32, 5), output.ffn_fused.m);
}
