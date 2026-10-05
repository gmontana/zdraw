//! Z-Image single-stream attention step.
//!
//! This is the reference CPU path: RMSNorm, q/k/v projection, q/k head norm,
//! Z-Image RoPE, scaled dot-product attention, and output projection.

const attention_fast = @import("attention_fast.zig");
const linear = @import("linear_fast.zig");
const mattn = @import("mattn.zig");
const mres = @import("mres.zig");
const mlinear = @import("mlinear.zig");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const zattn_res = @import("zattn_res.zig");
const zqkv = @import("zqkv.zig");
const zrope = @import("zrope.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
};

pub const Weights = struct {
    norm: tensor.View,
    q: tensor.View,
    k: tensor.View,
    v: tensor.View,
    o: tensor.View,
    q_norm: tensor.View,
    k_norm: tensor.View,
};

pub const Scratch = struct {
    norm: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    mix: []f32,
    scores: []f32,
};

pub fn run(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    out: []f32,
    input: []const f32,
    scale: ?[]const f32,
    pos: []const zrope.Pos,
    weights: Weights,
    scratch: Scratch,
    cfg: Config,
    rope_cache: zrope.Cache,
) !void {
    try check(out, input, scale, pos, scratch, cfg);
    try normalize(out, input, scale, weights.norm, cfg);
    if (try zattn_res.run(
        metal,
        attn,
        out,
        pos,
        resWeights(weights),
        resConfig(cfg),
        rope_cache,
    )) return;
    try project(metal, out, weights, scratch, cfg);
    try normHeads(scratch.q, scratch.norm, weights.q_norm, cfg, cfg.heads);
    try normHeads(scratch.k, scratch.norm, weights.k_norm, cfg, cfg.kv_heads);
    try rotate(scratch.q, cfg, cfg.heads, rope_cache, pos);
    try rotate(scratch.k, cfg, cfg.kv_heads, rope_cache, pos);

    try attention_fast.run(attn, scratch.mix, scratch.q, scratch.k, scratch.v, scratch.scores, .{
        .tokens = cfg.tokens,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .causal = false,
    });
    try linear.gemmBatch(metal, out, scratch.mix, weights.o, null, cfg.tokens);
}

fn project(
    metal: ?*mlinear.Context,
    normed: []f32,
    weights: Weights,
    scratch: Scratch,
    cfg: Config,
) !void {
    try zqkv.batch(
        metal,
        scratch.q,
        scratch.k,
        scratch.v,
        normed,
        weights.q,
        weights.k,
        weights.v,
        cfg.tokens,
    );
}

fn resWeights(weights: Weights) mres.Weights {
    return .{
        .q = weights.q,
        .k = weights.k,
        .v = weights.v,
        .o = weights.o,
        .q_norm = weights.q_norm,
        .k_norm = weights.k_norm,
    };
}

fn resConfig(cfg: Config) mres.Config {
    return .{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .norm_eps = cfg.norm_eps,
    };
}

fn normalize(
    out: []f32,
    input: []const f32,
    scale: ?[]const f32,
    weight: tensor.View,
    cfg: Config,
) !void {
    for (0..cfg.tokens) |tok| {
        const dst = outTok(out, cfg, tok);
        try ops.rmsNormView(dst, inTok(input, cfg, tok), weight, cfg.norm_eps);
        if (scale) |s| mul(dst, scaleTok(s, cfg, tok));
    }
}

fn normHeads(data: []f32, tmp: []f32, weight: tensor.View, cfg: Config, heads: usize) !void {
    if (tmp.len < cfg.head_dim) return error.InvalidShape;
    const buf = tmp[0..cfg.head_dim];
    for (0..cfg.tokens) |tok| {
        for (0..heads) |head| {
            const vec = headVec(data, cfg, tok, head, heads);
            try ops.rmsNormView(buf, vec, weight, cfg.norm_eps);
            copy(vec, buf);
        }
    }
}

fn rotate(
    data: []f32,
    cfg: Config,
    heads: usize,
    rope_cache: zrope.Cache,
    pos: []const zrope.Pos,
) !void {
    for (0..cfg.tokens) |tok| {
        for (0..heads) |head| {
            try rope_cache.apply(headVec(data, cfg, tok, head, heads), pos[tok]);
        }
    }
}

fn check(
    out: []const f32,
    input: []const f32,
    scale: ?[]const f32,
    pos: []const zrope.Pos,
    scratch: Scratch,
    cfg: Config,
) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0 or cfg.head_dim == 0) return error.InvalidShape;
    if (cfg.heads == 0 or cfg.kv_heads == 0) return error.InvalidShape;
    if (out.len != cfg.tokens * cfg.hidden or input.len != out.len) return error.InvalidShape;
    if (scale) |s| {
        if (s.len != cfg.hidden and s.len != out.len) return error.InvalidShape;
    }
    if (pos.len != cfg.tokens or scratch.norm.len != cfg.hidden) return error.InvalidShape;
    if (scratch.scores.len < cfg.tokens) return error.InvalidShape;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    if (scratch.q.len != q_len or scratch.mix.len != q_len) return error.InvalidShape;
    if (scratch.k.len != kv_len or scratch.v.len != kv_len) return error.InvalidShape;
}

fn mul(dst: []f32, scale: []const f32) void {
    for (dst, scale) |*value, s| value.* *= s;
}

fn copy(dst: []f32, src: []const f32) void {
    for (dst, src) |*value, in| value.* = in;
}

fn outTok(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn inTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn scaleTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    if (data.len == cfg.hidden) return data;
    return inTok(data, cfg, tok);
}

fn headVec(data: []f32, cfg: Config, tok: usize, head: usize, heads: usize) []f32 {
    const start = (tok * heads + head) * cfg.head_dim;
    return data[start..][0..cfg.head_dim];
}
