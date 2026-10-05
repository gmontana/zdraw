//! Small scaled dot-product attention reference kernel.

const std = @import("std");

pub const Error = error{
    InvalidShape,
};

pub const Config = struct {
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    /// Right-padding limit: keys at index >= valid are masked for every query
    /// (0 = no padding). Right padding plus causal masking means real-token
    /// rows are unaffected; only the pad rows change.
    valid: usize = 0,
};

pub fn run(
    out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    scratch: []f32,
    cfg: Config,
) !void {
    try check(out, q, k, v, scratch, cfg);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    for (0..cfg.tokens) |tok| {
        var limit = if (cfg.causal) tok + 1 else cfg.tokens;
        if (cfg.valid != 0 and limit > cfg.valid) limit = cfg.valid;
        for (0..cfg.heads) |head| {
            const kv_head = head * cfg.kv_heads / cfg.heads;
            score(qVec(q, cfg, tok, head), k, scratch[0..limit], cfg, kv_head, scale);
            softmax(scratch[0..limit]);
            mix(outVec(out, cfg, tok, head), v, scratch[0..limit], cfg, kv_head);
        }
    }
}

fn check(
    out: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    scratch: []const f32,
    cfg: Config,
) !void {
    if (cfg.tokens == 0 or cfg.heads == 0 or cfg.kv_heads == 0) return error.InvalidShape;
    if (cfg.head_dim == 0 or cfg.heads % cfg.kv_heads != 0) return error.InvalidShape;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    if (out.len != q_len or q.len != q_len) return error.InvalidShape;
    if (k.len != kv_len or v.len != kv_len) return error.InvalidShape;
    if (scratch.len < cfg.tokens) return error.InvalidShape;
}

fn score(
    qv: []const f32,
    k: []const f32,
    out: []f32,
    cfg: Config,
    kv_head: usize,
    scale: f32,
) void {
    for (out, 0..) |*dst, tok| {
        dst.* = dot(qv, kvVec(k, cfg, tok, kv_head)) * scale;
    }
}

fn softmax(values: []f32) void {
    var max = -std.math.inf(f32);
    for (values) |value| max = @max(max, value);

    var sum: f32 = 0.0;
    for (values) |*value| {
        value.* = @exp(value.* - max);
        sum += value.*;
    }
    for (values) |*value| value.* /= sum;
}

fn mix(out: []f32, v: []const f32, weights: []const f32, cfg: Config, kv_head: usize) void {
    @memset(out, 0.0);
    for (weights, 0..) |weight, tok| {
        const vv = kvVec(v, cfg, tok, kv_head);
        for (out, vv) |*dst, value| dst.* += weight * value;
    }
}

fn dot(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

fn qVec(data: []const f32, cfg: Config, tok: usize, head: usize) []const f32 {
    const start = (tok * cfg.heads + head) * cfg.head_dim;
    return data[start..][0..cfg.head_dim];
}

fn outVec(data: []f32, cfg: Config, tok: usize, head: usize) []f32 {
    const start = (tok * cfg.heads + head) * cfg.head_dim;
    return data[start..][0..cfg.head_dim];
}

fn kvVec(data: []const f32, cfg: Config, tok: usize, head: usize) []const f32 {
    const start = (tok * cfg.kv_heads + head) * cfg.head_dim;
    return data[start..][0..cfg.head_dim];
}

test "single token attention returns value" {
    var out = [_]f32{ 0.0, 0.0 };
    var scratch = [_]f32{0.0};
    const cfg = Config{ .tokens = 1, .heads = 1, .kv_heads = 1, .head_dim = 2, .causal = true };

    try run(&out, &.{ 1.0, 0.0 }, &.{ 1.0, 0.0 }, &.{ 3.0, 4.0 }, &scratch, cfg);
    try std.testing.expectApproxEqAbs(3.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(4.0, out[1], 0.0001);
}

test "causal attention ignores future tokens" {
    var out = [_]f32{ 0.0, 0.0 };
    var scratch = [_]f32{ 0.0, 0.0 };
    const cfg = Config{ .tokens = 2, .heads = 1, .kv_heads = 1, .head_dim = 1, .causal = true };

    try run(&out, &.{ 1.0, 1.0 }, &.{ 1.0, 1.0 }, &.{ 2.0, 6.0 }, &scratch, cfg);
    try std.testing.expectApproxEqAbs(2.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(4.0, out[1], 0.0001);
}

test "padding limit masks trailing keys without touching real rows" {
    // 3 tokens, 1 head, head_dim 1, causal. valid=2 means token 2 is padding:
    // real rows (0,1) must be identical with and without the limit, and row 2
    // must stop attending to itself.
    const q = [_]f32{ 1, 1, 1 };
    const k = [_]f32{ 0, 0, 100 };
    const v = [_]f32{ 1, 2, 99 };
    var scratch: [3]f32 = undefined;
    var open_out: [3]f32 = undefined;
    var masked_out: [3]f32 = undefined;
    const base = Config{ .tokens = 3, .heads = 1, .kv_heads = 1, .head_dim = 1, .causal = true };
    try run(&open_out, &q, &k, &v, &scratch, base);
    var limited = base;
    limited.valid = 2;
    try run(&masked_out, &q, &k, &v, &scratch, limited);
    try std.testing.expectEqual(open_out[0], masked_out[0]);
    try std.testing.expectEqual(open_out[1], masked_out[1]);
    // Without the limit the huge key 2 dominates row 2; with it, row 2 sees
    // only the real keys and matches row 1's average.
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), open_out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), masked_out[2], 1e-5);
}
