//! Qwen attention step built from the small reference kernels.

const std = @import("std");

const attention_fast = @import("../runtime/attention_fast.zig");
const linear = @import("../runtime/linear_fast.zig");
const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const ops = @import("../runtime/ops.zig");
const rope = @import("rope.zig");
const tensor = @import("../pack/tensor.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
    rope_theta: f32,
    causal: bool,
    // Route projections through the simdgroup GEMM path instead of the
    // rows-parallel linear kernel. Off for Z-Image (its penultimate path is
    // kept bit-identical); Klein's 512-token stacked encode opts in.
    use_gemm: bool = false,
    /// Projection-only override for bisecting the GEMM-route divergence;
    /// null = follow use_gemm.
    proj_gemm: ?bool = null,
    /// Number of real (non-padding) tokens; keys beyond it are masked for
    /// every query. 0 = unpadded. The references (diffusers, mflux) both pad
    /// to a fixed length and pass this mask; dropping it corrupts the padded
    /// rows, which the FLUX.2 DiT then consumes unmasked.
    valid: usize = 0,
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
    weights: Weights,
    scratch: Scratch,
    cfg: Config,
) !void {
    try check(out, input, scratch, cfg);
    try project(metal, out, input, weights, scratch, cfg);
    try normalizeHeads(scratch.q, scratch.norm, weights.q_norm, cfg, cfg.heads);
    try normalizeHeads(scratch.k, scratch.norm, weights.k_norm, cfg, cfg.kv_heads);
    try rotate(scratch.q, cfg, cfg.heads);
    try rotate(scratch.k, cfg, cfg.kv_heads);
    dumpPre(scratch, cfg);

    // Metal SDPA where the shape is supported; CPU kernel otherwise.
    try attention_fast.run(attn, scratch.mix, scratch.q, scratch.k, scratch.v, scratch.scores, .{
        .tokens = cfg.tokens,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .causal = cfg.causal,
        .valid = cfg.valid,
    });
    try project1(metal, out, scratch.mix, weights.o, cfg);
}

fn project1(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    cfg: Config,
) !void {
    if (cfg.proj_gemm orelse cfg.use_gemm) {
        try linear.gemmBatch(metal, out, input, weight, null, cfg.tokens);
    } else {
        try linear.runBatch(metal, out, input, weight, null, cfg.tokens);
    }
}

/// ZDRAW_QWEN_DUMP=<dir>: q/k/v exactly as attention is about to consume them.
/// Paired with the post-layer dump in qwen_encoder, this separates "attention
/// read stale buffers" from "something overwrote them afterwards".
fn dumpPre(scratch: Scratch, cfg: Config) void {
    const dir = std.c.getenv("ZDRAW_QWEN_DUMP") orelse return;
    if (dumped_pre) return; // layer 0 only
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    if (scratch.q.len < q_len or scratch.k.len < kv or scratch.v.len < kv) return;
    dumped_pre = true;
    writeBin(dir, "pre_q", scratch.q[0..q_len]);
    writeBin(dir, "pre_k", scratch.k[0..kv]);
    writeBin(dir, "pre_v", scratch.v[0..kv]);
}

var dumped_pre = false;

fn writeBin(dir: [*:0]const u8, tag: []const u8, values: []const f32) void {
    var buf: [512]u8 = undefined;
    const path = std.fmt.bufPrintZ(&buf, "{s}/{s}.bin", .{ std.mem.span(dir), tag }) catch return;
    const file = std.c.fopen(path.ptr, "wb") orelse return;
    defer _ = std.c.fclose(file);
    const bytes = std.mem.sliceAsBytes(values);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, file);
}

fn project(
    metal: ?*mlinear.Context,
    normed: []f32,
    input: []const f32,
    weights: Weights,
    scratch: Scratch,
    cfg: Config,
) !void {
    try normalize(normed, input, weights.norm, cfg);
    try project1(metal, scratch.q, normed, weights.q, cfg);
    try project1(metal, scratch.k, normed, weights.k, cfg);
    try project1(metal, scratch.v, normed, weights.v, cfg);
}

fn normalize(out: []f32, input: []const f32, weight: tensor.View, cfg: Config) !void {
    for (0..cfg.tokens) |tok| {
        try ops.rmsNormView(outTok(out, cfg, tok), inTok(input, cfg, tok), weight, cfg.norm_eps);
    }
}

fn normalizeHeads(
    data: []f32,
    tmp: []f32,
    weight: tensor.View,
    cfg: Config,
    heads: usize,
) !void {
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

fn rotate(data: []f32, cfg: Config, heads: usize) !void {
    // The rope frequencies depend only on (position, i), so they are tabled
    // once per token and reused across all heads instead of being recomputed
    // - four transcendentals per element - for every head of every layer.
    const half = cfg.head_dim / 2;
    if (cfg.head_dim % 2 != 0 or half == 0 or half > rope.max_half) {
        return error.InvalidShape;
    }
    var cos_tab: [rope.max_half]f32 = undefined;
    var sin_tab: [rope.max_half]f32 = undefined;
    for (0..cfg.tokens) |tok| {
        try rope.table(cos_tab[0..half], sin_tab[0..half], tok, cfg.rope_theta);
        for (0..heads) |head| {
            rope.applyTable(
                headVec(data, cfg, tok, head, heads),
                cos_tab[0..half],
                sin_tab[0..half],
            );
        }
    }
}

fn check(out: []const f32, input: []const f32, scratch: Scratch, cfg: Config) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0 or cfg.head_dim == 0) return error.InvalidShape;
    if (cfg.heads == 0 or cfg.kv_heads == 0) return error.InvalidShape;
    if (out.len != cfg.tokens * cfg.hidden or input.len != out.len) return error.InvalidShape;
    if (scratch.norm.len != cfg.hidden) return error.InvalidShape;
    if (scratch.scores.len < cfg.tokens) return error.InvalidShape;
    const q_len = cfg.tokens * cfg.heads * cfg.head_dim;
    const kv_len = cfg.tokens * cfg.kv_heads * cfg.head_dim;
    if (scratch.q.len != q_len or scratch.mix.len != q_len) return error.InvalidShape;
    if (scratch.k.len != kv_len or scratch.v.len != kv_len) return error.InvalidShape;
}

fn outTok(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn inTok(data: []const f32, cfg: Config, tok: usize) []const f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn copy(dst: []f32, src: []const f32) void {
    for (dst, src) |*value, in| value.* = in;
}

fn headVec(data: []f32, cfg: Config, tok: usize, head: usize, heads: usize) []f32 {
    const start = (tok * heads + head) * cfg.head_dim;
    return data[start..][0..cfg.head_dim];
}

test "single token attention step" {
    const cfg = Config{
        .tokens = 1,
        .hidden = 2,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .norm_eps = 0.0,
        .rope_theta = 10000.0,
        .causal = true,
    };
    const norm_bytes = [_]u8{ 0x00, 0x3c, 0x00, 0x3c };
    const norm = tensor.View{ .dtype = .f16, .shape = &.{2}, .bytes = &norm_bytes };
    const id = [_]u8{ 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c };
    const mat = tensor.View{ .dtype = .f16, .shape = &.{ 2, 2 }, .bytes = &id };
    var norm_buf = [_]f32{ 0.0, 0.0 };
    var q = [_]f32{ 0.0, 0.0 };
    var k = [_]f32{ 0.0, 0.0 };
    var v = [_]f32{ 0.0, 0.0 };
    var mix = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{0.0};
    var out = [_]f32{ 0.0, 0.0 };

    try run(null, null, &out, &.{ 1.0, 2.0 }, .{
        .norm = norm,
        .q = mat,
        .k = mat,
        .v = mat,
        .o = mat,
        .q_norm = norm,
        .k_norm = norm,
    }, .{
        .norm = &norm_buf,
        .q = &q,
        .k = &k,
        .v = &v,
        .mix = &mix,
        .scores = &scores,
    }, cfg);

    try std.testing.expectApproxEqAbs(0.6324, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.2649, out[1], 0.0001);
}
