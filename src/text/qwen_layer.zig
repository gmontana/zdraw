//! One Qwen decoder layer, written as a direct residual block.

const std = @import("std");

const linear = @import("../runtime/linear_fast.zig");
const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const ops = @import("../runtime/ops.zig");
const qattn = @import("qwen_attn.zig");
const qtext = @import("qwen_text.zig");
const tensor = @import("../pack/tensor.zig");

pub const Error = error{
    InvalidShape,
};

pub const Weights = struct {
    attn: qattn.Weights,
    post_norm: tensor.View,
    gate: tensor.View,
    up: tensor.View,
    down: tensor.View,
};

pub const Scratch = struct {
    attn: qattn.Scratch,
    attn_out: []f32,
    mlp: qtext.MlpScratch,
    mlp_out: []f32,
};

pub fn run(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    state: []f32,
    weights: Weights,
    scratch: Scratch,
    cfg: qattn.Config,
) !void {
    try check(state, scratch, cfg);
    try qattn.run(metal, attn, scratch.attn_out, state, weights.attn, scratch.attn, cfg);
    add(state, scratch.attn_out);
    try normTokens(scratch.attn_out, state, weights.post_norm, cfg);
    if (cfg.use_gemm) {
        // Simdgroup GEMM route (resident fused FFN when available).
        try linear.gemmFfn(
            metal,
            scratch.mlp_out,
            scratch.attn_out,
            weights.gate,
            weights.up,
            weights.down,
            scratch.mlp.gate,
            scratch.mlp.up,
            cfg.tokens,
        );
    } else {
        try qtext.mlpBatch(
            metal,
            scratch.mlp_out,
            scratch.attn_out,
            weights.gate,
            weights.up,
            weights.down,
            scratch.mlp,
            cfg.tokens,
        );
    }
    add(state, scratch.mlp_out);
}

fn check(state: []const f32, scratch: Scratch, cfg: qattn.Config) !void {
    if (state.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (scratch.attn_out.len != state.len) return error.InvalidShape;
    if (scratch.mlp_out.len != state.len) return error.InvalidShape;
}

fn normTokens(out: []f32, state: []const f32, weight: tensor.View, cfg: qattn.Config) !void {
    for (0..cfg.tokens) |tok| {
        try ops.rmsNormView(outTok(out, cfg, tok), inTok(state, cfg, tok), weight, cfg.norm_eps);
    }
}

fn add(dst: []f32, src: []const f32) void {
    for (dst, src) |*value, inc| value.* += inc;
}

fn outTok(data: []f32, cfg: qattn.Config, tok: usize) []f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

fn inTok(data: []const f32, cfg: qattn.Config, tok: usize) []const f32 {
    return data[tok * cfg.hidden ..][0..cfg.hidden];
}

test "layer applies attention residual" {
    const cfg = qattn.Config{
        .tokens = 1,
        .hidden = 2,
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .norm_eps = 0.0,
        .rope_theta = 10000.0,
        .causal = true,
    };
    const one = [_]u8{ 0x00, 0x3c, 0x00, 0x3c };
    const norm = tensor.View{ .dtype = .f16, .shape = &.{2}, .bytes = &one };
    const id = [_]u8{ 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c };
    const mat = tensor.View{ .dtype = .f16, .shape = &.{ 2, 2 }, .bytes = &id };
    const zero = [_]u8{ 0, 0, 0, 0 };
    const mlp_in = tensor.View{ .dtype = .f16, .shape = &.{ 1, 2 }, .bytes = &zero };
    const mlp_down = tensor.View{ .dtype = .f16, .shape = &.{ 2, 1 }, .bytes = &zero };

    var state = [_]f32{ 1.0, 2.0 };
    var norm_buf = [_]f32{ 0.0, 0.0 };
    var q = [_]f32{ 0.0, 0.0 };
    var k = [_]f32{ 0.0, 0.0 };
    var v = [_]f32{ 0.0, 0.0 };
    var mix = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{0.0};
    var attn_out = [_]f32{ 0.0, 0.0 };
    var gate = [_]f32{0.0};
    var up = [_]f32{0.0};
    var mlp_out = [_]f32{ 0.0, 0.0 };

    try run(null, null, &state, .{
        .attn = .{
            .norm = norm,
            .q = mat,
            .k = mat,
            .v = mat,
            .o = mat,
            .q_norm = norm,
            .k_norm = norm,
        },
        .post_norm = norm,
        .gate = mlp_in,
        .up = mlp_in,
        .down = mlp_down,
    }, .{
        .attn = .{
            .norm = &norm_buf,
            .q = &q,
            .k = &k,
            .v = &v,
            .mix = &mix,
            .scores = &scores,
        },
        .attn_out = &attn_out,
        .mlp = .{ .gate = &gate, .up = &up },
        .mlp_out = &mlp_out,
    }, cfg);

    try std.testing.expectApproxEqAbs(1.6324, state[0], 0.0001);
    try std.testing.expectApproxEqAbs(3.2649, state[1], 0.0001);
}
