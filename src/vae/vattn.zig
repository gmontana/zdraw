//! VAE spatial self-attention.
//!
//! Diffusers uses this in the AutoencoderKL mid block. The tensor is normalized
//! as NCHW, flattened to row-major spatial tokens, attended, then added back.

const std = @import("std");

const attention = @import("../runtime/attention.zig");
const attention_fast = @import("../runtime/attention_fast.zig");
const gnorm = @import("gnorm.zig");
const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const tensor = @import("../pack/tensor.zig");
const vproj = @import("vproj.zig");

pub const Config = struct {
    channels: usize,
    height: usize,
    width: usize,
    head_dim: usize,
    groups: usize = 32,
    eps: f32 = 0.000001,
};

pub const Views = struct {
    norm_w: tensor.View,
    norm_b: tensor.View,
    q_w: tensor.View,
    q_b: tensor.View,
    k_w: tensor.View,
    k_b: tensor.View,
    v_w: tensor.View,
    v_b: tensor.View,
    out_w: tensor.View,
    out_b: tensor.View,
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
    linear: ?*mlinear.Context,
    metal_attn: ?*mattn.Context,
    out: []f32,
    input: []const f32,
    views: Views,
    scratch: Scratch,
    cfg: Config,
) !void {
    try check(out, input, scratch, cfg);
    try gnorm.run(scratch.norm, input, views.norm_w, views.norm_b, normCfg(cfg));
    try projectAll(linear, views, scratch, cfg);
    copy(out, input);
    try attend(linear, metal_attn, scratch, cfg);
    try writeOut(linear, out, views, scratch, cfg);
}

fn projectAll(metal: ?*mlinear.Context, views: Views, scratch: Scratch, cfg: Config) !void {
    gatherAll(scratch.mix, scratch.norm, cfg);
    try vproj.run(metal, scratch.q, scratch.mix, views.q_w, views.q_b, tokens(cfg));
    try vproj.run(metal, scratch.k, scratch.mix, views.k_w, views.k_b, tokens(cfg));
    try vproj.run(metal, scratch.v, scratch.mix, views.v_w, views.v_b, tokens(cfg));
}

fn gatherAll(out: []f32, input: []const f32, cfg: Config) void {
    for (0..tokens(cfg)) |tok| {
        gather(seq(out, cfg, tok), input, cfg, tok);
    }
}

fn attend(
    linear: ?*mlinear.Context,
    metal: ?*mattn.Context,
    scratch: Scratch,
    cfg: Config,
) !void {
    const acfg = attention.Config{
        .tokens = tokens(cfg),
        .heads = heads(cfg),
        .kv_heads = heads(cfg),
        .head_dim = cfg.head_dim,
        .causal = false,
    };
    // Pool-borrowed Metal path when both contexts exist (mattn.runPooled);
    // an unsupported shape falls through, mirroring attention_fast.
    if (linear) |lin| {
        if (metal) |ctx| {
            if (ctx.runPooled(lin, scratch.mix, scratch.q, scratch.k, scratch.v, acfg)) |_| {
                return;
            } else |err| switch (err) {
                error.UnsupportedShape => {},
                else => return err,
            }
        }
    }
    try attention_fast.run(metal, scratch.mix, scratch.q, scratch.k, scratch.v, scratch.scores, acfg);
}

fn writeOut(
    metal: ?*mlinear.Context,
    out: []f32,
    views: Views,
    scratch: Scratch,
    cfg: Config,
) !void {
    try vproj.run(metal, scratch.q, scratch.mix, views.out_w, views.out_b, tokens(cfg));
    for (0..tokens(cfg)) |tok| {
        scatterAdd(out, seq(scratch.q, cfg, tok), cfg, tok);
    }
}

fn check(out: []const f32, input: []const f32, scratch: Scratch, cfg: Config) !void {
    if (cfg.channels == 0 or cfg.height == 0 or cfg.width == 0) return error.InvalidShape;
    if (cfg.head_dim == 0 or cfg.channels % cfg.head_dim != 0) return error.InvalidShape;
    const count = len(cfg);
    if (out.len != count or input.len != count or scratch.norm.len != count) {
        return error.InvalidShape;
    }
    if (scratch.scores.len < tokens(cfg)) return error.InvalidShape;
    if (scratch.q.len != count or scratch.k.len != count) return error.InvalidShape;
    if (scratch.v.len != count or scratch.mix.len != count) return error.InvalidShape;
}

fn gather(out: []f32, input: []const f32, cfg: Config, tok: usize) void {
    const row = tok / cfg.width;
    const col = tok % cfg.width;
    for (out, 0..) |*value, ch| value.* = input[pix(cfg, ch, row, col)];
}

fn scatterAdd(out: []f32, input: []const f32, cfg: Config, tok: usize) void {
    const row = tok / cfg.width;
    const col = tok % cfg.width;
    for (input, 0..) |value, ch| out[pix(cfg, ch, row, col)] += value;
}

fn copy(out: []f32, input: []const f32) void {
    for (out, input) |*dst, src| dst.* = src;
}

fn normCfg(cfg: Config) gnorm.Config {
    return .{
        .channels = cfg.channels,
        .height = cfg.height,
        .width = cfg.width,
        .groups = cfg.groups,
        .eps = cfg.eps,
    };
}

fn seq(data: []f32, cfg: Config, tok: usize) []f32 {
    return data[tok * cfg.channels ..][0..cfg.channels];
}

fn pix(cfg: Config, ch: usize, row: usize, col: usize) usize {
    return (ch * cfg.height + row) * cfg.width + col;
}

fn tokens(cfg: Config) usize {
    return cfg.height * cfg.width;
}

fn heads(cfg: Config) usize {
    return cfg.channels / cfg.head_dim;
}

fn len(cfg: Config) usize {
    return cfg.channels * cfg.height * cfg.width;
}

test "two tokens use row-major spatial order" {
    const cfg = Config{
        .channels = 1,
        .height = 1,
        .width = 2,
        .head_dim = 1,
        .groups = 1,
        .eps = 0.0,
    };
    var norm = [_]f32{0} ** 2;
    var q = [_]f32{0} ** 2;
    var k = [_]f32{0} ** 2;
    var v = [_]f32{0} ** 2;
    var mix = [_]f32{0} ** 2;
    var scores = [_]f32{0} ** 2;
    var out = [_]f32{0} ** 2;

    try run(null, null, &out, &.{ 1.0, 3.0 }, fixtureViews(&five1, &one1), .{
        .norm = &norm,
        .q = &q,
        .k = &k,
        .v = &v,
        .mix = &mix,
        .scores = &scores,
    }, cfg);
    try std.testing.expectApproxEqAbs(6.0, out[0], 0.0001);
    try std.testing.expectApproxEqAbs(8.0, out[1], 0.0001);
}

fn fixtureViews(v_bias: []const u8, out_gain: []const u8) Views {
    return .{
        .norm_w = view1(&one1),
        .norm_b = view1(&zero1),
        .q_w = mat1(&zero1),
        .q_b = view1(&zero1),
        .k_w = mat1(&zero1),
        .k_b = view1(&zero1),
        .v_w = mat1(&one1),
        .v_b = view1(v_bias),
        .out_w = mat1(out_gain),
        .out_b = view1(&zero1),
    };
}

fn view1(bytes: []const u8) tensor.View {
    return .{ .dtype = .u8, .shape = &shape1, .bytes = bytes };
}

fn mat1(bytes: []const u8) tensor.View {
    return .{ .dtype = .u8, .shape = &shape11, .bytes = bytes };
}

const shape1 = [_]usize{1};
const shape11 = [_]usize{ 1, 1 };
const zero1 = [_]u8{0};
const one1 = [_]u8{1};
const five1 = [_]u8{5};
