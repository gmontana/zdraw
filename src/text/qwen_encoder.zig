//! Qwen text encoder loop over the real safetensor weights.

const std = @import("std");

const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const progress = @import("../cli/progress.zig");
const qattn = @import("qwen_attn.zig");
const qlayer = @import("qwen_layer.zig");
const qnames = @import("qwen_names.zig");
const qscratch = @import("qwen_scratch.zig");
const qtext = @import("qwen_text.zig");
const shards = @import("../pack/shards.zig");
const tensor = @import("../pack/tensor.zig");
const weight_index = @import("../pack/weight_index.zig");
const zconfig = @import("../zimage/zimage_config.zig");

pub const Error = error{
    InvalidShape,
};

/// Which Qwen3 hidden state(s) the consumer wants.
///
/// The encoder is shared: Z-Image conditions on the penultimate hidden state;
/// FLUX.2 stacks several intermediate hidden states (per-token concatenated).
pub const Extract = union(enum) {
    /// hidden state after (layers-1) blocks. out.len == tokens*hidden.
    penultimate,
    /// snapshot after each listed block COUNT (e.g. {9,18,27} = after 9/18/27
    /// blocks), concatenated per token. out.len == tokens*groups.len*hidden.
    stacked: []const usize,
};

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    out: []f32,
    ids: []const u32,
    store: *const shards.Store,
    index: weight_index.Index,
    text: zconfig.Text,
    extract: Extract,
    scratch: *qscratch.Scratch,
    use_gemm: bool,
    proj_gemm: ?bool,
    valid: usize,
) !void {
    var cfg = maskedAttnCfg(text, ids.len, valid);
    cfg.use_gemm = use_gemm;
    cfg.proj_gemm = proj_gemm;
    const state_len = ids.len * cfg.hidden;
    if (scratch.state.len < state_len) return error.InvalidShape;
    const state = scratch.state[0..state_len];

    const embed = try store.view(index, qnames.embed);
    try qtext.embed(state, ids, embed);

    const ctx = Ctx{
        .io = io,
        .allocator = allocator,
        .metal = metal,
        .attn = attn,
        .store = store,
        .index = index,
        .scratch = scratch,
        .cfg = cfg,
    };
    switch (extract) {
        .penultimate => try runPenultimate(ctx, out, state, text),
        .stacked => |groups| try runStacked(ctx, out, state, ids.len, groups),
    }
}

const Ctx = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    store: *const shards.Store,
    index: weight_index.Index,
    scratch: *qscratch.Scratch,
    cfg: qattn.Config,
};

fn block(ctx: Ctx, state: []f32, layer: usize, total: usize) !void {
    try progress.layer(ctx.io, ctx.allocator, "text", layer + 1, total);
    const weights = try layerWeights(ctx.allocator, ctx.store, ctx.index, layer);
    try qlayer.run(ctx.metal, ctx.attn, state, weights, ctx.scratch.layer, ctx.cfg);
    try dumpLayer(ctx, state, layer);
}

/// ZDRAW_QWEN_DUMP=<dir>: write layer-0 q/k/v and the post-layer state, so two
/// projection routes can be diffed tensor-by-tensor instead of inferred from
/// the conditioning they eventually produce.
fn dumpLayer(ctx: Ctx, state: []const f32, layer: usize) !void {
    const dir = std.c.getenv("ZDRAW_QWEN_DUMP") orelse return;
    if (layer != 0) return;
    const a = ctx.scratch.layer.attn;
    const q_len = ctx.cfg.tokens * ctx.cfg.heads * ctx.cfg.head_dim;
    const kv = ctx.cfg.tokens * ctx.cfg.kv_heads * ctx.cfg.head_dim;
    const path = std.mem.span(dir);
    try dumpF32(ctx.allocator, path, "q", a.q[0..q_len]);
    try dumpF32(ctx.allocator, path, "k", a.k[0..kv]);
    try dumpF32(ctx.allocator, path, "v", a.v[0..kv]);
    try dumpF32(ctx.allocator, path, "mix", a.mix[0..q_len]);
    try dumpF32(ctx.allocator, path, "state", state);
}

fn dumpF32(
    allocator: std.mem.Allocator,
    dir: []const u8,
    tag: []const u8,
    values: []const f32,
) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}.bin", .{ dir, tag });
    defer allocator.free(path);
    const zpath = try allocator.dupeZ(u8, path);
    defer allocator.free(zpath);
    const file = std.c.fopen(zpath.ptr, "wb") orelse return error.AccessDenied;
    defer _ = std.c.fclose(file);
    const bytes = std.mem.sliceAsBytes(values);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, file);
}

/// Penultimate hidden state (hidden_states[-2]): after layers-1 blocks, before
/// the last block and final norm. The Z-Image path — kept bit-identical.
fn runPenultimate(ctx: Ctx, out: []f32, state: []f32, text: zconfig.Text) !void {
    if (out.len != state.len) return error.InvalidShape;
    const used: usize = @as(usize, @intCast(text.layers)) - 1;
    for (0..used) |layer| try block(ctx, state, layer, used);
    @memcpy(out, state);
}

/// FLUX.2: snapshot after each block count in `groups`, per-token concat.
fn runStacked(ctx: Ctx, out: []f32, state: []f32, tokens: usize, groups: []const usize) !void {
    if (out.len != tokens * groups.len * ctx.cfg.hidden) return error.InvalidShape;
    var last: usize = 0;
    for (groups) |g| last = @max(last, g);
    for (0..last) |layer| {
        try block(ctx, state, layer, last);
        for (groups, 0..) |g, slot| {
            if (g != layer + 1) continue;
            scatterSnapshot(out, state, tokens, ctx.cfg.hidden, slot, groups.len);
        }
    }
}

/// Copy a [tokens, hidden] snapshot into `slot` of a [tokens, groups*hidden]
/// per-token-concatenated output.
pub fn scatterSnapshot(
    out: []f32,
    state: []const f32,
    tokens: usize,
    hidden: usize,
    slot: usize,
    groups: usize,
) void {
    const stride = groups * hidden;
    for (0..tokens) |tok| {
        const src = state[tok * hidden ..][0..hidden];
        const dst = out[tok * stride + slot * hidden ..][0..hidden];
        @memcpy(dst, src);
    }
}

pub fn layerWeights(
    allocator: std.mem.Allocator,
    store: *const shards.Store,
    index: weight_index.Index,
    layer: usize,
) !qlayer.Weights {
    return .{
        .attn = .{
            .norm = try layerView(allocator, store, index, layer, .input_norm),
            .q = try layerView(allocator, store, index, layer, .q),
            .k = try layerView(allocator, store, index, layer, .k),
            .v = try layerView(allocator, store, index, layer, .v),
            .o = try layerView(allocator, store, index, layer, .o),
            .q_norm = try layerView(allocator, store, index, layer, .q_norm),
            .k_norm = try layerView(allocator, store, index, layer, .k_norm),
        },
        .post_norm = try layerView(allocator, store, index, layer, .post_norm),
        .gate = try layerView(allocator, store, index, layer, .gate),
        .up = try layerView(allocator, store, index, layer, .up),
        .down = try layerView(allocator, store, index, layer, .down),
    };
}

fn layerView(
    allocator: std.mem.Allocator,
    store: *const shards.Store,
    index: weight_index.Index,
    layer: usize,
    part: qnames.Layer,
) !tensor.View {
    const name = try qnames.layerName(allocator, layer, part);
    defer allocator.free(name);
    return try store.view(index, name);
}

pub fn attnConfig(text: zconfig.Text, tokens: usize) qattn.Config {
    return maskedAttnCfg(text, tokens, 0);
}

/// `valid` = count of real tokens when the sequence is right-padded.
pub fn maskedAttnCfg(text: zconfig.Text, tokens: usize, valid: usize) qattn.Config {
    return .{
        .valid = valid,
        .tokens = tokens,
        .hidden = @intCast(text.hidden_size),
        .heads = @intCast(text.heads),
        .kv_heads = @intCast(text.kv_heads),
        .head_dim = @intCast(text.head_dim),
        .norm_eps = @floatCast(text.rms_norm_eps),
        .rope_theta = @floatCast(text.rope_theta),
        .causal = true,
    };
}

test "stacked snapshot concatenates per token" {
    // 2 tokens, hidden 2, 3 groups -> out stride 6 per token.
    var out = [_]f32{0} ** 12;
    const s9 = [_]f32{ 1, 2, 3, 4 }; // tok0=[1,2] tok1=[3,4]
    const s18 = [_]f32{ 5, 6, 7, 8 };
    const s27 = [_]f32{ 9, 10, 11, 12 };
    scatterSnapshot(&out, &s9, 2, 2, 0, 3);
    scatterSnapshot(&out, &s18, 2, 2, 1, 3);
    scatterSnapshot(&out, &s27, 2, 2, 2, 3);
    // tok0 = [1,2, 5,6, 9,10]; tok1 = [3,4, 7,8, 11,12]
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12 }, &out);
}

test "build attention config from text config" {
    const cfg = attnConfig(.{
        .hidden_size = 2560,
        .intermediate_size = 9728,
        .layers = 36,
        .heads = 32,
        .kv_heads = 8,
        .head_dim = 128,
        .vocab_size = 151936,
        .rms_norm_eps = 0.000001,
        .rope_theta = 1000000.0,
    }, 512);

    try std.testing.expectEqual(@as(usize, 512), cfg.tokens);
    try std.testing.expectEqual(@as(usize, 2560), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 128), cfg.head_dim);
    try std.testing.expectApproxEqAbs(@as(f32, 1000000.0), cfg.rope_theta, 0.001);
}
