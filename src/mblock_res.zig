//! Resident Metal execution for one Z-Image transformer block.

const std = @import("std");

const c = @import("metal_c.zig");
const mattn = @import("mattn.zig");
const mbuffer = @import("mbuffer.zig");
const math = @import("mblock_math.zig");
const mattn_buf = @import("mattn_buf.zig");
const mffn_buf = @import("mffn_buf.zig");
const mlinear = @import("mlinear.zig");
const types = @import("mblock_chain_types.zig");
const zblock = @import("zblock.zig");
const zmod = @import("zmod.zig");
const zrope = @import("zrope.zig");

pub const Config = types.Config;

// Mod-slice helpers live in mblock_chain_types; re-bound to keep call sites short.
const attnScale = types.attnScale;
const attnGate = types.attnGate;
const mlpScale = types.mlpScale;
const mlpGate = types.mlpGate;

const Bufs = struct {
    state: mbuffer.Buffer,
    norm: mbuffer.Buffer,
    attn: mbuffer.Buffer,
    ffn: mbuffer.Buffer,

    fn deinit(self: *Bufs) void {
        self.state.deinit();
        self.norm.deinit();
        self.attn.deinit();
        self.ffn.deinit();
    }
};

pub fn run(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    state: []f32,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
) !void {
    try check(state, pos, cfg);
    var bufs = try initBufs(metal, state);
    defer bufs.deinit();
    try math.normScale(metal, bufs.norm, bufs.state, views.attn_in, attnScale(mods), normCfg(cfg));
    try mattn_buf.run(
        metal,
        attn,
        bufs.attn,
        bufs.norm,
        attnWeights(views),
        pos,
        rope,
        cfg,
    );
    try math.residual(metal, bufs.state, bufs.attn, views.attn_out, attnGate(mods), normCfg(cfg));
    try math.normScale(metal, bufs.norm, bufs.state, views.ffn_in, mlpScale(mods), normCfg(cfg));
    try mffn_buf.run(
        metal,
        bufs.ffn,
        bufs.norm,
        views.ffn_gate,
        views.ffn_up,
        views.ffn_down,
        cfg.tokens,
    );
    try math.residual(metal, bufs.state, bufs.ffn, views.ffn_out, mlpGate(mods), normCfg(cfg));
    readBack(bufs.state, state);
}

fn initBufs(metal: *mlinear.Context, state: []const f32) !Bufs {
    const bytes = std.mem.sliceAsBytes(state);
    var state_buf = try mbuffer.Buffer.fromBytes(metal.device, bytes);
    errdefer state_buf.deinit();
    var norm_buf = try empty(metal, state.len);
    errdefer norm_buf.deinit();
    var attn_buf = try empty(metal, state.len);
    errdefer attn_buf.deinit();
    const ffn_buf = try empty(metal, state.len);
    return .{ .state = state_buf, .norm = norm_buf, .attn = attn_buf, .ffn = ffn_buf };
}

fn readBack(buf: mbuffer.Buffer, state: []f32) void {
    c.zdraw_metal_read_buffer(
        buf.handle,
        std.mem.sliceAsBytes(state).ptr,
        state.len * @sizeOf(f32),
    );
}

fn check(state: []const f32, pos: []const zrope.Pos, cfg: Config) !void {
    if (cfg.tokens == 0 or cfg.hidden == 0) return error.InvalidShape;
    if (state.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (pos.len != cfg.tokens) return error.InvalidShape;
}

fn attnWeights(views: zblock.Views) mattn_buf.Weights {
    return .{
        .q = views.q,
        .k = views.k,
        .v = views.v,
        .o = views.proj,
        .q_norm = views.q_norm,
        .k_norm = views.k_norm,
    };
}

fn normCfg(cfg: Config) math.Config {
    return .{ .tokens = cfg.tokens, .hidden = cfg.hidden, .norm_eps = cfg.norm_eps };
}

fn empty(metal: *mlinear.Context, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(metal.device, count * @sizeOf(f32));
}
