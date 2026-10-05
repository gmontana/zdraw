const std = @import("std");

const c = @import("metal_c.zig");
const gmode = @import("../runtime/gemm_mode.zig");
const mattn = @import("mattn.zig");
const mbuffer = @import("mbuffer.zig");
const mgemm = @import("mgemm.zig");
const mbufs = @import("mres_buf.zig");
const mlinear = @import("mlinear.zig");
const mqk = @import("mqk.zig");
const util = @import("mres_util.zig");
const tensor = @import("../pack/tensor.zig");
const zrope = @import("../zimage/zrope.zig");

pub const Config = struct {
    tokens: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
};

pub const Weights = struct {
    q: tensor.View,
    k: tensor.View,
    v: tensor.View,
    o: tensor.View,
    q_norm: tensor.View,
    k_norm: tensor.View,
};

const Binds = struct { q: mbuffer.Bind, k: mbuffer.Bind, v: mbuffer.Bind };
const Params = struct { q: c.GemmParams, k: c.GemmParams, v: c.GemmParams };
const Temps = struct {
    q: ?mbuffer.Buffer = null,
    k: ?mbuffer.Buffer = null,
    v: ?mbuffer.Buffer = null,

    fn deinit(self: *Temps) void {
        if (self.q) |*buf| buf.deinit();
        if (self.k) |*buf| buf.deinit();
        if (self.v) |*buf| buf.deinit();
    }
};

pub fn run(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    out: []f32,
    input: []const f32,
    pos: []const zrope.Pos,
    weights: Weights,
    cfg: Config,
    rope: zrope.Cache,
) !void {
    try check(out, input, pos, weights, cfg);
    const pipe = gmode.pipeline(
        metal.gemm_mode,
        metal.gemm_exact_pipeline,
        metal.gemm_half_pipeline,
    ) orelse return error.GemmUnavailable;

    var bufs = try mbufs.init(metal, input, out.len, cfg.tokens, cfg.hidden);
    defer bufs.deinit();

    var temps = Temps{};
    defer temps.deinit();
    const binds = try bind(metal, weights, &temps);
    const ps = try makeParams(metal.gemm_mode, binds, weights, cfg);
    try qkv(metal, pipe, bufs.input, bufs.q, bufs.k, bufs.v, binds, ps);
    try qkNorm(metal, attn, bufs.q, bufs.k, weights, pos, rope, cfg);
    try attnRun(attn, bufs.mix, bufs.q, bufs.k, bufs.v, cfg);
    try mgemm.run(metal, bufs.out, bufs.mix, weights.o, cfg.tokens, cfg.hidden, cfg.hidden);
    c.zdraw_metal_read_buffer(
        bufs.out.handle,
        std.mem.sliceAsBytes(out).ptr,
        out.len * @sizeOf(f32),
    );
}

fn check(
    out: []const f32,
    input: []const f32,
    pos: []const zrope.Pos,
    weights: Weights,
    cfg: Config,
) !void {
    if (cfg.hidden != cfg.heads * cfg.head_dim) return error.InvalidShape;
    if (cfg.hidden != cfg.kv_heads * cfg.head_dim) return error.InvalidShape;
    if (out.len != input.len or out.len != cfg.tokens * cfg.hidden) return error.InvalidShape;
    if (pos.len != cfg.tokens) return error.InvalidShape;
    if (!util.fits(cfg.tokens, cfg.hidden, cfg.hidden)) return error.GemmShape;
    try sameShape(weights.q, weights.k);
    try sameShape(weights.q, weights.v);
    try sameShape(weights.q, weights.o);
}

fn bind(metal: *mlinear.Context, weights: Weights, temps: *Temps) !Binds {
    return .{
        .q = try metal.buffers.bindView(weights.q, &temps.q),
        .k = try metal.buffers.bindView(weights.k, &temps.k),
        .v = try metal.buffers.bindView(weights.v, &temps.v),
    };
}

fn qkNorm(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    q: mbuffer.Buffer,
    k: mbuffer.Buffer,
    weights: Weights,
    pos: []const zrope.Pos,
    rope: zrope.Cache,
    cfg: Config,
) !void {
    const qw = weights.q_norm;
    const kw = weights.k_norm;
    try mqk.run(attn, &metal.buffers, q, k, qw, kw, pos, rope, qkCfg(cfg));
}

fn makeParams(mode: gmode.Mode, binds: Binds, weights: Weights, cfg: Config) !Params {
    return .{
        .q = try param(mode, weights.q.dtype, binds.q, cfg),
        .k = try param(mode, weights.k.dtype, binds.k, cfg),
        .v = try param(mode, weights.v.dtype, binds.v, cfg),
    };
}

fn qkv(
    metal: *mlinear.Context,
    pipe: *anyopaque,
    input: mbuffer.Buffer,
    q_out: mbuffer.Buffer,
    k_out: mbuffer.Buffer,
    v_out: mbuffer.Buffer,
    binds: Binds,
    ps: Params,
) !void {
    const code = c.zdraw_metal_run_gemm_triple(
        metal.queue,
        pipe,
        input.handle,
        binds.q.handle,
        q_out.handle,
        &ps.q,
        binds.k.handle,
        k_out.handle,
        &ps.k,
        binds.v.handle,
        v_out.handle,
        &ps.v,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn attnRun(
    attn: *mattn.Context,
    out: mbuffer.Buffer,
    q: mbuffer.Buffer,
    k: mbuffer.Buffer,
    v: mbuffer.Buffer,
    cfg: Config,
) !void {
    const params = try attnParams(cfg);
    const picked = attn.pick(cfg.tokens, cfg.head_dim);
    const code = c.zdraw_metal_run_attention(
        attn.queue,
        picked.pipeline,
        q.handle,
        k.handle,
        v.handle,
        out.handle,
        &params,
        @intFromEnum(picked.kernel),
        picked.threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn param(mode: gmode.Mode, dtype: tensor.DType, bind_in: mbuffer.Bind, cfg: Config) !c.GemmParams {
    return .{
        .m = try util.toU32(cfg.tokens),
        .k = try util.toU32(cfg.hidden),
        .n = try util.toU32(cfg.hidden),
        .dtype = try util.dtype(dtype),
        .mode = util.mode(mode),
        .weight_offset = try util.toU64(bind_in.offset),
    };
}

fn attnParams(cfg: Config) !c.AttnParams {
    return .{
        .tokens = try util.toU32(cfg.tokens),
        .heads = try util.toU32(cfg.heads),
        .kv_heads = try util.toU32(cfg.kv_heads),
        .head_dim = try util.toU32(cfg.head_dim),
        .causal = 0,
    };
}

fn sameShape(a: tensor.View, b: tensor.View) !void {
    const as = try mlinear.shape(a);
    const bs = try mlinear.shape(b);
    if (as.rows != bs.rows or as.cols != bs.cols) return error.InvalidShape;
}

fn qkCfg(cfg: Config) mqk.Config {
    return .{
        .tokens = cfg.tokens,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .norm_eps = cfg.norm_eps,
    };
}
