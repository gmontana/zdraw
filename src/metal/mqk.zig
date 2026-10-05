//! Host wrapper for the resident Q/K norm plus RoPE kernel.

const std = @import("std");

const c = @import("metal_c.zig");
const mattn = @import("mattn.zig");
const mbuffer = @import("mbuffer.zig");
const mres_util = @import("mres_util.zig");
const tensor = @import("../pack/tensor.zig");
const zrope = @import("../zimage/zrope.zig");

pub const Config = struct {
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
};

pub fn run(
    ctx: *mattn.Context,
    cache: *mbuffer.Cache,
    q: mbuffer.Buffer,
    k: mbuffer.Buffer,
    q_weight: tensor.View,
    k_weight: tensor.View,
    pos: []const zrope.Pos,
    rope: zrope.Cache,
    cfg: Config,
) !void {
    const pipe = ctx.qk_pipeline orelse return error.GemmUnavailable;
    try check(q_weight, k_weight, pos, rope, cfg);
    var q_tmp: ?mbuffer.Buffer = null;
    defer if (q_tmp) |*buf| buf.deinit();
    var k_tmp: ?mbuffer.Buffer = null;
    defer if (k_tmp) |*buf| buf.deinit();
    const q_bind = try cache.bindView(q_weight, &q_tmp);
    const k_bind = try cache.bindView(k_weight, &k_tmp);
    var pos_buf = try mbuffer.Buffer.fromBytes(ctx.device, std.mem.sliceAsBytes(pos));
    defer pos_buf.deinit();
    var rope_buf = try mbuffer.Buffer.fromBytes(ctx.device, rope.pairBytes());
    defer rope_buf.deinit();
    const params = try makeParams(q_weight, k_weight, q_bind, k_bind, rope, cfg);
    const code = c.zdraw_metal_run_qk_norm_rope(
        ctx.queue,
        pipe,
        q.handle,
        k.handle,
        q_bind.handle,
        k_bind.handle,
        pos_buf.handle,
        rope_buf.handle,
        &params,
        ctx.qk_threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

fn check(
    q_weight: tensor.View,
    k_weight: tensor.View,
    pos: []const zrope.Pos,
    rope: zrope.Cache,
    cfg: Config,
) !void {
    if (pos.len != cfg.tokens or cfg.head_dim == 0) return error.InvalidShape;
    if (cfg.head_dim != rope.cfg.dims[0] + rope.cfg.dims[1] + rope.cfg.dims[2]) {
        return error.InvalidShape;
    }
    if (try q_weight.elems() != cfg.head_dim) return error.InvalidShape;
    if (try k_weight.elems() != cfg.head_dim) return error.InvalidShape;
    try q_weight.check();
    try k_weight.check();
}

fn makeParams(
    q_weight: tensor.View,
    k_weight: tensor.View,
    q_bind: mbuffer.Bind,
    k_bind: mbuffer.Bind,
    rope: zrope.Cache,
    cfg: Config,
) !c.QkNormParams {
    const bases = rope.axisBases();
    return .{
        .tokens = try mres_util.toU32(cfg.tokens),
        .heads = try mres_util.toU32(cfg.heads),
        .kv_heads = try mres_util.toU32(cfg.kv_heads),
        .head_dim = try mres_util.toU32(cfg.head_dim),
        .q_dtype = try mres_util.dtype(q_weight.dtype),
        .k_dtype = try mres_util.dtype(k_weight.dtype),
        .dim0 = try mres_util.toU32(rope.cfg.dims[0]),
        .dim1 = try mres_util.toU32(rope.cfg.dims[1]),
        .dim2 = try mres_util.toU32(rope.cfg.dims[2]),
        .eps = cfg.norm_eps,
        .q_offset = try u64Fit(q_bind.offset),
        .k_offset = try u64Fit(k_bind.offset),
        .base0 = try u64Fit(bases[0]),
        .base1 = try u64Fit(bases[1]),
        .base2 = try u64Fit(bases[2]),
    };
}

fn u64Fit(value: usize) !u64 {
    return @intCast(value);
}
