//! CPU-visible fallback path for one Z-Image transformer step.

const std = @import("std");

const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const zconfig = @import("zimage_config.zig");
const zfinal = @import("zfinal.zig");
const zmem = @import("zmem.zig");
const zpatch = @import("zpatch.zig");
const zprobe = @import("zprobe.zig");
const zrope = @import("zrope.zig");
const zrun = @import("zrun.zig");
const zseq = @import("zseq.zig");
const zshape = @import("zshape.zig");
const zs = @import("zstep_shape.zig");
const zstreams = @import("zstreams.zig");
const ztx = @import("ztx.zig");
const zunify = @import("zunify.zig");

pub const Request = struct {
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    out: []f32,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
    shape: zpatch.Shape,
    trace_unified: ?[]f32,
    trace_positions: ?[]zrope.Pos = null,
    layer_probe: ?zprobe.LayerProbe = null,
};

pub fn run(
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    req: Request,
    streams: zstreams.Streams,
    adaln: []const f32,
    dims: zs.Dims,
) !void {
    const unified = try makeUnified(temp_allocator, streams, dims);
    capturePos(req.trace_positions, unified.pos);
    try refine(
        req.metal,
        req.attn,
        cache_allocator,
        unified.state,
        unified.pos,
        req.tx,
        adaln,
        req.cfg,
        req.rope,
        req.layer_probe,
    );
    capture(req.trace_unified, unified.state);
    const projected = try project(
        req.metal,
        temp_allocator,
        unified.state,
        adaln,
        req.tx,
        req.cfg,
        dims,
    );
    try zpatch.unpatchify(req.out, projected[0 .. dims.img_raw * dims.patch_dim], req.shape);
}

fn refine(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    allocator: std.mem.Allocator,
    state: []f32,
    pos: []const zrope.Pos,
    tx: *const ztx.Loaded,
    adaln: ?[]const f32,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
    probe: ?zprobe.LayerProbe,
) !void {
    const layer_cfg = try zs.layerCfg(state, cfg);
    var mem = try zmem.init(allocator, layer_cfg, zshape.ffnDim(cfg));
    defer mem.deinit(allocator);
    try zrun.layers(
        allocator,
        metal,
        attn,
        state,
        tx.layers.items,
        adaln,
        pos,
        mem.layer,
        layer_cfg,
        rope,
        .main,
        probe,
    );
}

const Unified = struct {
    state: []f32,
    pos: []zseq.Pos,
};

fn makeUnified(
    allocator: std.mem.Allocator,
    streams: zstreams.Streams,
    dims: zs.Dims,
) !Unified {
    const state = try allocator.alloc(f32, dims.total * dims.hidden);
    const pos = try allocator.alloc(zseq.Pos, dims.total);
    _ = try zunify.basic(
        state,
        pos,
        streams.image,
        streams.img_pos,
        streams.cap,
        streams.cap_pos,
        dims.hidden,
    );
    return .{ .state = state, .pos = pos };
}

fn project(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    state: []const f32,
    adaln: []const f32,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) ![]f32 {
    const out = try allocator.alloc(f32, dims.total * dims.patch_dim);
    const scratch = zfinal.Scratch{
        .cond = try allocator.alloc(f32, adaln.len),
        .scale = try allocator.alloc(f32, dims.hidden),
        .norm = try allocator.alloc(f32, dims.hidden),
        .batch = try allocator.alloc(f32, dims.total * dims.hidden),
    };
    try zfinal.run(metal, out, state, adaln, finalViews(tx), scratch, zs.finalCfg(dims, cfg));
    return out;
}

fn finalViews(tx: *const ztx.Loaded) zfinal.Views {
    return .{
        .mod_w = tx.globals.final_mod_w,
        .mod_b = tx.globals.final_mod_b,
        .linear_w = tx.globals.final_w,
        .linear_b = tx.globals.final_b,
    };
}

fn capture(dst: ?[]f32, src: []const f32) void {
    if (dst) |buf| {
        if (buf.len == src.len) @memcpy(buf, src);
    }
}

fn capturePos(dst: ?[]zrope.Pos, src: []const zrope.Pos) void {
    if (dst) |buf| {
        if (buf.len == src.len) @memcpy(buf, src);
    }
}
