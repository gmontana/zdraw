//! Exact caption-stream cache for Z-Image denoising.
//!
//! Caption embedding and context refinement do not depend on timestep or image
//! latents, so a generation can compute them once and reuse them every step.

const std = @import("std");

const mattn = @import("mattn.zig");
const mlinear = @import("mlinear.zig");
const zconfig = @import("zimage_config.zig");
const zmem = @import("zmem.zig");
const zrope = @import("zrope.zig");
const zrun = @import("zrun.zig");
const zseq = @import("zseq.zig");
const zshape = @import("zshape.zig");
const zstep_shape = @import("zstep_shape.zig");
const zstreams = @import("zstreams.zig");
const ztx = @import("ztx.zig");

pub const Trace = struct {
    cap_embed: ?[]f32 = null,
    caption: ?[]f32 = null,
};

pub const Cache = struct {
    cap: ?[]f32 = null,
    pos: ?[]zseq.Pos = null,

    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        if (self.cap) |buf| allocator.free(buf);
        if (self.pos) |buf| allocator.free(buf);
        self.* = .{};
    }
};

pub fn get(
    cache: ?*Cache,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    temp_allocator: std.mem.Allocator,
    cache_allocator: std.mem.Allocator,
    input: zstep_shape.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zstep_shape.Dims,
    rope: zrope.Cache,
    trace: Trace,
) !zstreams.Caption {
    if (cache) |c| {
        if (cached(c, dims)) return .{ .cap = c.cap.?, .cap_pos = c.pos.? };
    }

    const cap = try zstreams.makeCaption(metal, temp_allocator, input, tx, cfg, dims);
    capture(trace.cap_embed, cap.cap);
    try refine(metal, attn, temp_allocator, cap.cap, cap.cap_pos, tx, cfg, rope);
    capture(trace.caption, cap.cap);

    if (cache) |c| try store(c, cache_allocator, cap);
    return cap;
}

fn cached(cache: *Cache, dims: zstep_shape.Dims) bool {
    const cap = cache.cap orelse return false;
    const pos = cache.pos orelse return false;
    return cap.len == dims.cap_total * dims.hidden and pos.len == dims.cap_total;
}

fn store(cache: *Cache, allocator: std.mem.Allocator, cap: zstreams.Caption) !void {
    const cap_copy = try allocator.dupe(f32, cap.cap);
    errdefer allocator.free(cap_copy);
    const pos_copy = try allocator.dupe(zseq.Pos, cap.cap_pos);
    if (cache.cap) |buf| allocator.free(buf);
    if (cache.pos) |buf| allocator.free(buf);
    cache.cap = cap_copy;
    cache.pos = pos_copy;
}

fn refine(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    allocator: std.mem.Allocator,
    cap: []f32,
    pos: []const zseq.Pos,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
) !void {
    const layer_cfg = try zstep_shape.layerCfg(cap, cfg);
    var mem = try zmem.init(allocator, layer_cfg, zshape.ffnDim(cfg));
    defer mem.deinit(allocator);
    try zrun.layers(
        allocator,
        metal,
        attn,
        cap,
        tx.context.items,
        null,
        pos,
        mem.layer,
        layer_cfg,
        rope,
        .context,
        null,
    );
}

fn capture(dst: ?[]f32, src: []const f32) void {
    if (dst) |buf| {
        if (buf.len == src.len) @memcpy(buf, src);
    }
}
