//! Per-step stream assembly for Z-Image.
//!
//! Image/noise is timestep-dependent and recomputed every step. Caption/context
//! is timestep-independent and can be served by the exact caption cache.

const std = @import("std");

const mattn = @import("mattn.zig");
const metrics = @import("metrics.zig");
const mlinear = @import("mlinear.zig");
const zblock = @import("zblock.zig");
const zcap = @import("zcap_cache.zig");
const zconfig = @import("zimage_config.zig");
const zlayer = @import("zlayer.zig");
const zmem = @import("zmem.zig");
const zrope = @import("zrope.zig");
const zrun = @import("zrun.zig");
const zshape = @import("zshape.zig");
const zs = @import("zstep_shape.zig");
const zstreams = @import("zstreams.zig");
const ztx = @import("ztx.zig");

pub const Trace = struct {
    cap_embed: ?[]f32 = null,
    image: ?[]f32 = null,
    caption: ?[]f32 = null,
};

pub fn make(
    cache: ?*zcap.Cache,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
    adaln: []const f32,
    rope: zrope.Cache,
    trace: Trace,
) !zstreams.Streams {
    const image = try timedImage(metal, temp_allocator, input, tx, cfg, dims);
    const cap = try timedCaption(
        cache,
        metal,
        attn,
        cache_allocator,
        temp_allocator,
        input,
        tx,
        cfg,
        dims,
        rope,
        trace,
    );
    try timedNoise(metal, attn, cache_allocator, image, tx, adaln, cfg, rope);
    capture(trace.image, image.image);
    return .{
        .image = image.image,
        .img_pos = image.img_pos,
        .cap = cap.cap,
        .cap_pos = cap.cap_pos,
    };
}

fn timedImage(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !zstreams.Image {
    const start = metrics.now();
    const snap = metrics.snapshot();
    const image = try zstreams.makeImage(metal, allocator, input, tx, cfg, dims);
    record("streams-image", start, snap);
    return image;
}

fn timedCaption(
    cache: ?*zcap.Cache,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
    rope: zrope.Cache,
    trace: Trace,
) !zstreams.Caption {
    const start = metrics.now();
    const snap = metrics.snapshot();
    const cap = try caption(
        cache,
        metal,
        attn,
        cache_allocator,
        temp_allocator,
        input,
        tx,
        cfg,
        dims,
        rope,
        trace,
    );
    record("streams-caption", start, snap);
    return cap;
}

fn timedNoise(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    allocator: std.mem.Allocator,
    image: zstreams.Image,
    tx: *const ztx.Loaded,
    adaln: []const f32,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
) !void {
    const start = metrics.now();
    const snap = metrics.snapshot();
    const items = tx.noise.items;
    try refine(metal, attn, allocator, image.image, image.img_pos, items, adaln, cfg, rope);
    record("streams-noise", start, snap);
}

fn caption(
    cache: ?*zcap.Cache,
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
    rope: zrope.Cache,
    trace: Trace,
) !zstreams.Caption {
    return zcap.get(
        cache,
        metal,
        attn,
        temp_allocator,
        cache_allocator,
        input,
        tx,
        cfg,
        dims,
        rope,
        .{ .cap_embed = trace.cap_embed, .caption = trace.caption },
    );
}

// Refiner scratch reused across steps (a fresh ~21 MB alloc costs ~13 ms
// in page zeroing per step). Process-lifetime; rebuilt on shape change.
var noise_scratch: ?zmem.Owned = null;
var noise_count: usize = 0;

pub fn clearScratch(allocator: std.mem.Allocator) void {
    if (noise_scratch) |*scratch| scratch.deinit(allocator);
    noise_scratch = null;
    noise_count = 0;
}

fn noiseScratch(gpa: std.mem.Allocator, cfg: zlayer.Config, ffn: usize) !*const zlayer.Scratch {
    const need = zmem.count(cfg, ffn);
    if (noise_scratch == null or noise_count != need) {
        if (noise_scratch) |*old| old.deinit(gpa);
        noise_scratch = try zmem.init(gpa, cfg, ffn);
        noise_count = need;
    }
    return &noise_scratch.?.layer;
}

fn refine(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    allocator: std.mem.Allocator,
    state: []f32,
    pos: []const zrope.Pos,
    layers: []const zblock.Views,
    adaln: []const f32,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
) !void {
    const layer_cfg = try zs.layerCfg(state, cfg);
    const scratch = try noiseScratch(allocator, layer_cfg, zshape.ffnDim(cfg));
    try zrun.layers(
        allocator,
        metal,
        attn,
        state,
        layers,
        adaln,
        pos,
        scratch.*,
        layer_cfg,
        rope,
        .noise,
        null,
    );
}

fn capture(dst: ?[]f32, src: []const f32) void {
    if (dst) |buf| {
        if (buf.len == src.len) @memcpy(buf, src);
    }
}

fn record(name: []const u8, start: u64, metal: metrics.Counters) void {
    metrics.record(name, metrics.now() - start);
    metrics.recordMetal(name, metal);
}
