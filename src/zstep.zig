//! One standard Z-Image transformer step.

const std = @import("std");

const mattn = @import("mattn.zig");
const metrics = @import("metrics.zig");
const mlinear = @import("mlinear.zig");
const zcap = @import("zcap_cache.zig");
const zconfig = @import("zimage_config.zig");
const zrope = @import("zrope.zig");
const zprobe = @import("zprobe.zig");
const zstep_res = @import("zstep_res.zig");
const zstep_cpu = @import("zstep_cpu.zig");
const zstep_streams = @import("zstep_streams.zig");
const zstreams = @import("zstreams.zig");
const zs = @import("zstep_shape.zig");
const ztime = @import("ztime.zig");
const ztx = @import("ztx.zig");

pub const Input = zs.Input;
pub const Cache = zcap.Cache;

pub const Trace = struct {
    cap_embed: ?[]f32 = null,
    image: ?[]f32 = null,
    caption: ?[]f32 = null,
    unified: ?[]f32 = null,
    adaln: ?[]f32 = null,
    positions: ?[]zrope.Pos = null,
    layer_probe: ?zprobe.LayerProbe = null,
};

pub const Request = struct {
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    out: []f32,
    input: Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
    trace: ?Trace,
};

pub fn run(allocator: std.mem.Allocator, req: Request) !void {
    try runCached(null, allocator, req);
}

pub fn clearCache(allocator: std.mem.Allocator) void {
    zstep_streams.clearScratch(allocator);
}

pub fn runCached(cache: ?*Cache, allocator: std.mem.Allocator, req: Request) !void {
    const dims = try zs.get(req.input, req.cfg);
    if (req.out.len != req.input.latent.len) return error.InvalidShape;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const adaln = try timedTime(a, req);
    if (req.trace) |trace| capture(trace.adaln, adaln);
    const streams = try timedStreams(cache, allocator, a, req, dims, adaln);
    if (req.trace == null and try timedResident(allocator, a, req, streams, adaln, dims)) {
        return;
    }
    const slow_start = metrics.now();
    const slow_metal = metrics.snapshot();
    try zstep_cpu.run(allocator, a, .{
        .metal = req.metal,
        .attn = req.attn,
        .out = req.out,
        .tx = req.tx,
        .cfg = req.cfg,
        .rope = req.rope,
        .shape = req.input.shape,
        .trace_unified = if (req.trace) |tr| tr.unified else null,
        .trace_positions = if (req.trace) |tr| tr.positions else null,
        .layer_probe = if (req.trace) |tr| tr.layer_probe else null,
    }, streams, adaln, dims);
    record("zstep-cpu", slow_start, slow_metal);
}

fn capture(dst: ?[]f32, src: []const f32) void {
    if (dst) |buf| {
        if (buf.len == src.len) @memcpy(buf, src);
    }
}

fn timedTime(allocator: std.mem.Allocator, req: Request) ![]f32 {
    const start = metrics.now();
    const metal = metrics.snapshot();
    const out = try makeTime(req.metal, allocator, req.input.time, req.cfg, req.tx);
    record("zstep-time", start, metal);
    return out;
}

fn timedStreams(
    cache: ?*Cache,
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    req: Request,
    dims: zs.Dims,
    adaln: []const f32,
) !zstreams.Streams {
    const start = metrics.now();
    const metal = metrics.snapshot();
    const streams = try makeStreams(cache, cache_allocator, temp_allocator, req, dims, adaln);
    record("zstep-streams", start, metal);
    return streams;
}

var res_disabled: ?bool = null;

fn timedResident(
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    req: Request,
    streams: zstreams.Streams,
    adaln: []const f32,
    dims: zs.Dims,
) !bool {
    // Read once, not per step: the value is fixed after startup (applyEnv
    // runs before the first generate), and this sits in the denoise hot loop.
    const disabled = res_disabled orelse blk: {
        const d = if (std.c.getenv("ZDRAW_ZSTEP_RES")) |raw|
            std.mem.eql(u8, std.mem.span(raw), "0")
        else
            false;
        res_disabled = d;
        break :blk d;
    };
    if (disabled) return false;
    const start = metrics.now();
    const metal = metrics.snapshot();
    const ok = try zstep_res.run(cache_allocator, temp_allocator, .{
        .metal = req.metal,
        .attn = req.attn,
        .out = req.out,
        .input = req.input,
        .tx = req.tx,
        .cfg = req.cfg,
        .rope = req.rope,
    }, streams, adaln, dims);
    record("zstep-resident", start, metal);
    return ok;
}

fn record(name: []const u8, start: u64, metal: metrics.Counters) void {
    metrics.record(name, metrics.now() - start);
    metrics.recordMetal(name, metal);
}

fn makeStreams(
    cache: ?*Cache,
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    req: Request,
    dims: zs.Dims,
    adaln: []const f32,
) !zstreams.Streams {
    const t = req.trace orelse Trace{};
    return zstep_streams.make(
        cache,
        req.metal,
        req.attn,
        cache_allocator,
        temp_allocator,
        req.input,
        req.tx,
        req.cfg,
        dims,
        adaln,
        req.rope,
        .{ .cap_embed = t.cap_embed, .image = t.image, .caption = t.caption },
    );
}

fn makeTime(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    time: f32,
    cfg: zconfig.Transformer,
    tx: *const ztx.Loaded,
) ![]f32 {
    const len = try tx.globals.t1_b.elems();
    const scratch_len = tx.globals.t0_w.shape[0] + tx.globals.t0_w.shape[1];
    const out = try allocator.alloc(f32, len);
    const scratch = try allocator.alloc(f32, scratch_len);
    try ztime.run(metal, out, scratch, time, @floatCast(cfg.t_scale), .{
        .w0 = tx.globals.t0_w,
        .b0 = tx.globals.t0_b,
        .w1 = tx.globals.t1_w,
        .b1 = tx.globals.t1_b,
    });
    return out;
}
