//! Z-Image denoising loop.
//!
//! The loop keeps the scheduler math separate from the transformer step. The
//! VAE decoder is still a later stage.

const std = @import("std");

const env = @import("../runtime/env.zig");
const mattn = @import("../metal/mattn.zig");
const metrics = @import("../metal/metrics.zig");
const mlinear = @import("../metal/mlinear.zig");
const progress = @import("../cli/progress.zig");
const rtload = @import("../runtime/runtime_load.zig");
const scheduler = @import("scheduler.zig");
const zconfig = @import("../zimage/zimage_config.zig");
const zpatch = @import("../zimage/zpatch.zig");
const zrope = @import("../zimage/zrope.zig");
const zstep = @import("../zimage/zstep.zig");
const zs = @import("../zimage/zstep_shape.zig");
const ztrace = @import("../control/ztrace_capture.zig");
const ztx = @import("../zimage/ztx.zig");

pub const Request = struct {
    cap: []const f32,
    shape: zpatch.Shape,
    schedule: scheduler.Schedule,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
    trace_capture: ?*ztrace.Capture = null,
};

pub const Prepared = struct {
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
};

const Options = struct {
    dump_pred: bool,
    clear_weight_cache_each_step: bool,

    fn fromEnv() Options {
        return .{
            .dump_pred = env.flag("ZDRAW_DUMP_PRED", false),
            .clear_weight_cache_each_step = env.flag("ZDRAW_CLEAR_WEIGHT_CACHE_EACH_STEP", false),
        };
    }
};

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    latents: []f32,
    request: Request,
) !void {
    var metal = try initMetal(io, allocator);
    defer if (metal) |*ctx| ctx.deinit();
    // Standalone path has no weights root; sidecar comes from env only.
    var sidecar = try rtload.loadZpack(io, allocator, "", &metal);
    defer if (sidecar) |*mapped| mapped.deinit(io);
    const metal_ptr = if (metal) |*ctx| ctx else null;
    var attn = try initAttention(io, allocator);
    defer if (attn) |*ctx| ctx.deinit();
    const attn_ptr = if (attn) |*ctx| ctx else null;

    try runPrepared(io, allocator, latents, request, .{
        .metal = metal_ptr,
        .attn = attn_ptr,
    });
}

pub fn runPrepared(
    io: std.Io,
    allocator: std.mem.Allocator,
    latents: []f32,
    request: Request,
    prepared: Prepared,
) !void {
    const pred = try allocator.alloc(f32, latents.len);
    defer allocator.free(pred);
    var cache = zstep.Cache{};
    defer cache.deinit(allocator);
    const options = Options.fromEnv();

    metrics.memtrace("denoise-start");
    for (request.schedule.timesteps, 0..) |time, idx| {
        try progress.step(io, allocator, idx + 1, request.schedule.timesteps.len);
        const step_start = std.Io.Timestamp.now(io, .awake);
        const step_metal = metrics.snapshot();
        const input = zstep.Input{
            .latent = latents,
            .cap = request.cap,
            .shape = request.shape,
            .time = timeNorm(time),
        };
        var trace_step = try beginTrace(request, input);
        defer if (trace_step) |*step| step.deinit();
        try zstep.runCached(&cache, allocator, .{
            .metal = prepared.metal,
            .attn = prepared.attn,
            .out = pred,
            .tx = request.tx,
            .cfg = request.cfg,
            .rope = request.rope,
            .trace = if (trace_step) |*step| step.trace() else null,
            .input = input,
        });
        if (trace_step) |*step| try commitTrace(io, request, input.time, step);
        metrics.record("denoise-step", @intCast(step_start.untilNow(io, .awake).toNanoseconds()));
        metrics.recordMetal("denoise-step", step_metal);
        if (options.dump_pred) dumpPred(io, pred, idx);
        apply(latents, pred, try stepSize(request.schedule, idx));
        memtraceStep(idx);
        // Diagnostic kill switch: drop the source-wrapper cache after each step.
        // If the per-step phys ramp vanishes with this on, the ramp is wrapper
        // accumulation (fixable); if it persists, it is real page residency.
        if (options.clear_weight_cache_each_step) {
            if (prepared.metal) |m| m.buffers.clearSources();
        }
    }
    if (request.trace_capture) |capture| try capture.finish(io);
}

fn beginTrace(
    request: Request,
    input: zstep.Input,
) !?ztrace.Step {
    const capture = request.trace_capture orelse return null;
    const dims = try zs.get(input, request.cfg);
    const step = try capture.begin(
        dims.total,
        dims.hidden,
        try request.tx.globals.t1_b.elems(),
    );
    return step;
}

fn commitTrace(
    io: std.Io,
    request: Request,
    time: f32,
    step: *const ztrace.Step,
) !void {
    const capture = request.trace_capture orelse return error.TraceNotEnabled;
    try capture.commit(io, time, step);
}

fn memtraceStep(idx: usize) void {
    if (!metrics.memtraceEnabled()) return;
    var buf: [32]u8 = undefined;
    const stage = std.fmt.bufPrint(&buf, "denoise-step-{d}", .{idx}) catch return;
    metrics.memtrace(stage);
}

fn initAttention(io: std.Io, allocator: std.mem.Allocator) !?mattn.Context {
    const ctx = mattn.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => {
            try progress.event(io, allocator, "Metal attention unavailable");
            return null;
        },
        else => return err,
    };
    try progress.event(io, allocator, "Metal attention ready");
    return ctx;
}

fn initMetal(io: std.Io, allocator: std.mem.Allocator) !?mlinear.Context {
    const ctx = mlinear.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => {
            try progress.event(io, allocator, "Metal unavailable; CPU linear fallback");
            return null;
        },
        else => return err,
    };
    try progress.event(io, allocator, "Metal denoise linear ready");
    return ctx;
}

pub fn fillNoise(latents: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (latents) |*value| value.* = random.floatNorm(f32);
}

/// Set the initial latent x0. `ZDRAW_LATENT_IN` (raw little-endian f32, the
/// same x0 dump refcheck consumes) injects a SHARED latent so cross-engine
/// comparisons are controlled rather than noise-confounded; otherwise the
/// seed-derived noise is used.
pub fn initLatents(io: std.Io, latents: []f32, seed: u64) !void {
    const raw = std.c.getenv("ZDRAW_LATENT_IN") orelse {
        fillNoise(latents, seed);
        return;
    };
    const path = std.mem.span(raw);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size != latents.len * @sizeOf(f32)) return error.LatentShapeMismatch;
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(std.mem.sliceAsBytes(latents));
}

fn dumpPred(io: std.Io, pred: []const f32, idx: usize) void {
    var buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "zpred_{d}.bin", .{idx}) catch return;
    const file = std.Io.Dir.cwd().createFile(io, path, .{}) catch return;
    defer file.close(io);
    var writer = file.writer(io, &.{});
    writer.interface.writeAll(std.mem.sliceAsBytes(pred)) catch return;
    writer.interface.flush() catch return;
}

pub fn apply(latents: []f32, pred: []const f32, size: f32) void {
    for (latents, pred) |*latent, value| latent.* += value * size;
}

pub fn stepSize(schedule: scheduler.Schedule, idx: usize) !f32 {
    if (idx + 1 >= schedule.sigmas.len) return error.InvalidSteps;
    return schedule.sigmas[idx] - schedule.sigmas[idx + 1];
}

pub fn timeNorm(timestep: f32) f32 {
    return (1000.0 - timestep) / 1000.0;
}

test "apply moves latents by positive sigma span" {
    var latents = [_]f32{ 1.0, 2.0 };
    apply(&latents, &.{ 3.0, -1.0 }, 0.5);

    try std.testing.expectApproxEqAbs(2.5, latents[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.5, latents[1], 0.0001);
}

test "step size is current sigma minus next sigma" {
    var timesteps = [_]f32{ 1000.0, 0.0 };
    var sigmas = [_]f32{ 1.0, 0.25, 0.0 };
    const sched = scheduler.Schedule{
        .timesteps = &timesteps,
        .sigmas = &sigmas,
    };

    try std.testing.expectApproxEqAbs(0.75, try stepSize(sched, 0), 0.0001);
    try std.testing.expectApproxEqAbs(0.25, try stepSize(sched, 1), 0.0001);
}

test "pipeline timestep is normalized before transformer" {
    try std.testing.expectApproxEqAbs(0.0, timeNorm(1000.0), 0.0001);
    try std.testing.expectApproxEqAbs(1.0, timeNorm(0.0), 0.0001);
}
