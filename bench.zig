//! Dev-only benchmark runner for zdraw.
//!
//! Times a fixed generate case — one cold run (includes model load), then a warm
//! median over N runs — and prints per-stage wall-clock plus peak memory and the
//! Metal dispatch/readback round-trip counts. Not part of the user-facing CLI;
//! like refcheck it exists to measure, not to ship.

const std = @import("std");

const c = @import("src/metal_c.zig");
const image = @import("src/image.zig");
const metrics = @import("src/metrics.zig");
const model = @import("src/model.zig");
const runtime = @import("src/model_runtime.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const measured_ffn_from = "1";

const Mode = enum {
    exact,
    naive,
    half,
    stack_half,
    stack_attn,
    stack_ffn,
    stack_measured,
    stack_down_w8,
    stack_down_w8_last,
    stack_down_w8_last8,
    stack_measured_down_w8_last8,
};

const Options = struct {
    weights: []const u8 = "",
    prompt: []const u8 = "a red boat at sunrise",
    width: u32 = 512,
    height: u32 = 512,
    steps: u32 = 8,
    runs: u32 = 3,
    seed: u64 = 42,
    cached: bool = false,
    mode: Mode = .exact,
};

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();

    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("bench: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    applyMode(opts.mode);
    if (opts.cached) return runCached(io, allocator, opts);
    metrics.reset();
    const cold = try once(io, allocator, opts);
    metrics.reset();

    var i: usize = 0;
    while (i < opts.runs) : (i += 1) {
        c.zdraw_metal_metrics_reset(); // counters reflect the last warm run; timing accumulates
        const total = try once(io, allocator, opts);
        metrics.record("TOTAL", total);
    }

    try report(io, allocator, opts, cold);
}

fn runCached(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    metrics.reset();
    const cold_start = std.Io.Timestamp.now(io, .awake);
    var rt = try runtime.Runtime.init(io, allocator, model.defaultKind(), opts.weights);
    defer rt.deinit(io, allocator);
    const cold_gen = try onceRuntime(io, allocator, &rt, opts);
    const cold_total = elapsed(cold_start, io);
    _ = cold_gen;
    metrics.reset();

    var i: usize = 0;
    while (i < opts.runs) : (i += 1) {
        c.zdraw_metal_metrics_reset();
        const total = try onceRuntime(io, allocator, &rt, opts);
        metrics.record("TOTAL", total);
    }
    try report(io, allocator, opts, cold_total);
}

fn once(io: std.Io, allocator: std.mem.Allocator, opts: Options) !u64 {
    const start = std.Io.Timestamp.now(io, .awake);
    const result = try model.generate(io, allocator, .{
        .kind = model.defaultKind(),
        .weights_dir = opts.weights,
        .prompt = opts.prompt,
        .width = opts.width,
        .height = opts.height,
        .steps = opts.steps,
        .seed = opts.seed,
    });
    defer allocator.free(result.pixels);
    try pixelGate(result.pixels);
    const total: u64 = @intCast(start.untilNow(io, .awake).toNanoseconds());

    const png = std.Io.Timestamp.now(io, .awake);
    try image.writePng(
        io,
        allocator,
        "/tmp/zdraw-bench.png",
        result.pixels,
        result.width,
        result.height,
    );
    metrics.record("png-write", @intCast(png.untilNow(io, .awake).toNanoseconds()));
    return total;
}

fn onceRuntime(
    io: std.Io,
    allocator: std.mem.Allocator,
    rt: *runtime.Runtime,
    opts: Options,
) !u64 {
    const start = std.Io.Timestamp.now(io, .awake);
    const result = try rt.generate(io, allocator, .{
        .prompt = opts.prompt,
        .width = opts.width,
        .height = opts.height,
        .steps = opts.steps,
        .seed = opts.seed,
    });
    defer allocator.free(result.pixels);
    try pixelGate(result.pixels);
    const total = elapsed(start, io);
    const png = std.Io.Timestamp.now(io, .awake);
    try image.writePng(
        io,
        allocator,
        "/tmp/zdraw-bench.png",
        result.pixels,
        result.width,
        result.height,
    );
    metrics.record("png-write", elapsed(png, io));
    return total;
}

// Lesson 17: a gate that never looks at the output is not a gate. A real
// generation never produces a flat image; blank output means NaN latents.
fn pixelGate(pixels: []const u8) !void {
    if (pixels.len == 0) return error.BlankImage;
    var lo: u8 = 255;
    var hi: u8 = 0;
    for (pixels) |v| {
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    if (hi - lo < 8) {
        std.debug.print("bench: PIXEL GATE FAIL (flat image, range {d})\n", .{hi - lo});
        return error.BlankImage;
    }
}

fn report(io: std.Io, allocator: std.mem.Allocator, opts: Options, cold: u64) !void {
    std.debug.print(
        "\nbench: {d}x{d}, {d} steps, {d} warm runs (+1 cold){s}, mode {s}\n" ++
            "  cold run (incl. load): {d:.1} ms\n",
        .{
            opts.width,
            opts.height,
            opts.steps,
            opts.runs,
            if (opts.cached) " runtime-cached" else "",
            @tagName(opts.mode),
            milliseconds(cold),
        },
    );
    try metrics.report(io, allocator);
}

fn elapsed(start: std.Io.Timestamp, io: std.Io) u64 {
    return @intCast(start.untilNow(io, .awake).toNanoseconds());
}

fn milliseconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn parse(iter: *std.process.Args.Iterator) !Options {
    _ = iter.next();
    var out = Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) {
            out.weights = try need(iter);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            out.prompt = try need(iter);
        } else if (std.mem.eql(u8, arg, "--width")) {
            out.width = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--height")) {
            out.height = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--steps")) {
            out.steps = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--runs")) {
            out.runs = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            out.seed = try std.fmt.parseInt(u64, try need(iter), 10);
        } else if (std.mem.eql(u8, arg, "--runtime")) {
            out.cached = true;
        } else if (std.mem.eql(u8, arg, "--mode")) {
            out.mode = try parseMode(try need(iter));
        } else {
            return error.UnknownOption;
        }
    }
    if (out.weights.len == 0) return error.MissingWeights;
    if (out.runs == 0) return error.InvalidRuns;
    return out;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn u32arg(iter: *std.process.Args.Iterator) !u32 {
    return std.fmt.parseInt(u32, try need(iter), 10);
}

fn parseMode(text: []const u8) !Mode {
    if (std.mem.eql(u8, text, "exact")) return .exact;
    if (std.mem.eql(u8, text, "naive")) return .naive;
    if (std.mem.eql(u8, text, "half")) return .half;
    if (std.mem.eql(u8, text, "stack-half")) return .stack_half;
    if (std.mem.eql(u8, text, "stack-attn")) return .stack_attn;
    if (std.mem.eql(u8, text, "stack-ffn")) return .stack_ffn;
    if (std.mem.eql(u8, text, "stack-measured")) return .stack_measured;
    if (std.mem.eql(u8, text, "stack-down-w8")) return .stack_down_w8;
    if (std.mem.eql(u8, text, "stack-down-w8-last")) return .stack_down_w8_last;
    if (std.mem.eql(u8, text, "stack-down-w8-last8")) return .stack_down_w8_last8;
    if (std.mem.eql(u8, text, "stack-measured-down-w8-last8")) {
        return .stack_measured_down_w8_last8;
    }
    return error.InvalidMode;
}

fn applyMode(mode: Mode) void {
    _ = setenv("ZDRAW_STACK_HALF_LAST", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_FROM", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_TO", "0", 1);
    switch (mode) {
        .exact => set("exact", "exact"),
        .naive => set("0", "exact"),
        .half => set("half", "half"),
        .stack_half => set("exact", "half"),
        .stack_attn => set("exact", "attn-half"),
        .stack_ffn => set("exact", "ffn-half"),
        .stack_down_w8 => set("exact", "down-w8"),
        .stack_down_w8_last => {
            _ = setenv("ZDRAW_STACK_HALF_LAST", "1", 1);
            set("exact", "down-w8");
        },
        .stack_down_w8_last8 => {
            _ = setenv("ZDRAW_STACK_HALF_LAST", "8", 1);
            set("exact", "down-w8");
        },
        .stack_measured => {
            _ = setenv("ZDRAW_STACK_MEASURED_FFN_FROM", measured_ffn_from, 1);
            set("exact", "measured-half");
        },
        .stack_measured_down_w8_last8 => {
            _ = setenv("ZDRAW_STACK_MEASURED_FFN_FROM", measured_ffn_from, 1);
            set("exact", "measured-down-w8-last8");
        },
    }
}

fn set(gemm: [*:0]const u8, stack: [*:0]const u8) void {
    _ = setenv("ZDRAW_GEMM", gemm, 1);
    _ = setenv("ZDRAW_STACK_GEMM", stack, 1);
}

fn usage() void {
    std.debug.print(
        \\zdraw bench
        \\  zig build bench -- --weights path/to/Z-Image-Turbo \
        \\    [--prompt "..."] [--width 512] [--height 512] [--steps 8] \
        \\    [--runs 3] [--runtime] [--mode exact|stack-measured|stack-down-w8|stack-down-w8-last|stack-down-w8-last8|stack-measured-down-w8-last8]
        \\
    , .{});
}
