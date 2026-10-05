//! Dev-only image-quality gate for zdraw.
//!
//! Generates a golden image (exact f32 GEMM) and a candidate (the mode under
//! test) with the SAME seed, so only kernel precision differs, then scores the
//! candidate against the golden with PSNR + SSIM. This is the gate that unlocks
//! the lossy speed levers (half, int8): a kernel that wins on speed must still
//! clear the perceptual bar here. Like refcheck/bench it loads the model and is
//! not part of the shipping CLI.
//!
//! Stack-policy modes share one loaded Runtime (golden and candidate differ
//! only in dispatch-time env), so an evaluation costs ~23 s instead of two
//! model loads; naive/half still take the two-load path.

const std = @import("std");

const c = @import("src/metal_c.zig");
const metrics = @import("src/quality_metrics.zig");
const model = @import("src/model.zig");
const runtime = @import("src/model_runtime.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const min_psnr: f64 = 38.0;
const min_ssim: f64 = 0.99;
const measured_ffn_from = "1";

const Mode = enum {
    naive,
    exact,
    half,
    stack_half,
    stack_attn,
    stack_ffn,
    stack_gateup,
    stack_down,
    stack_measured,
    stack_down_w8,
    stack_down_w8_last,
    stack_down_w8_last8,
    stack_measured_down_w8_last8,
};

const Env = struct {
    gemm: [*:0]const u8,
    stack: [*:0]const u8,
    measured_from: [*:0]const u8 = "0",
    last: [*:0]const u8 = "0",
};

fn modeEnv(mode: Mode) Env {
    return switch (mode) {
        .naive => .{ .gemm = "0", .stack = "exact" },
        .exact => .{ .gemm = "exact", .stack = "exact" },
        .half => .{ .gemm = "half", .stack = "half" },
        .stack_half => .{ .gemm = "exact", .stack = "half" },
        .stack_attn => .{ .gemm = "exact", .stack = "attn-half" },
        .stack_ffn => .{ .gemm = "exact", .stack = "ffn-half" },
        .stack_gateup => .{ .gemm = "exact", .stack = "gateup-half" },
        .stack_down => .{ .gemm = "exact", .stack = "down-half" },
        .stack_down_w8 => .{ .gemm = "exact", .stack = "down-w8" },
        .stack_down_w8_last => .{ .gemm = "exact", .stack = "down-w8", .last = "1" },
        .stack_down_w8_last8 => .{ .gemm = "exact", .stack = "down-w8", .last = "8" },
        .stack_measured_down_w8_last8 => .{
            .gemm = "exact",
            .stack = "measured-down-w8-last8",
            .measured_from = measured_ffn_from,
        },
        .stack_measured => .{
            .gemm = "exact",
            .stack = "measured-half",
            .measured_from = measured_ffn_from,
        },
    };
}

const Options = struct {
    weights: []const u8 = "",
    prompt: []const u8 = "a red boat at sunrise",
    width: u32 = 256,
    height: u32 = 256,
    steps: u32 = 8,
    seed: u64 = 42,
    mode: Mode = .naive,
    min_psnr: f64 = min_psnr,
    min_ssim: f64 = min_ssim,
    min_ms_ssim: f64 = 0.0,
    min_edge_psnr: f64 = 0.0,
    max_mae: f64 = 1.0e9,
    max_p99_abs: u8 = 255,
    max_abs: u8 = 255,
    ffn_from: ?[:0]const u8 = null,
    sweep: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const opts = parse(init) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("quality: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.weights.len == 0) return error.MissingWeights;

    var rt: ?runtime.Runtime = null;
    defer if (rt) |*r| r.deinit(io, allocator);
    if (singleLoad(opts.mode)) {
        applyEnv(modeEnv(.exact));
        rt = try runtime.Runtime.init(io, allocator, model.defaultKind(), opts.weights);
    }
    c.zdraw_metal_metrics_reset();
    const golden = try goldenPixels(io, allocator, &rt, opts);
    defer allocator.free(golden);

    if (opts.sweep) |list| return sweep(io, allocator, &rt, opts, golden, list);

    c.zdraw_metal_metrics_reset();
    var cand_env = modeEnv(opts.mode);
    if (opts.ffn_from) |value| cand_env.measured_from = value;
    const candidate = try pixelsFor(io, allocator, &rt, opts, cand_env);
    defer allocator.free(candidate);
    if (golden.len != candidate.len) return error.SizeMismatch;

    const mode_ok = activeModeMatches(opts.mode);
    const ps = metrics.psnr(golden, candidate);
    const ss = metrics.ssim(golden, candidate, opts.width, opts.height);
    const ms_ss = try metrics.msSsim(allocator, golden, candidate, opts.width, opts.height);
    const edge_ps = metrics.edgePsnr(golden, candidate, opts.width, opts.height);
    const mean_abs = metrics.mae(golden, candidate);
    const p99_abs = metrics.p99Abs(golden, candidate);
    const max_pixel_abs = metrics.maxAbs(golden, candidate);
    const ok = mode_ok and ps >= opts.min_psnr and ss >= opts.min_ssim and
        ms_ss >= opts.min_ms_ssim and edge_ps >= opts.min_edge_psnr and mean_abs <= opts.max_mae and
        p99_abs <= opts.max_p99_abs and max_pixel_abs <= opts.max_abs;

    std.debug.print(
        "\nquality: golden=exact  candidate={s}  {d}x{d}, {d} steps, seed {d}\n" ++
            "  mode active: {s}  gemm={d} exact={d} half={d} W8={d} MPS={d} fallback={d}\n" ++
            "  PSNR      {d:>7.2} dB  (>= {d:.0})\n" ++
            "  SSIM      {d:>7.4}     (>= {d:.2})\n" ++
            "  MS-SSIM   {d:>7.4}     (>= {d:.2})\n" ++
            "  edge PSNR {d:>7.2} dB  (>= {d:.0})\n" ++
            "  abs err   mean {d:.3} (<= {d:.3})  p99 {d} (<= {d})  max {d} (<= {d})\n" ++
            "  {s}\n",
        .{
            @tagName(opts.mode),            opts.width,                              opts.height,                      opts.steps,                      opts.seed,
            if (mode_ok) "yes" else "no",   c.zdraw_metal_gemm_count(),              c.zdraw_metal_gemm_exact_count(), c.zdraw_metal_gemm_half_count(), c.zdraw_metal_gemm_w8_count(),
            c.zdraw_metal_gemm_mps_count(), c.zdraw_metal_gemm_mps_fallback_count(), ps,                               opts.min_psnr,                   ss,
            opts.min_ssim,                  ms_ss,                                   opts.min_ms_ssim,                 edge_ps,                         opts.min_edge_psnr,
            mean_abs,                       opts.max_mae,                            p99_abs,                          opts.max_p99_abs,                max_pixel_abs,
            opts.max_abs,                   if (ok) "PASS" else "FAIL",
        },
    );
    if (!ok) std.process.exit(1);
}

fn activeModeMatches(mode: Mode) bool {
    const total = c.zdraw_metal_gemm_count();
    const exact = c.zdraw_metal_gemm_exact_count();
    const w8 = c.zdraw_metal_gemm_w8_count();
    // MPS dense substitutes for half-mode GEMMs (ZDRAW_DENSE), so either
    // counter proves the candidate exercised the policy's selected tensors.
    const half = c.zdraw_metal_gemm_half_count() + c.zdraw_metal_gemm_mps_count();
    return switch (mode) {
        .naive => total == 0,
        .exact => exact > 0 and half == 0 and w8 == 0,
        .half => half > 0,
        .stack_half,
        .stack_attn,
        .stack_ffn,
        .stack_gateup,
        .stack_down,
        .stack_measured,
        => exact > 0 and half > 0,
        .stack_down_w8 => exact > 0 and w8 > 0,
        .stack_down_w8_last => exact > 0 and w8 > 0,
        .stack_down_w8_last8 => exact > 0 and w8 > 0,
        .stack_measured_down_w8_last8 => exact > 0 and half > 0 and w8 > 0,
    };
}

// Owned pixels (caller frees). `mode_env` selects naive/exact/half via the same
// env switch the runtime reads in mlinear.Context.init.
fn applyEnv(env: Env) void {
    _ = setenv("ZDRAW_GEMM", env.gemm, 1);
    _ = setenv("ZDRAW_STACK_GEMM", env.stack, 1);
    _ = setenv("ZDRAW_STACK_HALF_LAST", env.last, 1);
    _ = setenv("ZDRAW_STACK_HALF_FROM", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_TO", "0", 1);
    _ = setenv("ZDRAW_STACK_MEASURED_FFN_FROM", env.measured_from, 1);
}

// Stack policies read their env at dispatch time, so golden and candidate can
// share one loaded Runtime; naive/half change ZDRAW_GEMM, which the Metal
// context caches at init, so those keep the two-load path.
fn singleLoad(mode: Mode) bool {
    return switch (mode) {
        .naive, .half => false,
        else => true,
    };
}

// Golden pixels are independent of the candidate env; cache them on disk so
// repeat evaluations skip the slow exact-mode generation entirely.
fn goldenPixels(io: std.Io, allocator: std.mem.Allocator, rt: *?runtime.Runtime, opts: Options) ![]u8 {
    var key_buf: [128]u8 = undefined;
    var hasher = std.hash.Wyhash.init(7);
    hasher.update("quality-golden-v2-env");
    hasher.update(opts.weights);
    hasher.update(opts.prompt);
    hasher.update(std.mem.asBytes(&opts.width));
    hasher.update(std.mem.asBytes(&opts.height));
    hasher.update(std.mem.asBytes(&opts.steps));
    hasher.update(std.mem.asBytes(&opts.seed));
    hashEnv(&hasher, "ZDRAW_DENSE");
    hashEnv(&hasher, "ZDRAW_ATTN");
    hashEnv(&hasher, "ZDRAW_ATTN_MFA");
    hashEnv(&hasher, "ZDRAW_MATH");
    hashEnv(&hasher, "ZDRAW_VAE");
    hashEnv(&hasher, "ZDRAW_STACK_W16");
    hashEnv(&hasher, "ZDRAW_ZPACK");
    const path = std.fmt.bufPrint(&key_buf, ".zig-cache/quality/golden-{x}.raw", .{hasher.final()}) catch unreachable;
    const want = @as(usize, opts.width) * opts.height * 3;
    if (std.Io.Dir.cwd().openFile(io, path, .{})) |file| {
        defer file.close(io);
        const bytes = try allocator.alloc(u8, want);
        errdefer allocator.free(bytes);
        var reader = file.reader(io, &.{});
        reader.interface.readSliceAll(bytes) catch {
            allocator.free(bytes);
            return try freshGolden(io, allocator, rt, opts, path);
        };
        std.debug.print("  golden: cached ({s})\n", .{path});
        return bytes;
    } else |_| {}
    return try freshGolden(io, allocator, rt, opts, path);
}

fn hashEnv(hasher: *std.hash.Wyhash, comptime name: [:0]const u8) void {
    hasher.update(name);
    hasher.update("=");
    if (std.c.getenv(name.ptr)) |raw| hasher.update(std.mem.span(raw));
    hasher.update("\n");
}

fn freshGolden(io: std.Io, allocator: std.mem.Allocator, rt: *?runtime.Runtime, opts: Options, path: []const u8) ![]u8 {
    const golden = try pixelsFor(io, allocator, rt, opts, modeEnv(.exact));
    if (!activeModeMatches(.exact)) return error.ExactModeUnavailable;
    std.Io.Dir.cwd().createDirPath(io, ".zig-cache/quality") catch {};
    if (std.Io.Dir.cwd().createFile(io, path, .{})) |file| {
        defer file.close(io);
        var writer = file.writer(io, &.{});
        writer.interface.writeAll(golden) catch {};
        writer.interface.flush() catch {};
    } else |_| {}
    return golden;
}

// One-process sweep: golden + runtime amortized across comma-separated
// ZDRAW_STACK_MEASURED_FFN_FROM values. Research tool: prints, no gate.
fn sweep(io: std.Io, allocator: std.mem.Allocator, rt: *?runtime.Runtime, opts: Options, golden: []const u8, list: []const u8) !void {
    var iter = std.mem.splitScalar(u8, list, ',');
    while (iter.next()) |value| {
        var env = modeEnv(opts.mode);
        var buf: [16]u8 = undefined;
        env.measured_from = try std.fmt.bufPrintZ(&buf, "{s}", .{value});
        c.zdraw_metal_metrics_reset();
        const candidate = try pixelsFor(io, allocator, rt, opts, env);
        defer allocator.free(candidate);
        const ps = metrics.psnr(golden, candidate);
        const ss = metrics.ssim(golden, candidate, opts.width, opts.height);
        std.debug.print("  sweep ffn-from={s}: PSNR {d:.2} SSIM {d:.4}\n", .{ value, ps, ss });
    }
}

fn pixelsFor(
    io: std.Io,
    allocator: std.mem.Allocator,
    rt: *?runtime.Runtime,
    opts: Options,
    env: Env,
) ![]u8 {
    applyEnv(env);
    if (rt.*) |*r| {
        const result = try r.generate(io, allocator, .{
            .prompt = opts.prompt,
            .width = opts.width,
            .height = opts.height,
            .steps = opts.steps,
            .seed = opts.seed,
        });
        return result.pixels;
    }
    return generate(io, allocator, opts);
}

fn generate(io: std.Io, allocator: std.mem.Allocator, opts: Options) ![]u8 {
    const result = try model.generate(io, allocator, .{
        .kind = model.defaultKind(),
        .weights_dir = opts.weights,
        .prompt = opts.prompt,
        .width = opts.width,
        .height = opts.height,
        .steps = opts.steps,
        .seed = opts.seed,
    });
    return result.pixels;
}

fn parse(init: std.process.Init) !Options {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    _ = iter.next();
    var out = Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) {
            out.weights = try need(&iter);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            out.prompt = try need(&iter);
        } else if (std.mem.eql(u8, arg, "--width")) {
            out.width = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--height")) {
            out.height = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--steps")) {
            out.steps = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            out.seed = try std.fmt.parseInt(u64, try need(&iter), 10);
        } else if (std.mem.eql(u8, arg, "--min-psnr")) {
            out.min_psnr = try std.fmt.parseFloat(f64, try need(&iter));
        } else if (std.mem.eql(u8, arg, "--min-ssim")) {
            out.min_ssim = try std.fmt.parseFloat(f64, try need(&iter));
        } else if (std.mem.eql(u8, arg, "--min-ms-ssim")) {
            out.min_ms_ssim = try std.fmt.parseFloat(f64, try need(&iter));
        } else if (std.mem.eql(u8, arg, "--min-edge-psnr")) {
            out.min_edge_psnr = try std.fmt.parseFloat(f64, try need(&iter));
        } else if (std.mem.eql(u8, arg, "--max-mae")) {
            out.max_mae = try std.fmt.parseFloat(f64, try need(&iter));
        } else if (std.mem.eql(u8, arg, "--max-p99-abs")) {
            out.max_p99_abs = try u8arg(&iter);
        } else if (std.mem.eql(u8, arg, "--max-abs")) {
            out.max_abs = try u8arg(&iter);
        } else if (std.mem.eql(u8, arg, "--mode")) {
            out.mode = try parseMode(try need(&iter));
        } else if (std.mem.eql(u8, arg, "--ffn-from")) {
            out.ffn_from = try init.gpa.dupeZ(u8, try need(&iter));
        } else if (std.mem.eql(u8, arg, "--ffn-sweep")) {
            out.sweep = try need(&iter);
        } else {
            return error.UnknownOption;
        }
    }
    return out;
}

fn parseMode(text: []const u8) !Mode {
    if (std.mem.eql(u8, text, "naive")) return .naive;
    if (std.mem.eql(u8, text, "exact")) return .exact;
    if (std.mem.eql(u8, text, "half")) return .half;
    if (std.mem.eql(u8, text, "stack-half")) return .stack_half;
    if (std.mem.eql(u8, text, "stack-attn")) return .stack_attn;
    if (std.mem.eql(u8, text, "stack-ffn")) return .stack_ffn;
    if (std.mem.eql(u8, text, "stack-gateup")) return .stack_gateup;
    if (std.mem.eql(u8, text, "stack-down")) return .stack_down;
    if (std.mem.eql(u8, text, "stack-measured")) return .stack_measured;
    if (std.mem.eql(u8, text, "stack-down-w8")) return .stack_down_w8;
    if (std.mem.eql(u8, text, "stack-down-w8-last")) return .stack_down_w8_last;
    if (std.mem.eql(u8, text, "stack-down-w8-last8")) return .stack_down_w8_last8;
    if (std.mem.eql(u8, text, "stack-measured-down-w8-last8")) {
        return .stack_measured_down_w8_last8;
    }
    return error.InvalidMode;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn u32arg(iter: *std.process.Args.Iterator) !u32 {
    return std.fmt.parseInt(u32, try need(iter), 10);
}

fn u8arg(iter: *std.process.Args.Iterator) !u8 {
    return std.fmt.parseInt(u8, try need(iter), 10);
}

fn usage() void {
    std.debug.print(
        \\zdraw quality gate
        \\  zig build quality -- --weights path/to/Z-Image-Turbo \
        \\    [--prompt "..."] [--width 256] [--height 256] [--steps 8] [--seed 42] \
        \\    [--min-psnr 38] [--min-ssim 0.99] [--min-ms-ssim 0] [--min-edge-psnr 0] \
        \\    [--max-mae N] [--max-p99-abs N] [--max-abs N] \
        \\    [--mode naive|exact|half|stack-half|stack-attn|stack-ffn|stack-gateup|stack-down|stack-measured|stack-down-w8|stack-down-w8-last|stack-down-w8-last8|stack-measured-down-w8-last8]
        \\
    , .{});
}
