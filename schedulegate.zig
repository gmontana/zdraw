//! Dev-only schedule comparison runner.
//!
//! This is not a hard perceptual gate. It writes an N-step reference image and
//! a lower-step candidate image, then reports PSNR/SSIM as context for visual
//! review. A good 4-step image is allowed to differ from an 8-step image.

const std = @import("std");

const c = @import("src/metal_c.zig");
const image = @import("src/image.zig");
const metrics = @import("src/quality_metrics.zig");
const model = @import("src/model.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const measured_ffn_from = "5";

const Mode = enum {
    naive,
    exact,
    half,
    stack_measured,
    stack_measured_down_w8_last8,
};

const Env = struct {
    gemm: [*:0]const u8,
    stack: [*:0]const u8,
    measured_from: [*:0]const u8 = "0",
};

const Options = struct {
    weights: []const u8 = "",
    prompt: []const u8 = "a red boat at sunrise",
    out_dir: []const u8 = "runs/schedule-gate",
    width: u32 = 256,
    height: u32 = 256,
    reference_steps: u32 = 8,
    candidate_steps: u32 = 4,
    reference_shift: []const u8 = "3.0",
    candidate_shift: []const u8 = "3.0",
    candidate_shift_sweep: ?[]const u8 = null,
    seed: u64 = 42,
    mode: Mode = .stack_measured_down_w8_last8,
};

pub fn main(init: std.process.Init) !void {
    const opts = parse(init) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("schedulegate: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.weights.len == 0) return error.MissingWeights;
    if (opts.reference_steps == 0 or opts.candidate_steps == 0) {
        return error.InvalidSteps;
    }

    try std.Io.Dir.cwd().createDirPath(io, opts.out_dir);

    c.zdraw_metal_metrics_reset();
    const reference = try generate(
        io,
        allocator,
        opts,
        opts.reference_steps,
        opts.reference_shift,
        modeEnv(.exact),
    );
    defer allocator.free(reference);
    if (!activeModeMatches(.exact)) return error.ExactModeUnavailable;

    const reference_path = try std.fmt.allocPrint(
        allocator,
        "{s}/reference_exact_{d}step.png",
        .{ opts.out_dir, opts.reference_steps },
    );
    defer allocator.free(reference_path);
    try image.writePng(io, allocator, reference_path, reference, opts.width, opts.height);

    if (opts.candidate_shift_sweep) |list| {
        var iter = std.mem.splitScalar(u8, list, ',');
        while (iter.next()) |raw| {
            const shift = std.mem.trim(u8, raw, " \t\r\n");
            if (shift.len == 0) continue;
            try runCandidate(io, allocator, opts, reference, reference_path, shift);
        }
        return;
    }

    try runCandidate(io, allocator, opts, reference, reference_path, opts.candidate_shift);
}

fn runCandidate(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    reference: []const u8,
    reference_path: []const u8,
    candidate_shift: []const u8,
) !void {
    c.zdraw_metal_metrics_reset();
    const candidate = try generate(
        io,
        allocator,
        opts,
        opts.candidate_steps,
        candidate_shift,
        modeEnv(opts.mode),
    );
    defer allocator.free(candidate);
    if (!activeModeMatches(opts.mode)) return error.CandidateModeUnavailable;
    if (reference.len != candidate.len) return error.SizeMismatch;

    const candidate_path = try std.fmt.allocPrint(
        allocator,
        "{s}/candidate_{s}_{d}step_shift-{s}.png",
        .{ opts.out_dir, @tagName(opts.mode), opts.candidate_steps, candidate_shift },
    );
    defer allocator.free(candidate_path);

    try image.writePng(io, allocator, candidate_path, candidate, opts.width, opts.height);

    const ps = metrics.psnr(reference, candidate);
    const ss = metrics.ssim(reference, candidate, opts.width, opts.height);

    std.debug.print(
        "\nschedulegate: {d}x{d}, seed {d}\n" ++
            "  reference: exact {d} steps -> {s}\n" ++
            "  candidate: {s} {d} steps -> {s}\n" ++
            "  shifts: reference={s} candidate={s}\n" ++
            "  candidate active: yes  gemm={d} exact={d} half={d} W8={d}\n" ++
            "  cross-step PSNR {d:.2} dB  SSIM {d:.4}\n" ++
            "  NOTE: cross-step metrics are diagnostic only; visual/product review decides schedule acceptance.\n",
        .{
            opts.width,
            opts.height,
            opts.seed,
            opts.reference_steps,
            reference_path,
            @tagName(opts.mode),
            opts.candidate_steps,
            candidate_path,
            opts.reference_shift,
            candidate_shift,
            c.zdraw_metal_gemm_count(),
            c.zdraw_metal_gemm_exact_count(),
            c.zdraw_metal_gemm_half_count(),
            c.zdraw_metal_gemm_w8_count(),
            ps,
            ss,
        },
    );
}

fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    steps: u32,
    shift: []const u8,
    env: Env,
) ![]u8 {
    var shift_buf: [32]u8 = undefined;
    const shift_z = try std.fmt.bufPrintZ(&shift_buf, "{s}", .{shift});
    _ = setenv("ZDRAW_GEMM", env.gemm, 1);
    _ = setenv("ZDRAW_STACK_GEMM", env.stack, 1);
    _ = setenv("ZDRAW_SHIFT", shift_z.ptr, 1);
    _ = setenv("ZDRAW_STACK_HALF_LAST", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_FROM", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_TO", "0", 1);
    _ = setenv("ZDRAW_STACK_MEASURED_FFN_FROM", env.measured_from, 1);

    const result = try model.generate(io, allocator, .{
        .kind = model.defaultKind(),
        .weights_dir = opts.weights,
        .prompt = opts.prompt,
        .width = opts.width,
        .height = opts.height,
        .steps = steps,
        .seed = opts.seed,
    });
    return result.pixels;
}

fn modeEnv(mode: Mode) Env {
    return switch (mode) {
        .naive => .{ .gemm = "0", .stack = "exact" },
        .exact => .{ .gemm = "exact", .stack = "exact" },
        .half => .{ .gemm = "half", .stack = "half" },
        .stack_measured => .{
            .gemm = "exact",
            .stack = "measured-half",
            .measured_from = measured_ffn_from,
        },
        .stack_measured_down_w8_last8 => .{
            .gemm = "exact",
            .stack = "measured-down-w8-last8",
            .measured_from = measured_ffn_from,
        },
    };
}

fn activeModeMatches(mode: Mode) bool {
    const total = c.zdraw_metal_gemm_count();
    const exact = c.zdraw_metal_gemm_exact_count();
    const half = c.zdraw_metal_gemm_half_count();
    const w8 = c.zdraw_metal_gemm_w8_count();
    return switch (mode) {
        .naive => total == 0,
        .exact => exact > 0 and half == 0 and w8 == 0,
        .half => half > 0,
        .stack_measured => exact > 0 and half > 0,
        .stack_measured_down_w8_last8 => exact > 0 and half > 0 and w8 > 0,
    };
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
        } else if (std.mem.eql(u8, arg, "--out-dir")) {
            out.out_dir = try need(&iter);
        } else if (std.mem.eql(u8, arg, "--width")) {
            out.width = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--height")) {
            out.height = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--reference-steps")) {
            out.reference_steps = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--candidate-steps")) {
            out.candidate_steps = try u32arg(&iter);
        } else if (std.mem.eql(u8, arg, "--reference-shift")) {
            out.reference_shift = try need(&iter);
        } else if (std.mem.eql(u8, arg, "--candidate-shift")) {
            out.candidate_shift = try need(&iter);
        } else if (std.mem.eql(u8, arg, "--candidate-shift-sweep")) {
            out.candidate_shift_sweep = try need(&iter);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            out.seed = try std.fmt.parseInt(u64, try need(&iter), 10);
        } else if (std.mem.eql(u8, arg, "--mode")) {
            out.mode = try parseMode(try need(&iter));
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
    if (std.mem.eql(u8, text, "stack-measured")) return .stack_measured;
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

fn usage() void {
    std.debug.print(
        \\zdraw schedule comparison gate
        \\  zig build schedulegate -- --weights path/to/Z-Image-Turbo \
        \\    [--prompt "..."] [--out-dir runs/schedule-gate] \
        \\    [--width 256] [--height 256] [--reference-steps 8] \
        \\    [--candidate-steps 4] [--reference-shift 3.0] \
        \\    [--candidate-shift 3.0] [--candidate-shift-sweep 2.5,3.0,3.5] \
        \\    [--seed 42] \
        \\    [--mode exact|half|stack-measured|stack-measured-down-w8-last8]
        \\
    , .{});
}
