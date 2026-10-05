//! Dev-only layer-band precision sensitivity probe.

const std = @import("std");

const c = @import("src/metal_c.zig");
const image_metrics = @import("src/quality_metrics.zig");
const model = @import("src/model.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const Options = struct {
    weights: []const u8 = "",
    prompt: []const u8 = "a red boat at sunrise",
    out: []const u8 = "runs/sensitivity.md",
    mode: []const u8 = "ffn-half",
    width: u32 = 64,
    height: u32 = 64,
    steps: u32 = 2,
    seed: u64 = 42,
    layers: usize = 30,
    band: usize = 5,
};

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("sensitivity: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.weights.len == 0) return error.MissingWeights;
    if (opts.layers == 0 or opts.band == 0) return error.InvalidShape;
    try std.Io.Dir.cwd().createDirPath(io, "runs");
    try setExact();
    const golden = try generate(io, allocator, opts);
    defer allocator.free(golden);

    var report = try std.ArrayList(u8).initCapacity(allocator, 4096);
    defer report.deinit(allocator);
    try appendHeader(allocator, &report, opts);
    var from: usize = 0;
    while (from < opts.layers) : (from += opts.band) {
        const to = @min(from + opts.band, opts.layers);
        try setCandidate(opts.mode, from, to);
        c.zdraw_metal_metrics_reset();
        {
            const candidate = try generate(io, allocator, opts);
            defer allocator.free(candidate);
            try appendBand(allocator, &report, golden, candidate, opts, from, to);
        }
    }
    try writeFile(io, opts.out, report.items);
    std.debug.print("sensitivity: wrote {s}\n", .{opts.out});
}

fn appendHeader(allocator: std.mem.Allocator, out: *std.ArrayList(u8), opts: Options) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "# zdraw Sensitivity Probe\n\n" ++
            "- mode: `{s}`\n- size: {d}x{d}\n- steps: {d}\n- seed: {d}\n\n" ++
            "| layers | PSNR dB | SSIM | half GEMM | exact GEMM |\n" ++
            "| --- | ---: | ---: | ---: | ---: |\n",
        .{ opts.mode, opts.width, opts.height, opts.steps, opts.seed },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendBand(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    golden: []const u8,
    candidate: []const u8,
    opts: Options,
    from: usize,
    to: usize,
) !void {
    const ps = image_metrics.psnr(golden, candidate);
    const ss = image_metrics.ssim(golden, candidate, opts.width, opts.height);
    const text = try std.fmt.allocPrint(
        allocator,
        "| {d}-{d} | {d:.2} | {d:.4} | {d} | {d} |\n",
        .{ from, to, ps, ss, c.zdraw_metal_gemm_half_count(), c.zdraw_metal_gemm_exact_count() },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
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

fn setExact() !void {
    _ = setenv("ZDRAW_GEMM", "exact", 1);
    _ = setenv("ZDRAW_STACK_GEMM", "exact", 1);
    _ = setenv("ZDRAW_STACK_HALF_LAST", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_FROM", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_TO", "0", 1);
}

fn setCandidate(mode: []const u8, from: usize, to: usize) !void {
    var start_buf: [32]u8 = undefined;
    var end_buf: [32]u8 = undefined;
    const start = try std.fmt.bufPrintZ(&start_buf, "{d}", .{from});
    const end = try std.fmt.bufPrintZ(&end_buf, "{d}", .{to});
    _ = setenv("ZDRAW_GEMM", "exact", 1);
    _ = setenv("ZDRAW_STACK_GEMM", try modeZ(mode), 1);
    _ = setenv("ZDRAW_STACK_HALF_LAST", "0", 1);
    _ = setenv("ZDRAW_STACK_HALF_FROM", start.ptr, 1);
    _ = setenv("ZDRAW_STACK_HALF_TO", end.ptr, 1);
}

fn modeZ(mode: []const u8) ![*:0]const u8 {
    if (std.mem.eql(u8, mode, "half")) return "half";
    if (std.mem.eql(u8, mode, "attn-half")) return "attn-half";
    if (std.mem.eql(u8, mode, "ffn-half")) return "ffn-half";
    if (std.mem.eql(u8, mode, "gateup-half")) return "gateup-half";
    if (std.mem.eql(u8, mode, "down-half")) return "down-half";
    return error.InvalidMode;
}

fn parse(iter: *std.process.Args.Iterator) !Options {
    _ = iter.next();
    var out = Options{};
    while (iter.next()) |arg| try parseArg(iter, &out, arg);
    return out;
}

fn parseArg(iter: *std.process.Args.Iterator, out: *Options, arg: []const u8) !void {
    if (std.mem.eql(u8, arg, "--weights")) {
        out.weights = try need(iter);
    } else if (std.mem.eql(u8, arg, "--prompt")) {
        out.prompt = try need(iter);
    } else if (std.mem.eql(u8, arg, "--out")) {
        out.out = try need(iter);
    } else if (std.mem.eql(u8, arg, "--mode")) {
        out.mode = try need(iter);
    } else if (std.mem.eql(u8, arg, "--width")) {
        out.width = try u32arg(iter);
    } else if (std.mem.eql(u8, arg, "--height")) {
        out.height = try u32arg(iter);
    } else if (std.mem.eql(u8, arg, "--steps")) {
        out.steps = try u32arg(iter);
    } else if (std.mem.eql(u8, arg, "--seed")) {
        out.seed = try std.fmt.parseInt(u64, try need(iter), 10);
    } else if (std.mem.eql(u8, arg, "--layers")) {
        out.layers = try usizeArg(iter);
    } else if (std.mem.eql(u8, arg, "--band")) {
        out.band = try usizeArg(iter);
    } else {
        return error.UnknownOption;
    }
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn u32arg(iter: *std.process.Args.Iterator) !u32 {
    return std.fmt.parseInt(u32, try need(iter), 10);
}

fn usizeArg(iter: *std.process.Args.Iterator) !usize {
    return std.fmt.parseInt(usize, try need(iter), 10);
}

fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buf);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

fn usage() void {
    std.debug.print(
        \\zdraw sensitivity
        \\  zig build sensitivity -- --weights path/to/Z-Image-Turbo [--mode ffn-half]
        \\
    , .{});
}
