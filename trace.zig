//! Dev-only engine trace for zdraw runtime decisions.
//!
//! Runs one real generation through the cached runtime path and writes a small
//! markdown report under `runs/`. This is the first diagnostic lab tool: it
//! records what the current engine actually does before we optimize it.

const std = @import("std");

const image = @import("src/image.zig");
const metrics = @import("src/metrics.zig");
const model = @import("src/model.zig");
const runtime = @import("src/model_runtime.zig");

const Options = struct {
    weights: []const u8 = "",
    prompt: []const u8 = "a red boat at sunrise",
    out: []const u8 = "runs/trace.md",
    image: []const u8 = "/tmp/zdraw-trace.png",
    width: u32 = 256,
    height: u32 = 256,
    steps: u32 = 8,
    seed: u64 = 42,
};

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();

    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("trace: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    try std.Io.Dir.cwd().createDirPath(io, "runs");

    const init_start = std.Io.Timestamp.now(io, .awake);
    var rt = try runtime.Runtime.init(io, allocator, model.defaultKind(), opts.weights);
    defer rt.deinit(io, allocator);
    const init_ns = elapsed(init_start, io);

    metrics.reset();
    const gen_start = std.Io.Timestamp.now(io, .awake);
    const result = try rt.generate(io, allocator, .{
        .prompt = opts.prompt,
        .width = opts.width,
        .height = opts.height,
        .steps = opts.steps,
        .seed = opts.seed,
    });
    defer allocator.free(result.pixels);
    const gen_ns = elapsed(gen_start, io);
    metrics.record("TOTAL", gen_ns);

    const png_start = std.Io.Timestamp.now(io, .awake);
    try image.writePng(io, allocator, opts.image, result.pixels, result.width, result.height);
    metrics.record("png-write", elapsed(png_start, io));

    const report = try makeReport(allocator, opts, init_ns, gen_ns);
    defer allocator.free(report);
    try writeFile(io, opts.out, report);
    try metrics.report(io, allocator);
    std.debug.print("trace: wrote {s}\n", .{opts.out});
}

fn makeReport(
    allocator: std.mem.Allocator,
    opts: Options,
    init_ns: u64,
    gen_ns: u64,
) ![]u8 {
    var out = try std.ArrayList(u8).initCapacity(allocator, 2048);
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "# zdraw Engine Trace\n\n");
    try appendRun(allocator, &out, opts, init_ns, gen_ns);
    try metrics.appendMarkdown(allocator, &out);
    try out.appendSlice(allocator, "\n## Diagnostic Read\n\n");
    try out.appendSlice(allocator, "- Treat readbacks as residency debt.\n");
    try out.appendSlice(allocator, "- Treat high weight bytes as packing debt.\n");
    try out.appendSlice(allocator, "- Optimize only after this report names the cost.\n");
    return out.toOwnedSlice(allocator);
}

fn appendRun(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    opts: Options,
    init_ns: u64,
    gen_ns: u64,
) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "## Run\n\n" ++
            "- weights: `{s}`\n- prompt: `{s}`\n- size: {d}x{d}\n" ++
            "- steps: {d}\n- seed: {d}\n- runtime init: {d:.2} ms\n" ++
            "- generation: {d:.2} ms\n- output image: `{s}`\n\n",
        .{
            opts.weights,
            opts.prompt,
            opts.width,
            opts.height,
            opts.steps,
            opts.seed,
            ms(init_ns),
            ms(gen_ns),
            opts.image,
        },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

fn parse(iter: *std.process.Args.Iterator) !Options {
    _ = iter.next();
    var out = Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) {
            out.weights = try need(iter);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            out.prompt = try need(iter);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out.out = try need(iter);
        } else if (std.mem.eql(u8, arg, "--image")) {
            out.image = try need(iter);
        } else if (std.mem.eql(u8, arg, "--width")) {
            out.width = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--height")) {
            out.height = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--steps")) {
            out.steps = try u32arg(iter);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            out.seed = try std.fmt.parseInt(u64, try need(iter), 10);
        } else return error.UnknownOption;
    }
    if (out.weights.len == 0) return error.MissingWeights;
    return out;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn u32arg(iter: *std.process.Args.Iterator) !u32 {
    return std.fmt.parseInt(u32, try need(iter), 10);
}

fn elapsed(start: std.Io.Timestamp, io: std.Io) u64 {
    return @intCast(start.untilNow(io, .awake).toNanoseconds());
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn usage() void {
    std.debug.print(
        \\zdraw engine trace
        \\  zig build trace -- --weights path/to/Z-Image-Turbo [--out runs/trace.md]
        \\
    , .{});
}
