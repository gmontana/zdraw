//! Dev-only packed-weight quantization report.

const std = @import("std");

const tensor = @import("src/tensor.zig");
const zblock = @import("src/zblock.zig");
const zimage = @import("src/zimage.zig");
const ztx = @import("src/ztx.zig");

const Options = struct {
    weights: []const u8 = "",
    out: []const u8 = "runs/quantreport.md",
    group: usize = 64,
    row_stride: usize = 128,
    layer_from: usize = 0,
    layer_to: usize = 0,
};

const Kind = enum { q, k, v, proj, gate, up, down };

const Tensors = [_]Kind{ .q, .k, .v, .proj, .gate, .up, .down };

const Stats = struct {
    rows: usize,
    cols: usize,
    samples: usize,
    original_bytes: usize,
    w8_bytes: usize,
    w4_bytes: usize,
    max_abs: f64,
    rms: f64,
    w8_rel: f64,
    w4_rel: f64,
};

const Summary = struct {
    w4: usize = 0,
    w8: usize = 0,
    keep: usize = 0,

    fn add(self: *Summary, text: []const u8) void {
        if (std.mem.eql(u8, text, "W4-screen")) self.w4 += 1 else if (std.mem.eql(
            u8,
            text,
            "W8-screen",
        )) self.w8 += 1 else self.keep += 1;
    }
};

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("quantreport: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.weights.len == 0) return error.MissingWeights;
    if (opts.group == 0 or opts.row_stride == 0) return error.InvalidShape;
    try std.Io.Dir.cwd().createDirPath(io, "runs");
    var meta = try zimage.load(io, allocator, opts.weights);
    defer meta.deinit(allocator);
    var tx = try ztx.load(io, allocator, opts.weights, meta.config.transformer, meta.indexes.transformer);
    defer tx.deinit(io, allocator);

    var report = try std.ArrayList(u8).initCapacity(allocator, 8192);
    defer report.deinit(allocator);
    try appendHeader(allocator, &report, opts, tx.layers.items.len);
    var summary = Summary{};
    try appendLayers(allocator, &report, &summary, tx.layers.items, opts);
    try appendSummary(allocator, &report, summary);
    try writeFile(io, opts.out, report.items);
    std.debug.print("quantreport: wrote {s}\n", .{opts.out});
}

fn appendHeader(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    opts: Options,
    layers: usize,
) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "# zdraw Quantization Report\n\n" ++
            "- layers: {d}\n- group size: {d}\n- sampled row stride: {d}\n\n" ++
            "| layer | tensor | shape | bf16 MiB | W8 MiB | W4 MiB | " ++
            "samples | outlier | W8 rel | W4 rel | screen |\n" ++
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | " ++
            "---: | ---: | --- |\n",
        .{ layers, opts.group, opts.row_stride },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendLayers(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    summary: *Summary,
    layers: []const zblock.Views,
    opts: Options,
) !void {
    const to = if (opts.layer_to == 0) layers.len else @min(opts.layer_to, layers.len);
    var layer = @min(opts.layer_from, to);
    while (layer < to) : (layer += 1) {
        for (Tensors) |kind| try appendTensor(allocator, out, summary, layers[layer], opts, layer, kind);
    }
}

fn appendTensor(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    summary: *Summary,
    block: zblock.Views,
    opts: Options,
    layer: usize,
    kind: Kind,
) !void {
    const view = viewFor(block, kind);
    const stats = try analyze(view, opts);
    const screen = screenKind(stats);
    summary.add(screen);
    const text = try std.fmt.allocPrint(
        allocator,
        "| {d} | {s} | {d}x{d} | {d:.1} | {d:.1} | {d:.1} | " ++
            "{d} | {d:.1} | {d:.4} | {d:.4} | {s} |\n",
        .{
            layer,               kindText(kind),            stats.rows,
            stats.cols,          mib(stats.original_bytes), mib(stats.w8_bytes),
            mib(stats.w4_bytes), stats.samples,             outlier(stats),
            stats.w8_rel,        stats.w4_rel,              screen,
        },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn analyze(view: tensor.View, opts: Options) !Stats {
    try view.check();
    if (view.shape.len != 2) return error.InvalidShape;
    const rows = view.shape[0];
    const cols = view.shape[1];
    var acc = Accum{};
    var row: usize = 0;
    while (row < rows) : (row += opts.row_stride) {
        try analyzeRow(view, opts.group, row, cols, &acc);
    }
    return finish(view, opts.group, rows, cols, acc);
}

fn analyzeRow(view: tensor.View, group: usize, row: usize, cols: usize, acc: *Accum) !void {
    var col: usize = 0;
    while (col < cols) : (col += group) {
        const end = @min(col + group, cols);
        const max_abs = groupScale(view, row, col, end);
        try groupErr(view, row, col, end, max_abs, acc);
    }
}

const Accum = struct {
    signal_sq: f64 = 0.0,
    err8_sq: f64 = 0.0,
    err4_sq: f64 = 0.0,
    max_abs: f64 = 0.0,
    samples: usize = 0,
};

fn groupScale(view: tensor.View, row: usize, from: usize, to: usize) f64 {
    var max_abs: f64 = 0.0;
    for (from..to) |col| {
        const value = absF(view.atF32Unchecked(row * view.shape[1] + col));
        max_abs = @max(max_abs, value);
    }
    return max_abs;
}

fn groupErr(
    view: tensor.View,
    row: usize,
    from: usize,
    to: usize,
    max_abs: f64,
    acc: *Accum,
) !void {
    const s8 = scale(max_abs, 127.0);
    const s4 = scale(max_abs, 7.0);
    for (from..to) |col| {
        const value: f64 = view.atF32Unchecked(row * view.shape[1] + col);
        acc.signal_sq += value * value;
        acc.err8_sq += errSq(value, s8, 127.0);
        acc.err4_sq += errSq(value, s4, 7.0);
        acc.max_abs = @max(acc.max_abs, absF(value));
        acc.samples += 1;
    }
}

fn finish(view: tensor.View, group: usize, rows: usize, cols: usize, acc: Accum) !Stats {
    const elems = try view.elems();
    const groups = rows * ceilDiv(cols, group);
    const signal = @max(acc.signal_sq, 1.0e-12);
    return .{
        .rows = rows,
        .cols = cols,
        .samples = acc.samples,
        .original_bytes = elems * tensor.byteSize(view.dtype),
        .w8_bytes = elems + groups * @sizeOf(f32),
        .w4_bytes = ceilDiv(elems, 2) + groups * @sizeOf(f32),
        .max_abs = acc.max_abs,
        .rms = @sqrt(signal / @as(f64, @floatFromInt(@max(acc.samples, 1)))),
        .w8_rel = @sqrt(acc.err8_sq / signal),
        .w4_rel = @sqrt(acc.err4_sq / signal),
    };
}

fn screenKind(stats: Stats) []const u8 {
    if (stats.w4_rel <= 0.035 and outlier(stats) <= 80.0) return "W4-screen";
    if (stats.w8_rel <= 0.008) return "W8-screen";
    return "keep/correct";
}

fn errSq(value: f64, s: f64, qmax: f64) f64 {
    if (s == 0.0) return 0.0;
    const q = clamp(@round(value / s), -qmax, qmax);
    const diff = value - q * s;
    return diff * diff;
}

fn scale(max_abs: f64, qmax: f64) f64 {
    if (max_abs == 0.0) return 0.0;
    return max_abs / qmax;
}

fn outlier(stats: Stats) f64 {
    if (stats.rms == 0.0) return 0.0;
    return stats.max_abs / stats.rms;
}

fn clamp(value: f64, low: f64, high: f64) f64 {
    return @min(@max(value, low), high);
}

fn absF(value: f64) f64 {
    return if (value < 0.0) -value else value;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1024.0 / 1024.0;
}

fn viewFor(block: zblock.Views, kind: Kind) tensor.View {
    return switch (kind) {
        .q => block.q,
        .k => block.k,
        .v => block.v,
        .proj => block.proj,
        .gate => block.ffn_gate,
        .up => block.ffn_up,
        .down => block.ffn_down,
    };
}

fn kindText(kind: Kind) []const u8 {
    return switch (kind) {
        .q => "q",
        .k => "k",
        .v => "v",
        .proj => "proj",
        .gate => "gate",
        .up => "up",
        .down => "down",
    };
}

fn appendSummary(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    summary: Summary,
) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "\n## Screen Summary\n\n- W4-screen: {d}\n- W8-screen: {d}\n" ++
            "- keep/correct: {d}\n",
        .{ summary.w4, summary.w8, summary.keep },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buf);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
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
    } else if (std.mem.eql(u8, arg, "--out")) {
        out.out = try need(iter);
    } else if (std.mem.eql(u8, arg, "--group")) {
        out.group = try usizeArg(iter);
    } else if (std.mem.eql(u8, arg, "--row-stride")) {
        out.row_stride = try usizeArg(iter);
    } else if (std.mem.eql(u8, arg, "--layer-from")) {
        out.layer_from = try usizeArg(iter);
    } else if (std.mem.eql(u8, arg, "--layer-to")) {
        out.layer_to = try usizeArg(iter);
    } else if (std.mem.eql(u8, arg, "--layers")) {
        out.layer_to = try usizeArg(iter);
    } else {
        return error.UnknownOption;
    }
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn usizeArg(iter: *std.process.Args.Iterator) !usize {
    return std.fmt.parseInt(usize, try need(iter), 10);
}

fn usage() void {
    std.debug.print(
        \\zdraw quantreport
        \\  zig build quantreport -- --weights path/to/Z-Image-Turbo
        \\
    , .{});
}
