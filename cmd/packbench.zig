//! Dev-only W8 fused-dequant GEMM microbenchmark.

const std = @import("std");

const c = @import("zdraw").metal_c;
const shader = @import("zdraw").mw8_shader;

const group_size = 64;

const Options = struct {
    runs: usize = 20,
    smoke: bool = false,
};

const Shape = struct {
    m: usize,
    k: usize,
    n: usize,
    label: []const u8,
};

const Err = struct {
    max_abs: f64,
    mean_abs: f64,
    rel_l1: f64,
};

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("packbench: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    const device = c.zdraw_metal_create_device() orelse return error.MetalNotAvailable;
    defer c.zdraw_metal_release_device(device);
    const queue = c.zdraw_metal_create_queue(device) orelse return error.MetalNotAvailable;
    defer c.zdraw_metal_release_queue(queue);
    var err_buf: [4096]u8 = undefined;
    const pipe = c.zdraw_metal_compile(
        device,
        shader.source.ptr,
        "gemm_w8",
        &err_buf,
        err_buf.len,
    ) orelse return error.MetalCompileFailed;
    defer c.zdraw_metal_release_pipeline(pipe);
    const shapes = if (opts.smoke) smokeShapes() else modelShapes();
    std.debug.print("packbench W8 fused-dequant, group {d}\n", .{group_size});
    for (shapes) |shape| try benchShape(io, allocator, device, queue, pipe, shape, opts.runs);
}

fn benchShape(
    io: std.Io,
    allocator: std.mem.Allocator,
    device: *anyopaque,
    queue: *anyopaque,
    pipe: *anyopaque,
    shape: Shape,
    runs: usize,
) !void {
    var data = try makeData(allocator, shape);
    defer data.deinit(allocator);
    const bufs = try makeBuffers(device, data, shape);
    defer bufs.deinit();
    const ns = try timeRuns(io, allocator, queue, pipe, bufs, shape, runs);
    const gflops = flops(shape) / @as(f64, @floatFromInt(ns));
    if (shape.m * shape.k * shape.n <= 1 << 28) {
        const got = try readOutput(allocator, bufs.c, shape);
        defer allocator.free(got);
        const e = try checkCpu(allocator, data.a, data.pack, got, shape);
        printChecked(shape, ns, gflops, e);
    } else {
        printFast(shape, ns, gflops);
    }
}

const Data = struct {
    a: []f32,
    pack: []u8,

    fn deinit(self: *Data, allocator: std.mem.Allocator) void {
        allocator.free(self.pack);
        allocator.free(self.a);
    }
};

fn makeData(allocator: std.mem.Allocator, shape: Shape) !Data {
    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();
    const a = try allocator.alloc(f32, shape.m * shape.k);
    errdefer allocator.free(a);
    const w = try allocator.alloc(f32, shape.n * shape.k);
    defer allocator.free(w);
    for (a) |*value| value.* = (random.float(f32) - 0.5) * 2.0;
    for (w) |*value| value.* = (random.float(f32) - 0.5) * 0.2;
    return .{ .a = a, .pack = try packW8(allocator, w, shape) };
}

const Buffers = struct {
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,

    fn deinit(self: Buffers) void {
        c.zdraw_metal_release_buffer(self.c);
        c.zdraw_metal_release_buffer(self.w);
        c.zdraw_metal_release_buffer(self.a);
    }
};

fn makeBuffers(device: *anyopaque, data: Data, shape: Shape) !Buffers {
    const a_buf = c.zdraw_metal_create_buffer_with_data(
        device,
        std.mem.sliceAsBytes(data.a).ptr,
        data.a.len * @sizeOf(f32),
    ) orelse return error.MetalNotAvailable;
    errdefer c.zdraw_metal_release_buffer(a_buf);
    const w_buf = c.zdraw_metal_create_buffer_with_data(
        device,
        data.pack.ptr,
        data.pack.len,
    ) orelse return error.MetalNotAvailable;
    errdefer c.zdraw_metal_release_buffer(w_buf);
    const c_buf = c.zdraw_metal_create_buffer(device, outputBytes(shape)) orelse {
        return error.MetalNotAvailable;
    };
    return .{
        .a = a_buf,
        .w = w_buf,
        .c = c_buf,
    };
}

fn outputBytes(shape: Shape) usize {
    return shape.m * shape.n * @sizeOf(f32);
}

fn timeRuns(
    io: std.Io,
    allocator: std.mem.Allocator,
    queue: *anyopaque,
    pipe: *anyopaque,
    bufs: Buffers,
    shape: Shape,
    runs: usize,
) !u64 {
    const count = @max(runs, 1);
    const samples = try allocator.alloc(u64, count);
    defer allocator.free(samples);
    const params = paramsFor(shape);
    for (samples) |*sample| {
        const start = std.Io.Timestamp.now(io, .awake);
        const code = c.zdraw_metal_run_gemm(queue, pipe, bufs.a, bufs.w, bufs.c, &params);
        if (code != 0) return error.MetalDispatchFailed;
        sample.* = @intCast(start.untilNow(io, .awake).toNanoseconds());
    }
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return samples[samples.len / 2];
}

fn paramsFor(shape: Shape) c.GemmParams {
    return .{
        .m = @intCast(shape.m),
        .k = @intCast(shape.k),
        .n = @intCast(shape.n),
        .dtype = 0,
        .mode = 0,
        .weight_offset = 0,
    };
}

fn packW8(allocator: std.mem.Allocator, w: []const f32, shape: Shape) ![]u8 {
    const groups = groupsPerRow(shape.k);
    const q_bytes = shape.n * shape.k;
    const scale_bytes = shape.n * groups * @sizeOf(f32);
    const out = try allocator.alloc(u8, q_bytes + scale_bytes);
    for (0..shape.n) |row| {
        for (0..groups) |group| {
            try packGroup(out, w, shape, row, group, q_bytes);
        }
    }
    return out;
}

fn packGroup(out: []u8, w: []const f32, shape: Shape, row: usize, group: usize, scales: usize) !void {
    const from = group * group_size;
    const to = @min(from + group_size, shape.k);
    const s = groupScale(w, shape, row, from, to);
    writeScale(out, scales, row * groupsPerRow(shape.k) + group, s);
    for (from..to) |col| {
        const value = w[row * shape.k + col];
        out[row * shape.k + col] = encode(value, s);
    }
}

fn groupScale(w: []const f32, shape: Shape, row: usize, from: usize, to: usize) f32 {
    var max_abs: f32 = 0.0;
    for (from..to) |col| max_abs = @max(max_abs, abs(w[row * shape.k + col]));
    if (max_abs == 0.0) return 0.0;
    return max_abs / 127.0;
}

fn writeScale(out: []u8, offset: usize, index: usize, value: f32) void {
    @memcpy(out[offset + index * @sizeOf(f32) ..][0..4], std.mem.asBytes(&value));
}

fn encode(value: f32, scale: f32) u8 {
    if (scale == 0.0) return 0;
    const qf = @round(value / scale);
    const qi: i32 = @intFromFloat(@min(@max(qf, -127.0), 127.0));
    return @intCast(if (qi < 0) qi + 256 else qi);
}

fn readOutput(allocator: std.mem.Allocator, buf: *anyopaque, shape: Shape) ![]f32 {
    const out = try allocator.alloc(f32, shape.m * shape.n);
    c.zdraw_metal_read_buffer(buf, std.mem.sliceAsBytes(out).ptr, out.len * @sizeOf(f32));
    return out;
}

fn checkCpu(
    allocator: std.mem.Allocator,
    a: []const f32,
    pack: []const u8,
    got: []const f32,
    shape: Shape,
) !Err {
    const gold = try allocator.alloc(f32, got.len);
    defer allocator.free(gold);
    cpuGemm(gold, a, pack, shape);
    return diff(gold, got);
}

fn cpuGemm(out: []f32, a: []const f32, pack: []const u8, shape: Shape) void {
    for (0..shape.m) |m| {
        for (0..shape.n) |n| {
            var sum: f32 = 0.0;
            for (0..shape.k) |k| sum += half(a[m * shape.k + k]) * wValue(pack, shape, n, k);
            out[m * shape.n + n] = sum;
        }
    }
}

fn wValue(pack: []const u8, shape: Shape, n: usize, k: usize) f32 {
    const q = signed(pack[n * shape.k + k]);
    const scale_offset = shape.n * shape.k;
    const scale_index = n * groupsPerRow(shape.k) + k / group_size;
    const bytes = pack[scale_offset + scale_index * @sizeOf(f32) ..][0..4];
    return half(@as(f32, @floatFromInt(q)) * std.mem.bytesToValue(f32, bytes));
}

fn diff(gold: []const f32, got: []const f32) Err {
    var max_abs: f64 = 0.0;
    var sum_abs: f64 = 0.0;
    var sum_ref: f64 = 0.0;
    for (gold, got) |r, g| {
        const d = fabs(@as(f64, r) - @as(f64, g));
        max_abs = @max(max_abs, d);
        sum_abs += d;
        sum_ref += fabs(r);
    }
    return .{
        .max_abs = max_abs,
        .mean_abs = sum_abs / @as(f64, @floatFromInt(gold.len)),
        .rel_l1 = sum_abs / @max(sum_ref, 1.0e-12),
    };
}

fn signed(byte: u8) i16 {
    return if (byte < 128) @intCast(byte) else @as(i16, byte) - 256;
}

fn half(value: f32) f32 {
    const h: f16 = @floatCast(value);
    return @floatCast(h);
}

fn groupsPerRow(k: usize) usize {
    return (k + group_size - 1) / group_size;
}

fn flops(shape: Shape) f64 {
    return @as(f64, @floatFromInt(2 * shape.m * shape.k * shape.n));
}

fn abs(value: f32) f32 {
    return if (value < 0.0) -value else value;
}

fn fabs(value: f64) f64 {
    return if (value < 0.0) -value else value;
}

fn printChecked(shape: Shape, ns: u64, gflops: f64, e: Err) void {
    std.debug.print(
        "{s:>10} {d}x{d}x{d}: {d:.0} GF/s  {d:.3} ms  " ++
            "err max {d:.5} mean {d:.5} rel {d:.5}\n",
        .{ shape.label, shape.m, shape.k, shape.n, gflops, ms(ns), e.max_abs, e.mean_abs, e.rel_l1 },
    );
}

fn printFast(shape: Shape, ns: u64, gflops: f64) void {
    std.debug.print(
        "{s:>10} {d}x{d}x{d}: {d:.0} GF/s  {d:.3} ms\n",
        .{ shape.label, shape.m, shape.k, shape.n, gflops, ms(ns) },
    );
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn smokeShapes() []const Shape {
    return &.{.{ .m = 64, .k = 128, .n = 128, .label = "smoke" }};
}

fn modelShapes() []const Shape {
    return &.{
        .{ .m = 256, .k = 3840, .n = 3840, .label = "qkv" },
        .{ .m = 256, .k = 3840, .n = 10240, .label = "ffn-up" },
        .{ .m = 256, .k = 10240, .n = 3840, .label = "ffn-down" },
    };
}

fn parse(iter: *std.process.Args.Iterator) !Options {
    _ = iter.next();
    var out = Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--runs")) {
            out.runs = try usizeArg(iter);
        } else if (std.mem.eql(u8, arg, "--smoke")) {
            out.smoke = true;
        } else {
            return error.UnknownOption;
        }
    }
    return out;
}

fn usizeArg(iter: *std.process.Args.Iterator) !usize {
    return std.fmt.parseInt(usize, iter.next() orelse return error.MissingValue, 10);
}

fn usage() void {
    std.debug.print(
        \\zdraw packbench
        \\  zig build packbench -- [--smoke] [--runs 20]
        \\
    , .{});
}
