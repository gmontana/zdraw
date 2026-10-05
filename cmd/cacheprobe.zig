//! Dev-only cross-step cacheability probe.

const std = @import("std");

const discovery_trace = @import("zdraw").discovery_trace;
const mattn = @import("zdraw").mattn;
const metrics = @import("zdraw").metrics;
const mlinear = @import("zdraw").mlinear;
const runtime = @import("zdraw").model_runtime;
const scheduler = @import("zdraw").scheduler;
const zdenoise = @import("zdraw").zdenoise;
const zlatent = @import("zdraw").zlatent;
const zprobe = @import("zdraw").zprobe;
const zstep = @import("zdraw").zstep;
const zs = @import("zdraw").zstep_shape;
const ztext = @import("zdraw").zimage_text;

const Options = struct {
    weights: []const u8 = "",
    prompt: []const u8 = "a red boat at sunrise",
    out: []const u8 = "runs/cacheprobe.md",
    width: u32 = 64,
    height: u32 = 64,
    steps: u32 = 4,
    seed: u64 = 42,
    trace_dir: []const u8 = "",
    model_revision: []const u8 = "",
    engine_revision: []const u8 = "",
};

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();
    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    run(init.io, init.gpa, opts) catch |err| {
        std.debug.print("cacheprobe: error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    try std.Io.Dir.cwd().createDirPath(io, "runs");
    metrics.reset();
    var rt = try runtime.Runtime.init(io, allocator, .z_image_turbo, opts.weights);
    defer rt.deinit(io, allocator);
    var text = try encode(io, allocator, &rt, opts.prompt);
    defer text.deinit(allocator);
    var sched = try scheduler.makeZImage(allocator, opts.steps);
    defer sched.deinit(allocator);
    var work = try Work.init(allocator, &rt, text.embeds, opts);
    defer work.deinit(allocator);
    if (opts.trace_dir.len > 0) {
        try std.Io.Dir.cwd().createDirPath(io, opts.trace_dir);
    }
    try probeSteps(io, allocator, &rt, &work, sched, text.embeds, opts);
    try writeReport(io, allocator, opts, &rt, &work);
    if (opts.trace_dir.len > 0) {
        try writeTraceManifest(io, allocator, opts, &work);
    }
}

fn probeSteps(
    io: std.Io,
    allocator: std.mem.Allocator,
    rt: *runtime.Runtime,
    work: *Work,
    sched: scheduler.Schedule,
    cap: []const f32,
    opts: Options,
) !void {
    var cache = zstep.Cache{};
    defer cache.deinit(allocator);
    for (sched.timesteps, 0..) |time, idx| {
        const start = std.Io.Timestamp.now(io, .awake);
        @memset(work.curr, 0);
        const normalized_time = zdenoise.timeNorm(time);
        try oneStep(&cache, allocator, rt, work, cap, normalized_time, idx);
        if (opts.trace_dir.len > 0) {
            try writeTraceStep(io, allocator, opts, work, idx, normalized_time);
        }
        if (idx > 0) try work.accumulate();
        std.mem.swap([]f32, &work.prev, &work.curr);
        zdenoise.apply(work.latents, work.pred, try zdenoise.stepSize(sched, idx));
        metrics.record("probe-step", @intCast(start.untilNow(io, .awake).toNanoseconds()));
    }
}

fn oneStep(
    cache: *zstep.Cache,
    allocator: std.mem.Allocator,
    rt: *runtime.Runtime,
    work: *Work,
    cap: []const f32,
    time: f32,
    step: usize,
) !void {
    const trace_enabled = work.trace_digests.len > 0;
    try zstep.runCached(cache, allocator, .{
        .metal = linearPtr(rt),
        .attn = attnPtr(rt),
        .out = work.pred,
        .input = .{ .latent = work.latents, .cap = cap, .shape = work.shape, .time = time },
        .tx = &rt.tx,
        .cfg = rt.loaded.config.transformer,
        .rope = rt.rope,
        .trace = .{
            .adaln = if (trace_enabled) work.stepAdaln(step) else null,
            .positions = if (trace_enabled) work.positions else null,
            .layer_probe = .{
                .before = work.input,
                .after = work.curr,
                .state_len = work.state_len,
            },
        },
    });
}

fn encode(
    io: std.Io,
    allocator: std.mem.Allocator,
    rt: *runtime.Runtime,
    prompt: []const u8,
) !ztext.Encoded {
    return ztext.encodePrepared(io, allocator, .{
        .metal = linearPtr(rt),
        .attn = attnPtr(rt),
        .store = &rt.text_store,
    }, rt.loaded.config.text, rt.loaded.tokens, rt.loaded.indexes.text, prompt);
}

fn writeReport(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    rt: *runtime.Runtime,
    work: *Work,
) !void {
    var out = try std.ArrayList(u8).initCapacity(allocator, 4096);
    defer out.deinit(allocator);
    try appendHeader(allocator, &out, opts, rt, work);
    for (work.stats, 0..) |stat, layer| try appendLayer(allocator, &out, layer, stat);
    try out.appendSlice(allocator, "\n");
    try metrics.appendMarkdown(allocator, &out);
    try writeFile(io, opts.out, out.items);
    std.debug.print("cacheprobe: wrote {s}\n", .{opts.out});
}

fn appendHeader(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    opts: Options,
    rt: *runtime.Runtime,
    work: *Work,
) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "# zdraw Cacheability Probe\n\n" ++
            "- size: {d}x{d}\n- steps: {d}\n- seed: {d}\n" ++
            "- layers: {d}\n- state floats/layer: {d}\n\n" ++
            "| layer | mean rel drift | max rel drift | mean cosine |\n" ++
            "| ---: | ---: | ---: | ---: |\n",
        .{ opts.width, opts.height, opts.steps, opts.seed, rt.tx.layers.items.len, work.state_len },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendLayer(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    layer: usize,
    stat: zprobe.LayerStats,
) !void {
    const text = try std.fmt.allocPrint(allocator, "| {d} | {d:.6} | {d:.6} | {d:.6} |\n", .{
        layer,
        stat.meanRel(),
        stat.rel_max,
        stat.meanCos(),
    });
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

const TraceDigest = struct {
    normalized_time: f32,
    input_sha256: [64]u8,
    layers_sha256: [64]u8,
    adaln_sha256: [64]u8,
};

fn writeTraceStep(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    work: *Work,
    index: usize,
    normalized_time: f32,
) !void {
    if (index >= work.trace_digests.len) return error.InvalidTraceStep;
    const input_name = try traceName(allocator, index, "input");
    defer allocator.free(input_name);
    const layers_name = try traceName(allocator, index, "layers");
    defer allocator.free(layers_name);
    const adaln_name = try traceName(allocator, index, "adaln");
    defer allocator.free(adaln_name);
    const input_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ opts.trace_dir, input_name },
    );
    defer allocator.free(input_path);
    const layers_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ opts.trace_dir, layers_name },
    );
    defer allocator.free(layers_path);
    const adaln_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ opts.trace_dir, adaln_name },
    );
    defer allocator.free(adaln_path);
    const input_bytes = std.mem.sliceAsBytes(work.input);
    const layer_bytes = std.mem.sliceAsBytes(work.curr);
    const adaln_bytes = std.mem.sliceAsBytes(work.stepAdaln(index));
    try writeFile(io, input_path, input_bytes);
    try writeFile(io, layers_path, layer_bytes);
    try writeFile(io, adaln_path, adaln_bytes);
    work.trace_digests[index] = .{
        .normalized_time = normalized_time,
        .input_sha256 = hashBytes(input_bytes),
        .layers_sha256 = hashBytes(layer_bytes),
        .adaln_sha256 = hashBytes(adaln_bytes),
    };
}

fn writeTraceManifest(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    work: *const Work,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const steps = try a.alloc(discovery_trace.TraceStep, work.trace_digests.len);
    const input_bytes = try byteCount(work.input.len);
    const layer_bytes = try byteCount(work.curr.len);
    const adaln_bytes = try byteCount(work.adaln_len);
    const positions_name = "positions.u32le";
    const positions_path = try std.fmt.allocPrint(
        a,
        "{s}/{s}",
        .{ opts.trace_dir, positions_name },
    );
    const position_data = try encodePositions(a, work.positions);
    try writeFile(io, positions_path, position_data);
    const positions_sha256 = hashBytes(position_data);
    for (steps, work.trace_digests, 0..) |*step, *digest, index| {
        step.* = .{
            .index = @intCast(index),
            .normalized_time = digest.normalized_time,
            .layer_input = .{
                .path = try traceName(a, index, "input"),
                .sha256 = &digest.input_sha256,
                .byte_count = input_bytes,
            },
            .layer_outputs = .{
                .path = try traceName(a, index, "layers"),
                .sha256 = &digest.layers_sha256,
                .byte_count = layer_bytes,
            },
            .adaln = .{
                .path = try traceName(a, index, "adaln"),
                .sha256 = &digest.adaln_sha256,
                .byte_count = adaln_bytes,
            },
        };
    }
    const prompt_sha256 = hashBytes(opts.prompt);
    const artifact = discovery_trace.ArtifactV2{
        .schema_version = discovery_trace.schema_version_v2,
        .subject = "zdraw",
        .model = "z-image-turbo",
        .model_revision = opts.model_revision,
        .engine_revision = opts.engine_revision,
        .hook = "transformer.layer",
        .capture_mode = .non_resident_layer_probe,
        .performance_eligible = false,
        .workload = .{
            .width = opts.width,
            .height = opts.height,
            .steps = opts.steps,
            .seed = opts.seed,
            .prompt_sha256 = &prompt_sha256,
        },
        .tensor = .{
            .dtype = "f32-le",
            .layer_input_shape = .{ work.tokens, work.hidden },
            .layer_output_shape = .{ work.stats.len, work.tokens, work.hidden },
        },
        .replay = .{
            .adaln_dtype = "f32-le",
            .adaln_shape = .{work.adaln_len},
            .position_dtype = "u32-le",
            .position_shape = .{ work.tokens, 3 },
            .positions = .{
                .path = positions_name,
                .sha256 = &positions_sha256,
                .byte_count = position_data.len,
            },
        },
        .trace_steps = steps,
    };
    const json = try discovery_trace.toJson(a, artifact);
    const path = try std.fmt.allocPrint(a, "{s}/manifest.json", .{opts.trace_dir});
    try writeFile(io, path, json);
    std.debug.print("cacheprobe: wrote semantic trace {s}\n", .{path});
}

fn traceName(allocator: std.mem.Allocator, index: usize, kind: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "step-{d}-{s}.f32le", .{ index, kind });
}

fn byteCount(elements: usize) !u64 {
    return std.math.mul(u64, elements, @sizeOf(f32));
}

fn encodePositions(allocator: std.mem.Allocator, positions: []const [3]usize) ![]u8 {
    const count = try std.math.mul(usize, positions.len, 3);
    const bytes = try allocator.alloc(u8, try std.math.mul(usize, count, @sizeOf(u32)));
    var offset: usize = 0;
    for (positions) |position| {
        for (position) |coordinate| {
            const value = std.math.cast(u32, coordinate) orelse return error.PositionOverflow;
            std.mem.writeInt(u32, bytes[offset..][0..@sizeOf(u32)], value, .little);
            offset += @sizeOf(u32);
        }
    }
    return bytes;
}

fn hashBytes(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

const Work = struct {
    latents: []f32,
    pred: []f32,
    input: []f32,
    prev: []f32,
    curr: []f32,
    stats: []zprobe.LayerStats,
    trace_digests: []TraceDigest,
    trace_adaln: []f32,
    positions: [][3]usize,
    shape: @import("zdraw").zpatch.Shape,
    state_len: usize,
    tokens: usize,
    hidden: usize,
    adaln_len: usize,

    fn init(
        allocator: std.mem.Allocator,
        rt: *runtime.Runtime,
        cap: []const f32,
        opts: Options,
    ) !Work {
        const shape = try zlatent.shape(opts.width, opts.height, rt.loaded.config.transformer);
        const latents = try allocator.alloc(f32, zlatent.len(shape));
        errdefer allocator.free(latents);
        zdenoise.fillNoise(latents, opts.seed);
        const dims = try zs.get(.{
            .latent = latents,
            .cap = cap,
            .shape = shape,
            .time = 1.0,
        }, rt.loaded.config.transformer);
        return makeBuffers(
            allocator,
            rt,
            latents,
            shape,
            dims.total,
            dims.hidden,
            try rt.tx.globals.t1_b.elems(),
            if (opts.trace_dir.len > 0) opts.steps else 0,
        );
    }

    fn deinit(self: *Work, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.trace_adaln);
        allocator.free(self.trace_digests);
        allocator.free(self.stats);
        allocator.free(self.curr);
        allocator.free(self.prev);
        allocator.free(self.input);
        allocator.free(self.pred);
        allocator.free(self.latents);
    }

    fn accumulate(self: *Work) !void {
        for (self.stats, 0..) |*stat, layer| {
            try stat.add(
                zprobe.layerSlice(self.prev, self.state_len, layer),
                zprobe.layerSlice(self.curr, self.state_len, layer),
            );
        }
    }

    fn stepAdaln(self: *Work, step: usize) []f32 {
        const start = step * self.adaln_len;
        return self.trace_adaln[start..][0..self.adaln_len];
    }
};

fn makeBuffers(
    allocator: std.mem.Allocator,
    rt: *runtime.Runtime,
    latents: []f32,
    shape: @import("zdraw").zpatch.Shape,
    tokens: usize,
    hidden: usize,
    adaln_len: usize,
    trace_steps: u32,
) !Work {
    const state_len = try std.math.mul(usize, tokens, hidden);
    const layers = rt.tx.layers.items.len;
    const pred = try allocator.alloc(f32, latents.len);
    errdefer allocator.free(pred);
    const input = try allocator.alloc(f32, state_len);
    errdefer allocator.free(input);
    const prev = try allocator.alloc(f32, layers * state_len);
    errdefer allocator.free(prev);
    const curr = try allocator.alloc(f32, layers * state_len);
    errdefer allocator.free(curr);
    const stats = try allocator.alloc(zprobe.LayerStats, layers);
    errdefer allocator.free(stats);
    @memset(stats, .{});
    const trace_digests = try allocator.alloc(TraceDigest, trace_steps);
    errdefer allocator.free(trace_digests);
    const trace_adaln = try allocator.alloc(f32, trace_steps * adaln_len);
    errdefer allocator.free(trace_adaln);
    const positions = try allocator.alloc([3]usize, if (trace_steps > 0) tokens else 0);
    return .{
        .latents = latents,
        .pred = pred,
        .input = input,
        .prev = prev,
        .curr = curr,
        .stats = stats,
        .trace_digests = trace_digests,
        .trace_adaln = trace_adaln,
        .positions = positions,
        .shape = shape,
        .state_len = state_len,
        .tokens = tokens,
        .hidden = hidden,
        .adaln_len = adaln_len,
    };
}

fn linearPtr(rt: *runtime.Runtime) ?*mlinear.Context {
    return if (rt.linear) |*ctx| ctx else null;
}

fn attnPtr(rt: *runtime.Runtime) ?*mattn.Context {
    return if (rt.attn) |*ctx| ctx else null;
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
    while (iter.next()) |arg| {
        try parseArg(iter, &out, arg);
    }
    if (out.weights.len == 0) return error.MissingWeights;
    if (out.steps < 2) return error.InvalidSteps;
    if (out.trace_dir.len > 0 and
        (out.model_revision.len == 0 or out.engine_revision.len == 0))
    {
        return error.MissingTraceRevision;
    }
    return out;
}

fn parseArg(iter: *std.process.Args.Iterator, out: *Options, arg: []const u8) !void {
    if (std.mem.eql(u8, arg, "--weights")) {
        out.weights = try need(iter);
    } else if (std.mem.eql(u8, arg, "--prompt")) {
        out.prompt = try need(iter);
    } else if (std.mem.eql(u8, arg, "--out")) {
        out.out = try need(iter);
    } else if (std.mem.eql(u8, arg, "--width")) {
        out.width = try u32arg(iter);
    } else if (std.mem.eql(u8, arg, "--height")) {
        out.height = try u32arg(iter);
    } else if (std.mem.eql(u8, arg, "--steps")) {
        out.steps = try u32arg(iter);
    } else if (std.mem.eql(u8, arg, "--seed")) {
        out.seed = try std.fmt.parseInt(u64, try need(iter), 10);
    } else if (std.mem.eql(u8, arg, "--trace-dir")) {
        out.trace_dir = try need(iter);
    } else if (std.mem.eql(u8, arg, "--model-revision")) {
        out.model_revision = try need(iter);
    } else if (std.mem.eql(u8, arg, "--engine-revision")) {
        out.engine_revision = try need(iter);
    } else return error.UnknownOption;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn u32arg(iter: *std.process.Args.Iterator) !u32 {
    return std.fmt.parseInt(u32, try need(iter), 10);
}

fn usage() void {
    std.debug.print(
        \\zdraw cacheprobe
        \\  zig build cacheprobe -- --weights path/to/Z-Image-Turbo [--steps 4]
        \\    [--trace-dir path --model-revision id --engine-revision id]
        \\
    , .{});
}
