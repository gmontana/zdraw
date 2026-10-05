//! Dev-only reference checker for Z-Image.
//!
//! This executable compares the Zig path against raw tensors dumped from the
//! official Python pipeline. It is not part of the user-facing CLI.

const std = @import("std");

const mattn = @import("zdraw").mattn;
const mconv = @import("zdraw").mconv;
const mlinear = @import("zdraw").mlinear;
const qenc = @import("zdraw").qwen_encoder;
const qscratch = @import("zdraw").qwen_scratch;
const scheduler = @import("zdraw").scheduler;
const shards = @import("zdraw").shards;
const tensor_file = @import("zdraw").tensor_file;
const tokenizer = @import("zdraw").tokenizer;
const vdecode = @import("zdraw").vdecode;
const vviews = @import("zdraw").vviews;
const zdenoise = @import("zdraw").zdenoise;
const zimage = @import("zdraw").zimage;
const zlatent = @import("zdraw").zlatent;
const zpatch = @import("zdraw").zpatch;
const zrope = @import("zdraw").zrope;
const zstep = @import("zdraw").zstep;
const ztx = @import("zdraw").ztx;

const max_prompt_tokens = 512;

const Backend = enum {
    auto,
    metal,
    cpu,
};

const Options = struct {
    weights: []const u8 = "",
    ref_dir: []const u8 = "",
    backend: Backend = .auto,
};

const Case = struct {
    prompt: []u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,

    fn deinit(self: *Case, allocator: std.mem.Allocator) void {
        allocator.free(self.prompt);
        self.* = undefined;
    }
};

const Text = struct {
    ids: tokenizer.Tokens,
    embeds: []f32,

    fn deinit(self: *Text, allocator: std.mem.Allocator) void {
        self.ids.deinit(allocator);
        allocator.free(self.embeds);
        self.* = undefined;
    }
};

const Metal = struct {
    linear: ?mlinear.Context = null,
    attn: ?mattn.Context = null,
    conv: ?mconv.Context = null,

    fn deinit(self: *Metal) void {
        if (self.conv) |*ctx| ctx.deinit();
        if (self.attn) |*ctx| ctx.deinit();
        if (self.linear) |*ctx| ctx.deinit();
        self.* = undefined;
    }
};

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

// The exactness anchor pins the full exact profile itself; it must never
// depend on the production defaults (which are the gated fast path).
fn pinExactProfile() void {
    _ = setenv("ZDRAW_GEMM", "exact", 1);
    _ = setenv("ZDRAW_STACK_GEMM", "exact", 1);
    _ = setenv("ZDRAW_VAE", "raw", 1);
    _ = setenv("ZDRAW_STACK_W16", "0", 1);
    _ = setenv("ZDRAW_DENSE", "off", 1);
    _ = setenv("ZDRAW_ATTN", "rows", 1);
}

pub fn main(init: std.process.Init) !void {
    pinExactProfile();
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();

    const opts = parse(&iter) catch |err| {
        usage();
        return err;
    };
    if (help(opts)) {
        usage();
        return;
    }

    run(init.io, init.gpa, opts) catch |err| {
        reportError(err);
        std.process.exit(1);
    };
}

fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    var failed = false;
    var case = try loadCase(io, allocator, opts.ref_dir);
    defer case.deinit(allocator);

    var loaded = try zimage.load(io, allocator, opts.weights);
    defer loaded.deinit(allocator);

    var metal = try initMetal(opts.backend);
    defer metal.deinit();
    var text = try checkText(io, allocator, opts, case, &loaded, &metal, &failed);
    defer text.deinit(allocator);

    try checkSchedule(io, allocator, opts, case, &failed);
    try checkDenoise(io, allocator, opts, case, &loaded, text, &metal, &failed);

    if (failed) return error.ReferenceMismatch;
    std.debug.print("refcheck: ok\n", .{});
}

fn checkText(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    case: Case,
    loaded: *const zimage.Loaded,
    metal: *Metal,
    failed: *bool,
) !Text {
    var ids = try loaded.tokens.encodePrompt(allocator, case.prompt, max_prompt_tokens);
    errdefer ids.deinit(allocator);
    try expectU32(io, allocator, opts.ref_dir, "ids.u32", ids.ids, failed);
    try expectU8(io, allocator, opts.ref_dir, "mask.u8", ids.mask, failed);

    const text = try encodeText(io, allocator, opts.weights, loaded, &ids, metal);
    errdefer allocator.free(text);
    try expectF32(io, allocator, opts.ref_dir, "text.f32", text, .text, failed);
    return .{ .ids = ids, .embeds = text };
}

fn encodeText(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    loaded: *const zimage.Loaded,
    ids: *const tokenizer.Tokens,
    metal: *Metal,
) ![]f32 {
    const used = countMask(ids.mask);
    const cfg = qenc.attnConfig(loaded.config.text, used);
    const out = try allocator.alloc(f32, used * cfg.hidden);
    errdefer allocator.free(out);

    var scratch = try qscratch.init(
        allocator,
        cfg,
        @intCast(loaded.config.text.intermediate_size),
    );
    defer scratch.deinit(allocator);

    const text_root = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{root});
    defer allocator.free(text_root);
    var store = try shards.open(io, allocator, text_root, loaded.indexes.text);
    defer store.deinit(io, allocator);

    const linear = if (metal.linear) |*ctx| ctx else null;
    const attn = if (metal.attn) |*ctx| ctx else null;
    try qenc.run(
        io,
        allocator,
        linear,
        attn,
        out,
        ids.ids[0..used],
        &store,
        loaded.indexes.text,
        loaded.config.text,
        .penultimate,
        &scratch,
        false,
        null,
        0,
    );
    return out;
}

fn checkSchedule(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    case: Case,
    failed: *bool,
) !void {
    var sched = try scheduler.makeZImage(allocator, case.steps);
    defer sched.deinit(allocator);
    try expectF32(io, allocator, opts.ref_dir, "t.f32", sched.timesteps, .schedule, failed);
    try expectF32(io, allocator, opts.ref_dir, "sigma.f32", sched.sigmas, .schedule, failed);
}

fn checkDenoise(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    case: Case,
    loaded: *const zimage.Loaded,
    text: Text,
    metal: *Metal,
    failed: *bool,
) !void {
    const x0 = try readOptF32(io, allocator, opts.ref_dir, "x0.f32") orelse return;
    defer allocator.free(x0);

    const shape = try zlatent.shape(case.width, case.height, loaded.config.transformer);
    if (x0.len != zlatent.len(shape)) {
        reportLen("x0.f32", x0.len, zlatent.len(shape));
        failed.* = true;
        return;
    }

    const latents = try allocator.dupe(f32, x0);
    defer allocator.free(latents);
    const pred = try allocator.alloc(f32, latents.len);
    defer allocator.free(pred);

    var sched = try scheduler.makeZImage(allocator, case.steps);
    defer sched.deinit(allocator);
    var rope = try makeRope(allocator, loaded);
    defer rope.deinit(allocator);
    var tx = try ztx.load(io, allocator, opts.weights, loaded.config.transformer, loaded.indexes.transformer);
    defer tx.deinit(io, allocator);
    var cache = zstep.Cache{};
    defer cache.deinit(allocator);

    const want_cape = try readOptF32(io, allocator, opts.ref_dir, "cape.f32");
    defer if (want_cape) |w| allocator.free(w);
    const want_imgr = try readOptF32(io, allocator, opts.ref_dir, "imgr.f32");
    defer if (want_imgr) |w| allocator.free(w);
    const want_capr = try readOptF32(io, allocator, opts.ref_dir, "capr.f32");
    defer if (want_capr) |w| allocator.free(w);
    const want_hlast = try readOptF32(io, allocator, opts.ref_dir, "hlast.f32");
    defer if (want_hlast) |w| allocator.free(w);

    var trace = zstep.Trace{};
    if (want_cape) |w| trace.cap_embed = try allocator.alloc(f32, w.len);
    defer if (trace.cap_embed) |b| allocator.free(b);
    if (want_imgr) |w| trace.image = try allocator.alloc(f32, w.len);
    defer if (trace.image) |b| allocator.free(b);
    if (want_capr) |w| trace.caption = try allocator.alloc(f32, w.len);
    defer if (trace.caption) |b| allocator.free(b);
    if (want_hlast) |w| trace.unified = try allocator.alloc(f32, w.len);
    defer if (trace.unified) |b| allocator.free(b);

    try firstStep(allocator, &cache, latents, pred, text.embeds, shape, sched, &tx, loaded, rope, metal, trace);

    if (want_cape) |w| stageStat("cape.f32", trace.cap_embed.?, w, failed);
    if (want_imgr) |w| stageStat("imgr.f32", trace.image.?, w, failed);
    if (want_capr) |w| stageStat("capr.f32", trace.caption.?, w, failed);
    if (want_hlast) |w| stageStat("hlast.f32", trace.unified.?, w, failed);
    try expectF32(io, allocator, opts.ref_dir, "pred0.f32", pred, .latent, failed);
    try expectF32(io, allocator, opts.ref_dir, "x1.f32", latents, .latent, failed);
    try restSteps(allocator, &cache, latents, pred, text.embeds, shape, sched, &tx, loaded, rope, metal);
    try expectF32(io, allocator, opts.ref_dir, "xf.f32", latents, .latent, failed);
    try checkVae(io, allocator, opts, case, latents, shape, metal, failed);
}

fn firstStep(
    allocator: std.mem.Allocator,
    cache: *zstep.Cache,
    latents: []f32,
    pred: []f32,
    text: []const f32,
    shape: zpatch.Shape,
    sched: scheduler.Schedule,
    tx: *const ztx.Loaded,
    loaded: *const zimage.Loaded,
    rope: zrope.Cache,
    metal: *Metal,
    trace: ?zstep.Trace,
) !void {
    try oneStep(cache, allocator, pred, latents, text, shape, sched.timesteps[0], tx, loaded, rope, metal, trace);
    zdenoise.apply(latents, pred, try zdenoise.stepSize(sched, 0));
}

fn restSteps(
    allocator: std.mem.Allocator,
    cache: *zstep.Cache,
    latents: []f32,
    pred: []f32,
    text: []const f32,
    shape: zpatch.Shape,
    sched: scheduler.Schedule,
    tx: *const ztx.Loaded,
    loaded: *const zimage.Loaded,
    rope: zrope.Cache,
    metal: *Metal,
) !void {
    for (sched.timesteps[1..], 1..) |time, idx| {
        try oneStep(cache, allocator, pred, latents, text, shape, time, tx, loaded, rope, metal, null);
        zdenoise.apply(latents, pred, try zdenoise.stepSize(sched, idx));
    }
}

fn oneStep(
    cache: *zstep.Cache,
    allocator: std.mem.Allocator,
    pred: []f32,
    latents: []const f32,
    text: []const f32,
    shape: zpatch.Shape,
    time: f32,
    tx: *const ztx.Loaded,
    loaded: *const zimage.Loaded,
    rope: zrope.Cache,
    metal: *Metal,
    trace: ?zstep.Trace,
) !void {
    const linear = if (metal.linear) |*ctx| ctx else null;
    const attn = if (metal.attn) |*ctx| ctx else null;
    try zstep.runCached(cache, allocator, .{
        .metal = linear,
        .attn = attn,
        .out = pred,
        .input = .{
            .latent = latents,
            .cap = text,
            .shape = shape,
            .time = zdenoise.timeNorm(time),
        },
        .tx = tx,
        .cfg = loaded.config.transformer,
        .rope = rope,
        .trace = trace,
    });
}

fn checkVae(
    io: std.Io,
    allocator: std.mem.Allocator,
    opts: Options,
    case: Case,
    latents: []const f32,
    shape: zpatch.Shape,
    metal: *Metal,
    failed: *bool,
) !void {
    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/vae/diffusion_pytorch_model.safetensors",
        .{opts.weights},
    );
    defer allocator.free(path);
    var vae = try tensor_file.open(io, allocator, path);
    defer vae.deinit(io, allocator);

    const views = try vviews.load(&vae);
    const sample = try allocator.alloc(f32, @as(usize, case.width) * case.height * 3);
    defer allocator.free(sample);
    const conv = if (metal.conv) |*ctx| ctx else null;
    const linear = if (metal.linear) |*ctx| ctx else null;
    const attn = if (metal.attn) |*ctx| ctx else null;
    try vdecode.run(conv, linear, attn, null, allocator, sample, latents, views, .{
        .height = shape.height,
        .width = shape.width,
    }, null);
    try expectF32(io, allocator, opts.ref_dir, "rgb.f32", sample, .rgb, failed);
}

fn makeRope(allocator: std.mem.Allocator, loaded: *const zimage.Loaded) !zrope.Cache {
    const cfg = loaded.config.transformer;
    return zrope.Cache.init(allocator, .{
        .dims = .{ cfg.axes_dims[0], cfg.axes_dims[1], cfg.axes_dims[2] },
        .lens = .{ cfg.axes_lens[0], cfg.axes_lens[1], cfg.axes_lens[2] },
        .theta = @floatCast(cfg.rope_theta),
    });
}

fn initMetal(backend: Backend) !Metal {
    if (backend == .cpu) return .{};
    var out = Metal{};
    out.linear = try initOne(mlinear.Context, backend);
    errdefer out.deinit();
    out.attn = try initOne(mattn.Context, backend);
    out.conv = try initOne(mconv.Context, backend);
    return out;
}

fn initOne(comptime T: type, backend: Backend) !?T {
    return T.init() catch |err| switch (err) {
        error.MetalNotAvailable => if (backend == .auto) null else return err,
        else => if (backend == .auto) null else return err,
    };
}

const Limits = enum {
    schedule,
    text,
    latent,
    rgb,
};

fn expectF32(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: []const u8,
    name: []const u8,
    got: []const f32,
    limits: Limits,
    failed: *bool,
) !void {
    const want = try readOptF32(io, allocator, dir, name) orelse return;
    defer allocator.free(want);
    if (want.len != got.len) {
        reportLen(name, got.len, want.len);
        failed.* = true;
        return;
    }
    const stats = statsF32(got, want);
    const is_ok = pass(stats, limits);
    if (!is_ok) failed.* = true;
    reportF32(name, stats, is_ok);
}

fn expectU32(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: []const u8,
    name: []const u8,
    got: []const u32,
    failed: *bool,
) !void {
    const want = try readU32(io, allocator, dir, name);
    defer allocator.free(want);
    if (std.mem.eql(u32, got, want)) return ok(name);
    failed.* = true;
    reportExact(name, firstU32(got, want));
}

fn expectU8(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: []const u8,
    name: []const u8,
    got: []const u8,
    failed: *bool,
) !void {
    const want = try readU8(io, allocator, dir, name);
    defer allocator.free(want);
    if (std.mem.eql(u8, got, want)) return ok(name);
    failed.* = true;
    reportExact(name, firstU8(got, want));
}

fn stageStat(name: []const u8, got: []const f32, want: []const f32, failed: *bool) void {
    if (got.len != want.len) {
        reportLen(name, got.len, want.len);
        failed.* = true;
        return;
    }
    const stats = statsF32(got, want);
    const is_ok = pass(stats, .latent);
    if (!is_ok) failed.* = true;
    reportF32(name, stats, is_ok);
}

const Stats = struct {
    max_abs: f64,
    mean_abs: f64,
    cosine: f64,
    index: usize,
};

fn statsF32(got: []const f32, want: []const f32) Stats {
    var max_abs: f64 = 0.0;
    var sum_abs: f64 = 0.0;
    var dot: f64 = 0.0;
    var got_norm: f64 = 0.0;
    var want_norm: f64 = 0.0;
    var index: usize = 0;
    for (got, want, 0..) |g, w, idx| {
        const gf: f64 = g;
        const wf: f64 = w;
        const diff = @abs(gf - wf);
        if (diff > max_abs) {
            max_abs = diff;
            index = idx;
        }
        sum_abs += diff;
        dot += gf * wf;
        got_norm += gf * gf;
        want_norm += wf * wf;
    }
    const den: f64 = @floatFromInt(got.len);
    return .{
        .max_abs = max_abs,
        .mean_abs = sum_abs / den,
        .cosine = cosine(dot, got_norm, want_norm),
        .index = index,
    };
}

fn pass(stats: Stats, limits: Limits) bool {
    return switch (limits) {
        .schedule => stats.max_abs <= 0.0001,
        .text => stats.mean_abs <= 0.02 and stats.cosine >= 0.999,
        .latent => stats.mean_abs <= 0.05 and stats.cosine >= 0.999,
        .rgb => stats.mean_abs <= 0.05 and stats.cosine >= 0.999,
    };
}

fn cosine(dot: f64, a: f64, b: f64) f64 {
    if (a == 0.0 and b == 0.0) return 1.0;
    if (a == 0.0 or b == 0.0) return 0.0;
    return dot / @sqrt(a * b);
}

fn readOptF32(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: []const u8,
    name: []const u8,
) !?[]f32 {
    return readF32(io, allocator, dir, name) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
}

fn readF32(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const bytes = try readRef(io, allocator, dir, name);
    defer allocator.free(bytes);
    if (bytes.len % 4 != 0) return error.InvalidReference;
    const out = try allocator.alloc(f32, bytes.len / 4);
    for (out, 0..) |*value, idx| {
        const raw = std.mem.readInt(u32, bytes[idx * 4 ..][0..4], .little);
        value.* = @bitCast(raw);
    }
    return out;
}

fn readU32(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u32 {
    const bytes = try readRef(io, allocator, dir, name);
    defer allocator.free(bytes);
    if (bytes.len % 4 != 0) return error.InvalidReference;
    const out = try allocator.alloc(u32, bytes.len / 4);
    for (out, 0..) |*value, idx| {
        value.* = std.mem.readInt(u32, bytes[idx * 4 ..][0..4], .little);
    }
    return out;
}

fn readU8(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const bytes = try readRef(io, allocator, dir, name);
    errdefer allocator.free(bytes);
    return bytes;
}

fn readRef(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, name });
    defer allocator.free(path);
    return readFile(io, allocator, path, 1024 * 1024 * 1024);
}

fn loadCase(io: std.Io, allocator: std.mem.Allocator, dir: []const u8) !Case {
    const path = try std.fmt.allocPrint(allocator, "{s}/case.json", .{dir});
    defer allocator.free(path);
    const bytes = try readFile(io, allocator, path, 1024 * 1024);
    defer allocator.free(bytes);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
    });
    defer parsed.deinit();
    const object = parsed.value.object;
    return .{
        .prompt = try allocator.dupe(u8, try string(object, "prompt")),
        .width = try u32Field(object, "width"),
        .height = try u32Field(object, "height"),
        .steps = try u32Field(object, "steps"),
        .seed = try u64Field(object, "seed"),
    };
}

fn readFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    max: u64,
) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > max) return error.FileTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

fn string(object: std.json.ObjectMap, key: []const u8) ![]const u8 {
    const value = object.get(key) orelse return error.InvalidReference;
    if (value != .string) return error.InvalidReference;
    return value.string;
}

fn u32Field(object: std.json.ObjectMap, key: []const u8) !u32 {
    const value = try u64Field(object, key);
    if (value > std.math.maxInt(u32)) return error.InvalidReference;
    return @intCast(value);
}

fn u64Field(object: std.json.ObjectMap, key: []const u8) !u64 {
    const value = object.get(key) orelse return error.InvalidReference;
    if (value != .integer or value.integer < 0) return error.InvalidReference;
    return @intCast(value.integer);
}

fn countMask(mask: []const u8) usize {
    var used: usize = 0;
    for (mask) |value| {
        if (value != 0) used += 1;
    }
    return used;
}

fn parse(iter: *std.process.Args.Iterator) !Options {
    _ = iter.next();
    var out = Options{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) {
            out.weights = try need(iter);
        } else if (std.mem.eql(u8, arg, "--ref")) {
            out.ref_dir = try need(iter);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            out.backend = try parseBackend(try need(iter));
        } else if (std.mem.eql(u8, arg, "--help")) {
            return out;
        } else {
            return error.UnknownOption;
        }
    }
    if (out.weights.len == 0 or out.ref_dir.len == 0) return error.MissingOption;
    return out;
}

fn help(opts: Options) bool {
    return opts.weights.len == 0 and opts.ref_dir.len == 0;
}

fn need(iter: *std.process.Args.Iterator) ![]const u8 {
    return iter.next() orelse error.MissingValue;
}

fn parseBackend(text: []const u8) !Backend {
    if (std.mem.eql(u8, text, "auto")) return .auto;
    if (std.mem.eql(u8, text, "metal")) return .metal;
    if (std.mem.eql(u8, text, "cpu")) return .cpu;
    return error.InvalidBackend;
}

fn firstU32(got: []const u32, want: []const u32) usize {
    const count = @min(got.len, want.len);
    for (0..count) |idx| if (got[idx] != want[idx]) return idx;
    return count;
}

fn firstU8(got: []const u8, want: []const u8) usize {
    const count = @min(got.len, want.len);
    for (0..count) |idx| if (got[idx] != want[idx]) return idx;
    return count;
}

fn ok(name: []const u8) void {
    std.debug.print("[ok] {s}\n", .{name});
}

fn reportExact(name: []const u8, index: usize) void {
    std.debug.print("[fail] {s}: first mismatch at {d}\n", .{ name, index });
}

fn reportLen(name: []const u8, got: usize, want: usize) void {
    std.debug.print("[fail] {s}: len got {d}, want {d}\n", .{ name, got, want });
}

fn reportF32(name: []const u8, stats: Stats, is_ok: bool) void {
    const tag = if (is_ok) "ok" else "fail";
    std.debug.print(
        "[{s}] {s}: max {d:.6}, mean {d:.6}, cos {d:.6}, idx {d}\n",
        .{ tag, name, stats.max_abs, stats.mean_abs, stats.cosine, stats.index },
    );
}

fn usage() void {
    std.debug.print(
        \\zdraw refcheck
        \\
        \\Usage:
        \\  zig build refcheck -- --weights path/to/Z-Image-Turbo \
        \\    --ref .zig-cache/ref/red_boat_64 --backend auto
        \\
        \\Backends: auto, metal, cpu
        \\
    , .{});
}

fn reportError(err: anyerror) void {
    switch (err) {
        error.FileNotFound => {
            std.debug.print("refcheck: missing reference file under --ref\n", .{});
        },
        error.ReferenceMismatch => {
            std.debug.print("refcheck: reference mismatch\n", .{});
        },
        else => std.debug.print("refcheck: error: {s}\n", .{@errorName(err)}),
    }
}
