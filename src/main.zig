// SPDX-License-Identifier: Apache-2.0
//! CLI dispatch. Argument parsing lives in args.zig; commands own their IO.
//! The preview command uses procedural images; generate runs a model.

const std = @import("std");

const args = @import("args.zig");
const cli_help = @import("cli_help.zig");
const image = @import("image.zig");
const progress = @import("progress.zig");
const composite = @import("composite.zig");
const recipe = @import("recipe.zig");
const metal_c = @import("metal_c.zig");
const image_quality = @import("image_quality.zig");
const metrics = @import("metrics.zig");
const model = @import("model.zig");
const plan_run = @import("plan_run.zig");
const runtime = @import("model_runtime.zig");
const render = @import("render.zig");
const safetensors = @import("safetensors.zig");
const bench_card = @import("bench_card.zig");
const doctor = @import("doctor.zig");
const fetch = @import("fetch.zig");
const safety = @import("safety.zig");
const session = @import("session.zig");
const version = @import("version.zig");
const terminal_preview = @import("terminal_preview.zig");
const terminal = @import("terminal.zig");
const util = @import("session_util.zig");
const runtime_options = @import("runtime_options.zig");
const model_kind = @import("model_kind.zig");
const model_paths = @import("model_paths.zig");
const weights = @import("weights.zig");

pub fn main(init: std.process.Init) !void {
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iter.deinit();

    var command = args.parse(&iter) catch |err| {
        try printParseError(init.io, init.gpa, err);
        std.process.exit(1);
    };

    const path = resolveWeights(init, &command) catch |err| {
        if (err != error.MissingWeights) {
            const text = try std.fmt.allocPrint(init.gpa, "zdraw: {s}\n", .{@errorName(err)});
            defer init.gpa.free(text);
            try std.Io.File.stderr().writeStreamingAll(init.io, text);
        }
        std.process.exit(1);
    };
    defer if (path) |value| init.gpa.free(value);
    dispatch(init, command) catch |err| {
        if (err == error.SafetyBlocked) std.process.exit(3);
        const text = try std.fmt.allocPrint(init.gpa, "zdraw: {s}\n", .{@errorName(err)});
        defer init.gpa.free(text);
        try std.Io.File.stderr().writeStreamingAll(init.io, text);
        std.process.exit(1);
    };
}

fn resolveWeights(init: std.process.Init, command: *args.Command) !?[]u8 {
    const selected: struct { kind: model.ModelKind, path: *[]const u8 } = switch (command.*) {
        .generate => |*r| .{ .kind = r.kind, .path = &r.weights_dir },
        .bench => |*r| .{ .kind = r.kind, .path = &r.weights_dir },
        .session => |*r| if (r.preview) return null else .{ .kind = r.kind, .path = &r.weights_dir },
        else => return null,
    };
    const path = model_paths.resolve(
        init.gpa,
        init.environ_map,
        selected.kind,
        selected.path.*,
    ) catch |err| {
        if (err == error.MissingWeights) {
            try printParseError(init.io, init.gpa, error.MissingWeights);
        }
        return err;
    };
    errdefer init.gpa.free(path);
    weights.validate(init.io, init.gpa, selected.kind, path) catch |err| {
        const text = try std.fmt.allocPrint(
            init.gpa,
            "zdraw: incomplete or unreadable model at {s} ({s})\n" ++
                "Run: zdraw fetch {s} --dir \"{s}\"\n",
            .{ path, @errorName(err), model_kind.cliName(selected.kind), path },
        );
        defer init.gpa.free(text);
        try std.Io.File.stderr().writeStreamingAll(init.io, text);
        return error.MissingWeights;
    };
    selected.path.* = path;
    return path;
}

fn dispatch(init: std.process.Init, command: args.Command) !void {
    switch (command) {
        .preview => |request| try runPreview(init.io, init.gpa, init.environ_map, request),
        .inspect => |request| try runInspect(init.io, init.gpa, request),
        .generate => |request| {
            var resolved = request;
            resolved.steps, resolved.guidance = model_kind.samplingRef(
                request.kind,
                request.steps,
                request.guidance,
                request.ref_image.len > 0,
            ) catch {
                try guidanceError(init.io, init.gpa, request.kind);
                std.process.exit(1);
            };
            if (request.safety) try safety.gate(init.io, init.gpa, request.prompt);
            try runGenerate(init.io, init.gpa, init.environ_map, resolved);
        },
        .session => |request| {
            var resolved = request;
            resolved.steps, resolved.guidance = model_kind.sampling(
                request.kind,
                request.steps,
                request.guidance,
            ) catch {
                try guidanceError(init.io, init.gpa, request.kind);
                std.process.exit(1);
            };
            try session.run(init.io, init.gpa, init.environ_map, resolved);
        },
        .doctor => |request| try doctor.run(init.io, init.gpa, init.environ_map, request),
        .bench => |request| try runBench(init.io, init.gpa, init.environ_map, request),
        .fetch => |request| try fetch.run(init.io, init.gpa, init.environ_map, request),
        .version => try runVersion(init.io, init.gpa),
        .help => |topic| try util.writeIo(init.io, cli_help.text(topic)),
    }
}

/// `zdraw version`: four `key value` lines, one fact each.
fn runVersion(io: std.Io, allocator: std.mem.Allocator) !void {
    const metal4 = switch (metal_c.zdraw_metal4_available()) {
        1 => "yes",
        0 => "no",
        else => "no metal device",
    };
    const text = try std.fmt.allocPrint(
        allocator,
        "zdraw {s}\ncommit {s}\nengine abi {d}\nmetal4 {s}\n",
        .{ version.semver, version.commit, version.abi, metal4 },
    );
    defer allocator.free(text);
    try util.writeIo(io, text);
}

fn guidanceError(io: std.Io, allocator: std.mem.Allocator, kind: model.ModelKind) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "{s} is step-distilled and only runs at --guidance 1.0; " ++
            "flux2-klein-base-4b accepts guidance (default 4)\n",
        .{model_kind.cliName(kind)},
    );
    defer allocator.free(text);
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

fn runPreview(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Preview,
) !void {
    try image.ensureParent(io, request.output_path);
    const pixels = try render.renderPreview(
        allocator,
        request.prompt,
        request.width,
        request.height,
        request.seed,
    );
    defer allocator.free(pixels);

    try image.writePng(io, allocator, request.output_path, pixels, request.width, request.height);
    if (request.show) {
        try terminal.writeImage(io, allocator, pixels, request.width, request.height, .{
            .env = env,
        });
    }
    const text = try std.fmt.allocPrint(allocator, "wrote {s}\n", .{request.output_path});
    defer allocator.free(text);
    try util.writeIo(io, text);
}

fn runInspect(
    io: std.Io,
    allocator: std.mem.Allocator,
    request: args.Inspect,
) !void {
    var header = try safetensors.readHeader(io, allocator, request.path);
    defer header.deinit(allocator);

    const head = try std.fmt.allocPrint(allocator, "{d} tensors\n", .{header.tensors.len});
    defer allocator.free(head);
    try util.writeIo(io, head);
    for (header.tensors) |tensor| {
        const prefix = try std.fmt.allocPrint(
            allocator,
            "{s} {s} [",
            .{ tensor.name, tensor.dtype },
        );
        defer allocator.free(prefix);
        try util.writeIo(io, prefix);
        for (tensor.shape, 0..) |dim, idx| {
            const chunk = try std.fmt.allocPrint(
                allocator,
                "{s}{d}",
                .{ if (idx > 0) ", " else "", dim },
            );
            defer allocator.free(chunk);
            try util.writeIo(io, chunk);
        }
        try util.writeIo(io, "]\n");
    }
}

/// Parse "--seeds 7,11,23" into an owned list; empty raw = the single --seed.
fn parseSeeds(allocator: std.mem.Allocator, raw: []const u8, fallback: u64) ![]u64 {
    if (raw.len == 0) {
        const one = try allocator.alloc(u64, 1);
        one[0] = fallback;
        return one;
    }
    var list: std.ArrayList(u64) = .empty;
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (trimmed.len == 0) continue;
        try list.append(allocator, try std.fmt.parseInt(u64, trimmed, 10));
    }
    if (list.items.len == 0) return error.InvalidNumber;
    return list.toOwnedSlice(allocator);
}

/// Save every candidate; a blank one is flagged, not fatal mid-batch (the
/// candidate-set contract). Returns how many were rejected.
/// SHA-256 of the raw RGB bytes of the last image saveResult wrote: the
/// certified hash (invariant to the PNG's recipe chunk).
var last_pixels_sha256: [64]u8 = [_]u8{'0'} ** 64;
/// Bits per weight of the sidecar the last Klein render loaded (the card's pack).
var last_pack_bits: u8 = 16;
var last_text_bits: u8 = 16;

/// The init image's file hash for the recipe (computed once per process).
var init_sha256: ?[64]u8 = null;

fn recipeFor(request: args.Generate, seed: u64) recipe.Recipe {
    const prof: []const u8 = if (request.kind == .z_image_turbo) @tagName(request.profile) else "-";
    const has_init = request.init_image.len > 0;
    return .{
        .model = model.cliName(request.kind),
        .profile = prof,
        .prompt = request.prompt,
        .seed = seed,
        .width = request.width,
        .height = request.height,
        .steps = request.steps,
        .guidance = request.guidance,
        .init_sha256 = if (has_init) (if (init_sha256) |*h| h else null) else null,
        .strength = if (has_init) request.strength else null,
    };
}

/// A masked edit comes back as the user's own photo at its own size, with
/// only the painted region changed; everything else is the render itself.
fn composited(
    allocator: std.mem.Allocator,
    request: args.Generate,
    result: model.KleinResult,
) !model.Result {
    const raw = model.Result{
        .pixels = result.pixels,
        .width = result.width,
        .height = result.height,
    };
    if (request.mask.len == 0 or request.init_image.len == 0) {
        // An instruction edit: give the photo back at its own resolution,
        // keeping its pixels wherever the edit changed nothing.
        if (request.ref_image.len == 0) return raw;
        const k = try composite.changedInto(
            allocator,
            request.ref_image,
            raw.pixels,
            raw.width,
            raw.height,
        );
        if (k) |c| return .{ .pixels = c.pixels, .width = c.width, .height = c.height };
        return raw;
    }
    const c = try composite.maskedInto(
        allocator,
        request.init_image,
        request.mask,
        result.pixels,
        result.width,
        result.height,
    );
    return .{ .pixels = c.pixels, .width = c.width, .height = c.height };
}

fn saveCandidates(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Generate,
    final: []const model.KleinResult,
    seeds: []const u64,
) !usize {
    var rejected: usize = 0;
    for (final, seeds) |r, seed| {
        if (request.safety) {
            try safety.imageGate(io, allocator, env, r.pixels, r.width, r.height);
        }
        const path = try seedPath(allocator, request.output_path, seed);
        defer allocator.free(path);
        saveResult(io, allocator, env, path, .{
            .pixels = r.pixels,
            .width = r.width,
            .height = r.height,
        }, request.show, recipeFor(request, seed)) catch |err| switch (err) {
            error.BlankImage => rejected += 1,
            else => return err,
        };
    }
    return rejected;
}

/// Per-seed output path: "out.png" -> "out-s7.png"; caller frees.
fn seedPath(allocator: std.mem.Allocator, base: []const u8, seed: u64) ![]u8 {
    const ext = std.fs.path.extension(base);
    const stem = base[0 .. base.len - ext.len];
    return std.fmt.allocPrint(allocator, "{s}-s{d}{s}", .{ stem, seed, ext });
}

/// Batched multi-seed Klein generation: one wider denoise per --repeat run,
/// one PNG per seed (suffix -s<seed>).
fn runKleinBatch(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Generate,
    krt: *model.KleinRuntime,
    kreq: model.KleinRequest,
    seeds: []const u64,
) !void {
    var results: ?[]model.KleinResult = null;
    defer if (results) |prev| {
        for (prev) |r| allocator.free(r.pixels);
        allocator.free(prev);
    };
    var krun: u32 = 0;
    while (krun < request.repeat) : (krun += 1) {
        const t0 = metrics.now();
        const out = try krt.generateMulti(io, allocator, kreq, seeds);
        if (results) |prev| {
            for (prev) |r| allocator.free(r.pixels);
            allocator.free(prev);
        }
        results = out;
        if (request.repeat > 1) {
            const secs = @as(f64, @floatFromInt(metrics.now() - t0)) / 1e9;
            const text = try std.fmt.allocPrint(allocator, "run {d}: {d:.1}s\n", .{ krun + 1, secs });
            defer allocator.free(text);
            try util.writeIo(io, text);
        }
    }
    const final = results.?;
    const rejected = try saveCandidates(io, allocator, env, request, final, seeds);
    try metrics.report(io, allocator);
    if (rejected > 0) {
        const text = try std.fmt.allocPrint(
            allocator,
            "{d} of {d} candidates rejected as blank\n",
            .{ rejected, seeds.len },
        );
        defer allocator.free(text);
        try std.Io.File.stderr().writeStreamingAll(io, text);
        std.process.exit(2);
    }
}

fn runGenerate(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Generate,
) !void {
    // Apply the profile before Metal initialization. An explicit --profile
    // wins over environment overrides; the default defers to them.
    // Z-Image scope: parse rejects an explicit --profile for other kinds.
    if (request.kind == .z_image_turbo) runtime_options.applyEnv(request.profile, request.profile_explicit);
    if (request.vae_reference) runtime_options.applyVaeRef();
    // Metal autoreleases per dispatch; a CLI has no pool, so drain at the end.
    const pool = metal_c.zdraw_metal_pool_push();
    defer metal_c.zdraw_metal_pool_pop(pool);
    runGenerateImpl(io, allocator, env, request) catch |err| {
        if (err == error.BlankImage) std.process.exit(2);
        return err;
    };
}

fn runGenerateImpl(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Generate,
) !void {
    try image.ensureParent(io, request.output_path);
    if (request.init_image.len > 0) init_sha256 = try plan_run.hashFile(io, request.init_image);
    if (request.init_image.len > 0 and request.kind == .z_image_turbo) {
        const msg = "zdraw: --init-image is Klein only (the Z-Image VAE encoder is not ported)\n";
        try util.writeIo(io, msg);
        std.process.exit(2);
    }
    if (request.kind != .z_image_turbo) {
        // Non-Z-Image kinds (Klein): hold a reusable runtime across --repeat
        // so runs 2+ are warm (weights/contexts loaded once).
        var krt = model.openKlein(
            io,
            allocator,
            request.kind,
            request.weights_dir,
        ) catch |err| switch (err) {
            error.MissingFile => {
                try missingWeights(io, allocator, request.weights_dir);
                std.process.exit(1);
            },
            error.MissingPackedSidecar => {
                try kleinPackError(io, allocator, request.kind, request.weights_dir);
                std.process.exit(1);
            },
            else => return err,
        };
        defer krt.deinit(io, allocator);
        const kreq = model.KleinRequest{
            .prompt = request.prompt,
            .width = request.width,
            .height = request.height,
            .steps = request.steps,
            .seed = request.seed,
            .guidance = request.guidance,
            .init_image = request.init_image,
            .strength = if (request.init_image.len > 0) request.strength else 1.0,
            .mask = request.mask,
            .ref_image = request.ref_image,
        };
        const seeds = try parseSeeds(allocator, request.seeds_raw, request.seed);
        defer allocator.free(seeds);
        last_pack_bits = krt.pack_bits;
        last_text_bits = krt.text_bits;
        if (request.seeds_raw.len > 0) {
            // Even one --seeds entry takes the batch path: it must render
            // THAT seed and write the -s<seed> suffix, not the --seed default.
            try runKleinBatch(io, allocator, env, request, &krt, kreq, seeds);
            return;
        }
        var krun: u32 = 0;
        var kresult: ?model.KleinResult = null;
        defer if (kresult) |prev| allocator.free(prev.pixels);
        while (krun < request.repeat) : (krun += 1) {
            const t0 = metrics.now();
            const show = request.show and request.progressive;
            const out = try kleinShown(io, allocator, env, &krt, kreq, show);
            if (kresult) |prev| allocator.free(prev.pixels);
            kresult = out;
            try noteRun(io, allocator, request.repeat, krun, t0);
        }
        const final = kresult.?;
        if (request.safety) {
            try safety.imageGate(io, allocator, env, final.pixels, final.width, final.height);
        }
        const shown = try composited(allocator, request, final);
        defer if (shown.pixels.ptr != final.pixels.ptr) allocator.free(shown.pixels);
        const rec = recipeFor(request, request.seed);
        // Leave the requested terminal image visible after diagnostics.
        if (request.show) try metrics.report(io, allocator);
        try saveResult(io, allocator, env, request.output_path, shown, request.show, rec);
        if (!request.show) try metrics.report(io, allocator);
        return;
    }

    var rt = runtime.Runtime.init(
        io,
        allocator,
        request.kind,
        request.weights_dir,
    ) catch |err| switch (err) {
        error.MissingFile => {
            try missingWeights(io, allocator, request.weights_dir);
            std.process.exit(1);
        },
        else => return err,
    };
    defer rt.deinit(io, allocator);

    // --repeat N reruns the same request with the weights resident, printing
    // per-run seconds: the warm in-session number competitor engines quote
    // (their first run hides the load; ours normally pays it every process).
    var result = try timedGenerate(io, allocator, &rt, request, 0);
    var run: u32 = 1;
    while (run < request.repeat) : (run += 1) {
        allocator.free(result.pixels);
        result = try timedGenerate(io, allocator, &rt, request, run);
    }
    defer allocator.free(result.pixels);
    if (request.safety) {
        try safety.imageGate(io, allocator, env, result.pixels, result.width, result.height);
    }

    try saveResult(io, allocator, env, request.output_path, .{
        .pixels = result.pixels,
        .width = result.width,
        .height = result.height,
    }, request.show, recipeFor(request, request.seed));
}

fn kleinShown(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    rt: *model.KleinRuntime,
    request: model.KleinRequest,
    show: bool,
) !model.KleinResult {
    var view = try terminal_preview.View.init(io, env, show);
    try view.start();
    defer view.deinit(io);
    return rt.generate(io, allocator, request);
}

fn timedGenerate(
    io: std.Io,
    allocator: std.mem.Allocator,
    rt: *runtime.Runtime,
    request: args.Generate,
    run: u32,
) !runtime.Result {
    const t0 = metrics.now();
    const result = try rt.generate(io, allocator, .{
        .prompt = request.prompt,
        .width = request.width,
        .height = request.height,
        .steps = request.steps,
        .seed = request.seed,
    });
    try noteRun(io, allocator, request.repeat, run, t0);
    return result;
}

/// Record one run's seconds under "generate" (the bench card reads them) and
/// print the warm-loop line competitor engines quote when --repeat > 1.
fn noteRun(io: std.Io, allocator: std.mem.Allocator, repeat: u32, run: u32, t0: u64) !void {
    const ns = metrics.now() - t0;
    metrics.record("generate", ns);
    if (repeat <= 1) return;
    const secs = @as(f64, @floatFromInt(ns)) / 1e9;
    const text = try std.fmt.allocPrint(allocator, "run {d}: {d:.1}s\n", .{ run + 1, secs });
    defer allocator.free(text);
    try util.writeIo(io, text);
}

/// `zdraw bench`: the census case through the same generate path, then the
/// card (hash vs certified, phases, routes, fallbacks, memory, thermal).
fn runBench(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Bench,
) !void {
    if (request.kind == .z_image_turbo) {
        runtime_options.applyEnv(request.profile, request.profile_explicit);
    }
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const out_path = if (request.output_path.len > 0)
        request.output_path
    else
        try std.fmt.allocPrint(arena, "zdraw-bench-{s}.png", .{model.cliName(request.kind)});
    const steps = bench_card.stepsFor(request.kind);
    const guidance = (model_kind.sampling(request.kind, steps, 0) catch {
        try guidanceError(io, allocator, request.kind);
        std.process.exit(1);
    })[1];
    const gen = args.Generate{
        .kind = request.kind,
        .weights_dir = request.weights_dir,
        .prompt = bench_card.prompt,
        .output_path = out_path,
        .width = bench_card.size,
        .height = bench_card.size,
        .steps = steps,
        .seed = bench_card.seed,
        .guidance = guidance,
        .repeat = request.repeat,
        .profile = request.profile,
        .profile_explicit = request.profile_explicit,
        .safety = request.safety,
    };
    if (request.safety) try safety.gate(io, allocator, bench_card.prompt);
    const thermal_before = plan_run.thermalState();
    const started = metrics.now();
    try runGenerateImpl(io, allocator, env, gen);
    const wall_ns = metrics.now() - started;
    // The certified hash is over the pixels (invariant to the PNG's recipe
    // chunk); the file hash rides along for the record.
    const digest = try plan_run.hashFile(io, out_path);
    const prof: []const u8 = if (request.kind == .z_image_turbo) @tagName(request.profile) else "-";
    const card = try bench_card.build(arena, env, .{
        .kind = request.kind,
        .profile = prof,
        .pack_bits = if (request.kind == .z_image_turbo) 16 else last_pack_bits,
        .text_bits = if (request.kind == .z_image_turbo) 16 else last_text_bits,
        .safety = request.safety,
        .wall_ns = wall_ns,
        .output_sha256 = &last_pixels_sha256,
        .png_sha256 = &digest,
        .thermal_before = thermal_before,
        .thermal_after = plan_run.thermalState(),
    });
    try bench_card.emit(io, arena, card);
}

fn saveResult(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    path: []const u8,
    result: model.Result,
    show: bool,
    rec: recipe.Recipe,
) !void {
    const stats = try image_quality.analyzeRgb(result.pixels, result.width, result.height);
    if (image_quality.isBlankLike(stats)) {
        const text = try std.fmt.allocPrint(
            allocator,
            "rejected blank/low-content image; no file written (luma {d}-{d}, mean {d:.1})\n",
            .{ stats.min_luma, stats.max_luma, stats.mean_luma },
        );
        defer allocator.free(text);
        try std.Io.File.stderr().writeStreamingAll(io, text);
        // Callers decide: runGenerate exits 2 after any plan receipt is
        // banked; the batch path keeps saving the remaining candidates
        // and fails at the end.
        return error.BlankImage;
    }
    const t = try recipe.texts(allocator, rec);
    defer allocator.free(t.buf);
    const px = result.pixels;
    try image.writePngText(io, allocator, path, px, result.width, result.height, &t.chunks);
    last_pixels_sha256 = image.pixelSha256(result.pixels);
    // Verbose: the certified (pixel) hash beside the path, for the app's
    // recipe check and for scripts.
    const hex: []const u8 = &last_pixels_sha256;
    const line = try std.fmt.allocPrint(allocator, "pixels {s} {s}", .{ hex, path });
    defer allocator.free(line);
    try progress.event(io, allocator, line);
    if (show) {
        try terminal.writeImage(io, allocator, result.pixels, result.width, result.height, .{
            .env = env,
        });
    }
    const text = try std.fmt.allocPrint(allocator, "wrote {s}\n", .{path});
    defer allocator.free(text);
    try util.writeIo(io, text);
}

fn missingWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "missing required model files under --weights {s}\n",
        .{path},
    );
    defer allocator.free(text);
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

fn kleinPackError(
    io: std.Io,
    allocator: std.mem.Allocator,
    kind: model.ModelKind,
    weights_dir: []const u8,
) !void {
    const text = try std.fmt.allocPrint(
        allocator,
        "missing Klein weight pack; run zdraw fetch {s} --dir \"{s}\"\n",
        .{ model_kind.cliName(kind), weights_dir },
    );
    defer allocator.free(text);
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

fn printParseError(io: std.Io, allocator: std.mem.Allocator, err: args.ParseError) !void {
    const message = switch (err) {
        error.MissingPrompt => "provide --prompt TEXT",
        error.MissingOutput => "provide --out FILE.png",
        error.MissingWeights => "provide --weights DIR for this model",
        error.MissingOutputDir => "provide --out-dir DIR or --no-auto-save",
        error.MissingOptionValue => "an option is missing its value",
        error.InvalidDimension => "use positive dimensions: multiples of 16 for Z-Image, 32 for Klein",
        error.UnsupportedKleinResolution => args.klein_resolution_hint,
        error.InvalidNumber => "provide a finite number in the option's allowed range",
        error.InvalidStrength => "--strength must be between 0 and 1",
        error.InvalidGuidance => "--guidance must be positive; the distilled models accept " ++
            "only 1, flux2-klein-base-4b any value (default 4)",
        error.EditUnsupported => "image editing requires a Klein model (flux2-klein-4b or base)",
        error.ConflictingEdits => "choose --edit or --init-image, not both",
        error.MissingInitImage => "--mask and --strength require --init-image",
        error.EditSeedsUnsupported => "editing uses --seed; --seeds is for text-to-image",
        error.DuplicateSeed => "--seeds values must be distinct to avoid overwriting output files",
        error.ConflictingSeeds => "choose --seed or --seeds, not both",
        error.ProfileUnsupported => "--profile is supported only by z-image-turbo",
        error.SeedsUnsupported => "--seeds requires a Klein model (flux2-klein-4b or -base-4b)",
        error.InvalidModel => "unknown model; use flux2-klein-4b, flux2-klein-base-4b " ++
            "or z-image-turbo",
        error.MissingModel => "provide a model name after fetch",
        error.MissingPath => "provide the safetensors file to inspect",
        error.UnknownOption => "unknown option or unexpected argument",
        error.InvalidCommand => "unknown command",
        else => @errorName(err),
    };
    const text = try std.fmt.allocPrint(
        allocator,
        "zdraw: {s}\nUse zdraw COMMAND --help for usage.\n",
        .{message},
    );
    defer allocator.free(text);
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

test "default model name is printable" {
    const name = model.kindName(model.defaultKind());
    try std.testing.expect(std.mem.indexOf(u8, name, "Z-Image") != null);
}

test "sampling resolves per-model defaults and gates guidance" {
    const distilled = try model_kind.sampling(.flux2_klein_9b, 0, 0);
    try std.testing.expectEqual(@as(u32, 4), distilled[0]);
    try std.testing.expectEqual(@as(f32, 1.0), distilled[1]);
    const base = try model_kind.sampling(.flux2_klein_base_9b, 0, 0);
    try std.testing.expectEqual(@as(u32, 50), base[0]);
    try std.testing.expectEqual(@as(f32, 4.0), base[1]);
    // Explicit values win; guidance > 1 is rejected on distilled models.
    const explicit = try model_kind.sampling(.flux2_klein_base_4b, 8, 2.5);
    try std.testing.expectEqual(@as(u32, 8), explicit[0]);
    try std.testing.expectEqual(@as(f32, 2.5), explicit[1]);
    const unsupported = error.GuidanceUnsupported;
    try std.testing.expectError(unsupported, model_kind.sampling(.flux2_klein_4b, 0, 3.0));
    try std.testing.expectError(unsupported, model_kind.sampling(.z_image_turbo, 0, 2.0));
}
