//! `zdraw fetch <model>`: exactly the files the engine reads, downloaded
//! from the official Hugging Face repository by `download.zig` (resumable,
//! hash-checked, no Python), then the weight sidecar built beside them, then
//! the certified hash to expect and the exact bench command. Only
//! Apache-licensed models are listed.
const std = @import("std");
const model_paths = @import("model_paths.zig");
const args = @import("args.zig");
const bench_card = @import("bench_card.zig");
const download = @import("download.zig");
const klein_packer = @import("klein_packer.zig");
const model = @import("model.zig");
const util = @import("session_util.zig");
const weights = @import("weights.zig");
const zflux2 = @import("zflux2.zig");
const zimage_packer = @import("zimage_packer.zig");
const zpack_families = @import("zpack_families.zig");
const zpack_kinds = @import("zpack_kinds.zig");

pub const Spec = struct {
    repo: []const u8,
    /// Download size of the files below, GB (approximate, for the plan line).
    snapshot_gb: u32,
    /// Sidecar the engine reads, built beside the weights.
    pack_gb: u32,
    pack_name: []const u8,
    /// The files the engine opens (weights.zig lists the required ones; the
    /// shards come from the index files at these fixed names).
    files: []const []const u8,
};

const tokenizer_files = [_][]const u8{
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/added_tokens.json",
};

const klein_files = tokenizer_files ++ [_][]const u8{
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
};

const zimage_files = tokenizer_files ++ [_][]const u8{
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
};

const classifier_files = [_][]const u8{
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
};

/// The distributable models. Klein 9B is absent: non-commercial licence.
pub fn spec(kind: model.ModelKind) ?Spec {
    return switch (kind) {
        .flux2_klein_4b => .{
            .repo = "black-forest-labs/FLUX.2-klein-4B",
            .snapshot_gb = 16,
            .pack_gb = 8,
            .pack_name = zflux2.Config.klein_4b.pack_name,
            .files = &klein_files,
        },
        .flux2_klein_base_4b => .{
            .repo = "black-forest-labs/FLUX.2-klein-base-4B",
            .snapshot_gb = 16,
            .pack_gb = 8,
            .pack_name = zflux2.Config.klein_base_4b.pack_name,
            .files = &klein_files,
        },
        .z_image_turbo => .{
            .repo = "Tongyi-MAI/Z-Image-Turbo",
            .snapshot_gb = 33,
            .pack_gb = 16,
            .pack_name = "zdraw-w16.zpack",
            .files = &zimage_files,
        },
        else => null,
    };
}

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    request: args.Fetch,
) !void {
    var buf: [4200]u8 = undefined;
    if (request.classifier) return fetchClassifier(io, allocator, environ, request);
    const sp = spec(request.kind) orelse {
        const fmt = "fetch: {s} is not distributed (non-commercial licence or unsupported)\n";
        try say(io, try std.fmt.bufPrint(&buf, fmt, .{model.cliName(request.kind)}));
        std.process.exit(1);
    };
    const dest = try model_paths.resolve(allocator, environ, request.kind, request.dir);
    defer allocator.free(dest);
    const plan = "fetch: {s} -> {s} (~{d} GB download, ~{d} GB sidecar)\n";
    try say(io, try std.fmt.bufPrint(&buf, plan, .{ sp.repo, dest, sp.snapshot_gb, sp.pack_gb }));
    if (weights.validate(io, allocator, request.kind, dest)) |_| {
        try say(io, "fetch: weights already complete, skipping the download\n");
    } else |_| {
        try fetchFiles(io, allocator, environ, sp.repo, sp.files, dest);
        weights.validate(io, allocator, request.kind, dest) catch {
            const fmt = "fetch: the snapshot is incomplete under {s}; rerun to resume\n";
            try say(io, try std.fmt.bufPrint(&buf, fmt, .{dest}));
            std.process.exit(1);
        };
    }
    const pack_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dest, sp.pack_name });
    defer allocator.free(pack_path);
    if (request.no_pack) {
        const next = "download complete; prepare the pack: zdraw fetch {s} --dir \"{s}\"\n";
        try say(io, try std.fmt.bufPrint(&buf, next, .{ model.cliName(request.kind), dest }));
        return;
    }
    try buildPack(io, allocator, request.kind, dest, pack_path);
    try printNext(io, allocator, request.kind, dest);
}

/// The safety filter's image classifier (Falconsai/nsfw_image_detection,
/// Apache-2.0, ~350 MB): its safetensors and configs, no sidecar.
fn fetchClassifier(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    request: args.Fetch,
) !void {
    var buf: [4200]u8 = undefined;
    var dest_buf: [4096]u8 = undefined;
    const dest = if (request.dir.len > 0) request.dir else blk: {
        const home = environ.get("HOME") orelse return error.NoHome;
        const fmt = "{s}/.zdraw/models/nsfw_image_detection";
        break :blk try std.fmt.bufPrint(&dest_buf, fmt, .{home});
    };
    const repo = "Falconsai/nsfw_image_detection";
    try say(io, try std.fmt.bufPrint(&buf, "fetch: {s} -> {s} (~350 MB)\n", .{ repo, dest }));
    const model_path = try std.fmt.bufPrint(&buf, "{s}/model.safetensors", .{dest});
    if (std.Io.Dir.cwd().access(io, model_path, .{})) |_| {
        try say(io, "fetch: classifier already present\n");
    } else |_| try fetchFiles(io, allocator, environ, repo, &classifier_files, dest);
    try say(io, "next: renders check their output against it automatically " ++
        "(ZDRAW_SAFETY_MODEL overrides the path)\n");
}

/// The listed files from the official repository, resumable; a failure names
/// the file and leaves its `.part` for the next run.
fn fetchFiles(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    repo: []const u8,
    files: []const []const u8,
    dest: []const u8,
) !void {
    var buf: [512]u8 = undefined;
    download.repoFiles(io, allocator, environ, repo, files, dest) catch |err| {
        const fmt = "fetch: download failed ({s}); rerun to resume (gated repo? " ++
            "accept the licence on huggingface.co first)\n";
        try say(io, try std.fmt.bufPrint(&buf, fmt, .{@errorName(err)}));
        std.process.exit(1);
    };
}

fn buildPack(
    io: std.Io,
    allocator: std.mem.Allocator,
    kind: model.ModelKind,
    dest: []const u8,
    pack_path: []const u8,
) !void {
    var buf: [4200]u8 = undefined;
    if (std.Io.Dir.cwd().access(io, pack_path, .{})) |_| {
        try say(io, try std.fmt.bufPrint(&buf, "fetch: sidecar present at {s}\n", .{pack_path}));
        return;
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }
    try say(io, try std.fmt.bufPrint(&buf, "fetch: building the sidecar {s}\n", .{pack_path}));
    if (kind == .z_image_turbo) {
        try zimage_packer.run(io, allocator, .{
            .weights = dest,
            .out = pack_path,
            .bits = 16,
            .last = 30,
            .kinds = try zpack_kinds.parse("all"),
            .families = try zpack_families.parse("all"),
        });
    } else {
        try klein_packer.run(io, allocator, .{ .weights = dest, .out = pack_path });
    }
}

fn printNext(
    io: std.Io,
    allocator: std.mem.Allocator,
    kind: model.ModelKind,
    dest: []const u8,
) !void {
    var buf: [4200]u8 = undefined;
    var parsed = try bench_card.parseCertified(allocator);
    defer parsed.deinit();
    const name = model.cliName(kind);
    const prof: []const u8 = if (kind == .z_image_turbo) "product" else "-";
    if (bench_card.lookup(parsed.value, name, prof, "w16")) |entry| {
        const fmt = "expected hash (1024, {d} steps, seed 46): {s}\n";
        try say(io, try std.fmt.bufPrint(&buf, fmt, .{ entry.steps, entry.sha256[0..12] }));
    }
    const fmt = "next: zdraw bench --model {s} --weights \"{s}\"\n";
    try say(io, try std.fmt.bufPrint(&buf, fmt, .{ name, dest }));
}

fn say(io: std.Io, text: []const u8) !void {
    try util.writeIo(io, text);
}

test "only the Apache-licensed models are fetchable" {
    try std.testing.expect(spec(.flux2_klein_4b) != null);
    try std.testing.expect(spec(.flux2_klein_base_4b) != null);
    try std.testing.expect(spec(.z_image_turbo) != null);
    try std.testing.expect(spec(.flux2_klein_9b) == null);
    try std.testing.expect(spec(.flux2_klein_base_9b) == null);
    // The base checkpoint shares the distilled model's file layout but has
    // its own sidecar name, so the two packs cannot be confused.
    const distilled_pack = spec(.flux2_klein_4b).?.pack_name;
    const base_pack = spec(.flux2_klein_base_4b).?.pack_name;
    try std.testing.expect(!std.mem.eql(u8, distilled_pack, base_pack));
}

test "the default destination lives under HOME/.zdraw/models" {
    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    try map.put("HOME", "/Users/x");
    const req = args.Fetch{ .kind = .flux2_klein_4b };
    const dest = try model_paths.resolve(std.testing.allocator, &map, req.kind, req.dir);
    defer std.testing.allocator.free(dest);
    try std.testing.expectEqualStrings("/Users/x/.zdraw/models/FLUX.2-klein-4B", dest);
}
