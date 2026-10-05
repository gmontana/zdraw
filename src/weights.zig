//! Required-file validation for supported model directories.

const std = @import("std");

const kinds = @import("model_kind.zig");
const weight_index = @import("weight_index.zig");

pub const ZImage = struct {
    text: weight_index.Index,
    transformer: weight_index.Index,

    pub fn deinit(self: *ZImage, allocator: std.mem.Allocator) void {
        self.text.deinit(allocator);
        self.transformer.deinit(allocator);
        self.* = undefined;
    }
};

const zimage_files = [_][]const u8{
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
};

const flux_files = [_][]const u8{
    "transformer/config.json",
    "text_encoder/config.json",
    // The text encoder's shard index is not listed: a text pack replaces the
    // shards, and the loader reports MissingFile when neither is present.
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
};

pub fn validate(
    io: std.Io,
    allocator: std.mem.Allocator,
    kind: kinds.ModelKind,
    root: []const u8,
) !void {
    for (required(kind)) |name| try checkFile(io, allocator, root, name);
}

pub fn loadZImage(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !ZImage {
    try validate(io, allocator, .z_image_turbo, root);

    const text_path = try join(
        allocator,
        root,
        "text_encoder/model.safetensors.index.json",
    );
    defer allocator.free(text_path);

    const transformer_path = try join(
        allocator,
        root,
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    );
    defer allocator.free(transformer_path);

    var text = try weight_index.read(io, allocator, text_path);
    errdefer text.deinit(allocator);
    const transformer = try weight_index.read(io, allocator, transformer_path);

    return .{ .text = text, .transformer = transformer };
}

fn required(kind: kinds.ModelKind) []const []const u8 {
    return switch (kind) {
        .z_image_turbo => &zimage_files,
        .flux2_klein_4b,
        .flux2_klein_9b,
        .flux2_klein_base_4b,
        .flux2_klein_base_9b,
        .flux2_klein_9b_kv,
        => &flux_files,
    };
}

fn checkFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    name: []const u8,
) !void {
    const path = try join(allocator, root, name);
    defer allocator.free(path);

    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.MissingFile,
        else => return err,
    };
    file.close(io);
}

fn join(allocator: std.mem.Allocator, root: []const u8, name: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, name });
}

fn touch(io: std.Io, allocator: std.mem.Allocator, root: []const u8, name: []const u8) !void {
    const path = try join(allocator, root, name);
    defer allocator.free(path);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    file.close(io);
}

fn makeDirs(io: std.Io, allocator: std.mem.Allocator, root: []const u8) !void {
    for ([_][]const u8{ "scheduler", "text_encoder", "tokenizer", "transformer", "vae" }) |dir| {
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, dir });
        defer allocator.free(path);
        try std.Io.Dir.cwd().createDirPath(io, path);
    }
}

fn writeFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    name: []const u8,
    text: []const u8,
) !void {
    const path = try join(allocator, root, name);
    defer allocator.free(path);

    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [256]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    try writer.interface.writeAll(text);
    try writer.interface.flush();
}

fn fixtureRoot(allocator: std.mem.Allocator, tmp: std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/model", .{tmp.sub_path});
}

test "validate z-image directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try fixtureRoot(std.testing.allocator, tmp);
    defer std.testing.allocator.free(root);
    try makeDirs(std.testing.io, std.testing.allocator, root);
    for (zimage_files) |name| {
        try touch(std.testing.io, std.testing.allocator, root, name);
    }

    try validate(std.testing.io, std.testing.allocator, .z_image_turbo, root);
}

test "reject incomplete model directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try fixtureRoot(std.testing.allocator, tmp);
    defer std.testing.allocator.free(root);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, root);

    try std.testing.expectError(
        error.MissingFile,
        validate(std.testing.io, std.testing.allocator, .flux2_klein_4b, root),
    );
}

test "load z-image shard indexes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try fixtureRoot(std.testing.allocator, tmp);
    defer std.testing.allocator.free(root);
    try makeDirs(std.testing.io, std.testing.allocator, root);
    for (zimage_files) |name| try touch(std.testing.io, std.testing.allocator, root, name);

    const index_json =
        \\{"metadata":{"total_size":1},"weight_map":{"x":"a.safetensors"}}
    ;
    try writeFile(
        std.testing.io,
        std.testing.allocator,
        root,
        "text_encoder/model.safetensors.index.json",
        index_json,
    );
    try writeFile(
        std.testing.io,
        std.testing.allocator,
        root,
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        index_json,
    );

    var loaded = try loadZImage(std.testing.io, std.testing.allocator, root);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("a.safetensors", loaded.text.find("x").?);
    try std.testing.expectEqualStrings("a.safetensors", loaded.transformer.find("x").?);
}
