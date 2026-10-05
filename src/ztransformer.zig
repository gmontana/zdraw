//! Z-Image transformer weight-index validation.

const std = @import("std");

const weight_index = @import("weight_index.zig");
const zconfig = @import("zimage_config.zig");

pub const Error = error{
    MissingTensor,
};

const globals = [_][]const u8{
    "all_x_embedder.2-1.weight",
    "all_x_embedder.2-1.bias",
    "all_final_layer.2-1.adaLN_modulation.1.weight",
    "all_final_layer.2-1.adaLN_modulation.1.bias",
    "all_final_layer.2-1.linear.weight",
    "all_final_layer.2-1.linear.bias",
    "cap_embedder.0.weight",
    "cap_embedder.1.weight",
    "cap_embedder.1.bias",
    "t_embedder.mlp.0.weight",
    "t_embedder.mlp.0.bias",
    "t_embedder.mlp.2.weight",
    "t_embedder.mlp.2.bias",
    "x_pad_token",
    "cap_pad_token",
};

const block_suffixes = [_][]const u8{
    "attention.norm_k.weight",
    "attention.norm_q.weight",
    "attention.to_k.weight",
    "attention.to_out.0.weight",
    "attention.to_q.weight",
    "attention.to_v.weight",
    "attention_norm1.weight",
    "attention_norm2.weight",
    "feed_forward.w1.weight",
    "feed_forward.w2.weight",
    "feed_forward.w3.weight",
    "ffn_norm1.weight",
    "ffn_norm2.weight",
};

const adaln_suffixes = [_][]const u8{
    "adaLN_modulation.0.weight",
    "adaLN_modulation.0.bias",
};

pub fn validate(
    allocator: std.mem.Allocator,
    config: zconfig.Transformer,
    index: weight_index.Index,
) !void {
    for (globals) |name| try require(index, name);
    for (0..config.layers) |layer| {
        try validateBlock(allocator, index, "layers", layer, true);
    }
    for (0..config.refiner_layers) |layer| {
        try validateBlock(allocator, index, "noise_refiner", layer, true);
        try validateBlock(allocator, index, "context_refiner", layer, false);
    }
}

fn validateBlock(
    allocator: std.mem.Allocator,
    index: weight_index.Index,
    prefix: []const u8,
    layer: usize,
    modulation: bool,
) !void {
    for (block_suffixes) |suffix| try requireLayer(allocator, index, prefix, layer, suffix);
    if (!modulation) return;
    for (adaln_suffixes) |suffix| try requireLayer(allocator, index, prefix, layer, suffix);
}

fn requireLayer(
    allocator: std.mem.Allocator,
    index: weight_index.Index,
    prefix: []const u8,
    layer: usize,
    suffix: []const u8,
) !void {
    const name = try std.fmt.allocPrint(allocator, "{s}.{d}.{s}", .{ prefix, layer, suffix });
    defer allocator.free(name);
    try require(index, name);
}

fn require(index: weight_index.Index, name: []const u8) !void {
    if (index.find(name) == null) return error.MissingTensor;
}

test "validate transformer index" {
    var entries = try std.ArrayList(weight_index.Entry).initCapacity(std.testing.allocator, 64);
    errdefer {
        for (entries.items) |entry| {
            std.testing.allocator.free(entry.name);
            std.testing.allocator.free(entry.file);
        }
        entries.deinit(std.testing.allocator);
    }

    try addAll(std.testing.allocator, &entries, txConfig(1, 1));
    const owned = try entries.toOwnedSlice(std.testing.allocator);
    entries.deinit(std.testing.allocator);
    var index = weight_index.Index{ .total_size = 1, .entries = owned };
    defer index.deinit(std.testing.allocator);

    try validate(std.testing.allocator, txConfig(1, 1), index);
    try std.testing.expectError(
        error.MissingTensor,
        validate(std.testing.allocator, txConfig(2, 1), index),
    );
}

fn addAll(
    allocator: std.mem.Allocator,
    entries: *std.ArrayList(weight_index.Entry),
    config: zconfig.Transformer,
) !void {
    for (globals) |name| try add(entries, allocator, name);
    for (0..config.layers) |layer| try addBlock(allocator, entries, "layers", layer, true);
    for (0..config.refiner_layers) |layer| {
        try addBlock(allocator, entries, "noise_refiner", layer, true);
        try addBlock(allocator, entries, "context_refiner", layer, false);
    }
}

fn addBlock(
    allocator: std.mem.Allocator,
    entries: *std.ArrayList(weight_index.Entry),
    prefix: []const u8,
    layer: usize,
    modulation: bool,
) !void {
    for (block_suffixes) |suffix| try addLayer(allocator, entries, prefix, layer, suffix);
    if (!modulation) return;
    for (adaln_suffixes) |suffix| try addLayer(allocator, entries, prefix, layer, suffix);
}

fn addLayer(
    allocator: std.mem.Allocator,
    entries: *std.ArrayList(weight_index.Entry),
    prefix: []const u8,
    layer: usize,
    suffix: []const u8,
) !void {
    const name = try std.fmt.allocPrint(allocator, "{s}.{d}.{s}", .{ prefix, layer, suffix });
    defer allocator.free(name);
    try add(entries, allocator, name);
}

fn add(
    entries: *std.ArrayList(weight_index.Entry),
    allocator: std.mem.Allocator,
    name: []const u8,
) !void {
    try entries.append(allocator, .{
        .name = try allocator.dupe(u8, name),
        .file = try allocator.dupe(u8, "a"),
    });
}

fn txConfig(layers: u32, refiners: u32) zconfig.Transformer {
    return .{
        .dim = 3840,
        .layers = layers,
        .refiner_layers = refiners,
        .heads = 30,
        .kv_heads = 30,
        .cap_feat_dim = 2560,
        .in_channels = 16,
        .axes_dims = .{ 32, 48, 48 },
        .axes_lens = .{ 1536, 512, 512 },
        .norm_eps = 0.00001,
        .rope_theta = 256.0,
        .t_scale = 1000.0,
        .qk_norm = true,
    };
}
