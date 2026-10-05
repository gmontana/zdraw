//! Tensor views for one Z-Image transformer block.
//!
//! Keeping the weight lookup here lets the math code talk about q, k, v,
//! norms, and feed-forward weights without carrying string names around.

const std = @import("std");

const shards = @import("shards.zig");
const tensor = @import("tensor.zig");
const weight_index = @import("weight_index.zig");

pub const Views = struct {
    q: tensor.View,
    k: tensor.View,
    v: tensor.View,
    proj: tensor.View,
    q_norm: tensor.View,
    k_norm: tensor.View,
    attn_in: tensor.View,
    attn_out: tensor.View,
    ffn_in: tensor.View,
    ffn_out: tensor.View,
    ffn_gate: tensor.View,
    ffn_down: tensor.View,
    ffn_up: tensor.View,
    ada_w: ?tensor.View,
    ada_b: ?tensor.View,
    // Derived fused [gate; up] f16 view, set by sidecar substitution only.
    ffn_fused: ?tensor.View = null,
};

pub fn load(
    allocator: std.mem.Allocator,
    store: *const shards.Store,
    index: weight_index.Index,
    prefix: []const u8,
    layer: usize,
    modulation: bool,
) !Views {
    return .{
        .q = try need(allocator, store, index, prefix, layer, "attention.to_q.weight"),
        .k = try need(allocator, store, index, prefix, layer, "attention.to_k.weight"),
        .v = try need(allocator, store, index, prefix, layer, "attention.to_v.weight"),
        .proj = try need(allocator, store, index, prefix, layer, "attention.to_out.0.weight"),
        .q_norm = try need(allocator, store, index, prefix, layer, "attention.norm_q.weight"),
        .k_norm = try need(allocator, store, index, prefix, layer, "attention.norm_k.weight"),
        .attn_in = try need(allocator, store, index, prefix, layer, "attention_norm1.weight"),
        .attn_out = try need(allocator, store, index, prefix, layer, "attention_norm2.weight"),
        .ffn_in = try need(allocator, store, index, prefix, layer, "ffn_norm1.weight"),
        .ffn_out = try need(allocator, store, index, prefix, layer, "ffn_norm2.weight"),
        .ffn_gate = try need(allocator, store, index, prefix, layer, "feed_forward.w1.weight"),
        .ffn_down = try need(allocator, store, index, prefix, layer, "feed_forward.w2.weight"),
        .ffn_up = try need(allocator, store, index, prefix, layer, "feed_forward.w3.weight"),
        .ada_w = if (modulation) try need(allocator, store, index, prefix, layer, adaW()) else null,
        .ada_b = if (modulation) try need(allocator, store, index, prefix, layer, adaB()) else null,
    };
}

pub fn name(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    layer: usize,
    suffix: []const u8,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.{d}.{s}", .{ prefix, layer, suffix });
}

fn need(
    allocator: std.mem.Allocator,
    store: *const shards.Store,
    index: weight_index.Index,
    prefix: []const u8,
    layer: usize,
    suffix: []const u8,
) !tensor.View {
    const key = try name(allocator, prefix, layer, suffix);
    defer allocator.free(key);
    return store.view(index, key);
}

fn adaW() []const u8 {
    return "adaLN_modulation.0.weight";
}

fn adaB() []const u8 {
    return "adaLN_modulation.0.bias";
}

test "formats block tensor name" {
    const got = try name(
        std.testing.allocator,
        "layers",
        12,
        "attention.to_q.weight",
    );
    defer std.testing.allocator.free(got);

    try std.testing.expectEqualStrings("layers.12.attention.to_q.weight", got);
}

test "formats refiner tensor name" {
    const got = try name(
        std.testing.allocator,
        "noise_refiner",
        1,
        adaW(),
    );
    defer std.testing.allocator.free(got);

    try std.testing.expectEqualStrings("noise_refiner.1.adaLN_modulation.0.weight", got);
}
