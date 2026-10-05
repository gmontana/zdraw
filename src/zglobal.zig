//! Global tensor views for the Z-Image transformer.
//!
//! These are the non-repeated weights: embedders, timestep MLP, final layer,
//! and learned pad tokens.

const std = @import("std");

const shards = @import("shards.zig");
const tensor = @import("tensor.zig");
const weight_index = @import("weight_index.zig");

pub const Views = struct {
    x_w: tensor.View,
    x_b: tensor.View,
    cap_norm: tensor.View,
    cap_w: tensor.View,
    cap_b: tensor.View,
    t0_w: tensor.View,
    t0_b: tensor.View,
    t1_w: tensor.View,
    t1_b: tensor.View,
    final_mod_w: tensor.View,
    final_mod_b: tensor.View,
    final_w: tensor.View,
    final_b: tensor.View,
    x_pad: tensor.View,
    cap_pad: tensor.View,
};

pub fn load(store: *const shards.Store, index: weight_index.Index) !Views {
    return .{
        .x_w = try need(store, index, xW()),
        .x_b = try need(store, index, xB()),
        .cap_norm = try need(store, index, "cap_embedder.0.weight"),
        .cap_w = try need(store, index, "cap_embedder.1.weight"),
        .cap_b = try need(store, index, "cap_embedder.1.bias"),
        .t0_w = try need(store, index, "t_embedder.mlp.0.weight"),
        .t0_b = try need(store, index, "t_embedder.mlp.0.bias"),
        .t1_w = try need(store, index, "t_embedder.mlp.2.weight"),
        .t1_b = try need(store, index, "t_embedder.mlp.2.bias"),
        .final_mod_w = try need(store, index, finalModW()),
        .final_mod_b = try need(store, index, finalModB()),
        .final_w = try need(store, index, "all_final_layer.2-1.linear.weight"),
        .final_b = try need(store, index, "all_final_layer.2-1.linear.bias"),
        .x_pad = try need(store, index, "x_pad_token"),
        .cap_pad = try need(store, index, "cap_pad_token"),
    };
}

fn need(store: *const shards.Store, index: weight_index.Index, name: []const u8) !tensor.View {
    return store.view(index, name);
}

fn xW() []const u8 {
    return "all_x_embedder.2-1.weight";
}

fn xB() []const u8 {
    return "all_x_embedder.2-1.bias";
}

fn finalModW() []const u8 {
    return "all_final_layer.2-1.adaLN_modulation.1.weight";
}

fn finalModB() []const u8 {
    return "all_final_layer.2-1.adaLN_modulation.1.bias";
}

test "global tensor names stay exact" {
    try std.testing.expectEqualStrings("all_x_embedder.2-1.weight", xW());
    try std.testing.expectEqualStrings(
        "all_final_layer.2-1.adaLN_modulation.1.bias",
        finalModB(),
    );
}
