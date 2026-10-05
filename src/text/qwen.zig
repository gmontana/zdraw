//! Qwen3 text-encoder weight-index validation.

const std = @import("std");

const weight_index = @import("../pack/weight_index.zig");
const zconfig = @import("../zimage/zimage_config.zig");

pub const Error = error{
    MissingTensor,
};

const global_names = [_][]const u8{
    "model.embed_tokens.weight",
    "model.norm.weight",
};

const layer_suffixes = [_][]const u8{
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};

pub fn validate(
    allocator: std.mem.Allocator,
    config: zconfig.Text,
    index: weight_index.Index,
) !void {
    for (global_names) |name| try require(index, name);
    for (0..config.layers) |layer| try validateLayer(allocator, index, layer);
}

fn validateLayer(
    allocator: std.mem.Allocator,
    index: weight_index.Index,
    layer: usize,
) !void {
    for (layer_suffixes) |suffix| {
        const name = try std.fmt.allocPrint(
            allocator,
            "model.layers.{d}.{s}",
            .{ layer, suffix },
        );
        defer allocator.free(name);
        try require(index, name);
    }
}

fn require(index: weight_index.Index, name: []const u8) !void {
    if (index.find(name) == null) return error.MissingTensor;
}

test "validate one qwen layer" {
    const json =
        \\{"metadata":{"total_size":1},"weight_map":{
        \\"model.embed_tokens.weight":"a",
        \\"model.norm.weight":"a",
        \\"model.layers.0.input_layernorm.weight":"a",
        \\"model.layers.0.post_attention_layernorm.weight":"a",
        \\"model.layers.0.self_attn.q_proj.weight":"a",
        \\"model.layers.0.self_attn.k_proj.weight":"a",
        \\"model.layers.0.self_attn.v_proj.weight":"a",
        \\"model.layers.0.self_attn.o_proj.weight":"a",
        \\"model.layers.0.self_attn.q_norm.weight":"a",
        \\"model.layers.0.self_attn.k_norm.weight":"a",
        \\"model.layers.0.mlp.gate_proj.weight":"a",
        \\"model.layers.0.mlp.up_proj.weight":"a",
        \\"model.layers.0.mlp.down_proj.weight":"a"}}
    ;
    var index = try weight_index.parse(std.testing.allocator, json);
    defer index.deinit(std.testing.allocator);

    try validate(std.testing.allocator, textConfig(1), index);
    try std.testing.expectError(
        error.MissingTensor,
        validate(std.testing.allocator, textConfig(2), index),
    );
}

fn textConfig(layers: u32) zconfig.Text {
    return .{
        .hidden_size = 2560,
        .intermediate_size = 9728,
        .layers = layers,
        .heads = 32,
        .kv_heads = 8,
        .head_dim = 128,
        .vocab_size = 151936,
        .rms_norm_eps = 0.000001,
        .rope_theta = 1000000.0,
    };
}
