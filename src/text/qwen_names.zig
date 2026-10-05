//! Qwen weight names, kept in one place.

const std = @import("std");

pub const embed = "model.embed_tokens.weight";
pub const final_norm = "model.norm.weight";

pub const Layer = enum {
    input_norm,
    post_norm,
    q,
    k,
    v,
    o,
    q_norm,
    k_norm,
    gate,
    up,
    down,
};

pub fn layerName(
    allocator: std.mem.Allocator,
    layer: usize,
    part: Layer,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "model.layers.{d}.{s}",
        .{ layer, suffix(part) },
    );
}

pub fn suffix(part: Layer) []const u8 {
    return switch (part) {
        .input_norm => "input_layernorm.weight",
        .post_norm => "post_attention_layernorm.weight",
        .q => "self_attn.q_proj.weight",
        .k => "self_attn.k_proj.weight",
        .v => "self_attn.v_proj.weight",
        .o => "self_attn.o_proj.weight",
        .q_norm => "self_attn.q_norm.weight",
        .k_norm => "self_attn.k_norm.weight",
        .gate => "mlp.gate_proj.weight",
        .up => "mlp.up_proj.weight",
        .down => "mlp.down_proj.weight",
    };
}

test "build layer weight name" {
    const name = try layerName(std.testing.allocator, 3, .q);
    defer std.testing.allocator.free(name);
    try std.testing.expectEqualStrings("model.layers.3.self_attn.q_proj.weight", name);
}
