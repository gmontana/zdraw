//! Official Z-Image VAE tensor views.
//!
//! The VAE is a single safetensors file. `load` names the decoder tensors
//! text-to-image needs; `loadEncoder` names the encoder side, which reference
//! images go through for editing. Both directions share the same resnet and
//! attention field layout, so the same prefix-generic loaders serve them.

const std = @import("std");

const tensor = @import("tensor.zig");
const tensor_file = @import("tensor_file.zig");
const vattn = @import("vattn.zig");
const vres = @import("vres.zig");

pub const Up = struct {
    res: [3]vres.Views,
    up_w: ?tensor.View,
    up_b: ?tensor.View,
};

pub const Views = struct {
    conv_in_w: tensor.View,
    conv_in_b: tensor.View,
    mid0: vres.Views,
    attn: vattn.Views,
    mid1: vres.Views,
    up: [4]Up,
    norm_w: tensor.View,
    norm_b: tensor.View,
    out_w: tensor.View,
    out_b: tensor.View,
};

/// One encoder down block: two resnets, then an optional stride-2 conv.
pub const Down = struct {
    res: [2]vres.Views,
    down_w: ?tensor.View,
    down_b: ?tensor.View,
};

/// Encoder path: image -> conv_in -> 4 down blocks -> mid (res, attn, res)
/// -> GroupNorm -> conv_out -> quant_conv, ending in mean/logvar pairs.
pub const EncoderViews = struct {
    conv_in_w: tensor.View,
    conv_in_b: tensor.View,
    down: [4]Down,
    mid0: vres.Views,
    attn: vattn.Views,
    mid1: vres.Views,
    norm_w: tensor.View,
    norm_b: tensor.View,
    out_w: tensor.View,
    out_b: tensor.View,
    quant_w: tensor.View,
    quant_b: tensor.View,
};

pub fn loadEncoder(file: *const tensor_file.Mapped) !EncoderViews {
    return .{
        .conv_in_w = try need(file, "encoder.conv_in.weight"),
        .conv_in_b = try need(file, "encoder.conv_in.bias"),
        .down = .{
            try loadDown(file, 0, true),
            try loadDown(file, 1, true),
            try loadDown(file, 2, true),
            try loadDown(file, 3, false),
        },
        .mid0 = try loadRes(file, "encoder.mid_block.resnets.0"),
        .attn = try loadAttn(file, "encoder.mid_block.attentions.0"),
        .mid1 = try loadRes(file, "encoder.mid_block.resnets.1"),
        .norm_w = try need(file, "encoder.conv_norm_out.weight"),
        .norm_b = try need(file, "encoder.conv_norm_out.bias"),
        .out_w = try need(file, "encoder.conv_out.weight"),
        .out_b = try need(file, "encoder.conv_out.bias"),
        .quant_w = try need(file, "quant_conv.weight"),
        .quant_b = try need(file, "quant_conv.bias"),
    };
}

fn loadDown(file: *const tensor_file.Mapped, block: usize, has_down: bool) !Down {
    var res: [2]vres.Views = undefined;
    for (&res, 0..) |*item, idx| {
        var prefix_buf: [96]u8 = undefined;
        item.* = try loadRes(file, try std.fmt.bufPrint(
            &prefix_buf,
            "encoder.down_blocks.{d}.resnets.{d}",
            .{ block, idx },
        ));
    }
    if (!has_down) return .{ .res = res, .down_w = null, .down_b = null };
    var weight_buf: [96]u8 = undefined;
    var bias_buf: [96]u8 = undefined;
    return .{
        .res = res,
        .down_w = try need(file, try downConvName(&weight_buf, block, "weight")),
        .down_b = try need(file, try downConvName(&bias_buf, block, "bias")),
    };
}

fn downConvName(buf: []u8, block: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(
        buf,
        "encoder.down_blocks.{d}.downsamplers.0.conv.{s}",
        .{ block, suffix },
    );
}

pub fn load(file: *const tensor_file.Mapped) !Views {
    return .{
        .conv_in_w = try need(file, "decoder.conv_in.weight"),
        .conv_in_b = try need(file, "decoder.conv_in.bias"),
        .mid0 = try loadRes(file, "decoder.mid_block.resnets.0"),
        .attn = try loadAttn(file, "decoder.mid_block.attentions.0"),
        .mid1 = try loadRes(file, "decoder.mid_block.resnets.1"),
        .up = .{
            try loadUp(file, 0, true),
            try loadUp(file, 1, true),
            try loadUp(file, 2, true),
            try loadUp(file, 3, false),
        },
        .norm_w = try need(file, "decoder.conv_norm_out.weight"),
        .norm_b = try need(file, "decoder.conv_norm_out.bias"),
        .out_w = try need(file, "decoder.conv_out.weight"),
        .out_b = try need(file, "decoder.conv_out.bias"),
    };
}

fn loadUp(file: *const tensor_file.Mapped, block: usize, has_up: bool) !Up {
    var res: [3]vres.Views = undefined;
    for (&res, 0..) |*item, idx| {
        var prefix_buf: [96]u8 = undefined;
        item.* = try loadRes(file, try upResPrefix(&prefix_buf, block, idx));
    }
    if (!has_up) return .{ .res = res, .up_w = null, .up_b = null };

    var weight_buf: [96]u8 = undefined;
    var bias_buf: [96]u8 = undefined;
    return .{
        .res = res,
        .up_w = try need(file, try upConvName(&weight_buf, block, "weight")),
        .up_b = try need(file, try upConvName(&bias_buf, block, "bias")),
    };
}

fn loadRes(file: *const tensor_file.Mapped, prefix: []const u8) !vres.Views {
    var buf: [128]u8 = undefined;
    return .{
        .norm1_w = try need(file, try field(&buf, prefix, "norm1.weight")),
        .norm1_b = try need(file, try field(&buf, prefix, "norm1.bias")),
        .conv1_w = try need(file, try field(&buf, prefix, "conv1.weight")),
        .conv1_b = try need(file, try field(&buf, prefix, "conv1.bias")),
        .norm2_w = try need(file, try field(&buf, prefix, "norm2.weight")),
        .norm2_b = try need(file, try field(&buf, prefix, "norm2.bias")),
        .conv2_w = try need(file, try field(&buf, prefix, "conv2.weight")),
        .conv2_b = try need(file, try field(&buf, prefix, "conv2.bias")),
        .skip_w = try maybe(file, try field(&buf, prefix, "conv_shortcut.weight")),
        .skip_b = try maybe(file, try field(&buf, prefix, "conv_shortcut.bias")),
    };
}

fn loadAttn(file: *const tensor_file.Mapped, prefix: []const u8) !vattn.Views {
    var buf: [128]u8 = undefined;
    return .{
        .norm_w = try need(file, try field(&buf, prefix, "group_norm.weight")),
        .norm_b = try need(file, try field(&buf, prefix, "group_norm.bias")),
        .q_w = try need(file, try field(&buf, prefix, "to_q.weight")),
        .q_b = try need(file, try field(&buf, prefix, "to_q.bias")),
        .k_w = try need(file, try field(&buf, prefix, "to_k.weight")),
        .k_b = try need(file, try field(&buf, prefix, "to_k.bias")),
        .v_w = try need(file, try field(&buf, prefix, "to_v.weight")),
        .v_b = try need(file, try field(&buf, prefix, "to_v.bias")),
        .out_w = try need(file, try field(&buf, prefix, "to_out.0.weight")),
        .out_b = try need(file, try field(&buf, prefix, "to_out.0.bias")),
    };
}

fn need(file: *const tensor_file.Mapped, name: []const u8) !tensor.View {
    return (try file.view(name)) orelse error.MissingTensor;
}

fn maybe(file: *const tensor_file.Mapped, name: []const u8) !?tensor.View {
    return try file.view(name);
}

fn field(buf: []u8, prefix: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ prefix, suffix });
}

fn upResPrefix(buf: []u8, block: usize, res: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "decoder.up_blocks.{d}.resnets.{d}", .{ block, res });
}

fn upConvName(buf: []u8, block: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(
        buf,
        "decoder.up_blocks.{d}.upsamplers.0.conv.{s}",
        .{ block, suffix },
    );
}

test "official VAE names stay exact" {
    var buf: [96]u8 = undefined;

    try std.testing.expectEqualStrings(
        "decoder.up_blocks.2.resnets.1",
        try upResPrefix(&buf, 2, 1),
    );
    try std.testing.expectEqualStrings(
        "decoder.up_blocks.0.upsamplers.0.conv.weight",
        try upConvName(&buf, 0, "weight"),
    );
}
