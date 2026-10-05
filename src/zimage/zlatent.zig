//! Latent tensor shape for Z-Image generation.
//!
//! The VAE scale is 8, and the transformer patches latents by 2, so requested
//! pixel sizes must be divisible by 16.

const std = @import("std");

const zconfig = @import("zimage_config.zig");
const zpatch = @import("zpatch.zig");

pub const Error = error{
    InvalidImageSize,
};

pub fn shape(width: u32, height: u32, cfg: zconfig.Transformer) !zpatch.Shape {
    if (width == 0 or height == 0) return error.InvalidImageSize;
    if (width % 16 != 0 or height % 16 != 0) return error.InvalidImageSize;
    return .{
        .channels = @intCast(cfg.in_channels),
        .frames = 1,
        .height = @intCast(height / 8),
        .width = @intCast(width / 8),
        .patch = 2,
        .f_patch = 1,
    };
}

pub fn len(latent_shape: zpatch.Shape) usize {
    return latent_shape.channels * latent_shape.frames *
        latent_shape.height * latent_shape.width;
}

test "pixel size maps to latent size" {
    const got = try shape(1024, 512, txConfig());

    try std.testing.expectEqual(@as(usize, 16), got.channels);
    try std.testing.expectEqual(@as(usize, 64), got.height);
    try std.testing.expectEqual(@as(usize, 128), got.width);
    try std.testing.expectEqual(@as(usize, 131072), len(got));
}

test "reject sizes not divisible by sixteen" {
    try std.testing.expectError(error.InvalidImageSize, shape(1000, 512, txConfig()));
}

fn txConfig() zconfig.Transformer {
    return .{
        .dim = 3840,
        .layers = 30,
        .refiner_layers = 2,
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
