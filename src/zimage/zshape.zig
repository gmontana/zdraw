//! Small Z-Image transformer shape formulas.
//!
//! These are fixed by the model architecture, not policy knobs.

const std = @import("std");

const zconfig = @import("zimage_config.zig");

pub const Error = error{
    InvalidShape,
};

pub fn headDim(cfg: zconfig.Transformer) !usize {
    if (cfg.heads == 0 or cfg.dim % cfg.heads != 0) return error.InvalidShape;
    const dim = cfg.dim / cfg.heads;
    if (dim != cfg.axes_dims[0] + cfg.axes_dims[1] + cfg.axes_dims[2]) {
        return error.InvalidShape;
    }
    return dim;
}

pub fn ffnDim(cfg: zconfig.Transformer) usize {
    return @as(usize, cfg.dim) / 3 * 8;
}

pub fn patchDim(cfg: zconfig.Transformer) usize {
    return @as(usize, cfg.in_channels) * 4;
}

test "official Z-Image dimensions" {
    const cfg = txConfig();
    try std.testing.expectEqual(@as(usize, 128), try headDim(cfg));
    try std.testing.expectEqual(@as(usize, 10240), ffnDim(cfg));
    try std.testing.expectEqual(@as(usize, 64), patchDim(cfg));
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
