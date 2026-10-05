//! Shape helpers for one Z-Image transformer step.

const std = @import("std");

const zconfig = @import("zimage_config.zig");
const zembed = @import("zembed.zig");
const zfinal = @import("zfinal.zig");
const zlayer = @import("zlayer.zig");
const zpatch = @import("zpatch.zig");
const zseq = @import("zseq.zig");
const zshape = @import("zshape.zig");

pub const Input = struct {
    latent: []const f32,
    cap: []const f32,
    shape: zpatch.Shape,
    time: f32,
};

pub const Dims = struct {
    hidden: usize,
    patch_dim: usize,
    img_raw: usize,
    img_total: usize,
    cap_tokens: usize,
    cap_total: usize,
    total: usize,
};

pub fn get(input: Input, cfg: zconfig.Transformer) !Dims {
    const patch_dim = try zpatch.patchDim(input.shape);
    if (patch_dim != zshape.patchDim(cfg)) return error.InvalidShape;
    const img_raw = try zpatch.patchCount(input.shape);
    const cap_dim: usize = @intCast(cfg.cap_feat_dim);
    if (cap_dim == 0 or input.cap.len % cap_dim != 0) return error.InvalidShape;
    const cap_tokens = input.cap.len / cap_dim;
    const img_total = zseq.paddedLen(img_raw);
    const cap_total = zseq.paddedLen(cap_tokens);
    return .{
        .hidden = @intCast(cfg.dim),
        .patch_dim = patch_dim,
        .img_raw = img_raw,
        .img_total = img_total,
        .cap_tokens = cap_tokens,
        .cap_total = cap_total,
        .total = img_total + cap_total,
    };
}

pub fn imgGrid(input: Input) zseq.Pos {
    return .{
        input.shape.frames / input.shape.f_patch,
        input.shape.height / input.shape.patch,
        input.shape.width / input.shape.patch,
    };
}

pub fn imgStart(dims: Dims) zseq.Pos {
    return .{ dims.cap_total + 1, 0, 0 };
}

pub fn capGrid(dims: Dims) zseq.Pos {
    // Grid spans the padded caption length so pad tokens keep advancing the frame
    // axis (reference uses pos_grid_size = padded length for the caption stream).
    return .{ dims.cap_total, 1, 1 };
}

pub fn imgCfg(dims: Dims, cfg: zconfig.Transformer) zembed.Config {
    return .{
        .tokens = dims.img_total,
        .in_dim = dims.patch_dim,
        .hidden = dims.hidden,
        .norm_eps = @floatCast(cfg.norm_eps),
    };
}

pub fn capCfg(dims: Dims, cfg: zconfig.Transformer) zembed.Config {
    return .{
        .tokens = dims.cap_total,
        .in_dim = @intCast(cfg.cap_feat_dim),
        .hidden = dims.hidden,
        .norm_eps = @floatCast(cfg.norm_eps),
    };
}

pub fn layerCfg(state: []const f32, cfg: zconfig.Transformer) !zlayer.Config {
    const hidden: usize = @intCast(cfg.dim);
    if (hidden == 0 or state.len % hidden != 0) return error.InvalidShape;
    return .{
        .tokens = state.len / hidden,
        .hidden = hidden,
        .heads = @intCast(cfg.heads),
        .kv_heads = @intCast(cfg.kv_heads),
        .head_dim = try zshape.headDim(cfg),
        .norm_eps = @floatCast(cfg.norm_eps),
    };
}

pub fn finalCfg(dims: Dims, cfg: zconfig.Transformer) zfinal.Config {
    return .{
        .tokens = dims.total,
        .hidden = dims.hidden,
        .out_dim = dims.patch_dim,
        .norm_eps = @floatCast(cfg.norm_eps),
    };
}

test "derive padded token counts" {
    const input = Input{
        .latent = &([_]f32{0} ** 256),
        .cap = &([_]f32{0} ** 5120),
        .shape = .{
            .channels = 16,
            .frames = 1,
            .height = 4,
            .width = 4,
            .patch = 2,
            .f_patch = 1,
        },
        .time = 1.0,
    };
    const dims = try get(input, txConfig());

    try std.testing.expectEqual(@as(usize, 4), dims.img_raw);
    try std.testing.expectEqual(@as(usize, 32), dims.img_total);
    try std.testing.expectEqual(@as(usize, 2), dims.cap_tokens);
    try std.testing.expectEqual(@as(usize, 32), dims.cap_total);
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
