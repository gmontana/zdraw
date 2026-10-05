//! Prepare image and caption streams for the Z-Image transformer.

const std = @import("std");

const mlinear = @import("../metal/mlinear.zig");
const tensor = @import("../pack/tensor.zig");
const zconfig = @import("zimage_config.zig");
const zembed = @import("zembed.zig");
const zpatch = @import("zpatch.zig");
const zseq = @import("zseq.zig");
const zs = @import("zstep_shape.zig");
const ztx = @import("ztx.zig");

pub const Streams = struct {
    image: []f32,
    img_pos: []zseq.Pos,
    cap: []f32,
    cap_pos: []zseq.Pos,
};

pub const Image = struct {
    image: []f32,
    img_pos: []zseq.Pos,
};

pub const Caption = struct {
    cap: []f32,
    cap_pos: []zseq.Pos,
};

pub fn make(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !Streams {
    const image = try makeImage(metal, allocator, input, tx, cfg, dims);
    const cap = try makeCaption(metal, allocator, input, tx, cfg, dims);
    return .{
        .image = image.image,
        .img_pos = image.img_pos,
        .cap = cap.cap,
        .cap_pos = cap.cap_pos,
    };
}

pub fn makeImage(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !Image {
    const raw = try imageRaw(allocator, input, dims);
    const padded_image = try imagePad(allocator, input, dims, raw);
    const image = try allocator.alloc(f32, dims.img_total * dims.hidden);
    try imageEmbed(metal, image, padded_image.data, tx, cfg, dims);
    const pad = try padVec(allocator, tx.globals.x_pad);
    try zseq.applyPad(image, padded_image.mask, pad, dims.hidden);
    return .{ .image = image, .img_pos = padded_image.pos };
}

pub fn makeCaption(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !Caption {
    const padded_cap = try capPad(allocator, input, cfg, dims);
    const cap = try allocator.alloc(f32, dims.cap_total * dims.hidden);
    try capEmbed(metal, allocator, cap, padded_cap.data, tx, cfg, dims);
    try zseq.applyPad(cap, padded_cap.mask, try padVec(allocator, tx.globals.cap_pad), dims.hidden);
    return .{ .cap = cap, .cap_pos = padded_cap.pos };
}

const Padded = struct {
    data: []f32,
    pos: []zseq.Pos,
    mask: []bool,
};

fn imageRaw(allocator: std.mem.Allocator, input: zs.Input, dims: zs.Dims) ![]f32 {
    const raw = try allocator.alloc(f32, dims.img_raw * dims.patch_dim);
    try zpatch.patchify(raw, input.latent, input.shape);
    return raw;
}

fn imagePad(
    allocator: std.mem.Allocator,
    input: zs.Input,
    dims: zs.Dims,
    raw: []const f32,
) !Padded {
    const out = try padded(allocator, dims.img_total, dims.patch_dim);
    _ = try zseq.padWithIds(
        out.data,
        out.pos,
        out.mask,
        raw,
        dims.patch_dim,
        zs.imgGrid(input),
        zs.imgStart(dims),
    );
    return out;
}

fn capPad(
    allocator: std.mem.Allocator,
    input: zs.Input,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !Padded {
    const dim: usize = @intCast(cfg.cap_feat_dim);
    const out = try padded(allocator, dims.cap_total, dim);
    _ = try zseq.padWithIds(
        out.data,
        out.pos,
        out.mask,
        input.cap,
        dim,
        zs.capGrid(dims),
        .{ 1, 0, 0 },
    );
    return out;
}

fn padded(allocator: std.mem.Allocator, tokens: usize, dim: usize) !Padded {
    return .{
        .data = try allocator.alloc(f32, tokens * dim),
        .pos = try allocator.alloc(zseq.Pos, tokens),
        .mask = try allocator.alloc(bool, tokens),
    };
}

fn imageEmbed(
    metal: ?*mlinear.Context,
    out: []f32,
    input: []const f32,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !void {
    try zembed.image(metal, out, input, tx.globals.x_w, tx.globals.x_b, zs.imgCfg(dims, cfg));
}

fn capEmbed(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    out: []f32,
    input: []const f32,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    dims: zs.Dims,
) !void {
    const cap_cfg = zs.capCfg(dims, cfg);
    const scratch = try allocator.alloc(f32, cap_cfg.tokens * cap_cfg.in_dim);
    try zembed.caption(
        metal,
        out,
        input,
        tx.globals.cap_norm,
        tx.globals.cap_w,
        tx.globals.cap_b,
        scratch,
        cap_cfg,
    );
}

fn padVec(allocator: std.mem.Allocator, view: tensor.View) ![]f32 {
    const elems = try view.elems();
    const out = try allocator.alloc(f32, elems);
    try view.copyF32(out);
    return out;
}
