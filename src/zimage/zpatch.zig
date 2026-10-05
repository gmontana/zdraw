//! Patch and unpatch Z-Image latent tensors.
//!
//! Input latents are stored as C, F, H, W. Patch tokens are emitted in
//! F-token, H-token, W-token order, with channels last inside each patch.

const std = @import("std");

pub const Error = error{
    InvalidShape,
};

pub const Shape = struct {
    channels: usize,
    frames: usize,
    height: usize,
    width: usize,
    patch: usize,
    f_patch: usize,
};

pub fn patchify(out: []f32, image: []const f32, shape: Shape) !void {
    try check(out, image, shape);
    var pos: usize = 0;
    for (0..shape.frames / shape.f_patch) |ft| {
        for (0..shape.height / shape.patch) |ht| {
            for (0..shape.width / shape.patch) |wt| {
                pos = copyPatch(out, pos, image, shape, .{ ft, ht, wt });
            }
        }
    }
}

pub fn unpatchify(out: []f32, patches: []const f32, shape: Shape) !void {
    try check(patches, out, shape);
    var pos: usize = 0;
    for (0..shape.frames / shape.f_patch) |ft| {
        for (0..shape.height / shape.patch) |ht| {
            for (0..shape.width / shape.patch) |wt| {
                pos = putPatch(out, pos, patches, shape, .{ ft, ht, wt });
            }
        }
    }
}

pub fn patchCount(shape: Shape) !usize {
    try checkShape(shape);
    return shape.frames / shape.f_patch * (shape.height / shape.patch) *
        (shape.width / shape.patch);
}

pub fn patchDim(shape: Shape) !usize {
    try checkShape(shape);
    return shape.f_patch * shape.patch * shape.patch * shape.channels;
}

fn copyPatch(out: []f32, pos: usize, image: []const f32, shape: Shape, at: [3]usize) usize {
    var dst = pos;
    for (0..shape.f_patch) |pf| {
        for (0..shape.patch) |ph| {
            for (0..shape.patch) |pw| {
                for (0..shape.channels) |chan| {
                    out[dst] = image[index(shape, chan, at, .{ pf, ph, pw })];
                    dst += 1;
                }
            }
        }
    }
    return dst;
}

fn putPatch(out: []f32, pos: usize, patches: []const f32, shape: Shape, at: [3]usize) usize {
    var src = pos;
    for (0..shape.f_patch) |pf| {
        for (0..shape.patch) |ph| {
            for (0..shape.patch) |pw| {
                for (0..shape.channels) |chan| {
                    out[index(shape, chan, at, .{ pf, ph, pw })] = patches[src];
                    src += 1;
                }
            }
        }
    }
    return src;
}

fn index(shape: Shape, chan: usize, at: [3]usize, inner: [3]usize) usize {
    const frame = at[0] * shape.f_patch + inner[0];
    const row = at[1] * shape.patch + inner[1];
    const col = at[2] * shape.patch + inner[2];
    return ((chan * shape.frames + frame) * shape.height + row) * shape.width + col;
}

fn check(patches: []const f32, image: []const f32, shape: Shape) !void {
    const pc = try patchCount(shape);
    const pd = try patchDim(shape);
    if (patches.len != pc * pd) return error.InvalidShape;
    if (image.len != shape.channels * shape.frames * shape.height * shape.width) {
        return error.InvalidShape;
    }
}

fn checkShape(shape: Shape) !void {
    if (shape.channels == 0 or shape.frames == 0) return error.InvalidShape;
    if (shape.height == 0 or shape.width == 0) return error.InvalidShape;
    if (shape.patch == 0 or shape.f_patch == 0) return error.InvalidShape;
    if (shape.height % shape.patch != 0) return error.InvalidShape;
    if (shape.width % shape.patch != 0) return error.InvalidShape;
    if (shape.frames % shape.f_patch != 0) return error.InvalidShape;
}

test "patchify stores channels last inside patch" {
    const shape = Shape{
        .channels = 2,
        .frames = 1,
        .height = 2,
        .width = 2,
        .patch = 2,
        .f_patch = 1,
    };
    const image = [_]f32{ 1, 2, 3, 4, 10, 20, 30, 40 };
    var out = [_]f32{0} ** 8;

    try patchify(&out, &image, shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 10, 2, 20, 3, 30, 4, 40 }, &out);
}

test "patchify and unpatchify round trip" {
    const shape = Shape{
        .channels = 1,
        .frames = 1,
        .height = 4,
        .width = 4,
        .patch = 2,
        .f_patch = 1,
    };
    const image = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var patches = [_]f32{0} ** 16;
    var back = [_]f32{0} ** 16;

    try patchify(&patches, &image, shape);
    try unpatchify(&back, &patches, shape);
    try std.testing.expectEqualSlices(f32, &image, &back);
}
