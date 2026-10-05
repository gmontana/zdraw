//! img2img: the initial image becomes the starting latent. Decode + cover
//! resize to the target size, encode with the VAE encoder (vencode) on the
//! runtime's Metal contexts, then the DiT-space packing diffusers' pipeline
//! applies (patchify 2x2 -> packed channel i*4 + dy*2 + dx, BN-normalise per
//! packed channel, one token per 2x2 latent cell). The sampler then starts
//! at the schedule step `strength` selects: x = (1 - sigma) * z0 + sigma * noise.
const std = @import("std");
const image = @import("../cli/image.zig");
const mattn = @import("../metal/mattn.zig");
const mconv = @import("../metal/mconv.zig");
const mlinear = @import("../metal/mlinear.zig");
const tensor_file = @import("../pack/tensor_file.zig");
const vencode = @import("../vae/vencode.zig");
const vviews = @import("../vae/vviews.zig");

/// The image as f32 [3, height, width] in [-1, 1], cover-resized (aspect
/// fill, centre crop) with bilinear sampling. Caller frees.
pub fn loadRgb(allocator: std.mem.Allocator, path: []const u8, width: usize, height: usize) ![]f32 {
    var src = try image.readRgba(allocator, path);
    defer src.deinit(allocator);
    const sw: f64 = @floatFromInt(src.width);
    const sh: f64 = @floatFromInt(src.height);
    const tw: f64 = @floatFromInt(width);
    const th: f64 = @floatFromInt(height);
    const scale = @max(tw / sw, th / sh);
    const off_x = (sw - tw / scale) / 2.0;
    const off_y = (sh - th / scale) / 2.0;
    const out = try allocator.alloc(f32, 3 * width * height);
    errdefer allocator.free(out);
    for (0..height) |y| {
        const yy: f64 = @floatFromInt(y);
        const fy = @min(@max(off_y + (yy + 0.5) / scale - 0.5, 0.0), sh - 1.0);
        for (0..width) |x| {
            const xx: f64 = @floatFromInt(x);
            const fx = @min(@max(off_x + (xx + 0.5) / scale - 0.5, 0.0), sw - 1.0);
            for (0..3) |ch| {
                const v = bilinear(src, fx, fy, ch);
                out[ch * width * height + y * width + x] = v / 127.5 - 1.0;
            }
        }
    }
    return out;
}

fn bilinear(src: image.Rgba, fx: f64, fy: f64, ch: usize) f32 {
    const x0: usize = @intFromFloat(@floor(fx));
    const y0: usize = @intFromFloat(@floor(fy));
    const x1 = @min(x0 + 1, src.width - 1);
    const y1 = @min(y0 + 1, src.height - 1);
    const ax: f64 = fx - @as(f64, @floatFromInt(x0));
    const ay: f64 = fy - @as(f64, @floatFromInt(y0));
    const p = struct {
        fn at(s: image.Rgba, x: usize, y: usize, c: usize) f64 {
            return @floatFromInt(s.pixels[(y * s.width + x) * 4 + c]);
        }
    };
    const top = p.at(src, x0, y0, ch) * (1 - ax) + p.at(src, x1, y0, ch) * ax;
    const bottom = p.at(src, x0, y1, ch) * (1 - ax) + p.at(src, x1, y1, ch) * ax;
    return @floatCast(top * (1 - ay) + bottom * ay);
}

/// The packed, BN-normalised DiT-space latent z0 ([tokens, 128]) of an RGB
/// image, through the VAE encoder on the given contexts. Caller frees.
pub fn encodePacked(
    allocator: std.mem.Allocator,
    lin: ?*mlinear.Context,
    conv: ?*mconv.Context,
    attn: ?*mattn.Context,
    vae: *const tensor_file.Mapped,
    rgb: []const f32,
    width: usize,
    height: usize,
) ![]f32 {
    const views = try vviews.loadEncoder(vae);
    const ecfg = vencode.Config{ .height = height, .width = width };
    const mean = try vencode.mean(allocator, lin, conv, attn, rgb, views, ecfg);
    defer allocator.free(mean);
    const lh = height / 8;
    const lw = width / 8;
    const gh = lh / 2;
    const gw = lw / 2;
    const bn_mean = (try vae.view("bn.running_mean")) orelse return error.MissingTensor;
    const bn_var = (try vae.view("bn.running_var")) orelse return error.MissingTensor;
    const packed_len = gh * gw * 128;
    const out = try allocator.alloc(f32, packed_len);
    errdefer allocator.free(out);
    for (0..gh) |th| for (0..gw) |tw| {
        const base = (th * gw + tw) * 128;
        for (0..32) |i| for (0..2) |dy| for (0..2) |dx| {
            const pc = i * 4 + dy * 2 + dx;
            const m = bn_mean.atF32Unchecked(pc);
            const sd = @sqrt(bn_var.atF32Unchecked(pc) + 1e-4);
            const v = mean[i * lh * lw + (2 * th + dy) * lw + (2 * tw + dx)];
            out[base + pc] = (v - m) / sd;
        };
    };
    return out;
}

/// x <- (1 - sigma) * z0 + sigma * x, with x holding the seeded noise.
pub fn mix(x: []f32, z0: []const f32, sigma: f32) void {
    for (x, z0) |*xi, zi| xi.* = (1.0 - sigma) * zi + sigma * xi.*;
}

/// "Fix a part": the mask as one weight per token (white = change), the
/// mean of the 16x16 pixel block each token covers, cover-resized like the
/// image. Caller frees.
pub fn loadMask(
    allocator: std.mem.Allocator,
    path: []const u8,
    width: usize,
    height: usize,
) ![]f32 {
    const rgb = try loadRgb(allocator, path, width, height);
    defer allocator.free(rgb);
    const gw = width / 16;
    const gh = height / 16;
    const out = try allocator.alloc(f32, gw * gh);
    for (0..gh) |ty| for (0..gw) |tx| {
        var acc: f32 = 0;
        for (0..16) |dy| for (0..16) |dx| {
            const px = (ty * 16 + dy) * width + tx * 16 + dx;
            acc += (rgb[px] + rgb[width * height + px] + rgb[2 * width * height + px]) / 3.0;
        };
        // rgb is in [-1, 1]; white = 1 = change.
        out[ty * gw + tx] = std.math.clamp((acc / 256.0 + 1.0) / 2.0, 0.0, 1.0);
    };
    // A one-token feather (3x3 box) so the held region blends into the
    // redrawn one in latent space; the pixel composite feathers again.
    const soft = try allocator.alloc(f32, gw * gh);
    defer allocator.free(soft);
    for (0..gh) |ty| for (0..gw) |tx| {
        var acc: f32 = 0;
        var n: f32 = 0;
        const y0 = ty -| 1;
        const x0 = tx -| 1;
        for (y0..@min(ty + 2, gh)) |yy| for (x0..@min(tx + 2, gw)) |xx| {
            acc += out[yy * gw + xx];
            n += 1;
        };
        soft[ty * gw + tx] = @max(out[ty * gw + tx], acc / n);
    };
    @memcpy(out, soft);
    return out;
}

/// Hold the unmasked tokens to the original: after each Euler step the
/// latent outside the mask is reset to the original's latent at the next
/// sigma, so only the painted region is redrawn and the rest lands exactly
/// on the input (through the VAE round trip).
pub fn repaint(
    x: []f32,
    z0: []const f32,
    noise0: []const f32,
    mask: []const f32,
    sigma_next: f32,
) void {
    for (mask, 0..) |m, t| {
        if (m >= 0.999) continue;
        const base = t * 128;
        for (0..128) |c| {
            const k = base + c;
            const held = (1.0 - sigma_next) * z0[k] + sigma_next * noise0[k];
            x[k] = m * x[k] + (1.0 - m) * held;
        }
    }
}

test "repaint holds unmasked tokens and frees masked ones" {
    var x = [_]f32{5} ** 256;
    const z0 = [_]f32{1} ** 256;
    const noise0 = [_]f32{3} ** 256;
    const mask = [_]f32{ 0, 1 };
    repaint(&x, &z0, &noise0, &mask, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[128], 1e-6);
}
