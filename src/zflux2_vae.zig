//! FLUX.2 Klein VAE decode pre-steps: packed-space BatchNorm denormalization
//! + 2x2 unpatchify (zflux2_latent.unpack) + post_quant_conv, feeding zdraw's
//! existing AutoencoderKL decoder (vviews/vdecode — Klein's VAE matches its
//! names and topology exactly: 3 resnets per up block, mid res+attn+res).
//!
//! Default path (ZDRAW_VAE_PQFOLD): the three linear maps compose into four
//! per-subpixel folded 32x32 matrices plus biases applied in ONE pass over
//! the packed latents; the two intermediate buffers and the strided conv
//! loop disappear. Exact algebra, float-order change only; `0` restores the
//! three-pass reference (the bisection lever).

const std = @import("std");

const env = @import("env.zig");
const tensor_file = @import("tensor_file.zig");
const zflux2_latent = @import("zflux2_latent.zig");

/// Token-major packed latents -> BN-denormalized (x*sqrt(var+eps)+mean per
/// packed channel) -> unpatchified 32-channel latents -> post_quant_conv
/// (1x1, 32->32; absent from the Z-Image path, required here).
pub fn prepare(
    allocator: std.mem.Allocator,
    latents: []const f32, // [tokens, 128]
    gh: usize,
    gw: usize,
    vae_file: *const tensor_file.Mapped,
) ![]f32 {
    const bn_mean = (try vae_file.view("bn.running_mean")) orelse return error.MissingTensor;
    const bn_var = (try vae_file.view("bn.running_var")) orelse return error.MissingTensor;
    var stds: [128]f32 = undefined;
    var means: [128]f32 = undefined;
    for (0..128) |c| {
        stds[c] = @sqrt(bn_var.atF32Unchecked(c) + 1e-4);
        means[c] = bn_mean.atF32Unchecked(c);
    }
    const pq_w = (try vae_file.view("post_quant_conv.weight")) orelse return error.MissingTensor;
    const pq_b = (try vae_file.view("post_quant_conv.bias")) orelse return error.MissingTensor;
    var wmat: [32][32]f32 = undefined;
    var bias: [32]f32 = undefined;
    for (0..32) |o| {
        bias[o] = pq_b.atF32Unchecked(o);
        for (0..32) |i| wmat[o][i] = pq_w.atF32Unchecked(o * 32 + i);
    }
    const hw = (gh * 2) * (gw * 2);
    const out = try allocator.alloc(f32, 32 * hw);
    errdefer allocator.free(out);
    if (env.flag("ZDRAW_VAE_PQFOLD", true)) {
        try foldApply(out, latents, gh, gw, &stds, &means, &wmat, &bias);
    } else {
        try threePass(allocator, out, latents, gh, gw, &stds, &means, &wmat, &bias);
    }
    return out;
}

/// One pass over the packed latents with the composed maps. For output pixel
/// (o, 2*oh+dy, 2*ow+dx): unpacked channel i reads packed channel i*4+s with
/// s = dy*2+dx, so W_f[s][o][i] = W_pq[o][i]*std[i*4+s] and b_f[s][o] =
/// b_pq[o] + sum_i W_pq[o][i]*mean[i*4+s].
fn foldApply(
    out: []f32,
    latents: []const f32,
    gh: usize,
    gw: usize,
    stds: *const [128]f32,
    means: *const [128]f32,
    wmat: *const [32][32]f32,
    bias: *const [32]f32,
) !void {
    if (latents.len != gh * gw * 128) return error.InvalidBufferLength;
    var wf: [4][32][32]f32 = undefined;
    var bf: [4][32]f32 = undefined;
    for (0..4) |s| {
        for (0..32) |o| {
            var acc: f32 = bias[o];
            for (0..32) |i| {
                wf[s][o][i] = wmat[o][i] * stds[i * 4 + s];
                acc += wmat[o][i] * means[i * 4 + s];
            }
            bf[s][o] = acc;
        }
    }
    const height = gh * 2;
    const width = gw * 2;
    const hw = height * width;
    for (0..gh) |oh| {
        for (0..gw) |ow| {
            const px = latents[(oh * gw + ow) * 128 ..][0..128];
            inline for (0..2) |dy| {
                inline for (0..2) |dx| {
                    const s = dy * 2 + dx;
                    const pos = (oh * 2 + dy) * width + (ow * 2 + dx);
                    for (0..32) |o| {
                        var acc: f32 = bf[s][o];
                        for (0..32) |i| acc += wf[s][o][i] * px[i * 4 + s];
                        out[o * hw + pos] = acc;
                    }
                }
            }
        }
    }
}

/// The three-pass reference: denormalize, unpack, then the strided 1x1 conv.
fn threePass(
    allocator: std.mem.Allocator,
    out: []f32,
    latents: []const f32,
    gh: usize,
    gw: usize,
    stds: *const [128]f32,
    means: *const [128]f32,
    wmat: *const [32][32]f32,
    bias: *const [32]f32,
) !void {
    const denorm = try allocator.alloc(f32, latents.len);
    defer allocator.free(denorm);
    for (0..latents.len / 128) |p| {
        for (0..128) |c| {
            denorm[p * 128 + c] = latents[p * 128 + c] * stds[c] + means[c];
        }
    }
    const hw = (gh * 2) * (gw * 2);
    const unpacked = try allocator.alloc(f32, 32 * hw);
    defer allocator.free(unpacked);
    try zflux2_latent.unpack(unpacked, denorm, .{ .height = gh * 2, .width = gw * 2 });
    for (0..hw) |p| {
        for (0..32) |o| {
            var acc: f32 = bias[o];
            for (0..32) |i| acc += wmat[o][i] * unpacked[i * hw + p];
            out[o * hw + p] = acc;
        }
    }
}

test "folded prepare matches the three-pass reference" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xf01d);
    const rand = prng.random();
    const gh = 3;
    const gw = 5;
    var stds: [128]f32 = undefined;
    var means: [128]f32 = undefined;
    for (&stds) |*v| v.* = 0.5 + rand.float(f32);
    for (&means) |*v| v.* = rand.floatNorm(f32) * 0.2;
    var wmat: [32][32]f32 = undefined;
    var bias: [32]f32 = undefined;
    for (&wmat) |*row| for (row) |*v| {
        v.* = rand.floatNorm(f32) * 0.3;
    };
    for (&bias) |*v| v.* = rand.floatNorm(f32) * 0.1;
    const latents = try alloc.alloc(f32, gh * gw * 128);
    defer alloc.free(latents);
    for (latents) |*v| v.* = rand.floatNorm(f32);

    const hw = (gh * 2) * (gw * 2);
    const want = try alloc.alloc(f32, 32 * hw);
    defer alloc.free(want);
    try threePass(alloc, want, latents, gh, gw, &stds, &means, &wmat, &bias);
    const got = try alloc.alloc(f32, 32 * hw);
    defer alloc.free(got);
    try foldApply(got, latents, gh, gw, &stds, &means, &wmat, &bias);
    for (want, got) |w, g| {
        try std.testing.expectApproxEqAbs(w, g, 1e-4);
    }
}
