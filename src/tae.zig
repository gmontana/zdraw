//! TAEF1 tiny VAE decoder (preview tier).
//!
//! madebyollin/taef1 (MIT): conv 16->64, three stages of three residual
//! blocks + nearest-2x upsample + conv, then a final block and conv 64->3.
//! Convs run on the GPU via mconv; pointwise glue stays on the CPU (the
//! whole net is ~10 MB and the conv work dominates).

const std = @import("std");

const mconv = @import("mconv.zig");
const tensor_file = @import("tensor_file.zig");

pub const Loaded = struct {
    file: tensor_file.Mapped,

    pub fn deinit(self: *Loaded, io: std.Io) void {
        self.file.deinit(io);
        self.* = undefined;
    }
};

fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Loaded {
    return .{ .file = try tensor_file.open(io, allocator, path) };
}

const vdecode = @import("vdecode.zig");

var cached: ?Loaded = null;

/// Preview decode tier: ZDRAW_VAE=tae + ZDRAW_TAE (or default discovery).
/// Returns false when disabled or weights are missing (caller falls back).
pub fn decodeIfEnabled(
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: ?*mconv.Context,
    out: []f32,
    latents: []const f32,
    cfg: struct { height: usize, width: usize },
) !bool {
    const raw = std.c.getenv("ZDRAW_VAE") orelse return false;
    if (!std.mem.eql(u8, std.mem.span(raw), "tae")) return false;
    const metal = ctx orelse return false;
    if (cached == null) {
        // No default: the preview tier needs an explicit ZDRAW_TAE path.
        const tae_env = std.c.getenv("ZDRAW_TAE") orelse return false;
        const path = std.mem.span(tae_env);
        cached = open(io, allocator, path) catch return false;
    }
    const norm = try allocator.alloc(f32, latents.len);
    defer allocator.free(norm);
    vdecode.denorm(norm, latents);
    try decode(allocator, metal, &cached.?, out, norm, cfg.height, cfg.width);
    return true;
}

const block_ids = [_]usize{ 2, 3, 4, 7, 8, 9, 12, 13, 14, 17 };
const up_conv_ids = [_]usize{ 6, 11, 16 };

/// Decode latents [16, h, w] (sampler-native scale) into rgb [3, 8h, 8w].
fn decode(
    allocator: std.mem.Allocator,
    ctx: *mconv.Context,
    tae: *const Loaded,
    out: []f32,
    latents: []const f32,
    h: usize,
    w: usize,
) !void {
    if (latents.len != 16 * h * w) return error.InvalidShape;
    if (out.len != 3 * 64 * h * w) return error.InvalidShape;
    var x = try allocator.alloc(f32, 64 * h * w);
    defer allocator.free(x);
    var y = try allocator.alloc(f32, 64 * h * w);
    defer allocator.free(y);
    // Input clamp: tanh(z/3)*3, then conv 16->64 + relu.
    const clamped = try allocator.alloc(f32, latents.len);
    defer allocator.free(clamped);
    for (clamped, latents) |*dst, v| dst.* = std.math.tanh(v / 3.0) * 3.0;
    try convLayer(ctx, tae, 0, x[0 .. 64 * h * w], clamped, 16, h, w, true);
    relu(x[0 .. 64 * h * w]);

    const final = try stages(allocator, ctx, tae, &x, &y, h, w);
    try residualBlock(allocator, ctx, tae, block_ids[9], x, final.h, final.w);
    try convLayerOut(ctx, tae, out, x[0 .. 64 * final.h * final.w], final.h, final.w);
    // TAE emits [0,1]; the pixel pipeline expects the full VAE's [-1,1].
    for (out) |*v| v.* = v.* * 2.0 - 1.0;
}

fn stages(
    allocator: std.mem.Allocator,
    ctx: *mconv.Context,
    tae: *const Loaded,
    cur: *[]f32,
    alt: *[]f32,
    h0: usize,
    w0: usize,
) !struct { h: usize, w: usize } {
    var ch = h0;
    var cw = w0;
    var blocks: usize = 0;
    for (0..3) |stage_i| {
        for (0..3) |_| {
            try residualBlock(allocator, ctx, tae, block_ids[blocks], cur.*, ch, cw);
            blocks += 1;
        }
        const big = try allocator.alloc(f32, 64 * ch * 2 * cw * 2);
        defer allocator.free(big);
        upsample2(big, cur.*[0 .. 64 * ch * cw], ch, cw);
        ch *= 2;
        cw *= 2;
        if (alt.*.len < 64 * ch * cw) {
            allocator.free(alt.*);
            alt.* = try allocator.alloc(f32, 64 * ch * cw);
        }
        const id = up_conv_ids[stage_i];
        try convLayer(ctx, tae, id, alt.*[0 .. 64 * ch * cw], big, 64, ch, cw, false);
        std.mem.swap([]f32, cur, alt);
        if (cur.*.len < 64 * ch * cw) return error.InvalidShape;
    }
    return .{ .h = ch, .w = cw };
}

fn residualBlock(
    allocator: std.mem.Allocator,
    ctx: *mconv.Context,
    tae: *const Loaded,
    id: usize,
    state: []f32,
    h: usize,
    w: usize,
) !void {
    const n = 64 * h * w;
    const t1 = try allocator.alloc(f32, n);
    defer allocator.free(t1);
    const t2 = try allocator.alloc(f32, n);
    defer allocator.free(t2);
    try convSub(ctx, tae, id, 0, t1, state[0..n], h, w);
    relu(t1);
    try convSub(ctx, tae, id, 2, t2, t1, h, w);
    relu(t2);
    try convSub(ctx, tae, id, 4, t1, t2, h, w);
    for (state[0..n], t1) |*s, v| s.* = @max(0.0, s.* + v);
}

fn convLayer(
    ctx: *mconv.Context,
    tae: *const Loaded,
    id: usize,
    out: []f32,
    input: []const f32,
    in_ch: usize,
    h: usize,
    w: usize,
    with_bias: bool,
) !void {
    var buf: [64]u8 = undefined;
    const wname = try std.fmt.bufPrint(&buf, "decoder.layers.{d}.weight", .{id});
    const weight = (try tae.file.view(wname)) orelse return error.MissingTensor;
    var bbuf: [64]u8 = undefined;
    const bname = try std.fmt.bufPrint(&bbuf, "decoder.layers.{d}.bias", .{id});
    const bias = if (with_bias) (try tae.file.view(bname)) else null;
    try ctx.run(out, input, weight, bias, .{
        .in_ch = in_ch,
        .out_ch = 64,
        .height = h,
        .width = w,
        .kernel = 3,
        .pad = 1,
    });
}

fn convLayerOut(
    ctx: *mconv.Context,
    tae: *const Loaded,
    out: []f32,
    input: []const f32,
    h: usize,
    w: usize,
) !void {
    const weight = (try tae.file.view("decoder.layers.18.weight")) orelse
        return error.MissingTensor;
    const bias = (try tae.file.view("decoder.layers.18.bias")) orelse
        return error.MissingTensor;
    try ctx.run(out, input, weight, bias, .{
        .in_ch = 64,
        .out_ch = 3,
        .height = h,
        .width = w,
        .kernel = 3,
        .pad = 1,
    });
}

fn convSub(
    ctx: *mconv.Context,
    tae: *const Loaded,
    id: usize,
    sub: usize,
    out: []f32,
    input: []const f32,
    h: usize,
    w: usize,
) !void {
    var buf: [64]u8 = undefined;
    const wname = try std.fmt.bufPrint(&buf, "decoder.layers.{d}.conv.{d}.weight", .{ id, sub });
    const weight = (try tae.file.view(wname)) orelse return error.MissingTensor;
    var bbuf: [64]u8 = undefined;
    const bname = try std.fmt.bufPrint(&bbuf, "decoder.layers.{d}.conv.{d}.bias", .{ id, sub });
    const bias = (try tae.file.view(bname)) orelse return error.MissingTensor;
    try ctx.run(out, input, weight, bias, .{
        .in_ch = 64,
        .out_ch = 64,
        .height = h,
        .width = w,
        .kernel = 3,
        .pad = 1,
    });
}

fn relu(values: []f32) void {
    for (values) |*v| v.* = @max(0.0, v.*);
}

fn upsample2(out: []f32, input: []const f32, h: usize, w: usize) void {
    for (0..64) |ch| {
        const src = input[ch * h * w ..][0 .. h * w];
        const dst = out[ch * h * 2 * w * 2 ..][0 .. h * 2 * w * 2];
        for (0..h * 2) |yy| {
            for (0..w * 2) |xx| {
                dst[yy * w * 2 + xx] = src[(yy / 2) * w + xx / 2];
            }
        }
    }
}
