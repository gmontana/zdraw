//! VAE encoder: image pixels -> latent distribution parameters.
//!
//! The mirror of vdecode: conv_in, four down blocks (two resnets each, the
//! first three followed by a stride-2 conv), the mid block (resnet, self
//! attention, resnet), GroupNorm + SiLU, conv_out, and quant_conv. Output is
//! the 64-channel mean/logvar pair at 1/8 resolution; editing uses the mean
//! half (`mean` below), matching the deterministic reference behaviour.
//!
//! CPU reference tier: correctness first, using the same vres/vattn/gnorm/conv
//! primitives the decoder runs on. The only primitive the decoder never needed
//! is the stride-2 downsample, which lives here.
//!
//! NOTE: nothing imports this module yet and `mean` has never been executed,
//! so treat it as unverified until it is gated against an oracle.

const std = @import("std");

const conv = @import("conv.zig");
const conv_fast = @import("conv_fast.zig");
const gnorm = @import("gnorm.zig");
const metrics = @import("../metal/metrics.zig");
const mattn = @import("../metal/mattn.zig");
const mconv = @import("../metal/mconv.zig");
const mlinear = @import("../metal/mlinear.zig");
const mvattn = @import("../metal/mvattn.zig");
const ops = @import("../runtime/ops.zig");
const tensor = @import("../pack/tensor.zig");
const vattn = @import("vattn.zig");
const vres = @import("vres.zig");
const vviews = @import("vviews.zig");

/// Channel widths per stage, matching AutoencoderKL's published config.
const block_ch = [4]usize{ 128, 256, 512, 512 };
const latent_ch: usize = 64; // 32 mean + 32 logvar, before the split

pub const Config = struct {
    height: usize, // pixel rows; must be a multiple of 8
    width: usize,
    groups: usize = 32,
    eps: f32 = 0.000001,
};

/// Encoded latent mean, [32, height/8, width/8], caller frees.
pub fn mean(
    allocator: std.mem.Allocator,
    linear: ?*mlinear.Context,
    metal_conv: ?*mconv.Context,
    metal_attn: ?*mattn.Context,
    rgb: []const f32, // [3, height, width], already in [-1, 1]
    views: vviews.EncoderViews,
    cfg: Config,
) ![]f32 {
    // 16, not 8: the 8x downsample must leave an EVEN latent grid for the
    // 2x2 patchify that follows (zflux2_latent.pack).
    if (cfg.height % 16 != 0 or cfg.width % 16 != 0) return error.InvalidImageSize;
    if (rgb.len != 3 * cfg.height * cfg.width) return error.InvalidShape;

    var state = try State.init(allocator, cfg);
    defer state.deinit(allocator);

    // conv_in: 3 -> 128 at full resolution.
    const first = state.cur[0 .. block_ch[0] * state.pixels()];
    try conv_fast.run(metal_conv, first, rgb, views.conv_in_w, views.conv_in_b, .{
        .in_ch = 3,
        .out_ch = block_ch[0],
        .height = state.h,
        .width = state.w,
        .kernel = 3,
        .pad = 1,
    });

    metrics.memtrace("ref-in");
    const down_names = [_][]const u8{ "ref-down-0", "ref-down-1", "ref-down-2", "ref-down-3" };
    for (views.down, 0..) |down, i| {
        try state.downBlock(allocator, metal_conv, down, i, cfg);
        metrics.memtrace(down_names[@min(i, down_names.len - 1)]);
    }

    try state.mid(allocator, linear, metal_conv, metal_attn, views, cfg);
    metrics.memtrace("ref-mid");
    const out = try state.finish(allocator, metal_conv, views, cfg);
    metrics.memtrace("ref-out");
    return out;
}

/// Working buffers for one encode. `cur` holds the live feature map; every
/// stage writes into `alt` and swaps, so no stage aliases its own input.
const State = struct {
    cur: []f32,
    alt: []f32,
    h: usize,
    w: usize,
    ch: usize,

    fn pixels(self: *const State) usize {
        return self.h * self.w;
    }

    fn init(allocator: std.mem.Allocator, cfg: Config) !State {
        // The widest stage is conv_in's output at full resolution; every later
        // stage halves area at least as fast as it doubles channels.
        const cap = block_ch[0] * cfg.height * cfg.width;
        const cur = try allocator.alloc(f32, cap);
        errdefer allocator.free(cur);
        const alt = try allocator.alloc(f32, cap);
        return .{ .cur = cur, .alt = alt, .h = cfg.height, .w = cfg.width, .ch = block_ch[0] };
    }

    fn deinit(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.alt);
        allocator.free(self.cur);
        self.* = undefined;
    }

    fn swap(self: *State) void {
        const tmp = self.cur;
        self.cur = self.alt;
        self.alt = tmp;
    }

    fn downBlock(
        self: *State,
        allocator: std.mem.Allocator,
        metal_conv: ?*mconv.Context,
        down: vviews.Down,
        index: usize,
        cfg: Config,
    ) !void {
        const out_ch = block_ch[index];
        for (down.res) |res| {
            const scratch = try resScratch(allocator, self.ch, out_ch, self.pixels());
            defer freeRes(allocator, scratch);
            try vres.run(
                metal_conv,
                self.alt[0 .. out_ch * self.pixels()],
                self.cur[0 .. self.ch * self.pixels()],
                res,
                scratch,
                .{
                    .in_ch = self.ch,
                    .out_ch = out_ch,
                    .height = self.h,
                    .width = self.w,
                    .groups = cfg.groups,
                    .eps = cfg.eps,
                },
            );
            self.ch = out_ch;
            self.swap();
        }
        const w = down.down_w orelse return;
        const b = down.down_b orelse return error.MissingTensor;
        const nh = self.h / 2;
        const nw = self.w / 2;
        try downsample(
            allocator,
            metal_conv,
            self.alt[0 .. self.ch * nh * nw],
            self.cur[0 .. self.ch * self.pixels()],
            w,
            b,
            self.ch,
            self.h,
            self.w,
        );
        self.h = nh;
        self.w = nw;
        self.swap();
    }

    fn mid(
        self: *State,
        allocator: std.mem.Allocator,
        linear: ?*mlinear.Context,
        metal_conv: ?*mconv.Context,
        metal_attn: ?*mattn.Context,
        views: vviews.EncoderViews,
        cfg: Config,
    ) !void {
        const n = self.ch * self.pixels();
        const res_cfg = vres.Config{
            .in_ch = self.ch,
            .out_ch = self.ch,
            .height = self.h,
            .width = self.w,
            .groups = cfg.groups,
            .eps = cfg.eps,
        };
        for ([_]vres.Views{views.mid0}) |res| {
            const scratch = try resScratch(allocator, self.ch, self.ch, self.pixels());
            defer freeRes(allocator, scratch);
            try vres.run(metal_conv, self.alt[0..n], self.cur[0..n], res, scratch, res_cfg);
            self.swap();
        }

        const acfg = vattn.Config{
            .channels = self.ch,
            .height = self.h,
            .width = self.w,
            .head_dim = self.ch,
            .groups = cfg.groups,
            .eps = cfg.eps,
        };
        // The decoder's resident GPU attention (mvattn, in place on `cur`)
        // when every context exists and the shape fits; else the pooled or
        // CPU path through vattn (16k tokens at 1024 would take minutes).
        if (metal_conv != null and linear != null and metal_attn != null and
            mvattn.fits(self.ch, self.pixels()))
        {
            try self.residentAttn(allocator, metal_conv.?, linear.?, metal_attn.?, views, acfg);
        } else {
            const attn_scratch = try attnScratch(allocator, self.ch, self.pixels());
            defer freeAttn(allocator, attn_scratch);
            const at = attn_scratch;
            try vattn.run(linear, metal_attn, self.alt[0..n], self.cur[0..n], views.attn, at, acfg);
            self.swap();
        }

        const scratch = try resScratch(allocator, self.ch, self.ch, self.pixels());
        defer freeRes(allocator, scratch);
        try vres.run(metal_conv, self.alt[0..n], self.cur[0..n], views.mid1, scratch, res_cfg);
        self.swap();
    }

    /// mvattn.run on the live feature map: the result replaces `cur`.
    fn residentAttn(
        self: *State,
        allocator: std.mem.Allocator,
        conv_ctx: *mconv.Context,
        lin: *mlinear.Context,
        att: *mattn.Context,
        views: vviews.EncoderViews,
        acfg: vattn.Config,
    ) !void {
        const n = self.ch * self.pixels();
        const q = try allocator.alloc(f32, n);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, n);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, n);
        defer allocator.free(v);
        const mix = try allocator.alloc(f32, n);
        defer allocator.free(mix);
        try mvattn.run(conv_ctx, lin, att, null, self.cur[0..n], views.attn, .{
            .q = q,
            .k = k,
            .v = v,
            .mix = mix,
        }, acfg);
    }

    /// GroupNorm + SiLU + conv_out + quant_conv, returning the mean half.
    fn finish(
        self: *State,
        allocator: std.mem.Allocator,
        metal_conv: ?*mconv.Context,
        views: vviews.EncoderViews,
        cfg: Config,
    ) ![]f32 {
        const n = self.ch * self.pixels();
        try gnorm.run(self.alt[0..n], self.cur[0..n], views.norm_w, views.norm_b, .{
            .channels = self.ch,
            .height = self.h,
            .width = self.w,
            .groups = cfg.groups,
            .eps = cfg.eps,
        });
        for (self.alt[0..n]) |*v| v.* = ops.silu(v.*);

        const wide = try allocator.alloc(f32, latent_ch * self.pixels());
        defer allocator.free(wide);
        try conv_fast.run(metal_conv, wide, self.alt[0..n], views.out_w, views.out_b, .{
            .in_ch = self.ch,
            .out_ch = latent_ch,
            .height = self.h,
            .width = self.w,
            .kernel = 3,
            .pad = 1,
        });

        const quantized = try allocator.alloc(f32, latent_ch * self.pixels());
        defer allocator.free(quantized);
        try conv.run(quantized, wide, views.quant_w, views.quant_b, .{
            .in_ch = latent_ch,
            .out_ch = latent_ch,
            .height = self.h,
            .width = self.w,
            .kernel = 1,
            .pad = 0,
        });

        // Deterministic encode: keep the mean, drop the logvar half.
        const half = latent_ch / 2;
        const out = try allocator.alloc(f32, half * self.pixels());
        @memcpy(out, quantized[0 .. half * self.pixels()]);
        return out;
    }
};

/// Stride-2, 3x3 convolution with the reference's asymmetric (0,1,0,1)
/// padding: taps past the last row/column read as zero, which is what
/// `F.pad(x, (0,1,0,1))` followed by an unpadded stride-2 conv computes.
/// The stride-2 downsample (diffusers: pad (0,1,0,1), then a 3x3 stride-2
/// conv with no padding). With a Metal context it is the decoder's 3x3
/// pad-1 stride-1 conv followed by sampling the odd positions: window rows
/// 2r..2r+2 of the padded input are exactly the pad-1 window around row
/// 2r+1, and the zero row/column past the edge matches the symmetric pad.
/// Four times the MACs of a strided kernel, on the GPU; the CPU loop below
/// is the reference tier.
fn downsample(
    allocator: std.mem.Allocator,
    metal_conv: ?*mconv.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    ch: usize,
    height: usize,
    width: usize,
) !void {
    const oh = height / 2;
    const ow = width / 2;
    if (out.len != ch * oh * ow or input.len != ch * height * width) return error.InvalidShape;
    try weight.check();
    if (weight.shape.len != 4 or weight.shape[0] != ch or weight.shape[1] != ch) {
        return error.InvalidShape;
    }
    if (metal_conv) |ctx| {
        return downsampleMetal(allocator, ctx, out, input, weight, bias, ch, height, width);
    }
    for (0..ch) |oc| {
        for (0..oh) |r| {
            for (0..ow) |c| {
                var sum = bias.atF32Unchecked(oc);
                for (0..ch) |ic| {
                    const plane = ic * height * width;
                    const wbase = ((oc * ch) + ic) * 9;
                    for (0..3) |kr| {
                        const sr = r * 2 + kr;
                        if (sr >= height) continue;
                        for (0..3) |kc| {
                            const sc = c * 2 + kc;
                            if (sc >= width) continue;
                            sum += input[plane + sr * width + sc] *
                                weight.atF32Unchecked(wbase + kr * 3 + kc);
                        }
                    }
                }
                out[(oc * oh + r) * ow + c] = sum;
            }
        }
    }
}

fn downsampleMetal(
    allocator: std.mem.Allocator,
    ctx: *mconv.Context,
    out: []f32,
    input: []const f32,
    weight: tensor.View,
    bias: tensor.View,
    ch: usize,
    height: usize,
    width: usize,
) !void {
    const oh = height / 2;
    const ow = width / 2;
    const full = try allocator.alloc(f32, ch * height * width);
    defer allocator.free(full);
    try conv_fast.run(ctx, full, input, weight, bias, .{
        .in_ch = ch,
        .out_ch = ch,
        .height = height,
        .width = width,
        .kernel = 3,
        .pad = 1,
    });
    for (0..ch) |oc| {
        for (0..oh) |r| {
            const src = (oc * height + 2 * r + 1) * width;
            const dst = (oc * oh + r) * ow;
            for (0..ow) |c| out[dst + c] = full[src + 2 * c + 1];
        }
    }
}

fn resScratch(
    allocator: std.mem.Allocator,
    in_ch: usize,
    out_ch: usize,
    pixels: usize,
) !vres.Scratch {
    return .{
        .norm = try allocator.alloc(f32, in_ch * pixels),
        .work = try allocator.alloc(f32, out_ch * pixels),
        .skip = try allocator.alloc(f32, out_ch * pixels),
    };
}

fn freeRes(allocator: std.mem.Allocator, s: vres.Scratch) void {
    allocator.free(s.skip);
    allocator.free(s.work);
    allocator.free(s.norm);
}

fn attnScratch(allocator: std.mem.Allocator, ch: usize, pixels: usize) !vattn.Scratch {
    return .{
        .norm = try allocator.alloc(f32, ch * pixels),
        .q = try allocator.alloc(f32, ch * pixels),
        .k = try allocator.alloc(f32, ch * pixels),
        .v = try allocator.alloc(f32, ch * pixels),
        .mix = try allocator.alloc(f32, ch * pixels),
        // One score row per token, per vattn.check; sized per token, not
        // per token-pair.
        .scores = try allocator.alloc(f32, pixels),
    };
}

fn freeAttn(allocator: std.mem.Allocator, s: vattn.Scratch) void {
    allocator.free(s.scores);
    allocator.free(s.mix);
    allocator.free(s.v);
    allocator.free(s.k);
    allocator.free(s.q);
    allocator.free(s.norm);
}

test "downsample halves each axis and honours the trailing-edge padding" {
    const alloc = std.testing.allocator;
    // 1 channel, 4x4 input of ones; identity-ish kernel picking one tap.
    const input = [_]f32{1} ** 16;
    var wdata = [_]f32{0} ** 9;
    wdata[0] = 1; // top-left tap only
    const bias = [_]f32{0};
    const out = try alloc.alloc(f32, 4);
    defer alloc.free(out);
    try downsample(
        alloc,
        null,
        out,
        &input,
        .{ .dtype = .f32, .shape = &.{ 1, 1, 3, 3 }, .bytes = std.mem.sliceAsBytes(&wdata) },
        .{ .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&bias) },
        1,
        4,
        4,
    );
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);

    // Bottom-right tap: the last output row/col read past the edge and see the
    // zero pad, so only the first row/col of outputs are fully covered.
    var w2 = [_]f32{0} ** 9;
    w2[8] = 1;
    try downsample(
        alloc,
        null,
        out,
        &input,
        .{ .dtype = .f32, .shape = &.{ 1, 1, 3, 3 }, .bytes = std.mem.sliceAsBytes(&w2) },
        .{ .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&bias) },
        1,
        4,
        4,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[3], 1e-6);
}
