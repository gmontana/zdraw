//! Parameter construction for resident VAE residual blocks.

const std = @import("std");

const c = @import("metal_c.zig");
const conv = @import("../vae/conv.zig");
const mbuffer = @import("mbuffer.zig");
const mconv = @import("mconv.zig");
const mres_util = @import("mres_util.zig");
const bufs = @import("mvres_buf.zig");
const tensor = @import("../pack/tensor.zig");
const vres = @import("../vae/vres.zig");

pub const NormParams = extern struct {
    channels: u32,
    height: u32,
    width: u32,
    groups: u32,
    dtype: u32,
    bias_dtype: u32,
    eps: f32,
    weight_offset: u64,
    bias_offset: u64,
};

pub const ResParams = extern struct {
    norm1: NormParams,
    conv1: c.ConvParams,
    norm2: NormParams,
    conv2: c.ConvParams,
    skip: c.ConvParams,
    has_skip: u32,
    out_count: u32,
};

pub fn check(out: []const f32, input: []const f32, views: vres.Views, cfg: vres.Config) !void {
    if (out.len != cfg.out_ch * cfg.height * cfg.width) return error.InvalidShape;
    if (input.len != cfg.in_ch * cfg.height * cfg.width) return error.InvalidShape;
    try checkViews(views, cfg);
}

// Weight-views-vs-config validation alone, for callers whose buffer lengths
// are derived from cfg by construction (no host slices to compare).
pub fn checkViews(views: vres.Views, cfg: vres.Config) !void {
    if (views.skip_w == null and cfg.in_ch != cfg.out_ch) return error.InvalidShape;
    try checkNorm(views.norm1_w, views.norm1_b, cfg.in_ch);
    try checkNorm(views.norm2_w, views.norm2_b, cfg.out_ch);
    try mconv.checkWeight(views.conv1_w, views.conv1_b, conv3(cfg.in_ch, cfg.out_ch, cfg));
    try mconv.checkWeight(views.conv2_w, views.conv2_b, conv3(cfg.out_ch, cfg.out_ch, cfg));
    if (views.skip_w) |weight| {
        try mconv.checkWeight(weight, views.skip_b, conv1(cfg.in_ch, cfg.out_ch, cfg));
    }
}

pub fn make(views: vres.Views, binds: bufs.Binds, cfg: vres.Config) !ResParams {
    const conv1_cfg = conv3(cfg.in_ch, cfg.out_ch, cfg);
    return .{
        .norm1 = try normParams(
            views.norm1_w,
            views.norm1_b,
            binds.norm1_w,
            binds.norm1_b,
            cfg,
            cfg.in_ch,
        ),
        .conv1 = try mconv.paramsFor(
            views.conv1_w,
            views.conv1_b,
            conv1_cfg,
            binds.conv1_w,
            binds.conv1_b,
        ),
        .norm2 = try normParams(
            views.norm2_w,
            views.norm2_b,
            binds.norm2_w,
            binds.norm2_b,
            cfg,
            cfg.out_ch,
        ),
        .conv2 = try mconv.paramsFor(
            views.conv2_w,
            views.conv2_b,
            conv3(cfg.out_ch, cfg.out_ch, cfg),
            binds.conv2_w,
            binds.conv2_b,
        ),
        .skip = try skipParams(views, binds, conv1(cfg.in_ch, cfg.out_ch, cfg)),
        .has_skip = if (views.skip_w == null) 0 else 1,
        .out_count = try mres_util.toU32(cfg.out_ch * cfg.height * cfg.width),
    };
}

fn skipParams(views: vres.Views, binds: bufs.Binds, cfg: conv.Config) !c.ConvParams {
    if (views.skip_w) |weight| {
        return mconv.paramsFor(weight, views.skip_b, cfg, binds.skip_w, binds.skip_b);
    }
    return std.mem.zeroes(c.ConvParams);
}

fn checkNorm(weight: tensor.View, bias: tensor.View, channels: usize) !void {
    if (try weight.elems() != channels or try bias.elems() != channels) {
        return error.InvalidShape;
    }
    try weight.check();
    try bias.check();
}

fn normParams(
    weight: tensor.View,
    bias: tensor.View,
    wb: mbuffer.Bind,
    bb: mbuffer.Bind,
    cfg: vres.Config,
    channels: usize,
) !NormParams {
    return .{
        .channels = try mres_util.toU32(channels),
        .height = try mres_util.toU32(cfg.height),
        .width = try mres_util.toU32(cfg.width),
        .groups = try mres_util.toU32(cfg.groups),
        .dtype = try mres_util.dtype(weight.dtype),
        .bias_dtype = try mres_util.dtype(bias.dtype),
        .eps = cfg.eps,
        .weight_offset = wb.offset,
        .bias_offset = bb.offset,
    };
}

fn conv3(in_ch: usize, out_ch: usize, cfg: vres.Config) conv.Config {
    return .{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = cfg.height,
        .width = cfg.width,
        .kernel = 3,
        .pad = 1,
    };
}

fn conv1(in_ch: usize, out_ch: usize, cfg: vres.Config) conv.Config {
    return .{
        .in_ch = in_ch,
        .out_ch = out_ch,
        .height = cfg.height,
        .width = cfg.width,
        .kernel = 1,
        .pad = 0,
    };
}
