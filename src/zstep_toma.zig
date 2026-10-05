//! Banded ToMA execution: pre-band at full tokens, merged interior band,
//! full-token tail fused with the final projection. Any band failure
//! returns false so the caller can fall back.

const std = @import("std");

const env = @import("env.zig");
const mattn = @import("mattn.zig");
const mchain = @import("mstack_chain.zig");
const mlinear = @import("mlinear.zig");
const mfallback = @import("metal_fallback.zig");
const mstack = @import("mstack_final.zig");
const mtoma = @import("mstack_toma.zig");
const toma = @import("toma.zig");
const toma_config = @import("toma_config.zig");
const toma_control = @import("toma_control.zig");
const zblock = @import("zblock.zig");
const zrope = @import("zrope.zig");
const zs = @import("zstep_shape.zig");

pub const Args = struct {
    projected: []f32,
    state: []f32,
    pos: []const zrope.Pos,
    views: []const zblock.Views,
    adaln: []const f32,
    cfg: mstack.Config,
    rope: zrope.Cache,
    final: mstack.Final,
    dims: zs.Dims,
};

pub fn run(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    args: Args,
) !bool {
    if (toma_control.plan()) |plan| {
        if (plan.source_tokens != args.dims.img_total) {
            return error.PlanTokenCountMismatch;
        }
        if (plan.layer_to > args.views.len) return error.PlanLayerRangeMismatch;
    }
    const band = toma.bandFor(args.views.len);
    const pre = args.views[0..band.from];
    if (!try runBand(
        allocator,
        metal,
        attn,
        args,
        args.state,
        pre,
        args.pos,
        args.cfg,
        0,
    )) {
        return false;
    }
    const mid = args.views[band.from..band.to];
    if (!try runTomaBand(allocator, metal, attn, args, mid, band.from)) {
        return false;
    }
    if (!try runTail(allocator, metal, attn, args, band.to)) return false;
    try toma_control.recordCompleted();
    return true;
}

fn runBand(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    args: Args,
    state: []f32,
    views: []const zblock.Views,
    pos: []const zrope.Pos,
    cfg: mstack.Config,
    first: usize,
) !bool {
    mchain.run(
        allocator,
        metal,
        attn,
        state,
        views,
        args.adaln,
        pos,
        cfg,
        args.rope,
        .main,
        .{ .first = first, .total = args.views.len },
    ) catch |err| {
        if (toma_control.plan() != null) return err;
        return false;
    };
    return true;
}

fn runTomaBand(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    args: Args,
    views: []const zblock.Views,
    first: usize,
) !bool {
    mtoma.run(
        allocator,
        metal,
        attn,
        args.state,
        views,
        args.adaln,
        args.pos,
        args.cfg,
        args.rope,
        .main,
        .{ .first = first, .total = args.views.len },
        .{
            .image_tokens = args.dims.img_total,
            .caption_tokens = args.dims.cap_total,
        },
        config(args.dims),
    ) catch |err| {
        if (toma_control.plan() != null) return err;
        return false;
    };
    return true;
}

fn config(dims: zs.Dims) toma_config.Config {
    if (toma_control.plan()) |plan| return plan.config;
    const destinations =
        env.usizeVar("ZDRAW_TOMA_DESTINATIONS", dims.img_total / 2);
    const regions = env.usizeVar("ZDRAW_TOMA_REGIONS", 64);
    const scale = scaleFromEnv();
    const route = std.c.getenv("ZDRAW_TOMA_ROUTE");
    if (route) |raw| {
        const value = std.mem.span(raw);
        if (std.mem.eql(u8, value, "paper-local")) return .{
            .mode = .paper_spec,
            .destination_tokens = destinations,
            .region_count = regions,
            .selection_layout = .tile,
            .assignment_scope = .region_local,
            .unmerge = .paper_normalized_transpose,
            .assignment_scale = scale,
        };
        if (std.mem.eql(u8, value, "official")) return .{
            .mode = .official_b578009,
            .destination_tokens = destinations,
            .region_count = regions,
            .selection_layout = .tile,
            .assignment_scope = .region_local,
            .unmerge = .official_raw_transpose,
            .assignment_scale = scale,
        };
    }
    return .{
        .mode = .paper_spec,
        .destination_tokens = destinations,
        .region_count = regions,
        .selection_layout = .tile,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = scale,
    };
}

fn scaleFromEnv() f32 {
    const raw = std.c.getenv("ZDRAW_TOMA_SCALE") orelse return 1000;
    return std.fmt.parseFloat(f32, std.mem.span(raw)) catch 1000;
}

fn runTail(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    args: Args,
    from: usize,
) !bool {
    mstack.run(
        allocator,
        metal,
        attn,
        args.projected,
        args.state,
        args.views[from..],
        args.adaln,
        args.pos,
        args.cfg,
        args.rope,
        args.final,
        .{ .first = from, .total = args.views.len },
    ) catch |err| {
        if (mfallback.isGemmRefusal(err) and toma_control.plan() == null) return false;
        return err;
    };
    return true;
}
