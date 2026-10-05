//! Resident fast path for one Z-Image denoise step.

const std = @import("std");

const env = @import("env.zig");
const hook_bypass = @import("hook_bypass_control.zig");
const linear = @import("linear_fast.zig");
const mattn = @import("mattn.zig");
const metrics = @import("metrics.zig");
const mlinear = @import("mlinear.zig");
const mfallback = @import("metal_fallback.zig");
const mselect = @import("mstack_select.zig");
const zblock = @import("zblock.zig");
const mstack = @import("mstack_final.zig");
const token_selection = @import("token_selection_control.zig");
const toma = @import("toma.zig");
const ztoma = @import("zstep_toma.zig");
const ops = @import("ops.zig");
const zconfig = @import("zimage_config.zig");
const zlayer = @import("zlayer.zig");
const zpatch = @import("zpatch.zig");
const zrope = @import("zrope.zig");
const zseq = @import("zseq.zig");
const zs = @import("zstep_shape.zig");
const zstreams = @import("zstreams.zig");
const ztx = @import("ztx.zig");
const zunify = @import("zunify.zig");

pub const Request = struct {
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    out: []f32,
    input: zs.Input,
    tx: *const ztx.Loaded,
    cfg: zconfig.Transformer,
    rope: zrope.Cache,
};

pub fn run(
    cache_allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    req: Request,
    streams: zstreams.Streams,
    adaln: []const f32,
    dims: zs.Dims,
) !bool {
    const metal = req.metal orelse return false;
    const attn = req.attn orelse return false;
    const unified = try timedUnified(temp_allocator, streams, dims);
    const scale = try timedScale(metal, temp_allocator, adaln, req.tx);
    const projected = try temp_allocator.alloc(f32, dims.total * dims.patch_dim);
    const ok = try timedStack(
        cache_allocator,
        metal,
        attn,
        projected,
        unified,
        req,
        adaln,
        scale,
        dims,
    );
    if (!ok) return false;
    const patch_start = metrics.now();
    const patch_metal = metrics.snapshot();
    try zpatch.unpatchify(req.out, projected[0 .. dims.img_raw * dims.patch_dim], req.input.shape);
    record("resident-unpatch", patch_start, patch_metal);
    return true;
}

fn limitLayers(views: []const zblock.Views) []const zblock.Views {
    const n = env.usizeVar("ZDRAW_STACK_LAYER_LIMIT", views.len);
    return views[0..@min(n, views.len)];
}

fn timedUnified(
    allocator: std.mem.Allocator,
    streams: zstreams.Streams,
    dims: zs.Dims,
) !Unified {
    const start = metrics.now();
    const metal = metrics.snapshot();
    const unified = try makeUnified(allocator, streams, dims);
    record("resident-unify", start, metal);
    return unified;
}

fn timedScale(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    adaln: []const f32,
    tx: *const ztx.Loaded,
) ![]f32 {
    const start = metrics.now();
    const snap = metrics.snapshot();
    const scale = try makeScale(metal, allocator, adaln, tx);
    record("resident-final-scale", start, snap);
    return scale;
}

fn timedStack(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    projected: []f32,
    unified: Unified,
    req: Request,
    adaln: []const f32,
    scale: []const f32,
    dims: zs.Dims,
) !bool {
    const start = metrics.now();
    const snap = metrics.snapshot();
    const ok = try runStack(allocator, metal, attn, projected, unified, req, adaln, scale, dims);
    record("resident-stack-final", start, snap);
    return ok;
}

const TomaArgs = struct {
    p: []f32,
    u: Unified,
    s: []const f32,
    d: zs.Dims,
    c: mstack.Config,
    v: []const zblock.Views,
};

fn tomaArgsRun(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    req: Request,
    adaln: []const f32,
    ta: TomaArgs,
) !bool {
    return ztoma.run(allocator, metal, attn, .{
        .projected = ta.p,
        .state = ta.u.state,
        .pos = ta.u.pos,
        .views = ta.v,
        .adaln = adaln,
        .cfg = ta.c,
        .rope = req.rope,
        .final = finalReq(req, ta.d, ta.s),
        .dims = ta.d,
    });
}

fn runStack(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    projected: []f32,
    unified: Unified,
    req: Request,
    adaln: []const f32,
    scale: []const f32,
    dims: zs.Dims,
) !bool {
    const cfg = stackCfg(try zs.layerCfg(unified.state, req.cfg));
    const views = if (hook_bypass.managed() or token_selection.managed())
        req.tx.layers.items
    else
        limitLayers(req.tx.layers.items);
    const selection_step = if (token_selection.managed())
        token_selection.current() orelse return error.TooManyTokenSelectionSteps
    else
        null;
    if (selection_step) |step| {
        if (step.applies) {
            try mselect.run(
                allocator,
                metal,
                attn,
                projected,
                unified.state,
                views,
                adaln,
                unified.pos,
                cfg,
                req.rope,
                finalReq(req, dims, scale),
                step.plan,
            );
        } else {
            try mstack.run(
                allocator,
                metal,
                attn,
                projected,
                unified.state,
                views,
                adaln,
                unified.pos,
                cfg,
                req.rope,
                finalReq(req, dims, scale),
                .{ .total = views.len },
            );
        }
        try token_selection.recordCompleted(step.applies);
        return true;
    }
    if (toma.enabled(dims)) {
        const ta =
            TomaArgs{ .p = projected, .u = unified, .s = scale, .d = dims, .c = cfg, .v = views };
        return tomaArgsRun(allocator, metal, attn, req, adaln, ta);
    }
    const bypass_step = if (hook_bypass.managed())
        hook_bypass.current() orelse return error.TooManyBypassSteps
    else
        null;
    var selected = try selectViews(allocator, views, bypass_step);
    defer selected.deinit(allocator);
    mstack.run(
        allocator,
        metal,
        attn,
        projected,
        unified.state,
        selected.views,
        adaln,
        unified.pos,
        cfg,
        req.rope,
        finalReq(req, dims, scale),
        selected.band(views.len),
    ) catch |err| {
        if (mfallback.isGemmRefusal(err) and !hook_bypass.managed()) return false;
        return err;
    };
    if (bypass_step) |step| try hook_bypass.recordCompleted(step.applies);
    return true;
}

const SelectedViews = struct {
    views: []const zblock.Views,
    owned_views: ?[]zblock.Views = null,
    indices: ?[]usize = null,

    fn deinit(self: *SelectedViews, allocator: std.mem.Allocator) void {
        if (self.owned_views) |values| allocator.free(values);
        if (self.indices) |values| allocator.free(values);
        self.* = undefined;
    }

    fn band(self: SelectedViews, total: usize) @import("mstack_chain.zig").Band {
        return .{ .total = total, .indices = self.indices };
    }
};

fn selectViews(
    allocator: std.mem.Allocator,
    views: []const zblock.Views,
    step: ?hook_bypass.Step,
) !SelectedViews {
    const bypass = step orelse return .{ .views = views };
    if (!bypass.applies) return .{ .views = views };
    if (bypass.layer_to > views.len) return error.PlanLayerRangeMismatch;
    const count = views.len - (bypass.layer_to - bypass.layer_from);
    if (count == 0) return error.EmptyResidentStack;
    const selected = try allocator.alloc(zblock.Views, count);
    errdefer allocator.free(selected);
    const indices = try allocator.alloc(usize, count);
    var output: usize = 0;
    for (views, 0..) |view, layer| {
        if (layer >= bypass.layer_from and layer < bypass.layer_to) continue;
        selected[output] = view;
        indices[output] = layer;
        output += 1;
    }
    return .{
        .views = selected,
        .owned_views = selected,
        .indices = indices,
    };
}

// ToMA band: pre-band full tokens, merged interior band, full-token tail
// with the fused final. Falls back hard on any band error (quality gates
// cover the rest).

const Unified = struct {
    state: []f32,
    pos: []zseq.Pos,
};

fn makeUnified(
    allocator: std.mem.Allocator,
    streams: zstreams.Streams,
    dims: zs.Dims,
) !Unified {
    const state = try allocator.alloc(f32, dims.total * dims.hidden);
    const pos = try allocator.alloc(zseq.Pos, dims.total);
    _ = try zunify.basic(
        state,
        pos,
        streams.image,
        streams.img_pos,
        streams.cap,
        streams.cap_pos,
        dims.hidden,
    );
    return .{ .state = state, .pos = pos };
}

fn makeScale(
    metal: ?*mlinear.Context,
    allocator: std.mem.Allocator,
    adaln: []const f32,
    tx: *const ztx.Loaded,
) ![]f32 {
    const cond = try allocator.alloc(f32, adaln.len);
    for (cond, adaln) |*dst, value| dst.* = ops.silu(value);
    const scale = try allocator.alloc(f32, try tx.globals.final_mod_b.elems());
    try linear.run(metal, scale, cond, tx.globals.final_mod_w, tx.globals.final_mod_b);
    for (scale) |*value| value.* += 1.0;
    return scale;
}

fn stackCfg(cfg: zlayer.Config) mstack.Config {
    return .{
        .tokens = cfg.tokens,
        .hidden = cfg.hidden,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .head_dim = cfg.head_dim,
        .norm_eps = cfg.norm_eps,
    };
}

fn finalReq(req: Request, dims: zs.Dims, scale: []const f32) mstack.Final {
    return .{
        .scale = scale,
        .weight = req.tx.globals.final_w,
        .bias = req.tx.globals.final_b,
        .cfg = .{
            .tokens = dims.total,
            .hidden = dims.hidden,
            .out_dim = dims.patch_dim,
            .eps = @floatCast(req.cfg.norm_eps),
        },
    };
}

fn record(name: []const u8, start: u64, metal: metrics.Counters) void {
    metrics.record(name, metrics.now() - start);
    metrics.recordMetal(name, metal);
}
