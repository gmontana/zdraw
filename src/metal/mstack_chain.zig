//! One-command-buffer resident transformer-stack execution.

const std = @import("std");
const metrics = @import("metrics.zig");

const c = @import("metal_c.zig");
const bufs = @import("mblock_chain_buf.zig");
const chain_c = @import("mblock_chain_c.zig");
const mattn = @import("mattn.zig");
const mlinear = @import("mlinear.zig");
const mmod = @import("mmod_batch.zig");
const params = @import("mblock_chain_param.zig");
const policy = @import("mstack_policy.zig");
const types = @import("mblock_chain_types.zig");
const weights = @import("mblock_chain_weight.zig");
const zblock = @import("../zimage/zblock.zig");
const zpack_file = @import("../pack/zpack_file.zig");
const zrope = @import("../zimage/zrope.zig");
const zw16 = @import("../pack/zw16.zig");

pub const Config = types.Config;
pub const Pipes = struct {
    norm: *anyopaque,
    resid: *anyopaque,
    gemm_exact: *anyopaque,
    gemm_half: ?*anyopaque,
    gemm_w8: ?*anyopaque,
    qk: *anyopaque,
    attn: *anyopaque,
    attn_threads: usize,
    attn_kernel: usize,
    swiglu: *anyopaque,
    resid_norm: *anyopaque,
};

pub const Set = struct {
    bounds: []weights.Bound,
    cweights: []chain_c.Weights,
    params: []chain_c.Params,
    filled: usize,

    pub fn deinit(self: *Set, allocator: std.mem.Allocator) void {
        for (self.bounds[0..self.filled]) |*item| item.deinit();
        allocator.free(self.params);
        allocator.free(self.cweights);
        allocator.free(self.bounds);
    }
};

pub fn run(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    attn: *mattn.Context,
    state: []f32,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    family: zpack_file.Family,
    band: Band,
) !void {
    if (views.len == 0) return;
    try validateBand(band, views.len);
    const pol = policy.fromEnv();
    const select = policy.selectFromEnv();
    const first_modes = policy.modes(
        metal.gemm_mode,
        pol,
        select,
        band.index(0),
        band.total,
    );
    if (policy.allOff(first_modes)) return error.GemmUnavailable;
    const pipes = try getPipes(metal, attn, pol, attn.pick(cfg.tokens, cfg.head_dim));
    try params.check(state, pos, null, cfg);
    const t0 = metrics.now();
    var block_bufs = try bufs.make(metal, state, views[0], cfg);
    defer block_bufs.deinit();
    const t1 = metrics.now();
    var set =
        try makeSet(allocator, metal, views, adaln, pos, cfg, rope, pol, select, family, band);
    defer set.deinit(allocator);
    const t2 = metrics.now();
    try dispatch(metal, attn, pipes, block_bufs.c, set);
    const t3 = metrics.now();
    readBack(block_bufs.c.state, state);
    if (family == .noise) {
        metrics.record("nstack-bufs", t1 - t0);
        metrics.record("nstack-set", t2 - t1);
        metrics.record("nstack-gpu", t3 - t2);
        metrics.record("nstack-read", metrics.now() - t3);
    }
}

/// Absolute layer window: `first` offsets policy and sidecar lookups when
/// `views` is a slice of a larger stack; `total` is the full stack depth.
pub const Band = struct {
    first: usize = 0,
    total: usize,
    indices: ?[]const usize = null,

    pub fn index(self: Band, relative: usize) usize {
        if (self.indices) |values| return values[relative];
        return self.first + relative;
    }
};

pub fn validateBand(band: Band, count: usize) !void {
    if (band.total == 0) return error.InvalidLayerBand;
    if (band.indices) |indices| {
        if (indices.len != count) return error.InvalidLayerBand;
        for (indices, 0..) |index, position| {
            if (index >= band.total or
                (position > 0 and indices[position - 1] >= index))
            {
                return error.InvalidLayerBand;
            }
        }
    } else if (band.first > band.total or count > band.total - band.first) {
        return error.InvalidLayerBand;
    }
}

pub const Geometry = struct {
    pos: *anyopaque,
    rope: *anyopaque,
};

pub fn makeSet(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    stack_policy: policy.Policy,
    select: policy.Select,
    family: zpack_file.Family,
    band: Band,
) !Set {
    var mods = if (adaln) |input| try mmod.run(allocator, metal, input, views) else null;
    defer if (mods) |*batch| batch.deinit(allocator);
    return makeSetBound(
        allocator,
        metal,
        views,
        if (mods) |*batch| batch else null,
        null,
        pos,
        cfg,
        rope,
        stack_policy,
        select,
        family,
        band,
        null,
    );
}

pub fn makeSetGeo(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: []const zblock.Views,
    adaln: ?[]const f32,
    cfg: Config,
    rope: zrope.Cache,
    stack_policy: policy.Policy,
    select: policy.Select,
    family: zpack_file.Family,
    band: Band,
    geometry: Geometry,
) !Set {
    var mods = if (adaln) |input| try mmod.run(allocator, metal, input, views) else null;
    defer if (mods) |*batch| batch.deinit(allocator);
    return makeSetBound(
        allocator,
        metal,
        views,
        if (mods) |*batch| batch else null,
        null,
        null,
        cfg,
        rope,
        stack_policy,
        select,
        family,
        band,
        geometry,
    );
}

/// Build one layer set from a modulation batch aligned to the full stack.
///
/// The caller owns `mods`; `band` maps each view back to its absolute layer.
pub fn makeSetFromMods(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: []const zblock.Views,
    mods: *const mmod.Batch,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    stack_policy: policy.Policy,
    select: policy.Select,
    family: zpack_file.Family,
    band: Band,
) !Set {
    return makeSetBound(
        allocator,
        metal,
        views,
        mods,
        band,
        pos,
        cfg,
        rope,
        stack_policy,
        select,
        family,
        band,
        null,
    );
}

/// Build a geometry-specialized layer set from full-stack modulations.
pub fn makeSetGeoMods(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: []const zblock.Views,
    mods: *const mmod.Batch,
    cfg: Config,
    rope: zrope.Cache,
    stack_policy: policy.Policy,
    select: policy.Select,
    family: zpack_file.Family,
    band: Band,
    geometry: Geometry,
) !Set {
    return makeSetBound(
        allocator,
        metal,
        views,
        mods,
        band,
        null,
        cfg,
        rope,
        stack_policy,
        select,
        family,
        band,
        geometry,
    );
}

fn makeSetBound(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: []const zblock.Views,
    mods: ?*const mmod.Batch,
    modulation_band: ?Band,
    pos: ?[]const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    stack_policy: policy.Policy,
    select: policy.Select,
    family: zpack_file.Family,
    band: Band,
    geometry: ?Geometry,
) !Set {
    try validateBand(band, views.len);
    var set = try initSet(allocator, views.len);
    errdefer set.deinit(allocator);
    try fillSet(
        &set,
        allocator,
        metal,
        views,
        mods,
        modulation_band,
        pos,
        cfg,
        rope,
        stack_policy,
        select,
        family,
        band,
        geometry,
    );
    return set;
}

fn fillSet(
    set: *Set,
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: []const zblock.Views,
    mods: ?*const mmod.Batch,
    modulation_band: ?Band,
    pos: ?[]const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
    stack_policy: policy.Policy,
    select: policy.Select,
    family: zpack_file.Family,
    band: Band,
    geometry: ?Geometry,
) !void {
    const sidecar = metal.buffers.sidecarBytes();
    const w16_opts = zw16.Options.fromEnv();
    for (views, 0..) |view, rel| {
        const index = band.index(rel);
        const modes = policy.modes(metal.gemm_mode, stack_policy, select, index, band.total);
        const modulation_index = modulationIndex(modulation_band, rel);
        const layer_mods = if (mods) |batch| try batch.parts(modulation_index) else null;
        var fshape: [2]usize = undefined;
        const bound_view =
            try zw16.substitute(w16_opts, sidecar, modes, view, family, @intCast(index), &fshape);
        set.bounds[rel] = if (geometry) |geo|
            try weights.makePackedGeo(
                allocator,
                metal,
                bound_view,
                layer_mods,
                geo.pos,
                geo.rope,
                policy.packedModes(modes),
                family,
                index,
            )
        else
            try weights.makePacked(
                allocator,
                metal,
                bound_view,
                layer_mods,
                pos.?,
                rope,
                policy.packedModes(modes),
                family,
                index,
            );
        set.filled += 1;
        set.cweights[rel] = set.bounds[rel].c;
        const binds = set.bounds[rel].binds;
        set.params[rel] = try params.makeModes(modes, binds, bound_view, layer_mods, cfg, rope);
    }
}

fn modulationIndex(band: ?Band, relative: usize) usize {
    return if (band) |value| value.index(relative) else relative;
}

fn initSet(allocator: std.mem.Allocator, count: usize) !Set {
    const bounds = try allocator.alloc(weights.Bound, count);
    errdefer allocator.free(bounds);
    const cweights = try allocator.alloc(chain_c.Weights, count);
    errdefer allocator.free(cweights);
    const ps = try allocator.alloc(chain_c.Params, count);
    return .{ .bounds = bounds, .cweights = cweights, .params = ps, .filled = 0 };
}

pub fn dispatch(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    pipes: Pipes,
    buffers: chain_c.Buffers,
    set: Set,
) !void {
    const threads = chain_c.Threads{
        .block = metal.block_threads,
        .qk = attn.qk_threads,
        .attn = pipes.attn_threads,
        .attn_kernel = pipes.attn_kernel,
        .swiglu = metal.swiglu_threads,
    };
    const code = chain_c.zdraw_metal_run_stack_chain(
        metal.queue,
        pipes.norm,
        pipes.resid,
        pipes.gemm_exact,
        pipes.gemm_half,
        pipes.gemm_w8,
        pipes.qk,
        pipes.attn,
        pipes.swiglu,
        metal.swiglu_fused_pipeline,
        pipes.resid_norm,
        &buffers,
        set.cweights.ptr,
        set.params.ptr,
        set.filled,
        &threads,
    );
    if (code != 0) return error.MetalDispatchFailed;
}

pub fn getPipes(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    stack_policy: policy.Policy,
    pick: mattn.Context.Pick,
) !Pipes {
    return .{
        .norm = metal.norm_pipeline orelse return error.GemmUnavailable,
        .resid = metal.resid_pipeline orelse return error.GemmUnavailable,
        .gemm_exact = metal.gemm_exact_pipeline orelse return error.GemmUnavailable,
        .gemm_half = if (policy.needsHalf(stack_policy))
            metal.gemm_half_pipeline orelse return error.GemmUnavailable
        else
            metal.gemm_half_pipeline,
        .gemm_w8 = if (policy.needsW8(stack_policy))
            metal.gemm_w8_pipeline orelse return error.GemmUnavailable
        else
            metal.gemm_w8_pipeline,
        .qk = attn.qk_pipeline orelse return error.GemmUnavailable,
        .attn = pick.pipeline,
        .attn_threads = pick.threads,
        .attn_kernel = @intFromEnum(pick.kernel),
        .swiglu = metal.swiglu_pipeline orelse return error.GemmUnavailable,
        .resid_norm = metal.resid_norm_pipeline orelse return error.GemmUnavailable,
    };
}

fn readBack(handle: *anyopaque, state: []f32) void {
    const bytes = std.mem.sliceAsBytes(state);
    c.zdraw_metal_read_buffer(handle, bytes.ptr, bytes.len);
}

test "explicit layer bands preserve absolute sidecar identity" {
    const indices = [_]usize{ 0, 2, 5 };
    const band = Band{ .total = 6, .indices = &indices };
    try validateBand(band, indices.len);
    try std.testing.expectEqual(@as(usize, 2), band.index(1));
    try std.testing.expectError(
        error.InvalidLayerBand,
        validateBand(.{ .total = 6, .indices = &.{ 0, 2, 2 } }, 3),
    );
    try std.testing.expectEqual(@as(usize, 5), modulationIndex(band, 2));
    try std.testing.expectEqual(@as(usize, 2), modulationIndex(null, 2));
}
