//! Weight and dynamic-buffer binding for chained block execution.

const std = @import("std");

const chain_c = @import("mblock_chain_c.zig");
const dynbuf = @import("mblock_dyn.zig");
const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const tensor = @import("../pack/tensor.zig");
const zblock = @import("../zimage/zblock.zig");
const zmod = @import("../zimage/zmod.zig");
const zpack_file = @import("../pack/zpack_file.zig");
const zrope = @import("../zimage/zrope.zig");

const w8_group = 64;

pub const Binds = struct {
    attn_in: mbuffer.Bind,
    q: mbuffer.Bind,
    k: mbuffer.Bind,
    v: mbuffer.Bind,
    q_norm: mbuffer.Bind,
    k_norm: mbuffer.Bind,
    proj: mbuffer.Bind,
    attn_out: mbuffer.Bind,
    ffn_in: mbuffer.Bind,
    ffn_gate: mbuffer.Bind,
    ffn_up: mbuffer.Bind,
    ffn_down: mbuffer.Bind,
    ffn_out: mbuffer.Bind,
    ffn_fused: mbuffer.Bind,
};

const Temps = struct {
    weight: [14]?mbuffer.Buffer = [_]?mbuffer.Buffer{null} ** 14,
    dyn: [6]?mbuffer.Buffer = [_]?mbuffer.Buffer{null} ** 6,

    fn deinit(self: *Temps) void {
        for (&self.weight) |*buf| if (buf.*) |*value| value.deinit();
        for (&self.dyn) |*buf| if (buf.*) |*value| value.deinit();
    }
};

pub const Packed = struct {
    q: bool = false,
    k: bool = false,
    v: bool = false,
    proj: bool = false,
    ffn_gate: bool = false,
    ffn_up: bool = false,
    ffn_down: bool = false,
};

pub const Bound = struct {
    c: chain_c.Weights,
    binds: Binds,
    temps: Temps,

    pub fn deinit(self: *Bound) void {
        self.temps.deinit();
    }
};

pub fn make(
    metal: *mlinear.Context,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    rope: zrope.Cache,
) !Bound {
    var temps = Temps{};
    errdefer temps.deinit();
    const binds = try bindWeights(null, metal, views, &temps, .{}, .main, 0);
    const dyn = try dynbuf.make(metal, mods, pos, rope, &temps.dyn);
    return .{ .c = cweights(binds, dyn), .binds = binds, .temps = temps };
}

pub fn makePacked(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    rope: zrope.Cache,
    packed_in: Packed,
    family: zpack_file.Family,
    layer: usize,
) !Bound {
    var temps = Temps{};
    errdefer temps.deinit();
    const binds = try bindWeights(allocator, metal, views, &temps, packed_in, family, layer);
    const dyn = try dynbuf.make(metal, mods, pos, rope, &temps.dyn);
    return .{ .c = cweights(binds, dyn), .binds = binds, .temps = temps };
}

pub fn makePackedGeo(
    allocator: std.mem.Allocator,
    metal: *mlinear.Context,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: *anyopaque,
    rope_handle: *anyopaque,
    packed_in: Packed,
    family: zpack_file.Family,
    layer: usize,
) !Bound {
    var temps = Temps{};
    errdefer temps.deinit();
    const binds = try bindWeights(
        allocator,
        metal,
        views,
        &temps,
        packed_in,
        family,
        layer,
    );
    const dyn = try dynbuf.makeBorrowed(
        metal,
        mods,
        pos,
        rope_handle,
        &temps.dyn,
    );
    return .{ .c = cweights(binds, dyn), .binds = binds, .temps = temps };
}

fn bindWeights(
    allocator: ?std.mem.Allocator,
    metal: *mlinear.Context,
    views: zblock.Views,
    temps: *Temps,
    packed_in: Packed,
    family: zpack_file.Family,
    layer: usize,
) !Binds {
    const v = views;
    const p = packed_in;
    const a = allocator;
    const fam = family;
    return .{
        .attn_in = try bind(metal, v.attn_in, temps, 0),
        .q = try bindLin(a, metal, v.q, temps, 1, p.q, fam, layer, .q),
        .k = try bindLin(a, metal, v.k, temps, 2, p.k, fam, layer, .k),
        .v = try bindLin(a, metal, v.v, temps, 3, p.v, fam, layer, .v),
        .q_norm = try bind(metal, v.q_norm, temps, 4),
        .k_norm = try bind(metal, v.k_norm, temps, 5),
        .proj = try bindLin(a, metal, v.proj, temps, 6, p.proj, fam, layer, .proj),
        .attn_out = try bind(metal, v.attn_out, temps, 7),
        .ffn_in = try bind(metal, v.ffn_in, temps, 8),
        .ffn_gate = try bindLin(a, metal, v.ffn_gate, temps, 9, p.ffn_gate, fam, layer, .ffn_gate),
        .ffn_up = try bindLin(a, metal, v.ffn_up, temps, 10, p.ffn_up, fam, layer, .ffn_up),
        .ffn_down = try bindLin(a, metal, v.ffn_down, temps, 11, p.ffn_down, fam, layer, .ffn_down),
        .ffn_out = try bind(metal, v.ffn_out, temps, 12),
        .ffn_fused = try bindFused(metal, v.ffn_fused, temps, 13),
    };
}

// The fused gate+up weight is optional; absent layers bind the shared zero
// buffer and leave the fused GemmParams disabled (mode 0).
fn bindFused(
    metal: *mlinear.Context,
    view: ?tensor.View,
    temps: *Temps,
    index: usize,
) !mbuffer.Bind {
    const v = view orelse return .{ .handle = metal.buffers.zero, .offset = 0 };
    return bind(metal, v, temps, index);
}

fn cweights(binds: Binds, dyn: dynbuf.Dyn) chain_c.Weights {
    return .{
        .attn_in = binds.attn_in.handle,
        .attn_scale = dyn.attn_scale,
        .q = binds.q.handle,
        .k = binds.k.handle,
        .v = binds.v.handle,
        .q_norm = binds.q_norm.handle,
        .k_norm = binds.k_norm.handle,
        .pos = dyn.pos,
        .rope = dyn.rope,
        .proj = binds.proj.handle,
        .attn_out = binds.attn_out.handle,
        .attn_gate = dyn.attn_gate,
        .ffn_in = binds.ffn_in.handle,
        .mlp_scale = dyn.mlp_scale,
        .ffn_gate = binds.ffn_gate.handle,
        .ffn_up = binds.ffn_up.handle,
        .ffn_down = binds.ffn_down.handle,
        .ffn_out = binds.ffn_out.handle,
        .mlp_gate = dyn.mlp_gate,
        .ffn_fused = binds.ffn_fused.handle,
    };
}

fn bind(metal: *mlinear.Context, view: tensor.View, temps: *Temps, index: usize) !mbuffer.Bind {
    return metal.buffers.bindView(view, &temps.weight[index]);
}

fn bindLin(
    allocator: ?std.mem.Allocator,
    metal: *mlinear.Context,
    view: tensor.View,
    temps: *Temps,
    index: usize,
    use_w8: bool,
    family: zpack_file.Family,
    layer: usize,
    kind: zpack_file.Kind,
) !mbuffer.Bind {
    if (!use_w8) return bind(metal, view, temps, index);
    const alloc = allocator orelse return error.MissingAllocator;
    return metal.buffers.bindW8Layer(alloc, view, w8_group, family, @intCast(layer), kind);
}
