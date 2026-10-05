//! Dynamic Metal buffers for chained block execution.

const std = @import("std");

const mbuffer = @import("mbuffer.zig");
const mlinear = @import("mlinear.zig");
const types = @import("mblock_chain_types.zig");
const zmod = @import("zmod.zig");
const zrope = @import("zrope.zig");

pub const Dyn = struct {
    attn_scale: *anyopaque,
    attn_gate: *anyopaque,
    mlp_scale: *anyopaque,
    mlp_gate: *anyopaque,
    pos: *anyopaque,
    rope: *anyopaque,
};

pub fn make(
    metal: *mlinear.Context,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    rope: zrope.Cache,
    temps: *[6]?mbuffer.Buffer,
) !Dyn {
    try makeMods(metal, mods, temps);
    temps[4] = try mbuffer.Buffer.fromBytes(metal.device, std.mem.sliceAsBytes(pos));
    temps[5] = try mbuffer.Buffer.fromBytes(metal.device, rope.pairBytes());
    return .{
        .attn_scale = handleOrZero(metal, temps[0]),
        .attn_gate = handleOrZero(metal, temps[1]),
        .mlp_scale = handleOrZero(metal, temps[2]),
        .mlp_gate = handleOrZero(metal, temps[3]),
        .pos = temps[4].?.handle,
        .rope = temps[5].?.handle,
    };
}

pub fn makeBorrowed(
    metal: *mlinear.Context,
    mods: ?zmod.Parts,
    pos: *anyopaque,
    rope: *anyopaque,
    temps: *[6]?mbuffer.Buffer,
) !Dyn {
    try makeMods(metal, mods, temps);
    return .{
        .attn_scale = handleOrZero(metal, temps[0]),
        .attn_gate = handleOrZero(metal, temps[1]),
        .mlp_scale = handleOrZero(metal, temps[2]),
        .mlp_gate = handleOrZero(metal, temps[3]),
        .pos = pos,
        .rope = rope,
    };
}

fn makeMods(
    metal: *mlinear.Context,
    mods: ?zmod.Parts,
    temps: *[6]?mbuffer.Buffer,
) !void {
    temps[0] = try optBuf(metal, types.attnScale(mods));
    temps[1] = try optBuf(metal, types.attnGate(mods));
    temps[2] = try optBuf(metal, types.mlpScale(mods));
    temps[3] = try optBuf(metal, types.mlpGate(mods));
}

fn optBuf(metal: *mlinear.Context, values: ?[]const f32) !?mbuffer.Buffer {
    if (values) |slice| {
        return try mbuffer.Buffer.fromBytes(metal.device, std.mem.sliceAsBytes(slice));
    }
    return null;
}

fn handleOrZero(metal: *mlinear.Context, buf: ?mbuffer.Buffer) *anyopaque {
    if (buf) |value| return value.handle;
    return metal.buffers.zero;
}
