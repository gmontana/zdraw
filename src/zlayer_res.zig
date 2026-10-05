//! Resident block adapter for `zlayer`.

const mattn = @import("mattn.zig");
const mblock_chain = @import("mblock_chain.zig");
const mblock_res = @import("mblock_res.zig");
const mlinear = @import("mlinear.zig");
const mfallback = @import("metal_fallback.zig");
const zblock = @import("zblock.zig");
const zmod = @import("zmod.zig");
const zrope = @import("zrope.zig");

pub const Config = @import("mblock_chain_types.zig").Config;

pub fn run(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    state: []f32,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
) !bool {
    const m = metal orelse return false;
    const a = attn orelse return false;
    if (try chain(m, a, state, views, mods, pos, cfg, rope)) return true;
    mblock_res.run(m, a, state, views, mods, pos, cfg, rope) catch |err| {
        if (mfallback.isFallback(err)) return false;
        return err;
    };
    return true;
}

fn chain(
    metal: *mlinear.Context,
    attn: *mattn.Context,
    state: []f32,
    views: zblock.Views,
    mods: ?zmod.Parts,
    pos: []const zrope.Pos,
    cfg: Config,
    rope: zrope.Cache,
) !bool {
    mblock_chain.run(metal, attn, state, views, mods, pos, cfg, rope) catch |err| {
        if (mfallback.isFallback(err)) return false;
        return err;
    };
    return true;
}
