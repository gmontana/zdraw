//! Z-attention resident path adapter.

const mattn = @import("mattn.zig");
const mlinear = @import("mlinear.zig");
const mfallback = @import("metal_fallback.zig");
const mres = @import("mres.zig");
const zrope = @import("zrope.zig");

pub fn run(
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    out: []f32,
    pos: []const zrope.Pos,
    weights: mres.Weights,
    cfg: mres.Config,
    rope_cache: zrope.Cache,
) !bool {
    const m = metal orelse return false;
    const a = attn orelse return false;
    mres.run(m, a, out, out, pos, weights, cfg, rope_cache) catch |err| {
        if (mfallback.isFallback(err)) return false;
        return err;
    };
    return true;
}
