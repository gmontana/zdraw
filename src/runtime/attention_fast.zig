//! Shared attention helper with an optional Metal backend.

const attention = @import("attention.zig");
const mattn = @import("../metal/mattn.zig");

pub fn run(
    metal: ?*mattn.Context,
    out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    scratch: []f32,
    cfg: attention.Config,
) !void {
    if (metal) |ctx| {
        if (ctx.run(out, q, k, v, cfg)) |_| {
            return; // Metal handled the shape
        } else |err| switch (err) {
            error.UnsupportedShape => {}, // fall through to the CPU kernel
            else => return err,
        }
    }
    try attention.run(out, q, k, v, scratch, cfg);
}
