//! Wander's coordinates: the starting latent at (u, v) in a field whose four
//! corners are seeds. Bilinear spherical interpolation of the four seeds'
//! Gaussian noises (slerp along u on each row, then along v), so every
//! coordinate is a stable picture and neighbours morph (ledger
//! slerp-probe-20260828). Corners all zero = no field.
const std = @import("std");
const zdenoise = @import("zdenoise.zig");

pub const Field = struct {
    corners: [4]u64 = .{ 0, 0, 0, 0 },
    u: f32 = 0,
    v: f32 = 0,

    pub fn active(self: Field) bool {
        for (self.corners) |c| if (c != 0) return true;
        return false;
    }
};

/// Fill `x` with the blended noise for `field`.
pub fn fill(allocator: std.mem.Allocator, x: []f32, field: Field) !void {
    const n = x.len;
    const tmp = try allocator.alloc(f32, 4 * n);
    defer allocator.free(tmp);
    for (0..4) |i| zdenoise.fillNoise(tmp[i * n ..][0..n], field.corners[i]);
    const top = try allocator.alloc(f32, n);
    defer allocator.free(top);
    const bottom = try allocator.alloc(f32, n);
    defer allocator.free(bottom);
    slerp(top, tmp[0..n], tmp[n .. 2 * n], std.math.clamp(field.u, 0, 1));
    slerp(bottom, tmp[2 * n .. 3 * n], tmp[3 * n .. 4 * n], std.math.clamp(field.u, 0, 1));
    slerp(x, top, bottom, std.math.clamp(field.v, 0, 1));
}

fn slerp(out: []f32, a: []const f32, b: []const f32, t: f32) void {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |ai, bi| {
        dot += @as(f64, ai) * bi;
        na += @as(f64, ai) * ai;
        nb += @as(f64, bi) * bi;
    }
    const cos_omega = std.math.clamp(dot / (@sqrt(na) * @sqrt(nb) + 1e-12), -1.0, 1.0);
    const omega = std.math.acos(cos_omega);
    if (omega < 1e-6) {
        for (out, a, b) |*o, ai, bi| o.* = (1 - t) * ai + t * bi;
        return;
    }
    const so = @sin(omega);
    const wa: f32 = @floatCast(@sin((1 - @as(f64, t)) * omega) / so);
    const wb: f32 = @floatCast(@sin(@as(f64, t) * omega) / so);
    for (out, a, b) |*o, ai, bi| o.* = wa * ai + wb * bi;
}

test "the corners reproduce their own seeds" {
    var x: [64]f32 = undefined;
    var want: [64]f32 = undefined;
    zdenoise.fillNoise(&want, 7);
    try fill(std.testing.allocator, &x, .{ .corners = .{ 7, 11, 13, 17 }, .u = 0, .v = 0 });
    for (x, want) |got, w| try std.testing.expectApproxEqAbs(w, got, 1e-5);
    zdenoise.fillNoise(&want, 17);
    try fill(std.testing.allocator, &x, .{ .corners = .{ 7, 11, 13, 17 }, .u = 1, .v = 1 });
    for (x, want) |got, w| try std.testing.expectApproxEqAbs(w, got, 1e-4);
}
