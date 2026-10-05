//! Split-half rotary position embedding used by Qwen.

const std = @import("std");

pub const Error = error{
    InvalidShape,
};

/// Largest head_dim/2 a caller may table. 128 covers every Qwen head_dim we
/// run (128) with room to spare, and keeps the buffers stack-sized.
pub const max_half = 128;

pub fn apply(vec: []f32, position: usize, theta: f32) !void {
    if (vec.len == 0 or vec.len % 2 != 0) return error.InvalidShape;
    const half = vec.len / 2;
    if (half > max_half) return error.InvalidShape;
    var cos_tab: [max_half]f32 = undefined;
    var sin_tab: [max_half]f32 = undefined;
    try table(cos_tab[0..half], sin_tab[0..half], position, theta);
    applyTable(vec, cos_tab[0..half], sin_tab[0..half]);
}

/// Fill the cos/sin pair for one position. The frequencies depend only on
/// (position, i) - not on the head or the layer - so a caller rotating many
/// heads at the same position builds this once and reuses it.
pub fn table(cos_tab: []f32, sin_tab: []f32, position: usize, theta: f32) !void {
    if (cos_tab.len == 0 or cos_tab.len != sin_tab.len) return error.InvalidShape;
    const pos: f32 = @floatFromInt(position);
    const dim: f32 = @floatFromInt(cos_tab.len * 2);
    for (0..cos_tab.len) |i| {
        const freq = pos / @exp(@log(theta) * exponent(i, dim));
        cos_tab[i] = @cos(freq);
        sin_tab[i] = @sin(freq);
    }
}

/// Rotate one head vector against a table built by `table`. Bit-identical to
/// `apply` for the same position: the arithmetic is unchanged, only the
/// transcendentals are hoisted.
pub fn applyTable(vec: []f32, cos_tab: []const f32, sin_tab: []const f32) void {
    const half = cos_tab.len;
    for (0..half) |i| {
        const c = cos_tab[i];
        const s = sin_tab[i];
        const a = vec[i];
        const b = vec[i + half];
        vec[i] = a * c - b * s;
        vec[i + half] = b * c + a * s;
    }
}

pub fn applyPair(q: []f32, k: []f32, position: usize, theta: f32) !void {
    if (q.len != k.len) return error.InvalidShape;
    try apply(q, position, theta);
    try apply(k, position, theta);
}

fn exponent(i: usize, dim: f32) f32 {
    const fi: f32 = @floatFromInt(i);
    return (2.0 * fi) / dim;
}

test "position zero leaves vector unchanged" {
    var v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try apply(&v, 0, 10000.0);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, &v);
}

test "tabled rope is bit-identical to the per-element formula" {
    // The pre-table implementation, inlined so the comparison is against the
    // original arithmetic rather than against apply() calling the same table.
    const reference = struct {
        fn call(vec: []f32, position: usize, theta: f32) void {
            const half = vec.len / 2;
            const pos: f32 = @floatFromInt(position);
            const dim: f32 = @floatFromInt(vec.len);
            for (0..half) |i| {
                const freq = pos / @exp(@log(theta) * exponent(i, dim));
                const c = @cos(freq);
                const s = @sin(freq);
                const a = vec[i];
                const b = vec[i + half];
                vec[i] = a * c - b * s;
                vec[i + half] = b * c + a * s;
            }
        }
    }.call;

    var prng = std.Random.DefaultPrng.init(0x0f5e);
    const rng = prng.random();
    for ([_]usize{ 0, 1, 7, 63, 511 }) |pos| {
        var want: [128]f32 = undefined;
        for (&want) |*v| v.* = rng.floatNorm(f32);
        var got = want;
        reference(&want, pos, 1_000_000.0);
        try apply(&got, pos, 1_000_000.0);
        try std.testing.expectEqualSlices(f32, &want, &got);
    }
}

test "rotate split halves" {
    var v = [_]f32{ 1.0, 2.0 };
    try apply(&v, 1, 1.0);

    try std.testing.expectApproxEqAbs(@cos(1.0) - 2.0 * @sin(1.0), v[0], 0.0001);
    try std.testing.expectApproxEqAbs(2.0 * @cos(1.0) + @sin(1.0), v[1], 0.0001);
}
