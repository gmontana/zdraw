//! Complex-pair rotary embedding used by the Z-Image transformer.
//!
//! The model splits each attention head into three position axes. Each axis
//! gets its own frequency table, then adjacent float pairs are rotated as a
//! complex number. The table is small, so we precompute it once.

const std = @import("std");

pub const Pos = [3]usize;

pub const Config = struct {
    dims: [3]usize,
    lens: [3]usize,
    theta: f32,
};

pub const Error = error{
    InvalidConfig,
    InvalidShape,
    BadPosition,
};

const Pair = struct {
    c: f32,
    s: f32,
};

pub const Cache = struct {
    cfg: Config,
    values: []Pair,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) !Cache {
        try checkConfig(cfg);
        const values = try allocator.alloc(Pair, count(cfg));
        errdefer allocator.free(values);
        fill(values, cfg);
        return .{ .cfg = cfg, .values = values };
    }

    pub fn deinit(self: Cache, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }

    pub fn apply(self: Cache, vec: []f32, pos: Pos) !void {
        if (vec.len != headDim(self.cfg)) return error.InvalidShape;
        var vpos: usize = 0;
        for (0..3) |ax| {
            if (pos[ax] >= self.cfg.lens[ax]) return error.BadPosition;
            const dim = self.cfg.dims[ax];
            const pairs = self.axis(ax, pos[ax]);
            applyAxis(vec[vpos..][0..dim], pairs);
            vpos += dim;
        }
    }

    pub fn applyPair(self: Cache, q: []f32, k: []f32, pos: Pos) !void {
        if (q.len != k.len) return error.InvalidShape;
        try self.apply(q, pos);
        try self.apply(k, pos);
    }

    pub fn pairBytes(self: Cache) []const u8 {
        return std.mem.sliceAsBytes(self.values);
    }

    pub fn axisBases(self: Cache) [3]usize {
        return .{
            axisBase(self.cfg, 0),
            axisBase(self.cfg, 1),
            axisBase(self.cfg, 2),
        };
    }

    fn axis(self: Cache, axis_id: usize, pos: usize) []const Pair {
        const pairs = pairCount(self.cfg.dims[axis_id]);
        const base = axisBase(self.cfg, axis_id) + pos * pairs;
        return self.values[base..][0..pairs];
    }
};

fn checkConfig(cfg: Config) !void {
    if (cfg.theta <= 0.0) return error.InvalidConfig;
    for (cfg.dims, cfg.lens) |dim, len| {
        if (dim == 0 or len == 0 or dim % 2 != 0) return error.InvalidConfig;
    }
}

fn fill(values: []Pair, cfg: Config) void {
    var off: usize = 0;
    for (0..3) |axis| {
        const n = cfg.lens[axis] * pairCount(cfg.dims[axis]);
        fillAxis(values[off..][0..n], cfg.dims[axis], cfg.theta);
        off += n;
    }
}

fn fillAxis(values: []Pair, dim: usize, theta: f32) void {
    const pairs = pairCount(dim);
    for (0..values.len / pairs) |pos| {
        const fpos: f32 = @floatFromInt(pos);
        for (0..pairs) |pair| {
            const freq = fpos * @exp(-@log(theta) * power(pair, dim));
            values[pos * pairs + pair] = .{ .c = @cos(freq), .s = @sin(freq) };
        }
    }
}

fn applyAxis(vec: []f32, pairs: []const Pair) void {
    for (pairs, 0..) |rot, pair| {
        const i = pair * 2;
        const a = vec[i];
        const b = vec[i + 1];
        vec[i] = a * rot.c - b * rot.s;
        vec[i + 1] = b * rot.c + a * rot.s;
    }
}

fn count(cfg: Config) usize {
    var total: usize = 0;
    for (0..3) |axis| total += cfg.lens[axis] * pairCount(cfg.dims[axis]);
    return total;
}

fn axisBase(cfg: Config, axis_id: usize) usize {
    var base: usize = 0;
    for (0..axis_id) |axis| base += cfg.lens[axis] * pairCount(cfg.dims[axis]);
    return base;
}

fn headDim(cfg: Config) usize {
    return cfg.dims[0] + cfg.dims[1] + cfg.dims[2];
}

fn pairCount(dim: usize) usize {
    return dim / 2;
}

fn power(pair: usize, dim: usize) f32 {
    return 2.0 * @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(dim));
}

test "position zero leaves head unchanged" {
    const allocator = std.testing.allocator;
    const cfg = Config{ .dims = .{ 2, 2, 2 }, .lens = .{ 2, 2, 2 }, .theta = 256.0 };
    const cache = try Cache.init(allocator, cfg);
    defer cache.deinit(allocator);

    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try cache.apply(&vec, .{ 0, 0, 0 });
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, &vec);
}

test "rotates adjacent pair for one axis" {
    const allocator = std.testing.allocator;
    const cfg = Config{ .dims = .{ 2, 2, 2 }, .lens = .{ 3, 2, 2 }, .theta = 1.0 };
    const cache = try Cache.init(allocator, cfg);
    defer cache.deinit(allocator);

    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try cache.apply(&vec, .{ 1, 0, 0 });

    try std.testing.expectApproxEqAbs(@cos(1.0) - 2.0 * @sin(1.0), vec[0], 0.0001);
    try std.testing.expectApproxEqAbs(2.0 * @cos(1.0) + @sin(1.0), vec[1], 0.0001);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 4.0, 5.0, 6.0 }, vec[2..]);
}

test "uses pair frequency exponent" {
    const allocator = std.testing.allocator;
    const cfg = Config{ .dims = .{ 4, 2, 2 }, .lens = .{ 3, 1, 1 }, .theta = 100.0 };
    const cache = try Cache.init(allocator, cfg);
    defer cache.deinit(allocator);

    var vec = [_]f32{ 1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0 };
    try cache.apply(&vec, .{ 2, 0, 0 });

    try std.testing.expectApproxEqAbs(@cos(2.0), vec[0], 0.0001);
    try std.testing.expectApproxEqAbs(@sin(2.0), vec[1], 0.0001);
    try std.testing.expectApproxEqAbs(-@sin(0.2), vec[2], 0.0001);
    try std.testing.expectApproxEqAbs(@cos(0.2), vec[3], 0.0001);
}
