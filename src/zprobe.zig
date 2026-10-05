//! Diagnostic hidden-state capture for cacheability probes.

const std = @import("std");

pub const LayerProbe = struct {
    before: ?[]f32 = null,
    after: []f32,
    state_len: usize,

    pub fn captureBefore(self: LayerProbe, state: []const f32) !void {
        const output = self.before orelse return;
        if (output.len != self.state_len or state.len != self.state_len) {
            return error.InvalidShape;
        }
        @memcpy(output, state);
    }

    pub fn capture(self: LayerProbe, layer: usize, state: []const f32) !void {
        if (self.state_len != state.len) return error.InvalidShape;
        const start = layer * self.state_len;
        if (start + self.state_len > self.after.len) return error.InvalidShape;
        @memcpy(self.after[start..][0..self.state_len], state);
    }
};

pub const LayerStats = struct {
    rel_sum: f64 = 0,
    rel_max: f64 = 0,
    cos_sum: f64 = 0,
    count: usize = 0,

    pub fn add(self: *LayerStats, prev: []const f32, curr: []const f32) !void {
        if (prev.len != curr.len) return error.InvalidShape;
        var diff2: f64 = 0;
        var prev2: f64 = 0;
        var curr2: f64 = 0;
        var dot: f64 = 0;
        for (prev, curr) |a, b| {
            const da: f64 = @floatCast(a);
            const db: f64 = @floatCast(b);
            const d = db - da;
            diff2 += d * d;
            prev2 += da * da;
            curr2 += db * db;
            dot += da * db;
        }
        const rel = std.math.sqrt(diff2 / @max(prev2, 1.0e-20));
        const cos = dot / @max(std.math.sqrt(prev2 * curr2), 1.0e-20);
        self.rel_sum += rel;
        self.rel_max = @max(self.rel_max, rel);
        self.cos_sum += cos;
        self.count += 1;
    }

    pub fn meanRel(self: LayerStats) f64 {
        return if (self.count == 0) 0 else self.rel_sum / @as(f64, @floatFromInt(self.count));
    }

    pub fn meanCos(self: LayerStats) f64 {
        return if (self.count == 0) 0 else self.cos_sum / @as(f64, @floatFromInt(self.count));
    }
};

pub fn layerSlice(buf: []const f32, state_len: usize, layer: usize) []const f32 {
    return buf[layer * state_len ..][0..state_len];
}

test "layer probe captures a layer slice" {
    var out = [_]f32{0} ** 6;
    try (LayerProbe{ .after = &out, .state_len = 3 }).capture(1, &.{ 1, 2, 3 });
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 1, 2, 3 }, &out);
}

test "layer probe optionally captures stack input" {
    var before = [_]f32{0} ** 3;
    var after = [_]f32{0} ** 3;
    const probe = LayerProbe{
        .before = &before,
        .after = &after,
        .state_len = 3,
    };
    try probe.captureBefore(&.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, &before);
}
