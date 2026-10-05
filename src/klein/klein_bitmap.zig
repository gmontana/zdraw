//! Per-slot bit widths for the Klein packer, read from a JSON map.
//!
//! The packer's built-in allocations (`--bits N`, the W6 scope ladder and
//! the hard-coded `mixed` profile) are fixed policies. A map makes the
//! allocation data: an experiment (the response-theory DP allocation, a
//! sensitivity sweep over one class or one block) writes a JSON file and
//! packs it, with no code change and no kernel change, because every width
//! here is one the loaders already decode.
//!
//! Keys: a class name (`to_q`, `ff_in`, ..., `qkv_mlp`, `out`) sets every
//! block of that class; `double.<i>.<class>` or `single.<i>.<class>` sets one
//! block and wins over the class; `w4_qmax` (3 or 7) selects the 4-bit grid.
//! Values are 2, 4, 6 or 16. Classes the map leaves out keep the packer's own
//! policy, so a partial map is a delta on top of `--bits`/`--w6-scope`.

const std = @import("std");

const zflux2_pack = @import("zflux2_pack.zig");

pub const Map = struct {
    arena: std.heap.ArenaAllocator,
    double: [double_count]?u8 = @splat(null),
    single: [single_count]?u8 = @splat(null),
    /// Per-block overrides keyed by the canonical "double.<i>.<class>" text.
    overrides: std.StringHashMapUnmanaged(u8) = .{},
    w4_qmax: f32 = 7.0,

    const double_count = @typeInfo(zflux2_pack.Double).@"enum".fields.len;
    const single_count = @typeInfo(zflux2_pack.Single).@"enum".fields.len;

    pub fn parse(backing: std.mem.Allocator, text: []const u8) !Map {
        var map = Map{ .arena = std.heap.ArenaAllocator.init(backing) };
        errdefer map.deinit();
        const alloc = map.arena.allocator();
        var parsed = try std.json.parseFromSlice(std.json.Value, alloc, text, .{});
        defer parsed.deinit();
        const root = switch (parsed.value) {
            .object => |o| o,
            else => return error.BadMap,
        };
        var it = root.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (std.mem.eql(u8, key, "w4_qmax")) {
                const q = try integer(kv.value_ptr.*);
                if (q != 3 and q != 7) return error.BadQmax;
                map.w4_qmax = @floatFromInt(q);
                continue;
            }
            const bits = try width(kv.value_ptr.*);
            if (std.meta.stringToEnum(zflux2_pack.Double, key)) |d| {
                map.double[@intFromEnum(d)] = bits;
            } else if (std.meta.stringToEnum(zflux2_pack.Single, key)) |s| {
                map.single[@intFromEnum(s)] = bits;
            } else if (try blockKey(key)) {
                try map.overrides.put(alloc, try alloc.dupe(u8, key), bits);
            } else return error.UnknownKey;
        }
        return map;
    }

    pub fn deinit(self: *Map) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// The width for one double-block weight, or null when the map is silent.
    pub fn widthDouble(self: *const Map, block: usize, class: zflux2_pack.Double) ?u8 {
        var buf: [64]u8 = undefined;
        const name = @tagName(class);
        const key = std.fmt.bufPrint(&buf, "double.{d}.{s}", .{ block, name }) catch return null;
        if (self.overrides.get(key)) |b| return b;
        return self.double[@intFromEnum(class)];
    }

    pub fn widthSingle(self: *const Map, block: usize, class: zflux2_pack.Single) ?u8 {
        var buf: [64]u8 = undefined;
        const name = @tagName(class);
        const key = std.fmt.bufPrint(&buf, "single.{d}.{s}", .{ block, name }) catch return null;
        if (self.overrides.get(key)) |b| return b;
        return self.single[@intFromEnum(class)];
    }
};

fn integer(v: std.json.Value) !i64 {
    return switch (v) {
        .integer => |i| i,
        else => error.BadWidth,
    };
}

fn width(v: std.json.Value) !u8 {
    const i = try integer(v);
    return switch (i) {
        2, 4, 6, 16 => @intCast(i),
        else => error.BadWidth,
    };
}

/// "double.<i>.<class>" or "single.<i>.<class>" with a known class.
fn blockKey(key: []const u8) !bool {
    var parts = std.mem.splitScalar(u8, key, '.');
    const kind = parts.next() orelse return false;
    const idx = parts.next() orelse return false;
    const class = parts.next() orelse return false;
    if (parts.next() != null) return false;
    _ = std.fmt.parseInt(usize, idx, 10) catch return false;
    if (std.mem.eql(u8, kind, "double")) {
        return std.meta.stringToEnum(zflux2_pack.Double, class) != null;
    }
    if (std.mem.eql(u8, kind, "single")) {
        return std.meta.stringToEnum(zflux2_pack.Single, class) != null;
    }
    return false;
}

test "a class sets every block and a block key overrides it" {
    const text =
        \\{"to_q": 4, "ff_in": 2, "qkv_mlp": 2,
        \\ "double.3.to_q": 6, "single.7.qkv_mlp": 16, "w4_qmax": 3}
    ;
    var map = try Map.parse(std.testing.allocator, text);
    defer map.deinit();
    try std.testing.expectEqual(@as(?u8, 4), map.widthDouble(0, .to_q));
    try std.testing.expectEqual(@as(?u8, 6), map.widthDouble(3, .to_q));
    try std.testing.expectEqual(@as(?u8, 2), map.widthDouble(4, .ff_in));
    try std.testing.expectEqual(@as(?u8, null), map.widthDouble(0, .to_v)); // silent: packer policy
    try std.testing.expectEqual(@as(?u8, 2), map.widthSingle(0, .qkv_mlp));
    try std.testing.expectEqual(@as(?u8, 16), map.widthSingle(7, .qkv_mlp));
    try std.testing.expectEqual(@as(?u8, null), map.widthSingle(7, .out));
    try std.testing.expectEqual(@as(f32, 3.0), map.w4_qmax);
}

fn rejects(comptime err: anyerror, text: []const u8) !void {
    try std.testing.expectError(err, Map.parse(std.testing.allocator, text));
}

test "the map rejects unknown keys, widths the loaders cannot decode and odd grids" {
    try rejects(error.UnknownKey, "{\"to_qq\": 4}");
    try rejects(error.UnknownKey, "{\"double.x.to_q\": 4}");
    try rejects(error.BadWidth, "{\"to_q\": 3}");
    try rejects(error.BadWidth, "{\"to_q\": \"4\"}");
    try rejects(error.BadQmax, "{\"w4_qmax\": 5}");
    try rejects(error.BadMap, "[4]");
}
