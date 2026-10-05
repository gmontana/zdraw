//! Load a list of Z-Image transformer block views.
//!
//! Views point into already-mapped safetensors files. This file only owns the
//! small slice that groups those views into layer order.

const std = @import("std");

const shards = @import("../pack/shards.zig");
const weight_index = @import("../pack/weight_index.zig");
const zblock = @import("zblock.zig");

pub const List = struct {
    items: []zblock.Views,

    pub fn deinit(self: *List, allocator: std.mem.Allocator) void {
        allocator.free(self.items);
        self.* = undefined;
    }
};

pub fn load(
    allocator: std.mem.Allocator,
    store: *const shards.Store,
    index: weight_index.Index,
    prefix: []const u8,
    count: usize,
    modulation: bool,
) !List {
    const items = try allocator.alloc(zblock.Views, count);
    errdefer allocator.free(items);

    for (items, 0..) |*item, layer| {
        item.* = try zblock.load(allocator, store, index, prefix, layer, modulation);
    }
    return .{ .items = items };
}

test "load empty block list" {
    var store: shards.Store = undefined;
    const index = weight_index.Index{ .total_size = 0, .entries = &.{} };

    var list = try load(std.testing.allocator, &store, index, "layers", 0, true);
    defer list.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), list.items.len);
}
