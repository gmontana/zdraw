//! Safetensors shard index loading.

const std = @import("std");

const max_index_bytes = 64 * 1024 * 1024;

pub const Entry = struct {
    name: []u8,
    file: []u8,

    fn deinit(self: Entry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.file);
    }
};

pub const Index = struct {
    total_size: u64,
    entries: []Entry,

    pub fn deinit(self: *Index, allocator: std.mem.Allocator) void {
        for (self.entries) |entry| entry.deinit(allocator);
        allocator.free(self.entries);
        self.* = undefined;
    }

    pub fn find(self: Index, name: []const u8) ?[]const u8 {
        for (self.entries) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.file;
        }
        return null;
    }
};

pub const Error = error{
    IndexTooLarge,
    InvalidIndex,
};

/// An index with no entries (a pack-only text store).
pub fn empty(allocator: std.mem.Allocator) !Index {
    return .{ .total_size = 0, .entries = try allocator.alloc(Entry, 0) };
}

pub fn read(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !Index {
    const bytes = try readFile(io, allocator, path);
    defer allocator.free(bytes);
    return parse(allocator, bytes);
}

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Index {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
    });
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidIndex;
    const object = parsed.value.object;
    const metadata = object.get("metadata") orelse return error.InvalidIndex;
    const weight_map = object.get("weight_map") orelse return error.InvalidIndex;
    if (metadata != .object or weight_map != .object) return error.InvalidIndex;

    const size_value = metadata.object.get("total_size") orelse return error.InvalidIndex;
    const total_size = try readU64(size_value);
    const entries = try parseMap(allocator, weight_map.object);
    return .{ .total_size = total_size, .entries = entries };
}

fn parseMap(
    allocator: std.mem.Allocator,
    map: std.json.ObjectMap,
) ![]Entry {
    var entries = try std.ArrayList(Entry).initCapacity(allocator, map.count());
    errdefer {
        for (entries.items) |entry| entry.deinit(allocator);
        entries.deinit(allocator);
    }

    var iter = map.iterator();
    while (iter.next()) |item| {
        if (item.value_ptr.* != .string) return error.InvalidIndex;
        try entries.append(allocator, .{
            .name = try allocator.dupe(u8, item.key_ptr.*),
            .file = try allocator.dupe(u8, item.value_ptr.string),
        });
    }

    return entries.toOwnedSlice(allocator);
}

fn readFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.size > max_index_bytes) return error.IndexTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));

    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

fn readU64(value: std.json.Value) !u64 {
    if (value != .integer or value.integer < 0) return error.InvalidIndex;
    return @intCast(value.integer);
}

test "parse shard index" {
    const json =
        \\{"metadata":{"total_size":12},"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}
    ;
    var index = try parse(std.testing.allocator, json);
    defer index.deinit(std.testing.allocator);

    const size: u64 = 12;
    try std.testing.expectEqual(size, index.total_size);
    try std.testing.expectEqualStrings("one.safetensors", index.find("a").?);
    try std.testing.expectEqualStrings("two.safetensors", index.find("b").?);
    try std.testing.expect(index.find("missing") == null);
}

test "reject malformed index" {
    try std.testing.expectError(
        error.InvalidIndex,
        parse(std.testing.allocator, "{}"),
    );
}
