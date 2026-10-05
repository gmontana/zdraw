//! Tests for the `.zpack` sidecar container.

const std = @import("std");

const zpack_file = @import("zpack_file.zig");

test "write and find W8 entry" {
    var out = try std.ArrayList(u8).initCapacity(std.testing.allocator, 64);
    defer out.deinit(std.testing.allocator);
    const payload = [_]u8{ 1, 2, 3, 4 };
    try zpack_file.append(std.testing.allocator, &out, &.{.{
        .layer = 7,
        .kind = .ffn_down,
        .rows = 2,
        .cols = 2,
        .group = 64,
        .bytes = &payload,
    }});
    const got = (try zpack_file.find(out.items, 7, .ffn_down)).?;
    try std.testing.expectEqual(@as(u32, 2), got.rows);
    try std.testing.expectEqual(@as(usize, 0), got.offset % 16);
    try std.testing.expectEqualSlices(u8, &payload, got.bytes);
    try std.testing.expect(try zpack_file.find(out.items, 8, .ffn_down) == null);
}

test "map sidecar bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/weights.zpack",
        .{tmp.sub_path},
    );
    defer std.testing.allocator.free(path);

    var out = try std.ArrayList(u8).initCapacity(std.testing.allocator, 64);
    defer out.deinit(std.testing.allocator);
    const payload = [_]u8{ 5, 6, 7, 8 };
    try zpack_file.append(std.testing.allocator, &out, &.{.{
        .layer = 3,
        .kind = .ffn_down,
        .rows = 1,
        .cols = 4,
        .group = 64,
        .bytes = &payload,
    }});
    try writeFile(path, out.items);
    var mapped = try zpack_file.open(std.testing.io, path);
    defer mapped.deinit(std.testing.io);
    const got = (try zpack_file.find(mapped.bytes(), 3, .ffn_down)).?;
    try std.testing.expectEqual(@as(usize, 0), got.offset % 16);
    try std.testing.expectEqualSlices(u8, &payload, got.bytes);
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
    defer file.close(std.testing.io);
    var buffer: [256]u8 = undefined;
    var writer = file.writerStreaming(std.testing.io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}
