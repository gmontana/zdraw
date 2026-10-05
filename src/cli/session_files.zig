//! Numbered automatic saves never replace files from an earlier session.
const std = @import("std");

/// Write to the first unused number at or after `first`; caller owns the path.
/// Exclusive creation also prevents concurrent sessions selecting the same file.
pub fn writeNumbered(
    io: std.Io,
    allocator: std.mem.Allocator,
    directory: []const u8,
    first: u32,
    bytes: []const u8,
) ![]u8 {
    var index = first;
    while (true) {
        const path = try std.fmt.allocPrint(allocator, "{s}/{d}.png", .{ directory, index });
        errdefer allocator.free(path);
        const file = std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true }) catch |err| {
            if (err != error.PathAlreadyExists) return err;
            index = std.math.add(u32, index, 1) catch return error.OutputIndexExhausted;
            allocator.free(path);
            continue;
        };
        defer file.close(io);
        file.writeStreamingAll(io, bytes) catch |err| {
            try std.Io.Dir.cwd().deleteFile(io, path);
            return err;
        };
        return path;
    }
}

test "restarted sessions preserve existing images and skip reserved names" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const directory = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer allocator.free(directory);
    const first = try writeNumbered(io, allocator, directory, 1, "existing image");
    defer allocator.free(first);
    try tmp.dir.createDir(io, "2.png", .default_dir);
    const next = try writeNumbered(io, allocator, directory, 1, "new image");
    defer allocator.free(next);
    try std.testing.expect(std.mem.endsWith(u8, next, "/3.png"));
    const old = try tmp.dir.readFileAlloc(io, "1.png", allocator, .limited(100));
    defer allocator.free(old);
    try std.testing.expectEqualStrings("existing image", old);
    const new = try tmp.dir.readFileAlloc(io, "3.png", allocator, .limited(100));
    defer allocator.free(new);
    try std.testing.expectEqualStrings("new image", new);
}

test "exhausted numbering returns an error without replacing the last file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const directory = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer allocator.free(directory);
    const path = try writeNumbered(io, allocator, directory, std.math.maxInt(u32), "last image");
    defer allocator.free(path);
    try std.testing.expectError(
        error.OutputIndexExhausted,
        writeNumbered(io, allocator, directory, std.math.maxInt(u32), "replacement"),
    );
}

test "an invalid output directory returns its error and releases the path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const allocator = std.testing.allocator;
    const directory = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/missing",
        .{tmp.sub_path},
    );
    defer allocator.free(directory);
    try std.testing.expectError(
        error.FileNotFound,
        writeNumbered(std.testing.io, allocator, directory, 1, "image"),
    );
}
