const std = @import("std");

const image = @import("image.zig");

test "encode png with multiple deflate blocks" {
    const pixels = try std.testing.allocator.alloc(u8, 256 * 256 * 3);
    defer std.testing.allocator.free(pixels);
    @memset(pixels, 128);

    const bytes = try image.encodePng(std.testing.allocator, pixels, 256, 256);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", bytes[0..8]);
}
