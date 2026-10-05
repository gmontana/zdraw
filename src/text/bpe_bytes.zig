//! GPT/Qwen byte-level alphabet.

const std = @import("std");

pub fn symbol(allocator: std.mem.Allocator, byte: u8) ![]u8 {
    var buf: [4]u8 = undefined;
    const len = try std.unicode.utf8Encode(codepoint(byte), &buf);
    return allocator.dupe(u8, buf[0..len]);
}

fn codepoint(byte: u8) u21 {
    if (direct(byte)) return byte;
    var cp: u21 = 256;
    var cur: u16 = 0;
    while (cur < byte) : (cur += 1) {
        const other: u8 = @intCast(cur);
        if (!direct(other)) cp += 1;
    }
    return cp;
}

fn direct(byte: u8) bool {
    return (byte >= '!' and byte <= '~') or
        (byte >= 0xa1 and byte <= 0xac) or byte >= 0xae;
}

test "byte mapping matches qwen byte-level alphabet" {
    const space = try symbol(std.testing.allocator, ' ');
    defer std.testing.allocator.free(space);
    const newline = try symbol(std.testing.allocator, '\n');
    defer std.testing.allocator.free(newline);

    try std.testing.expectEqualStrings("Ġ", space);
    try std.testing.expectEqualStrings("Ċ", newline);
}
