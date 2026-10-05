//! Session output bridge for Stanza-backed prompts.

const std = @import("std");
const stanza = @import("stanza");

pub const Output = struct {
    io: std.Io,
    editor: ?*stanza.Editor = null,

    pub fn write(self: *Output, allocator: std.mem.Allocator, text: []const u8) !void {
        const ed = self.editor orelse return writeRaw(self.io, text);
        if (!hasBareLf(text)) return ed.printAbove(text);

        const fixed = try crlf(allocator, text);
        defer allocator.free(fixed);
        try ed.printAbove(fixed);
    }
};

fn writeRaw(io: std.Io, text: []const u8) !void {
    try std.Io.File.stdout().writeStreamingAll(io, text);
}

fn hasBareLf(text: []const u8) bool {
    for (text, 0..) |byte, i| {
        if (byte == '\n' and (i == 0 or text[i - 1] != '\r')) return true;
    }
    return false;
}

fn crlf(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var extra: usize = 0;
    for (text, 0..) |byte, i| {
        if (byte == '\n' and (i == 0 or text[i - 1] != '\r')) extra += 1;
    }

    var out = try std.ArrayList(u8).initCapacity(allocator, text.len + extra);
    errdefer out.deinit(allocator);
    for (text, 0..) |byte, i| {
        if (byte == '\n' and (i == 0 or text[i - 1] != '\r')) {
            try out.append(allocator, '\r');
        }
        try out.append(allocator, byte);
    }
    return try out.toOwnedSlice(allocator);
}

test "crlf leaves existing carriage returns alone" {
    const got = try crlf(std.testing.allocator, "a\nb\r\n");
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("a\r\nb\r\n", got);
}
