//! Terminal inline-image escape protocols.

const std = @import("std");

pub const Protocol = enum {
    auto,
    ansi,
    kitty,
    iterm,
};

pub const Options = struct {
    image_width: u32,
    image_height: u32,
    max_columns: u32,
    viewport_rows: ?u32 = null,
    image_id: u32 = 0,
};

const chunk_size = 4096;

pub fn resolve(
    wanted: Protocol,
    env: ?*const std.process.Environ.Map,
) Protocol {
    if (wanted != .auto) return wanted;
    const map = env orelse return .ansi;
    if (map.contains("KITTY_WINDOW_ID")) return .kitty;
    if (map.contains("GHOSTTY_RESOURCES_DIR")) return .kitty;
    if (map.contains("WEZTERM_PANE")) return .kitty;
    if (map.contains("KONSOLE_VERSION")) return .kitty;
    if (map.contains("ITERM_SESSION_ID")) return .iterm;
    if (map.get("TERM_PROGRAM")) |value| {
        if (std.mem.eql(u8, value, "iTerm.app")) return .iterm;
    }
    return .ansi;
}

pub fn render(
    allocator: std.mem.Allocator,
    protocol: Protocol,
    png: []const u8,
    options: Options,
) ![]u8 {
    if (options.viewport_rows) |rows| {
        return renderViewport(allocator, protocol, png, options, rows);
    }
    const encoded = try encode64(allocator, png);
    defer allocator.free(encoded);

    return switch (protocol) {
        .kitty => renderKitty(allocator, encoded, displayColumns(options)),
        .iterm => renderIterm(allocator, png.len, encoded, displayPixels(options)),
        .auto, .ansi => error.UnsupportedProtocol,
    };
}

fn renderViewport(
    allocator: std.mem.Allocator,
    protocol: Protocol,
    png: []const u8,
    options: Options,
    rows: u32,
) ![]u8 {
    const encoded = try encode64(allocator, png);
    defer allocator.free(encoded);
    if (protocol == .iterm) return std.fmt.allocPrint(
        allocator,
        "\x1b]1337;File=inline=1;width={d};height={d};preserveAspectRatio=1;size={d}:{s}\x07",
        .{ options.max_columns, rows, png.len, encoded },
    );
    if (protocol != .kitty) return error.UnsupportedProtocol;
    var out = try std.ArrayList(u8).initCapacity(allocator, encoded.len + 256);
    errdefer out.deinit(allocator);
    var offset: usize = 0;
    while (offset < encoded.len) {
        const take = @min(chunk_size, encoded.len - offset);
        const more = offset + take < encoded.len;
        if (offset == 0) {
            const header = try std.fmt.allocPrint(
                allocator,
                "\x1b_Ga=T,f=100,q=2,C=1,i={d},c={d},r={d},m={d};",
                .{ options.image_id, options.max_columns, rows, @intFromBool(more) },
            );
            defer allocator.free(header);
            try out.appendSlice(allocator, header);
        } else try kittyMore(allocator, &out, more);
        try out.appendSlice(allocator, encoded[offset..][0..take]);
        try out.appendSlice(allocator, "\x1b\\");
        offset += take;
    }
    return out.toOwnedSlice(allocator);
}

fn encode64(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
    _ = std.base64.standard.Encoder.encode(out, bytes);
    return out;
}

fn renderKitty(
    allocator: std.mem.Allocator,
    encoded: []const u8,
    columns: u32,
) ![]u8 {
    var out = try std.ArrayList(u8).initCapacity(allocator, encoded.len + 128);
    errdefer out.deinit(allocator);

    var offset: usize = 0;
    while (offset < encoded.len) {
        const take = @min(chunk_size, encoded.len - offset);
        const more = offset + take < encoded.len;
        if (offset == 0) {
            try kittyStart(allocator, &out, more, columns);
        } else {
            try kittyMore(allocator, &out, more);
        }
        try out.appendSlice(allocator, encoded[offset..][0..take]);
        try out.appendSlice(allocator, "\x1b\\");
        offset += take;
    }

    return out.toOwnedSlice(allocator);
}

fn kittyStart(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    more: bool,
    columns: u32,
) !void {
    const flag: u8 = if (more) '1' else '0';
    var buffer: [64]u8 = undefined;
    const text = try std.fmt.bufPrint(&buffer, "\x1b_Ga=T,f=100,c={d},m=", .{columns});
    try out.appendSlice(allocator, text);
    try out.append(allocator, flag);
    try out.append(allocator, ';');
}

fn kittyMore(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    more: bool,
) !void {
    const flag: u8 = if (more) '1' else '0';
    try out.appendSlice(allocator, "\x1b_Gm=");
    try out.append(allocator, flag);
    try out.append(allocator, ';');
}

fn renderIterm(
    allocator: std.mem.Allocator,
    byte_len: usize,
    encoded: []const u8,
    pixels: u32,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "\x1b]1337;File=inline=1;width={d}px;size={d}:{s}\x07",
        .{ pixels, byte_len, encoded },
    );
}

fn displayColumns(options: Options) u32 {
    const natural = @max(@as(u32, 1), (options.image_width + 7) / 8);
    return @max(@as(u32, 1), @min(natural, options.max_columns));
}

fn displayPixels(options: Options) u32 {
    const max_pixels = @max(@as(u32, 1), options.max_columns) * 8;
    return @max(@as(u32, 1), @min(options.image_width, max_pixels));
}

test "detect terminal protocols" {
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();

    try env.put("KITTY_WINDOW_ID", "1");
    try std.testing.expectEqual(Protocol.kitty, resolve(.auto, &env));
    _ = env.swapRemove("KITTY_WINDOW_ID");

    try env.put("TERM_PROGRAM", "iTerm.app");
    try std.testing.expectEqual(Protocol.iterm, resolve(.auto, &env));
}

test "render inline protocols" {
    const opts = Options{ .image_width = 512, .image_height = 512, .max_columns = 80 };
    const kitty = try render(std.testing.allocator, .kitty, "abc", opts);
    defer std.testing.allocator.free(kitty);
    try std.testing.expect(std.mem.indexOf(u8, kitty, "\x1b_Ga=T,f=100,c=64") != null);
    try std.testing.expect(std.mem.indexOf(u8, kitty, "YWJj") != null);

    const iterm = try render(std.testing.allocator, .iterm, "abc", opts);
    defer std.testing.allocator.free(iterm);
    try std.testing.expect(std.mem.indexOf(u8, iterm, "width=512px;size=3:YWJj") != null);
}

test "viewport images reserve dimensions and kitty finishes chunked transmission" {
    const options = Options{
        .image_width = 128,
        .image_height = 128,
        .max_columns = 40,
        .viewport_rows = 20,
        .image_id = 42,
    };
    const iterm = try render(std.testing.allocator, .iterm, "abc", options);
    defer std.testing.allocator.free(iterm);
    try std.testing.expectEqualStrings(
        "\x1b]1337;File=inline=1;width=40;height=20;preserveAspectRatio=1;size=3:YWJj\x07",
        iterm,
    );
    const data: [3075]u8 = @splat(0);
    const kitty = try render(std.testing.allocator, .kitty, &data, options);
    defer std.testing.allocator.free(kitty);
    const header = "\x1b_Ga=T,f=100,q=2,C=1,i=42,c=40,r=20,m=1;";
    try std.testing.expect(std.mem.startsWith(u8, kitty, header));
    try std.testing.expect(std.mem.endsWith(u8, kitty, "\x1b\\\x1b_Gm=0;AAAA\x1b\\"));
    try std.testing.expectError(
        error.UnsupportedProtocol,
        render(std.testing.allocator, .ansi, "abc", options),
    );
}
