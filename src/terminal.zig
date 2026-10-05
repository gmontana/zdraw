//! Terminal output from tightly packed RGB: inline images or ANSI half-blocks.
//! writeImage writes to stdout; renderAnsi returns owned bytes.

const std = @import("std");

const image = @import("image.zig");
const terminal_image = @import("terminal_image.zig");

pub const Options = struct {
    max_width: u32 = 80,
    protocol: terminal_image.Protocol = .auto,
    env: ?*const std.process.Environ.Map = null,
};

const Cells = struct {
    cols: u32,
    rows: u32,
};

const Pair = struct {
    top: [3]u8,
    bottom: [3]u8,
};

pub const Error = error{
    InvalidPixels,
    InvalidImageSize,
};

pub fn writeImage(
    io: std.Io,
    allocator: std.mem.Allocator,
    pixels: []const u8,
    width: u32,
    height: u32,
    options: Options,
) !void {
    const protocol = terminal_image.resolve(options.protocol, options.env);
    if (protocol != .ansi) {
        const png = try image.encodePng(allocator, pixels, width, height);
        defer allocator.free(png);
        const image_text = try terminal_image.render(allocator, protocol, png, .{
            .image_width = width,
            .image_height = height,
            .max_columns = options.max_width,
        });
        defer allocator.free(image_text);
        try std.Io.File.stdout().writeStreamingAll(io, image_text);
        try std.Io.File.stdout().writeStreamingAll(io, "\n");
        return;
    }

    const ansi = try renderAnsi(allocator, pixels, width, height, options);
    defer allocator.free(ansi);

    try std.Io.File.stdout().writeStreamingAll(io, ansi);
}

pub fn renderAnsi(
    allocator: std.mem.Allocator,
    pixels: []const u8,
    width: u32,
    height: u32,
    options: Options,
) ![]u8 {
    try checkPixels(pixels, width, height);
    const cells = fit(width, height, options.max_width);

    var out = try std.ArrayList(u8).initCapacity(allocator, ansiCap(cells));
    errdefer out.deinit(allocator);

    for (0..cells.rows) |row| {
        for (0..cells.cols) |col| {
            const pair = samplePair(pixels, width, height, cells, col, row);
            try appendCell(allocator, &out, pair);
        }
        try out.appendSlice(allocator, "\x1b[0m\n");
    }

    return out.toOwnedSlice(allocator);
}

fn checkPixels(pixels: []const u8, width: u32, height: u32) !void {
    if (width == 0 or height == 0) return error.InvalidImageSize;
    const count = try std.math.mul(usize, @as(usize, width), @as(usize, height));
    const need = try std.math.mul(usize, count, 3);
    if (pixels.len != need) return error.InvalidPixels;
}

fn fit(width: u32, height: u32, max_width: u32) Cells {
    const cols = @max(@as(u32, 1), @min(width, @max(@as(u32, 1), max_width)));
    const num = @as(u64, height) * cols + @as(u64, width) * 2 - 1;
    const rows = @max(@as(u32, 1), @min(height, @as(u32, @intCast(num / (width * 2)))));
    return .{ .cols = cols, .rows = rows };
}

fn ansiCap(cells: Cells) usize {
    return @as(usize, cells.cols) * @as(usize, cells.rows) * 48;
}

fn appendCell(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    pair: Pair,
) !void {
    var buffer: [64]u8 = undefined;
    const text = try std.fmt.bufPrint(
        &buffer,
        "\x1b[38;2;{d};{d};{d}m\x1b[48;2;{d};{d};{d}m\xe2\x96\x80",
        .{
            pair.top[0],    pair.top[1],    pair.top[2],
            pair.bottom[0], pair.bottom[1], pair.bottom[2],
        },
    );
    try out.appendSlice(allocator, text);
}

fn samplePair(
    pixels: []const u8,
    width: u32,
    height: u32,
    cells: Cells,
    col: usize,
    row: usize,
) Pair {
    return .{
        .top = sample(pixels, width, height, cells, col, row, 0),
        .bottom = sample(pixels, width, height, cells, col, row, 1),
    };
}

fn sample(
    pixels: []const u8,
    width: u32,
    height: u32,
    cells: Cells,
    col: usize,
    row: usize,
    half: usize,
) [3]u8 {
    const x = @min(width - 1, @as(u32, @intCast((col * width) / cells.cols)));
    const y_num = (row * 2 + half) * height;
    const y_den = @as(usize, cells.rows) * 2;
    const y = @min(height - 1, @as(u32, @intCast(y_num / y_den)));
    const idx = (@as(usize, y) * width + x) * 3;
    return .{ pixels[idx], pixels[idx + 1], pixels[idx + 2] };
}

test "renders rgb as ansi cells" {
    const pixels = [_]u8{ 255, 0, 0 };
    const ansi = try renderAnsi(std.testing.allocator, &pixels, 1, 1, .{});
    defer std.testing.allocator.free(ansi);

    try std.testing.expect(std.mem.indexOf(u8, ansi, "\x1b[38;2;255;0;0m") != null);
    try std.testing.expect(std.mem.endsWith(u8, ansi, "\x1b[0m\n"));
}

test "rejects mismatched pixels" {
    try std.testing.expectError(
        error.InvalidPixels,
        renderAnsi(std.testing.allocator, &.{ 1, 2 }, 1, 1, .{}),
    );
}
