//! PNG output from tightly packed RGB; ImageIO input returns owned RGBA8.
//! PNG chunks include CRCs and the stored-deflate stream includes Adler-32.

const std = @import("std");
const builtin = @import("builtin");

const png_sig = "\x89PNG\r\n\x1a\n";

const native_image = if (builtin.target.os.tag == .macos) struct {
    extern fn zdraw_image_decode_rgba8(
        path: [*:0]const u8,
        out_pixels: *?[*]u8,
        out_width: *u32,
        out_height: *u32,
    ) c_int;
    extern fn zdraw_image_free(ptr: ?*anyopaque) void;
} else struct {};

pub const ImageError = error{
    InvalidPixels,
    ImageTooLarge,
    ImageDecodeFailed,
    UnsupportedImageDecode,
};

pub const Rgba = struct {
    pixels: []u8,
    width: u32,
    height: u32,

    pub fn deinit(self: *Rgba, allocator: std.mem.Allocator) void {
        allocator.free(self.pixels);
        self.pixels = &.{};
        self.width = 0;
        self.height = 0;
    }
};

pub fn readRgba(
    allocator: std.mem.Allocator,
    path: []const u8,
) !Rgba {
    if (builtin.target.os.tag != .macos) return error.UnsupportedImageDecode;

    const c_path = try allocator.dupeZ(u8, path);
    defer allocator.free(c_path);

    var raw: ?[*]u8 = null;
    var width: u32 = 0;
    var height: u32 = 0;
    const rc = native_image.zdraw_image_decode_rgba8(
        c_path.ptr,
        &raw,
        &width,
        &height,
    );
    if (rc != 0 or raw == null or width == 0 or height == 0) {
        return error.ImageDecodeFailed;
    }
    defer native_image.zdraw_image_free(raw);

    const len = try rgbaLen(width, height);
    const pixels = try allocator.dupe(u8, raw.?[0..len]);
    unpremultiply(pixels);
    return .{ .pixels = pixels, .width = width, .height = height };
}

pub fn writePng(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    pixels: []const u8,
    width: u32,
    height: u32,
) !void {
    return writePngText(io, allocator, path, pixels, width, height, &.{});
}

pub fn writePngText(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    pixels: []const u8,
    width: u32,
    height: u32,
    texts: []const Text,
) !void {
    _ = try writePngHashed(io, allocator, path, pixels, width, height, texts);
}

/// writePngText that also returns the sha256 (lower-case hex) of the bytes it
/// wrote: the artifact's file hash for the render receipt.
pub fn writePngHashed(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    pixels: []const u8,
    width: u32,
    height: u32,
    texts: []const Text,
) ![64]u8 {
    const bytes = try encodePngText(allocator, pixels, width, height, texts);
    defer allocator.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});

    try ensureParent(io, path);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buffer: [4096]u8 = [_]u8{0} ** 4096;
    var writer = file.writerStreaming(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn ensureParent(io: std.Io, path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    if (parent.len == 0 or std.mem.eql(u8, parent, ".")) return;
    // createDirPath fails with NotDir when an existing component is a symlink
    // (macOS /tmp -> private/tmp), so only create when the parent doesn't open.
    var dir = std.Io.Dir.cwd().openDir(io, parent, .{}) catch {
        try std.Io.Dir.cwd().createDirPath(io, parent);
        return;
    };
    dir.close(io);
}

/// A PNG tEXt chunk: a Latin-1 keyword (1-79 bytes) and its text.
pub const Text = struct {
    keyword: []const u8,
    text: []const u8,
};

pub fn encodePng(
    allocator: std.mem.Allocator,
    pixels: []const u8,
    width: u32,
    height: u32,
) ![]u8 {
    return encodePngText(allocator, pixels, width, height, &.{});
}

/// The same PNG with tEXt chunks between IHDR and IDAT (the recipe; the
/// machine-readable marking of a generated image). Chunk order and content
/// are deterministic, so the file bytes are a pure function of the inputs.
pub fn encodePngText(
    allocator: std.mem.Allocator,
    pixels: []const u8,
    width: u32,
    height: u32,
    texts: []const Text,
) ![]u8 {
    const pixel_len = try pixelLen(width, height);
    if (pixels.len != pixel_len) return error.InvalidPixels;

    var out = try std.ArrayList(u8).initCapacity(allocator, pixel_len + 512);
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, png_sig);
    try appendIhdr(allocator, &out, width, height);
    for (texts) |t| try appendText(allocator, &out, t);
    try appendIdat(allocator, &out, pixels, width, height);
    try appendChunk(allocator, &out, "IEND", &.{});

    return out.toOwnedSlice(allocator);
}

/// SHA-256 of the raw RGB bytes as lowercase hex: the certified hash of an
/// image, invariant to the PNG's metadata chunks.
pub fn pixelSha256(pixels: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(pixels, &digest, .{});
    const table = "0123456789abcdef";
    var hex: [64]u8 = undefined;
    for (digest, 0..) |byte, i| {
        hex[2 * i] = table[byte >> 4];
        hex[2 * i + 1] = table[byte & 0x0f];
    }
    return hex;
}

fn appendText(allocator: std.mem.Allocator, out: *std.ArrayList(u8), t: Text) !void {
    if (t.keyword.len == 0 or t.keyword.len > 79) return error.InvalidKeyword;
    const data = try allocator.alloc(u8, t.keyword.len + 1 + t.text.len);
    defer allocator.free(data);
    @memcpy(data[0..t.keyword.len], t.keyword);
    data[t.keyword.len] = 0;
    @memcpy(data[t.keyword.len + 1 ..], t.text);
    try appendChunk(allocator, out, "tEXt", data);
}

fn appendIhdr(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    width: u32,
    height: u32,
) !void {
    var data: [13]u8 = [_]u8{0} ** 13;
    std.mem.writeInt(u32, data[0..4], width, .big);
    std.mem.writeInt(u32, data[4..8], height, .big);
    data[8] = 8;
    data[9] = 2;
    data[10] = 0;
    data[11] = 0;
    data[12] = 0;
    try appendChunk(allocator, out, "IHDR", &data);
}

fn appendIdat(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    pixels: []const u8,
    width: u32,
    height: u32,
) !void {
    const scanline = try checkedMul(usizeU32(width), 3);
    const row_len = try checkedAdd(scanline, 1);
    const raw_len = try checkedMul(usizeU32(height), row_len);

    const filtered = try allocator.alloc(u8, raw_len);
    defer allocator.free(filtered);

    for (0..height) |row| {
        const dst = row * row_len;
        const src = row * scanline;
        filtered[dst] = 0;
        @memcpy(filtered[dst + 1 ..][0..scanline], pixels[src..][0..scanline]);
    }

    const zlib = try zlibStore(allocator, filtered);
    defer allocator.free(zlib);
    try appendChunk(allocator, out, "IDAT", zlib);
}

fn appendChunk(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    tag: *const [4]u8,
    data: []const u8,
) !void {
    var header: [8]u8 = [_]u8{0} ** 8;
    std.mem.writeInt(u32, header[0..4], @intCast(data.len), .big);
    @memcpy(header[4..8], tag);

    try out.appendSlice(allocator, &header);
    try out.appendSlice(allocator, data);

    var crc = Crc32.init();
    crc.update(tag);
    crc.update(data);

    var tail: [4]u8 = [_]u8{0} ** 4;
    std.mem.writeInt(u32, &tail, crc.final(), .big);
    try out.appendSlice(allocator, &tail);
}

fn zlibStore(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const max_block: usize = 65_535;
    const blocks = (data.len + max_block - 1) / max_block;
    const deflate_len = try checkedAdd(data.len, blocks * 5);
    const out = try allocator.alloc(u8, try checkedAdd(deflate_len, 6));

    out[0] = 0x78;
    out[1] = 0x01;

    var src: usize = 0;
    var dst: usize = 2;
    while (src < data.len or (data.len == 0 and dst == 2)) {
        const remaining: usize = data.len - src;
        const take: usize = @min(remaining, max_block);
        const final: u8 = if (src + take == data.len) 1 else 0;
        out[dst] = final;
        out[dst + 1] = @intCast(take & 0xff);
        out[dst + 2] = @intCast((take >> 8) & 0xff);
        out[dst + 3] = ~out[dst + 1];
        out[dst + 4] = ~out[dst + 2];
        @memcpy(out[dst + 5 ..][0..take], data[src..][0..take]);
        src += take;
        dst += 5 + take;
        if (data.len == 0) break;
    }

    std.mem.writeInt(u32, out[dst..][0..4], adler32(data), .big);
    return out;
}

fn pixelLen(width: u32, height: u32) !usize {
    return checkedMul(try checkedMul(usizeU32(width), usizeU32(height)), 3);
}

fn rgbaLen(width: u32, height: u32) !usize {
    return checkedMul(try checkedMul(usizeU32(width), usizeU32(height)), 4);
}

fn usizeU32(value: u32) usize {
    return @intCast(value);
}

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.ImageTooLarge;
}

fn checkedAdd(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.ImageTooLarge;
}

fn adler32(data: []const u8) u32 {
    // Reduce once per 5552-byte run (the largest run whose sums fit u32),
    // not twice per byte; same result, ~10x fewer divisions on a 3 MB PNG.
    const nmax = 5552;
    var s1: u32 = 1;
    var s2: u32 = 0;
    var rest = data;
    while (rest.len > 0) {
        const run = rest[0..@min(rest.len, nmax)];
        for (run) |byte| {
            s1 += byte;
            s2 += s1;
        }
        s1 %= 65_521;
        s2 %= 65_521;
        rest = rest[run.len..];
    }
    return (s2 << 16) | s1;
}

test "adler32 matches the byte-at-a-time reference" {
    var data: [20_000]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(&data);
    var s1: u32 = 1;
    var s2: u32 = 0;
    for (data) |byte| {
        s1 = (s1 + byte) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    try std.testing.expectEqual((s2 << 16) | s1, adler32(&data));
    try std.testing.expectEqual(@as(u32, 1), adler32(""));
}

fn unpremultiply(pixels: []u8) void {
    var i: usize = 0;
    while (i + 3 < pixels.len) : (i += 4) {
        const alpha = pixels[i + 3];
        if (alpha == 0 or alpha == 255) continue;
        const a: u32 = alpha;
        pixels[i] = unpremulChannel(pixels[i], a);
        pixels[i + 1] = unpremulChannel(pixels[i + 1], a);
        pixels[i + 2] = unpremulChannel(pixels[i + 2], a);
    }
}

fn unpremulChannel(value: u8, alpha: u32) u8 {
    const wide: u32 = value;
    const expanded = (wide * 255 + alpha / 2) / alpha;
    return @intCast(@min(expanded, 255));
}

const Crc32 = struct {
    value: u32 = 0xffff_ffff,

    fn init() Crc32 {
        return .{};
    }

    fn update(self: *Crc32, data: []const u8) void {
        for (data) |byte| {
            var v = self.value ^ byte;
            for (0..8) |_| {
                v = if (v & 1 == 1) (v >> 1) ^ 0xedb8_8320 else v >> 1;
            }
            self.value = v;
        }
    }

    fn final(self: Crc32) u32 {
        return self.value ^ 0xffff_ffff;
    }
};

test "encode png writes signature and ihdr" {
    const pixels = [_]u8{ 255, 0, 0 };
    const bytes = try encodePng(std.testing.allocator, &pixels, 1, 1);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, png_sig, bytes[0..8]);
    const ihdr_len: u32 = 13;
    try std.testing.expectEqual(ihdr_len, std.mem.readInt(u32, bytes[8..12], .big));
    try std.testing.expectEqualSlices(u8, "IHDR", bytes[12..16]);
}

test "checks pixel length" {
    try std.testing.expectError(
        error.InvalidPixels,
        encodePng(std.testing.allocator, &.{ 0, 0 }, 1, 1),
    );
}

test "checksums match known values" {
    const empty_adler: u32 = 1;
    const abc_adler: u32 = 0x024d_0127;
    try std.testing.expectEqual(empty_adler, adler32(""));
    try std.testing.expectEqual(abc_adler, adler32("abc"));

    var crc = Crc32.init();
    crc.update("IEND");
    const iend_crc: u32 = 0xae42_6082;
    try std.testing.expectEqual(iend_crc, crc.final());
}

test "unpremultiply expands alpha-weighted channels" {
    var pixels = [_]u8{ 64, 32, 16, 128 };
    unpremultiply(&pixels);
    const r: u8 = 128;
    const g: u8 = 64;
    const b: u8 = 32;
    const a: u8 = 128;
    try std.testing.expectEqual(r, pixels[0]);
    try std.testing.expectEqual(g, pixels[1]);
    try std.testing.expectEqual(b, pixels[2]);
    try std.testing.expectEqual(a, pixels[3]);
}

test "png write/read preserves row orientation" {
    if (builtin.target.os.tag != .macos) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/orient.png",
        .{tmp.sub_path},
    );
    defer std.testing.allocator.free(path);
    const pixels = [_]u8{
        255, 0,   0,
        0,   255, 0,
        0,   0,   255,
        255, 255, 255,
    };
    try writePng(std.testing.io, std.testing.allocator, path, &pixels, 2, 2);
    var rgba = try readRgba(std.testing.allocator, path);
    defer rgba.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u8, 255), rgba.pixels[0]);
    try std.testing.expectEqual(@as(u8, 0), rgba.pixels[1]);
    try std.testing.expectEqual(@as(u8, 0), rgba.pixels[2]);
    try std.testing.expectEqual(@as(u8, 255), rgba.pixels[3]);
    try std.testing.expectEqual(@as(u8, 0), rgba.pixels[8]);
    try std.testing.expectEqual(@as(u8, 0), rgba.pixels[9]);
    try std.testing.expectEqual(@as(u8, 255), rgba.pixels[10]);
    try std.testing.expectEqual(@as(u8, 255), rgba.pixels[11]);
}
