//! Cheap final-frame sanity gates for product-facing image output.

const std = @import("std");

pub const Stats = struct {
    min_luma: u8,
    max_luma: u8,
    mean_luma: f32,

    pub fn range(self: Stats) u8 {
        return self.max_luma - self.min_luma;
    }
};

pub fn analyzeRgb(pixels: []const u8, width: u32, height: u32) !Stats {
    const expected = try rgbLen(width, height);
    if (pixels.len != expected) return error.InvalidPixels;

    var min_luma: u8 = 255;
    var max_luma: u8 = 0;
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < pixels.len) : (i += 3) {
        const y = luminance(pixels[i], pixels[i + 1], pixels[i + 2]);
        min_luma = @min(min_luma, y);
        max_luma = @max(max_luma, y);
        sum += y;
    }

    const count = try pixelCount(width, height);
    return .{
        .min_luma = min_luma,
        .max_luma = max_luma,
        .mean_luma = @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(count)),
    };
}

pub fn isBlankLike(stats: Stats) bool {
    if (stats.range() <= 6) return true;
    return stats.mean_luma >= 250.0 and stats.range() <= 24;
}

fn luminance(r: u8, g: u8, b: u8) u8 {
    const rr: u32 = r;
    const gg: u32 = g;
    const bb: u32 = b;
    return @intCast((rr * 77 + gg * 150 + bb * 29) >> 8);
}

fn pixelCount(width: u32, height: u32) !usize {
    return std.math.mul(usize, @as(usize, width), @as(usize, height)) catch error.ImageTooLarge;
}

fn rgbLen(width: u32, height: u32) !usize {
    return std.math.mul(usize, try pixelCount(width, height), 3) catch error.ImageTooLarge;
}

test "white image is blank-like" {
    const pixels = [_]u8{255} ** (4 * 4 * 3);
    const stats = try analyzeRgb(&pixels, 4, 4);
    try std.testing.expect(isBlankLike(stats));
}

test "white background with product contrast is accepted" {
    var pixels = [_]u8{255} ** (4 * 4 * 3);
    pixels[0] = 120;
    pixels[1] = 40;
    pixels[2] = 60;
    const stats = try analyzeRgb(&pixels, 4, 4);
    try std.testing.expect(!isBlankLike(stats));
}
