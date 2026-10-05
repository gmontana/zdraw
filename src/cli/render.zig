//! Procedural images for CLI/PNG smoke tests, without model weights.
//! Returns owned RGB pixels, deterministic for the prompt, seed and dimensions.

const std = @import("std");

pub fn renderPreview(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    width: u32,
    height: u32,
    seed: u64,
) ![]u8 {
    const len = try std.math.mul(usize, @as(usize, width) * @as(usize, height), 3);
    const pixels = try allocator.alloc(u8, len);

    const h = mix(hash(prompt), seed);
    const scene = Scene.init(prompt, h);

    for (0..height) |y| {
        for (0..width) |x| {
            const fx = scale(x, width);
            const fy = scale(y, height);
            var rgb = backdrop(scene, fx, fy, h);
            rgb = sun(scene, rgb, fx, fy);
            rgb = hills(scene, rgb, fx, fy, h);
            rgb = waves(scene, rgb, fx, fy, h);
            rgb = boat(scene, rgb, fx, fy);

            const idx = (@as(usize, y) * width + x) * 3;
            pixels[idx + 0] = rgb[0];
            pixels[idx + 1] = rgb[1];
            pixels[idx + 2] = rgb[2];
        }
    }

    return pixels;
}

const Scene = struct {
    boat: bool,
    storm: bool,
    sunrise: bool,
    sky: [3]u8,
    water: [3]u8,
    warm: [3]u8,

    fn init(prompt: []const u8, h: u64) Scene {
        return .{
            .boat = hasText(prompt, "boat"),
            .storm = hasText(prompt, "storm") or hasText(prompt, "rain"),
            .sunrise = hasText(prompt, "sunrise") or hasText(prompt, "sunset"),
            .sky = color(h),
            .water = color(rot(h, 19)),
            .warm = .{ 244, 142, 65 },
        };
    }
};

fn backdrop(scene: Scene, x: f32, y: f32, h: u64) [3]u8 {
    const horizon: f32 = 0.58;
    const flicker = 0.08 * @sin(x * 19.0 + unit(h) * 6.2831855);
    if (y < horizon) {
        const high = if (scene.storm) [_]u8{ 30, 35, 54 } else scene.sky;
        const low = if (scene.sunrise) scene.warm else [_]u8{ 115, 139, 180 };
        return mixColor(high, low, std.math.clamp(y / horizon + flicker, 0.0, 1.0));
    }
    const t = std.math.clamp((y - horizon) / (1.0 - horizon), 0.0, 1.0);
    return mixColor(scene.water, .{ 18, 34, 54 }, t);
}

fn sun(scene: Scene, rgb: [3]u8, x: f32, y: f32) [3]u8 {
    const cy: f32 = if (scene.sunrise) 0.34 else 0.22;
    const d = distance(x - 0.74, y - cy);
    if (d > 0.13 or scene.storm) return rgb;
    return blend(rgb, .{ 255, 207, 104 }, 1.0 - d / 0.13);
}

fn hills(scene: Scene, rgb: [3]u8, x: f32, y: f32, h: u64) [3]u8 {
    const phase = unit(rot(h, 9)) * 6.2831855;
    const ridge = 0.48 + 0.10 * @sin(x * 9.0 + phase) + 0.05 * @sin(x * 23.0);
    if (y < ridge or y > 0.64) return rgb;
    const shade = if (scene.storm) [_]u8{ 16, 20, 31 } else [_]u8{ 35, 55, 70 };
    return blend(rgb, shade, std.math.clamp((y - ridge) * 9.0, 0.0, 1.0));
}

fn waves(scene: Scene, rgb: [3]u8, x: f32, y: f32, h: u64) [3]u8 {
    if (y < 0.58) return rgb;
    const line = @sin(x * 72.0 + y * 31.0 + unit(rot(h, 17)) * 6.2831855);
    if (line < 0.84) return rgb;
    const tint = if (scene.storm) [_]u8{ 160, 176, 190 } else [_]u8{ 214, 230, 238 };
    return blend(rgb, tint, 0.35);
}

fn boat(scene: Scene, rgb: [3]u8, x: f32, y: f32) [3]u8 {
    if (!scene.boat) return rgb;
    const dx = @abs(x - 0.48);
    const hull = y > 0.67 and y < 0.74 and dx < 0.24 - (y - 0.67) * 2.2;
    const mast = @abs(x - 0.49) < 0.008 and y > 0.38 and y < 0.70;
    const sail = x > 0.50 and x < 0.50 + (0.68 - y) * 0.42 and y > 0.40 and y < 0.66;
    if (sail or mast) return blend(rgb, .{ 236, 229, 207 }, 0.95);
    if (hull) return blend(rgb, .{ 164, 31, 28 }, 0.95);
    return rgb;
}

fn scale(value: usize, size: u32) f32 {
    if (size <= 1) return 0.0;
    return @as(f32, @floatFromInt(value)) / @as(f32, @floatFromInt(size - 1));
}

fn distance(x: f32, y: f32) f32 {
    return @sqrt(x * x + y * y);
}

fn blend(a: [3]u8, b: [3]u8, t: f32) [3]u8 {
    return mixColor(a, b, std.math.clamp(t, 0.0, 1.0));
}

fn mixColor(a: [3]u8, b: [3]u8, t: f32) [3]u8 {
    return .{
        mixChan(a[0], b[0], t),
        mixChan(a[1], b[1], t),
        mixChan(a[2], b[2], t),
    };
}

fn mixChan(a: u8, b: u8, t: f32) u8 {
    const af: f32 = @floatFromInt(a);
    const bf: f32 = @floatFromInt(b);
    return @intFromFloat(std.math.clamp(af * (1.0 - t) + bf * t, 0.0, 255.0));
}

fn color(value: u64) [3]u8 {
    return .{
        @intCast(48 + (value & 0x7f)),
        @intCast(48 + ((value >> 16) & 0x7f)),
        @intCast(96 + ((value >> 32) & 0x7f)),
    };
}

fn hash(bytes: []const u8) u64 {
    var h: u64 = 0xcbf2_9ce4_8422_2325;
    for (bytes) |byte| {
        h ^= byte;
        h *%= 0x0000_0100_0000_01b3;
    }
    return h;
}

fn mix(a: u64, b: u64) u64 {
    var x = a ^ (b +% 0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)) *% 0xbf58_476d_1ce4_e5b9;
    x = (x ^ (x >> 27)) *% 0x94d0_49bb_1331_11eb;
    return x ^ (x >> 31);
}

fn rot(x: u64, amount: u6) u64 {
    if (amount == 0) return x;
    return (x << amount) | (x >> @intCast(64 - @as(u7, amount)));
}

fn unit(value: u64) f32 {
    return @as(f32, @floatFromInt(value & 0xffff)) / 65_535.0;
}

fn hasText(hay: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or hay.len < needle.len) return false;
    for (0..hay.len - needle.len + 1) |i| {
        for (needle, 0..) |want, j| {
            const got = std.ascii.toLower(hay[i + j]);
            if (got != std.ascii.toLower(want)) break;
        } else return true;
    }
    return false;
}

test "preview is deterministic" {
    const a = try renderPreview(std.testing.allocator, "cat", 8, 8, 1);
    defer std.testing.allocator.free(a);
    const b = try renderPreview(std.testing.allocator, "cat", 8, 8, 1);
    defer std.testing.allocator.free(b);

    try std.testing.expectEqualSlices(u8, a, b);
}

test "preview has visible contrast" {
    const pixels = try renderPreview(std.testing.allocator, "a red boat at sunrise", 32, 32, 1);
    defer std.testing.allocator.free(pixels);

    var min: u8 = 255;
    var max: u8 = 0;
    for (pixels) |value| {
        min = @min(min, value);
        max = @max(max, value);
    }
    try std.testing.expect(max - min > 80);
}
