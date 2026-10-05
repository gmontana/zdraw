//! Image-fidelity metrics for the quality gate: PSNR, SSIM, MS-SSIM, and
//! cheap deterministic error/detail checks.
//!
//! Both operate on packed u8 RGB (HWC, the actually-displayed image), so the
//! gate measures perceptual closeness of a candidate generation to a golden one.
//! SSIM is a sliding 8x8 window, uniform weights, averaged over windows and the
//! three channels — enough to catch structural drift from a lossy kernel without
//! pulling in a Gaussian-window dependency. MS-SSIM repeats the structural
//! check over a downsampled pyramid. Edge PSNR compares Sobel-like luminance
//! gradients so detail loss does not hide behind average pixel scores. Dev-only,
//! like refcheck's stats.

const std = @import("std");

const win = 8;
const stride = 4;
const c1: f64 = 6.5025; // (0.01 * 255)^2
const c2: f64 = 58.5225; // (0.03 * 255)^2
const ms_weights = [_]f64{ 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 };

pub fn mse(a: []const u8, b: []const u8) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0;
    for (a, b) |x, y| {
        const d = @as(f64, @floatFromInt(x)) - @as(f64, @floatFromInt(y));
        sum += d * d;
    }
    if (a.len == 0) return 0;
    return sum / @as(f64, @floatFromInt(a.len));
}

pub fn mae(a: []const u8, b: []const u8) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @floatFromInt(absDiff(x, y));
    if (a.len == 0) return 0;
    return sum / @as(f64, @floatFromInt(a.len));
}

pub fn maxAbs(a: []const u8, b: []const u8) u8 {
    std.debug.assert(a.len == b.len);
    var out: u8 = 0;
    for (a, b) |x, y| out = @max(out, absDiff(x, y));
    return out;
}

pub fn p99Abs(a: []const u8, b: []const u8) u8 {
    std.debug.assert(a.len == b.len);
    var hist = [_]usize{0} ** 256;
    for (a, b) |x, y| hist[absDiff(x, y)] += 1;
    if (a.len == 0) return 0;
    const want = (a.len * 99 + 99) / 100;
    var seen: usize = 0;
    for (hist, 0..) |count, value| {
        seen += count;
        if (seen >= want) return @intCast(value);
    }
    return 255;
}

/// Peak signal-to-noise ratio in dB; identical images are capped at 99 dB.
pub fn psnr(a: []const u8, b: []const u8) f64 {
    const m = mse(a, b);
    if (m <= 1e-12) return 99.0;
    return 10.0 * std.math.log10(255.0 * 255.0 / m);
}

/// PSNR over grayscale edge magnitudes. This catches detail/edge loss that can
/// look acceptable under RGB PSNR/SSIM alone.
pub fn edgePsnr(a: []const u8, b: []const u8, width: usize, height: usize) f64 {
    if (width < 3 or height < 3) return psnr(a, b);
    var sum: f64 = 0;
    var count: usize = 0;
    var row: usize = 1;
    while (row + 1 < height) : (row += 1) {
        var col: usize = 1;
        while (col + 1 < width) : (col += 1) {
            const da = edgeMagnitude(a, width, row, col);
            const db = edgeMagnitude(b, width, row, col);
            const d = da - db;
            sum += d * d;
            count += 1;
        }
    }
    if (count == 0) return 99.0;
    const m = sum / @as(f64, @floatFromInt(count));
    if (m <= 1e-12) return 99.0;
    return 10.0 * std.math.log10(255.0 * 255.0 / m);
}

/// Mean SSIM over 8x8 windows across all three channels of a packed HWC image.
pub fn ssim(a: []const u8, b: []const u8, width: usize, height: usize) f64 {
    if (width < win or height < win) return globalSsim(a, b);
    var total: f64 = 0;
    var count: usize = 0;
    for (0..3) |ch| {
        var row: usize = 0;
        while (row + win <= height) : (row += stride) {
            var col: usize = 0;
            while (col + win <= width) : (col += stride) {
                total += windowSsim(a, b, width, ch, row, col);
                count += 1;
            }
        }
    }
    if (count == 0) return 1.0;
    return total / @as(f64, @floatFromInt(count));
}

/// Multi-scale SSIM over a simple 2x downsample pyramid. This is deterministic
/// and dependency-free; learned/perceptual metrics belong in the external
/// certification harness.
pub fn msSsim(allocator: std.mem.Allocator, a: []const u8, b: []const u8, width: usize, height: usize) !f64 {
    std.debug.assert(a.len == b.len);
    var owned_a: ?[]u8 = null;
    var owned_b: ?[]u8 = null;
    defer if (owned_a) |buf| allocator.free(buf);
    defer if (owned_b) |buf| allocator.free(buf);

    var cur_a = a;
    var cur_b = b;
    var w = width;
    var h = height;
    var score: f64 = 1.0;

    for (ms_weights, 0..) |weight, scale| {
        const parts = ssimParts(cur_a, cur_b, w, h);
        const term = if (scale + 1 == ms_weights.len or w < win * 2 or h < win * 2)
            @max(0.0, @min(1.0, parts.ssim))
        else
            @max(0.0, @min(1.0, parts.contrast_structure));
        score *= std.math.pow(f64, term, weight);

        if (scale + 1 == ms_weights.len or w < 2 or h < 2) break;

        const next_a = try downsample2x(allocator, cur_a, w, h);
        const next_b = try downsample2x(allocator, cur_b, w, h);
        if (owned_a) |buf| allocator.free(buf);
        if (owned_b) |buf| allocator.free(buf);
        owned_a = next_a;
        owned_b = next_b;
        cur_a = next_a;
        cur_b = next_b;
        w /= 2;
        h /= 2;
    }
    return score;
}

fn edgeMagnitude(img: []const u8, width: usize, row: usize, col: usize) f64 {
    const left = luma(img, width, row, col - 1);
    const right = luma(img, width, row, col + 1);
    const up = luma(img, width, row - 1, col);
    const down = luma(img, width, row + 1, col);
    const gx = right - left;
    const gy = down - up;
    return @sqrt(gx * gx + gy * gy);
}

fn luma(img: []const u8, width: usize, row: usize, col: usize) f64 {
    const i = (row * width + col) * 3;
    return 0.299 * @as(f64, @floatFromInt(img[i])) +
        0.587 * @as(f64, @floatFromInt(img[i + 1])) +
        0.114 * @as(f64, @floatFromInt(img[i + 2]));
}

fn absDiff(a: u8, b: u8) u8 {
    return if (a >= b) a - b else b - a;
}

fn windowSsim(a: []const u8, b: []const u8, width: usize, ch: usize, r0: usize, c0: usize) f64 {
    const parts = windowSsimParts(a, b, width, ch, r0, c0);
    return parts.ssim;
}

const SsimParts = struct {
    ssim: f64,
    contrast_structure: f64,
};

fn ssimParts(a: []const u8, b: []const u8, width: usize, height: usize) SsimParts {
    if (width < win or height < win) {
        const global = globalSsimParts(a, b);
        return .{ .ssim = global.ssim, .contrast_structure = global.contrast_structure };
    }
    var total_ssim: f64 = 0;
    var total_cs: f64 = 0;
    var count: usize = 0;
    for (0..3) |ch| {
        var row: usize = 0;
        while (row + win <= height) : (row += stride) {
            var col: usize = 0;
            while (col + win <= width) : (col += stride) {
                const parts = windowSsimParts(a, b, width, ch, row, col);
                total_ssim += parts.ssim;
                total_cs += parts.contrast_structure;
                count += 1;
            }
        }
    }
    if (count == 0) return .{ .ssim = 1.0, .contrast_structure = 1.0 };
    const denom: f64 = @floatFromInt(count);
    return .{ .ssim = total_ssim / denom, .contrast_structure = total_cs / denom };
}

fn windowSsimParts(a: []const u8, b: []const u8, width: usize, ch: usize, r0: usize, c0: usize) SsimParts {
    var sa: f64 = 0;
    var sb: f64 = 0;
    var saa: f64 = 0;
    var sbb: f64 = 0;
    var sab: f64 = 0;
    for (0..win) |dr| {
        for (0..win) |dc| {
            const i = ((r0 + dr) * width + (c0 + dc)) * 3 + ch;
            const x: f64 = @floatFromInt(a[i]);
            const y: f64 = @floatFromInt(b[i]);
            sa += x;
            sb += y;
            saa += x * x;
            sbb += y * y;
            sab += x * y;
        }
    }
    const n: f64 = win * win;
    const ma = sa / n;
    const mb = sb / n;
    return ssimFrom(ma, mb, saa / n - ma * ma, sbb / n - mb * mb, sab / n - ma * mb);
}

fn ssimFrom(ma: f64, mb: f64, va: f64, vb: f64, cov: f64) SsimParts {
    const luminance = (2.0 * ma * mb + c1) / (ma * ma + mb * mb + c1);
    const contrast_structure = (2.0 * cov + c2) / (va + vb + c2);
    return .{ .ssim = luminance * contrast_structure, .contrast_structure = contrast_structure };
}

fn downsample2x(allocator: std.mem.Allocator, img: []const u8, width: usize, height: usize) ![]u8 {
    const out_w = width / 2;
    const out_h = height / 2;
    var out = try allocator.alloc(u8, out_w * out_h * 3);
    for (0..out_h) |row| {
        for (0..out_w) |col| {
            for (0..3) |ch| {
                const idx00 = ((row * 2) * width + col * 2) * 3 + ch;
                const idx01 = idx00 + 3;
                const idx10 = (((row * 2) + 1) * width + col * 2) * 3 + ch;
                const idx11 = idx10 + 3;
                const sum: u16 = @as(u16, img[idx00]) + img[idx01] + img[idx10] + img[idx11];
                out[(row * out_w + col) * 3 + ch] = @intCast((sum + 2) / 4);
            }
        }
    }
    return out;
}

// Fallback for images smaller than one window: a single global SSIM term.
fn globalSsim(a: []const u8, b: []const u8) f64 {
    return globalSsimParts(a, b).ssim;
}

fn globalSsimParts(a: []const u8, b: []const u8) SsimParts {
    var sa: f64 = 0;
    var sb: f64 = 0;
    var saa: f64 = 0;
    var sbb: f64 = 0;
    var sab: f64 = 0;
    for (a, b) |xi, yi| {
        const x: f64 = @floatFromInt(xi);
        const y: f64 = @floatFromInt(yi);
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    const n: f64 = @floatFromInt(a.len);
    if (n == 0) return .{ .ssim = 1.0, .contrast_structure = 1.0 };
    const ma = sa / n;
    const mb = sb / n;
    return ssimFrom(ma, mb, saa / n - ma * ma, sbb / n - mb * mb, sab / n - ma * mb);
}

test "identical images score perfect" {
    var img: [8 * 8 * 3]u8 = undefined;
    for (&img, 0..) |*p, i| p.* = @truncate(i * 7);
    try std.testing.expectApproxEqAbs(99.0, psnr(&img, &img), 1e-9);
    try std.testing.expectApproxEqAbs(1.0, ssim(&img, &img, 8, 8), 1e-9);
    try std.testing.expectApproxEqAbs(1.0, try msSsim(std.testing.allocator, &img, &img, 8, 8), 1e-9);
    try std.testing.expectApproxEqAbs(0.0, mae(&img, &img), 1e-9);
    try std.testing.expectEqual(@as(u8, 0), p99Abs(&img, &img));
    try std.testing.expectEqual(@as(u8, 0), maxAbs(&img, &img));
    try std.testing.expectApproxEqAbs(99.0, edgePsnr(&img, &img, 8, 8), 1e-9);
}

test "psnr matches a known constant shift" {
    var a: [8 * 8 * 3]u8 = undefined;
    var b: [8 * 8 * 3]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = @truncate(100 + (i % 16));
    for (&b, &a) |*p, av| p.* = av + 16;
    // mse = 256 -> psnr = 10*log10(65025/256) = 24.05 dB
    try std.testing.expectApproxEqAbs(24.05, psnr(&a, &b), 0.1);
    try std.testing.expectApproxEqAbs(16.0, mae(&a, &b), 1e-9);
    try std.testing.expectEqual(@as(u8, 16), p99Abs(&a, &b));
    try std.testing.expectEqual(@as(u8, 16), maxAbs(&a, &b));
}

test "ssim drops on structural noise" {
    var a: [8 * 8 * 3]u8 = undefined;
    var b: [8 * 8 * 3]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for (&a, &b) |*pa, *pb| {
        pa.* = rng.int(u8);
        pb.* = rng.int(u8);
    }
    try std.testing.expect(ssim(&a, &b, 8, 8) < 0.5);
}
