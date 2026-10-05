//! "Fix a part" returns the user's photo, not a re-framed copy: the render
//! (which ran at a size the engine likes, over the photo's centre-cover
//! crop) is put back into the original at its own resolution, and only
//! inside the painted mask, with a feathered edge. Everything the brush did
//! not touch stays the original's pixels.
const std = @import("std");
const image = @import("image.zig");

pub const Result = struct {
    pixels: []u8, // RGB, original size
    width: u32,
    height: u32,
};

/// `rendered` is RGB at rw x rh; `original_path` and `mask_path` are files.
/// Caller frees `pixels`.
pub fn maskedInto(
    allocator: std.mem.Allocator,
    original_path: []const u8,
    mask_path: []const u8,
    rendered: []const u8,
    rw: usize,
    rh: usize,
) !Result {
    var orig = try image.readRgba(allocator, original_path);
    defer orig.deinit(allocator);
    var mask_img = try image.readRgba(allocator, mask_path);
    defer mask_img.deinit(allocator);
    const ow: usize = orig.width;
    const oh: usize = orig.height;
    const out = try allocator.alloc(u8, ow * oh * 3);
    errdefer allocator.free(out);

    // The render covers the centre crop of the original with the render's
    // aspect (init_image.loadRgb's cover transform); invert it.
    const fw: f64 = @floatFromInt(ow);
    const fh: f64 = @floatFromInt(oh);
    const scale = @max(@as(f64, @floatFromInt(rw)) / fw, @as(f64, @floatFromInt(rh)) / fh);
    const crop_w = @as(f64, @floatFromInt(rw)) / scale;
    const crop_h = @as(f64, @floatFromInt(rh)) / scale;
    const off_x = (fw - crop_w) / 2.0;
    const off_y = (fh - crop_h) / 2.0;

    const mask = try feathered(allocator, mask_img, ow, oh);
    defer allocator.free(mask);
    const map = Map{ .scale = scale, .off_x = off_x, .off_y = off_y, .rw = rw, .rh = rh };
    blend(out, orig.pixels, rendered, mask, ow, oh, map);
    return .{ .pixels = out, .width = @intCast(ow), .height = @intCast(oh) };
}

/// An instruction edit ("add a bird") redraws the whole frame at the size the
/// engine likes, so a three-megapixel photo would come back at half a
/// megapixel. This puts the edit back into the photo at the photo's own
/// resolution and keeps the photo's own pixels wherever nothing really
/// changed: the mask is the difference between the two, blurred so that
/// re-drawn texture does not count, only real changes. Null when almost
/// everything changed (a new light, a new season): then the render stands.
pub fn changedInto(
    allocator: std.mem.Allocator,
    original_path: []const u8,
    rendered: []const u8,
    rw: usize,
    rh: usize,
) !?Result {
    var orig = try image.readRgba(allocator, original_path);
    defer orig.deinit(allocator);
    const ow: usize = orig.width;
    const oh: usize = orig.height;
    if (ow == 0 or oh == 0) return null;
    const fw: f64 = @floatFromInt(ow);
    const fh: f64 = @floatFromInt(oh);
    const scale = @max(@as(f64, @floatFromInt(rw)) / fw, @as(f64, @floatFromInt(rh)) / fh);
    const map = Map{
        .scale = scale,
        .off_x = (fw - @as(f64, @floatFromInt(rw)) / scale) / 2.0,
        .off_y = (fh - @as(f64, @floatFromInt(rh)) / scale) / 2.0,
        .rw = rw,
        .rh = rh,
    };
    const mask = try diffMask(allocator, orig, rendered, ow, oh, map);
    defer allocator.free(mask);
    var sum: f64 = 0;
    for (mask) |m| sum += m;
    if (sum / @as(f64, @floatFromInt(mask.len)) > 0.85) return null;
    const out = try allocator.alloc(u8, ow * oh * 3);
    errdefer allocator.free(out);
    blend(out, orig.pixels, rendered, mask, ow, oh, map);
    return .{ .pixels = out, .width = @intCast(ow), .height = @intCast(oh) };
}

/// Where the edit really differs from the photo, 0..1. Measured on a coarse
/// grid (a sixteenth of the pixels) and blurred there, so a re-drawn grain of
/// sand counts for nothing and an added bird counts for everything.
fn diffMask(
    allocator: std.mem.Allocator,
    orig: image.Rgba,
    rendered: []const u8,
    ow: usize,
    oh: usize,
    map: Map,
) ![]f64 {
    const cw = @max(32, ow / 4);
    const ch = @max(32, oh / 4);
    const coarse = try allocator.alloc(f64, cw * ch);
    defer allocator.free(coarse);
    for (0..ch) |cy| {
        const y = @min(cy * oh / ch, oh - 1);
        const fy = (@as(f64, @floatFromInt(y)) + 0.5 - map.off_y) * map.scale - 0.5;
        for (0..cw) |cx| {
            const x = @min(cx * ow / cw, ow - 1);
            const fx = (@as(f64, @floatFromInt(x)) + 0.5 - map.off_x) * map.scale - 0.5;
            const src = (y * ow + x) * 4;
            var d: f64 = 0;
            for (0..3) |c| {
                const a: f64 = @floatFromInt(orig.pixels[src + c]);
                d = @max(d, @abs(a - sample(rendered, map.rw, map.rh, fx, fy, c)));
            }
            coarse[cy * cw + cx] = d;
        }
    }
    const diag = std.math.sqrt(@as(f64, @floatFromInt(cw * cw + ch * ch)));
    const radius: usize = @intFromFloat(@max(2.0, diag * 0.02));
    const tmp = try allocator.alloc(f64, cw * ch);
    defer allocator.free(tmp);
    for (0..2) |_| {
        blurX(coarse, tmp, cw, ch, radius);
        blurY(tmp, coarse, cw, ch, radius);
    }
    // 18/255 of difference is texture; 40/255 is a change. Between them it fades.
    for (coarse) |*v| v.* = std.math.clamp((v.* - 18.0) / 22.0, 0.0, 1.0);
    return expand(allocator, coarse, cw, ch, ow, oh);
}

/// The coarse mask at the photo's size (bilinear; it is already smooth).
fn expand(
    allocator: std.mem.Allocator,
    coarse: []const f64,
    cw: usize,
    ch: usize,
    ow: usize,
    oh: usize,
) ![]f64 {
    const out = try allocator.alloc(f64, ow * oh);
    errdefer allocator.free(out);
    const sy = @as(f64, @floatFromInt(ch)) / @as(f64, @floatFromInt(oh));
    const sx = @as(f64, @floatFromInt(cw)) / @as(f64, @floatFromInt(ow));
    for (0..oh) |y| {
        const gy = (@as(f64, @floatFromInt(y)) + 0.5) * sy - 0.5;
        const y0: usize = @intFromFloat(@max(0.0, @floor(gy)));
        const y1 = @min(y0 + 1, ch - 1);
        const ay = std.math.clamp(gy - @floor(gy), 0.0, 1.0);
        for (0..ow) |x| {
            const gx = (@as(f64, @floatFromInt(x)) + 0.5) * sx - 0.5;
            const x0: usize = @intFromFloat(@max(0.0, @floor(gx)));
            const x1 = @min(x0 + 1, cw - 1);
            const ax = std.math.clamp(gx - @floor(gx), 0.0, 1.0);
            const top = coarse[y0 * cw + x0] * (1 - ax) + coarse[y0 * cw + x1] * ax;
            const bot = coarse[y1 * cw + x0] * (1 - ax) + coarse[y1 * cw + x1] * ax;
            out[y * ow + x] = top * (1 - ay) + bot * ay;
        }
    }
    return out;
}

/// Where a pixel of the original lands in the render.
const Map = struct {
    scale: f64,
    off_x: f64,
    off_y: f64,
    rw: usize,
    rh: usize,
};

fn blend(
    out: []u8,
    orig: []const u8,
    rendered: []const u8,
    mask: []const f64,
    ow: usize,
    oh: usize,
    map: Map,
) void {
    for (0..oh) |y| {
        const yy: f64 = @floatFromInt(y);
        const fy = (yy + 0.5 - map.off_y) * map.scale - 0.5;
        for (0..ow) |x| {
            const o = (y * ow + x) * 3;
            const src = (y * ow + x) * 4;
            const m = mask[y * ow + x];
            if (m <= 0.001) {
                out[o] = orig[src];
                out[o + 1] = orig[src + 1];
                out[o + 2] = orig[src + 2];
                continue;
            }
            const xx: f64 = @floatFromInt(x);
            const fx = (xx + 0.5 - map.off_x) * map.scale - 0.5;
            for (0..3) |c| {
                const r = sample(rendered, map.rw, map.rh, fx, fy, c);
                const a: f64 = @floatFromInt(orig[src + c]);
                const v = a * (1.0 - m) + r * m;
                out[o + c] = @intFromFloat(std.math.clamp(v, 0.0, 255.0) + 0.5);
            }
        }
    }
}

/// The mask at the original's size, 0..1, with a soft edge (a box blur of
/// about 1% of the diagonal, run twice).
fn feathered(allocator: std.mem.Allocator, mask_img: image.Rgba, ow: usize, oh: usize) ![]f64 {
    const m = try allocator.alloc(f64, ow * oh);
    errdefer allocator.free(m);
    const mw: usize = mask_img.width;
    const mh: usize = mask_img.height;
    for (0..oh) |y| {
        const sy = @min(y * mh / oh, mh - 1);
        for (0..ow) |x| {
            const sx = @min(x * mw / ow, mw - 1);
            const p = (sy * mw + sx) * 4;
            const r0: f64 = @floatFromInt(mask_img.pixels[p]);
            const g0: f64 = @floatFromInt(mask_img.pixels[p + 1]);
            const b0: f64 = @floatFromInt(mask_img.pixels[p + 2]);
            const lum = (r0 + g0 + b0) / 3.0;
            m[y * ow + x] = lum / 255.0;
        }
    }
    const diag = std.math.sqrt(@as(f64, @floatFromInt(ow * ow + oh * oh)));
    const radius: usize = @intFromFloat(@max(2.0, diag * 0.01));
    const tmp = try allocator.alloc(f64, ow * oh);
    defer allocator.free(tmp);
    for (0..2) |_| {
        blurX(m, tmp, ow, oh, radius);
        blurY(tmp, m, ow, oh, radius);
    }
    return m;
}

fn blurX(src: []const f64, dst: []f64, w: usize, h: usize, r: usize) void {
    for (0..h) |y| {
        const row = y * w;
        for (0..w) |x| {
            const lo = x -| r;
            const hi = @min(x + r, w - 1);
            var acc: f64 = 0;
            for (lo..hi + 1) |i| acc += src[row + i];
            dst[row + x] = acc / @as(f64, @floatFromInt(hi - lo + 1));
        }
    }
}

fn blurY(src: []const f64, dst: []f64, w: usize, h: usize, r: usize) void {
    for (0..w) |x| {
        for (0..h) |y| {
            const lo = y -| r;
            const hi = @min(y + r, h - 1);
            var acc: f64 = 0;
            for (lo..hi + 1) |i| acc += src[i * w + x];
            dst[y * w + x] = acc / @as(f64, @floatFromInt(hi - lo + 1));
        }
    }
}

/// Bilinear sample of channel `c` from the rendered RGB image.
fn sample(rgb: []const u8, w: usize, h: usize, fx: f64, fy: f64, c: usize) f64 {
    const cx = std.math.clamp(fx, 0.0, @as(f64, @floatFromInt(w - 1)));
    const cy = std.math.clamp(fy, 0.0, @as(f64, @floatFromInt(h - 1)));
    const x0: usize = @intFromFloat(@floor(cx));
    const y0: usize = @intFromFloat(@floor(cy));
    const x1 = @min(x0 + 1, w - 1);
    const y1 = @min(y0 + 1, h - 1);
    const ax = cx - @as(f64, @floatFromInt(x0));
    const ay = cy - @as(f64, @floatFromInt(y0));
    const p = struct {
        fn at(px: []const u8, ww: usize, xx: usize, yy: usize, cc: usize) f64 {
            return @floatFromInt(px[(yy * ww + xx) * 3 + cc]);
        }
    };
    const top = p.at(rgb, w, x0, y0, c) * (1 - ax) + p.at(rgb, w, x1, y0, c) * ax;
    const bot = p.at(rgb, w, x0, y1, c) * (1 - ax) + p.at(rgb, w, x1, y1, c) * ax;
    return top * (1 - ay) + bot * ay;
}
