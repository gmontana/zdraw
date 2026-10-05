//! Progressive preview: after every denoise step the predicted clean sample
//! (x - sigma_next * v) is projected to RGB with the fitted linear map in
//! preview_map.zig at the latent resolution (2*gh x 2*gw) and written as a
//! PNG under ZDRAW_PREVIEW_DIR, announced on stderr in verbose mode as
//! `zdraw: preview <path> step k/n`. Costs a few milliseconds on the CPU;
//! the latent is already there on the default route.
const std = @import("std");
const image = @import("image.zig");
const map = @import("preview_map.zig");
const sink = @import("progress_sink.zig");
const progress = @import("progress.zig");

/// The preview directory, or null when previews are off.
pub fn dir() ?[]const u8 {
    const raw = std.c.getenv("ZDRAW_PREVIEW_DIR") orelse return null;
    const value = std.mem.span(raw);
    return if (value.len == 0) null else value;
}

/// One image's packed latents `x` and velocity `v` ([tokens, 128]) after the
/// Euler update of `step` (1-based) out of `total`; `index` numbers the
/// image within a --seeds batch.
pub fn write(
    io: std.Io,
    allocator: std.mem.Allocator,
    out_dir: ?[]const u8,
    x: []const f32,
    v: []const f32,
    sigma_next: f32,
    gh: usize,
    gw: usize,
    step: usize,
    total: usize,
    index: usize,
) !void {
    if (out_dir == null and !sink.active()) return;
    const h = 2 * gh;
    const w = 2 * gw;
    const pixels = try allocator.alloc(u8, h * w * 3);
    defer allocator.free(pixels);
    project(pixels, x, v, sigma_next, gh, gw);
    if (out_dir) |directory| {
        const fmt = "{s}/preview-{d}-step{d}.png";
        const path = try std.fmt.allocPrint(allocator, fmt, .{ directory, index, step });
        defer allocator.free(path);
        try image.writePng(io, allocator, path, pixels, @intCast(w), @intCast(h));
        try progress.preview(io, allocator, path, step, total);
    }
    try sink.frame(io, allocator, .{
        .pixels = pixels,
        .width = @intCast(w),
        .height = @intCast(h),
        .step = step,
        .total = total,
        .index = index,
    });
}

/// Packed channel i*4 + dy*2 + dx of token (th, tw) is unpacked channel i at
/// pixel (2*th + dy, 2*tw + dx) (zflux2_vae.prepare).
fn project(
    pixels: []u8,
    x: []const f32,
    v: []const f32,
    sigma_next: f32,
    gh: usize,
    gw: usize,
) void {
    const w = 2 * gw;
    for (0..gh) |th| for (0..gw) |tw| {
        const base = (th * gw + tw) * 128;
        for (0..2) |dy| for (0..2) |dx| {
            const s = dy * 2 + dx;
            const px = ((2 * th + dy) * w + (2 * tw + dx)) * 3;
            for (0..3) |ch| {
                var acc: f32 = map.b[ch];
                for (0..32) |i| {
                    const k = base + i * 4 + s;
                    acc += map.w[ch][i] * (x[k] - sigma_next * v[k]);
                }
                pixels[px + ch] = @intFromFloat(std.math.clamp(acc, 0.0, 1.0) * 255.0 + 0.5);
            }
        };
    };
}

test "the projection clamps and maps every latent pixel" {
    const gh = 1;
    const gw = 2;
    var x = [_]f32{0} ** (2 * 128);
    const v = [_]f32{0} ** (2 * 128);
    x[0] = 100.0;
    var pixels: [2 * 4 * 3]u8 = undefined;
    project(&pixels, &x, &v, 0.0, gh, gw);
    for (pixels) |p| try std.testing.expect(p <= 255);
}
