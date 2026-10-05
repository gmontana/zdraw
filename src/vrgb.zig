//! Convert VAE decoder samples to packed RGB.
//!
//! Diffusers postprocess maps decoder output from roughly [-1, 1] into
//! [0, 1], clamps it, and stores the image as channel-last pixels.

const std = @import("std");

pub fn run(out: []u8, sample: []const f32, width: u32, height: u32) !void {
    const pixels = @as(usize, width) * @as(usize, height);
    if (out.len != pixels * 3 or sample.len != pixels * 3) return error.InvalidShape;
    // Debug instrument for the sparse-exact-correction study: dump the
    // pre-quantization f32 samples so fast-vs-strict float error and
    // quantization-boundary proximity can be measured offline.
    if (std.c.getenv("ZDRAW_DUMP_RGBF")) |path| {
        dumpF32(std.mem.span(path), sample);
    }
    for (0..@as(usize, height)) |row| {
        for (0..@as(usize, width)) |col| {
            const dst = (row * width + col) * 3;
            out[dst + 0] = chan(sample[idx(0, row, col, width, height)]);
            out[dst + 1] = chan(sample[idx(1, row, col, width, height)]);
            out[dst + 2] = chan(sample[idx(2, row, col, width, height)]);
        }
    }
}

fn idx(ch: usize, row: usize, col: usize, width: u32, height: u32) usize {
    return (ch * @as(usize, height) + row) * @as(usize, width) + col;
}

fn dumpF32(path: []const u8, sample: []const f32) void {
    var pbuf: [256]u8 = undefined;
    const pz = std.fmt.bufPrintZ(&pbuf, "{s}", .{path}) catch return;
    const fh = std.c.fopen(pz.ptr, "wb") orelse return;
    defer _ = std.c.fclose(fh);
    const bytes = std.mem.sliceAsBytes(sample);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, fh);
}

fn chan(value: f32) u8 {
    const scaled = std.math.clamp(value * 0.5 + 0.5, 0.0, 1.0) * 255.0;
    return @intFromFloat(scaled + 0.5);
}

test "sample channels become packed rgb" {
    const sample = [_]f32{
        -1.0, 1.0,
        0.0,  2.0,
        1.0,  -2.0,
    };
    var out = [_]u8{0} ** 6;

    try run(&out, &sample, 2, 1);
    try std.testing.expectEqualSlices(u8, &.{ 0, 128, 255, 255, 255, 0 }, &out);
}
