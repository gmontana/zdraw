//! Small helpers for compiling Metal compute pipelines.

const c = @import("metal_c.zig");

pub fn required(
    device: *anyopaque,
    source: [*:0]const u8,
    entry: [*:0]const u8,
    err: *[1024]u8,
) !*anyopaque {
    return optional(device, source, entry, err) orelse error.MetalCompileFailed;
}

pub fn optional(
    device: *anyopaque,
    source: [*:0]const u8,
    entry: [*:0]const u8,
    err: *[1024]u8,
) ?*anyopaque {
    return c.zdraw_metal_compile(device, source, entry, err, err.len);
}

pub fn threadCount(max_threads: usize) usize {
    if (max_threads >= 256) return 256;
    if (max_threads >= 128) return 128;
    if (max_threads >= 64) return 64;
    if (max_threads >= 32) return 32;
    return 16;
}

pub fn optionalThreads(pipe: ?*anyopaque) usize {
    if (pipe) |p| return threadCount(c.zdraw_metal_pipeline_threads(p));
    return 0;
}
