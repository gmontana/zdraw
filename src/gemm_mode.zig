//! Precision mode for the FFN GEMM path.
//!
//! Shared by the linear context (which owns the compiled pipelines) and the
//! GEMM wrapper (which dispatches them), so neither has to import the other just
//! for this. `off` keeps the naive per-row GEMV; `exact` is bit-exact f32;
//! `half` is the lossy fast path gated by the image-quality suite.

const std = @import("std");

const c = @import("metal_c.zig");
const mgemm_shader = @import("mgemm_shader.zig");
const mw8_shader = @import("mw8_shader.zig");

pub const Mode = enum { off, exact, half, w8, w6 };

/// ZDRAW_GEMM: unset -> half (the gated fast default); "exact" opts out;
/// "0" -> off; unknown values keep the default.
pub fn fromEnv() Mode {
    // Production default is the gated fast path; ZDRAW_GEMM=exact opts out.
    const raw = std.c.getenv("ZDRAW_GEMM") orelse return .half;
    return parse(std.mem.span(raw), .half);
}

fn parse(value: []const u8, fallback: Mode) Mode {
    if (std.mem.eql(u8, value, "half")) return .half;
    if (std.mem.eql(u8, value, "0")) return .off;
    if (std.mem.eql(u8, value, "exact")) return .exact;
    return fallback;
}

pub fn compile(device: *anyopaque, comptime entry: [*:0]const u8, err: *[1024]u8) ?*anyopaque {
    return c.zdraw_metal_compile(device, mgemm_shader.gemm.ptr, entry, err, err.len);
}

pub fn compileW8(device: *anyopaque, err: *[1024]u8) ?*anyopaque {
    return c.zdraw_metal_compile(device, mw8_shader.source.ptr, "gemm_w8", err, err.len);
}

/// Fall back to a mode whose kernel actually compiled (else off).
pub fn resolve(want: Mode, exact_pipe: ?*anyopaque, half_pipe: ?*anyopaque) Mode {
    return switch (want) {
        .off => .off,
        .exact => if (exact_pipe != null) .exact else .off,
        .half => if (half_pipe != null) .half else if (exact_pipe != null) .exact else .off,
        .w8 => if (exact_pipe != null) .exact else .off,
        .w6 => if (half_pipe != null) .half else if (exact_pipe != null) .exact else .off,
    };
}

/// The pipeline for the active mode (null when off or the kernel is missing).
pub fn pipeline(mode: Mode, exact_pipe: ?*anyopaque, half_pipe: ?*anyopaque) ?*anyopaque {
    return switch (mode) {
        .off => null,
        .exact => exact_pipe,
        .half => half_pipe,
        .w8 => null,
        .w6 => half_pipe,
    };
}
