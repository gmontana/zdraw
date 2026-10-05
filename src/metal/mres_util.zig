//! Small shared helpers for resident Metal attention.

const std = @import("std");

const gmode = @import("../runtime/gemm_mode.zig");
const tensor = @import("../pack/tensor.zig");

const tile = 32;
const k_step = 8;

pub fn dtype(dtype_in: tensor.DType) !u32 {
    return switch (dtype_in) {
        .f16 => 1,
        .bf16 => 2,
        .f32 => 3,
        else => error.UnsupportedDType,
    };
}

pub fn mode(mode_in: gmode.Mode) u32 {
    return switch (mode_in) {
        .half => 2,
        .w8 => 3,
        .w6 => 4,
        .exact => 1,
        .off => 0,
    };
}

/// The staged fast-GEMM admission dims (mirrors ours16_ok / ours16_ok_w16 in
/// metal_api.m): 32-aligned m/n/k. The dtype leg differs per entry (the
/// half-A kernel has no bf16 variant), so callers add their own dtype check.
pub fn ours16Dims(m: u32, k: u32, n: u32) bool {
    return m % 32 == 0 and n % 32 == 0 and k % 32 == 0;
}

pub fn fits(m: usize, k: usize, n: usize) bool {
    if (m == 0 or k == 0 or n == 0) return false;
    return m % tile == 0 and n % tile == 0 and k % k_step == 0;
}

pub fn toU32(value: usize) !u32 {
    if (value > std.math.maxInt(u32)) return error.InvalidShape;
    return @intCast(value);
}

pub fn toU64(value: usize) !u64 {
    return @intCast(value);
}
