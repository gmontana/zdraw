//! Packed low-bit weight tiers in the resident executor: the W6 and W4
//! `.u8` marker views from zflux2_pack.swap bind without a copy and dispatch
//! the split-scales packed GEMM for their width. Lives beside
//! zflux2_resident.zig so that file stays inside its size ratchet.

const std = @import("std");

const metal_c = @import("metal_c.zig");
const zflux2_pack = @import("zflux2_pack.zig");
const zw2 = @import("zw2.zig");
const zw4 = @import("zw4.zig");
const zw6 = @import("zw6.zig");

/// GemmParams dtype codes the Metal side dispatches on: 4 = W6 (established
/// by the gemmbench gate), 5 = W4 (the same convention one width down).
pub const dtype_f16: u32 = 1;
pub const dtype_w6: u32 = 4;
pub const dtype_w4: u32 = 5;
/// 6 = W2 (steel route only: no f32-A split kernel exists for it).
pub const dtype_w2: u32 = 6;

/// A bound weight in the resident executor (the handle of the resident or
/// no-copy buffer, the byte offset of this matrix, its GemmParams dtype).
pub const WeightBind = struct {
    handle: *anyopaque,
    offset: u64 = 0,
    dtype: u32 = dtype_f16,
    // W6 only: absolute byte offset of the matrix's f16 scales region inside
    // the bound buffer (scales live after ALL rows' codes, so sub-row binds
    // cannot derive it from their own n).
    scales_off: u64 = 0,
};

pub fn isPacked(dtype: u32) bool {
    return dtype == dtype_w6 or dtype == dtype_w4 or dtype == dtype_w2;
}

/// The dtype code for a marker view's `packed_bits`.
pub fn dtypeFor(bits: u8) !u32 {
    return switch (bits) {
        6 => dtype_w6,
        4 => dtype_w4,
        2 => dtype_w2,
        else => error.InvalidDType,
    };
}

/// Byte offset of the f16 scales region for a whole matrix of this width.
pub fn scalesBase(bits: u8, rows: usize, cols: usize) !usize {
    return switch (bits) {
        6 => zw6.scalesBase(rows, cols, zflux2_pack.w6_group),
        4 => zw4.scalesBase(rows, cols, zflux2_pack.w6_group),
        2 => zw2.scalesBase(rows, cols, zflux2_pack.w6_group),
        else => error.InvalidDType,
    };
}

/// Packed code bytes per row for the dtype code (row-offset sub-binding).
pub fn codesPerRow(dtype: u32, cols: usize) usize {
    if (dtype == dtype_w4) return zw4.codesPerRow(cols, zflux2_pack.w6_group);
    if (dtype == dtype_w2) return zw2.codesPerRow(cols, zflux2_pack.w6_group);
    return zw6.codesPerRow(cols, zflux2_pack.w6_group);
}

/// Encode the packed GEMM for the dtype code into the live batch.
pub fn runEnc(
    dtype: u32,
    batch: *anyopaque,
    a: *anyopaque,
    w: *anyopaque,
    c: *anyopaque,
    p: *const metal_c.GemmParams,
    a_off: u64,
    c_off: u64,
    scales_off: u64,
) c_int {
    if (dtype == dtype_w2) return 1; // no f32-A split kernel: W2 runs on the f16-A steel route
    if (dtype == dtype_w4) {
        return metal_c.zdraw_metal_run_gemm_w4_enc(batch, a, w, c, p, a_off, c_off, scales_off);
    }
    return metal_c.zdraw_metal_run_gemm_w6_enc(batch, a, w, c, p, a_off, c_off, scales_off);
}

test "packed dtype codes and layouts follow the marker width" {
    try std.testing.expectEqual(dtype_w6, try dtypeFor(6));
    try std.testing.expectEqual(dtype_w4, try dtypeFor(4));
    try std.testing.expectEqual(dtype_w2, try dtypeFor(2));
    try std.testing.expectEqual(@as(usize, 3072 / 4), codesPerRow(dtype_w2, 3072));
    try std.testing.expectError(error.InvalidDType, dtypeFor(8));
    try std.testing.expect(isPacked(dtype_w4) and isPacked(dtype_w6) and !isPacked(1));
    // 3072 cols: W6 packs 4 codes in 3 bytes, W4 packs 2 per byte.
    try std.testing.expectEqual(@as(usize, 3072 / 4 * 3), codesPerRow(dtype_w6, 3072));
    try std.testing.expectEqual(@as(usize, 3072 / 2), codesPerRow(dtype_w4, 3072));
}

/// The f16-A route for a packed weight: the steel dequant kernel for its width
/// (f16 A x packed W^T -> f16, cast to the f32 C the consumers read). `batch`
/// is the executor's open command batch. Widths other
/// than W6/W4 are refused as before.
pub fn steel(
    batch: ?*anyopaque,
    a: *anyopaque,
    w: WeightBind,
    c: *anyopaque,
    m: usize,
    k: usize,
    n: usize,
    w_off: usize,
    a_off: u64,
    c_off: u64,
) !void {
    if (!isPacked(w.dtype)) return error.InvalidDType;
    // w_off is the f16 byte offset of the row sub-block the executor binds;
    // the packed codes and the scales of that sub-block start at row0.
    if (w_off % (k * 2) != 0) return error.InvalidShape;
    const row0 = w_off / (k * 2);
    const groups_per_row = (k + zflux2_pack.w6_group - 1) / zflux2_pack.w6_group;
    const p = metal_c.GemmParams{
        .m = @intCast(m),
        .k = @intCast(k),
        .n = @intCast(n),
        .dtype = w.dtype,
        .mode = 2,
        .weight_offset = w.offset + row0 * codesPerRow(w.dtype, k),
    };
    const scales_off = w.scales_off + row0 * groups_per_row * 2;
    const bt = batch orelse return error.MetalDispatchFailed;
    const rc = metal_c.zdraw_metal_run_gemm_steel_enc(
        bt,
        a,
        w.handle,
        c,
        &p,
        a_off,
        c_off,
        scales_off,
    );
    if (rc != 0) return error.MetalDispatchFailed;
}

test "steel refuses unpacked widths before touching Metal" {
    var dummy: u8 = 0;
    const d: *anyopaque = &dummy;
    const w = WeightBind{ .handle = d, .dtype = dtype_f16 };
    try std.testing.expectError(error.InvalidDType, steel(null, d, w, d, 64, 64, 64, 0, 0, 0));
}
