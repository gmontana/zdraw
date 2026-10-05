//! f16 weight-view substitution from `.zpack` sidecars.
//!
//! W16 entries are plain little-endian f16 images of the original matrices,
//! marked by `group == 0`. Substituting the tensor view (dtype f16, sidecar
//! bytes) lets every downstream consumer — param dtype, MPS descriptors, raw
//! kernels — run the existing f16-weight path with half the weight traffic.
//! Enabled by `ZDRAW_STACK_W16=1`; only half-mode tensors are swapped, so the
//! exact profile never sees substituted weights. With `ZDRAW_REQUIRE_ZPACK=1`
//! a missing entry is an error instead of a silent f32 fallback.

const std = @import("std");

const env = @import("../runtime/env.zig");
const gmode = @import("../runtime/gemm_mode.zig");
const params = @import("../metal/mblock_chain_param.zig");
const tensor = @import("tensor.zig");
const zblock = @import("../zimage/zblock.zig");
const zpack_file = @import("zpack_file.zig");
const zpack_trace = @import("zpack_trace.zig");

pub const Options = struct {
    enabled: bool = true,
    strict: bool = false,

    pub fn fromEnv() Options {
        return .{
            .enabled = env.flag("ZDRAW_STACK_W16", true),
            .strict = env.flag("ZDRAW_REQUIRE_ZPACK", false),
        };
    }
};

pub fn enabled() bool {
    return Options.fromEnv().enabled;
}

pub fn substitute(
    opts: Options,
    sidecar: []const u8,
    modes: params.Modes,
    views: zblock.Views,
    family: zpack_file.Family,
    layer: u32,
    fused_shape: *[2]usize, // caller-owned backing for the fused view's shape
) !zblock.Views {
    if (!opts.enabled) return views;
    var out = views;
    out.q = try swap(opts, sidecar, modes.q, views.q, family, layer, .q);
    out.k = try swap(opts, sidecar, modes.k, views.k, family, layer, .k);
    out.v = try swap(opts, sidecar, modes.v, views.v, family, layer, .v);
    out.proj = try swap(opts, sidecar, modes.proj, views.proj, family, layer, .proj);
    out.ffn_gate = try swap(opts, sidecar, modes.ffn_gate, views.ffn_gate, family, layer, .ffn_gate);
    out.ffn_up = try swap(opts, sidecar, modes.ffn_up, views.ffn_up, family, layer, .ffn_up);
    out.ffn_down = try swap(opts, sidecar, modes.ffn_down, views.ffn_down, family, layer, .ffn_down);
    out.ffn_fused = try fused(sidecar, modes, views.ffn_gate, family, layer, fused_shape);
    return out;
}

// One [2*inner, hidden] f16 matrix replacing the gate+up pair when both run
// in half mode; absence is not an error (the split entries cover strictness).
fn fused(
    sidecar: []const u8,
    modes: params.Modes,
    gate: tensor.View,
    family: zpack_file.Family,
    layer: u32,
    shape: *[2]usize,
) !?tensor.View {
    if (modes.ffn_gate != .half or modes.ffn_up != .half) return null;
    const entry = try findW16(sidecar, family, layer, .ffn_gateup) orelse return null;
    if (gate.shape.len != 2) return error.InvalidShape;
    if (entry.rows != gate.shape[0] * 2 or entry.cols != gate.shape[1]) return error.InvalidShape;
    if (entry.bytes.len != entry.rows * entry.cols * 2) return error.InvalidShape;
    shape.* = .{ entry.rows, entry.cols };
    return .{
        .dtype = .f16,
        .shape = shape,
        .bytes = entry.bytes,
        .source = .{ .bytes = sidecar, .offset = entry.offset },
    };
}

fn swap(
    opts: Options,
    sidecar: []const u8,
    mode: gmode.Mode,
    view: tensor.View,
    family: zpack_file.Family,
    layer: u32,
    kind: zpack_file.Kind,
) !tensor.View {
    if (mode == .w6) return swapW6(sidecar, view, family, layer, kind);
    if (mode != .half) return view; // exact/W8 tensors keep their own paths
    const entry = try findW16(sidecar, family, layer, kind) orelse {
        zpack_trace.recordFallback(kind);
        if (opts.strict) return error.MissingPackedSidecar;
        return view;
    };
    if (view.shape.len != 2) return error.InvalidShape;
    if (entry.rows != view.shape[0] or entry.cols != view.shape[1]) return error.InvalidShape;
    if (entry.bytes.len != entry.rows * entry.cols * 2) return error.InvalidShape;
    zpack_trace.recordPacked(kind);
    return .{
        .dtype = .f16,
        .shape = view.shape,
        .bytes = entry.bytes,
        // Source points at the whole mmap'd sidecar so binding wraps it once
        // as a no-copy Metal buffer and addresses entries by offset.
        .source = .{ .bytes = sidecar, .offset = entry.offset },
    };
}

// 6-bit entry: same binding shape as W16; the kernel decodes (zw6.zig).
fn swapW6(
    sidecar: []const u8,
    view: tensor.View,
    family: zpack_file.Family,
    layer: u32,
    kind: zpack_file.Kind,
) !tensor.View {
    if (sidecar.len == 0) return missingW6(kind);
    const entry = try zpack_file.findInBits(sidecar, family, layer, kind, 6) orelse {
        return missingW6(kind);
    };
    if (view.shape.len != 2) return error.InvalidShape;
    if (entry.rows != view.shape[0] or entry.cols != view.shape[1]) return error.InvalidShape;
    zpack_trace.recordPacked(kind);
    // dtype is a placeholder; mode w6 sets the param dtype itself.
    const src = tensor.Source{ .bytes = sidecar, .offset = entry.offset };
    return .{ .dtype = .f32, .shape = view.shape, .bytes = entry.bytes, .source = src };
}

fn missingW6(kind: zpack_file.Kind) error{MissingPackedSidecar} {
    zpack_trace.recordFallback(kind);
    return error.MissingPackedSidecar;
}

fn findW16(
    sidecar: []const u8,
    family: zpack_file.Family,
    layer: u32,
    kind: zpack_file.Kind,
) !?zpack_file.Entry {
    if (sidecar.len == 0) return null;
    // Bits-aware scan: a W6/W8 twin of the same (family, layer, kind) earlier
    // in the file must not shadow the W16 image (entry order is build-defined).
    // entryBits == 16 iff group == 0, so this keeps the plain-f16 invariant.
    return try zpack_file.findInBits(sidecar, family, layer, kind, 16);
}

test "substitute swaps half tensors to sidecar f16 views" {
    const zpack = @import("zpack.zig");
    const allocator = std.testing.allocator;
    const w = [_]f32{ 1.0, -2.0, 0.5, 4.0 };
    const shape = [_]usize{ 2, 2 };
    const view = tensor.View{ .dtype = .f32, .shape = &shape, .bytes = std.mem.sliceAsBytes(&w) };
    var w16 = try zpack.packW16(allocator, view);
    defer w16.deinit(allocator);
    var file = try std.ArrayList(u8).initCapacity(allocator, 256);
    defer file.deinit(allocator);
    try zpack_file.append(allocator, &file, &.{.{
        .family = .main,
        .layer = 3,
        .kind = .q,
        .rows = 2,
        .cols = 2,
        .group = 0,
        .bytes = w16.bytes,
    }});

    const opts = Options{};
    const got = try swap(opts, file.items, .half, view, .main, 3, .q);
    try std.testing.expectEqual(tensor.DType.f16, got.dtype);
    try std.testing.expectEqual(@as(usize, 8), got.bytes.len);
    const kept = try swap(opts, file.items, .exact, view, .main, 3, .q);
    try std.testing.expectEqual(tensor.DType.f32, kept.dtype);
    const missing = try swap(opts, file.items, .half, view, .main, 9, .q);
    try std.testing.expectEqual(tensor.DType.f32, missing.dtype);
}

test "substitute rejects missing w6 sidecar" {
    const shape = [_]usize{ 2, 2 };
    const bytes = [_]u8{0} ** 16;
    const view = tensor.View{ .dtype = .f32, .shape = &shape, .bytes = &bytes };

    try std.testing.expectError(
        error.MissingPackedSidecar,
        swap(.{}, &.{}, .w6, view, .main, 0, .q),
    );
}
