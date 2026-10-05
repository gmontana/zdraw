//! FLUX.2 Klein W16 sidecar slot map and view substitution.

const std = @import("std");

const tensor = @import("../pack/tensor.zig");
const util = @import("../cli/session_util.zig");
const zflux2 = @import("zflux2.zig");
const tensor_file = @import("../pack/tensor_file.zig");
const zpack_file = @import("../pack/zpack_file.zig");
const zpack_trace = @import("../pack/zpack_trace.zig");
const zw6 = @import("../pack/zw6.zig");
const zw2 = @import("../pack/zw2.zig");
const zw4 = @import("../pack/zw4.zig");

/// The one W6 group size the resident GEMM dispatch is pinned to.
pub const w6_group: usize = 64;

const family = zpack_file.Family.flux2;
const kind = zpack_file.Kind.flux2_weight;

pub const Global = enum(u32) {
    x_embed = 0,
    context_embed = 1,
    mod_img = 2,
    mod_txt = 3,
    mod_single = 4,
    time_in_1 = 5,
    time_in_2 = 6,
    norm_out = 7,
    proj_out = 8,
};

pub const Double = enum(u32) {
    to_q = 0,
    to_k = 1,
    to_v = 2,
    add_q = 3,
    add_k = 4,
    add_v = 5,
    to_out = 6,
    to_add_out = 7,
    ff_in = 8,
    ff_out = 9,
    ffc_in = 10,
    ffc_out = 11,
};

pub const Single = enum(u32) {
    qkv_mlp = 0,
    out = 1,
};

pub const global_weights = .{
    .{ .field = "x_embed", .slot = Global.x_embed },
    .{ .field = "context_embed", .slot = Global.context_embed },
    .{ .field = "proj_out", .slot = Global.proj_out },
};

pub const double_weights = .{
    .{ .field = "to_q", .slot = Double.to_q },
    .{ .field = "to_k", .slot = Double.to_k },
    .{ .field = "to_v", .slot = Double.to_v },
    .{ .field = "add_q", .slot = Double.add_q },
    .{ .field = "add_k", .slot = Double.add_k },
    .{ .field = "add_v", .slot = Double.add_v },
    .{ .field = "to_out", .slot = Double.to_out },
    .{ .field = "to_add_out", .slot = Double.to_add_out },
    .{ .field = "ff_in", .slot = Double.ff_in },
    .{ .field = "ff_out", .slot = Double.ff_out },
    .{ .field = "ffc_in", .slot = Double.ffc_in },
    .{ .field = "ffc_out", .slot = Double.ffc_out },
};

pub const single_weights = .{
    .{ .field = "qkv_mlp", .slot = Single.qkv_mlp },
    .{ .field = "out", .slot = Single.out },
};

/// Checkpoint tensor-name stems per slot, without the trailing ".weight".
/// These mirror the names zflux2.load reads, and LoRA adapters key off the
/// same stems (diffusers publishes "<stem>.lora_A.weight"), so one table
/// serves both the base weights and every adapter that targets them.
pub fn globalStem(g: Global) ?[]const u8 {
    return switch (g) {
        .x_embed => "x_embedder",
        .context_embed => "context_embedder",
        .proj_out => "proj_out",
        // Modulation/time weights are not GEMM-substituted and no published
        // adapter targets them; excluded rather than guessed.
        else => null,
    };
}

pub fn doubleStem(d: Double) []const u8 {
    return switch (d) {
        .to_q => "transformer_blocks.{d}.attn.to_q",
        .to_k => "transformer_blocks.{d}.attn.to_k",
        .to_v => "transformer_blocks.{d}.attn.to_v",
        .add_q => "transformer_blocks.{d}.attn.add_q_proj",
        .add_k => "transformer_blocks.{d}.attn.add_k_proj",
        .add_v => "transformer_blocks.{d}.attn.add_v_proj",
        .to_out => "transformer_blocks.{d}.attn.to_out.0",
        .to_add_out => "transformer_blocks.{d}.attn.to_add_out",
        .ff_in => "transformer_blocks.{d}.ff.linear_in",
        .ff_out => "transformer_blocks.{d}.ff.linear_out",
        .ffc_in => "transformer_blocks.{d}.ff_context.linear_in",
        .ffc_out => "transformer_blocks.{d}.ff_context.linear_out",
    };
}

pub fn singleStem(s: Single) []const u8 {
    return switch (s) {
        .qkv_mlp => "single_transformer_blocks.{d}.attn.to_qkv_mlp_proj",
        .out => "single_transformer_blocks.{d}.attn.to_out",
    };
}

/// Raw (verbatim) entries: the kinds and the dtype code carried in `group`.
pub const raw_kind = zpack_file.Kind.flux2_raw;
pub const norm_kind = zpack_file.Kind.flux2_norm;

pub fn rawGroup(dtype: tensor.DType) u32 {
    const code: u32 = switch (dtype) {
        .f32 => 1,
        .f16 => 2,
        .bf16 => 3,
        .u8 => 4,
    };
    return code << 16;
}

pub fn rawDType(group: u32) ?tensor.DType {
    return switch (group >> 16) {
        1 => .f32,
        2 => .f16,
        3 => .bf16,
        4 => .u8,
        else => null,
    };
}

/// Per-block norm weights ride under norm_kind in the block slot space:
/// doubles carry four (norm_q, norm_k, norm_added_q, norm_added_k), singles two.
pub fn normSlotDouble(layer: usize, idx: u32) u32 {
    return 100 + @as(u32, @intCast(layer)) * 12 + idx;
}

pub fn normSlotSingle(layer: usize, idx: u32) u32 {
    return 1000 + @as(u32, @intCast(layer)) * 2 + idx;
}

pub fn globalSlot(g: Global) u32 {
    return @intFromEnum(g);
}

pub fn doubleSlot(layer: usize, d: Double) u32 {
    return 100 + @as(u32, @intCast(layer)) * 12 + @intFromEnum(d);
}

pub fn singleSlot(layer: usize, s: Single) u32 {
    return 1000 + @as(u32, @intCast(layer)) * 2 + @intFromEnum(s);
}

pub fn swapLoaded(sidecar: []const u8, loaded: *zflux2.Loaded) !void {
    // Only resident GEMM weights move to W16. CPU-side modulation/time/final
    // norm math stays on the original bf16/f32 views; swapping those changes
    // scalar-side rounding and drifts the image.
    inline for (global_weights) |entry| {
        @field(loaded.globals, entry.field) =
            try swap(sidecar, globalSlot(entry.slot), @field(loaded.globals, entry.field));
    }

    for (loaded.doubles, 0..) |*blk, i| {
        inline for (double_weights) |entry| {
            @field(blk.*, entry.field) =
                try swap(sidecar, doubleSlot(i, entry.slot), @field(blk.*, entry.field));
        }
    }

    for (loaded.singles, 0..) |*blk, i| {
        inline for (single_weights) |entry| {
            @field(blk.*, entry.field) =
                try swap(sidecar, singleSlot(i, entry.slot), @field(blk.*, entry.field));
        }
    }
}

pub fn swap(sidecar: []const u8, slot: u32, view: tensor.View) !tensor.View {
    if (sidecar.len == 0) return fallback(view);
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 16)) |found| {
        if (view.shape.len != 2) return error.InvalidShape;
        if (found.rows != view.shape[0] or found.cols != view.shape[1]) return error.InvalidShape;
        if (found.bytes.len != @as(usize, found.rows) * found.cols * 2) return error.InvalidShape;
        zpack_trace.recordPacked(kind);
        return .{
            .dtype = .f16,
            .shape = view.shape,
            .bytes = found.bytes,
            .source = .{ .bytes = sidecar, .offset = found.offset },
        };
    }
    // The .u8 packed marker-view CONTRACT: a W6 or W4 entry is returned as
    // dtype .u8 with source set and packed_bits = 6 or 4, carrying the packed
    // codes+scales bytes (zw6 / zw4 layout, group w6_group). The ONLY
    // consumer that understands it is zflux2_resident.Ctx.weight, which
    // no-copy binds it and dispatches the split-scales packed GEMM for that
    // width; every other consumer fails loudly (assertFloatView / dtypeCode
    // -> error) instead of reinterpreting. tensor.View.check()/atF32* are
    // invalid on these views by design.
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 6)) |found| {
        return packedView(sidecar, view, found, 6, zw6.byteLen(found.rows, found.cols, w6_group));
    }
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 4)) |found| {
        return packedView(sidecar, view, found, 4, zw4.byteLen(found.rows, found.cols, w6_group));
    }
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 2)) |found| {
        return packedView(sidecar, view, found, 2, zw2.byteLen(found.rows, found.cols, w6_group));
    }
    return fallback(view);
}

fn packedView(
    sidecar: []const u8,
    view: tensor.View,
    found: zpack_file.Entry,
    bits: u8,
    want_len: usize,
) !tensor.View {
    if (view.shape.len != 2) return error.InvalidShape;
    if (found.rows != view.shape[0] or found.cols != view.shape[1]) return error.InvalidShape;
    if (zpack_file.entryGroupSize(found) != w6_group) return error.InvalidShape;
    if (found.bytes.len != want_len) return error.InvalidShape;
    zpack_trace.recordPacked(kind);
    return .{
        .dtype = .u8,
        .shape = view.shape,
        .bytes = found.bytes,
        .source = .{ .bytes = sidecar, .offset = found.offset },
        .packed_bits = bits,
    };
}

fn fallback(view: tensor.View) !tensor.View {
    zpack_trace.recordFallback(kind);
    if (requireSidecar()) return error.MissingPackedSidecar;
    return view;
}

/// The bit width a sidecar packs its block linears at (16, 6, 4 or 2), read
/// from single block 0's fused projection; 16 when the sidecar has none.
/// The tier value sidecarBits returns for a mixed-allocation pack (the single
/// blocks and the double v/out classes at different widths); no real width.
pub const mixed_bits: u8 = 9;

pub fn sidecarBits(sidecar: []const u8) !u8 {
    if (sidecar.len == 0) return 16;
    const singles = (try slotBits(sidecar, singleSlot(0, .qkv_mlp))) orelse 16;
    const v = (try slotBits(sidecar, doubleSlot(0, .to_v))) orelse singles;
    if (singles != 16 and v != singles) return mixed_bits;
    return singles;
}

/// The width of one packed slot, or null when the sidecar has no entry for it.
fn slotBits(sidecar: []const u8, slot: u32) !?u8 {
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 16)) |_| return 16;
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 6)) |_| return 6;
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 4)) |_| return 4;
    if (try zpack_file.findInBits(sidecar, family, slot, kind, 2)) |_| return 2;
    return null;
}

/// Does the sidecar carry the globals and norms a shard-less load needs?
pub fn hasRaw(sidecar: []const u8) bool {
    const slot = globalSlot(.mod_img);
    const found = zpack_file.findIn(sidecar, family, slot, raw_kind) catch return false;
    return found != null;
}

/// Build a Loaded from a `--globals` sidecar alone: block linears as packed
/// (or W16) views, globals and per-block norms as verbatim views. Every view
/// borrows the sidecar's bytes, so the sidecar must outlive the Loaded.
pub fn loadFromSidecar(
    allocator: std.mem.Allocator,
    sidecar: []const u8,
    cfg: zflux2.Config,
) !zflux2.Loaded {
    const n_views = 9 + cfg.double_layers * 16 + cfg.single_layers * 4;
    const shapes = try allocator.alloc(usize, n_views * 2);
    errdefer allocator.free(shapes);
    var cursor = ShapeCursor{ .buf = shapes };
    // SAFETY: the field loop initializes every global before the value is returned.
    var globals: zflux2.Globals = undefined;
    inline for (std.meta.fields(zflux2.Globals)) |f| {
        const g = @field(Global, f.name);
        @field(globals, f.name) = try rawView(sidecar, globalSlot(g), raw_kind, &cursor);
    }
    const doubles = try allocator.alloc(zflux2.Double, cfg.double_layers);
    errdefer allocator.free(doubles);
    for (doubles, 0..) |*d, i| {
        inline for (std.meta.fields(Double)) |f| {
            const slot = doubleSlot(i, @field(Double, f.name));
            @field(d.*, f.name) = try linearView(sidecar, slot, &cursor);
        }
        const norms = .{ "norm_q", "norm_k", "norm_added_q", "norm_added_k" };
        inline for (norms, 0..) |name, idx| {
            @field(d.*, name) = try rawView(sidecar, normSlotDouble(i, idx), norm_kind, &cursor);
        }
    }
    const singles = try allocator.alloc(zflux2.Single, cfg.single_layers);
    errdefer allocator.free(singles);
    for (singles, 0..) |*sgl, i| {
        inline for (std.meta.fields(Single)) |f| {
            const slot = singleSlot(i, @field(Single, f.name));
            @field(sgl.*, f.name) = try linearView(sidecar, slot, &cursor);
        }
        const norms = .{ "norm_q", "norm_k" };
        inline for (norms, 0..) |name, idx| {
            @field(sgl.*, name) = try rawView(sidecar, normSlotSingle(i, idx), norm_kind, &cursor);
        }
    }
    return .{
        .files = .{ .items = try allocator.alloc(tensor_file.Mapped, 0) },
        .cfg = cfg,
        .globals = globals,
        .doubles = doubles,
        .singles = singles,
        .raw_shapes = shapes,
    };
}

const ShapeCursor = struct {
    buf: []usize,
    used: usize = 0,

    fn take(self: *ShapeCursor, n: usize) ![]usize {
        if (self.used + n > self.buf.len) return error.InvalidShape;
        const out = self.buf[self.used .. self.used + n];
        self.used += n;
        return out;
    }
};

/// A verbatim entry: 2-D under raw_kind, 1-D under norm_kind.
fn rawView(
    sidecar: []const u8,
    slot: u32,
    which: zpack_file.Kind,
    cursor: *ShapeCursor,
) !tensor.View {
    const found = (try zpack_file.findIn(sidecar, family, slot, which)) orelse
        return error.MissingWeights;
    const dtype = rawDType(found.group) orelse return error.InvalidShape;
    const shape = if (which == raw_kind) try cursor.take(2) else try cursor.take(1);
    if (which == raw_kind) {
        shape[0] = found.rows;
        shape[1] = found.cols;
    } else {
        shape[0] = found.cols;
    }
    return .{ .dtype = dtype, .shape = shape, .bytes = found.bytes };
}

/// A block linear from its packed (or W16) entry, through the same swap the
/// shard path uses, with a shape taken from the entry itself.
fn linearView(sidecar: []const u8, slot: u32, cursor: *ShapeCursor) !tensor.View {
    const found = (try zpack_file.findIn(sidecar, family, slot, kind)) orelse
        return error.MissingWeights;
    const shape = try cursor.take(2);
    shape[0] = found.rows;
    shape[1] = found.cols;
    const probe = tensor.View{ .dtype = .bf16, .shape = shape, .bytes = &.{} };
    return swap(sidecar, slot, probe);
}

pub fn allowUnpacked() bool {
    return util.envFlag("ZDRAW_KLEIN_ALLOW_UNPACKED", false) and
        !util.envFlag("ZDRAW_REQUIRE_ZPACK", false);
}

pub fn requireSidecar() bool {
    return !allowUnpacked();
}

test "swap is bits-aware: W6 entries load as .u8, W16 twins win" {
    const allocator = std.testing.allocator;
    const rows = 2;
    const cols = 64;
    const f16_bytes = [_]u8{0xAA} ** (rows * cols * 2);
    const w6_len = zw6.byteLen(rows, cols, w6_group);
    const w6_bytes = try allocator.alloc(u8, w6_len);
    defer allocator.free(w6_bytes);
    @memset(w6_bytes, 0xBB);
    const w6_tag: u32 = @intCast(w6_group | (6 << 16));
    var pack: std.ArrayList(u8) = .empty;
    defer pack.deinit(allocator);
    try zpack_file.append(allocator, &pack, &.{
        // slot 0: W16 + W6 twin (W16 must win); slot 1: W6 only.
        .{ .family = family, .layer = 0, .kind = kind, .rows = rows, .cols = cols, .group = 0, .bytes = &f16_bytes },
        .{
            .family = family,
            .layer = 0,
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w6_tag,
            .bytes = w6_bytes,
        },
        .{
            .family = family,
            .layer = 1,
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w6_tag,
            .bytes = w6_bytes,
        },
    });
    const orig = [_]u8{0} ** (rows * cols * 4);
    const view = tensor.View{ .dtype = .f32, .shape = &.{ rows, cols }, .bytes = &orig };
    const w16 = try swap(pack.items, 0, view);
    try std.testing.expectEqual(tensor.DType.f16, w16.dtype);
    try std.testing.expectEqual(@as(usize, rows * cols * 2), w16.bytes.len);
    const w6 = try swap(pack.items, 1, view);
    try std.testing.expectEqual(tensor.DType.u8, w6.dtype);
    try std.testing.expectEqual(w6_len, w6.bytes.len);
    try std.testing.expect(w6.source != null);
    try std.testing.expectEqual(@as(u8, 6), w6.packed_bits);
}

test "sidecarBits reads the tier from single block 0" {
    const allocator = std.testing.allocator;
    const rows = 2;
    const cols = 64;
    const w4_len = zw4.byteLen(rows, cols, w6_group);
    const w4_bytes = try allocator.alloc(u8, w4_len);
    defer allocator.free(w4_bytes);
    @memset(w4_bytes, 0xCC);
    const w4_tag: u32 = @intCast(w6_group | (4 << 16));
    var pack: std.ArrayList(u8) = .empty;
    defer pack.deinit(allocator);
    try zpack_file.append(allocator, &pack, &.{
        .{
            .family = family,
            .layer = singleSlot(0, .qkv_mlp),
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w4_tag,
            .bytes = w4_bytes,
        },
    });
    try std.testing.expectEqual(@as(u8, 4), try sidecarBits(pack.items));
    try std.testing.expectEqual(@as(u8, 16), try sidecarBits(&.{}));
}

test "sidecarBits names a mixed pack when the singles and v/out widths differ" {
    const allocator = std.testing.allocator;
    const rows = 2;
    const cols = 64;
    const w2_len = zw2.byteLen(rows, cols, w6_group);
    const w2_bytes = try allocator.alloc(u8, w2_len);
    defer allocator.free(w2_bytes);
    @memset(w2_bytes, 0x11);
    const w6_len = zw6.byteLen(rows, cols, w6_group);
    const w6_bytes = try allocator.alloc(u8, w6_len);
    defer allocator.free(w6_bytes);
    @memset(w6_bytes, 0xBB);
    const w2_tag: u32 = @intCast(w6_group | (2 << 16));
    const w6_tag: u32 = @intCast(w6_group | (6 << 16));
    var mixed: std.ArrayList(u8) = .empty;
    defer mixed.deinit(allocator);
    try zpack_file.append(allocator, &mixed, &.{
        .{
            .family = family,
            .layer = singleSlot(0, .qkv_mlp),
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w2_tag,
            .bytes = w2_bytes,
        },
        .{
            .family = family,
            .layer = doubleSlot(0, .to_v),
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w6_tag,
            .bytes = w6_bytes,
        },
    });
    try std.testing.expectEqual(mixed_bits, try sidecarBits(mixed.items));
    var all2: std.ArrayList(u8) = .empty;
    defer all2.deinit(allocator);
    try zpack_file.append(allocator, &all2, &.{
        .{
            .family = family,
            .layer = singleSlot(0, .qkv_mlp),
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w2_tag,
            .bytes = w2_bytes,
        },
        .{
            .family = family,
            .layer = doubleSlot(0, .to_v),
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w2_tag,
            .bytes = w2_bytes,
        },
    });
    try std.testing.expectEqual(@as(u8, 2), try sidecarBits(all2.items));
}

test "swap loads a W4 entry as a .u8 marker with packed_bits 4" {
    const allocator = std.testing.allocator;
    const rows = 2;
    const cols = 64;
    const w4_len = zw4.byteLen(rows, cols, w6_group);
    const w4_bytes = try allocator.alloc(u8, w4_len);
    defer allocator.free(w4_bytes);
    @memset(w4_bytes, 0xCC);
    const w4_tag: u32 = @intCast(w6_group | (4 << 16));
    var pack: std.ArrayList(u8) = .empty;
    defer pack.deinit(allocator);
    try zpack_file.append(allocator, &pack, &.{
        .{
            .family = family,
            .layer = 3,
            .kind = kind,
            .rows = rows,
            .cols = cols,
            .group = w4_tag,
            .bytes = w4_bytes,
        },
    });
    const orig = [_]u8{0} ** (rows * cols * 4);
    const view = tensor.View{ .dtype = .f32, .shape = &.{ rows, cols }, .bytes = &orig };
    const w4 = try swap(pack.items, 3, view);
    try std.testing.expectEqual(tensor.DType.u8, w4.dtype);
    try std.testing.expectEqual(w4_len, w4.bytes.len);
    try std.testing.expectEqual(@as(u8, 4), w4.packed_bits);
}

test "flux2 slots are stable" {
    try std.testing.expectEqual(@as(u32, 0), globalSlot(.x_embed));
    try std.testing.expectEqual(@as(u32, 100), doubleSlot(0, .to_q));
    try std.testing.expectEqual(@as(u32, 112), doubleSlot(1, .to_q));
    try std.testing.expectEqual(@as(u32, 1000), singleSlot(0, .qkv_mlp));
    try std.testing.expectEqual(@as(u32, 1003), singleSlot(1, .out));
}

test "a raw entry round-trips its bytes, shape and dtype" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 0.5, -0.25, 0.125, -1.0 };
    var pack: std.ArrayList(u8) = .empty;
    defer pack.deinit(allocator);
    try zpack_file.append(allocator, &pack, &.{
        .{
            .family = family,
            .layer = globalSlot(.mod_img),
            .kind = raw_kind,
            .rows = 1,
            .cols = 4,
            .group = rawGroup(.f32),
            .bytes = std.mem.sliceAsBytes(&data),
        },
    });
    const slot = globalSlot(.mod_img);
    const found = (try zpack_file.findIn(pack.items, family, slot, raw_kind)) orelse
        return error.NotFound;
    try std.testing.expectEqual(@as(u32, 4), found.cols);
    try std.testing.expectEqual(tensor.DType.f32, rawDType(found.group).?);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&data), found.bytes);
}

test "loadFromSidecar builds every view from a --globals sidecar" {
    const allocator = std.testing.allocator;
    const cfg = zflux2.Config.klein_4b;
    const rows = 2;
    const cols = 64;
    const f16_bytes = [_]u8{0x3C} ** (rows * cols * 2);
    const norm_bytes = [_]u8{0x3F} ** (cols * 2);
    var entries: std.ArrayList(zpack_file.Entry) = .empty;
    defer entries.deinit(allocator);
    inline for (std.meta.fields(zflux2.Globals)) |f| {
        try entries.append(allocator, .{
            .family = family,
            .layer = globalSlot(@field(Global, f.name)),
            .kind = raw_kind,
            .rows = rows,
            .cols = cols,
            .group = rawGroup(.bf16),
            .bytes = &f16_bytes,
        });
    }
    for (0..cfg.double_layers) |i| {
        inline for (std.meta.fields(Double)) |f| {
            try entries.append(allocator, .{
                .family = family,
                .layer = doubleSlot(i, @field(Double, f.name)),
                .kind = kind,
                .rows = rows,
                .cols = cols,
                .group = 0,
                .bytes = &f16_bytes,
            });
        }
        for (0..4) |idx| {
            try entries.append(allocator, .{
                .family = family,
                .layer = normSlotDouble(i, @intCast(idx)),
                .kind = norm_kind,
                .rows = 1,
                .cols = cols,
                .group = rawGroup(.bf16),
                .bytes = &norm_bytes,
            });
        }
    }
    for (0..cfg.single_layers) |i| {
        inline for (std.meta.fields(Single)) |f| {
            try entries.append(allocator, .{
                .family = family,
                .layer = singleSlot(i, @field(Single, f.name)),
                .kind = kind,
                .rows = rows,
                .cols = cols,
                .group = 0,
                .bytes = &f16_bytes,
            });
        }
        for (0..2) |idx| {
            try entries.append(allocator, .{
                .family = family,
                .layer = normSlotSingle(i, @intCast(idx)),
                .kind = norm_kind,
                .rows = 1,
                .cols = cols,
                .group = rawGroup(.bf16),
                .bytes = &norm_bytes,
            });
        }
    }
    var pack: std.ArrayList(u8) = .empty;
    defer pack.deinit(allocator);
    try zpack_file.append(allocator, &pack, entries.items);
    try std.testing.expect(hasRaw(pack.items));
    var loaded = try loadFromSidecar(allocator, pack.items, cfg);
    defer loaded.deinit(std.testing.io, allocator);
    try std.testing.expectEqual(@as(usize, cols), loaded.globals.mod_img.shape[1]);
    try std.testing.expectEqual(tensor.DType.bf16, loaded.globals.mod_img.dtype);
    try std.testing.expectEqual(tensor.DType.f16, loaded.doubles[0].to_q.dtype);
    try std.testing.expectEqual(@as(usize, 1), loaded.singles[0].norm_q.shape.len);
    try std.testing.expectEqual(@as(usize, cols), loaded.singles[0].norm_q.shape[0]);
}
