//! LoRA adapter loading and merge for FLUX.2 Klein.
//!
//! An adapter stores two low-rank factors per targeted weight — A [rank, in]
//! and B [out, rank] — whose product is the delta applied to the base weight:
//! `W' = W + (alpha/rank) * scale * (B @ A)`. zdraw merges that delta while
//! building the W16 sidecar rather than at inference time, so adapted runs
//! keep the mmap'd single-buffer weight path, its memory profile, and every
//! GEMM route unchanged (see lorapack.zig).
//!
//! Key naming: the lookup starts from the diffusers stem (the base tensor
//! name minus `.weight`, zflux2_pack stem table) and accepts three published
//! layouts for it:
//!   - diffusers: `[transformer.]<stem>.lora_{A,B}.weight` (what the BFL
//!     training recipe and diffusers emit);
//!   - native / PEFT / Comfy-converted: BFL module names under an optional
//!     `diffusion_model.` or `base_model.model.` prefix, e.g.
//!     `double_blocks.N.img_attn.qkv` with q, k and v stacked along B's rows,
//!     `single_blocks.N.linear1`, `img_in`; the alias table below maps each
//!     stem to that name plus the row window of the fused factor;
//!   - `lora_{down,up}.weight` as an alias for `lora_{A,B}` in either layout.
//! Modulation, time and final-norm weights are never merged: they are not
//! GEMM-substituted (globalStem returns null), so an adapter's deltas for
//! them are reported as unmerged rather than applied.
//!
//! Scaling: the delta is `(alpha/rank) * scale * (B @ A)`. kohya-style files
//! carry `<stem>.alpha` tensors; PEFT/diffusers files carry `lora_alpha`,
//! `r` and `use_rslora` in `__metadata__.lora_adapter_metadata` instead, and
//! applying them at alpha == rank silently merges the delta rank/alpha times
//! too strong (a rank-32, alpha-4 adapter destroyed the model at 8x; ledger
//! klein-lora-bake-20260827). Both sources are honoured; no alpha anywhere
//! means alpha == rank.

const std = @import("std");

const tensor = @import("../pack/tensor.zig");
const tensor_file = @import("../pack/tensor_file.zig");
const zflux2_pack = @import("zflux2_pack.zig");

pub const Entry = struct {
    a: tensor.View, // [rank, in]
    b: tensor.View, // [out, rank], or [3*out, rank] when `part` is set
    alpha: ?f32, // adapter-declared alpha; null = no rank rescale
    /// For a fused native factor (q|k|v stacked along B's rows): which third
    /// of B belongs to this stem. Null = B covers exactly this weight.
    part: ?u8 = null,
};

/// A native-layout name for a diffusers stem, with the fused-row part.
pub const Alias = struct {
    name: []const u8,
    part: ?u8,
};

const prefixes = [_][]const u8{ "", "transformer.", "diffusion_model.", "base_model.model." };
const pairs = [_][2][]const u8{
    .{ "lora_A", "lora_B" },
    .{ "lora_down", "lora_up" },
};

fn layerAfter(stem: []const u8, head: []const u8) ?struct { layer: []const u8, rest: []const u8 } {
    if (!std.mem.startsWith(u8, stem, head)) return null;
    const tail = stem[head.len..];
    const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return null;
    if (dot == 0) return null;
    for (tail[0..dot]) |c| if (!std.ascii.isDigit(c)) return null;
    return .{ .layer = tail[0..dot], .rest = tail[dot + 1 ..] };
}

/// The BFL module name an adapter trained on the native checkpoint uses for
/// a diffusers stem. Written into `buf`; null when the stem has no native
/// counterpart. Mirrors the diffusers<->BFL conversion tables for FLUX.2:
/// double blocks fuse q|k|v into one `qkv` (rows in that order), the single
/// block keeps its fused `linear1` (zdraw's `to_qkv_mlp_proj` slot is the
/// same fused tensor), and the embedders/final layer rename.
pub fn nativeAlias(stem: []const u8, buf: []u8) ?Alias {
    if (layerAfter(stem, "transformer_blocks.")) |d| {
        const map = [_]struct { rest: []const u8, native: []const u8, part: ?u8 }{
            .{ .rest = "attn.to_q", .native = "img_attn.qkv", .part = 0 },
            .{ .rest = "attn.to_k", .native = "img_attn.qkv", .part = 1 },
            .{ .rest = "attn.to_v", .native = "img_attn.qkv", .part = 2 },
            .{ .rest = "attn.add_q_proj", .native = "txt_attn.qkv", .part = 0 },
            .{ .rest = "attn.add_k_proj", .native = "txt_attn.qkv", .part = 1 },
            .{ .rest = "attn.add_v_proj", .native = "txt_attn.qkv", .part = 2 },
            .{ .rest = "attn.to_out.0", .native = "img_attn.proj", .part = null },
            .{ .rest = "attn.to_add_out", .native = "txt_attn.proj", .part = null },
            .{ .rest = "ff.linear_in", .native = "img_mlp.0", .part = null },
            .{ .rest = "ff.linear_out", .native = "img_mlp.2", .part = null },
            .{ .rest = "ff_context.linear_in", .native = "txt_mlp.0", .part = null },
            .{ .rest = "ff_context.linear_out", .native = "txt_mlp.2", .part = null },
        };
        for (map) |m| {
            if (std.mem.eql(u8, d.rest, m.rest)) {
                const name = std.fmt.bufPrint(
                    buf,
                    "double_blocks.{s}.{s}",
                    .{ d.layer, m.native },
                ) catch return null;
                return .{ .name = name, .part = m.part };
            }
        }
        return null;
    }
    if (layerAfter(stem, "single_transformer_blocks.")) |sgl| {
        const native: []const u8 = if (std.mem.eql(u8, sgl.rest, "attn.to_qkv_mlp_proj"))
            "linear1"
        else if (std.mem.eql(u8, sgl.rest, "attn.to_out"))
            "linear2"
        else
            return null;
        const name = std.fmt.bufPrint(
            buf,
            "single_blocks.{s}.{s}",
            .{ sgl.layer, native },
        ) catch return null;
        return .{ .name = name, .part = null };
    }
    const globals = [_][2][]const u8{
        .{ "x_embedder", "img_in" },
        .{ "context_embedder", "txt_in" },
        .{ "proj_out", "final_layer.linear" },
    };
    for (globals) |g| {
        if (std.mem.eql(u8, stem, g[0])) return .{ .name = g[1], .part = null };
    }
    return null;
}

pub const Adapter = struct {
    map: tensor_file.Mapped,
    /// User-facing strength multiplier applied on top of alpha/rank.
    scale: f32,
    /// PEFT `lora_alpha / r` (or `lora_alpha / sqrt(r)` under rslora) from
    /// the file's metadata; used for every stem without its own `.alpha`.
    peft_factor: ?f32 = null,

    pub fn deinit(self: *Adapter, io: std.Io, allocator: std.mem.Allocator) void {
        self.map.deinit(io, allocator);
        self.* = undefined;
    }

    /// Adapter entry for one base tensor stem, or null when the adapter does
    /// not target it. Half a pair (A without B) is a malformed adapter.
    pub fn find(self: *const Adapter, stem: []const u8) !?Entry {
        if (try self.lookup(stem, null)) |e| return e;
        var alias_buf: [256]u8 = undefined;
        if (nativeAlias(stem, &alias_buf)) |alias| {
            if (try self.lookup(alias.name, alias.part)) |e| return e;
        }
        return null;
    }

    fn lookup(self: *const Adapter, name: []const u8, part: ?u8) !?Entry {
        var buf: [256]u8 = undefined;
        for (prefixes) |prefix| {
            for (pairs) |pair| {
                const a_name = std.fmt.bufPrint(
                    &buf,
                    "{s}{s}.{s}.weight",
                    .{ prefix, name, pair[0] },
                ) catch continue;
                const a = try self.map.view(a_name) orelse continue;
                var b_buf: [256]u8 = undefined;
                const b_name = try std.fmt.bufPrint(
                    &b_buf,
                    "{s}{s}.{s}.weight",
                    .{ prefix, name, pair[1] },
                );
                const b = try self.map.view(b_name) orelse return error.MissingTensor;
                var alpha_buf: [256]u8 = undefined;
                const alpha_name = try std.fmt.bufPrint(
                    &alpha_buf,
                    "{s}{s}.alpha",
                    .{ prefix, name },
                );
                var alpha: ?f32 = if (try self.map.view(alpha_name)) |v|
                    try v.atF32(0)
                else
                    null;
                if (alpha == null) {
                    if (self.peft_factor) |f| {
                        // merge() divides alpha by the rank it reads from A.
                        alpha = f * @as(f32, @floatFromInt(a.shape[0]));
                    }
                }
                return .{ .a = a, .b = b, .alpha = alpha, .part = part };
            }
        }
        return null;
    }
};

pub fn open(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    scale: f32,
) !Adapter {
    const map = try tensor_file.open(io, allocator, path);
    const factor = peftFactor(allocator, map.headerJson()) catch |err| blk: {
        std.debug.print(
            "zlora: unreadable PEFT metadata ({s}); assuming alpha == rank\n",
            .{@errorName(err)},
        );
        break :blk null;
    };
    if (factor) |f| std.debug.print("zlora: PEFT metadata scaling alpha/r = {d:.4}\n", .{f});
    return .{ .map = map, .scale = scale, .peft_factor = factor };
}

/// `lora_alpha / r` from a PEFT adapter's `__metadata__.lora_adapter_metadata`
/// (a JSON document stored as a string), or null when the file carries no
/// such block. Keys are prefixed by the adapted component (`transformer.`).
/// Per-module `alpha_pattern` / `rank_pattern` overrides are reported and
/// ignored: the global pair is applied to every stem.
pub fn peftFactor(allocator: std.mem.Allocator, header_json: []const u8) !?f32 {
    var outer = try std.json.parseFromSlice(std.json.Value, allocator, header_json, .{});
    defer outer.deinit();
    const meta = switch (outer.value) {
        .object => |o| o.get("__metadata__") orelse return null,
        else => return null,
    };
    const inner_text = switch (meta) {
        .object => |o| switch (o.get("lora_adapter_metadata") orelse return null) {
            .string => |t| t,
            else => return null,
        },
        else => return null,
    };
    var inner = try std.json.parseFromSlice(std.json.Value, allocator, inner_text, .{});
    defer inner.deinit();
    const cfg = switch (inner.value) {
        .object => |o| o,
        else => return null,
    };
    var alpha: ?f32 = null;
    var rank: ?f32 = null;
    var rslora = false;
    var it = cfg.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (std.mem.endsWith(u8, key, ".lora_alpha") or std.mem.eql(u8, key, "lora_alpha")) {
            alpha = jsonNumber(kv.value_ptr.*);
        } else if (std.mem.endsWith(u8, key, ".r") or std.mem.eql(u8, key, "r")) {
            rank = jsonNumber(kv.value_ptr.*);
        } else if (std.mem.endsWith(u8, key, "use_rslora")) {
            rslora = switch (kv.value_ptr.*) {
                .bool => |b| b,
                else => false,
            };
        } else if (std.mem.endsWith(u8, key, "alpha_pattern") or
            std.mem.endsWith(u8, key, "rank_pattern"))
        {
            switch (kv.value_ptr.*) {
                .object => |o| if (o.count() > 0) std.debug.print(
                    "zlora: {s} has {d} per-module overrides; applying the global alpha/r\n",
                    .{ key, o.count() },
                ),
                else => {},
            }
        }
    }
    const a = alpha orelse return null;
    const r = rank orelse return null;
    if (r <= 0) return error.InvalidShape;
    return if (rslora) a / @sqrt(r) else a / r;
}

fn jsonNumber(v: std.json.Value) ?f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => null,
    };
}

/// Base weight plus the adapter delta, as freshly allocated f32 rows in the
/// base's [out, in] layout. Caller frees.
pub fn merge(
    allocator: std.mem.Allocator,
    base: tensor.View,
    entry: Entry,
    scale: f32,
) ![]f32 {
    if (base.shape.len != 2 or entry.a.shape.len != 2 or entry.b.shape.len != 2) {
        return error.InvalidShape;
    }
    const out = base.shape[0];
    const in = base.shape[1];
    const rank = entry.a.shape[0];
    // A fused native factor stacks three weights along B's rows; this stem
    // owns one third of them.
    const b_rows_expected: usize = if (entry.part != null) 3 * out else out;
    if (entry.a.shape[1] != in or entry.b.shape[0] != b_rows_expected or entry.b.shape[1] != rank) {
        return error.InvalidShape;
    }
    if (entry.part) |p| if (p > 2) return error.InvalidShape;
    if (rank == 0) return error.InvalidShape;
    const b_row_off: usize = if (entry.part) |p| @as(usize, p) * out else 0;

    // alpha/rank is the adapter's own normalization; `scale` is the user's
    // strength on top of it.
    const rank_f: f32 = @floatFromInt(rank);
    const factor = scale * if (entry.alpha) |alpha| alpha / rank_f else 1.0;

    const a = try allocator.alloc(f32, rank * in);
    defer allocator.free(a);
    try entry.a.copyF32(a);
    const b_all = try allocator.alloc(f32, b_rows_expected * rank);
    defer allocator.free(b_all);
    try entry.b.copyF32(b_all);
    const b = b_all[b_row_off * rank ..][0 .. out * rank];

    const merged = try allocator.alloc(f32, out * in);
    errdefer allocator.free(merged);
    try base.copyF32(merged);
    for (0..out) |o| {
        const row = merged[o * in ..][0..in];
        for (0..rank) |r| {
            const w = b[o * rank + r] * factor;
            if (w == 0) continue;
            const a_row = a[r * in ..][0..in];
            for (row, a_row) |*dst, av| dst.* += w * av;
        }
    }
    return merged;
}

test "merge applies the scaled low-rank delta in base layout" {
    const alloc = std.testing.allocator;
    // base [2,3] = zeros; A [1,3] = (1,2,3); B [2,1] = (1,10) -> delta rows
    // (1,2,3) and (10,20,30), scaled by alpha/rank * scale = (2/1) * 0.5 = 1.
    const base_data = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const a_data = [_]f32{ 1, 2, 3 };
    const b_data = [_]f32{ 1, 10 };
    const base = tensor.View{
        .dtype = .f32,
        .shape = &.{ 2, 3 },
        .bytes = std.mem.sliceAsBytes(&base_data),
    };
    const entry = Entry{
        .a = .{ .dtype = .f32, .shape = &.{ 1, 3 }, .bytes = std.mem.sliceAsBytes(&a_data) },
        .b = .{ .dtype = .f32, .shape = &.{ 2, 1 }, .bytes = std.mem.sliceAsBytes(&b_data) },
        .alpha = 2.0,
    };
    const merged = try merge(alloc, base, entry, 0.5);
    defer alloc.free(merged);
    const want = [_]f32{ 1, 2, 3, 10, 20, 30 };
    for (merged, want) |got, expect| try std.testing.expectApproxEqAbs(expect, got, 1e-6);
}

test "merge rejects factors that do not match the base shape" {
    const alloc = std.testing.allocator;
    const base_data = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const a_data = [_]f32{ 1, 2 };
    const b_data = [_]f32{ 1, 10 };
    const base = tensor.View{
        .dtype = .f32,
        .shape = &.{ 2, 3 },
        .bytes = std.mem.sliceAsBytes(&base_data),
    };
    // A is [1,2] but the base takes 3 inputs.
    const entry = Entry{
        .a = .{ .dtype = .f32, .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&a_data) },
        .b = .{ .dtype = .f32, .shape = &.{ 2, 1 }, .bytes = std.mem.sliceAsBytes(&b_data) },
        .alpha = null,
    };
    try std.testing.expectError(error.InvalidShape, merge(alloc, base, entry, 1.0));
}

test "merge takes one third of a fused native factor" {
    const alloc = std.testing.allocator;
    // base to_k [2,3] = zeros; A [1,3] = (1,2,3); fused B [6,1] = rows
    // 1..6 for q|k|v; part 1 (k) owns rows (3,4) -> deltas (3,6,9), (4,8,12).
    const base_data = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const a_data = [_]f32{ 1, 2, 3 };
    const b_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const base = tensor.View{
        .dtype = .f32,
        .shape = &.{ 2, 3 },
        .bytes = std.mem.sliceAsBytes(&base_data),
    };
    const entry = Entry{
        .a = .{ .dtype = .f32, .shape = &.{ 1, 3 }, .bytes = std.mem.sliceAsBytes(&a_data) },
        .b = .{ .dtype = .f32, .shape = &.{ 6, 1 }, .bytes = std.mem.sliceAsBytes(&b_data) },
        .alpha = null,
        .part = 1,
    };
    const merged = try merge(alloc, base, entry, 1.0);
    defer alloc.free(merged);
    const want = [_]f32{ 3, 6, 9, 4, 8, 12 };
    for (merged, want) |got, expect| try std.testing.expectApproxEqAbs(expect, got, 1e-6);
    // The same factor without `part` is a shape mismatch, never a silent merge.
    var whole = entry;
    whole.part = null;
    try std.testing.expectError(error.InvalidShape, merge(alloc, base, whole, 1.0));
}

test "PEFT metadata yields alpha over rank, rslora and absence handled" {
    const alloc = std.testing.allocator;
    const peft = "{\"__metadata__\":{\"format\":\"pt\",\"lora_adapter_metadata\":" ++
        "\"{\\\"transformer.lora_alpha\\\": 4, \\\"transformer.r\\\": 32, " ++
        "\\\"transformer.use_rslora\\\": false, \\\"transformer.alpha_pattern\\\": {}}\"}," ++
        "\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    const f = (try peftFactor(alloc, peft)) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), f, 1e-6);
    const rs = "{\"__metadata__\":{\"lora_adapter_metadata\":" ++
        "\"{\\\"transformer.lora_alpha\\\": 8, \\\"transformer.r\\\": 16, " ++
        "\\\"transformer.use_rslora\\\": true}\"}}";
    const g = (try peftFactor(alloc, rs)) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), g, 1e-6);
    const kohya = "{\"__metadata__\":{\"format\":\"pt\",\"ss_output_name\":\"x\"}," ++
        "\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    try std.testing.expect((try peftFactor(alloc, kohya)) == null);
    const bare = "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    try std.testing.expect((try peftFactor(alloc, bare)) == null);
}

test "native alias maps every substituted stem to its BFL name" {
    var buf: [256]u8 = undefined;
    const Case = struct { stem: []const u8, name: []const u8, part: ?u8 };
    const cases = [_]Case{
        .{
            .stem = "transformer_blocks.3.attn.to_k",
            .name = "double_blocks.3.img_attn.qkv",
            .part = 1,
        },
        .{
            .stem = "transformer_blocks.12.attn.add_v_proj",
            .name = "double_blocks.12.txt_attn.qkv",
            .part = 2,
        },
        .{
            .stem = "transformer_blocks.0.attn.to_out.0",
            .name = "double_blocks.0.img_attn.proj",
            .part = null,
        },
        .{
            .stem = "transformer_blocks.0.ff_context.linear_out",
            .name = "double_blocks.0.txt_mlp.2",
            .part = null,
        },
        .{
            .stem = "single_transformer_blocks.7.attn.to_qkv_mlp_proj",
            .name = "single_blocks.7.linear1",
            .part = null,
        },
        .{
            .stem = "single_transformer_blocks.7.attn.to_out",
            .name = "single_blocks.7.linear2",
            .part = null,
        },
        .{ .stem = "x_embedder", .name = "img_in", .part = null },
        .{ .stem = "proj_out", .name = "final_layer.linear", .part = null },
    };
    for (cases) |c| {
        const alias = nativeAlias(c.stem, &buf) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(c.name, alias.name);
        try std.testing.expectEqual(c.part, alias.part);
    }
    try std.testing.expect(nativeAlias("transformer_blocks.x.attn.to_q", &buf) == null);
    try std.testing.expect(nativeAlias("norm_out", &buf) == null);
    // Every stem the pack substitutes has a native counterpart, so an adapter
    // in either layout can reach every merged weight.
    inline for (zflux2_pack.double_weights) |e| {
        var sb: [128]u8 = undefined;
        const stem = try std.fmt.bufPrint(&sb, zflux2_pack.doubleStem(e.slot), .{@as(usize, 1)});
        try std.testing.expect(nativeAlias(stem, &buf) != null);
    }
    inline for (zflux2_pack.single_weights) |e| {
        var sb: [128]u8 = undefined;
        const stem = try std.fmt.bufPrint(&sb, zflux2_pack.singleStem(e.slot), .{@as(usize, 1)});
        try std.testing.expect(nativeAlias(stem, &buf) != null);
    }
}

test "stem table covers every substituted slot" {
    // The LoRA lookup is only as complete as the stem table; a new slot must
    // not silently become un-adaptable.
    inline for (zflux2_pack.double_weights) |e| {
        try std.testing.expect(zflux2_pack.doubleStem(e.slot).len > 0);
    }
    inline for (zflux2_pack.single_weights) |e| {
        try std.testing.expect(zflux2_pack.singleStem(e.slot).len > 0);
    }
    inline for (zflux2_pack.global_weights) |e| {
        try std.testing.expect(zflux2_pack.globalStem(e.slot) != null);
    }
}
