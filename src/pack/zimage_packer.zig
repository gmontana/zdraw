//! Dev-only `.zpack` sidecar builder.

const std = @import("std");

const tensor = @import("tensor.zig");
const zblock = @import("../zimage/zblock.zig");
const zimage = @import("../zimage/zimage.zig");
const zlora = @import("../klein/zlora.zig");
const zpack = @import("zpack.zig");
const zpack_families = @import("zpack_families.zig");
const zpack_file = @import("zpack_file.zig");
const zpack_kinds = @import("zpack_kinds.zig");
const zw6 = @import("zw6.zig");
const ztx = @import("../zimage/ztx.zig");

pub const Options = struct {
    weights: []const u8 = "",
    out: []const u8 = "runs/zdraw-down-last8.zpack",
    last: usize = 8,
    group: usize = 64,
    bits: usize = 8, // 8 = grouped W8 quant, 16 = plain f16 image (group 0)
    w6_down: bool = false, // additionally emit W6 entries for ffn_down
    w16_refiners: bool = false, // additionally emit W16 q/k/v/gate/up for refiners
    kinds: zpack_kinds.Set = zpack_kinds.Set.downOnly(),
    families: zpack_families.Set = zpack_families.Set.mainOnly(),
    /// LoRA adapter merged into every packed weight it targets (bake-time,
    /// same mechanism as kleinpack); "" = none.
    lora: []const u8 = "",
    lora_scale: f32 = 1.0,
};

/// A packed weight's source view: the checkpoint view, or the LoRA-merged
/// f32 rows the builder owns until the entry is written.
const Src = struct {
    view: tensor.View,
    owned: ?[]f32 = null,

    fn deinit(self: *Src, allocator: std.mem.Allocator) void {
        if (self.owned) |rows| allocator.free(rows);
        self.* = undefined;
    }
};

fn familyPrefix(family: zpack_file.Family) []const u8 {
    return switch (family) {
        .main => "layers",
        .noise => "noise_refiner",
        .context => "context_refiner",
        .flux2, .text => unreachable, // kleinpack owns FLUX.2 and the text pack
    };
}

/// Checkpoint tensor-name suffix per packed kind, minus ".weight": the names
/// zblock.load reads and the stems Z-Image adapters key off.
fn kindSuffix(kind: zpack_file.Kind) []const u8 {
    return switch (kind) {
        .q => "attention.to_q",
        .k => "attention.to_k",
        .v => "attention.to_v",
        .proj => "attention.to_out.0",
        .ffn_gate => "feed_forward.w1",
        .ffn_down => "feed_forward.w2",
        .ffn_up => "feed_forward.w3",
        .ffn_gateup => unreachable, // derived from gate and up, each merged on its own
        .flux2_weight, .flux2_raw, .flux2_norm => unreachable,
    };
}

fn source(
    allocator: std.mem.Allocator,
    adapter: ?*const zlora.Adapter,
    family: zpack_file.Family,
    index: u32,
    kind: zpack_file.Kind,
    base: tensor.View,
    adapted: *usize,
) !Src {
    const ad = adapter orelse return .{ .view = base };
    var buf: [128]u8 = undefined;
    const stem = try std.fmt.bufPrint(
        &buf,
        "{s}.{d}.{s}",
        .{ familyPrefix(family), index, kindSuffix(kind) },
    );
    const found = try ad.find(stem) orelse return .{ .view = base };
    const rows = try zlora.merge(allocator, base, found, ad.scale);
    adapted.* += 1;
    return .{
        .view = .{ .dtype = .f32, .shape = base.shape, .bytes = std.mem.sliceAsBytes(rows) },
        .owned = rows,
    };
}

// Streams one entry at a time so a full-model sidecar never needs an
// in-memory image (a W16 all-layers build is ~11 GiB).
const Stream = struct {
    writer: *std.Io.Writer,
    chunk: std.ArrayList(u8),
    offset: usize = 0,
    count: usize = 0,

    fn emit(self: *Stream) !void {
        try self.writer.writeAll(self.chunk.items);
        self.offset += self.chunk.items.len;
        self.chunk.clearRetainingCapacity();
    }

    fn deinit(self: *Stream, allocator: std.mem.Allocator) void {
        self.chunk.deinit(allocator);
    }
};

pub fn run(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.weights.len == 0) return error.MissingWeights;
    try ensureParent(io, opts.out);
    var meta = try zimage.load(io, allocator, opts.weights);
    defer meta.deinit(allocator);
    var tx = try ztx.load(
        io,
        allocator,
        opts.weights,
        meta.config.transformer,
        meta.indexes.transformer,
    );
    defer tx.deinit(io, allocator);
    var adapter: ?zlora.Adapter = if (opts.lora.len == 0)
        null
    else
        try zlora.open(io, allocator, opts.lora, opts.lora_scale);
    defer if (adapter) |*a| a.deinit(io, allocator);
    try packAll(io, allocator, tx, opts, if (adapter) |*a| a else null);
}

fn packAll(
    io: std.Io,
    allocator: std.mem.Allocator,
    tx: ztx.Loaded,
    opts: Options,
    adapter: ?*const zlora.Adapter,
) !void {
    const cap = packCap(tx, opts);
    if (cap == 0) return error.MissingFamily;
    if (opts.kinds.count() == 0) return error.MissingKind;
    const file = try std.Io.Dir.cwd().createFile(io, opts.out, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var writer = file.writerStreaming(io, &buf);
    var stream = Stream{
        .writer = &writer.interface,
        .chunk = try std.ArrayList(u8).initCapacity(allocator, 1 << 20),
    };
    defer stream.deinit(allocator);
    try zpack_file.beginStream(allocator, &stream.chunk, cap);
    try stream.emit();
    var adapted: usize = 0;
    try packFamily(io, allocator, &stream, .main, tx.layers.items, opts, adapter, &adapted);
    try packFamily(io, allocator, &stream, .noise, tx.noise.items, opts, adapter, &adapted);
    try packFamily(io, allocator, &stream, .context, tx.context.items, opts, adapter, &adapted);
    try writer.interface.flush();
    if (stream.count != cap) return error.InvalidShape;
    if (adapter) |_| {
        // A LoRA that matched nothing means the key convention did not fit;
        // silently shipping the base weights would look like a weak adapter.
        if (adapted == 0) return error.MissingTensor;
        var msg: [128]u8 = undefined;
        const fmt = "zpackbuild: merged LoRA into {d} weights\n";
        try note(io, try std.fmt.bufPrint(&msg, fmt, .{adapted}));
        try writeLoraNote(io, allocator, opts.out, opts.lora, opts.lora_scale, adapted);
        // The strict profile's exact GEMMs read the checkpoint, never the
        // sidecar (runtime_options.zig), so the adapter reaches product only.
        try note(io, "zpackbuild: note: the adapter applies to --profile product; " ++
            "strict renders the base checkpoint\n");
    }
    var msg: [1200]u8 = undefined;
    try note(io, try std.fmt.bufPrint(&msg, "zpackbuild: wrote {s} ({d:.1} MiB)\n", .{
        opts.out,
        mib(stream.offset),
    }));
}

fn packCap(tx: ztx.Loaded, opts: Options) usize {
    const kinds = opts.kinds.count() + @intFromBool(fusedWanted(opts)) +
        @as(usize, if (opts.w6_down) 7 else 0);
    // Refiner layers additionally pack W16 q/k/v/gate/up under --w16-refiners.
    const ref_kinds = kinds + @as(usize, if (opts.w16_refiners) 5 else 0);
    var total: usize = 0;
    if (opts.families.main) total += @min(opts.last, tx.layers.items.len) * kinds;
    if (opts.families.noise) total += @min(opts.last, tx.noise.items.len) * ref_kinds;
    if (opts.families.context) total += @min(opts.last, tx.context.items.len) * ref_kinds;
    return total;
}

fn packFamily(
    io: std.Io,
    allocator: std.mem.Allocator,
    stream: *Stream,
    family: zpack_file.Family,
    layers: []const zblock.Views,
    opts: Options,
    adapter: ?*const zlora.Adapter,
    adapted: *usize,
) !void {
    if (!opts.families.has(family)) return;
    const count = @min(opts.last, layers.len);
    const from = layers.len - count;
    for (layers[from..], 0..) |layer, i| {
        const index: u32 = @intCast(from + i);
        try packLayer(io, allocator, stream, family, layer, index, opts, adapter, adapted);
    }
}

/// Shared arguments of the per-layer packers.
const Layer = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    stream: *Stream,
    family: zpack_file.Family,
    layer: zblock.Views,
    index: u32,
    opts: Options,
    adapter: ?*const zlora.Adapter,
    adapted: *usize,
};

fn packLayer(
    io: std.Io,
    allocator: std.mem.Allocator,
    stream: *Stream,
    family: zpack_file.Family,
    layer: zblock.Views,
    index: u32,
    opts: Options,
    adapter: ?*const zlora.Adapter,
    adapted: *usize,
) !void {
    const l = Layer{
        .io = io,
        .allocator = allocator,
        .stream = stream,
        .family = family,
        .layer = layer,
        .index = index,
        .opts = opts,
        .adapter = adapter,
        .adapted = adapted,
    };
    for (zpack_kinds.order) |kind| {
        if (!opts.kinds.has(kind)) continue;
        errdefer failNote(io, "pack", family, index, kind);
        var srcv = try source(allocator, adapter, family, index, kind, view(layer, kind), adapted);
        defer srcv.deinit(allocator);
        const src = srcv.view;
        if (opts.bits == 16) {
            var w16 = try zpack.packW16(allocator, src);
            defer w16.deinit(allocator);
            const e = entry(family, index, kind, w16.rows, w16.cols, 0, w16.bytes);
            try zpack_file.appendEntryAt(allocator, &stream.chunk, e, stream.offset);
        } else if (opts.bits == 6) {
            var w6 = try zw6.packW6(allocator, src, opts.group);
            defer w6.deinit(allocator);
            const tag: u32 = @intCast(opts.group | (6 << 16));
            const e = entry(family, index, kind, w6.rows, w6.cols, tag, w6.bytes);
            try zpack_file.appendEntryAt(allocator, &stream.chunk, e, stream.offset);
        } else {
            var w8 = try zpack.packW8(allocator, src, opts.group);
            defer w8.deinit(allocator);
            const e = entry(family, index, kind, w8.rows, w8.cols, w8.group, w8.bytes);
            try zpack_file.appendEntryAt(allocator, &stream.chunk, e, stream.offset);
        }
        try stream.emit();
        stream.count += 1;
    }
    if (opts.w6_down) try packW6Down(l);
    if (opts.w16_refiners and family != .main) try packRefiners(l);
    if (fusedWanted(opts)) try packFused(l);
}

/// Additional W6 entries for the attention and FFN weights (--w6-down).
fn packW6Down(l: Layer) !void {
    const w6_kinds = [_]zpack_file.Kind{
        .q, .k, .v, .proj, .ffn_gate, .ffn_up, .ffn_down,
    };
    for (w6_kinds) |kind| {
        errdefer failNote(l.io, "w6 pack", l.family, l.index, kind);
        const base = view(l.layer, kind);
        var srcv = try source(l.allocator, l.adapter, l.family, l.index, kind, base, l.adapted);
        defer srcv.deinit(l.allocator);
        var w6 = try zw6.packW6(l.allocator, srcv.view, 64);
        defer w6.deinit(l.allocator);
        const tag: u32 = @intCast(64 | (@as(u32, 6) << 16));
        const e = entry(l.family, l.index, kind, w6.rows, w6.cols, tag, w6.bytes);
        try zpack_file.appendEntryAt(l.allocator, &l.stream.chunk, e, l.stream.offset);
        try l.stream.emit();
        l.stream.count += 1;
    }
}

/// W16 (plain f16, group 0) for the refiner GEMMs that run half mode but
/// have no sidecar entry today (they read original f32): the quality-
/// certified memory profile's lever, never the strict default.
fn packRefiners(l: Layer) !void {
    const ref_kinds = [_]zpack_file.Kind{ .q, .k, .v, .ffn_gate, .ffn_up };
    for (ref_kinds) |kind| {
        errdefer failNote(l.io, "w16 refiner pack", l.family, l.index, kind);
        const base = view(l.layer, kind);
        var srcv = try source(l.allocator, l.adapter, l.family, l.index, kind, base, l.adapted);
        defer srcv.deinit(l.allocator);
        var w16 = try zpack.packW16(l.allocator, srcv.view);
        defer w16.deinit(l.allocator);
        const e = entry(l.family, l.index, kind, w16.rows, w16.cols, 0, w16.bytes);
        try zpack_file.appendEntryAt(l.allocator, &l.stream.chunk, e, l.stream.offset);
        try l.stream.emit();
        l.stream.count += 1;
    }
}

/// The fused gate|up W16 pair.
fn packFused(l: Layer) !void {
    errdefer {
        var eb: [200]u8 = undefined;
        const fmt = "fused pack failed family={d} layer={d} gate={any} up={any}\n";
        const line = std.fmt.bufPrint(&eb, fmt, .{
            @intFromEnum(l.family), l.index, l.layer.ffn_gate.shape, l.layer.ffn_up.shape,
        }) catch "pack failed\n";
        noteQuiet(l.io, line);
    }
    const a = l.allocator;
    var gate = try source(a, l.adapter, l.family, l.index, .ffn_gate, l.layer.ffn_gate, l.adapted);
    defer gate.deinit(a);
    var up = try source(a, l.adapter, l.family, l.index, .ffn_up, l.layer.ffn_up, l.adapted);
    defer up.deinit(a);
    var w16 = try zpack.packW16Pair(a, gate.view, up.view);
    defer w16.deinit(a);
    const e = entry(l.family, l.index, .ffn_gateup, w16.rows, w16.cols, 0, w16.bytes);
    try zpack_file.appendEntryAt(a, &l.stream.chunk, e, l.stream.offset);
    try l.stream.emit();
    l.stream.count += 1;
}

/// errdefer diagnostic for one weight: which stage, layer and kind failed.
fn failNote(
    io: std.Io,
    comptime what: []const u8,
    family: zpack_file.Family,
    index: u32,
    kind: zpack_file.Kind,
) void {
    var eb: [200]u8 = undefined;
    const line = std.fmt.bufPrint(&eb, what ++ " failed family={d} layer={d} kind={s}\n", .{
        @intFromEnum(family), index, @tagName(kind),
    }) catch "pack failed\n";
    noteQuiet(io, line);
}

fn fusedWanted(opts: Options) bool {
    return opts.bits == 16 and opts.kinds.has(.ffn_gate) and opts.kinds.has(.ffn_up);
}

fn view(layer: zblock.Views, kind: zpack_file.Kind) tensor.View {
    return switch (kind) {
        .q => layer.q,
        .k => layer.k,
        .v => layer.v,
        .proj => layer.proj,
        .ffn_gate => layer.ffn_gate,
        .ffn_up => layer.ffn_up,
        .ffn_down => layer.ffn_down,
        .ffn_gateup => unreachable, // emitted directly by packLayer, not via kinds
        .flux2_weight, .flux2_raw, .flux2_norm => unreachable, // emitted by kleinpack, not Z-Image zpackbuild
    };
}

fn entry(
    family: zpack_file.Family,
    layer: u32,
    kind: zpack_file.Kind,
    rows: usize,
    cols: usize,
    group: usize,
    bytes: []const u8,
) zpack_file.Entry {
    return .{
        .family = family,
        .layer = layer,
        .kind = kind,
        .rows = @intCast(rows),
        .cols = @intCast(cols),
        .group = @intCast(group),
        .bytes = bytes,
    };
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1024.0 / 1024.0;
}

/// A `<pack>.lora.json` note beside an adapted sidecar: the runtime refuses
/// such a pack under the strict profile (strict reads the checkpoint, so the
/// adapter would silently not apply; ledger zimage-lora-bake-20260903), and
/// tools can tell an adapted pack from a base one.
fn writeLoraNote(
    io: std.Io,
    allocator: std.mem.Allocator,
    pack: []const u8,
    adapter: []const u8,
    scale: f32,
    merged: usize,
) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}.lora.json", .{pack});
    defer allocator.free(path);
    const text = try std.fmt.allocPrint(
        allocator,
        "{{\"adapter\": \"{s}\", \"scale\": {d:.4}, \"merged_weights\": {d}}}\n",
        .{ adapter, scale, merged },
    );
    defer allocator.free(text);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [512]u8 = undefined;
    var writer = file.writerStreaming(io, &buf);
    try writer.interface.writeAll(text);
    try writer.interface.flush();
}

/// One line to stderr (the builders run under the CLI, never std.debug).
fn note(io: std.Io, text: []const u8) !void {
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

/// `note` for errdefer paths, where nothing can be returned.
fn noteQuiet(io: std.Io, text: []const u8) void {
    note(io, text) catch |err| {
        std.log.warn("zpackbuild: stderr write failed: {s}", .{@errorName(err)});
    };
}

/// The output's directory, created if missing.
fn ensureParent(io: std.Io, path: []const u8) !void {
    const dir = std.fs.path.dirname(path) orelse return;
    try std.Io.Dir.cwd().createDirPath(io, dir);
}
