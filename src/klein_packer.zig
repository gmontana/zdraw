//! FLUX.2 Klein W16/W6 sidecar builder (library): `run` packs a loaded
//! checkpoint into one `.zpack`; the `kleinpack` tool and `zdraw fetch` call it.

const std = @import("std");

const tensor = @import("tensor.zig");
const zflux2 = @import("zflux2.zig");
const zlora = @import("zlora.zig");
const zflux2_pack = @import("zflux2_pack.zig");
const zpack = @import("zpack.zig");
const zpack_file = @import("zpack_file.zig");
const qnames = @import("qwen_names.zig");
const qwen_pack = @import("qwen_pack.zig");
const shards = @import("shards.zig");
const weight_index = @import("weight_index.zig");
const zw6 = @import("zw6.zig");
const zw2 = @import("zw2.zig");
const zw4 = @import("zw4.zig");
const klein_bitmap = @import("klein_bitmap.zig");

/// W6 widening ladder: each scope is a pack profile, so staging and rollback
/// are pack rebuilds, never code changes. Globals always stay W16.
pub const W6Scope = enum(u8) {
    singles_out = 0,
    singles = 1,
    singles_dff = 2,
    all = 3,
};

pub const Options = struct {
    weights: []const u8 = "",
    // Empty = derive "runs/<cfg.pack_name>" after parsing, so --size picks
    // the per-variant filename and packs for both sizes can coexist.
    out: []const u8 = "",
    cfg: zflux2.Config = zflux2.Config.klein_4b,
    bits: usize = 16,
    /// The mixed allocation of the compact model: single blocks and the image
    /// FFN at 2 bits, q/k and the text FFN at 4 (3-bit classes stored as
    /// nibbles), v/out at 6; the CLI's --bits mixed.
    mixed: bool = false,
    /// A per-slot width map (--bits map:FILE). A class or block the map names
    /// takes that width; everything else keeps the scope/mixed policy below.
    bit_map: ?*const klein_bitmap.Map = null,
    /// With --bits mixed3: the 4-bit classes hold a trained 3-bit grid, so
    /// they are packed with scale absmax/3 (codes -3..3) and reproduce the
    /// checkpoint exactly, instead of being rounded again onto the 4-bit grid.
    mixed_w3: bool = false,
    /// --globals: also store the nine global tensors and every per-block norm
    /// weight verbatim (checkpoint dtype), so a release can drop the
    /// transformer shard once the loader reads them from the sidecar.
    globals: bool = false,
    /// --text-bits 4: also write the text pack (the encoder's block linears
    /// at 4 bits, its other tensors verbatim) to --text-out; --text-only
    /// skips the transformer.
    text_bits: usize = 0,
    text_out: []const u8 = "",
    text_only: bool = false,
    scope: W6Scope = .singles,
    // Optional LoRA adapter merged into the packed weights: adapted runs then
    // use the ordinary mmap'd sidecar path with no runtime or memory cost.
    lora: []const u8 = "",
    lora_scale: f32 = 1.0,
};

fn w6Double(scope: W6Scope, d: zflux2_pack.Double) bool {
    return switch (d) {
        .ff_in,
        .ff_out,
        .ffc_in,
        .ffc_out,
        => @intFromEnum(scope) >= @intFromEnum(W6Scope.singles_dff),
        else => scope == .all,
    };
}

fn w6Single(scope: W6Scope, s: zflux2_pack.Single) bool {
    return switch (s) {
        .out => true,
        .qkv_mlp => @intFromEnum(scope) >= @intFromEnum(W6Scope.singles),
    };
}

const Stream = struct {
    writer: *std.Io.Writer,
    chunk: std.ArrayList(u8),
    offset: usize = 0,
    count: usize = 0,
    /// The 4-bit classes' scale rule: 7 (the 4-bit grid) or 3 (--bits mixed3).
    w4_qmax: f32 = 7.0,
    /// The entries' family: FLUX.2 for the transformer, text for the text pack.
    family: zpack_file.Family = .flux2,

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
    if (opts.text_bits != 0) try runText(io, allocator, opts);
    if (opts.text_only) return;
    try ensureParent(io, opts.out);
    const tx_dir = try std.fmt.allocPrint(allocator, "{s}/transformer", .{opts.weights});
    defer allocator.free(tx_dir);
    var loaded = try zflux2.load(io, allocator, tx_dir, opts.cfg);
    defer loaded.deinit(io, allocator);
    var adapter: ?zlora.Adapter = if (opts.lora.len == 0)
        null
    else
        try zlora.open(io, allocator, opts.lora, opts.lora_scale);
    defer if (adapter) |*a| a.deinit(io, allocator);
    try packAll(io, allocator, loaded, opts, if (adapter) |*a| a else null);
}

fn packAll(
    io: std.Io,
    allocator: std.mem.Allocator,
    loaded: zflux2.Loaded,
    opts: Options,
    adapter: ?*const zlora.Adapter,
) !void {
    const raw_count: usize = if (opts.globals)
        9 + loaded.doubles.len * 4 + loaded.singles.len * 2
    else
        0;
    const cap = 3 + loaded.doubles.len * 12 + loaded.singles.len * 2 + raw_count;
    const file = try std.Io.Dir.cwd().createFile(io, opts.out, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var writer = file.writerStreaming(io, &buf);
    var stream = Stream{
        .writer = &writer.interface,
        .chunk = try std.ArrayList(u8).initCapacity(allocator, 1 << 20),
        .w4_qmax = if (opts.bit_map) |m| m.w4_qmax else if (opts.mixed_w3) 3.0 else 7.0,
    };
    defer stream.deinit(allocator);
    try zpack_file.beginStream(allocator, &stream.chunk, cap);
    try stream.emit();

    var adapted: usize = 0;
    inline for (zflux2_pack.global_weights) |entry| {
        const stem = zflux2_pack.globalStem(entry.slot);
        const slot = zflux2_pack.globalSlot(entry.slot);
        const view = @field(loaded.globals, entry.field);
        try emitAdapted(allocator, &stream, slot, view, 16, adapter, stem, &adapted);
    }

    try emitBlocks(allocator, &stream, loaded, opts, adapter, &adapted);
    if (opts.globals) try emitRaw(allocator, &stream, loaded);

    try writer.interface.flush();
    if (stream.count != cap) return error.InvalidShape;
    if (adapter) |_| try finishLora(io, allocator, opts, adapted);
    try noteWrote(io, opts.out, stream.offset);
}

/// The 100 blocks' linears: per-slot width from the scope and the mixed policy.
fn emitBlocks(
    allocator: std.mem.Allocator,
    stream: *Stream,
    loaded: zflux2.Loaded,
    opts: Options,
    adapter: ?*const zlora.Adapter,
    adapted: *usize,
) !void {
    var buf_name: [256]u8 = undefined;
    for (loaded.doubles, 0..) |blk, i| {
        inline for (zflux2_pack.double_weights) |entry| {
            const chosen = w6Double(opts.scope, entry.slot);
            const mapped: ?u8 = if (opts.bit_map) |m| m.widthDouble(i, entry.slot) else null;
            const use_w6: usize = mapped orelse packedFor(opts, chosen, mixedDouble(entry.slot));
            const stem = try std.fmt.bufPrint(&buf_name, zflux2_pack.doubleStem(entry.slot), .{i});
            const slot = zflux2_pack.doubleSlot(i, entry.slot);
            const view = @field(blk, entry.field);
            try emitAdapted(allocator, stream, slot, view, use_w6, adapter, stem, adapted);
        }
    }
    for (loaded.singles, 0..) |blk, i| {
        inline for (zflux2_pack.single_weights) |entry| {
            const chosen = w6Single(opts.scope, entry.slot);
            const mapped: ?u8 = if (opts.bit_map) |m| m.widthSingle(i, entry.slot) else null;
            const use_w6: usize = mapped orelse packedFor(opts, chosen, mixedSingle(entry.slot));
            const stem = try std.fmt.bufPrint(&buf_name, zflux2_pack.singleStem(entry.slot), .{i});
            const slot = zflux2_pack.singleSlot(i, entry.slot);
            const view = @field(blk, entry.field);
            try emitAdapted(allocator, stream, slot, view, use_w6, adapter, stem, adapted);
        }
    }
}

/// The text pack: every tensor the encoder reads, by slot (qwen_pack.zig).
fn runText(io: std.Io, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.text_bits != qwen_pack.bits) return error.BadBits;
    try ensureParent(io, opts.text_out);
    const root = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{opts.weights});
    defer allocator.free(root);
    const index_path = try std.fmt.allocPrint(
        allocator,
        "{s}/model.safetensors.index.json",
        .{root},
    );
    defer allocator.free(index_path);
    var index = try weight_index.read(io, allocator, index_path);
    defer index.deinit(allocator);
    var store = try shards.open(io, allocator, root, index);
    defer store.deinit(io, allocator);
    const layers = try textLayers(allocator, index);
    const cap = 2 + layers * std.meta.fields(qnames.Layer).len;
    const file = try std.Io.Dir.cwd().createFile(io, opts.text_out, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var writer = file.writerStreaming(io, &buf);
    var stream = Stream{
        .writer = &writer.interface,
        .chunk = try std.ArrayList(u8).initCapacity(allocator, 1 << 20),
        .family = qwen_pack.family,
    };
    defer stream.deinit(allocator);
    try zpack_file.beginStream(allocator, &stream.chunk, cap);
    try stream.emit();
    const embed = try store.view(index, qnames.embed);
    try emitVerbatim(allocator, &stream, qwen_pack.embed_slot, qwen_pack.raw_kind, embed);
    const final_norm = try store.view(index, qnames.final_norm);
    try emitVerbatim(allocator, &stream, qwen_pack.final_slot, qwen_pack.raw_kind, final_norm);
    for (0..layers) |layer| try emitTextLayer(allocator, &stream, &store, index, layer);
    try writer.interface.flush();
    if (stream.count != cap) return error.InvalidShape;
    try noteWrote(io, opts.text_out, stream.offset);
}

/// Layers are counted from the index: consecutive input norms from 0.
fn textLayers(allocator: std.mem.Allocator, index: weight_index.Index) !usize {
    var n: usize = 0;
    while (true) : (n += 1) {
        const name = try qnames.layerName(allocator, n, .input_norm);
        defer allocator.free(name);
        if (index.find(name) == null) return n;
    }
}

fn emitTextLayer(
    allocator: std.mem.Allocator,
    stream: *Stream,
    store: *const shards.Store,
    index: weight_index.Index,
    layer: usize,
) !void {
    inline for (std.meta.fields(qnames.Layer)) |f| {
        const part: qnames.Layer = @enumFromInt(f.value);
        const name = try qnames.layerName(allocator, layer, part);
        defer allocator.free(name);
        const view = try store.view(index, name);
        const slot = qwen_pack.layerSlot(layer, part);
        if (qwen_pack.isLinear(part)) {
            try emit(allocator, stream, slot, view, qwen_pack.bits);
        } else {
            try emitVerbatim(allocator, stream, slot, qwen_pack.raw_kind, view);
        }
    }
}

/// A LoRA that matched nothing means the key convention did not fit; silently
/// shipping the base weights would look like a weak adapter.
fn finishLora(io: std.Io, allocator: std.mem.Allocator, opts: Options, adapted: usize) !void {
    if (adapted == 0) return error.MissingTensor;
    var msg: [128]u8 = undefined;
    const fmt = "kleinpack: merged LoRA into {d} weights\n";
    try note(io, try std.fmt.bufPrint(&msg, fmt, .{adapted}));
    try writeLoraNote(io, allocator, opts.out, opts.lora, opts.lora_scale, adapted);
}

fn noteWrote(io: std.Io, out: []const u8, bytes: usize) !void {
    var msg: [1200]u8 = undefined;
    const fmt = "kleinpack: wrote {s} ({d:.1} MiB)\n";
    try note(io, try std.fmt.bufPrint(&msg, fmt, .{ out, mib(bytes) }));
}

/// The packed width for a weight the scope selects: the CLI's bits when
/// they are a packed tier (4 or 6), else 16 (an ordinary W16 entry).
fn packedFor(opts: Options, selected: bool, mixed_bits: usize) usize {
    if (!selected) return 16;
    if (opts.mixed) return mixed_bits;
    return if (opts.bits == 6 or opts.bits == 4 or opts.bits == 2) opts.bits else 16;
}

/// The mixed allocation's width per double-block class.
fn mixedDouble(d: zflux2_pack.Double) usize {
    return switch (d) {
        .to_q, .to_k, .add_q, .add_k => 4,
        .to_v, .add_v, .to_out, .to_add_out => 6,
        .ff_in, .ff_out => 2,
        .ffc_in, .ffc_out => 4,
    };
}

/// The mixed allocation's width per single-block class (both at 2 bits).
fn mixedSingle(s: zflux2_pack.Single) usize {
    return switch (s) {
        .qkv_mlp, .out => 2,
    };
}

/// The nine globals and every per-block norm weight, verbatim, under the raw
/// kinds (zflux2_pack.raw_kind / norm_kind).
fn emitRaw(allocator: std.mem.Allocator, stream: *Stream, loaded: zflux2.Loaded) !void {
    inline for (std.meta.fields(zflux2.Globals)) |f| {
        const g = @field(zflux2_pack.Global, f.name);
        const slot = zflux2_pack.globalSlot(g);
        try emitVerbatim(allocator, stream, slot, zflux2_pack.raw_kind, @field(loaded.globals, f.name));
    }
    const double_norms = .{ "norm_q", "norm_k", "norm_added_q", "norm_added_k" };
    for (loaded.doubles, 0..) |blk, i| {
        inline for (double_norms, 0..) |name, idx| {
            const slot = zflux2_pack.normSlotDouble(i, idx);
            try emitVerbatim(allocator, stream, slot, zflux2_pack.norm_kind, @field(blk, name));
        }
    }
    const single_norms = .{ "norm_q", "norm_k" };
    for (loaded.singles, 0..) |blk, i| {
        inline for (single_norms, 0..) |name, idx| {
            const slot = zflux2_pack.normSlotSingle(i, idx);
            try emitVerbatim(allocator, stream, slot, zflux2_pack.norm_kind, @field(blk, name));
        }
    }
}

fn emitVerbatim(
    allocator: std.mem.Allocator,
    stream: *Stream,
    slot: u32,
    kind: zpack_file.Kind,
    view: tensor.View,
) !void {
    const rows: u32 = if (view.shape.len == 2) @intCast(view.shape[0]) else 1;
    const cols: u32 = @intCast(if (view.shape.len == 2) view.shape[1] else view.shape[0]);
    const e = zpack_file.Entry{
        .family = stream.family,
        .layer = slot,
        .kind = kind,
        .rows = rows,
        .cols = cols,
        .group = zflux2_pack.rawGroup(view.dtype),
        .bytes = view.bytes,
    };
    try zpack_file.appendEntryAt(allocator, &stream.chunk, e, stream.offset);
    try stream.emit();
    stream.count += 1;
}

/// emit() with the adapter delta folded in when this weight is targeted.
fn emitAdapted(
    allocator: std.mem.Allocator,
    stream: *Stream,
    slot: u32,
    view: tensor.View,
    use_w6: usize,
    adapter: ?*const zlora.Adapter,
    stem: ?[]const u8,
    adapted: *usize,
) !void {
    const ad = adapter orelse return emit(allocator, stream, slot, view, use_w6);
    const name = stem orelse return emit(allocator, stream, slot, view, use_w6);
    const entry = try ad.find(name) orelse return emit(allocator, stream, slot, view, use_w6);
    const rows = try zlora.merge(allocator, view, entry, ad.scale);
    defer allocator.free(rows);
    adapted.* += 1;
    try emit(allocator, stream, slot, .{
        .dtype = .f32,
        .shape = view.shape,
        .bytes = std.mem.sliceAsBytes(rows),
    }, use_w6);
}

fn emit(
    allocator: std.mem.Allocator,
    stream: *Stream,
    slot: u32,
    view: tensor.View,
    use_w6: usize,
) !void {
    if (use_w6 == 6 or use_w6 == 4 or use_w6 == 2) {
        // SAFETY: every branch assigns packed_bytes before it is read or freed.
        var packed_bytes: []u8 = undefined;
        var rows: usize = 0;
        var cols: usize = 0;
        if (use_w6 == 6) {
            const w6 = try zw6.packW6(allocator, view, zflux2_pack.w6_group);
            packed_bytes = w6.bytes;
            rows = w6.rows;
            cols = w6.cols;
        } else if (use_w6 == 4) {
            const w4 = try zw4.packW4Q(allocator, view, zflux2_pack.w6_group, stream.w4_qmax);
            packed_bytes = w4.bytes;
            rows = w4.rows;
            cols = w4.cols;
        } else {
            const w2 = try zw2.packW2(allocator, view, zflux2_pack.w6_group);
            packed_bytes = w2.bytes;
            rows = w2.rows;
            cols = w2.cols;
        }
        defer allocator.free(packed_bytes);
        const e = zpack_file.Entry{
            .family = stream.family,
            .layer = slot,
            .kind = .flux2_weight,
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .group = @intCast(zflux2_pack.w6_group | (use_w6 << 16)),
            .bytes = packed_bytes,
        };
        try zpack_file.appendEntryAt(allocator, &stream.chunk, e, stream.offset);
        try stream.emit();
        stream.count += 1;
        return;
    }
    var w16 = try zpack.packW16(allocator, view);
    defer w16.deinit(allocator);
    const e = zpack_file.Entry{
        .family = stream.family,
        .layer = slot,
        .kind = .flux2_weight,
        .rows = @intCast(w16.rows),
        .cols = @intCast(w16.cols),
        .group = 0,
        .bytes = w16.bytes,
    };
    try zpack_file.appendEntryAt(allocator, &stream.chunk, e, stream.offset);
    try stream.emit();
    stream.count += 1;
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

/// The output's directory, created if missing (packs go beside the weights
/// or under runs/).
fn ensureParent(io: std.Io, path: []const u8) !void {
    const dir = std.fs.path.dirname(path) orelse return;
    try std.Io.Dir.cwd().createDirPath(io, dir);
}
