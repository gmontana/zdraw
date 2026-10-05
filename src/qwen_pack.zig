//! The text pack: the Qwen encoder's block linears packed at 4 bits
//! (absmax/7, group 64: the rule te-quant-final-20260905 measured as free)
//! and its other tensors (the token embedding, the final norm, the per-layer
//! norms) stored verbatim, keyed by tensor name through fixed slots. A Klein
//! release with a text pack drops the text_encoder shards (7.5 GB bf16).

const std = @import("std");

const qnames = @import("qwen_names.zig");
const shards = @import("shards.zig");
const tensor = @import("tensor.zig");
const zflux2_pack = @import("zflux2_pack.zig");
const zpack_file = @import("zpack_file.zig");
const zw4 = @import("zw4.zig");

pub const pack_name = "zdraw-klein-text-w4.zpack";
pub const family = zpack_file.Family.text;
/// Packed linears reuse the packed-weight kind; verbatim tensors the raw kind.
pub const linear_kind = zpack_file.Kind.flux2_weight;
pub const raw_kind = zpack_file.Kind.flux2_raw;
pub const bits: u8 = 4;
pub const embed_slot: u32 = 0;
pub const final_slot: u32 = 1;
const layer_base: u32 = 100;
const layer_stride: u32 = 16;

pub fn layerSlot(layer: usize, part: qnames.Layer) u32 {
    const l: u32 = @intCast(layer);
    return layer_base + l * layer_stride + @intFromEnum(part);
}

pub fn isLinear(part: qnames.Layer) bool {
    return switch (part) {
        .q, .k, .v, .o, .gate, .up, .down => true,
        .input_norm, .post_norm, .q_norm, .k_norm => false,
    };
}

/// The slot a tensor name maps to; null for a name the pack never carries.
pub fn slotOf(name: []const u8) ?u32 {
    if (std.mem.eql(u8, name, qnames.embed)) return embed_slot;
    if (std.mem.eql(u8, name, qnames.final_norm)) return final_slot;
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    const tail = rest[dot + 1 ..];
    inline for (std.meta.fields(qnames.Layer)) |f| {
        const part: qnames.Layer = @enumFromInt(f.value);
        if (std.mem.eql(u8, tail, qnames.suffix(part))) return layerSlot(layer, part);
    }
    return null;
}

/// Every view a text pack carries, by name; the store consults it before
/// the shards, and alone when there are none. Lives on the heap because the
/// store keeps a pointer to `overrides`.
pub const Table = struct {
    names: [][]const u8,
    views: []tensor.View,
    dims: [][2]usize,
    overrides: shards.Overrides,

    pub fn deinit(self: *Table, allocator: std.mem.Allocator) void {
        for (self.names) |n| allocator.free(n);
        allocator.free(self.names);
        allocator.free(self.views);
        allocator.free(self.dims);
        self.* = undefined;
    }
};

/// The number of layers the pack carries: consecutive input norms from 0.
pub fn layerCount(pack: []const u8) !usize {
    var n: usize = 0;
    while (true) : (n += 1) {
        const slot = layerSlot(n, .input_norm);
        if ((try zpack_file.findIn(pack, family, slot, raw_kind)) == null) return n;
    }
}

pub fn create(allocator: std.mem.Allocator, pack: []const u8) !*Table {
    const layers = try layerCount(pack);
    if (layers == 0) return error.MissingWeights;
    const count = 2 + layers * std.meta.fields(qnames.Layer).len;
    const names = try allocator.alloc([]const u8, count);
    errdefer allocator.free(names);
    const views = try allocator.alloc(tensor.View, count);
    errdefer allocator.free(views);
    const dims = try allocator.alloc([2]usize, count);
    errdefer allocator.free(dims);
    var filled: usize = 0;
    errdefer for (names[0..filled]) |n| allocator.free(n);
    names[0] = try allocator.dupe(u8, qnames.embed);
    filled = 1;
    views[0] = try rawAt(pack, embed_slot, &dims[0]);
    names[1] = try allocator.dupe(u8, qnames.final_norm);
    filled = 2;
    views[1] = try rawAt(pack, final_slot, &dims[1]);
    for (0..layers) |layer| {
        inline for (std.meta.fields(qnames.Layer)) |f| {
            const part: qnames.Layer = @enumFromInt(f.value);
            const i = filled;
            names[i] = try qnames.layerName(allocator, layer, part);
            filled += 1;
            const slot = layerSlot(layer, part);
            views[i] = if (isLinear(part))
                try linearAt(pack, slot, &dims[i])
            else
                try rawAt(pack, slot, &dims[i]);
        }
    }
    const t = try allocator.create(Table);
    t.* = .{
        .names = names,
        .views = views,
        .dims = dims,
        .overrides = .{ .names = names, .views = views },
    };
    return t;
}

pub fn destroy(allocator: std.mem.Allocator, t: *Table) void {
    t.deinit(allocator);
    allocator.destroy(t);
}

/// A verbatim tensor; a vector was stored as one row and comes back 1-D.
fn rawAt(pack: []const u8, slot: u32, dims: *[2]usize) !tensor.View {
    const found = (try zpack_file.findIn(pack, family, slot, raw_kind)) orelse
        return error.MissingTensor;
    const dtype = zflux2_pack.rawDType(found.group) orelse return error.InvalidShape;
    dims.* = .{ found.rows, found.cols };
    const shape: []const usize = if (found.rows == 1) dims[1..2] else dims[0..2];
    const view = tensor.View{
        .dtype = dtype,
        .shape = shape,
        .bytes = found.bytes,
        .source = .{ .bytes = pack, .offset = found.offset },
    };
    try view.check();
    return view;
}

/// A 4-bit linear as the `.u8` marker view the resident encoder binds no-copy.
fn linearAt(pack: []const u8, slot: u32, dims: *[2]usize) !tensor.View {
    const found = (try zpack_file.findInBits(pack, family, slot, linear_kind, bits)) orelse
        return error.MissingTensor;
    if (zpack_file.entryGroupSize(found) != zflux2_pack.w6_group) return error.InvalidShape;
    const want = zw4.byteLen(found.rows, found.cols, zflux2_pack.w6_group);
    if (found.bytes.len != want) return error.InvalidShape;
    dims.* = .{ found.rows, found.cols };
    return .{
        .dtype = .u8,
        .shape = dims[0..2],
        .bytes = found.bytes,
        .source = .{ .bytes = pack, .offset = found.offset },
        .packed_bits = bits,
    };
}

fn testEntry(
    slot: u32,
    kind: zpack_file.Kind,
    rows: u32,
    cols: u32,
    group: u32,
    bytes: []const u8,
) zpack_file.Entry {
    return .{
        .family = family,
        .layer = slot,
        .kind = kind,
        .rows = rows,
        .cols = cols,
        .group = group,
        .bytes = bytes,
    };
}

test "slots follow the tensor names" {
    try std.testing.expectEqual(@as(?u32, embed_slot), slotOf(qnames.embed));
    try std.testing.expectEqual(@as(?u32, final_slot), slotOf(qnames.final_norm));
    const down = "model.layers.3.mlp.down_proj.weight";
    try std.testing.expectEqual(@as(?u32, layerSlot(3, .down)), slotOf(down));
    try std.testing.expectEqual(@as(?u32, null), slotOf("model.layers.x.mlp.down_proj.weight"));
    try std.testing.expectEqual(@as(?u32, null), slotOf("lm_head.weight"));
}

test "a table serves packed linears and verbatim tensors by name" {
    const allocator = std.testing.allocator;
    const rows = 2;
    const cols = 64;
    const f = [_]f32{0.5} ** (rows * cols);
    const lin = tensor.View{
        .dtype = .f32,
        .shape = &.{ rows, cols },
        .bytes = std.mem.sliceAsBytes(&f),
    };
    var w4 = try zw4.packW4(allocator, lin, zflux2_pack.w6_group);
    defer w4.deinit(allocator);
    const norm = [_]u8{0x3F} ** (cols * 2);
    const embed = [_]u8{0x3F} ** (4 * cols * 2);
    const raw_group = zflux2_pack.rawGroup(.bf16);
    const lin_group: u32 = @intCast(zflux2_pack.w6_group | (@as(usize, bits) << 16));
    var entries: std.ArrayList(zpack_file.Entry) = .empty;
    defer entries.deinit(allocator);
    try entries.append(allocator, testEntry(embed_slot, raw_kind, 4, cols, raw_group, &embed));
    try entries.append(allocator, testEntry(final_slot, raw_kind, 1, cols, raw_group, &norm));
    inline for (std.meta.fields(qnames.Layer)) |fld| {
        const part: qnames.Layer = @enumFromInt(fld.value);
        const slot = layerSlot(0, part);
        if (isLinear(part)) {
            try entries.append(allocator, testEntry(slot, linear_kind, rows, cols, lin_group, w4.bytes));
        } else {
            try entries.append(allocator, testEntry(slot, raw_kind, 1, cols, raw_group, &norm));
        }
    }
    var pack: std.ArrayList(u8) = .empty;
    defer pack.deinit(allocator);
    try zpack_file.append(allocator, &pack, entries.items);
    const t = try create(allocator, pack.items);
    defer destroy(allocator, t);
    try std.testing.expectEqual(@as(usize, 2 + 11), t.names.len);
    const q = t.overrides.find("model.layers.0.self_attn.q_proj.weight") orelse
        return error.TestUnexpectedResult;
    try std.testing.expectEqual(tensor.DType.u8, q.dtype);
    try std.testing.expectEqual(@as(u8, 4), q.packed_bits);
    try std.testing.expectEqual(@as(usize, rows), q.shape[0]);
    const n = t.overrides.find(qnames.final_norm) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 1), n.shape.len);
    try std.testing.expectEqual(@as(usize, cols), n.shape[0]);
    const e = t.overrides.find(qnames.embed) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 2), e.shape.len);
    try std.testing.expect(t.overrides.find("lm_head.weight") == null);
}
