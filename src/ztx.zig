//! Loaded Z-Image transformer views.
//!
//! This owns the mapped safetensors store and the small arrays of block views.
//! The views themselves do not allocate.

const std = @import("std");

const shards = @import("shards.zig");
const weight_index = @import("weight_index.zig");
const zblocks = @import("zblocks.zig");
const zconfig = @import("zimage_config.zig");
const zglobal = @import("zglobal.zig");

pub const Loaded = struct {
    store: shards.Store,
    globals: zglobal.Views,
    noise: zblocks.List,
    context: zblocks.List,
    layers: zblocks.List,

    pub fn deinit(
        self: *Loaded,
        io: std.Io,
        allocator: std.mem.Allocator,
    ) void {
        self.layers.deinit(allocator);
        self.context.deinit(allocator);
        self.noise.deinit(allocator);
        self.store.deinit(io, allocator);
        self.* = undefined;
    }
};

pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    cfg: zconfig.Transformer,
    index: weight_index.Index,
) !Loaded {
    const tx_root = try rootPath(allocator, root);
    defer allocator.free(tx_root);

    var store = try shards.open(io, allocator, tx_root, index);
    errdefer store.deinit(io, allocator);
    const globals = try zglobal.load(&store, index);
    var noise = try loadList(allocator, &store, index, "noise_refiner", cfg.refiner_layers, true);
    errdefer noise.deinit(allocator);
    var context = try loadList(
        allocator,
        &store,
        index,
        "context_refiner",
        cfg.refiner_layers,
        false,
    );
    errdefer context.deinit(allocator);
    const layers = try zblocks.load(allocator, &store, index, "layers", cfg.layers, true);

    return .{
        .store = store,
        .globals = globals,
        .noise = noise,
        .context = context,
        .layers = layers,
    };
}

pub fn rootPath(allocator: std.mem.Allocator, root: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/transformer", .{root});
}

fn loadList(
    allocator: std.mem.Allocator,
    store: *const shards.Store,
    index: weight_index.Index,
    prefix: []const u8,
    count: usize,
    modulation: bool,
) !zblocks.List {
    return zblocks.load(allocator, store, index, prefix, count, modulation);
}

test "transformer root path" {
    const path = try rootPath(std.testing.allocator, "model");
    defer std.testing.allocator.free(path);

    try std.testing.expectEqualStrings("model/transformer", path);
}
