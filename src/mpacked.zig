//! Packed-weight Metal buffer cache.

const std = @import("std");

const c = @import("metal_c.zig");
const env = @import("env.zig");
const tensor = @import("tensor.zig");
const zpack = @import("zpack.zig");
const zpack_file = @import("zpack_file.zig");

const max_cached = 64;

pub const Bind = struct {
    handle: *anyopaque,
    offset: usize,
};

pub const Cache = struct {
    device: *anyopaque,
    cached: [max_cached]Item = [_]Item{.{}} ** max_cached,
    len: usize = 0,
    sidecar: []const u8 = &.{},
    sidecar_handle: ?*anyopaque = null,
    require_sidecar: bool = false,

    pub fn init(device: *anyopaque) Cache {
        return .{ .device = device, .require_sidecar = strictEnv() };
    }

    pub fn deinit(self: *Cache) void {
        for (self.cached[0..self.len]) |item| {
            if (item.owned) {
                if (item.handle) |handle| c.zdraw_metal_release_buffer(handle);
            }
        }
        if (self.sidecar_handle) |handle| c.zdraw_metal_release_weight_buffer(handle);
        self.* = undefined;
    }

    pub fn setSidecar(self: *Cache, bytes: []const u8) void {
        self.sidecar = bytes;
    }

    pub fn bindW8(
        self: *Cache,
        allocator: std.mem.Allocator,
        view: tensor.View,
        group: usize,
    ) !Bind {
        return self.bindEntry(allocator, view, group, null);
    }

    pub fn bindW8Layer(
        self: *Cache,
        allocator: std.mem.Allocator,
        view: tensor.View,
        group: usize,
        family: zpack_file.Family,
        layer: u32,
        kind: zpack_file.Kind,
    ) !Bind {
        return self.bindEntry(allocator, view, group, .{
            .family = family,
            .layer = layer,
            .kind = kind,
        });
    }

    fn bindEntry(
        self: *Cache,
        allocator: std.mem.Allocator,
        view: tensor.View,
        group: usize,
        entry: ?EntryKey,
    ) !Bind {
        const key = try packedKey(view, group);
        if (self.find(key)) |item| return item.bind();
        if (self.len == max_cached) return error.MetalCacheFull;
        if (try self.sidecarEntry(entry, key)) |found| return try self.addSidecar(key, found);
        if (entry != null and self.require_sidecar) return error.MissingPackedSidecar;
        var w8 = try zpack.packW8(allocator, view, group);
        defer w8.deinit(allocator);
        return try self.addOwned(key, w8.bytes);
    }

    fn find(self: *Cache, key: PackedKey) ?Item {
        for (self.cached[0..self.len]) |item| {
            if (samePacked(item.key, key)) return item;
        }
        return null;
    }

    fn addOwned(self: *Cache, key: PackedKey, bytes: []const u8) !Bind {
        const handle = c.zdraw_metal_create_buffer_with_data(
            self.device,
            bytes.ptr,
            bytes.len,
        ) orelse return error.MetalBufferFailed;
        return self.push(.{ .key = key, .handle = handle, .owned = true });
    }

    fn addSidecar(self: *Cache, key: PackedKey, entry: zpack_file.Entry) !Bind {
        const handle = try self.sidecarBuffer();
        return self.push(.{ .key = key, .handle = handle, .offset = entry.offset });
    }

    fn push(self: *Cache, item: Item) Bind {
        self.cached[self.len] = item;
        self.len += 1;
        return item.bind();
    }

    fn sidecarBuffer(self: *Cache) !*anyopaque {
        if (self.sidecar_handle) |handle| return handle;
        const handle = c.zdraw_metal_create_buffer_no_copy(
            self.device,
            self.sidecar.ptr,
            self.sidecar.len,
        ) orelse return error.MetalBufferFailed;
        self.sidecar_handle = handle;
        return handle;
    }

    fn sidecarEntry(self: *Cache, entry: ?EntryKey, key: PackedKey) !?zpack_file.Entry {
        const want = entry orelse return null;
        if (self.sidecar.len == 0) return null;
        const got = (try zpack_file.findIn(
            self.sidecar,
            want.family,
            want.layer,
            want.kind,
        )) orelse return null;
        if (got.rows != key.rows or got.cols != key.cols or got.group != key.group) {
            return error.InvalidShape;
        }
        return got;
    }
};

const Item = struct {
    key: PackedKey = .{},
    handle: ?*anyopaque = null,
    offset: usize = 0,
    owned: bool = false,

    fn bind(self: Item) Bind {
        return .{ .handle = self.handle.?, .offset = self.offset };
    }
};

const PackedKey = struct {
    ptr: [*]const u8 = &.{},
    len: usize = 0,
    rows: usize = 0,
    cols: usize = 0,
    group: usize = 0,
};

const EntryKey = struct {
    family: zpack_file.Family,
    layer: u32,
    kind: zpack_file.Kind,
};

fn packedKey(view: tensor.View, group: usize) !PackedKey {
    if (view.shape.len != 2) return error.InvalidShape;
    return .{
        .ptr = view.bytes.ptr,
        .len = view.bytes.len,
        .rows = view.shape[0],
        .cols = view.shape[1],
        .group = group,
    };
}

fn samePacked(a: PackedKey, b: PackedKey) bool {
    return a.ptr == b.ptr and a.len == b.len and a.rows == b.rows and
        a.cols == b.cols and a.group == b.group;
}

fn strictEnv() bool {
    return env.flag("ZDRAW_REQUIRE_ZPACK", false);
}
