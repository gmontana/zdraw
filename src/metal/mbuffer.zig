//! Metal buffers for the linear fast path.
//!
//! Safetensor shards are mmap-backed, so model weights can be wrapped once as
//! no-copy Metal buffers. Temporary activation buffers stay explicit.

const std = @import("std");

const c = @import("metal_c.zig");
const mpacked = @import("mpacked.zig");
const tensor = @import("../pack/tensor.zig");
const zpack_file = @import("../pack/zpack_file.zig");

const max_cached = 2048;

pub const Bind = mpacked.Bind;

// Ownership handoff for a CPU activation slice that a runner copies into a GPU
// input buffer. When present, the runner frees this slice the instant the copy
// completes (see Buffer.fromInput), so the caller must not free it again.
pub const Recycle = struct {
    allocator: std.mem.Allocator,
    input: []f32,
};

pub const Cache = struct {
    device: *anyopaque,
    zero: *anyopaque,
    cached: [max_cached]Cached = [_]Cached{.{}} ** max_cached,
    cached_len: usize = 0,
    packs: mpacked.Cache,

    pub fn init(device: *anyopaque) !Cache {
        const zero = [_]u8{ 0, 0 };
        const handle = c.zdraw_metal_create_buffer_with_data(
            device,
            zero[0..].ptr,
            zero.len,
        ) orelse return error.MetalBufferFailed;
        return .{ .device = device, .zero = handle, .packs = mpacked.Cache.init(device) };
    }

    pub fn deinit(self: *Cache) void {
        self.clearSources();
        self.packs.deinit();
        c.zdraw_metal_release_buffer(self.zero);
        self.* = undefined;
    }

    pub fn clearSources(self: *Cache) void {
        for (self.cached[0..self.cached_len]) |item| {
            if (item.handle) |handle| c.zdraw_metal_release_weight_buffer(handle);
        }
        self.cached_len = 0;
    }

    pub fn bindBias(self: *Cache, bias: ?tensor.View, temp: *?Buffer) !Bind {
        if (bias) |b| return self.bindView(b, temp);
        return .{ .handle = self.zero, .offset = 0 };
    }

    pub fn setSidecar(self: *Cache, bytes: []const u8) void {
        self.packs.setSidecar(bytes);
    }

    pub fn sidecarBytes(self: *const Cache) []const u8 {
        return self.packs.sidecar;
    }

    pub fn bindView(self: *Cache, view: tensor.View, temp: *?Buffer) !Bind {
        if (view.source) |source| {
            if (try self.trySliceBind(view.bytes)) |exact| return exact;
            return .{ .handle = try self.sourceBuffer(source), .offset = source.offset };
        }
        const buf = try Buffer.fromBytes(self.device, view.bytes);
        temp.* = buf;
        return .{ .handle = buf.handle, .offset = 0 };
    }

    fn trySliceBind(self: *Cache, bytes: []const u8) !?Bind {
        for (self.cached[0..self.cached_len]) |item| {
            if (sameBytes(item.source.bytes, bytes)) {
                if (item.handle) |handle| return .{ .handle = handle, .offset = 0 };
            }
        }
        if (self.cached_len == max_cached) return error.MetalCacheFull;
        const handle = c.zdraw_metal_create_buffer_no_copy(
            self.device,
            bytes.ptr,
            bytes.len,
        ) orelse return null;
        self.cached[self.cached_len] = .{
            .source = .{ .bytes = bytes, .offset = 0 },
            .handle = handle,
        };
        self.cached_len += 1;
        return .{ .handle = handle, .offset = 0 };
    }

    pub fn bindW8(
        self: *Cache,
        allocator: std.mem.Allocator,
        view: tensor.View,
        group: usize,
    ) !Bind {
        return self.packs.bindW8(allocator, view, group);
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
        return self.packs.bindW8Layer(allocator, view, group, family, layer, kind);
    }

    fn sourceBuffer(self: *Cache, source: tensor.Source) !*anyopaque {
        for (self.cached[0..self.cached_len]) |item| {
            if (sameSource(item.source, source)) {
                if (item.handle) |handle| return handle;
            }
        }
        if (self.cached_len == max_cached) return error.MetalCacheFull;
        const handle = c.zdraw_metal_create_buffer_no_copy(
            self.device,
            source.bytes.ptr,
            source.bytes.len,
        ) orelse return error.MetalBufferFailed;
        self.cached[self.cached_len] = .{ .source = source, .handle = handle };
        self.cached_len += 1;
        return handle;
    }
};

pub const Buffer = struct {
    handle: *anyopaque,
    ownership: Ownership,

    pub const Ownership = enum {
        owned,
        borrowed,
    };

    pub fn borrow(handle: *anyopaque) Buffer {
        return .{ .handle = handle, .ownership = .borrowed };
    }

    pub fn empty(device: *anyopaque, size: usize) !Buffer {
        const handle = c.zdraw_metal_create_buffer(device, size) orelse {
            return error.MetalBufferFailed;
        };
        return .{ .handle = handle, .ownership = .owned };
    }

    pub fn fromBytes(device: *anyopaque, bytes: []const u8) !Buffer {
        const handle = c.zdraw_metal_create_buffer_with_data(
            device,
            bytes.ptr,
            bytes.len,
        ) orelse return error.MetalBufferFailed;
        return .{ .handle = handle, .ownership = .owned };
    }

    // Build the input buffer from a CPU slice and, if a recycle is supplied,
    // free that CPU slice immediately. create_buffer_with_data does a
    // synchronous newBufferWithBytes copy, so the source bytes are no longer
    // read after this returns; freeing here drops the old high-res activation
    // from the live footprint before the (much larger) GPU working set and
    // readback, without changing any computed value.
    pub fn fromInput(device: *anyopaque, input: []const f32, recycle: ?Recycle) !Buffer {
        const buf = try fromBytes(device, std.mem.sliceAsBytes(input));
        if (recycle) |r| r.allocator.free(r.input);
        return buf;
    }

    pub fn deinit(self: *Buffer) void {
        std.debug.assert(self.ownership == .owned);
        c.zdraw_metal_release_buffer(self.handle);
        self.* = undefined;
    }
};

const Cached = struct {
    source: tensor.Source = .{ .bytes = &.{}, .offset = 0 },
    handle: ?*anyopaque = null,
};

fn sameSource(a: tensor.Source, b: tensor.Source) bool {
    return sameBytes(a.bytes, b.bytes);
}

fn sameBytes(a: []const u8, b: []const u8) bool {
    return a.ptr == b.ptr and a.len == b.len;
}
