//! A tiny lookup layer over safetensors shard files.

const std = @import("std");

const tensor = @import("tensor.zig");
const tensor_file = @import("tensor_file.zig");
const weight_index = @import("weight_index.zig");

pub const Error = error{
    MissingShard,
    MissingTensor,
};

pub const File = struct {
    name: []u8,
    mapped: tensor_file.Mapped,

    fn deinit(
        self: *File,
        io: std.Io,
        allocator: std.mem.Allocator,
    ) void {
        allocator.free(self.name);
        self.mapped.deinit(io, allocator);
        self.* = undefined;
    }
};

/// Views served by name ahead of the shards (a text pack's table). The
/// store borrows it; the owner keeps it alive for the store's lifetime.
pub const Overrides = struct {
    names: []const []const u8,
    views: []const tensor.View,

    pub fn find(self: Overrides, name: []const u8) ?tensor.View {
        for (self.names, self.views) |n, v| {
            if (std.mem.eql(u8, n, name)) return v;
        }
        return null;
    }
};

pub const Store = struct {
    files: []File,
    overrides: ?*const Overrides = null,

    pub fn deinit(
        self: *Store,
        io: std.Io,
        allocator: std.mem.Allocator,
    ) void {
        closeFiles(self.files, io, allocator);
        allocator.free(self.files);
        self.* = undefined;
    }

    pub fn view(
        self: *const Store,
        index: weight_index.Index,
        name: []const u8,
    ) !tensor.View {
        if (self.overrides) |o| {
            if (o.find(name)) |v| return v;
        }
        const shard = index.find(name) orelse return error.MissingTensor;
        const file = self.find(shard) orelse return error.MissingShard;
        return (try file.mapped.view(name)) orelse error.MissingTensor;
    }

    fn find(self: *const Store, name: []const u8) ?*const File {
        for (self.files) |*file| {
            if (std.mem.eql(u8, file.name, name)) return file;
        }
        return null;
    }
};

/// A store with no shards: every view comes from the overrides.
pub fn packOnly(allocator: std.mem.Allocator, overrides: *const Overrides) !Store {
    return .{ .files = try allocator.alloc(File, 0), .overrides = overrides };
}

pub fn open(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    index: weight_index.Index,
) !Store {
    var files = try std.ArrayList(File).initCapacity(allocator, 4);
    errdefer {
        closeFiles(files.items, io, allocator);
        files.deinit(allocator);
    }

    for (index.entries) |entry| {
        if (hasFile(files.items, entry.file)) continue;
        var file = try openFile(io, allocator, root, entry.file);
        var kept = false;
        defer if (!kept) file.deinit(io, allocator);
        try files.append(allocator, file);
        kept = true;
    }

    return .{ .files = try files.toOwnedSlice(allocator) };
}

fn openFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    name: []const u8,
) !File {
    const owned = try allocator.dupe(u8, name);
    errdefer allocator.free(owned);
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, name });
    defer allocator.free(path);

    var mapped = try tensor_file.open(io, allocator, path);
    errdefer mapped.deinit(io, allocator);
    return .{ .name = owned, .mapped = mapped };
}

fn hasFile(files: []const File, name: []const u8) bool {
    for (files) |file| {
        if (std.mem.eql(u8, file.name, name)) return true;
    }
    return false;
}

fn closeFiles(files: []File, io: std.Io, allocator: std.mem.Allocator) void {
    for (files) |*file| file.deinit(io, allocator);
}

fn fixturePath(
    allocator: std.mem.Allocator,
    tmp: std.testing.TmpDir,
    name: []const u8,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/{s}",
        .{ tmp.sub_path, name },
    );
}

fn writeShard(io: std.Io, path: []const u8) !void {
    const json =
        \\{"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}
    ;
    var bytes: [8 + json.len + 2]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    @memcpy(bytes[8..][0..json.len], json);
    bytes[8 + json.len] = 0x00;
    bytes[8 + json.len + 1] = 0x3c;

    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [256]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    try writer.interface.writeAll(&bytes);
    try writer.interface.flush();
}

test "open unique shards and read tensor view" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try fixturePath(std.testing.allocator, tmp, "model");
    defer std.testing.allocator.free(root);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, root);

    const path = try fixturePath(std.testing.allocator, tmp, "model/one.safetensors");
    defer std.testing.allocator.free(path);
    try writeShard(std.testing.io, path);

    const json =
        \\{"metadata":{"total_size":2},"weight_map":{"x":"one.safetensors"}}
    ;
    var index = try weight_index.parse(std.testing.allocator, json);
    defer index.deinit(std.testing.allocator);

    var store = try open(std.testing.io, std.testing.allocator, root, index);
    defer store.deinit(std.testing.io, std.testing.allocator);

    const view = try store.view(index, "x");
    try std.testing.expectApproxEqAbs(1.0, try view.atF32(0), 0.0001);
}

test "overrides answer before the shards, and alone in a pack-only store" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1.0, 2.0 };
    const names = [_][]const u8{"model.norm.weight"};
    const views = [_]tensor.View{.{
        .dtype = .f32,
        .shape = &.{2},
        .bytes = std.mem.sliceAsBytes(&data),
    }};
    const overrides = Overrides{ .names = &names, .views = &views };
    var store = try packOnly(allocator, &overrides);
    defer allocator.free(store.files);
    var index = try weight_index.empty(allocator);
    defer index.deinit(allocator);
    const got = try store.view(index, "model.norm.weight");
    try std.testing.expectEqual(@as(usize, 2), got.shape[0]);
    const missing = store.view(index, "model.embed_tokens.weight");
    try std.testing.expectError(error.MissingTensor, missing);
}
