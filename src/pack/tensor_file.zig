//! Mmap-backed safetensors file access.

const std = @import("std");

const safetensors = @import("safetensors.zig");
const tensor = @import("tensor.zig");

pub const TensorBytes = struct {
    info: *const safetensors.TensorInfo,
    bytes: []const u8,
};

pub const Mapped = struct {
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    header: safetensors.Header,
    data_start: usize,

    pub fn deinit(
        self: *Mapped,
        io: std.Io,
        allocator: std.mem.Allocator,
    ) void {
        self.header.deinit(allocator);
        self.map.destroy(io);
        self.file.close(io);
        self.* = undefined;
    }

    /// The raw safetensors header JSON (tensor table plus `__metadata__`),
    /// for callers that need producer metadata the tensor table drops.
    pub fn headerJson(self: *const Mapped) []const u8 {
        return self.map.memory[8..self.data_start];
    }

    pub fn find(self: *const Mapped, name: []const u8) ?TensorBytes {
        for (self.header.tensors) |*info| {
            if (!std.mem.eql(u8, info.name, name)) continue;
            const start = self.data_start + toUsize(info.offsets[0]);
            const end = self.data_start + toUsize(info.offsets[1]);
            return .{ .info = info, .bytes = self.map.memory[start..end] };
        }
        return null;
    }

    pub fn view(self: *const Mapped, name: []const u8) !?tensor.View {
        const found = self.find(name) orelse return null;
        const view_out = tensor.View{
            .dtype = try tensor.parseDType(found.info.dtype),
            .shape = found.info.shape,
            .bytes = found.bytes,
            .source = .{
                .bytes = self.map.memory,
                .offset = dataStart(self, found.info.offsets[0]),
            },
        };
        try view_out.check();
        return view_out;
    }
};

pub const Error = error{
    FileTooLarge,
    InvalidHeader,
    InvalidOffsets,
};

pub fn open(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !Mapped {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);

    const stat = try file.stat(io);
    const size = try fitUsize(stat.size);
    var map = try std.Io.File.MemoryMap.create(io, file, .{
        .len = size,
        .protection = .{ .read = true, .write = false },
    });
    errdefer map.destroy(io);

    if (map.memory.len < 8) return error.InvalidHeader;
    const header_len = std.mem.readInt(u64, map.memory[0..8], .little);
    if (header_len > safetensors.max_header_bytes) return error.HeaderTooLarge;

    const start = try fitUsize(8 + header_len);
    if (start > map.memory.len) return error.InvalidHeader;

    var header = try safetensors.parseHeader(allocator, map.memory[8..start]);
    errdefer header.deinit(allocator);
    try checkOffsets(header, map.memory.len - start);

    return .{
        .file = file,
        .map = map,
        .header = header,
        .data_start = start,
    };
}

fn dataStart(mapped: *const Mapped, offset: u64) usize {
    return mapped.data_start + toUsize(offset);
}

fn checkOffsets(header: safetensors.Header, data_len: usize) !void {
    for (header.tensors) |info| {
        const end = try fitUsize(info.offsets[1]);
        if (end > data_len) return error.InvalidOffsets;
    }
}

fn fitUsize(value: u64) !usize {
    if (value > std.math.maxInt(usize)) return error.FileTooLarge;
    return @intCast(value);
}

fn toUsize(value: u64) usize {
    return @intCast(value);
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

fn writeFixture(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    json: []const u8,
    payload: []const u8,
) !void {
    var bytes = try std.ArrayList(u8).initCapacity(allocator, 8 + json.len + payload.len);
    defer bytes.deinit(allocator);

    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, json.len, .little);
    try bytes.appendSlice(allocator, &prefix);
    try bytes.appendSlice(allocator, json);
    try bytes.appendSlice(allocator, payload);

    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [256]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    try writer.interface.writeAll(bytes.items);
    try writer.interface.flush();
}

test "map tensor bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try fixturePath(std.testing.allocator, tmp, "ok.safetensors");
    defer std.testing.allocator.free(path);

    const json =
        \\{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}
    ;
    try writeFixture(std.testing.io, std.testing.allocator, path, json, "abcd");

    var mapped = try open(std.testing.io, std.testing.allocator, path);
    defer mapped.deinit(std.testing.io, std.testing.allocator);

    const found = mapped.find("x").?;
    try std.testing.expectEqualStrings("U8", found.info.dtype);
    try std.testing.expectEqualSlices(u8, "abcd", found.bytes);

    const view = (try mapped.view("x")).?;
    const elem_count: usize = 4;
    try std.testing.expectEqual(elem_count, try view.elems());
}

test "reject tensor outside file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try fixturePath(std.testing.allocator, tmp, "bad.safetensors");
    defer std.testing.allocator.free(path);

    const json =
        \\{"x":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}
    ;
    try writeFixture(std.testing.io, std.testing.allocator, path, json, "abcd");
    try std.testing.expectError(
        error.InvalidOffsets,
        open(std.testing.io, std.testing.allocator, path),
    );
}
