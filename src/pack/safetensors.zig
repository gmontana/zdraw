//! Safetensors metadata reader. An eight-byte little-endian length prefixes
//! the JSON header; tensor offsets are relative to the payload that follows it.
//! Caps header size and validates shapes/offsets without reading tensor data.

const std = @import("std");

pub const max_header_bytes = 128 * 1024 * 1024;

pub const TensorInfo = struct {
    name: []u8,
    dtype: []u8,
    shape: []usize,
    offsets: [2]u64,

    pub fn deinit(self: TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dtype);
        allocator.free(self.shape);
    }
};

pub const Header = struct {
    tensors: []TensorInfo,

    pub fn deinit(self: *Header, allocator: std.mem.Allocator) void {
        for (self.tensors) |tensor| tensor.deinit(allocator);
        allocator.free(self.tensors);
    }
};

pub const Error = error{
    HeaderTooLarge,
    InvalidHeader,
    InvalidShape,
    InvalidOffsets,
};

pub fn readHeader(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !Header {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var read_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buffer);

    const header_len = try reader.interface.takeInt(u64, .little);
    if (header_len > max_header_bytes) return error.HeaderTooLarge;

    const header_bytes = try allocator.alloc(u8, @intCast(header_len));
    defer allocator.free(header_bytes);
    try reader.interface.readSliceAll(header_bytes);

    return parseHeader(allocator, header_bytes);
}

pub fn parseHeader(allocator: std.mem.Allocator, bytes: []const u8) !Header {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
    });
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidHeader;

    var tensors = try std.ArrayList(TensorInfo).initCapacity(
        allocator,
        parsed.value.object.count(),
    );
    errdefer {
        for (tensors.items) |tensor| tensor.deinit(allocator);
        tensors.deinit(allocator);
    }

    var iter = parsed.value.object.iterator();
    while (iter.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        const tensor = try parseTensor(allocator, entry.key_ptr.*, entry.value_ptr.*);
        try tensors.append(allocator, tensor);
    }

    return .{ .tensors = try tensors.toOwnedSlice(allocator) };
}

fn parseTensor(
    allocator: std.mem.Allocator,
    name: []const u8,
    value: std.json.Value,
) !TensorInfo {
    if (value != .object) return error.InvalidHeader;
    const object = value.object;

    const dtype_value = object.get("dtype") orelse return error.InvalidHeader;
    const shape_value = object.get("shape") orelse return error.InvalidHeader;
    const offsets_value = object.get("data_offsets") orelse return error.InvalidHeader;

    if (dtype_value != .string) return error.InvalidHeader;
    if (shape_value != .array) return error.InvalidShape;
    if (offsets_value != .array or offsets_value.array.items.len != 2) {
        return error.InvalidOffsets;
    }

    const shape = try allocator.alloc(usize, shape_value.array.items.len);
    errdefer allocator.free(shape);
    for (shape_value.array.items, 0..) |dim, idx| {
        if (dim != .integer or dim.integer < 0) return error.InvalidShape;
        shape[idx] = @intCast(dim.integer);
    }

    const start = try parseOffset(offsets_value.array.items[0]);
    const end = try parseOffset(offsets_value.array.items[1]);
    if (start > end) return error.InvalidOffsets;

    return .{
        .name = try allocator.dupe(u8, name),
        .dtype = try allocator.dupe(u8, dtype_value.string),
        .shape = shape,
        .offsets = .{ start, end },
    };
}

fn parseOffset(value: std.json.Value) !u64 {
    if (value != .integer or value.integer < 0) return error.InvalidOffsets;
    return @intCast(value.integer);
}

test "parse simple header" {
    const json =
        \\{"x":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}
    ;
    var header = try parseHeader(std.testing.allocator, json);
    defer header.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), header.tensors.len);
    try std.testing.expectEqualStrings("x", header.tensors[0].name);
    try std.testing.expectEqualStrings("F32", header.tensors[0].dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, header.tensors[0].shape);
    try std.testing.expectEqual(@as(u64, 24), header.tensors[0].offsets[1]);
}

test "reject reversed offsets" {
    const json =
        \\{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}
    ;
    try std.testing.expectError(error.InvalidOffsets, parseHeader(std.testing.allocator, json));
}
