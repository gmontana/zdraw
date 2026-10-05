//! Load byte-level BPE vocabulary and merge files.

const std = @import("std");

const bpe = @import("bpe.zig");

pub const Error = error{
    InvalidBpe,
    BpeTooLarge,
};

pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    vocab_path: []const u8,
    merges_path: []const u8,
) !bpe.Encoder {
    var out = bpe.Encoder.init(allocator);
    errdefer out.deinit(allocator);

    const vocab_bytes = try readFile(io, allocator, vocab_path);
    defer allocator.free(vocab_bytes);
    try parseVocab(&out, allocator, vocab_bytes);

    const merge_bytes = try readFile(io, allocator, merges_path);
    defer allocator.free(merge_bytes);
    try parseMerges(&out, allocator, merge_bytes);
    return out;
}

fn parseVocab(
    encoder: *bpe.Encoder,
    allocator: std.mem.Allocator,
    bytes: []const u8,
) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
    });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidBpe;

    var iter = parsed.value.object.iterator();
    while (iter.next()) |entry| {
        if (entry.value_ptr.* != .integer or entry.value_ptr.integer < 0) {
            return error.InvalidBpe;
        }
        const id: u32 = @intCast(entry.value_ptr.integer);
        try encoder.addVocab(allocator, entry.key_ptr.*, id);
    }
}

fn parseMerges(
    encoder: *bpe.Encoder,
    allocator: std.mem.Allocator,
    bytes: []const u8,
) !void {
    var rank: u32 = 0;
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, "\r");
        if (line.len == 0 or line[0] == '#') continue;
        const mid = std.mem.indexOfScalar(u8, line, ' ') orelse {
            return error.InvalidBpe;
        };
        try encoder.addMerge(allocator, line[0..mid], line[mid + 1 ..], rank);
        rank += 1;
    }
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > 20 * 1024 * 1024) return error.BpeTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

test "parse small vocab and merge set" {
    var enc = bpe.Encoder.init(std.testing.allocator);
    defer enc.deinit(std.testing.allocator);

    try parseVocab(&enc, std.testing.allocator, "{\"a\":1,\"b\":2,\"ab\":3}");
    try parseMerges(&enc, std.testing.allocator, "#version: 0.2\na b\n");

    const ids = try enc.encode(std.testing.allocator, "ab", .{
        .im_start_id = 10,
        .im_end_id = 11,
        .eos_id = 11,
        .pad_id = 12,
    });
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{3}, ids);
}
