//! Tiny `.zpack` sidecar container for derived packed weights.

const std = @import("std");
const builtin = @import("builtin");

const magic = "ZDWPACK1";
const version: u32 = 4;
const min_version: u32 = 2;
const payload_align: usize = 16;

pub const Family = enum(u8) {
    main = 1,
    noise = 2,
    context = 3,
    flux2 = 4,
    /// The Qwen text encoder's pack (qwen_pack.zig).
    text = 5,
};

pub const Kind = enum(u8) {
    ffn_down = 1,
    ffn_gate = 2,
    ffn_up = 3,
    q = 4,
    k = 5,
    v = 6,
    proj = 7,
    ffn_gateup = 8, // fused [gate; up] image, W16 only
    flux2_weight = 9,
    /// A tensor stored verbatim in its checkpoint dtype (the Klein globals):
    /// group's high 16 bits carry the dtype code (zflux2_pack.rawGroup).
    flux2_raw = 10,
    /// A per-block norm weight stored verbatim (same encoding as flux2_raw).
    flux2_norm = 11,
};

pub const Entry = struct {
    family: Family = .main,
    layer: u32,
    kind: Kind,
    rows: u32,
    cols: u32,
    group: u32,
    offset: usize = 0,
    bytes: []const u8,
};

// group encoding: low 16 = group size, high 16 = code bits; legacy high==0
// means group==0 -> f16, group>0 -> 8-bit codes.
pub fn entryBits(entry: Entry) u32 {
    const hi = entry.group >> 16;
    return if (hi != 0) hi else if (entry.group == 0) 16 else 8;
}

pub fn entryGroupSize(entry: Entry) u32 {
    return entry.group & 0xFFFF;
}

pub const Mapped = struct {
    file: std.Io.File,
    map: std.Io.File.MemoryMap,

    pub fn bytes(self: *const Mapped) []const u8 {
        return self.map.memory;
    }

    pub fn deinit(self: *Mapped, io: std.Io) void {
        self.map.destroy(io);
        self.file.close(io);
        self.* = undefined;
    }
};

pub fn open(io: std.Io, path: []const u8) !Mapped {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    const stat = try file.stat(io);
    var map = try std.Io.File.MemoryMap.create(io, file, .{
        .len = try fitUsize(stat.size),
        .protection = .{ .read = true, .write = false },
    });
    errdefer map.destroy(io);
    var pos: usize = 0;
    _ = try checkHeader(map.memory, &pos);
    _ = try takeU32(map.memory, &pos);
    readAhead(file, map.memory.len);
    return .{ .file = file, .map = map };
}

/// Asynchronous sequential read-ahead of the whole sidecar (Darwin
/// F_RDADVISE, one hint per GiB). A one-shot render streams the pack from
/// the page cache; on a churned cache the mmap faults pay SSD latency per
/// layer, which is the unattributed cross-session shift in the one-shot
/// Klein numbers (ledger klein-oneshot-prefetch-20260827). The hint costs no
/// RSS (file-backed pages) and never changes numerics. On by default since
/// the cold-cache A/B (`sudo purge` before every render: median -1.1 s,
/// 6/6 pairs); ZDRAW_PACK_READAHEAD=0 disables it. Warm-cache runs are a wash.
fn readAhead(file: std.Io.File, len: usize) void {
    if (builtin.os.tag != .macos) return;
    if (std.c.getenv("ZDRAW_PACK_READAHEAD")) |raw| {
        if (raw[0] == '0') return;
    }
    const chunk: usize = 1 << 30;
    var off: usize = 0;
    while (off < len) : (off += chunk) {
        const count: usize = @min(chunk, len - off);
        var advice = Radvisory{ .ra_offset = @intCast(off), .ra_count = @intCast(count) };
        _ = std.c.fcntl(file.handle, std.c.F.RDADVISE, &advice);
    }
}

/// Darwin `struct radvisory` for F_RDADVISE.
const Radvisory = extern struct {
    ra_offset: i64,
    ra_count: c_int,
};

pub fn append(allocator: std.mem.Allocator, out: *std.ArrayList(u8), entries: []const Entry) !void {
    try beginStream(allocator, out, entries.len);
    for (entries) |entry| try appendEntryAt(allocator, out, entry, 0);
}

// Streaming writer: emit the header once, then entry chunks whose payload
// alignment is computed against `base` (the absolute file offset where the
// chunk starts), so large sidecars never need one in-memory image.
pub fn beginStream(allocator: std.mem.Allocator, out: *std.ArrayList(u8), count: usize) !void {
    try out.appendSlice(allocator, magic);
    try putU32(allocator, out, version);
    try putU32(allocator, out, @intCast(count));
}

pub fn appendEntryAt(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    entry: Entry,
    base: usize,
) !void {
    try putU32(allocator, out, entry.layer);
    try out.append(allocator, @intFromEnum(entry.family));
    try out.append(allocator, @intFromEnum(entry.kind));
    try putU32(allocator, out, entry.rows);
    try putU32(allocator, out, entry.cols);
    try putU32(allocator, out, entry.group);
    try putU64(allocator, out, entry.bytes.len);
    while ((base + out.items.len) % payload_align != 0) try out.append(allocator, 0);
    try out.appendSlice(allocator, entry.bytes);
}

pub fn find(bytes: []const u8, layer: u32, kind: Kind) !?Entry {
    return findIn(bytes, .main, layer, kind);
}

pub fn findInBits(bytes: []const u8, family: Family, layer: u32, kind: Kind, bits: u32) !?Entry {
    return scan(bytes, family, layer, kind, bits);
}

pub fn findIn(bytes: []const u8, family: Family, layer: u32, kind: Kind) !?Entry {
    return scan(bytes, family, layer, kind, null);
}

fn scan(bytes: []const u8, family: Family, layer: u32, kind: Kind, bits: ?u32) !?Entry {
    var pos: usize = 0;
    const fmt = try checkHeader(bytes, &pos);
    const count = try takeU32(bytes, &pos);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const entry = try takeEntry(bytes, &pos, fmt);
        if (!sameEntry(entry, family, layer, kind)) continue;
        if (bits == null or entryBits(entry) == bits.?) return entry;
    }
    return null;
}

fn checkHeader(bytes: []const u8, pos: *usize) !u32 {
    const got = try take(bytes, pos, magic.len);
    if (!std.mem.eql(u8, got, magic)) return error.BadMagic;
    const fmt = try takeU32(bytes, pos);
    if (fmt < min_version or fmt > version) return error.BadVersion;
    return fmt;
}

fn takeEntry(bytes: []const u8, pos: *usize, fmt: u32) !Entry {
    const layer = try takeU32(bytes, pos);
    const family = if (fmt >= 3) try familyFrom(try takeByte(bytes, pos)) else .main;
    const kind = try kindFrom(try takeByte(bytes, pos));
    const rows = try takeU32(bytes, pos);
    const cols = try takeU32(bytes, pos);
    const group = try takeU32(bytes, pos);
    const len: usize = @intCast(try takeU64(bytes, pos));
    try alignPayload(bytes, pos);
    const offset = pos.*;
    return .{
        .family = family,
        .layer = layer,
        .kind = kind,
        .rows = rows,
        .cols = cols,
        .group = group,
        .offset = offset,
        .bytes = try take(bytes, pos, len),
    };
}

fn sameEntry(entry: Entry, family: Family, layer: u32, kind: Kind) bool {
    return entry.family == family and entry.layer == layer and entry.kind == kind;
}

fn familyFrom(value: u8) !Family {
    return switch (value) {
        1 => .main,
        2 => .noise,
        3 => .context,
        4 => .flux2,
        5 => .text,
        else => error.BadFamily,
    };
}

fn kindFrom(value: u8) !Kind {
    return switch (value) {
        1 => .ffn_down,
        2 => .ffn_gate,
        3 => .ffn_up,
        4 => .q,
        5 => .k,
        6 => .v,
        7 => .proj,
        8 => .ffn_gateup,
        9 => .flux2_weight,
        10 => .flux2_raw,
        11 => .flux2_norm,
        else => error.BadKind,
    };
}

fn putU32(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, value, .little);
    try out.appendSlice(allocator, &buf);
}

fn putU64(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: usize) !void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, @intCast(value), .little);
    try out.appendSlice(allocator, &buf);
}

fn takeU32(bytes: []const u8, pos: *usize) !u32 {
    return std.mem.readInt(u32, (try take(bytes, pos, 4))[0..4], .little);
}

fn takeU64(bytes: []const u8, pos: *usize) !u64 {
    return std.mem.readInt(u64, (try take(bytes, pos, 8))[0..8], .little);
}

fn takeByte(bytes: []const u8, pos: *usize) !u8 {
    return (try take(bytes, pos, 1))[0];
}

fn take(bytes: []const u8, pos: *usize, len: usize) ![]const u8 {
    // len comes from the file, so pos.* + len can wrap in a ReleaseFast build
    // and slip past the guard; subtract instead of adding.
    if (pos.* > bytes.len or len > bytes.len - pos.*) return error.Truncated;
    defer pos.* += len;
    return bytes[pos.*..][0..len];
}

fn alignPayload(bytes: []const u8, pos: *usize) !void {
    const next = alignForward(pos.*, payload_align);
    if (next > bytes.len) return error.Truncated;
    pos.* = next;
}

fn alignForward(value: usize, alignment: usize) usize {
    return (value + alignment - 1) / alignment * alignment;
}

fn fitUsize(value: u64) !usize {
    if (value > std.math.maxInt(usize)) return error.FileTooLarge;
    return @intCast(value);
}

test "take rejects a length that would wrap the cursor" {
    var pos: usize = 8;
    try std.testing.expectError(
        error.Truncated,
        take("0123456789", &pos, std.math.maxInt(usize) - 4),
    );
    try std.testing.expectEqual(@as(usize, 8), pos);
}

test "take still yields the bytes it is given room for" {
    var pos: usize = 2;
    try std.testing.expectEqualStrings("234", try take("0123456789", &pos, 3));
    try std.testing.expectEqual(@as(usize, 5), pos);
    try std.testing.expectError(error.Truncated, take("0123456789", &pos, 6));
}
