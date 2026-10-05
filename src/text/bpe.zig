//! Byte-level BPE encoder used by the Qwen tokenizer.

const std = @import("std");

const byte_map = @import("bpe_bytes.zig");
const split = @import("bpe_split.zig");

pub const Error = error{
    UnknownToken,
};

pub const Specials = struct {
    im_start_id: u32,
    im_end_id: u32,
    eos_id: u32,
    pad_id: u32,
    /// Qwen3 reasoning markers. Optional because only some checkpoints
    /// declare them; when absent the text falls through to ordinary BPE,
    /// which is what zdraw did for every FLUX.2 Klein prompt before
    /// 2026-08-06 and is NOT what the reference tokenizer does.
    think_id: ?u32 = null,
    think_end_id: ?u32 = null,

    /// Test-visible wrapper: the marker set is a correctness contract with
    /// the reference tokenizer, so it is pinned by a test.
    pub fn matchForTest(self: Specials, text: []const u8) ?u32 {
        const m = self.match(text) orelse return null;
        return m.id;
    }

    fn match(self: Specials, text: []const u8) ?Match {
        if (std.mem.startsWith(u8, text, "<|im_start|>")) {
            return .{ .id = self.im_start_id, .len = 12 };
        }
        if (std.mem.startsWith(u8, text, "<|im_end|>")) {
            return .{ .id = self.im_end_id, .len = 10 };
        }
        if (std.mem.startsWith(u8, text, "<|endoftext|>")) {
            return .{ .id = self.pad_id, .len = 13 };
        }
        // Longer marker first: "</think>" also starts with '<'.
        if (self.think_end_id) |id| {
            if (std.mem.startsWith(u8, text, "</think>")) return .{ .id = id, .len = 8 };
        }
        if (self.think_id) |id| {
            if (std.mem.startsWith(u8, text, "<think>")) return .{ .id = id, .len = 7 };
        }
        return null;
    }
};

const Match = struct {
    id: u32,
    len: usize,
};

pub const Encoder = struct {
    vocab: std.StringHashMap(u32),
    ranks: std.StringHashMap(u32),

    pub fn init(allocator: std.mem.Allocator) Encoder {
        return .{
            .vocab = std.StringHashMap(u32).init(allocator),
            .ranks = std.StringHashMap(u32).init(allocator),
        };
    }

    pub fn deinit(self: *Encoder, allocator: std.mem.Allocator) void {
        freeKeys(&self.vocab, allocator);
        freeKeys(&self.ranks, allocator);
        self.vocab.deinit();
        self.ranks.deinit();
        self.* = undefined;
    }

    pub fn encode(
        self: Encoder,
        allocator: std.mem.Allocator,
        text: []const u8,
        specials: Specials,
    ) ![]u32 {
        var ids = try std.ArrayList(u32).initCapacity(allocator, text.len / 2 + 4);
        errdefer ids.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            if (specials.match(text[i..])) |m| {
                try ids.append(allocator, m.id);
                i += m.len;
                continue;
            }
            const piece = split.next(text[i..]);
            try self.encodePiece(allocator, &ids, piece);
            i += piece.len;
        }
        return ids.toOwnedSlice(allocator);
    }

    pub fn addVocab(
        self: *Encoder,
        allocator: std.mem.Allocator,
        token: []const u8,
        id: u32,
    ) !void {
        const key = try allocator.dupe(u8, token);
        self.vocab.put(key, id) catch |err| {
            allocator.free(key);
            return err;
        };
    }

    pub fn addMerge(
        self: *Encoder,
        allocator: std.mem.Allocator,
        left: []const u8,
        right: []const u8,
        rank: u32,
    ) !void {
        const key = try makePair(allocator, left, right);
        self.ranks.put(key, rank) catch |err| {
            allocator.free(key);
            return err;
        };
    }

    fn encodePiece(
        self: Encoder,
        allocator: std.mem.Allocator,
        ids: *std.ArrayList(u32),
        piece: []const u8,
    ) !void {
        var symbols = try byteSymbols(allocator, piece);
        defer freeSymbols(allocator, &symbols);

        while (symbols.items.len > 1) {
            const pos = try self.bestPair(allocator, symbols.items) orelse break;
            const merged = try join(allocator, symbols.items[pos], symbols.items[pos + 1]);
            allocator.free(symbols.items[pos]);
            allocator.free(symbols.items[pos + 1]);
            symbols.items[pos] = merged;
            std.mem.copyForwards(
                []u8,
                symbols.items[pos + 1 ..],
                symbols.items[pos + 2 ..],
            );
            symbols.items.len -= 1;
        }

        for (symbols.items) |token| {
            const id = self.vocab.get(token) orelse return error.UnknownToken;
            try ids.append(allocator, id);
        }
    }

    fn bestPair(
        self: Encoder,
        allocator: std.mem.Allocator,
        symbols: []const []u8,
    ) !?usize {
        var best_rank: ?u32 = null;
        var best_pos: usize = 0;
        var i: usize = 0;
        while (i + 1 < symbols.len) : (i += 1) {
            const key = try makePair(allocator, symbols[i], symbols[i + 1]);
            defer allocator.free(key);
            const rank = self.ranks.get(key) orelse continue;
            if (best_rank == null or rank < best_rank.?) {
                best_rank = rank;
                best_pos = i;
            }
        }
        return if (best_rank == null) null else best_pos;
    }
};

fn byteSymbols(allocator: std.mem.Allocator, piece: []const u8) !std.ArrayList([]u8) {
    var out = try std.ArrayList([]u8).initCapacity(allocator, piece.len);
    errdefer freeSymbols(allocator, &out);
    for (piece) |byte| try out.append(allocator, try byte_map.symbol(allocator, byte));
    return out;
}

fn makePair(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, left.len + 1 + right.len);
    @memcpy(out[0..left.len], left);
    out[left.len] = 0;
    @memcpy(out[left.len + 1 ..], right);
    return out;
}

fn join(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, left.len + right.len);
    @memcpy(out[0..left.len], left);
    @memcpy(out[left.len..], right);
    return out;
}

fn freeSymbols(allocator: std.mem.Allocator, symbols: *std.ArrayList([]u8)) void {
    for (symbols.items) |token| allocator.free(token);
    symbols.deinit(allocator);
}

fn freeKeys(map: *std.StringHashMap(u32), allocator: std.mem.Allocator) void {
    var keys = map.keyIterator();
    while (keys.next()) |key| allocator.free(key.*);
}

test "merge ranks produce vocabulary ids" {
    var enc = Encoder.init(std.testing.allocator);
    defer enc.deinit(std.testing.allocator);
    try enc.addVocab(std.testing.allocator, "a", 1);
    try enc.addVocab(std.testing.allocator, "b", 2);
    try enc.addVocab(std.testing.allocator, "ab", 3);
    try enc.addVocab(std.testing.allocator, "Ġa", 4);
    try enc.addMerge(std.testing.allocator, "a", "b", 0);
    try enc.addMerge(std.testing.allocator, "Ġ", "a", 1);

    const ids = try enc.encode(std.testing.allocator, "ab a", .{
        .im_start_id = 10,
        .im_end_id = 11,
        .eos_id = 11,
        .pad_id = 12,
    });
    defer std.testing.allocator.free(ids);

    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, ids);
}
