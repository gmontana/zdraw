//! Qwen tokenizer metadata and prompt formatting for Z-Image.

const std = @import("std");

const bpe = @import("bpe.zig");
const bpe_load = @import("bpe_load.zig");

pub const Config = struct {
    max_len: u32,
    eos_id: u32,
    im_start_id: u32,
    im_end_id: u32,
    pad_id: u32,
    think_id: ?u32 = null,
    think_end_id: ?u32 = null,
};

pub const Error = error{
    InvalidTokenizer,
    TokenizerTooLarge,
};

pub const Loaded = struct {
    config: Config,
    encoder: bpe.Encoder,

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator) void {
        self.encoder.deinit(allocator);
        self.* = undefined;
    }

    pub fn encodePrompt(
        self: Loaded,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        max_len: usize,
    ) !Tokens {
        const text = try formatChat(allocator, prompt);
        defer allocator.free(text);

        const ids = try self.encoder.encode(allocator, text, self.specials());
        defer allocator.free(ids);
        return padTokens(allocator, ids, max_len, self.config.pad_id);
    }

    /// FLUX.2 Klein prompt encoding: the Qwen3 chat template with the empty
    /// think block (enable_thinking=false), right-padded to max_len.
    pub fn encodeKlein(
        self: Loaded,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        max_len: usize,
    ) !Tokens {
        const text = try std.fmt.allocPrint(
            allocator,
            "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            .{prompt},
        );
        defer allocator.free(text);
        const ids = try self.encoder.encode(allocator, text, self.specials());
        defer allocator.free(ids);
        return padTokens(allocator, ids, max_len, self.config.pad_id);
    }

    fn specials(self: Loaded) bpe.Specials {
        return .{
            .im_start_id = self.config.im_start_id,
            .im_end_id = self.config.im_end_id,
            .think_id = self.config.think_id,
            .think_end_id = self.config.think_end_id,
            .eos_id = self.config.eos_id,
            .pad_id = self.config.pad_id,
        };
    }
};

pub const Tokens = struct {
    ids: []u32,
    mask: []u8,

    pub fn deinit(self: *Tokens, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.mask);
        self.* = undefined;
    }
};

pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !Loaded {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer/tokenizer_config.json", .{root});
    defer allocator.free(path);
    const bytes = try readFile(io, allocator, path);
    defer allocator.free(bytes);
    const config = try parseConfig(allocator, bytes);

    const vocab_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer/vocab.json", .{root});
    defer allocator.free(vocab_path);
    const merges_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer/merges.txt", .{root});
    defer allocator.free(merges_path);

    // Qwen2TokenizerFast checkpoints (FLUX.2 Klein) ship merges inside
    // tokenizer.json with no merges.txt; extract once into a sidecar so the
    // existing BPE loader applies unchanged.
    ensureMerges(io, allocator, root, merges_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };

    return .{
        .config = config,
        .encoder = try bpe_load.load(io, allocator, vocab_path, merges_path),
    };
}

fn ensureMerges(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    merges_path: []const u8,
) !void {
    if (std.Io.Dir.cwd().openFile(io, merges_path, .{})) |f| {
        var fh = f;
        fh.close(io);
        return; // already present
    } else |_| {}
    const tj_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer/tokenizer.json", .{root});
    defer allocator.free(tj_path);
    const bytes = try readFile(io, allocator, tj_path);
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidTokenizer;
    const model = parsed.value.object.get("model") orelse return error.InvalidTokenizer;
    if (model != .object) return error.InvalidTokenizer;
    const merges = model.object.get("merges") orelse return error.InvalidTokenizer;
    if (merges != .array) return error.InvalidTokenizer;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    for (merges.array.items) |m| switch (m) {
        .string => |line| {
            try out.appendSlice(allocator, line);
            try out.append(allocator, '\n');
        },
        .array => |pair| {
            if (pair.items.len != 2) return error.InvalidTokenizer;
            if (pair.items[0] != .string or pair.items[1] != .string) return error.InvalidTokenizer;
            try out.appendSlice(allocator, pair.items[0].string);
            try out.append(allocator, ' ');
            try out.appendSlice(allocator, pair.items[1].string);
            try out.append(allocator, '\n');
        },
        else => return error.InvalidTokenizer,
    };
    const file = try std.Io.Dir.cwd().createFile(io, merges_path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    try writer.interface.writeAll(out.items);
    try writer.interface.flush();
}

pub fn formatChat(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n",
        .{prompt},
    );
}

pub fn parseConfig(allocator: std.mem.Allocator, bytes: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
    });
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidTokenizer;
    const object = parsed.value.object;
    const token_map = object.get("added_tokens_decoder") orelse {
        return error.InvalidTokenizer;
    };
    if (token_map != .object) return error.InvalidTokenizer;

    return .{
        .max_len = try getU32(object, "model_max_length"),
        .eos_id = try findToken(token_map.object, "<|im_end|>"),
        .im_start_id = try findToken(token_map.object, "<|im_start|>"),
        .im_end_id = try findToken(token_map.object, "<|im_end|>"),
        .pad_id = try findToken(token_map.object, "<|endoftext|>"),
        .think_id = findToken(token_map.object, "<think>") catch null,
        .think_end_id = findToken(token_map.object, "</think>") catch null,
    };
}

fn findToken(map: std.json.ObjectMap, content: []const u8) !u32 {
    var iter = map.iterator();
    while (iter.next()) |entry| {
        if (entry.value_ptr.* != .object) continue;
        const value = entry.value_ptr.object.get("content") orelse continue;
        if (value != .string) continue;
        if (!std.mem.eql(u8, value.string, content)) continue;
        return std.fmt.parseInt(u32, entry.key_ptr.*, 10) catch error.InvalidTokenizer;
    }
    return error.InvalidTokenizer;
}

fn getU32(object: std.json.ObjectMap, key: []const u8) !u32 {
    const value = object.get(key) orelse return error.InvalidTokenizer;
    if (value != .integer or value.integer < 0) return error.InvalidTokenizer;
    return @intCast(value.integer);
}

fn padTokens(
    allocator: std.mem.Allocator,
    raw: []const u32,
    max_len: usize,
    pad_id: u32,
) !Tokens {
    const ids = try allocator.alloc(u32, max_len);
    errdefer allocator.free(ids);
    const mask = try allocator.alloc(u8, max_len);
    errdefer allocator.free(mask);

    const used = @min(raw.len, max_len);
    @memcpy(ids[0..used], raw[0..used]);
    @memset(mask[0..used], 1);
    @memset(ids[used..], pad_id);
    @memset(mask[used..], 0);
    return .{ .ids = ids, .mask = mask };
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > 1024 * 1024) return error.TokenizerTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

test "format chat prompt like z-image pipeline" {
    const text = try formatChat(std.testing.allocator, "a red boat");
    defer std.testing.allocator.free(text);

    try std.testing.expectEqualStrings(
        "<|im_start|>user\na red boat<|im_end|>\n<|im_start|>assistant\n",
        text,
    );
}

test "parse tokenizer special ids" {
    const json =
        \\{"model_max_length":512,"added_tokens_decoder":{
        \\"151643":{"content":"<|endoftext|>"},
        \\"151644":{"content":"<|im_start|>"},
        \\"151645":{"content":"<|im_end|>"}}}
    ;
    const cfg = try parseConfig(std.testing.allocator, json);
    const max_len: u32 = 512;
    const start_id: u32 = 151644;
    const end_id: u32 = 151645;

    try std.testing.expectEqual(max_len, cfg.max_len);
    try std.testing.expectEqual(start_id, cfg.im_start_id);
    try std.testing.expectEqual(end_id, cfg.eos_id);
}

test "pad prompt ids and mask" {
    const tokens = try padTokens(std.testing.allocator, &.{ 1, 2, 3 }, 5, 0);
    var owned = tokens;
    defer owned.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 0, 0 }, owned.ids);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 1, 0, 0 }, owned.mask);
}

test "klein prompt tokenizes think markers as single special tokens" {
    // The reference tokenizer emits one id per reasoning marker; splitting
    // them into ordinary sub-words changed every Klein prompt's conditioning
    // from token 12 onward until 2026-08-06.
    const alloc = std.testing.allocator;
    var specials = bpe.Specials{
        .im_start_id = 151644,
        .im_end_id = 151645,
        .eos_id = 151645,
        .pad_id = 151643,
        .think_id = 151667,
        .think_end_id = 151668,
    };
    try std.testing.expectEqual(@as(?u32, 151668), matchId(specials, "</think>\n\n"));
    try std.testing.expectEqual(@as(?u32, 151667), matchId(specials, "<think>\n\n"));
    // Without the ids declared, the markers fall through to ordinary BPE.
    specials.think_id = null;
    specials.think_end_id = null;
    try std.testing.expectEqual(@as(?u32, null), matchId(specials, "<think>\n\n"));
    _ = alloc;
}

fn matchId(specials: bpe.Specials, text: []const u8) ?u32 {
    return specials.matchForTest(text);
}
