//! Prompt-to-embedding path for Z-Image's Qwen text encoder.

const std = @import("std");

const mattn = @import("../metal/mattn.zig");
const mlinear = @import("../metal/mlinear.zig");
const progress = @import("../cli/progress.zig");
const qenc = @import("../text/qwen_encoder.zig");
const qscratch = @import("../text/qwen_scratch.zig");
const shards = @import("../pack/shards.zig");
const tokenizer = @import("../text/tokenizer.zig");
const weight_index = @import("../pack/weight_index.zig");
const zconfig = @import("zimage_config.zig");

const max_prompt_tokens = 512;

pub const Encoded = struct {
    embeds: []f32,
    mask: []u8,
    tokens: usize,
    hidden: usize,

    pub fn deinit(self: *Encoded, allocator: std.mem.Allocator) void {
        allocator.free(self.embeds);
        allocator.free(self.mask);
        self.* = undefined;
    }
};

pub const Prepared = struct {
    metal: ?*mlinear.Context,
    attn: ?*mattn.Context,
    store: *const shards.Store,
};

pub fn encode(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
    text: zconfig.Text,
    tokens: tokenizer.Loaded,
    index: weight_index.Index,
    prompt: []const u8,
) !Encoded {
    var metal = try initMetal(io, allocator);
    defer if (metal) |*ctx| ctx.deinit();
    var attn = mattn.Context.init() catch null;
    defer if (attn) |*ctx| ctx.deinit();

    const text_root = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{root});
    defer allocator.free(text_root);
    var store = try shards.open(io, allocator, text_root, index);
    defer store.deinit(io, allocator);

    return try encodeWith(io, allocator, .{
        .metal = if (metal) |*ctx| ctx else null,
        .attn = if (attn) |*ctx| ctx else null,
        .store = &store,
    }, text, tokens, index, prompt);
}

pub fn encodePrepared(
    io: std.Io,
    allocator: std.mem.Allocator,
    prepared: Prepared,
    text: zconfig.Text,
    tokens: tokenizer.Loaded,
    index: weight_index.Index,
    prompt: []const u8,
) !Encoded {
    return try encodeWith(io, allocator, prepared, text, tokens, index, prompt);
}

fn encodeWith(
    io: std.Io,
    allocator: std.mem.Allocator,
    prepared: Prepared,
    text: zconfig.Text,
    tokens: tokenizer.Loaded,
    index: weight_index.Index,
    prompt: []const u8,
) !Encoded {
    var ids = try tokens.encodePrompt(allocator, prompt, max_prompt_tokens);
    defer ids.deinit(allocator);

    const hidden: usize = @intCast(text.hidden_size);
    const used = countMask(ids.mask);
    try progress.tokens(io, allocator, used);
    const embeds = try allocator.alloc(f32, used * hidden);
    errdefer allocator.free(embeds);

    const cfg = qenc.attnConfig(text, used);
    var scratch = try qscratch.init(
        allocator,
        cfg,
        @intCast(text.intermediate_size),
    );
    defer scratch.deinit(allocator);

    try qenc.run(
        io,
        allocator,
        prepared.metal,
        prepared.attn,
        embeds,
        ids.ids[0..used],
        prepared.store,
        index,
        text,
        .penultimate,
        &scratch,
        false, // Z-Image path stays on the rows kernel, bit-identical.
        null, // projections follow use_gemm
        0, // ids are already trimmed to `used`; no padding to mask.
    );
    return owned(allocator, embeds, used, hidden);
}

fn initMetal(io: std.Io, allocator: std.mem.Allocator) !?mlinear.Context {
    const ctx = mlinear.Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => {
            try progress.event(io, allocator, "Metal unavailable; CPU linear fallback");
            return null;
        },
        else => return err,
    };
    try progress.event(io, allocator, "Metal linear ready");
    return ctx;
}

fn owned(
    allocator: std.mem.Allocator,
    embeds: []f32,
    used: usize,
    hidden: usize,
) !Encoded {
    const mask = try allocator.alloc(u8, used);
    errdefer allocator.free(mask);
    @memset(mask, 1);
    return .{ .embeds = embeds, .mask = mask, .tokens = used, .hidden = hidden };
}

fn compact(
    allocator: std.mem.Allocator,
    embeds: []f32,
    mask: []const u8,
    hidden: usize,
) !Encoded {
    defer allocator.free(embeds);
    const used = countMask(mask);
    const out = try allocator.alloc(f32, used * hidden);
    errdefer allocator.free(out);
    const out_mask = try allocator.alloc(u8, used);
    errdefer allocator.free(out_mask);
    copyUsed(out, embeds, mask, hidden);
    @memset(out_mask, 1);
    return .{ .embeds = out, .mask = out_mask, .tokens = used, .hidden = hidden };
}

fn countMask(mask: []const u8) usize {
    var used: usize = 0;
    for (mask) |value| {
        if (value != 0) used += 1;
    }
    return used;
}

fn copyUsed(out: []f32, embeds: []const f32, mask: []const u8, hidden: usize) void {
    var dst: usize = 0;
    for (mask, 0..) |value, tok| {
        if (value == 0) continue;
        const row = embeds[tok * hidden ..][0..hidden];
        @memcpy(out[dst * hidden ..][0..hidden], row);
        dst += 1;
    }
}

test "encoded shape owns embeddings and mask" {
    var out = Encoded{
        .embeds = try std.testing.allocator.alloc(f32, 4),
        .mask = try std.testing.allocator.alloc(u8, 2),
        .tokens = 2,
        .hidden = 2,
    };
    out.deinit(std.testing.allocator);
}

test "compact removes padding embeddings" {
    const embeds = try std.testing.allocator.dupe(f32, &.{ 1, 2, 3, 4, 5, 6 });
    var out = try compact(std.testing.allocator, embeds, &.{ 1, 0, 1 }, 2);
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), out.tokens);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6 }, out.embeds);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, out.mask);
}
