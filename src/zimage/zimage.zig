//! Z-Image-Turbo model metadata loader.

const std = @import("std");

const qwen = @import("../text/qwen.zig");
const weights = @import("../pack/weights.zig");
const tokenizer = @import("../text/tokenizer.zig");
const zconfig = @import("zimage_config.zig");
const ztransformer = @import("ztransformer.zig");

pub const Loaded = struct {
    config: zconfig.Config,
    tokens: tokenizer.Loaded,
    indexes: weights.ZImage,

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator) void {
        self.tokens.deinit(allocator);
        self.indexes.deinit(allocator);
        self.* = undefined;
    }
};

pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !Loaded {
    const config = try zconfig.load(io, allocator, root);
    var indexes = try weights.loadZImage(io, allocator, root);
    errdefer indexes.deinit(allocator);
    try qwen.validate(allocator, config.text, indexes.text);
    try ztransformer.validate(allocator, config.transformer, indexes.transformer);

    return .{
        .config = config,
        .tokens = try tokenizer.load(io, allocator, root),
        .indexes = indexes,
    };
}
