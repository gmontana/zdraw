//! Generation settings embedded in CLI generate/bench PNGs as a `zdraw`
//! tEXt chunk beside a `Software` marking. Session PNGs currently omit it.
//! The chunk contains a prompt hash, never the full prompt. Reproduction also
//! needs the original prompt, exact weights, inputs and engine configuration.
//! Metadata is deterministic: it contains no date, version or local path.
const std = @import("std");
const image = @import("image.zig");

pub const Recipe = struct {
    model: []const u8,
    profile: []const u8,
    prompt: []const u8,
    seed: u64,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    /// img2img: the SHA-256 of the init image file and the strength; a
    /// recipe with an init image re-derives only with that same file.
    init_sha256: ?[]const u8 = null,
    strength: ?f32 = null,
};

/// The JSON written into the PNG (field order fixed).
const Chunk = struct {
    schema: u32 = 1,
    generator: []const u8 = "zdraw",
    ai_generated: bool = true,
    model: []const u8,
    profile: []const u8,
    prompt_sha256: []const u8,
    seed: u64,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    init_sha256: ?[]const u8 = null,
    strength: ?f32 = null,
};

pub fn promptSha256(prompt: []const u8) [64]u8 {
    return image.pixelSha256(prompt);
}

/// The JSON text; caller frees.
pub fn json(allocator: std.mem.Allocator, r: Recipe) ![]u8 {
    const sha = promptSha256(r.prompt);
    return std.json.Stringify.valueAlloc(allocator, Chunk{
        .model = r.model,
        .profile = r.profile,
        .prompt_sha256 = &sha,
        .seed = r.seed,
        .width = r.width,
        .height = r.height,
        .steps = r.steps,
        .guidance = r.guidance,
        .init_sha256 = r.init_sha256,
        .strength = r.strength,
    }, .{ .emit_null_optional_fields = false });
}

/// The two tEXt chunks for a PNG; `buf` holds the JSON (caller frees it).
pub fn texts(allocator: std.mem.Allocator, r: Recipe) !struct { chunks: [2]image.Text, buf: []u8 } {
    const buf = try json(allocator, r);
    return .{
        .chunks = .{
            .{ .keyword = "Software", .text = "zdraw (AI-generated image)" },
            .{ .keyword = "zdraw", .text = buf },
        },
        .buf = buf,
    };
}

test "the recipe json is deterministic and carries the prompt hash only" {
    const r = Recipe{
        .model = "flux2-klein-4b",
        .profile = "-",
        .prompt = "a red fox sitting in deep snow, golden hour light",
        .seed = 46,
        .width = 1024,
        .height = 1024,
        .steps = 4,
        .guidance = 1.0,
    };
    const a = try json(std.testing.allocator, r);
    defer std.testing.allocator.free(a);
    const b = try json(std.testing.allocator, r);
    defer std.testing.allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
    try std.testing.expect(std.mem.indexOf(u8, a, "red fox") == null);
    try std.testing.expect(std.mem.indexOf(u8, a, "909c1660128ee00f") != null);
    try std.testing.expect(std.mem.indexOf(u8, a, "\"seed\":46") != null);
}
