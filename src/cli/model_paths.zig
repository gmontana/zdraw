//! Shared checkpoint-directory policy for fetch and CLI model commands.
const std = @import("std");
const kinds = @import("model_kind.zig");

/// Return an owned path. An explicit directory always wins, including when HOME is unset.
pub fn resolve(
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    kind: kinds.ModelKind,
    explicit: []const u8,
) ![]u8 {
    if (explicit.len > 0) return allocator.dupe(u8, explicit);
    const name = switch (kind) {
        .flux2_klein_4b => "FLUX.2-klein-4B",
        .flux2_klein_base_4b => "FLUX.2-klein-base-4B",
        .z_image_turbo => "Z-Image-Turbo",
        else => return error.MissingWeights,
    };
    const home = environ.get("HOME") orelse return error.NoHome;
    return std.fs.path.join(allocator, &.{ home, ".zdraw", "models", name });
}

test "model commands share the download directories" {
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();
    try env.put("HOME", "/Users/example");
    const klein = try resolve(std.testing.allocator, &env, .flux2_klein_4b, "");
    defer std.testing.allocator.free(klein);
    try std.testing.expectEqualStrings("/Users/example/.zdraw/models/FLUX.2-klein-4B", klein);
    const zimage = try resolve(std.testing.allocator, &env, .z_image_turbo, "");
    defer std.testing.allocator.free(zimage);
    try std.testing.expectEqualStrings("/Users/example/.zdraw/models/Z-Image-Turbo", zimage);
    const base = try resolve(std.testing.allocator, &env, .flux2_klein_base_4b, "");
    defer std.testing.allocator.free(base);
    try std.testing.expectEqualStrings("/Users/example/.zdraw/models/FLUX.2-klein-base-4B", base);
}

test "explicit paths work without HOME and unsupported models need an explicit path" {
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();
    const path = try resolve(std.testing.allocator, &env, .flux2_klein_4b, "models/my model");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("models/my model", path);
    try std.testing.expectError(
        error.NoHome,
        resolve(std.testing.allocator, &env, .z_image_turbo, ""),
    );
    try std.testing.expectError(
        error.MissingWeights,
        resolve(std.testing.allocator, &env, .flux2_klein_9b, ""),
    );
}
