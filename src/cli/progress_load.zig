//! Compact progress metadata for model startup stages.

const std = @import("std");

pub const total: usize = 7;

pub fn isStage(name: []const u8) bool {
    return label(name).len > 0;
}

pub fn isStart(name: []const u8) bool {
    return std.mem.eql(u8, name, "checking model files");
}

pub fn label(name: []const u8) []const u8 {
    inline for (stages) |stage| {
        if (std.mem.eql(u8, name, stage.name)) return stage.label;
    }
    return "";
}

const stages = [_]struct {
    name: []const u8,
    label: []const u8,
}{
    .{ .name = "checking model files", .label = "files" },
    .{ .name = "loading model metadata", .label = "metadata" },
    .{ .name = "loading transformer", .label = "transformer" },
    .{ .name = "preparing rope", .label = "rope" },
    .{ .name = "loading text encoder", .label = "text encoder" },
    .{ .name = "loading zpack sidecar", .label = "zpack" },
    .{ .name = "loading VAE decoder", .label = "vae" },
};
