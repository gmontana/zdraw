//! Shared semantic configuration for CPU-oracle and Metal ToMA execution.

const std = @import("std");

const execution_plan = @import("execution_plan.zig");

pub const Mode = enum {
    paper_spec,
    official_b578009,
};

pub const Config = struct {
    mode: Mode,
    destination_tokens: usize,
    region_count: usize = 1,
    selection_layout: execution_plan.RegionLayout,
    assignment_scope: execution_plan.AssignmentScope,
    unmerge: execution_plan.Unmerge,
    assignment_scale: f32,
};

pub const Shape = struct {
    source_tokens: usize,
    feature_width: usize,
    region_size: usize,
    destinations_per_region: usize,
    token_side: usize,
    region_side: usize,
};

pub fn shape(source_tokens: usize, feature_width: usize, config: Config) !Shape {
    if (source_tokens == 0 or feature_width == 0) return error.InvalidInputShape;
    if (config.destination_tokens == 0 or config.destination_tokens > source_tokens) {
        return error.InvalidTokenCount;
    }
    if (config.region_count == 0 or source_tokens % config.region_count != 0 or
        config.destination_tokens % config.region_count != 0)
    {
        return error.InvalidRegionCount;
    }
    if (!std.math.isFinite(config.assignment_scale) or config.assignment_scale <= 0) {
        return error.InvalidAssignmentScale;
    }
    if (config.selection_layout == .global and config.region_count != 1) {
        return error.InvalidRegionCount;
    }
    var token_side: usize = 0;
    var region_side: usize = 0;
    if (config.selection_layout == .tile) {
        token_side = squareRoot(source_tokens) orelse return error.InvalidTileGrid;
        region_side = squareRoot(config.region_count) orelse return error.InvalidTileGrid;
        if (token_side % region_side != 0) return error.InvalidTileGrid;
    }
    if (config.mode == .official_b578009 and
        (config.selection_layout != .tile or config.assignment_scope != .region_local or
            config.unmerge != .official_raw_transpose))
    {
        return error.InvalidOfficialMode;
    }
    return .{
        .source_tokens = source_tokens,
        .feature_width = feature_width,
        .region_size = source_tokens / config.region_count,
        .destinations_per_region = config.destination_tokens / config.region_count,
        .token_side = token_side,
        .region_side = region_side,
    };
}

fn squareRoot(value: usize) ?usize {
    const root: usize = @intFromFloat(@sqrt(@as(f64, @floatFromInt(value))));
    return if (root * root == value) root else null;
}

test "paper and official modes have distinct assignment semantics" {
    _ = try shape(4096, 3072, .{
        .mode = .paper_spec,
        .destination_tokens = 2048,
        .region_count = 64,
        .selection_layout = .tile,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 1000,
    });
    try std.testing.expectError(error.InvalidOfficialMode, shape(4096, 3072, .{
        .mode = .official_b578009,
        .destination_tokens = 2048,
        .region_count = 64,
        .selection_layout = .tile,
        .assignment_scope = .global,
        .unmerge = .official_raw_transpose,
        .assignment_scale = 1000,
    }));
}
