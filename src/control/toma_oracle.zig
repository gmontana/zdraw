//! Independent CPU oracle for the ToMA equations.
//!
//! This is deliberately not the production path. It provides deterministic
//! destinations, assignment matrices, merge, and unmerge outputs against which
//! Metal kernels can be tested stage by stage.

const std = @import("std");

const execution_plan = @import("execution_plan.zig");
const toma_config = @import("toma_config.zig");

pub const Mode = toma_config.Mode;
pub const Config = toma_config.Config;

pub const Pattern = struct {
    source_tokens: usize,
    destination_tokens: usize,
    region_count: usize,
    destinations: []u32,
    // Both matrices are row-major [destination, source].
    assignment: []f32,
    merge_matrix: []f32,
    unmerge: execution_plan.Unmerge,

    pub fn deinit(self: *Pattern, allocator: std.mem.Allocator) void {
        allocator.free(self.destinations);
        allocator.free(self.assignment);
        allocator.free(self.merge_matrix);
        self.* = undefined;
    }
};

pub fn build(
    allocator: std.mem.Allocator,
    features: []const f32,
    source_tokens: usize,
    hidden: usize,
    config: Config,
) !Pattern {
    try validate(features, source_tokens, hidden, config);
    const normalized = try normalize(allocator, features, source_tokens, hidden, config.mode);
    defer allocator.free(normalized);
    var selection_copy: ?[]f32 = null;
    defer if (selection_copy) |values| allocator.free(values);
    const selection_input: []const f32 = if (config.mode == .official_b578009) select: {
        // The released path normalizes once in merge_helper and a second time
        // inside batched_facility_location.
        selection_copy = try normalize(
            allocator,
            normalized,
            source_tokens,
            hidden,
            .official_b578009,
        );
        break :select selection_copy.?;
    } else normalized;
    const destinations = try allocator.alloc(u32, config.destination_tokens);
    errdefer allocator.free(destinations);
    try selectAll(allocator, selection_input, source_tokens, hidden, config, destinations);
    const assignment = try allocator.alloc(f32, config.destination_tokens * source_tokens);
    errdefer allocator.free(assignment);
    @memset(assignment, 0);
    try assign(normalized, source_tokens, hidden, config, destinations, assignment);
    const merge_matrix = try allocator.alloc(f32, assignment.len);
    errdefer allocator.free(merge_matrix);
    try normalizeRows(assignment, source_tokens, merge_matrix);
    return .{
        .source_tokens = source_tokens,
        .destination_tokens = config.destination_tokens,
        .region_count = config.region_count,
        .destinations = destinations,
        .assignment = assignment,
        .merge_matrix = merge_matrix,
        .unmerge = config.unmerge,
    };
}

pub fn applyMerge(
    allocator: std.mem.Allocator,
    pattern: Pattern,
    input: []const f32,
    hidden: usize,
) ![]f32 {
    if (hidden == 0 or input.len != pattern.source_tokens * hidden) {
        return error.InvalidInputShape;
    }
    const output = try allocator.alloc(f32, pattern.destination_tokens * hidden);
    for (0..pattern.destination_tokens) |destination| {
        for (0..hidden) |channel| {
            var sum: f32 = 0;
            for (0..pattern.source_tokens) |source| {
                sum += pattern.merge_matrix[destination * pattern.source_tokens + source] *
                    input[source * hidden + channel];
            }
            output[destination * hidden + channel] = sum;
        }
    }
    return output;
}

pub fn applyUnmerge(
    allocator: std.mem.Allocator,
    pattern: Pattern,
    input: []const f32,
    hidden: usize,
) ![]f32 {
    if (hidden == 0 or input.len != pattern.destination_tokens * hidden) {
        return error.InvalidInputShape;
    }
    return switch (pattern.unmerge) {
        .paper_normalized_transpose => applyTranspose(
            allocator,
            pattern.merge_matrix,
            pattern.source_tokens,
            pattern.destination_tokens,
            input,
            hidden,
        ),
        .official_raw_transpose => applyTranspose(
            allocator,
            pattern.assignment,
            pattern.source_tokens,
            pattern.destination_tokens,
            input,
            hidden,
        ),
        .pseudo_inverse => applyPseudo(allocator, pattern, input, hidden),
    };
}

pub fn gatherRows(
    allocator: std.mem.Allocator,
    input: []const f32,
    columns: usize,
    indices: []const u32,
) ![]f32 {
    if (columns == 0 or input.len % columns != 0) return error.InvalidInputShape;
    const rows = input.len / columns;
    const output = try allocator.alloc(f32, indices.len * columns);
    errdefer allocator.free(output);
    for (indices, 0..) |raw, output_row| {
        const input_row: usize = raw;
        if (input_row >= rows) return error.InvalidIndex;
        @memcpy(
            output[output_row * columns ..][0..columns],
            input[input_row * columns ..][0..columns],
        );
    }
    return output;
}

fn validate(
    features: []const f32,
    source_tokens: usize,
    hidden: usize,
    config: Config,
) !void {
    if (features.len != source_tokens * hidden) {
        return error.InvalidInputShape;
    }
    _ = try toma_config.shape(source_tokens, hidden, config);
}

fn normalize(
    allocator: std.mem.Allocator,
    input: []const f32,
    tokens: usize,
    hidden: usize,
    mode: Mode,
) ![]f32 {
    const output = try allocator.alloc(f32, input.len);
    errdefer allocator.free(output);
    for (0..tokens) |token| {
        const row = input[token * hidden ..][0..hidden];
        var norm_sq: f32 = 0;
        for (row) |value| norm_sq += value * value;
        const norm = @sqrt(norm_sq);
        for (row, 0..) |value, channel| {
            output[token * hidden + channel] = switch (mode) {
                .paper_spec => if (norm > 1e-12) value / norm else 0,
                // The pinned code performs raw division before its later
                // F.normalize call, so zero rows intentionally remain NaN.
                .official_b578009 => value / norm,
            };
        }
    }
    return output;
}

fn selectAll(
    allocator: std.mem.Allocator,
    normalized: []const f32,
    source_tokens: usize,
    hidden: usize,
    config: Config,
    destinations: []u32,
) !void {
    const region_size = source_tokens / config.region_count;
    const per_region = config.destination_tokens / config.region_count;
    for (0..config.region_count) |region| {
        const output = destinations[region * per_region ..][0..per_region];
        try selectRegion(
            allocator,
            normalized,
            source_tokens,
            hidden,
            config,
            region,
            region_size,
            output,
        );
    }
}

fn selectRegion(
    allocator: std.mem.Allocator,
    normalized: []const f32,
    source_tokens: usize,
    hidden: usize,
    config: Config,
    region: usize,
    region_size: usize,
    output: []u32,
) !void {
    const similarity = try allocator.alloc(f32, region_size * region_size);
    defer allocator.free(similarity);
    fillSimilarity(similarity, normalized, source_tokens, hidden, config, region);
    const selected = try allocator.alloc(bool, region_size);
    defer allocator.free(selected);
    @memset(selected, false);
    const max_sim = try allocator.alloc(f32, region_size);
    defer allocator.free(max_sim);
    const first = firstCenter(similarity, region_size);
    output[0] = @intCast(regionToken(source_tokens, config, region, first));
    selected[first] = true;
    @memcpy(max_sim, similarity[first * region_size ..][0..region_size]);
    for (1..output.len) |slot| {
        const next = nextCenter(similarity, max_sim, selected, region_size, config.mode);
        output[slot] = @intCast(regionToken(source_tokens, config, region, next));
        selected[next] = true;
        for (max_sim, similarity[next * region_size ..][0..region_size]) |*old, value| {
            old.* = @max(old.*, value);
        }
    }
}

fn fillSimilarity(
    similarity: []f32,
    normalized: []const f32,
    source_tokens: usize,
    hidden: usize,
    config: Config,
    region: usize,
) void {
    const region_size = source_tokens / config.region_count;
    for (0..region_size) |row| {
        const row_token = regionToken(source_tokens, config, region, row);
        for (0..region_size) |column| {
            const column_token = regionToken(source_tokens, config, region, column);
            similarity[row * region_size + column] = dotRows(
                normalized,
                hidden,
                row_token,
                column_token,
            );
        }
    }
}

fn firstCenter(similarity: []const f32, region_size: usize) usize {
    var best: usize = 0;
    var best_score: f32 = -std.math.inf(f32);
    for (0..region_size) |candidate| {
        var score: f32 = 0;
        for (similarity[candidate * region_size ..][0..region_size]) |value| score += value;
        if (score > best_score) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

fn nextCenter(
    similarity: []const f32,
    max_sim: []const f32,
    selected: []const bool,
    region_size: usize,
    mode: Mode,
) usize {
    var best: usize = 0;
    var best_gain: f32 = -std.math.inf(f32);
    for (0..region_size) |candidate| {
        if (mode == .paper_spec and selected[candidate]) continue;
        var gain: f32 = 0;
        for (similarity[candidate * region_size ..][0..region_size], max_sim) |value, old| {
            gain += @max(@as(f32, 0), value - old);
        }
        if (gain > best_gain) {
            best = candidate;
            best_gain = gain;
        }
    }
    return best;
}

fn assign(
    normalized: []const f32,
    source_tokens: usize,
    hidden: usize,
    config: Config,
    destinations: []const u32,
    assignment: []f32,
) !void {
    if (config.assignment_scope == .global) {
        for (0..source_tokens) |source| {
            try assignColumn(
                normalized,
                source_tokens,
                hidden,
                destinations,
                0,
                destinations.len,
                source,
                config.assignment_scale,
                assignment,
            );
        }
        return;
    }
    const region_size = source_tokens / config.region_count;
    const per_region = destinations.len / config.region_count;
    for (0..config.region_count) |region| {
        for (0..region_size) |local_source| {
            try assignColumn(
                normalized,
                source_tokens,
                hidden,
                destinations,
                region * per_region,
                per_region,
                regionToken(source_tokens, config, region, local_source),
                config.assignment_scale,
                assignment,
            );
        }
    }
}

fn assignColumn(
    normalized: []const f32,
    source_tokens: usize,
    hidden: usize,
    destinations: []const u32,
    destination_from: usize,
    destination_count: usize,
    source: usize,
    scale: f32,
    assignment: []f32,
) !void {
    var maximum: f32 = -std.math.inf(f32);
    for (destination_from..destination_from + destination_count) |destination| {
        const score = scale * dotRows(normalized, hidden, destinations[destination], source);
        maximum = @max(maximum, score);
    }
    var denominator: f32 = 0;
    for (destination_from..destination_from + destination_count) |destination| {
        const score = scale * dotRows(normalized, hidden, destinations[destination], source);
        const weight = @exp(score - maximum);
        assignment[destination * source_tokens + source] = weight;
        denominator += weight;
    }
    if (!std.math.isFinite(denominator) or denominator <= 0) {
        return error.InvalidAssignment;
    }
    for (destination_from..destination_from + destination_count) |destination| {
        assignment[destination * source_tokens + source] /= denominator;
    }
}

fn normalizeRows(assignment: []const f32, source_tokens: usize, output: []f32) !void {
    const destinations = assignment.len / source_tokens;
    for (0..destinations) |destination| {
        const row = assignment[destination * source_tokens ..][0..source_tokens];
        var mass: f32 = 0;
        for (row) |value| mass += value;
        if (!std.math.isFinite(mass) or mass <= 0) return error.InvalidAssignment;
        for (row, 0..) |value, source| {
            output[destination * source_tokens + source] = value / mass;
        }
    }
}

fn applyTranspose(
    allocator: std.mem.Allocator,
    matrix: []const f32,
    source_tokens: usize,
    destination_tokens: usize,
    input: []const f32,
    hidden: usize,
) ![]f32 {
    const output = try allocator.alloc(f32, source_tokens * hidden);
    for (0..source_tokens) |source| {
        for (0..hidden) |channel| {
            var sum: f32 = 0;
            for (0..destination_tokens) |destination| {
                sum += matrix[destination * source_tokens + source] *
                    input[destination * hidden + channel];
            }
            output[source * hidden + channel] = sum;
        }
    }
    return output;
}

fn applyPseudo(
    allocator: std.mem.Allocator,
    pattern: Pattern,
    input: []const f32,
    hidden: usize,
) ![]f32 {
    const destinations = pattern.destination_tokens;
    const gram = try allocator.alloc(f64, destinations * destinations);
    defer allocator.free(gram);
    fillGram(gram, pattern.merge_matrix, pattern.source_tokens, destinations);
    const inverse = try invert(allocator, gram, destinations);
    defer allocator.free(inverse);
    const intermediate = try allocator.alloc(f64, destinations * hidden);
    defer allocator.free(intermediate);
    multiplyInverse(intermediate, inverse, input, destinations, hidden);
    return pseudoTranspose(allocator, pattern, intermediate, hidden);
}

fn fillGram(gram: []f64, matrix: []const f32, sources: usize, destinations: usize) void {
    for (0..destinations) |row| {
        for (0..destinations) |column| {
            var sum: f64 = 0;
            for (0..sources) |source| {
                sum += @as(f64, matrix[row * sources + source]) *
                    @as(f64, matrix[column * sources + source]);
            }
            gram[row * destinations + column] = sum;
        }
    }
}

fn invert(
    allocator: std.mem.Allocator,
    matrix: []const f64,
    size: usize,
) ![]f64 {
    const stride = size * 2;
    const augmented = try allocator.alloc(f64, size * stride);
    defer allocator.free(augmented);
    for (0..size) |row| {
        @memcpy(augmented[row * stride ..][0..size], matrix[row * size ..][0..size]);
        @memset(augmented[row * stride + size ..][0..size], 0);
        augmented[row * stride + size + row] = 1;
    }
    for (0..size) |column| {
        const pivot = pivotRow(augmented, stride, size, column);
        if (@abs(augmented[pivot * stride + column]) <= 1e-12) return error.SingularMatrix;
        swapRows(augmented, stride, pivot, column);
        eliminateColumn(augmented, stride, size, column);
    }
    const inverse = try allocator.alloc(f64, size * size);
    for (0..size) |row| {
        @memcpy(inverse[row * size ..][0..size], augmented[row * stride + size ..][0..size]);
    }
    return inverse;
}

fn pivotRow(matrix: []const f64, stride: usize, size: usize, column: usize) usize {
    var pivot = column;
    var magnitude = @abs(matrix[column * stride + column]);
    for (column + 1..size) |row| {
        const candidate = @abs(matrix[row * stride + column]);
        if (candidate > magnitude) {
            pivot = row;
            magnitude = candidate;
        }
    }
    return pivot;
}

fn swapRows(matrix: []f64, stride: usize, a: usize, b: usize) void {
    if (a == b) return;
    for (0..stride) |column| {
        std.mem.swap(f64, &matrix[a * stride + column], &matrix[b * stride + column]);
    }
}

fn eliminateColumn(matrix: []f64, stride: usize, size: usize, column: usize) void {
    const divisor = matrix[column * stride + column];
    for (0..stride) |entry| matrix[column * stride + entry] /= divisor;
    for (0..size) |row| {
        if (row == column) continue;
        const factor = matrix[row * stride + column];
        for (0..stride) |entry| {
            matrix[row * stride + entry] -= factor * matrix[column * stride + entry];
        }
    }
}

fn multiplyInverse(
    output: []f64,
    inverse: []const f64,
    input: []const f32,
    destinations: usize,
    hidden: usize,
) void {
    for (0..destinations) |row| {
        for (0..hidden) |channel| {
            var sum: f64 = 0;
            for (0..destinations) |column| {
                sum += inverse[row * destinations + column] *
                    @as(f64, input[column * hidden + channel]);
            }
            output[row * hidden + channel] = sum;
        }
    }
}

fn pseudoTranspose(
    allocator: std.mem.Allocator,
    pattern: Pattern,
    intermediate: []const f64,
    hidden: usize,
) ![]f32 {
    const output = try allocator.alloc(f32, pattern.source_tokens * hidden);
    for (0..pattern.source_tokens) |source| {
        for (0..hidden) |channel| {
            var sum: f64 = 0;
            for (0..pattern.destination_tokens) |destination| {
                sum += @as(f64, pattern.merge_matrix[
                    destination * pattern.source_tokens + source
                ]) * intermediate[destination * hidden + channel];
            }
            output[source * hidden + channel] = @floatCast(sum);
        }
    }
    return output;
}

fn regionToken(source_tokens: usize, config: Config, region: usize, local: usize) usize {
    const region_size = source_tokens / config.region_count;
    return switch (config.selection_layout) {
        .global, .stripe => region * region_size + local,
        .tile => tileToken(source_tokens, config.region_count, region, local),
    };
}

fn tileToken(source_tokens: usize, regions: usize, region: usize, local: usize) usize {
    const token_side = squareRoot(source_tokens).?;
    const region_side = squareRoot(regions).?;
    const tile_side = token_side / region_side;
    const region_y = region / region_side;
    const region_x = region % region_side;
    const local_y = local / tile_side;
    const local_x = local % tile_side;
    return (region_y * tile_side + local_y) * token_side +
        region_x * tile_side + local_x;
}

fn dotRows(input: []const f32, hidden: usize, a: usize, b: usize) f32 {
    var sum: f32 = 0;
    for (0..hidden) |channel| {
        sum += input[a * hidden + channel] * input[b * hidden + channel];
    }
    return sum;
}

fn squareRoot(value: usize) ?usize {
    const root: usize = @intFromFloat(@sqrt(@as(f64, @floatFromInt(value))));
    return if (root * root == value) root else null;
}

test "paper facility location selects diverse tokens" {
    const features = [_]f32{
        1, 0,
        1, 0,
        0, 1,
        0, 1,
    };
    var pattern = try build(std.testing.allocator, &features, 4, 2, .{
        .mode = .paper_spec,
        .destination_tokens = 2,
        .selection_layout = .global,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 1,
    });
    defer pattern.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, pattern.destinations);
}

test "paper and pinned code expose their tie behavior" {
    const identical = [_]f32{
        1, 0,
        1, 0,
        1, 0,
        1, 0,
    };
    var paper = try build(std.testing.allocator, &identical, 4, 2, .{
        .mode = .paper_spec,
        .destination_tokens = 2,
        .selection_layout = .global,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 1,
    });
    defer paper.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, paper.destinations);

    var official = try build(std.testing.allocator, &identical, 4, 2, .{
        .mode = .official_b578009,
        .destination_tokens = 2,
        .region_count = 1,
        .selection_layout = .tile,
        .assignment_scope = .region_local,
        .unmerge = .official_raw_transpose,
        .assignment_scale = 1,
    });
    defer official.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, official.destinations);
}

test "assignment columns and merge rows are normalized" {
    const features = [_]f32{
        1, 0,
        1, 0,
        0, 1,
        0, 1,
    };
    var pattern = try build(std.testing.allocator, &features, 4, 2, .{
        .mode = .paper_spec,
        .destination_tokens = 2,
        .selection_layout = .global,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 1,
    });
    defer pattern.deinit(std.testing.allocator);
    for (0..pattern.source_tokens) |source| {
        var sum: f32 = 0;
        for (0..pattern.destination_tokens) |destination| {
            sum += pattern.assignment[destination * pattern.source_tokens + source];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    }
    for (0..pattern.destination_tokens) |destination| {
        var sum: f32 = 0;
        const offset = destination * pattern.source_tokens;
        const row = pattern.merge_matrix[offset..][0..pattern.source_tokens];
        for (row) |value| sum += value;
        try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    }
}

test "paper raw and pseudo unmerge are distinct" {
    var destinations = [_]u32{0};
    var assignment = [_]f32{ 0.5, 0.5 };
    var merge_matrix = [_]f32{ 0.25, 0.75 };
    const input = [_]f32{10};
    var pattern = Pattern{
        .source_tokens = 2,
        .destination_tokens = 1,
        .region_count = 1,
        .destinations = &destinations,
        .assignment = &assignment,
        .merge_matrix = &merge_matrix,
        .unmerge = .paper_normalized_transpose,
    };
    const paper = try applyUnmerge(std.testing.allocator, pattern, &input, 1);
    defer std.testing.allocator.free(paper);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), paper[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), paper[1], 1e-6);

    pattern.unmerge = .official_raw_transpose;
    const official = try applyUnmerge(std.testing.allocator, pattern, &input, 1);
    defer std.testing.allocator.free(official);
    try std.testing.expectEqualSlices(f32, &.{ 5, 5 }, official);

    pattern.unmerge = .pseudo_inverse;
    const pseudo = try applyUnmerge(std.testing.allocator, pattern, &input, 1);
    defer std.testing.allocator.free(pseudo);
    try std.testing.expectApproxEqAbs(@as(f32, 4), pseudo[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12), pseudo[1], 1e-5);
}

test "tile region order is a complete row-major permutation" {
    const expected = [_]usize{
        0,  1,  4,  5,
        2,  3,  6,  7,
        8,  9,  12, 13,
        10, 11, 14, 15,
    };
    const config = Config{
        .mode = .paper_spec,
        .destination_tokens = 8,
        .region_count = 4,
        .selection_layout = .tile,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 1,
    };
    for (0..4) |region| {
        for (0..4) |local| {
            try std.testing.expectEqual(
                expected[region * 4 + local],
                regionToken(16, config, region, local),
            );
        }
    }
}

test "zero vectors are finite in paper mode" {
    const features = [_]f32{
        0, 0,
        1, 0,
        0, 1,
        1, 1,
    };
    var pattern = try build(std.testing.allocator, &features, 4, 2, .{
        .mode = .paper_spec,
        .destination_tokens = 2,
        .selection_layout = .global,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 1,
    });
    defer pattern.deinit(std.testing.allocator);
    for (pattern.assignment) |value| try std.testing.expect(std.math.isFinite(value));
}

test "official small fixture destinations match the pinned oracle" {
    const features = [_]f32{
        1, 0, 0, 0.9, 0.1, 0, 1, 0, 0, 0.9, 0.1, 0,
        0, 1, 0, 0,   0,   1, 0, 1, 0, 0,   0,   1,
        1, 0, 0, 0.9, 0.1, 0, 1, 0, 0, 0.9, 0.1, 0,
        0, 1, 0, 0,   0,   1, 0, 1, 0, 0,   0,   1,
    };
    var pattern = try build(std.testing.allocator, &features, 16, 3, .{
        .mode = .official_b578009,
        .destination_tokens = 8,
        .region_count = 4,
        .selection_layout = .tile,
        .assignment_scope = .region_local,
        .unmerge = .official_raw_transpose,
        .assignment_scale = 3,
    });
    defer pattern.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(
        u32,
        &.{ 1, 5, 3, 7, 9, 13, 11, 15 },
        pattern.destinations,
    );
}
