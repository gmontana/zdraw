//! Scoped control and evidence state for plan-driven ToMA execution.
//!
//! Normal product and lab runs remain environment-driven. An execution plan
//! installs one typed control value for the duration of a single CLI command;
//! while installed, it takes precedence over environment flags. This prevents
//! a baseline plan from accidentally inheriting ToMA and prevents a ToMA plan
//! from silently falling back while still producing a successful receipt.

const std = @import("std");

const toma_config = @import("toma_config.zig");

pub const Plan = struct {
    source_tokens: usize,
    layer_from: usize,
    layer_to: usize,
    config: toma_config.Config,
    expected_steps: u32,
};

pub const Counts = struct {
    expected: u32,
    completed: u32,
};

const State = struct {
    plan: ?Plan,
    expected_steps: u32,
    completed_steps: u32 = 0,
};

threadlocal var state: ?State = null;

pub fn startBaseline(expected_steps: u32) !void {
    try activate(.{
        .plan = null,
        .expected_steps = expected_steps,
    });
}

pub fn activateToma(value: Plan) !void {
    if (value.expected_steps == 0) return error.InvalidExpectedSteps;
    if (value.layer_from >= value.layer_to) return error.InvalidLayerRange;
    try activate(.{
        .plan = value,
        .expected_steps = value.expected_steps,
    });
}

fn activate(value: State) !void {
    if (state != null) return error.ExecutionPlanAlreadyActive;
    state = value;
}

pub fn deactivate() void {
    state = null;
}

pub fn managed() bool {
    return state != null;
}

pub fn plan() ?Plan {
    const current = state orelse return null;
    return current.plan;
}

pub fn recordCompleted() !void {
    const current = if (state) |*value| value else return;
    if (current.plan == null) return error.UnexpectedTomaExecution;
    if (current.completed_steps >= current.expected_steps) {
        return error.TooManyTomaExecutions;
    }
    current.completed_steps += 1;
}

pub fn counts() Counts {
    const current = state orelse return .{ .expected = 0, .completed = 0 };
    return .{
        .expected = if (current.plan == null) 0 else current.expected_steps,
        .completed = current.completed_steps,
    };
}

pub fn verifyCompleted() !void {
    const current = state orelse return;
    if (current.plan == null) {
        if (current.completed_steps != 0) return error.UnexpectedTomaExecution;
        return;
    }
    if (current.completed_steps != current.expected_steps) {
        return error.IncompleteTomaExecution;
    }
}

test "baseline control suppresses ToMA and verifies no executions" {
    try startBaseline(4);
    defer deactivate();
    try std.testing.expect(managed());
    try std.testing.expect(plan() == null);
    try verifyCompleted();
    try std.testing.expectError(error.UnexpectedTomaExecution, recordCompleted());
}

test "ToMA control requires every expected denoising step" {
    try activateToma(.{
        .source_tokens = 4096,
        .layer_from = 12,
        .layer_to = 24,
        .config = .{
            .mode = .paper_spec,
            .destination_tokens = 2048,
            .region_count = 64,
            .selection_layout = .tile,
            .assignment_scope = .region_local,
            .unmerge = .paper_normalized_transpose,
            .assignment_scale = 1000,
        },
        .expected_steps = 2,
    });
    defer deactivate();
    try recordCompleted();
    try std.testing.expectError(error.IncompleteTomaExecution, verifyCompleted());
    try recordCompleted();
    try verifyCompleted();
    const got = counts();
    try std.testing.expectEqual(@as(u32, 2), got.expected);
    try std.testing.expectEqual(@as(u32, 2), got.completed);
}
