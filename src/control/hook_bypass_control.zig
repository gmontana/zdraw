//! Scoped execution state for plan-driven transformer-hook bypass.
//!
//! The plan boundary owns activation. The resident Z-Image path may only
//! inspect the current step and report a completed, non-fallback execution.

const std = @import("std");

pub const Plan = struct {
    layer_from: usize,
    layer_to: usize,
    timesteps: []const u32,
    expected_steps: u32,
};

pub const Step = struct {
    index: u32,
    applies: bool,
    layer_from: usize,
    layer_to: usize,
};

pub const Counts = struct {
    expected: u32,
    completed: u32,
};

const State = struct {
    plan: Plan,
    observed_steps: u32 = 0,
    completed: u32 = 0,
};

threadlocal var state: ?State = null;

pub fn activate(plan: Plan) !void {
    if (state != null) return error.ExecutionPlanAlreadyActive;
    if (plan.expected_steps == 0 or plan.layer_from >= plan.layer_to) {
        return error.InvalidBypassPlan;
    }
    if (plan.timesteps.len == 0) return error.InvalidBypassPlan;
    for (plan.timesteps, 0..) |timestep, index| {
        if (timestep >= plan.expected_steps or
            (index > 0 and plan.timesteps[index - 1] >= timestep))
        {
            return error.InvalidBypassPlan;
        }
    }
    state = .{ .plan = plan };
}

pub fn deactivate() void {
    state = null;
}

pub fn managed() bool {
    return state != null;
}

pub fn current() ?Step {
    const value = state orelse return null;
    if (value.observed_steps >= value.plan.expected_steps) return null;
    return .{
        .index = value.observed_steps,
        .applies = contains(value.plan.timesteps, value.observed_steps),
        .layer_from = value.plan.layer_from,
        .layer_to = value.plan.layer_to,
    };
}

pub fn recordCompleted(applied: bool) !void {
    const value = if (state) |*item| item else return;
    const step = current() orelse return error.TooManyBypassSteps;
    if (step.applies != applied) return error.BypassExecutionMismatch;
    value.observed_steps += 1;
    if (applied) value.completed += 1;
}

pub fn counts() Counts {
    const value = state orelse return .{ .expected = 0, .completed = 0 };
    return .{
        .expected = @intCast(value.plan.timesteps.len),
        .completed = value.completed,
    };
}

pub fn verifyCompleted() !void {
    const value = state orelse return;
    if (value.observed_steps != value.plan.expected_steps or
        value.completed != @as(u32, @intCast(value.plan.timesteps.len)))
    {
        return error.IncompleteBypassExecution;
    }
}

fn contains(values: []const u32, needle: u32) bool {
    for (values) |value| {
        if (value == needle) return true;
        if (value > needle) return false;
    }
    return false;
}

test "bypass policy accounts for every denoising step" {
    try activate(.{
        .layer_from = 17,
        .layer_to = 19,
        .timesteps = &.{ 0, 2 },
        .expected_steps = 3,
    });
    defer deactivate();
    try std.testing.expect(current().?.applies);
    try recordCompleted(true);
    try std.testing.expect(!current().?.applies);
    try recordCompleted(false);
    try std.testing.expect(current().?.applies);
    try recordCompleted(true);
    try verifyCompleted();
    try std.testing.expectEqual(@as(u32, 2), counts().completed);
}
