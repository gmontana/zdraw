//! Scoped execution state for one capsule-bound token-selection route.
//!
//! The plan boundary owns activation. The resident stack may only inspect the
//! immutable current-step policy and report whether the requested route ran.

const std = @import("std");

pub const Score = enum {
    feature_mean,
    zero,
};

pub const Reconstruction = enum {
    reference_scatter,
    cosine_residual_interpolate,
};

pub const Plan = struct {
    semantic_program_sha256: []const u8,
    selection_program_sha256: []const u8,
    realization_sha256: []const u8,
    schedule_sha256: []const u8,
    source_sha256: []const u8,
    source_tokens: usize,
    feature_width: usize,
    selected_tokens: usize,
    layers: []const u32,
    timesteps: []const u32,
    expected_steps: u32,
    score: Score = .feature_mean,
    reconstruction: Reconstruction = .reference_scatter,
    reconstruction_weight: f32 = 0,
};

pub const Step = struct {
    index: u32,
    applies: bool,
    plan: Plan,
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
    if (plan.expected_steps == 0 or
        plan.source_tokens == 0 or
        plan.feature_width == 0 or
        plan.selected_tokens == 0 or
        plan.selected_tokens >= plan.source_tokens or
        plan.source_tokens % 32 != 0 or
        plan.selected_tokens % 32 != 0 or
        plan.layers.len == 0 or
        plan.timesteps.len == 0 or
        !std.math.isFinite(plan.reconstruction_weight))
    {
        return error.InvalidTokenSelectionPlan;
    }
    switch (plan.reconstruction) {
        .reference_scatter => if (plan.reconstruction_weight != 0) {
            return error.InvalidTokenSelectionPlan;
        },
        .cosine_residual_interpolate => if (plan.reconstruction_weight < 0 or
            plan.reconstruction_weight > 1)
        {
            return error.InvalidTokenSelectionPlan;
        },
    }
    for (plan.layers, 0..) |layer, index| {
        if (index > 0 and plan.layers[index - 1] >= layer) {
            return error.InvalidTokenSelectionPlan;
        }
    }
    for (plan.timesteps, 0..) |timestep, index| {
        if (timestep >= plan.expected_steps or
            (index > 0 and plan.timesteps[index - 1] >= timestep))
        {
            return error.InvalidTokenSelectionPlan;
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
        .plan = value.plan,
    };
}

pub fn recordCompleted(applied: bool) !void {
    const value = if (state) |*item| item else return;
    const step = current() orelse return error.TooManyTokenSelectionSteps;
    if (step.applies != applied) return error.TokenSelectionExecutionMismatch;
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
        return error.IncompleteTokenSelectionExecution;
    }
}

fn contains(values: []const u32, needle: u32) bool {
    for (values) |value| {
        if (value == needle) return true;
        if (value > needle) return false;
    }
    return false;
}

test "token selection accounts for applied and baseline denoising steps" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    try activate(.{
        .semantic_program_sha256 = hash,
        .selection_program_sha256 = hash,
        .realization_sha256 = hash,
        .schedule_sha256 = hash,
        .source_sha256 = hash,
        .source_tokens = 64,
        .feature_width = 3840,
        .selected_tokens = 32,
        .layers = &.{ 0, 10, 20 },
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

test "token selection rejects shapes the resident MPS realizer cannot encode" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    try std.testing.expectError(error.InvalidTokenSelectionPlan, activate(.{
        .semantic_program_sha256 = hash,
        .selection_program_sha256 = hash,
        .realization_sha256 = hash,
        .schedule_sha256 = hash,
        .source_sha256 = hash,
        .source_tokens = 64,
        .feature_width = 32,
        .selected_tokens = 31,
        .layers = &.{0},
        .timesteps = &.{0},
        .expected_steps = 1,
    }));
}
