//! FLUX.2 Klein FlowMatch schedule.
//!
//! Klein uses dynamic shifting: the base sigma ramp is shifted by an empirical
//! mu that depends on latent sequence length and step count. Keep this separate
//! from Z-Image's fixed-shift scheduler; the formulas are model-family policy,
//! not generic flow-matching infrastructure.

const std = @import("std");

pub const Schedule = struct {
    timesteps: []f32,
    sigmas: []f32,

    pub fn deinit(self: Schedule, allocator: std.mem.Allocator) void {
        allocator.free(self.timesteps);
        allocator.free(self.sigmas);
    }
};

pub const Error = error{
    InvalidSteps,
    InvalidSequenceLength,
};

pub fn make(
    allocator: std.mem.Allocator,
    image_seq_len: usize,
    steps: u32,
) !Schedule {
    if (steps == 0) return error.InvalidSteps;
    if (image_seq_len == 0) return error.InvalidSequenceLength;

    const count: usize = @intCast(steps);
    const timesteps = try allocator.alloc(f32, count);
    errdefer allocator.free(timesteps);
    const sigmas = try allocator.alloc(f32, count + 1);
    errdefer allocator.free(sigmas);

    const mu = shiftMu(image_seq_len, steps);
    const last: f32 = 1.0 / @as(f32, @floatFromInt(count));
    for (0..count) |idx| {
        sigmas[idx] = timeShift(mu, lerp(1.0, last, idx, count));
        timesteps[idx] = sigmas[idx] * 1000.0;
    }
    sigmas[count] = 0.0;

    return .{ .timesteps = timesteps, .sigmas = sigmas };
}

/// Empirical Klein mu from diffusers' FLUX.2 pipeline. The constants are
/// verified below against the captured 1024px/4-step oracle.
pub fn shiftMu(image_seq_len: usize, steps: u32) f32 {
    const seq: f32 = @floatFromInt(image_seq_len);
    const step_count: f32 = @floatFromInt(steps);
    const a1: f32 = 8.73809524e-05;
    const b1: f32 = 1.89833333;
    const a2: f32 = 0.00016927;
    const b2: f32 = 0.45666666;
    const m10 = a1 * seq + b1;
    const m200 = a2 * seq + b2;
    const slope = (m200 - m10) / 190.0;
    const intercept = m200 - 200.0 * slope;
    return slope * step_count + intercept;
}

/// img2img: squeeze the whole schedule into [start, 0] so the sampler
/// begins at exactly `start` (the noise fraction) and still takes every
/// step; the model is conditioned on the scaled sigma through timesteps.
pub fn scaleTo(schedule: *Schedule, start: f32) void {
    for (schedule.sigmas) |*sg| sg.* *= start;
    for (schedule.timesteps, 0..) |*t, i| t.* = schedule.sigmas[i] * 1000.0;
}

pub fn delta(schedule: Schedule, index: usize) !f32 {
    if (index + 1 >= schedule.sigmas.len) return error.InvalidSteps;
    return schedule.sigmas[index + 1] - schedule.sigmas[index];
}

fn timeShift(mu: f32, sigma: f32) f32 {
    if (sigma >= 1.0) return 1.0;
    const e = @exp(mu);
    return e / (e + (1.0 / sigma - 1.0));
}

fn lerp(start: f32, end: f32, idx: usize, count: usize) f32 {
    if (count == 1) return start;
    const pos: f32 = @floatFromInt(idx);
    const den: f32 = @floatFromInt(count - 1);
    return start + (end - start) * (pos / den);
}

test "klein schedule matches captured 1024px anchor" {
    const schedule = try make(std.testing.allocator, 4096, 4);
    defer schedule.deinit(std.testing.allocator);

    const want = [_]f32{ 1000.0, 967.384, 908.144, 767.200 };
    try std.testing.expectEqual(@as(usize, 4), schedule.timesteps.len);
    try std.testing.expectEqual(@as(usize, 5), schedule.sigmas.len);
    for (want, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(value, schedule.timesteps[i], 0.002);
        try std.testing.expectApproxEqAbs(value / 1000.0, schedule.sigmas[i], 0.000002);
    }
    try std.testing.expectApproxEqAbs(0.0, schedule.sigmas[4], 0.0);
}

test "klein schedule at the base model's 50 steps matches diffusers" {
    // diffusers 0.39.0, FlowMatchEulerDiscreteScheduler with
    // sigmas = linspace(1, 1/50, 50) and mu = compute_empirical_mu(4096, 50),
    // captured 2026-10-04: an anchor against drift, not a derivation.
    try std.testing.expectApproxEqAbs(@as(f32, 2.02335116), shiftMu(4096, 50), 0.00001);
    const schedule = try make(std.testing.allocator, 4096, 50);
    defer schedule.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 50), schedule.timesteps.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1000.0), schedule.timesteps[0], 0.002);
    try std.testing.expectApproxEqAbs(@as(f32, 997.30908), schedule.timesteps[1], 0.002);
    try std.testing.expectApproxEqAbs(@as(f32, 883.22711), schedule.timesteps[25], 0.002);
    try std.testing.expectApproxEqAbs(@as(f32, 133.71895), schedule.timesteps[49], 0.002);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), schedule.sigmas[50], 0.0);
}

test "klein schedule rejects empty inputs" {
    try std.testing.expectError(error.InvalidSteps, make(std.testing.allocator, 4096, 0));
    try std.testing.expectError(error.InvalidSequenceLength, make(std.testing.allocator, 0, 4));
}

test "delta is next sigma minus current sigma" {
    const schedule = try make(std.testing.allocator, 4096, 4);
    defer schedule.deinit(std.testing.allocator);

    try std.testing.expect(try delta(schedule, 0) < 0.0);
    try std.testing.expectError(error.InvalidSteps, delta(schedule, 4));
}
