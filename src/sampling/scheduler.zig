//! FlowMatch Euler schedule foundation.

const std = @import("std");

pub const Config = struct {
    train_steps: u32 = 1000,
    shift: f32 = 3.0,
    dynamic: bool = false,

    pub fn zImageTurbo() Config {
        return .{ .train_steps = 1000, .shift = 3.0, .dynamic = false };
    }
};

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
    UnsupportedDynamic,
};

pub fn make(
    allocator: std.mem.Allocator,
    config: Config,
    steps: u32,
) !Schedule {
    if (steps == 0) return error.InvalidSteps;
    if (config.dynamic) return error.UnsupportedDynamic;

    const count: usize = @intCast(steps);
    const timesteps = try allocator.alloc(f32, count);
    errdefer allocator.free(timesteps);
    const sigmas = try allocator.alloc(f32, count + 1);
    errdefer allocator.free(sigmas);

    const train: f32 = @floatFromInt(config.train_steps);
    const last: f32 = 1.0 / @as(f32, @floatFromInt(count));
    for (0..count) |idx| {
        // Base sigmas are linspace(1, 1/N, N) (diffusers get_default_z_image_sigmas),
        // then the static FlowMatch shift; the timestep follows the shifted sigma.
        sigmas[idx] = shift(config.shift, lerp(1.0, last, idx, count));
        timesteps[idx] = sigmas[idx] * train;
    }
    sigmas[count] = 0.0;

    return .{ .timesteps = timesteps, .sigmas = sigmas };
}

pub fn makeZImage(allocator: std.mem.Allocator, steps: u32) !Schedule {
    var config = Config.zImageTurbo();
    if (std.c.getenv("ZDRAW_SHIFT")) |raw| {
        const text = std.mem.span(raw);
        config.shift = std.fmt.parseFloat(f32, text) catch config.shift;
    }
    return make(allocator, config, steps);
}

pub fn delta(schedule: Schedule, index: usize) !f32 {
    if (index + 1 >= schedule.sigmas.len) return error.InvalidSteps;
    return schedule.sigmas[index + 1] - schedule.sigmas[index];
}

fn shift(amount: f32, sigma: f32) f32 {
    return amount * sigma / (1.0 + (amount - 1.0) * sigma);
}

fn lerp(start: f32, end: f32, idx: usize, count: usize) f32 {
    if (count == 1) return start;
    const pos: f32 = @floatFromInt(idx);
    const den: f32 = @floatFromInt(count - 1);
    return start + (end - start) * (pos / den);
}

test "z-image schedule matches fixed shift shape" {
    const schedule = try make(std.testing.allocator, Config.zImageTurbo(), 9);
    defer schedule.deinit(std.testing.allocator);

    const timestep_count: usize = 9;
    const sigma_count: usize = 10;
    const first_time: f32 = 1000.0;
    const first_sigma: f32 = 1.0;
    const last_sigma: f32 = 0.0;

    try std.testing.expectEqual(timestep_count, schedule.timesteps.len);
    try std.testing.expectEqual(sigma_count, schedule.sigmas.len);
    try std.testing.expectApproxEqAbs(first_time, schedule.timesteps[0], 0.001);
    try std.testing.expectApproxEqAbs(first_sigma, schedule.sigmas[0], 0.0001);
    try std.testing.expectApproxEqAbs(last_sigma, schedule.sigmas[9], 0.0);
    try std.testing.expect(schedule.sigmas[1] < schedule.sigmas[0]);
}

test "delta is next minus current" {
    const schedule = try make(std.testing.allocator, Config.zImageTurbo(), 2);
    defer schedule.deinit(std.testing.allocator);

    try std.testing.expect(try delta(schedule, 0) < 0.0);
    try std.testing.expectError(error.InvalidSteps, delta(schedule, 2));
}

test "z-image schedule follows pipeline sigmas" {
    const schedule = try makeZImage(std.testing.allocator, 4);
    defer schedule.deinit(std.testing.allocator);

    try std.testing.expectApproxEqAbs(1.0, schedule.sigmas[0], 0.0001);
    try std.testing.expectApproxEqAbs(0.9, schedule.sigmas[1], 0.0001);
    try std.testing.expectApproxEqAbs(0.75, schedule.sigmas[2], 0.0001);
    try std.testing.expectApproxEqAbs(0.5, schedule.sigmas[3], 0.0001);
    try std.testing.expectApproxEqAbs(0.0, schedule.sigmas[4], 0.0001);
    try std.testing.expectApproxEqAbs(500.0, schedule.timesteps[3], 0.0001);
}
