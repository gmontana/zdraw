//! Command helpers for the interactive zdraw session.

const std = @import("std");
const stanza = @import("stanza");

const args = @import("args.zig");
const completion = @import("session_complete.zig");
const metrics = @import("metrics.zig");
const session_state = @import("session_state.zig");
const util = @import("session_util.zig");

pub const Dimensions = struct {
    width: u32,
    height: u32,
};

pub const complete = completion.complete;
pub const paint = completion.paint;

pub fn setSeed(
    out: *util.Output,
    allocator: std.mem.Allocator,
    request: *args.Session,
    value: []const u8,
) !void {
    const clean = util.clean(value);
    request.seed = std.fmt.parseInt(u64, clean, 10) catch {
        try util.writeText(out, allocator, "invalid seed\n");
        return;
    };
    try util.showPath(out, allocator, "seed ", clean);
}

pub fn setSteps(
    out: *util.Output,
    allocator: std.mem.Allocator,
    request: *args.Session,
    value: []const u8,
) !void {
    const steps = std.fmt.parseInt(u32, util.clean(value), 10) catch {
        try util.writeText(out, allocator, "invalid steps\n");
        return;
    };
    if (steps == 0) {
        try util.writeText(out, allocator, "steps must be > 0\n");
        return;
    }
    request.steps = steps;
    request.steps_explicit = true;
    const msg = try std.fmt.allocPrint(allocator, "steps {d}\n", .{steps});
    defer allocator.free(msg);
    try util.writeText(out, allocator, msg);
}

pub fn setSize(
    out: *util.Output,
    allocator: std.mem.Allocator,
    request: *args.Session,
    value: []const u8,
) !void {
    const dims = parseDimensions(value) catch |err| {
        try sizeError(out, allocator, err);
        return;
    };
    args.checkDimensions(request.kind, dims.width, dims.height, request.preview) catch |err| {
        if (err == error.UnsupportedKleinResolution) {
            return util.writeText(out, allocator, args.klein_resolution_hint ++ "\n");
        }
        return util.writeText(out, allocator, "use multiples of 16 for Z-Image, 32 for Klein\n");
    };
    request.width = dims.width;
    request.height = dims.height;
    const msg = try std.fmt.allocPrint(
        allocator,
        "size {d}x{d}\n",
        .{ dims.width, dims.height },
    );
    defer allocator.free(msg);
    try util.writeText(out, allocator, msg);
}

pub fn writeResult(
    out: *util.Output,
    allocator: std.mem.Allocator,
    request: args.Session,
    seed: u64,
    ns: u64,
    path: ?[]const u8,
    stats: session_state.Stats,
) !void {
    const memory = metrics.memory();
    const elapsed = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
    const footprint = try formatGiB(allocator, memory.phys_footprint_bytes);
    defer allocator.free(footprint);
    const text = if (path) |saved_path|
        try std.fmt.allocPrint(
            allocator,
            "  saved   {s}\n" ++
                "  result  {d}x{d} | {d} steps | seed {d} | {d:.2}s | mem {s}\n",
            .{
                saved_path,
                request.width,
                request.height,
                request.steps,
                seed,
                elapsed,
                footprint,
            },
        )
    else
        try std.fmt.allocPrint(
            allocator,
            "  result  {d}x{d} | {d} steps | seed {d} | {d:.2}s | mem {s}\n",
            .{
                request.width,
                request.height,
                request.steps,
                seed,
                elapsed,
                footprint,
            },
        );
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
    if (stats.verbose) try writeStats(out, allocator, memory);
}

pub fn setStats(
    out: *util.Output,
    allocator: std.mem.Allocator,
    stats: *session_state.Stats,
    text: []const u8,
) !void {
    const arg = util.clean(text["stats".len..]);
    if (arg.len == 0) return statsStatus(out, allocator, stats.*);
    if (std.mem.eql(u8, arg, "on")) {
        stats.verbose = true;
        return statsStatus(out, allocator, stats.*);
    }
    if (std.mem.eql(u8, arg, "off")) {
        stats.verbose = false;
        return statsStatus(out, allocator, stats.*);
    }
    try util.writeText(out, allocator, "usage: stats | stats on | stats off\n");
}

pub fn showPrompt(
    out: *util.Output,
    allocator: std.mem.Allocator,
    prompt: []const u8,
) !void {
    if (prompt.len == 0) {
        try util.writeText(out, allocator, "prompt empty\n");
        return;
    }
    const text = try std.fmt.allocPrint(allocator, "prompt {s}\n", .{prompt});
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
}

pub fn hint(_: ?*anyopaque, line: []const u8) ?stanza.Hint {
    if (line.len != 0) return null;
    return .{ .text = "type a prompt, or help for commands" };
}

fn writeStats(
    out: *util.Output,
    allocator: std.mem.Allocator,
    memory: metrics.Memory,
) !void {
    const footprint = try formatGiB(allocator, memory.phys_footprint_bytes);
    defer allocator.free(footprint);
    const gpu = try formatGiB(allocator, memory.peak_gpu_live_bytes);
    defer allocator.free(gpu);
    const weights = try formatGiB(allocator, memory.weight_bytes);
    defer allocator.free(weights);
    const rss = try formatGiB(allocator, memory.peak_rss_bytes);
    defer allocator.free(rss);
    const text = try std.fmt.allocPrint(
        allocator,
        "  stats   footprint {s} | gpu {s} | weights {s} | rss {s}\n",
        .{ footprint, gpu, weights, rss },
    );
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
}

fn statsStatus(
    out: *util.Output,
    allocator: std.mem.Allocator,
    stats: session_state.Stats,
) !void {
    const status = if (stats.verbose) "on" else "off";
    const text = try std.fmt.allocPrint(allocator, "stats {s}\n", .{status});
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
}

fn formatGiB(allocator: std.mem.Allocator, bytes: u64) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{d:.1} GB",
        .{@as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0)},
    );
}

pub fn parseDimensions(value: []const u8) !Dimensions {
    const text = util.clean(value);
    const split = std.mem.indexOfScalar(u8, text, 'x') orelse
        std.mem.indexOfScalar(u8, text, 'X') orelse
        return error.MissingSeparator;
    const width = try parseDim(text[0..split]);
    const height = try parseDim(text[split + 1 ..]);
    return .{ .width = width, .height = height };
}

fn parseDim(text: []const u8) !u32 {
    const value = std.fmt.parseInt(u32, util.clean(text), 10) catch {
        return error.InvalidDimension;
    };
    if (value == 0) return error.InvalidDimension;
    return value;
}

fn sizeError(out: *util.Output, allocator: std.mem.Allocator, err: anyerror) !void {
    const message = switch (err) {
        error.MissingSeparator => "size must be WIDTHxHEIGHT\n",
        else => "size dimensions must be positive integers\n",
    };
    try util.writeText(out, allocator, message);
}

test "parse session dimensions" {
    const dims = try parseDimensions("256x512");
    try std.testing.expectEqual(@as(u32, 256), dims.width);
    try std.testing.expectEqual(@as(u32, 512), dims.height);
    try std.testing.expectError(error.MissingSeparator, parseDimensions("256"));
    try std.testing.expectError(error.InvalidDimension, parseDimensions("0x512"));
}
