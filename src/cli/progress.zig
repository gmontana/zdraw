//! Small progress lines for slow model runs.
//!
//! The CLI can sit in one matrix multiply for a long time while the CPU
//! reference path is still young. These lines go to stderr so image output and
//! saved PNGs stay untouched.

const std = @import("std");

const sink = @import("progress_sink.zig");
const metrics = @import("../metal/metrics.zig");
const bar = @import("progress_bar.zig");
const load = @import("progress_load.zig");

const Mode = enum {
    verbose,
    compact,
    quiet,
};

var load_current: usize = 0;
var load_label: []const u8 = "";

pub const Stage = struct {
    name: []const u8,
    start: std.Io.Timestamp,
    metal: metrics.Counters,
};

pub fn begin(
    io: std.Io,
    allocator: std.mem.Allocator,
    name: []const u8,
) !Stage {
    switch (mode()) {
        .quiet => {},
        .compact => try compactBegin(io, allocator, name),
        .verbose => {
            const text = try std.fmt.allocPrint(allocator, "zdraw: {s}...\n", .{name});
            defer allocator.free(text);
            try write(io, text);
        },
    }
    return .{
        .name = name,
        .start = std.Io.Timestamp.now(io, .awake),
        .metal = metrics.snapshot(),
    };
}

fn compactBegin(io: std.Io, allocator: std.mem.Allocator, name: []const u8) !void {
    if (load.isStage(name)) {
        load_label = load.label(name);
        if (load.isStart(name)) {
            load_current = 0;
            return bar.write(io, allocator, .{
                .label = "loading ",
                .current = load_current,
                .total = load.total,
                .detail = load_label,
            });
        }
        return;
    }
    const label = compactLabel(name) orelse return;
    const text = try std.fmt.allocPrint(allocator, "  {s}\n", .{label});
    defer allocator.free(text);
    try write(io, text);
}

fn compactLabel(name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "encoding prompt")) return "encode   prompt";
    if (std.mem.eql(u8, name, "decoding image")) return "decode   image";
    return null;
}

pub fn done(io: std.Io, allocator: std.mem.Allocator, stage: Stage) !void {
    const elapsed = stage.start.untilNow(io, .awake);
    metrics.record(stage.name, @intCast(elapsed.toNanoseconds()));
    metrics.recordMetal(stage.name, stage.metal);
    switch (mode()) {
        .quiet => return,
        .compact => return compactDone(io, allocator, stage.name),
        .verbose => {},
    }

    const text = try std.fmt.allocPrint(allocator, "zdraw: {s} done ({d} ms)\n", .{
        stage.name,
        elapsed.toMilliseconds(),
    });
    defer allocator.free(text);
    try write(io, text);
}

/// A preview PNG for the app's progressive display (verbose mode only).
pub fn preview(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    current: usize,
    total: usize,
) !void {
    if (mode() != .verbose) return;
    const fmt = "zdraw: preview {s} step {d}/{d}\n";
    const text = try std.fmt.allocPrint(allocator, fmt, .{ path, current, total });
    defer allocator.free(text);
    try write(io, text);
}

pub fn event(io: std.Io, allocator: std.mem.Allocator, name: []const u8) !void {
    if (mode() != .verbose) return;

    const text = try std.fmt.allocPrint(allocator, "zdraw: {s}\n", .{name});
    defer allocator.free(text);
    try write(io, text);
}

pub fn tokens(io: std.Io, allocator: std.mem.Allocator, count: usize) !void {
    if (mode() != .verbose) return;

    const text = try std.fmt.allocPrint(allocator, "zdraw: prompt tokens: {d}\n", .{count});
    defer allocator.free(text);
    try write(io, text);
}

pub fn layer(
    io: std.Io,
    allocator: std.mem.Allocator,
    name: []const u8,
    current: usize,
    total: usize,
) !void {
    if (mode() != .verbose) return;

    const text = try std.fmt.allocPrint(allocator, "zdraw: {s} layer {d}/{d}\n", .{
        name,
        current,
        total,
    });
    defer allocator.free(text);
    try write(io, text);
}

pub fn step(
    io: std.Io,
    allocator: std.mem.Allocator,
    current: usize,
    total: usize,
) !void {
    switch (mode()) {
        .quiet => return,
        .compact => return bar.write(io, allocator, .{
            .label = "denoise ",
            .current = current,
            .total = total,
            .finish_newline = true,
            .show_count = true,
        }),
        .verbose => {},
    }

    const text = try std.fmt.allocPrint(allocator, "zdraw: denoise step {d}/{d}\n", .{
        current,
        total,
    });
    defer allocator.free(text);
    try write(io, text);
}

fn compactDone(io: std.Io, allocator: std.mem.Allocator, name: []const u8) !void {
    if (!load.isStage(name)) return;
    load_current = @min(load.total, load_current + 1);
    const detail = if (load_current >= load.total) "ready" else load_label;
    try bar.write(io, allocator, .{
        .label = "loading ",
        .current = load_current,
        .total = load.total,
        .finish_newline = true,
        .detail = detail,
    });
}

fn mode() Mode {
    const raw = std.c.getenv("ZDRAW_PROGRESS") orelse return .verbose;
    const value = std.mem.span(raw);
    if (std.mem.eql(u8, value, "quiet")) return .quiet;
    if (std.mem.eql(u8, value, "off")) return .quiet;
    if (std.mem.eql(u8, value, "compact")) return .compact;
    if (std.mem.eql(u8, value, "bar")) return .compact;
    return .verbose;
}

fn write(io: std.Io, text: []const u8) !void {
    try sink.text(io, text);
}

test "stage keeps its label" {
    const stage = Stage{ .name = "load", .start = .zero, .metal = .{} };
    try std.testing.expectEqualStrings("load", stage.name);
}
