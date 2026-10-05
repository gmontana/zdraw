//! Compact terminal progress bars shared by loading and generation stages.

const std = @import("std");
const sink = @import("progress_sink.zig");

pub const Options = struct {
    label: []const u8,
    current: usize,
    total: usize,
    finish_newline: bool = false,
    show_count: bool = false,
    detail: []const u8 = "",
};

pub fn write(io: std.Io, allocator: std.mem.Allocator, options: Options) !void {
    const width: usize = 24;
    const filled = filledCells(width, options.current, options.total);
    var out = try std.ArrayList(u8).initCapacity(allocator, 96);
    defer out.deinit(allocator);

    try out.appendSlice(allocator, "\r  ");
    try out.appendSlice(allocator, options.label);
    try out.appendSlice(allocator, " [");
    for (0..width) |idx| {
        try out.append(allocator, if (idx < filled) '=' else '.');
    }
    const tail = try suffix(allocator, options);
    defer allocator.free(tail);
    try out.appendSlice(allocator, tail);
    try sink.text(io, out.items);
}

fn suffix(allocator: std.mem.Allocator, options: Options) ![]u8 {
    const newline = if (options.finish_newline and options.current >= options.total) "\n" else "";
    if (options.show_count) {
        return std.fmt.allocPrint(allocator, "] {d}/{d} {d}%{s}", .{
            options.current,
            options.total,
            percent(options.current, options.total),
            newline,
        });
    }
    if (options.detail.len > 0) {
        return std.fmt.allocPrint(allocator, "] {d}%  {s}{s}", .{
            percent(options.current, options.total),
            options.detail,
            newline,
        });
    }
    return std.fmt.allocPrint(allocator, "] {d}%{s}", .{
        percent(options.current, options.total),
        newline,
    });
}

fn filledCells(width: usize, current: usize, total: usize) usize {
    if (total == 0) return width;
    return @min(width, (current * width) / total);
}

fn percent(current: usize, total: usize) usize {
    if (total == 0) return 100;
    return @min(@as(usize, 100), (current * 100) / total);
}

test "progress bar fill is bounded" {
    try std.testing.expectEqual(@as(usize, 0), filledCells(24, 0, 4));
    try std.testing.expectEqual(@as(usize, 12), filledCells(24, 2, 4));
    try std.testing.expectEqual(@as(usize, 24), filledCells(24, 8, 4));
}

test "progress bar percent is bounded" {
    try std.testing.expectEqual(@as(usize, 0), percent(0, 4));
    try std.testing.expectEqual(@as(usize, 50), percent(2, 4));
    try std.testing.expectEqual(@as(usize, 100), percent(8, 4));
}

test "compact bars respect the active progress observer" {
    const Observer = struct {
        fn text(_: *sink.Sink, _: std.Io, _: []const u8) !void {
            return error.ProgressObserved;
        }
        fn frame(_: *sink.Sink, _: std.Io, _: std.mem.Allocator, _: sink.Frame) !void {}
    };
    var observer = sink.Sink{ .on_text = Observer.text, .on_frame = Observer.frame };
    try observer.activate();
    defer observer.deactivate();
    const observed = write(std.testing.io, std.testing.allocator, .{
        .label = "denoise",
        .current = 4,
        .total = 4,
        .finish_newline = true,
    });
    try std.testing.expectError(error.ProgressObserved, observed);
}
