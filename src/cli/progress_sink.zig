//! Borrowed, synchronous progress observer. The CLI owns its scoped lifetime.
const std = @import("std");

pub const Frame = struct {
    pixels: []const u8,
    width: u32,
    height: u32,
    step: usize,
    total: usize,
    index: usize,
};

pub const Sink = struct {
    on_text: *const fn (*Sink, std.Io, []const u8) anyerror!void,
    on_frame: *const fn (*Sink, std.Io, std.mem.Allocator, Frame) anyerror!void,

    pub fn activate(self: *Sink) !void {
        if (current != null) return error.ProgressAlreadyActive;
        current = self;
    }

    pub fn deactivate(self: *Sink) void {
        std.debug.assert(current == self);
        current = null;
    }
};

threadlocal var current: ?*Sink = null;

pub fn active() bool {
    return current != null;
}

pub fn text(io: std.Io, message: []const u8) !void {
    if (current) |sink| return sink.on_text(sink, io, message);
    try std.Io.File.stderr().writeStreamingAll(io, message);
}

/// Pixels are borrowed only for this call; an observer must not retain them.
pub fn frame(io: std.Io, allocator: std.mem.Allocator, value: Frame) !void {
    if (current) |sink| try sink.on_frame(sink, io, allocator, value);
}

test "progress observers reject nesting and release their scope" {
    const Noop = struct {
        fn text(_: *Sink, _: std.Io, _: []const u8) !void {}
        fn frame(_: *Sink, _: std.Io, _: std.mem.Allocator, _: Frame) !void {}
    };
    var first = Sink{ .on_text = Noop.text, .on_frame = Noop.frame };
    var second = first;
    try first.activate();
    try std.testing.expectError(error.ProgressAlreadyActive, second.activate());
    first.deactivate();
    try std.testing.expect(!active());
    try second.activate();
    second.deactivate();
    try std.testing.expect(!active());
}

test "a failing observer cannot survive its render scope" {
    const Failing = struct {
        fn text(_: *Sink, _: std.Io, _: []const u8) !void {
            return error.OutputFailed;
        }
        fn failFrame(_: *Sink, _: std.Io, _: std.mem.Allocator, _: Frame) !void {
            return error.OutputFailed;
        }
        fn render(sink: *Sink) !void {
            try sink.activate();
            defer sink.deactivate();
            try frame(std.testing.io, std.testing.allocator, .{
                .pixels = &.{ 0, 0, 0 },
                .width = 1,
                .height = 1,
                .step = 1,
                .total = 4,
                .index = 0,
            });
        }
    };
    var sink = Sink{ .on_text = Failing.text, .on_frame = Failing.failFrame };
    try std.testing.expectError(error.OutputFailed, Failing.render(&sink));
    try std.testing.expect(!active());
    try sink.activate();
    sink.deactivate();
}
