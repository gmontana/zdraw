//! A temporary terminal image region, owned by one synchronous CLI render.
//! No alternate screen, cursor hiding, terminal modes, or preview files.
const std = @import("std");
const image = @import("image.zig");
const sink_mod = @import("progress_sink.zig");
const terminal = @import("terminal.zig");
const protocol_mod = @import("terminal_image.zig");

const image_id = 2053407344;

pub const View = struct {
    sink: sink_mod.Sink = .{ .on_text = onText, .on_frame = onFrame },
    enabled: bool,
    watching: bool = false,
    occupied: u32 = 0,
    columns: u32 = 0,
    rows: u32 = 0,
    protocol: protocol_mod.Protocol = .ansi,

    pub fn init(io: std.Io, env: *const std.process.Environ.Map, show: bool) !View {
        var view = View{ .enabled = false };
        if (!show) return view;
        if (!try std.Io.File.stdout().isTty(io) or !try std.Io.File.stderr().isTty(io)) return view;
        if (env.get("TERM")) |term| if (std.mem.eql(u8, term, "dumb")) return view;
        var size: std.c.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
        if (std.c.ioctl(std.Io.File.stdout().handle, std.c.T.IOCGWINSZ, &size) != 0) return view;
        if (size.row < 12 or size.col < 40) return view;
        view.enabled = true;
        view.columns = @min(64, size.col - 4);
        view.rows = @min(28, size.row - 6);
        view.protocol = protocol_mod.resolve(.auto, env);
        return view;
    }

    /// Activate only after the View is at its final stack address.
    pub fn start(self: *View) !void {
        if (!self.enabled) return;
        try self.sink.activate();
        self.watching = true;
    }

    pub fn deinit(self: *View, io: std.Io) void {
        if (!self.watching) return;
        self.sink.deactivate();
        self.watching = false;
        self.clear(io) catch |err| std.log.warn("preview cleanup: {s}", .{@errorName(err)});
    }

    fn clear(self: *View, io: std.Io) !void {
        if (self.occupied == 0) return;
        if (self.protocol == .kitty) {
            try write(io, "\x1b_Ga=d,d=I,i=2053407344,q=2;\x1b\\");
        }
        var buffer: [48]u8 = @splat(0);
        const move = try std.fmt.bufPrint(&buffer, "\r\x1b[{d}A\x1b[J", .{self.occupied});
        try write(io, move);
        self.occupied = 0;
    }

    fn onText(sink: *sink_mod.Sink, io: std.Io, text: []const u8) !void {
        // SAFETY: this observer is embedded in the live View that activated it.
        const self: *View = @fieldParentPtr("sink", sink);
        // The frame owns the live region; leave it stable until the next step.
        if (self.occupied == 0) try std.Io.File.stderr().writeStreamingAll(io, text);
    }

    fn onFrame(
        sink: *sink_mod.Sink,
        io: std.Io,
        allocator: std.mem.Allocator,
        frame: sink_mod.Frame,
    ) !void {
        // SAFETY: callbacks are synchronous and cannot outlive the owning View.
        const self: *View = @fieldParentPtr("sink", sink);
        if (frame.index != 0) return;
        const columns = fitColumns(self.columns, self.rows, frame.width, frame.height);
        const rows = (frame.height * columns + frame.width * 2 - 1) / (frame.width * 2);
        const rendered = try self.render(allocator, frame, columns, rows);
        defer allocator.free(rendered);
        try self.clear(io);
        var buffer: [96]u8 = @splat(0);
        const label = try std.fmt.bufPrint(
            &buffer,
            "\r\x1b[K  preview  {d}/{d} (approximate)\n",
            .{ frame.step, frame.total },
        );
        try write(io, label);
        for (0..rows) |_| try write(io, "\n");
        self.occupied = rows + 1;
        const up = try std.fmt.bufPrint(&buffer, "\r\x1b[{d}A\x1b7", .{rows});
        var down_buffer: [32]u8 = @splat(0);
        const down = try std.fmt.bufPrint(&down_buffer, "\x1b8\x1b[{d}B\r", .{rows});
        try write(io, up);
        defer write(io, down) catch |err| std.log.warn("preview cursor: {s}", .{@errorName(err)});
        try write(io, rendered);
    }

    fn render(
        self: *View,
        allocator: std.mem.Allocator,
        frame: sink_mod.Frame,
        columns: u32,
        rows: u32,
    ) ![]u8 {
        if (self.protocol == .ansi) return terminal.renderAnsi(
            allocator,
            frame.pixels,
            frame.width,
            frame.height,
            .{ .max_width = columns },
        );
        const png = try image.encodePng(allocator, frame.pixels, frame.width, frame.height);
        defer allocator.free(png);
        return protocol_mod.render(allocator, self.protocol, png, .{
            .image_width = frame.width,
            .image_height = frame.height,
            .max_columns = columns,
            .viewport_rows = rows,
            .image_id = image_id,
        });
    }
};

fn write(io: std.Io, text: []const u8) !void {
    try std.Io.File.stdout().writeStreamingAll(io, text);
}

fn fitColumns(columns: u32, rows: u32, width: u32, height: u32) u32 {
    return @max(1, @min(columns, rows * 2 * width / height));
}

test "preview viewport fits portrait and landscape images" {
    try std.testing.expectEqual(@as(u32, 36), fitColumns(64, 18, 128, 128));
    try std.testing.expectEqual(@as(u32, 18), fitColumns(64, 18, 64, 128));
    try std.testing.expectEqual(@as(u32, 64), fitColumns(64, 18, 256, 128));
}
