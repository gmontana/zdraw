//! Small text helpers for the interactive session.

const std = @import("std");

const env_reader = @import("../runtime/env.zig");

const cyan = "\x1b[36m";
const sgr_reset = "\x1b[0m";

pub const Output = @import("session_output.zig").Output;

pub fn help(out: *Output, allocator: std.mem.Allocator) !void {
    try out.write(allocator,
        \\prompt text       append to the prompt and render
        \\new <prompt>      replace the prompt and render
        \\prompt / clear    show or clear the current prompt
        \\reroll            render the current prompt with a new seed
        \\undo              restore the previous prompt and render
        \\model <name>      switch between downloaded models
        \\steps <n>         set a positive denoise step count
        \\size WxH          set the next image size
        \\seed <n>          set the base seed (image index is added)
        \\save <path>       copy the last image
        \\stats on/off      show detailed memory statistics
        \\help              show this help
        \\quit / exit       leave the session (Ctrl-D also exits)
        \\
        \\Tab completes commands; Up/Down recall history; Ctrl-C clears input.
        \\Preview mode is only a smoke test and runs no model.
        \\
    );
}

pub fn merge(
    allocator: std.mem.Allocator,
    old: []const u8,
    text: []const u8,
    reset: bool,
) ![]u8 {
    if (reset or old.len == 0) return allocator.dupe(u8, text);
    return std.fmt.allocPrint(allocator, "{s}; {s}", .{ old, text });
}

pub fn saveTarget(text: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, text, "save ")) return null;
    const path = clean(text[5..]);
    return if (path.len == 0) null else path;
}

pub fn isQuit(text: []const u8) bool {
    return std.mem.eql(u8, text, "quit") or
        std.mem.eql(u8, text, "exit") or
        std.mem.eql(u8, text, ":q") or
        std.mem.eql(u8, text, "/quit") or
        std.mem.eql(u8, text, "/exit");
}

pub fn showPath(
    out: *Output,
    allocator: std.mem.Allocator,
    prefix: []const u8,
    path: []const u8,
) !void {
    const text = try std.fmt.allocPrint(allocator, "{s}{s}\n", .{ prefix, path });
    defer allocator.free(text);
    try out.write(allocator, text);
}

pub fn renderHeader(
    out: *Output,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    label: []const u8,
    width: u32,
    height: u32,
    steps: u32,
) !void {
    const text = if (colorEnabled(env))
        try std.fmt.allocPrint(allocator, "\n{s}{s}{s}  {d}x{d} | {d} steps\n", .{
            cyan,
            label,
            sgr_reset,
            width,
            height,
            steps,
        })
    else
        try std.fmt.allocPrint(
            allocator,
            "\n{s}  {d}x{d} | {d} steps\n",
            .{ label, width, height, steps },
        );
    defer allocator.free(text);
    try out.write(allocator, text);
}

pub fn writeText(out: *Output, allocator: std.mem.Allocator, text: []const u8) !void {
    try out.write(allocator, text);
}

pub fn writeIo(io: std.Io, text: []const u8) !void {
    try std.Io.File.stdout().writeStreamingAll(io, text);
}

pub fn colorEnabled(env: *const std.process.Environ.Map) bool {
    if (env.contains("NO_COLOR")) return false;
    if (env.get("CLICOLOR")) |value| {
        if (std.mem.eql(u8, value, "0")) return false;
    }
    if (env.get("TERM")) |value| {
        if (std.mem.eql(u8, value, "dumb")) return false;
    }
    return true;
}

pub fn clean(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, " \t\r\n");
}

test "merge appends refinements" {
    const text = try merge(std.testing.allocator, "a red boat", "stormy sky", false);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("a red boat; stormy sky", text);
}

test "save target parses path" {
    try std.testing.expectEqualStrings("final.png", saveTarget("save final.png").?);
    try std.testing.expect(saveTarget("save   ") == null);
}

/// Read a boolean env flag: unset -> default; set to "" or "0" -> false;
/// anything else -> true. One getenv call, no byte-indexing pitfalls.
pub fn envFlag(name: [*:0]const u8, default: bool) bool {
    return env_reader.flag(name, default);
}
