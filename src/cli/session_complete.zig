//! Line-aware completions for the interactive zdraw session.

const std = @import("std");
const stanza = @import("stanza");

// Candidate with the dimmed annotation the stanza menu shows next to it.
const Choice = struct { insert: []const u8, detail: []const u8 = "" };

const command_args = [_]Choice{
    .{ .insert = "help", .detail = "commands and settings" },
    .{ .insert = "clear", .detail = "empty the prompt" },
    .{ .insert = "model ", .detail = "switch model" },
    .{ .insert = "new ", .detail = "start a fresh prompt" },
    .{ .insert = "prompt", .detail = "show current prompt" },
    .{ .insert = "quit", .detail = "leave the session" },
    .{ .insert = "reroll", .detail = "same prompt, new seed" },
    .{ .insert = "save ", .detail = "copy last image to a path" },
    .{ .insert = "seed ", .detail = "set the seed" },
    .{ .insert = "size ", .detail = "set output resolution" },
    .{ .insert = "stats ", .detail = "timing readout on/off" },
    .{ .insert = "steps ", .detail = "set denoise steps" },
    .{ .insert = "undo", .detail = "back to previous prompt" },
};

const model_args = [_]Choice{
    .{ .insert = "z-image-turbo", .detail = "text-to-image" },
    .{ .insert = "flux2-klein-4b", .detail = "text-to-image" },
    .{ .insert = "flux2-klein-base-4b", .detail = "text-to-image, 50 steps with guidance" },
};
const stats_args = [_]Choice{ .{ .insert = "on" }, .{ .insert = "off" } };
const seed_args = [_]Choice{
    .{ .insert = "42" },
    .{ .insert = "1234" },
    .{ .insert = "2026" },
};
const size_args = [_]Choice{
    .{ .insert = "256x256", .detail = "fast preview" },
    .{ .insert = "512x512", .detail = "balanced" },
    .{ .insert = "768x768", .detail = "detail" },
    .{ .insert = "1024x1024", .detail = "high detail" },
    .{ .insert = "128x128", .detail = "thumbnail" },
};
const step_args = [_]Choice{
    .{ .insert = "1", .detail = "fastest draft" },
    .{ .insert = "2" },
    .{ .insert = "3" },
    .{ .insert = "4", .detail = "turbo default" },
    .{ .insert = "8", .detail = "extra refinement" },
    .{ .insert = "9", .detail = "max refinement" },
};

pub fn complete(
    _: ?*anyopaque,
    line: []const u8,
    cursor: usize,
    word: []const u8,
    out: *stanza.Completions,
) anyerror!void {
    const head = std.mem.trimStart(u8, line[0 .. cursor - word.len], " \t");
    if (std.mem.eql(u8, head, "model ")) return addMatches(model_args[0..], word, out);
    if (std.mem.eql(u8, head, "size ")) return addMatches(size_args[0..], word, out);
    if (std.mem.eql(u8, head, "steps ")) return addMatches(step_args[0..], word, out);
    if (std.mem.eql(u8, head, "seed ")) return addMatches(seed_args[0..], word, out);
    if (std.mem.eql(u8, head, "stats ")) return addMatches(stats_args[0..], word, out);
    if (isCommandHead(head)) return addMatches(command_args[0..], word, out);
}

pub fn paint(_: ?*anyopaque, line: []const u8, out: *stanza.Painter) anyerror!void {
    const split = std.mem.indexOfAny(u8, line, " \t") orelse line.len;
    const command = line[0..split];
    if (!isCommand(command)) return out.plain(line);
    try out.put(command, .{ .color = .cyan, .bold = true });
    if (split < line.len) {
        try out.put(line[split..], .{ .color = .gray });
    }
}

fn addMatches(
    values: []const Choice,
    text: []const u8,
    out: *stanza.Completions,
) !void {
    for (values) |value| {
        if (!std.mem.startsWith(u8, value.insert, text)) continue;
        if (value.detail.len == 0) {
            try out.add(value.insert);
        } else {
            try out.addDetail(value.insert, value.detail);
        }
    }
}

fn isCommandHead(head: []const u8) bool {
    return std.mem.trim(u8, head, " \t").len == 0;
}

fn isCommand(word: []const u8) bool {
    for (command_args) |command| {
        const clean = std.mem.trim(u8, command.insert, " ");
        if (std.mem.eql(u8, clean, word)) return true;
    }
    return false;
}

test "completion suggests size values with detail" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var out = stanza.Completions{ .arena = arena.allocator() };

    try complete(null, "size 2", "size 2".len, "2", &out);
    try std.testing.expect(out.items.items.len >= 1);
    try std.testing.expectEqualStrings("256x256", out.items.items[0].insert);
    try std.testing.expectEqualStrings("fast preview", out.items.items[0].detail);
}

test "empty prompt offers commands" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var out = stanza.Completions{ .arena = arena.allocator() };

    try complete(null, "", 0, "", &out);
    try std.testing.expect(out.items.items.len >= 1);
    try std.testing.expectEqualStrings("help", out.items.items[0].insert);
    try std.testing.expectEqualStrings("commands and settings", out.items.items[0].detail);
}

test "model completion excludes research variants and prompt fragments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var out = stanza.Completions{ .arena = arena.allocator() };
    try complete(null, "model flux", 10, "flux", &out);
    try std.testing.expectEqual(@as(usize, 2), out.items.items.len);
    try std.testing.expectEqualStrings("flux2-klein-4b", out.items.items[0].insert);
    try std.testing.expectEqualStrings("flux2-klein-base-4b", out.items.items[1].insert);
    var prompt = stanza.Completions{ .arena = arena.allocator() };
    try complete(null, "a model flux", 12, "flux", &prompt);
    try std.testing.expectEqual(@as(usize, 0), prompt.items.items.len);
}
