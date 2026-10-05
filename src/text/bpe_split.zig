//! Small tokenizer pre-splitter matching the ASCII path of Qwen's regex.

const std = @import("std");

pub fn next(text: []const u8) []const u8 {
    if (text.len >= 2 and text[0] == '\'') {
        if (contractionLen(text)) |n| return text[0..n];
    }
    if (letterStart(text)) |i| return letters(text, i);
    if (std.ascii.isDigit(text[0])) return text[0..1];
    if (punctStart(text)) |i| return punct(text, i);
    return spaces(text);
}

fn contractionLen(text: []const u8) ?usize {
    inline for ([_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }) |part| {
        if (text.len >= part.len and std.ascii.eqlIgnoreCase(text[0..part.len], part)) {
            return part.len;
        }
    }
    return null;
}

fn letterStart(text: []const u8) ?usize {
    if (isLetter(text[0])) return 0;
    if (text.len > 1 and isPrefix(text[0]) and isLetter(text[1])) return 1;
    return null;
}

fn letters(text: []const u8, first: usize) []const u8 {
    var end = first;
    while (end < text.len and isLetter(text[end])) : (end += 1) {}
    return text[0..end];
}

fn punctStart(text: []const u8) ?usize {
    if (isPunct(text[0])) return 0;
    if (text.len > 1 and text[0] == ' ' and isPunct(text[1])) return 1;
    return null;
}

fn punct(text: []const u8, first: usize) []const u8 {
    var end = first;
    while (end < text.len and isPunct(text[end])) : (end += 1) {}
    while (end < text.len and (text[end] == '\r' or text[end] == '\n')) : (end += 1) {}
    return text[0..end];
}

fn spaces(text: []const u8) []const u8 {
    var end: usize = 1;
    while (end < text.len and std.ascii.isWhitespace(text[end])) : (end += 1) {}
    return text[0..end];
}

fn isLetter(byte: u8) bool {
    return std.ascii.isAlphabetic(byte) or byte >= 0x80;
}

fn isPrefix(byte: u8) bool {
    return byte != '\r' and byte != '\n' and !std.ascii.isAlphanumeric(byte);
}

fn isPunct(byte: u8) bool {
    return !std.ascii.isWhitespace(byte) and !std.ascii.isAlphanumeric(byte) and byte < 0x80;
}

test "split common prompt pieces" {
    try std.testing.expectEqualStrings("a", next("a red"));
    try std.testing.expectEqualStrings(" red", next(" red boat"));
    try std.testing.expectEqualStrings("'s", next("'s bright"));
    try std.testing.expectEqualStrings("!\n", next("!\n"));
}
