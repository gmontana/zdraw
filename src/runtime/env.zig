//! Shared environment parsing for runtime/debug flags.
//!
//! Convention: unset means the caller-provided default; "0", "false", "no",
//! and the empty string mean false; everything else means true. Numeric readers
//! return their fallback on malformed input so debug knobs cannot crash a run.

const std = @import("std");

pub fn flag(name: [*:0]const u8, default: bool) bool {
    const raw = std.c.getenv(name) orelse return default;
    return truthy(std.mem.span(raw));
}

pub fn equals(name: [*:0]const u8, value: []const u8) bool {
    const raw = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), value);
}

pub fn usizeVar(name: [*:0]const u8, fallback: usize) usize {
    const raw = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(usize, std.mem.span(raw), 10) catch fallback;
}

fn truthy(value: []const u8) bool {
    if (value.len == 0) return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.ascii.eqlIgnoreCase(value, "false")) return false;
    if (std.ascii.eqlIgnoreCase(value, "no")) return false;
    if (std.ascii.eqlIgnoreCase(value, "off")) return false;
    return true;
}

test "flag parses common false spellings" {
    try std.testing.expect(!truthy(""));
    try std.testing.expect(!truthy("0"));
    try std.testing.expect(!truthy("false"));
    try std.testing.expect(!truthy("NO"));
    try std.testing.expect(!truthy("off"));
    try std.testing.expect(truthy("1"));
    try std.testing.expect(truthy("yes"));
}
