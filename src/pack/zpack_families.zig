//! Shared parser for packed-weight block-family sets.

const std = @import("std");

const zpack_file = @import("zpack_file.zig");

pub const Set = struct {
    main: bool = false,
    noise: bool = false,
    context: bool = false,
    flux2: bool = false,
    text: bool = false,

    pub fn mainOnly() Set {
        return .{ .main = true };
    }

    pub fn count(self: Set) usize {
        var total: usize = 0;
        for (order) |family| {
            if (self.has(family)) total += 1;
        }
        return total;
    }

    pub fn has(self: Set, family: zpack_file.Family) bool {
        return switch (family) {
            .main => self.main,
            .noise => self.noise,
            .context => self.context,
            .flux2 => self.flux2,
            .text => self.text,
        };
    }

    pub fn add(self: *Set, family: zpack_file.Family) void {
        switch (family) {
            .main => self.main = true,
            .noise => self.noise = true,
            .context => self.context = true,
            .flux2 => self.flux2 = true,
            .text => self.text = true,
        }
    }
};

/// The block families a selector can name; the text pack is not one.
pub const order = [_]zpack_file.Family{ .main, .noise, .context, .flux2 };

pub fn parse(text: []const u8) !Set {
    var out = Set{};
    var parts = std.mem.splitScalar(u8, text, ',');
    while (parts.next()) |raw| try addToken(&out, std.mem.trim(u8, raw, " \t\r\n"));
    return out;
}

fn addToken(out: *Set, token: []const u8) !void {
    if (token.len == 0) return;
    if (std.mem.eql(u8, token, "all")) return addAll(out);
    if (std.mem.eql(u8, token, "refiners")) return addRefiners(out);
    out.add(try oneFamily(token));
}

fn addAll(out: *Set) void {
    for (order) |family| out.add(family);
}

fn addRefiners(out: *Set) void {
    out.add(.noise);
    out.add(.context);
}

fn oneFamily(token: []const u8) !zpack_file.Family {
    if (std.mem.eql(u8, token, "main")) return .main;
    if (std.mem.eql(u8, token, "layers")) return .main;
    if (std.mem.eql(u8, token, "noise")) return .noise;
    if (std.mem.eql(u8, token, "noise_refiner")) return .noise;
    if (std.mem.eql(u8, token, "context")) return .context;
    if (std.mem.eql(u8, token, "context_refiner")) return .context;
    if (std.mem.eql(u8, token, "flux2")) return .flux2;
    if (std.mem.eql(u8, token, "klein")) return .flux2;
    return error.BadFamily;
}

test "parse packed family sets" {
    const got = try parse("main,noise");
    try std.testing.expect(got.main and got.noise);
    try std.testing.expect(!got.context);
    try std.testing.expectEqual(@as(usize, 2), got.count());
}

test "parse flux2 family aliases" {
    const flux = try parse("flux2");
    try std.testing.expect(flux.flux2);
    try std.testing.expectEqual(@as(usize, 1), flux.count());

    const klein = try parse("klein");
    try std.testing.expect(klein.flux2);
    try std.testing.expectEqual(@as(usize, 1), klein.count());
}

test "all includes flux2" {
    const got = try parse("all");
    try std.testing.expect(got.main);
    try std.testing.expect(got.noise);
    try std.testing.expect(got.context);
    try std.testing.expect(got.flux2);
    try std.testing.expectEqual(@as(usize, 4), got.count());
}

test "default main-only family set" {
    const got = Set.mainOnly();
    try std.testing.expect(got.main);
    try std.testing.expectEqual(@as(usize, 1), got.count());
}
