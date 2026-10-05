//! Shared parser for packed-weight tensor-kind sets.

const std = @import("std");

const zpack_file = @import("zpack_file.zig");

pub const Set = struct {
    q: bool = false,
    k: bool = false,
    v: bool = false,
    proj: bool = false,
    ffn_gate: bool = false,
    ffn_up: bool = false,
    ffn_down: bool = false,

    pub fn downOnly() Set {
        return .{ .ffn_down = true };
    }

    pub fn count(self: Set) usize {
        var total: usize = 0;
        for (order) |kind| {
            if (self.has(kind)) total += 1;
        }
        return total;
    }

    pub fn has(self: Set, kind: zpack_file.Kind) bool {
        return switch (kind) {
            .q => self.q,
            .k => self.k,
            .v => self.v,
            .proj => self.proj,
            .ffn_gate => self.ffn_gate,
            .ffn_up => self.ffn_up,
            .ffn_down => self.ffn_down,
            .ffn_gateup => false, // derived fused entry, never policy-selected
            .flux2_weight, .flux2_raw, .flux2_norm => false, // FLUX.2 has its own slot map
        };
    }

    pub fn add(self: *Set, kind: zpack_file.Kind) void {
        switch (kind) {
            .q => self.q = true,
            .k => self.k = true,
            .v => self.v = true,
            .proj => self.proj = true,
            .ffn_gate => self.ffn_gate = true,
            .ffn_up => self.ffn_up = true,
            .ffn_down => self.ffn_down = true,
            .ffn_gateup => {}, // derived fused entry, never policy-selected
            .flux2_weight, .flux2_raw, .flux2_norm => {}, // FLUX.2 has its own slot map
        }
    }
};

pub const order = [_]zpack_file.Kind{
    .q,
    .k,
    .v,
    .proj,
    .ffn_gate,
    .ffn_up,
    .ffn_down,
};

pub fn parse(text: []const u8) !Set {
    var out = Set{};
    var parts = std.mem.splitScalar(u8, text, ',');
    while (parts.next()) |raw| try addToken(&out, std.mem.trim(u8, raw, " \t\r\n"));
    return out;
}

pub fn env(name: [*:0]const u8, fallback: Set) Set {
    const raw = std.c.getenv(name) orelse return fallback;
    return parse(std.mem.span(raw)) catch fallback;
}

fn addToken(out: *Set, token: []const u8) !void {
    if (token.len == 0) return;
    if (std.mem.eql(u8, token, "none")) return;
    if (std.mem.eql(u8, token, "all")) return addAll(out);
    if (std.mem.eql(u8, token, "attn")) return addAttn(out);
    if (std.mem.eql(u8, token, "ffn")) return addFfn(out);
    out.add(try oneKind(token));
}

fn addAll(out: *Set) void {
    for (order) |kind| out.add(kind);
}

fn addAttn(out: *Set) void {
    out.add(.q);
    out.add(.k);
    out.add(.v);
    out.add(.proj);
}

fn addFfn(out: *Set) void {
    out.add(.ffn_gate);
    out.add(.ffn_up);
    out.add(.ffn_down);
}

fn oneKind(token: []const u8) !zpack_file.Kind {
    if (std.mem.eql(u8, token, "q")) return .q;
    if (std.mem.eql(u8, token, "k")) return .k;
    if (std.mem.eql(u8, token, "v")) return .v;
    if (std.mem.eql(u8, token, "proj")) return .proj;
    if (std.mem.eql(u8, token, "out")) return .proj;
    if (std.mem.eql(u8, token, "gate")) return .ffn_gate;
    if (std.mem.eql(u8, token, "up")) return .ffn_up;
    if (std.mem.eql(u8, token, "down")) return .ffn_down;
    if (std.mem.eql(u8, token, "ffn_gate")) return .ffn_gate;
    if (std.mem.eql(u8, token, "ffn_up")) return .ffn_up;
    if (std.mem.eql(u8, token, "ffn_down")) return .ffn_down;
    return error.BadKind;
}

test "parse packed kind sets" {
    const got = try parse("attn,gate,down");
    try std.testing.expect(got.q and got.k and got.v and got.proj);
    try std.testing.expect(got.ffn_gate and got.ffn_down);
    try std.testing.expect(!got.ffn_up);
    try std.testing.expectEqual(@as(usize, 6), got.count());
}

test "default down-only kind set" {
    const got = Set.downOnly();
    try std.testing.expect(got.ffn_down);
    try std.testing.expectEqual(@as(usize, 1), got.count());
}
