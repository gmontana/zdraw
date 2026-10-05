//! Precision policy for resident modulated transformer stacks.

const std = @import("std");

const gmode = @import("gemm_mode.zig");
const params = @import("mblock_chain_param.zig");
const weights = @import("mblock_chain_weight.zig");
const zpack_kinds = @import("zpack_kinds.zig");

pub const Policy = enum {
    inherit,
    allExact,
    allHalf,
    attnHalf,
    ffnHalf,
    gateupHalf,
    downHalf,
    measuredHalf,
    downW8,
    measuredDownW8Last8,
};

pub const Select = struct {
    from: usize = 0,
    to: usize = 0,
    last: usize = 0,
};

pub fn fromEnv() Policy {
    const raw = std.c.getenv("ZDRAW_STACK_GEMM") orelse return parse(null);
    return parse(std.mem.span(raw));
}

pub fn parse(raw: ?[]const u8) Policy {
    const value = raw orelse return .inherit;
    if (std.mem.eql(u8, value, "exact")) return .allExact;
    if (std.mem.eql(u8, value, "half")) return .allHalf;
    if (std.mem.eql(u8, value, "attn-half")) return .attnHalf;
    if (std.mem.eql(u8, value, "ffn-half")) return .ffnHalf;
    if (std.mem.eql(u8, value, "gateup-half")) return .gateupHalf;
    if (std.mem.eql(u8, value, "down-half")) return .downHalf;
    if (std.mem.eql(u8, value, "measured-half")) return .measuredHalf;
    if (std.mem.eql(u8, value, "quality-half")) return .measuredHalf;
    if (std.mem.eql(u8, value, "down-w8")) return .downW8;
    if (std.mem.eql(u8, value, "measured-down-w8-last8")) return .measuredDownW8Last8;
    return .inherit;
}

pub fn selectFromEnv() Select {
    return .{
        .from = usizeEnv("ZDRAW_STACK_HALF_FROM", 0),
        .to = usizeEnv("ZDRAW_STACK_HALF_TO", 0),
        .last = usizeEnv("ZDRAW_STACK_HALF_LAST", 0),
    };
}

pub fn modes(
    base: gmode.Mode,
    policy: Policy,
    select: Select,
    index: usize,
    count: usize,
) params.Modes {
    if (!useLayer(select, index, count)) return params.uniform(base);
    var out = params.uniform(base);
    switch (policy) {
        .inherit => {},
        .allExact => out = params.uniform(.exact),
        .allHalf => out = params.uniform(.half),
        .attnHalf => {
            out.q = .half;
            out.k = .half;
            out.v = .half;
            out.proj = .half;
        },
        .ffnHalf => {
            out.ffn_gate = .half;
            out.ffn_up = .half;
            out.ffn_down = .half;
        },
        .gateupHalf => {
            out.ffn_gate = .half;
            out.ffn_up = .half;
        },
        .downHalf => out.ffn_down = .half,
        .measuredHalf => out = measuredModes(base, index, count),
        .downW8 => out.ffn_down = .w8,
        .measuredDownW8Last8 => out = measuredW8Modes(base, index, count),
    }
    return out;
}

pub fn needsHalf(policy: Policy) bool {
    return switch (policy) {
        .allHalf,
        .attnHalf,
        .ffnHalf,
        .gateupHalf,
        .downHalf,
        .measuredHalf,
        .measuredDownW8Last8,
        => true,
        .inherit,
        .allExact,
        .downW8,
        => false,
    };
}

pub fn needsW8(policy: Policy) bool {
    return switch (policy) {
        .downW8,
        .measuredDownW8Last8,
        => true,
        else => false,
    };
}

pub fn packedModes(modes_in: params.Modes) weights.Packed {
    return .{
        .q = modes_in.q == .w8,
        .k = modes_in.k == .w8,
        .v = modes_in.v == .w8,
        .proj = modes_in.proj == .w8,
        .ffn_gate = modes_in.ffn_gate == .w8,
        .ffn_up = modes_in.ffn_up == .w8,
        .ffn_down = modes_in.ffn_down == .w8,
    };
}

fn measuredModes(base: gmode.Mode, index: usize, count: usize) params.Modes {
    var out = params.uniform(base);
    out.q = .half;
    out.k = .half;
    out.v = .half;
    out.proj = .half;
    if (index >= measuredFfnFrom(count)) {
        const w6 = w6Kinds();
        out.ffn_gate = if (w6.gate) .w6 else .half;
        out.ffn_up = if (w6.up) .w6 else .half;
        out.ffn_down = if (w6.down) .w6 else .half;
        if (w6.qkv) {
            out.q = .w6;
            out.k = .w6;
            out.v = .w6;
        }
        if (w6.proj) out.proj = .w6;
    } else {
        // Exact region: pinned to .exact explicitly (never inherits base —
        // the production base is half). ZDRAW_STACK_EXACT_OPS selects which
        // of gate/up/down stay exact; the others go half.
        out.ffn_gate = if (exactOp("gate")) .exact else .half;
        out.ffn_up = if (exactOp("up")) .exact else .half;
        out.ffn_down = if (exactOp("down")) .exact else .half;
    }
    applyPins(&out, pinsFromEnv(), index);
    return out;
}

// Per-op-class exact prefixes on the measured stack: layers below the bound
// run that op class exact (never W16-substituted). The Q2 sensitivity search
// explores these three axes; whatever it certifies ships as profile pins.
const Pins = struct {
    attn_below: usize = 0,
    gateup_below: usize = 0,
    down_below: usize = 0,
};

fn pinsFromEnv() Pins {
    return .{
        .attn_below = usizeEnv("ZDRAW_STACK_EXACT_ATTN_BELOW", 0),
        .gateup_below = usizeEnv("ZDRAW_STACK_EXACT_GATEUP_BELOW", 0),
        .down_below = usizeEnv("ZDRAW_STACK_EXACT_DOWN_BELOW", 0),
    };
}

fn applyPins(out: *params.Modes, pins: Pins, index: usize) void {
    if (index < pins.attn_below) {
        out.q = .exact;
        out.k = .exact;
        out.v = .exact;
        out.proj = .exact;
    }
    if (index < pins.gateup_below) {
        out.ffn_gate = .exact;
        out.ffn_up = .exact;
    }
    if (index < pins.down_below) out.ffn_down = .exact;
}

fn exactOp(name: []const u8) bool {
    // Per-op bisect 2026-06-06: layer-0 ffn_down alone carries quality
    // (down-only 66.42 dB; any subset without it collapses to 12.96).
    const raw = std.c.getenv("ZDRAW_STACK_EXACT_OPS") orelse
        return std.mem.eql(u8, name, "down");
    return std.mem.indexOf(u8, std.mem.span(raw), name) != null;
}

fn measuredW8Modes(base: gmode.Mode, index: usize, count: usize) params.Modes {
    var out = measuredModes(base, index, count);
    const last = usizeEnv("ZDRAW_STACK_W8_LAST", 8);
    if (last > 0 and index + last >= count) applyW8(&out, w8Kinds());
    return out;
}

fn applyW8(out: *params.Modes, kinds: zpack_kinds.Set) void {
    if (kinds.q) out.q = .w8;
    if (kinds.k) out.k = .w8;
    if (kinds.v) out.v = .w8;
    if (kinds.proj) out.proj = .w8;
    if (kinds.ffn_gate) out.ffn_gate = .w8;
    if (kinds.ffn_up) out.ffn_up = .w8;
    if (kinds.ffn_down) out.ffn_down = .w8;
}

const W6Kinds = struct {
    gate: bool = false,
    up: bool = false,
    down: bool = false,
    qkv: bool = false,
    proj: bool = false,
};

fn w6Kinds() W6Kinds {
    const raw = std.c.getenv("ZDRAW_STACK_W6_KINDS") orelse return .{};
    const text = std.mem.span(raw);
    return .{
        .gate = std.mem.indexOf(u8, text, "gate") != null,
        .up = std.mem.indexOf(u8, text, "up") != null,
        .down = std.mem.indexOf(u8, text, "down") != null,
        .qkv = std.mem.indexOf(u8, text, "qkv") != null,
        .proj = std.mem.indexOf(u8, text, "proj") != null,
    };
}

fn w8Kinds() zpack_kinds.Set {
    return zpack_kinds.env("ZDRAW_STACK_W8_KINDS", zpack_kinds.Set.downOnly());
}

fn measuredFfnFrom(count: usize) usize {
    // Boundary sweep 2026-06-06: only layer 0's FFN is quality-critical
    // (ffn-from 1..4 all score 64-69 dB; 0 collapses to 13 dB).
    _ = count;
    return usizeEnv("ZDRAW_STACK_MEASURED_FFN_FROM", 1);
}

fn useLayer(select: Select, index: usize, count: usize) bool {
    if (select.last > 0) return index + select.last >= count;
    if (index < select.from) return false;
    return select.to == 0 or index < select.to;
}

fn usizeEnv(comptime name: [*:0]const u8, fallback: usize) usize {
    const raw = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(usize, std.mem.span(raw), 10) catch fallback;
}

pub fn allOff(modes_in: params.Modes) bool {
    return modes_in.q == .off and modes_in.k == .off and modes_in.v == .off and
        modes_in.proj == .off and modes_in.ffn_gate == .off and
        modes_in.ffn_up == .off and modes_in.ffn_down == .off;
}

test "stack policy inherits unless an explicit known policy is selected" {
    try std.testing.expectEqual(Policy.inherit, parse(null));
    try std.testing.expectEqual(Policy.inherit, parse(""));
    try std.testing.expectEqual(Policy.inherit, parse("unknown"));
    try std.testing.expectEqual(Policy.allExact, parse("exact"));
    try std.testing.expectEqual(Policy.allHalf, parse("half"));
    try std.testing.expectEqual(Policy.measuredHalf, parse("measured-half"));
    try std.testing.expectEqual(Policy.measuredHalf, parse("quality-half"));
    try std.testing.expectEqual(Policy.downW8, parse("down-w8"));
    try std.testing.expectEqual(
        Policy.measuredDownW8Last8,
        parse("measured-down-w8-last8"),
    );
}

test "exact stack policy overrides a lossy base mode" {
    const selected = modes(
        .half,
        parse("exact"),
        .{},
        0,
        30,
    );
    try std.testing.expectEqual(params.uniform(.exact), selected);
}

test "inherited stack policy preserves the resolved base mode" {
    try std.testing.expectEqual(
        params.uniform(.exact),
        modes(.exact, parse(null), .{}, 0, 30),
    );
    try std.testing.expectEqual(
        params.uniform(.half),
        modes(.half, parse(null), .{}, 0, 30),
    );
}

test "exact prefix pins override the measured half assignments" {
    var m = measuredModes(.half, 1, 30);
    applyPins(&m, .{ .attn_below = 2, .down_below = 2 }, 1);
    try std.testing.expectEqual(gmode.Mode.exact, m.q);
    try std.testing.expectEqual(gmode.Mode.exact, m.proj);
    try std.testing.expectEqual(gmode.Mode.half, m.ffn_gate);
    try std.testing.expectEqual(gmode.Mode.exact, m.ffn_down);
    var out = measuredModes(.half, 5, 30);
    applyPins(&out, .{ .attn_below = 2, .down_below = 2 }, 5);
    try std.testing.expectEqual(gmode.Mode.half, out.q);
    try std.testing.expectEqual(gmode.Mode.half, out.ffn_down);
}
