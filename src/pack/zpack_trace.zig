//! Read-the-truth counters for packed-weight resolution.
//!
//! Every DiT linear in the resident chain resolves to either a packed sidecar
//! entry (W6/W16) or falls back to the original safetensor view. Under
//! `ZDRAW_MEMTRACE` these counters record which, per weight-kind, so the report
//! can prove whether (e.g.) proj or the attention weights page in the 23 GB
//! original transformer instead of being served from the small sidecar.
//!
//! The chain build is single-threaded, but the counters are plain globals only
//! touched on that path; keep them simple. Disabled paths never call record().

const std = @import("std");

const zpack_file = @import("zpack_file.zig");
const metrics = @import("../metal/metrics.zig");

const Kind = zpack_file.Kind;

const kind_count = std.meta.fields(Kind).len; // contiguous from 1, see zpack_file.Kind

var packed_counts = [_]u64{0} ** kind_count;
var fallback_counts = [_]u64{0} ** kind_count;

fn index(kind: Kind) usize {
    return @intFromEnum(kind) - 1; // enum starts at 1
}

pub fn recordPacked(kind: Kind) void {
    if (!metrics.memtraceEnabled()) return;
    packed_counts[index(kind)] += 1;
}

pub fn recordFallback(kind: Kind) void {
    if (!metrics.memtraceEnabled()) return;
    fallback_counts[index(kind)] += 1;
}

pub fn reset() void {
    packed_counts = [_]u64{0} ** kind_count;
    fallback_counts = [_]u64{0} ** kind_count;
}

fn name(kind: Kind) []const u8 {
    return switch (kind) {
        .ffn_down => "down",
        .ffn_gate => "gate",
        .ffn_up => "up",
        .q => "q",
        .k => "k",
        .v => "v",
        .proj => "proj",
        .ffn_gateup => "gateup",
        .flux2_weight => "flux2",
        .flux2_raw => "flux2_raw",
        .flux2_norm => "flux2_norm",
    };
}

// Print one `ZPACK <kind> packed=<n> fallback=<n>` line per kind that saw any
// resolution. Uses std.debug.print so it needs no io/allocator and matches the
// MEMTRACE lines. No-op unless ZDRAW_MEMTRACE is set.
pub fn report() void {
    if (!metrics.memtraceEnabled()) return;
    inline for (.{ Kind.q, .k, .v, .proj, .ffn_gate, .ffn_up, .ffn_down, .ffn_gateup, .flux2_weight }) |kind| {
        const p = packed_counts[index(kind)];
        const f = fallback_counts[index(kind)];
        if (p != 0 or f != 0) {
            std.debug.print("ZPACK {s} packed={d} fallback={d}\n", .{ name(kind), p, f });
        }
    }
    // Counters cover one report window (per generation in sessions), not the
    // process lifetime — cumulative counts mislead in --repeat runs.
    reset();
}

test "records packed and fallback per kind when enabled" {
    // memtraceEnabled() gates recording; the test forces it on via the env.
    reset();
    if (!metrics.memtraceEnabled()) return; // only meaningful with the gate set
    recordPacked(.proj);
    recordPacked(.proj);
    recordFallback(.proj);
    try std.testing.expectEqual(@as(u64, 2), packed_counts[index(.proj)]);
    try std.testing.expectEqual(@as(u64, 1), fallback_counts[index(.proj)]);
}
