//! Dev-only metrics for the benchmark harness.
//!
//! Records named wall-clock samples (nanoseconds) into a fixed static table and
//! reads the Metal/host counters from the C bridge. No allocation in the hot
//! path, single-threaded. `record` is a cheap no-op until a stage names itself,
//! so wiring it into the normal run costs nothing.

const std = @import("std");

const c = @import("metal_c.zig");
const env = @import("../runtime/env.zig");
const metric_counts = @import("metric_counts.zig");
const stack_profile = @import("../runtime/stack_profile.zig");

pub const max_timing_labels = 48;
const max_labels = max_timing_labels;
const max_samples = 64;

const Label = struct {
    name: []const u8 = "",
    samples: [max_samples]u64 = [_]u64{0} ** max_samples,
    count: usize = 0,
};

var labels = [_]Label{.{}} ** max_labels;
var label_count: usize = 0;

pub const Counters = metric_counts.Counters;

pub const Memory = struct {
    peak_rss_bytes: u64,
    phys_footprint_bytes: u64,
    peak_gpu_live_bytes: u64,
    weight_bytes: u64,
    /// Resident pages of the no-copy weight mappings (mincore), which neither
    /// RSS nor phys_footprint count; total = phys_footprint + this.
    mapped_resident_bytes: u64,
    /// Peaks over the phase-boundary samples (memtrace samples even when it
    /// does not print): the honest "what the host had to hold" numbers.
    mapped_resident_peak_bytes: u64,
    total_peak_bytes: u64,
};

pub const TimingSummary = struct {
    stage: []const u8,
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
    samples: u32,
};

pub fn reset() void {
    labels = [_]Label{.{}} ** max_labels;
    label_count = 0;
    metric_counts.reset();
    c.zdraw_metal_metrics_reset();
}

pub fn record(name: []const u8, ns: u64) void {
    const slot = find(name) orelse return;
    if (slot.count < max_samples) {
        slot.samples[slot.count] = ns;
        slot.count += 1;
    }
}

pub fn now() u64 {
    return c.zdraw_now_ns();
}

// Record the span since `since` under `name` and return the new lap start.
pub fn lap(name: []const u8, since: u64) u64 {
    const at = now();
    record(name, at - since);
    return at;
}

pub fn snapshot() Counters {
    return metric_counts.snapshot();
}

// ZDRAW_MEMTRACE prints `MEMTRACE <stage> <GB>` from the honest phys_footprint
// number at each pipeline boundary, so the per-stage memory growth (encoder vs
// DiT vs activations vs the VAE-decode spike) is read from the OS instead of
// guessed. Cheap: one task_info call plus a stderr line; gated OFF by default.
var memtrace_cached: ?bool = null;

pub fn memtraceEnabled() bool {
    if (memtrace_cached) |value| return value;
    const value = env.flag("ZDRAW_MEMTRACE", false);
    memtrace_cached = value;
    return value;
}

// No `io`/`allocator` so it can be called from any pipeline depth (the VAE
// decoder has neither). std.debug.print locks and writes stderr directly.
pub fn memtrace(stage: []const u8) void {
    // Always sample (a strided estimate for the card's peaks); the exact
    // walk below runs only when printing.
    if (!memtraceEnabled()) {
        c.zdraw_proc_sample_memory();
        return;
    }
    // Field 3 stays phys (GB) for back-compat; live/peak are MTLBuffer temp
    // bytes and weights is mmap'd-weight bytes wrapped, so a per-step ramp can
    // be split into real residency vs temp leak vs weight-wrapper accumulation.
    // mapped_res is the resident part of those wrapped mappings (mincore) and
    // total = phys + mapped_res, the number a device actually has to hold.
    const phys = c.zdraw_proc_phys_footprint_bytes();
    const mapped = c.zdraw_proc_mapped_resident_bytes();
    const fmt = "MEMTRACE {s} {d:.3} live={d:.3} peak={d:.3} weights={d:.3}" ++
        " mapped_res={d:.3} total={d:.3}\n";
    std.debug.print(fmt, .{
        stage,
        gb(phys),
        gb(c.zdraw_metal_live_bytes()),
        gb(c.zdraw_metal_peak_bytes()),
        gb(c.zdraw_metal_weight_bytes()),
        gb(mapped),
        gb(phys + mapped),
    });
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

pub fn recordMetal(name: []const u8, before: Counters) void {
    metric_counts.record(name, before);
}

pub fn memory() Memory {
    return .{
        .peak_rss_bytes = c.zdraw_proc_peak_rss_bytes(),
        .phys_footprint_bytes = c.zdraw_proc_phys_footprint_bytes(),
        .peak_gpu_live_bytes = c.zdraw_metal_peak_bytes(),
        .weight_bytes = c.zdraw_metal_weight_bytes(),
        .mapped_resident_bytes = c.zdraw_proc_mapped_resident_bytes(),
        .mapped_resident_peak_bytes = c.zdraw_proc_mapped_resident_peak_bytes(),
        .total_peak_bytes = c.zdraw_proc_total_peak_bytes(),
    };
}

fn find(name: []const u8) ?*Label {
    for (labels[0..label_count]) |*l| {
        if (std.mem.eql(u8, l.name, name)) return l;
    }
    if (label_count == max_labels) return null;
    const l = &labels[label_count];
    l.* = .{ .name = name };
    label_count += 1;
    return l;
}

const Stat = struct { med: u64, min: u64, max: u64 };

fn stats(l: *const Label) Stat {
    if (l.count == 0) return .{ .med = 0, .min = 0, .max = 0 };
    var sorted = [_]u64{0} ** max_samples;
    @memcpy(sorted[0..l.count], l.samples[0..l.count]);
    std.mem.sort(u64, sorted[0..l.count], {}, std.sort.asc(u64));
    return .{ .med = sorted[l.count / 2], .min = sorted[0], .max = sorted[l.count - 1] };
}

/// Every recorded sample of one label, in order (empty when unknown).
pub fn samplesOf(name: []const u8) []const u64 {
    for (labels[0..label_count]) |*l| {
        if (std.mem.eql(u8, l.name, name)) return l.samples[0..l.count];
    }
    return &.{};
}

pub fn timingSummary(index: usize) ?TimingSummary {
    if (index >= label_count) return null;
    const label = &labels[index];
    const summary = stats(label);
    return .{
        .stage = label.name,
        .median_ns = summary.med,
        .min_ns = summary.min,
        .max_ns = summary.max,
        .samples = @intCast(label.count),
    };
}

pub fn report(io: std.Io, allocator: std.mem.Allocator) !void {
    try reportTiming(io, allocator);
    try reportMetal(io, allocator);
    try stack_profile.report(io, allocator);
    try reportSystem(io, allocator);
}

pub fn appendMarkdown(allocator: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    try out.appendSlice(allocator, "## Timing\n\n");
    try out.appendSlice(allocator, "| stage | median ms | min ms | max ms | n |\n");
    try out.appendSlice(allocator, "| --- | ---: | ---: | ---: | ---: |\n");
    for (labels[0..label_count]) |*l| try appendTiming(allocator, out, l);
    try out.appendSlice(allocator, "\n## Memory And Round Trips\n\n");
    try appendMegabytes(allocator, out, "peak RSS", c.zdraw_proc_peak_rss_bytes());
    try appendMegabytes(allocator, out, "phys footprint", c.zdraw_proc_phys_footprint_bytes());
    try appendMegabytes(allocator, out, "peak GPU live", c.zdraw_metal_peak_bytes());
    try appendMegabytes(allocator, out, "weights wrapped", c.zdraw_metal_weight_bytes());
    try appendCounter(allocator, out, "Metal dispatches", c.zdraw_metal_dispatch_count());
    try appendCounter(allocator, out, "command buffers", c.zdraw_metal_command_count());
    try appendCounter(allocator, out, "completion waits", c.zdraw_metal_wait_count());
    try appendCounter(allocator, out, "GEMM dispatches", c.zdraw_metal_gemm_count());
    try appendCounter(allocator, out, "exact GEMM", c.zdraw_metal_gemm_exact_count());
    try appendCounter(allocator, out, "half GEMM", c.zdraw_metal_gemm_half_count());
    try appendCounter(allocator, out, "W8 GEMM", c.zdraw_metal_gemm_w8_count());
    try appendCounter(allocator, out, "W6 GEMM", c.zdraw_metal_gemm_w6_count());
    try appendCounter(allocator, out, "W4 GEMM", c.zdraw_metal_gemm_w4_count());
    try appendCounter(allocator, out, "W2 GEMM", c.zdraw_metal_gemm_w2_count());
    try appendCounter(allocator, out, "MPS GEMM", c.zdraw_metal_gemm_mps_count());
    try appendCounter(allocator, out, "MPS fallback", c.zdraw_metal_gemm_mps_fallback_count());
    try appendCounter(allocator, out, "steel fallback", c.zdraw_metal_attn_steel_fallback_count());
    try appendCounter(allocator, out, "MPP fallback", c.zdraw_metal_gemm_mpp_fallback_count());
    try appendCounter(allocator, out, "GPU readbacks", c.zdraw_metal_readback_count());
    try appendCounter(allocator, out, "GPU busy ms", c.zdraw_metal_gpu_ns() / 1_000_000);
}

fn reportTiming(io: std.Io, allocator: std.mem.Allocator) !void {
    try stderr(io, "--- timing (ms): median / min / max ---\n");
    for (labels[0..label_count]) |*l| {
        const s = stats(l);
        const text = try std.fmt.allocPrint(
            allocator,
            "  {s:<18}{d:>9.2}{d:>9.2}{d:>9.2}  (n={d})\n",
            .{ l.name, ms(s.med), ms(s.min), ms(s.max), l.count },
        );
        defer allocator.free(text);
        try stderr(io, text);
    }
}

fn reportMetal(io: std.Io, allocator: std.mem.Allocator) !void {
    try metric_counts.report(io, allocator);
}

fn reportSystem(io: std.Io, allocator: std.mem.Allocator) !void {
    try stderr(io, "--- memory / round-trips ---\n");
    try megabytes(io, allocator, "peak RSS        ", c.zdraw_proc_peak_rss_bytes());
    try megabytes(io, allocator, "phys footprint  ", c.zdraw_proc_phys_footprint_bytes());
    try megabytes(io, allocator, "peak GPU live   ", c.zdraw_metal_peak_bytes());
    try megabytes(io, allocator, "weights wrapped ", c.zdraw_metal_weight_bytes());
    try counter(io, allocator, "GPU busy ms     ", c.zdraw_metal_gpu_ns() / 1_000_000);
    try counter(io, allocator, "Metal dispatches", c.zdraw_metal_dispatch_count());
    try counter(io, allocator, "command buffers ", c.zdraw_metal_command_count());
    try counter(io, allocator, "completion waits", c.zdraw_metal_wait_count());
    try counter(io, allocator, "  of which GEMM ", c.zdraw_metal_gemm_count());
    try counter(io, allocator, "    exact GEMM  ", c.zdraw_metal_gemm_exact_count());
    try counter(io, allocator, "    half GEMM   ", c.zdraw_metal_gemm_half_count());
    try counter(io, allocator, "    W8 GEMM     ", c.zdraw_metal_gemm_w8_count());
    try counter(io, allocator, "    W6 GEMM     ", c.zdraw_metal_gemm_w6_count());
    try counter(io, allocator, "    W4 GEMM     ", c.zdraw_metal_gemm_w4_count());
    try counter(io, allocator, "    W2 GEMM     ", c.zdraw_metal_gemm_w2_count());
    try counter(io, allocator, "    MPS GEMM    ", c.zdraw_metal_gemm_mps_count());
    try counter(io, allocator, "    MPS fallback", c.zdraw_metal_gemm_mps_fallback_count());
    try counter(io, allocator, "    steel fallback", c.zdraw_metal_attn_steel_fallback_count());
    try counter(io, allocator, "    MPP fallback", c.zdraw_metal_gemm_mpp_fallback_count());
    try counter(io, allocator, "GPU readbacks   ", c.zdraw_metal_readback_count());
}

fn megabytes(io: std.Io, allocator: std.mem.Allocator, name: []const u8, bytes: u64) !void {
    const text = try std.fmt.allocPrint(allocator, "  {s} {d:>9.1} MB\n", .{ name, mb(bytes) });
    defer allocator.free(text);
    try stderr(io, text);
}

fn counter(io: std.Io, allocator: std.mem.Allocator, name: []const u8, value: u64) !void {
    const text = try std.fmt.allocPrint(allocator, "  {s} {d:>9}\n", .{ name, value });
    defer allocator.free(text);
    try stderr(io, text);
}

fn stderr(io: std.Io, text: []const u8) !void {
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

fn appendTiming(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    label: *const Label,
) !void {
    const s = stats(label);
    const text = try std.fmt.allocPrint(
        allocator,
        "| {s} | {d:.2} | {d:.2} | {d:.2} | {d} |\n",
        .{ label.name, ms(s.med), ms(s.min), ms(s.max), label.count },
    );
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendMegabytes(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    name: []const u8,
    bytes: u64,
) !void {
    const text = try std.fmt.allocPrint(allocator, "- {s}: {d:.1} MB\n", .{ name, mb(bytes) });
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendCounter(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    name: []const u8,
    value: u64,
) !void {
    const text = try std.fmt.allocPrint(allocator, "- {s}: {d}\n", .{ name, value });
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn mb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

test "records samples and computes order stats" {
    reset();
    record("a", 10);
    record("a", 30);
    record("b", 5);
    const a = find("a").?;
    try std.testing.expectEqual(@as(usize, 2), a.count);
    const s = stats(a);
    try std.testing.expectEqual(@as(u64, 30), s.max);
    try std.testing.expectEqual(@as(u64, 10), s.min);
    try std.testing.expectEqual(@as(usize, 2), label_count);
    const summary = timingSummary(0).?;
    try std.testing.expectEqualStrings("a", summary.stage);
    try std.testing.expectEqual(@as(u64, 30), summary.median_ns);
    try std.testing.expectEqual(@as(u32, 2), summary.samples);
    try std.testing.expect(timingSummary(2) == null);
}
