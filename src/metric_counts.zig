//! Per-stage Metal counter deltas for the benchmark report.

const std = @import("std");

const c = @import("metal_c.zig");

const max_labels = 24;
const max_samples = 64;

pub const Counters = struct {
    dispatch: u64 = 0,
    command: u64 = 0,
    readback: u64 = 0,
    gemm: u64 = 0,
    exact: u64 = 0,
    half: u64 = 0,
    w8: u64 = 0,
    w6: u64 = 0,
    w4: u64 = 0,
    w2: u64 = 0,
    mps: u64 = 0,
    mps_fallback: u64 = 0,
    steel_fallback: u64 = 0,
    mpp_fallback: u64 = 0,
    gpu_ns: u64 = 0,
};

const Label = struct {
    name: []const u8 = "",
    samples: [max_samples]Counters = [_]Counters{.{}} ** max_samples,
    count: usize = 0,
};

var labels = [_]Label{.{}} ** max_labels;
var label_count: usize = 0;

pub fn reset() void {
    labels = [_]Label{.{}} ** max_labels;
    label_count = 0;
}

pub fn snapshot() Counters {
    return .{
        .dispatch = c.zdraw_metal_dispatch_count(),
        .command = c.zdraw_metal_command_count(),
        .readback = c.zdraw_metal_readback_count(),
        .gemm = c.zdraw_metal_gemm_count(),
        .exact = c.zdraw_metal_gemm_exact_count(),
        .half = c.zdraw_metal_gemm_half_count(),
        .w8 = c.zdraw_metal_gemm_w8_count(),
        .w6 = c.zdraw_metal_gemm_w6_count(),
        .w4 = c.zdraw_metal_gemm_w4_count(),
        .w2 = c.zdraw_metal_gemm_w2_count(),
        .mps = c.zdraw_metal_gemm_mps_count(),
        .mps_fallback = c.zdraw_metal_gemm_mps_fallback_count(),
        .steel_fallback = c.zdraw_metal_attn_steel_fallback_count(),
        .mpp_fallback = c.zdraw_metal_gemm_mpp_fallback_count(),
        .gpu_ns = c.zdraw_metal_gpu_ns(),
    };
}

pub fn record(name: []const u8, before: Counters) void {
    const slot = find(name) orelse return;
    if (slot.count == max_samples) return;
    slot.samples[slot.count] = sub(snapshot(), before);
    slot.count += 1;
}

pub fn report(io: std.Io, allocator: std.mem.Allocator) !void {
    try stderr(
        io,
        "--- Metal by stage: dispatch / cmd / read / gemm / exact / half / " ++
            "W8 / W6 / W4 / W2 / MPS / fallback / GPU ms ---\n",
    );
    for (labels[0..label_count]) |*l| {
        const m = sum(l);
        if (m.dispatch == 0 and m.command == 0 and m.readback == 0) continue;
        const text = try std.fmt.allocPrint(
            allocator,
            "  {s:<18}{d:>6}{d:>6}{d:>6}{d:>6}{d:>7}{d:>7}{d:>6}{d:>6}" ++
                "{d:>6}{d:>6}{d:>6}{d:>9}{d:>9.1}\n",
            .{
                l.name,
                m.dispatch,
                m.command,
                m.readback,
                m.gemm,
                m.exact,
                m.half,
                m.w8,
                m.w6,
                m.w4,
                m.w2,
                m.mps,
                m.mps_fallback + m.steel_fallback + m.mpp_fallback,
                gpuMs(m),
            },
        );
        defer allocator.free(text);
        try stderr(io, text);
        if (l.count > 1 and m.gpu_ns > 0) {
            try stderr(io, "        gpu-ms per sample:");
            for (l.samples[0..l.count]) |s| {
                const t = try std.fmt.allocPrint(allocator, " {d:.1}", .{gpuMs(s)});
                defer allocator.free(t);
                try stderr(io, t);
            }
            try stderr(io, "\n");
        }
    }
}

fn gpuMs(m: Counters) f64 {
    return @as(f64, @floatFromInt(m.gpu_ns)) / 1_000_000.0;
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

fn sub(a: Counters, b: Counters) Counters {
    return .{
        .dispatch = a.dispatch - b.dispatch,
        .command = a.command - b.command,
        .readback = a.readback - b.readback,
        .gemm = a.gemm - b.gemm,
        .exact = a.exact - b.exact,
        .half = a.half - b.half,
        .w8 = a.w8 - b.w8,
        .w6 = a.w6 - b.w6,
        .w4 = a.w4 - b.w4,
        .w2 = a.w2 - b.w2,
        .mps = a.mps - b.mps,
        .mps_fallback = a.mps_fallback - b.mps_fallback,
        .steel_fallback = a.steel_fallback - b.steel_fallback,
        .mpp_fallback = a.mpp_fallback - b.mpp_fallback,
        .gpu_ns = a.gpu_ns - b.gpu_ns,
    };
}

fn sum(l: *const Label) Counters {
    var out = Counters{};
    for (l.samples[0..l.count]) |m| {
        out.dispatch += m.dispatch;
        out.command += m.command;
        out.readback += m.readback;
        out.gemm += m.gemm;
        out.exact += m.exact;
        out.half += m.half;
        out.w8 += m.w8;
        out.w6 += m.w6;
        out.w4 += m.w4;
        out.w2 += m.w2;
        out.mps += m.mps;
        out.mps_fallback += m.mps_fallback;
        out.steel_fallback += m.steel_fallback;
        out.mpp_fallback += m.mpp_fallback;
        out.gpu_ns += m.gpu_ns;
    }
    return out;
}

fn stderr(io: std.Io, text: []const u8) !void {
    try std.Io.File.stderr().writeStreamingAll(io, text);
}
