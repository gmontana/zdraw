//! Report the diagnostic resident-stack kernel-family profile.

const std = @import("std");

const c = @import("stack_profile_c.zig");

pub fn report(io: std.Io, allocator: std.mem.Allocator) !void {
    const n = c.zdraw_metal_stack_profile_count();
    if (!hasSamples(n)) return;
    try stderr(io, "--- stack profile: wall sum / avg / GPU sum ms (segmented) ---\n");
    var i: u64 = 0;
    while (i < n) : (i += 1) {
        const samples = c.zdraw_metal_stack_profile_samples(i);
        if (samples == 0) continue;
        const ns = c.zdraw_metal_stack_profile_ns(i);
        const gpu_ns = c.zdraw_metal_stack_profile_gpu_ns(i);
        const name = std.mem.span(c.zdraw_metal_stack_profile_name(i));
        const text = try std.fmt.allocPrint(
            allocator,
            "  {s:<18}{d:>9.2}{d:>9.2}{d:>9.2}  (n={d})\n",
            .{ name, ms(ns), avg(ns, samples), ms(gpu_ns), samples },
        );
        defer allocator.free(text);
        try stderr(io, text);
    }
}

fn hasSamples(n: u64) bool {
    var i: u64 = 0;
    while (i < n) : (i += 1) {
        if (c.zdraw_metal_stack_profile_samples(i) > 0) return true;
    }
    return false;
}

fn avg(ns: u64, samples: u64) f64 {
    if (samples == 0) return 0;
    return ms(ns) / @as(f64, @floatFromInt(samples));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn stderr(io: std.Io, text: []const u8) !void {
    try std.Io.File.stderr().writeStreamingAll(io, text);
}
