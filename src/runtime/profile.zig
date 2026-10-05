//! Hardware capabilities — NOT the execution profile.
//!
//! Ownership split (kept intentionally separate, do not merge):
//!   - profile.zig (this file): detect hardware/OS capabilities.
//!   - runtime_options.zig: EXECUTION profile — the typed product/strict
//!     quality intent and its algorithm knobs.
//!   - vae_mode.zig: VAE tier intent (strict/product/preview/legacy).
//!
//! Precision policy is deliberately absent here. It changes model behavior and
//! must be selected by a quality-certified execution profile, never inferred
//! from RAM alone.

const std = @import("std");

const c = @import("../metal/metal_c.zig");
const progress = @import("../cli/progress.zig");

pub const Device = struct {
    name: [128]u8 = @splat(0),
    name_len: usize = 0,
    ram_bytes: u64 = 0,
    gpu_working_set: u64 = 0,
    os_major: u32 = 0,

    pub fn chip(self: *const Device) []const u8 {
        return self.name[0..self.name_len];
    }
};

pub fn detect() ?Device {
    var d = Device{};
    const rc = c.zdraw_device_identity(
        &d.name,
        d.name.len,
        &d.ram_bytes,
        &d.gpu_working_set,
        &d.os_major,
    );
    if (rc != 0) return null;
    d.name_len = std.mem.indexOfScalar(u8, &d.name, 0) orelse d.name.len;
    return d;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Detect the host, announce it, and apply capability-only fallbacks.
pub fn apply(io: std.Io, allocator: std.mem.Allocator) void {
    const dev = detect() orelse return;
    const msg = std.fmt.allocPrint(allocator, "device {s}, {d} GiB, macOS {d}", .{
        dev.chip(), dev.ram_bytes / (1 << 30), dev.os_major,
    }) catch return;
    defer allocator.free(msg);
    progress.event(io, allocator, msg) catch return;
    if (dev.os_major < 14 and std.c.getenv("ZDRAW_ATTN") == null) {
        _ = setenv("ZDRAW_ATTN", "rows", 0);
    }
}
