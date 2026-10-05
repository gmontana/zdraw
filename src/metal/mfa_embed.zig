//! Hands the embedded MFA kernel sources to the Metal bridge.
//!
//! `metal_c.zig` forces this file into every compile that links the bridge,
//! so `zdraw_mfa_source` is always defined when `mfa_library` asks for it.

const mfa = @import("mfa");

/// The MFA kernel source for the device family (Apple9: 16-row tiles;
/// M1/M2: 32-row tiles). Static lifetime; `len` excludes the terminator.
fn mfaSource(apple9: c_int, len: *usize) callconv(.c) [*]const u8 {
    const src: [:0]const u8 = if (apple9 != 0) mfa.apple9 else mfa.m1m2;
    len.* = src.len;
    return src.ptr;
}

comptime {
    @export(&mfaSource, .{ .name = "zdraw_mfa_source" });
}
