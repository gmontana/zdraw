//! Compile-time embedding of the vendored metal-flash-attention kernels.
//!
//! The Metal bridge compiles these sources at runtime (newLibraryWithSource);
//! embedding them makes the binary self-contained. Before this, the bridge
//! read `vendor/mfa/*.metal` relative to the working directory, so any
//! render above 2048 tokens failed with MetalDispatchFailed unless the
//! process ran from the repository root (found 2026-08-26).

pub const apple9: [:0]const u8 = @embedFile("mfa-fwd-d128.metal");
pub const m1m2: [:0]const u8 = @embedFile("mfa-fwd-d128-m1m2.metal");
