//! Tests for the Z-Image VAE decoder.

const std = @import("std");

const vdecode = @import("vdecode.zig");

test "latent denormalization matches official config" {
    var out = [_]f32{0};
    vdecode.denorm(&out, &.{0.3611});
    try std.testing.expectApproxEqAbs(1.1159, out[0], 0.0001);
}
