//! Activation storage precision for the VAE decoder.
//!
//! ZDRAW_VAE_F16=1 stores the high-res inter-op VAE activations as f16 instead
//! of f32, roughly halving the 1024px decode memory peak. Every kernel still
//! accumulates in f32 (conv MACs in simdgroup_float8x8, norm reductions in
//! float); only the buffers handed between ops become half. Default OFF: f32
//! stays the shipping default and is pixel-identical to before this flag.

const std = @import("std");

// dtype code shared with the Metal kernels: 1 == f16, 3 == f32 (matches the
// `read_value` dtype switch in the conv/norm shaders).
pub const code_f16: u32 = 1;
pub const code_f32: u32 = 3;

/// True when ZDRAW_VAE_F16=1; the VAE activation buffers then store f16.
pub fn enabled() bool {
    const raw = std.c.getenv("ZDRAW_VAE_F16") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

/// Byte size of one activation element for the active mode.
pub fn elemSize() usize {
    return if (enabled()) @sizeOf(f16) else @sizeOf(f32);
}

/// The kernel dtype code for the activation storage in the active mode.
pub fn code() u32 {
    return if (enabled()) code_f16 else code_f32;
}
