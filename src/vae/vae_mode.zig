//! Typed VAE decode mode (migration layer).
//!
//! VAE mode intent was scattered across ZDRAW_VAE* env flags read deep in the
//! decode path. This names the tiers as a typed enum and populates the
//! existing env defaults (overwrite=0, same migration style as
//! runtime_options) WITHOUT rewriting any VAE internals or removing read
//! sites. strict/product map to ALREADY-GATED behavior; preview/legacy are
//! reserved (named, not yet certified) and are not wired into the production
//! profiles.
//!
//! Debug flags stay env-only by design: ZDRAW_VAE_DUMP_STAGES,
//! ZDRAW_VAE_F16SIM, ZDRAW_VAE_F16_STAGES, ZDRAW_VAE_UNFUSE, …

const std = @import("std");

const Flag = @import("../runtime/runtime_options.zig").Flag;

pub const VaeMode = enum {
    /// Bit-exact streamed VAE (the v7 conv path; max|d|=0 vs raw f32).
    strict,
    /// f16 fast streamed VAE — the "fast creative default" (the shipped default).
    /// NOT precision-preserving: perceptual metrics (LPIPS/DISTS, runs/product-
    /// proof) confirm detail-dependent divergence from strict — identical on
    /// smooth content but LPIPS up to ~0.24 on detailed textures; reviewed, no
    /// obvious artifacts. Do NOT claim "≤1-LSB"/"visually matched"/"exact".
    product,
    /// Tiny-AE fast preview: gates on ZDRAW_VAE=tae (the weights path comes
    /// from the optional ZDRAW_TAE env). Reserved / not certified.
    preview,
    /// Bare binary default / debug fallback (no profile flags applied).
    legacy,

    pub fn parse(name: []const u8) ?VaeMode {
        inline for (.{ .strict, .product, .preview, .legacy }) |m| {
            if (std.mem.eql(u8, name, @tagName(m))) return m;
        }
        return null;
    }
};

const strict_flags = [_]Flag{
    .{ .name = "ZDRAW_VAE", .value = "raw" },
    .{ .name = "ZDRAW_VAE_STREAM", .value = "1" },
    .{ .name = "ZDRAW_VAE_V7", .value = "1" },
    // Strip height is exactness-neutral (the windowed split is byte-for-byte
    // the whole-image decode); 256 rows is 4x fewer strips and command
    // buffers than the default and was the product tier's value since June.
    .{ .name = "ZDRAW_VAE_STRIP", .value = "256" },
};

const product_flags = [_]Flag{
    // Product decodes on the owned streamed decoder with f16 activation
    // storage and the owned half MMA conv kernels (conv2d_window_h): no
    // MPSGraph anywhere on the render path (owner rule 2026-08-26). Measured
    // 2026-08-27 on the dev box, interleaved: 1.98-2.11 s vs MPSGraph's
    // 1.43-1.45 s vs the f32 route's 7.3-7.9 s (ledger
    // vae-decoder-routes-20260827); the conv kernel is the open item (W7).
    .{ .name = "ZDRAW_VAE", .value = "raw" },
    .{ .name = "ZDRAW_VAE_STREAM", .value = "1" },
    .{ .name = "ZDRAW_VAE_V7", .value = "1" },
    .{ .name = "ZDRAW_VAE_F16", .value = "1" },
    .{ .name = "ZDRAW_VAE_FINAL_H", .value = "1" },
    .{ .name = "ZDRAW_VAE_MID_F16", .value = "1" },
    .{ .name = "ZDRAW_VAE_FULL_H", .value = "1" },
    .{ .name = "ZDRAW_VAE_STATSSQ", .value = "1" },
    .{ .name = "ZDRAW_VAE_STRIP", .value = "256" },
    // Winograd F(4x4,3x3) on NOVA's fp16-exact points for every 3x3 stride-1
    // conv of the half route (vae-winograd-20260827): Klein 1024 decode
    // 1.40 -> 0.75 s, Z-Image up-blocks 1124 -> 449 ms, product-vs-strict
    // 48.7 dB (the direct kernels were at 48.7 too). ZDRAW_VAE_WINO=0 is the
    // direct-kernel A/B arm.
    .{ .name = "ZDRAW_VAE_WINO", .value = "1" },
    // Resident mid-attention (mvattn.zig) on the owned row-blocked path;
    // product takes its half-precision operands (f32 accumulate), strict
    // the exact f32 path. The GPU stats reduction is the one value-changing
    // element vs strict.
    .{ .name = "ZDRAW_VAE_ATTN_GPU", .value = "1" },
    .{ .name = "ZDRAW_VAE_ATTN_HALF", .value = "1" },
};

const preview_flags = [_]Flag{
    // tae.decodeIfEnabled gates ONLY on ZDRAW_VAE=tae (src/tae.zig:40-41).
    // ZDRAW_TAE is an OPTIONAL weights-PATH override (it defaults to a local
    // taef1.safetensors), so it stays env-only — the old ZDRAW_TAE=1 was both
    // insufficient (the gate never fired) and wrong (a bogus "1" path).
    .{ .name = "ZDRAW_VAE", .value = "tae" },
};

/// The env defaults for a VAE tier. strict/product are the gated production
/// sets; preview names the ZDRAW_VAE=tae gate; legacy is empty (bare default).
pub fn flagsFor(mode: VaeMode) []const Flag {
    return switch (mode) {
        .strict => &strict_flags,
        .product => &product_flags,
        .preview => &preview_flags,
        .legacy => &.{},
    };
}

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Populate the VAE env defaults, never overwriting an explicit value.
/// `explicit` = the tier came from a typed user choice (--profile): it then
/// overwrites pre-existing env values; a defaulted tier defers to them.
pub fn applyEnv(mode: VaeMode, explicit: bool) void {
    const ow: c_int = if (explicit) 1 else 0;
    for (flagsFor(mode)) |f| _ = setenv(f.name.ptr, f.value.ptr, ow);
}

test "strict and product map to the gated flag sets" {
    try std.testing.expectEqual(@as(usize, 4), flagsFor(.strict).len);
    try std.testing.expectEqual(@as(usize, 12), flagsFor(.product).len);
    try std.testing.expectEqual(@as(usize, 0), flagsFor(.legacy).len);
}

test "preview includes every env tae.decodeIfEnabled requires" {
    // The decode path hard-gates on ZDRAW_VAE == "tae" (src/tae.zig:40-41);
    // that pair must be present for the named tier to actually engage. (The
    // ZDRAW_TAE weights path is an optional override, not a gate, so it is
    // intentionally NOT asserted here.)
    const required = Flag{ .name = "ZDRAW_VAE", .value = "tae" };
    var found = false;
    for (preview_flags) |f| {
        if (std.mem.eql(u8, f.name, required.name) and
            std.mem.eql(u8, f.value, required.value)) found = true;
    }
    try std.testing.expect(found);
}

test "vae mode parses canonical names" {
    try std.testing.expectEqual(VaeMode.product, VaeMode.parse("product").?);
    try std.testing.expectEqual(VaeMode.preview, VaeMode.parse("preview").?);
    try std.testing.expect(VaeMode.parse("nope") == null);
}
