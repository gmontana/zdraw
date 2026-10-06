//! Typed production runtime profiles.
//!
//! The winning Z-Image configurations used to live ONLY as env strings in
//! tools/p0bench.sh / scale_bench.sh — scripts were the source of truth for
//! product behavior. Here the stable algorithm knobs are encoded as typed
//! presets; `applyEnv` populates the environment the existing read sites
//! consume, WITHOUT overwriting anything already set (explicit env / a lab
//! override still wins). This migration step does NOT remove the getenv read
//! sites; it just stops scripts from being the only source of truth.
//!
//! VAE flags are owned by vae_mode.zig and composed here (not duplicated).
//!
//! Deliberately excluded:
//!   - ZDRAW_ZPACK: a machine/CWD-relative weights PATH (a deployment detail,
//!     not an algorithm knob); left to env/script so we never silently bake a
//!     fragile path. An explicit ZDRAW_ZPACK always wins regardless.
//!   - Debug/probe flags (METRICS, MEMTRACE, GPU_TRACE, DUMP_*, F16SIM,
//!     MIX_PROBE, …): lab-only, stay env-only by design.

const std = @import("std");
const vae_mode = @import("../vae/vae_mode.zig");

pub const Flag = struct { name: [:0]const u8, value: [:0]const u8 };

pub const Quality = enum {
    /// Exact transformer GEMMs plus the bit-exact reference VAE.
    strict,
    /// Fast creative default: f16 transformer GEMMs and the f16 VAE tier. NOT
    /// precision-preserving — LPIPS/DISTS (runs/product-proof) confirm VAE
    /// detail-dependent divergence from strict (identical on smooth content,
    /// LPIPS up to ~0.24 on detailed textures); reviewed, no obvious artifacts.
    /// Not an "≤1-LSB"/"visually matched"/"exact quality" claim.
    product,

    pub fn parse(name: []const u8) ?Quality {
        if (std.mem.eql(u8, name, "strict")) return .strict;
        if (std.mem.eql(u8, name, "product")) return .product;
        return null;
    }

    /// The VAE tier each quality composes (the VAE flags live in vae_mode).
    pub fn vaeMode(self: Quality) vae_mode.VaeMode {
        return switch (self) {
            .strict => .strict,
            .product => .product,
        };
    }
};

/// Non-VAE algorithm knobs shared by both tiers. W16 substitution is ON: the
/// f16-sidecar -> gemm_f16_direct substrate is recertified post-MFA-fix as
/// byte-identical to the generic tier and 12.7% faster at 1024 (ledger
/// zimage-w16-ours16-recert, 2026-08-04; the 45a2094 pin's NaN evidence was
/// the W6 route inside the corruption window). Strict is unaffected: exact
/// GEMMs never substitute. W6/steel stay lab-only.
const common = [_]Flag{
    .{ .name = "ZDRAW_DENSE", .value = "ours-f16" },
    .{ .name = "ZDRAW_STACK_GEMM", .value = "inherit" },
    .{ .name = "ZDRAW_STACK_W16", .value = "1" },
    .{ .name = "ZDRAW_QK_HM", .value = "1" },
};

const strict = [_]Flag{
    .{ .name = "ZDRAW_GEMM", .value = "exact" },
};

const product = [_]Flag{
    .{ .name = "ZDRAW_GEMM", .value = "half" },
};

pub fn commonFlags() []const Flag {
    return &common;
}

pub fn qualityFlags(quality: Quality) []const Flag {
    return switch (quality) {
        .strict => &strict,
        .product => &product,
    };
}

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Populate the environment for `quality`: the non-VAE common knobs plus the
/// composed VAE tier. An EXPLICIT --profile overwrites pre-existing env
/// values (the typed CLI choice is the product contract, architecture.md
/// section 4.4); the bare default defers to env so exported experiment
/// variables keep working. Must run before any Metal init so the read sites
/// observe it.
/// The whole-image f32 reference VAE decode: the strict tier's streamed
/// decode is certified byte-identical to it, so it is the arm quality work
/// compares against. Applied AFTER the profile (overwrite=1) because an
/// explicit --profile otherwise wins over env by design (section 4.4).
pub fn applyVaeRef() void {
    _ = setenv("ZDRAW_VAE", "raw", 1);
    _ = setenv("ZDRAW_VAE_STREAM", "0", 1);
}

pub fn applyEnv(quality: Quality, explicit: bool) void {
    const ow: c_int = if (explicit) 1 else 0;
    for (common) |f| _ = setenv(f.name.ptr, f.value.ptr, ow);
    for (qualityFlags(quality)) |f| _ = setenv(f.name.ptr, f.value.ptr, ow);
    vae_mode.applyEnv(quality.vaeMode(), explicit);
}

test "common holds only the non-VAE algorithm knobs" {
    const want = [_][]const u8{
        "ZDRAW_DENSE",
        "ZDRAW_STACK_GEMM",
        "ZDRAW_STACK_W16",
        "ZDRAW_QK_HM",
    };
    try std.testing.expectEqual(want.len, common.len);
    for (want, common) |w, g| try std.testing.expect(std.mem.eql(u8, w, g.name));
}

test "quality owns transformer precision" {
    try std.testing.expectEqualStrings("ZDRAW_GEMM", qualityFlags(.strict)[0].name);
    try std.testing.expectEqualStrings("exact", qualityFlags(.strict)[0].value);
    try std.testing.expectEqualStrings("half", qualityFlags(.product)[0].value);
}

// The stack policy must be an explicit pin, never an accident of absence.
// Since the 2026-08-04 recertification the pinned policy is inherit + W16
// substitution (byte-identical to the generic tier, ledger
// zimage-w16-ours16-recert); strict is untouched because exact GEMMs never
// substitute.
test "shipping profiles explicitly pin the certified transformer stack" {
    const stack = for (common) |flag| {
        if (std.mem.eql(u8, flag.name, "ZDRAW_STACK_GEMM")) break flag.value;
    } else return error.MissingStackPolicy;
    try std.testing.expect(std.mem.eql(u8, stack, "inherit"));
    const w16 = for (common) |flag| {
        if (std.mem.eql(u8, flag.name, "ZDRAW_STACK_W16")) break flag.value;
    } else return error.MissingStackPolicy;
    try std.testing.expect(std.mem.eql(u8, w16, "1"));
}

test "quality parses canonical names and maps to its VAE tier" {
    try std.testing.expectEqual(Quality.product, Quality.parse("product").?);
    try std.testing.expectEqual(Quality.strict, Quality.parse("strict").?);
    try std.testing.expect(Quality.parse("fast") == null);
    try std.testing.expectEqual(vae_mode.VaeMode.product, Quality.product.vaeMode());
    try std.testing.expectEqual(vae_mode.VaeMode.strict, Quality.strict.vaeMode());
}

test "a defaulted profile defers to exported env overrides" {
    defer clearApplied(.product);
    _ = setenv("ZDRAW_VAE_ATTN_HALF", "0", 1); // exported experiment override
    applyEnv(.product, false); // would otherwise set 1 via the product tier
    const got = std.c.getenv("ZDRAW_VAE_ATTN_HALF") orelse return error.Unexpected;
    try std.testing.expect(std.mem.eql(u8, std.mem.span(got), "0"));
}

test "an explicit profile wins over stale env values" {
    defer clearApplied(.product);
    _ = setenv("ZDRAW_VAE_ATTN_HALF", "0", 1); // stale exported value
    applyEnv(.product, true); // typed user choice: the product contract wins
    const got = std.c.getenv("ZDRAW_VAE_ATTN_HALF") orelse return error.Unexpected;
    try std.testing.expect(std.mem.eql(u8, std.mem.span(got), "1"));
}

/// Tests share one process: every flag applyEnv exported is removed again so
/// later tests (the GEMM route tests among them) see the default environment.
fn clearApplied(quality: Quality) void {
    for (common) |f| _ = unsetenv(f.name.ptr);
    for (qualityFlags(quality)) |f| _ = unsetenv(f.name.ptr);
    for (vae_mode.flagsFor(quality.vaeMode())) |f| _ = unsetenv(f.name.ptr);
}

extern fn unsetenv(name: [*:0]const u8) c_int;
