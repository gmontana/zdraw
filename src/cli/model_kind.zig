const std = @import("std");

pub const klein_max_image_tokens: usize = 4096;

/// The default half-activation GEMM requires image rows divisible by 32.
pub fn kleinDimsFit(width: u32, height: u32) bool {
    if (width == 0 or height == 0 or width % 32 != 0 or height % 32 != 0) return false;
    const tokens = (@as(u64, width) / 16) * (@as(u64, height) / 16);
    return tokens % 32 == 0 and tokens <= klein_max_image_tokens;
}

pub const ModelKind = enum {
    z_image_turbo,
    flux2_klein_4b,
    flux2_klein_9b,
    flux2_klein_base_4b,
    flux2_klein_base_9b,
    flux2_klein_9b_kv,
};

// Per-variant sampling policy (mflux parity): distilled Klein models are
// step-distilled to 4 steps at guidance 1.0 and reject other guidance in
// txt2img; base models default to the diffusers 50-step / guidance-4.0
// regime via true CFG; only the kv fine-tune caches reference K/V.
pub const Policy = struct {
    default_steps: u32,
    default_guidance: f32,
    cfg_allowed: bool,
    kv_cache: bool,
    /// Steps for an instruction edit (a reference present) when the caller
    /// left steps at 0. Measured 2026-09-03: on the distilled models two
    /// steps edit as well as four (background LPIPS 0.024 vs 0.018, edit
    /// region 0.042 vs 0.038) and the cliff is between two and one.
    edit_steps: u32,
};

pub fn policy(kind: ModelKind) Policy {
    return switch (kind) {
        .z_image_turbo,
        .flux2_klein_4b,
        .flux2_klein_9b,
        => .{
            .default_steps = 4,
            .default_guidance = 1.0,
            .cfg_allowed = false,
            .kv_cache = false,
            .edit_steps = 2,
        },
        .flux2_klein_9b_kv => .{
            .default_steps = 4,
            .default_guidance = 1.0,
            .cfg_allowed = false,
            .kv_cache = true,
            .edit_steps = 2,
        },
        .flux2_klein_base_4b,
        .flux2_klein_base_9b,
        => .{
            .default_steps = 50,
            .default_guidance = 4.0,
            .cfg_allowed = true,
            .kv_cache = false,
            .edit_steps = 8,
        },
    };
}

pub fn defaultKind() ModelKind {
    return .z_image_turbo;
}

pub fn parseKind(name: []const u8) ?ModelKind {
    if (std.mem.eql(u8, name, "z-image-turbo")) return .z_image_turbo;
    if (std.mem.eql(u8, name, "flux2-klein-4b")) return .flux2_klein_4b;
    if (std.mem.eql(u8, name, "flux2-klein-9b")) return .flux2_klein_9b;
    if (std.mem.eql(u8, name, "flux2-klein-base-4b")) return .flux2_klein_base_4b;
    if (std.mem.eql(u8, name, "flux2-klein-base-9b")) return .flux2_klein_base_9b;
    if (std.mem.eql(u8, name, "flux2-klein-9b-kv")) return .flux2_klein_9b_kv;
    return null;
}

/// Models supported by the public CLI; research variants remain internal.
pub fn parsePublic(name: []const u8) ?ModelKind {
    const kind = parseKind(name) orelse return null;
    return switch (kind) {
        .z_image_turbo, .flux2_klein_4b, .flux2_klein_base_4b => kind,
        else => null,
    };
}

/// Resolve the 0-sentinels against the model's sampling policy and reject
/// guidance on step-distilled models (mflux parity: they are trained at a
/// fixed guidance of 1.0 and have no unconditional branch to steer with).
/// Shared by the CLI at startup and by the session's `model` swap.
pub fn sampling(kind: ModelKind, steps: u32, guidance: f32) !struct { u32, f32 } {
    return samplingRef(kind, steps, guidance, false);
}

/// `sampling` for a request that may carry a reference image: an edit with
/// steps left at 0 gets the model's `edit_steps` instead of `default_steps`.
pub fn samplingRef(
    kind: ModelKind,
    steps: u32,
    guidance: f32,
    has_ref: bool,
) !struct { u32, f32 } {
    const pol = policy(kind);
    const g = if (guidance == 0) pol.default_guidance else guidance;
    if (g != 1.0 and !pol.cfg_allowed) return error.GuidanceUnsupported;
    const fallback = if (has_ref) pol.edit_steps else pol.default_steps;
    return .{ if (steps == 0) fallback else steps, g };
}

pub fn cliName(kind: ModelKind) []const u8 {
    return switch (kind) {
        .z_image_turbo => "z-image-turbo",
        .flux2_klein_4b => "flux2-klein-4b",
        .flux2_klein_9b => "flux2-klein-9b",
        .flux2_klein_base_4b => "flux2-klein-base-4b",
        .flux2_klein_base_9b => "flux2-klein-base-9b",
        .flux2_klein_9b_kv => "flux2-klein-9b-kv",
    };
}

pub fn kindName(kind: ModelKind) []const u8 {
    return switch (kind) {
        .z_image_turbo => "Z-Image-Turbo",
        .flux2_klein_4b => "FLUX.2 [klein] 4B",
        .flux2_klein_9b => "FLUX.2 [klein] 9B",
        .flux2_klein_base_4b => "FLUX.2 [klein] base 4B",
        .flux2_klein_base_9b => "FLUX.2 [klein] base 9B",
        .flux2_klein_9b_kv => "FLUX.2 [klein] 9B KV",
    };
}

test "default target is the fast Z-Image model" {
    try std.testing.expectEqual(ModelKind.z_image_turbo, defaultKind());
}

test "Klein dimensions fit the default kernel alignment and capacity" {
    const supported = .{
        .{ 128, 128 },  .{ 1024, 1024 }, .{ 1024, 768 },
        .{ 768, 1024 }, .{ 512, 2048 },  .{ 96, 256 },
    };
    inline for (supported) |size| {
        try std.testing.expect(kleinDimsFit(size[0], size[1]));
    }
    const rejected = .{
        .{ 64, 64 }, .{ 544, 800 }, .{ 1536, 1536 },             .{ 2048, 2048 },
        .{ 0, 128 }, .{ 528, 512 }, .{ 0xffffffe0, 0xffffffe0 },
    };
    inline for (rejected) |size| {
        try std.testing.expect(!kleinDimsFit(size[0], size[1]));
    }
}

test "parse supported model names" {
    try std.testing.expectEqual(ModelKind.z_image_turbo, parseKind("z-image-turbo").?);
    try std.testing.expectEqual(ModelKind.flux2_klein_4b, parseKind("flux2-klein-4b").?);
    try std.testing.expectEqual(ModelKind.flux2_klein_9b, parseKind("flux2-klein-9b").?);
    try std.testing.expectEqual(
        ModelKind.flux2_klein_base_4b,
        parseKind("flux2-klein-base-4b").?,
    );
    try std.testing.expectEqual(
        ModelKind.flux2_klein_base_9b,
        parseKind("flux2-klein-base-9b").?,
    );
    try std.testing.expectEqual(ModelKind.flux2_klein_9b_kv, parseKind("flux2-klein-9b-kv").?);
    try std.testing.expect(parseKind("flux") == null);
}

test "the public CLI exposes the Apache-licensed models only" {
    const base = parsePublic("flux2-klein-base-4b").?;
    try std.testing.expectEqual(ModelKind.flux2_klein_base_4b, base);
    try std.testing.expectEqual(ModelKind.flux2_klein_4b, parsePublic("flux2-klein-4b").?);
    try std.testing.expect(parsePublic("flux2-klein-9b") == null);
    try std.testing.expect(parsePublic("flux2-klein-base-9b") == null);
    try std.testing.expect(parsePublic("flux2-klein-9b-kv") == null);
}

test "the base model samples at 50 steps with guidance 4 unless told otherwise" {
    const defaults = try samplingRef(.flux2_klein_base_4b, 0, 0, false);
    try std.testing.expectEqual(@as(u32, 50), defaults[0]);
    try std.testing.expectEqual(@as(f32, 4.0), defaults[1]);
    // An explicit guidance 1 disables the unconditional pass.
    const unguided = try samplingRef(.flux2_klein_base_4b, 0, 1.0, false);
    try std.testing.expectEqual(@as(f32, 1.0), unguided[1]);
    const distilled = samplingRef(.flux2_klein_4b, 0, 4.0, false);
    try std.testing.expectError(error.GuidanceUnsupported, distilled);
}

test "variant policy matches mflux sampling semantics" {
    try std.testing.expectEqual(@as(u32, 4), policy(.flux2_klein_9b).default_steps);
    try std.testing.expect(!policy(.flux2_klein_9b).cfg_allowed);
    try std.testing.expectEqual(@as(u32, 50), policy(.flux2_klein_base_4b).default_steps);
    try std.testing.expect(policy(.flux2_klein_base_9b).cfg_allowed);
    try std.testing.expect(policy(.flux2_klein_9b_kv).kv_cache);
    try std.testing.expect(!policy(.flux2_klein_4b).kv_cache);
}

test "an instruction edit defaults to the model's edit steps" {
    const edit = try samplingRef(.flux2_klein_4b, 0, 0, true);
    try std.testing.expectEqual(@as(u32, 2), edit[0]);
    const plain = try samplingRef(.flux2_klein_4b, 0, 0, false);
    try std.testing.expectEqual(@as(u32, 4), plain[0]);
    const explicit = try samplingRef(.flux2_klein_4b, 4, 0, true);
    try std.testing.expectEqual(@as(u32, 4), explicit[0]);
    const base = try samplingRef(.flux2_klein_base_4b, 0, 0, true);
    try std.testing.expectEqual(@as(u32, 8), base[0]);
}
