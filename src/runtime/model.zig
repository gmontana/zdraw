//! Shared inference requests, results and model dispatch.
//! Unsupported model execution returns an error.

const std = @import("std");

const kinds = @import("../cli/model_kind.zig");
const progress = @import("../cli/progress.zig");
const weights = @import("../pack/weights.zig");
const zflux2 = @import("../klein/zflux2.zig");
const zflux2_run = @import("../klein/zflux2_run.zig");

pub const ModelKind = kinds.ModelKind;

pub const Request = struct {
    kind: ModelKind,
    weights_dir: []const u8,
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    // 0 = unset: resolved from the model policy (1.0 distilled, 4.0 base),
    // matching the CLI. Lab tools that omit it now benchmark base variants
    // at their real guided cost instead of a silent 1.0.
    guidance: f32 = 0,
    /// img2img (Klein): empty = text to image.
    init_image: []const u8 = "",
    strength: f32 = 1.0,
    mask: []const u8 = "",
    ref_image: []const u8 = "",
};

pub const Result = struct {
    pixels: []u8,
    width: u32,
    height: u32,
};

pub const Error = error{
    EmptyPrompt,
    InvalidImageSize,
    UnsupportedInference,
    // Z-Image's production path is runtime.Runtime (main.zig); the one-shot
    // generate() exists only for Klein and rejects Z-Image with this.
    UseRuntimePath,
};

pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    request: Request,
) !Result {
    if (request.prompt.len == 0) return error.EmptyPrompt;
    if (request.width == 0 or request.height == 0) return error.InvalidImageSize;

    return switch (request.kind) {
        // Z-Image's production path is runtime.Runtime (main.zig); the old
        // one-shot path was dead code that could drift from the real engine.
        .z_image_turbo => error.UseRuntimePath,
        .flux2_klein_4b,
        .flux2_klein_9b,
        .flux2_klein_base_4b,
        .flux2_klein_base_9b,
        .flux2_klein_9b_kv,
        => generateKlein(io, allocator, request),
    };
}

// FLUX.2 Klein 4B: the oracle-verified native path (zflux2_dit forward +
// flow-match loop + native VAE), assembled behind the standard CLI.
// Correctness tier: GEMMs on the shared Metal path, glue on CPU; the speed
// pass (resident chain/fused glue/batching) follows the roadmap.
pub const KleinRuntime = zflux2_run.Runtime;
pub const KleinRequest = zflux2_run.Request;
pub const KleinResult = zflux2_run.Result;

/// Open a reusable Klein runtime (validates files, selects the size config,
/// loads weights/contexts once). The caller drives generate()+deinit() — used
/// by the CLI's --repeat path for warm numbers.
pub fn openKlein(
    io: std.Io,
    allocator: std.mem.Allocator,
    kind: ModelKind,
    weights_dir: []const u8,
) !KleinRuntime {
    try weights.validate(io, allocator, kind, weights_dir);
    return zflux2_run.Runtime.init(io, allocator, try kleinConfig(kind), weights_dir);
}

/// Transformer dims per variant: base models share their distilled sibling's
/// architecture (verified: identical tensor sets, guidance_embeds false); the
/// kv fine-tune is dimensionally a 9B.
/// Sidecar filename + size tag for user-facing messages (main.zig).
pub fn kleinPackName(kind: ModelKind) []const u8 {
    const cfg = kleinConfig(kind) catch return zflux2.Config.klein_4b.pack_name;
    return cfg.pack_name;
}

pub fn kleinSizeTag(kind: ModelKind) []const u8 {
    return switch (kind) {
        .flux2_klein_4b, .flux2_klein_base_4b => "4b",
        else => "9b",
    };
}

fn kleinConfig(kind: ModelKind) !zflux2.Config {
    return switch (kind) {
        .flux2_klein_4b => zflux2.Config.klein_4b,
        .flux2_klein_base_4b => zflux2.Config.klein_base_4b,
        .flux2_klein_9b => zflux2.Config.klein_9b,
        .flux2_klein_base_9b => zflux2.Config.klein_base_9b,
        .flux2_klein_9b_kv => zflux2.Config.klein_9b_kv,
        .z_image_turbo => error.UnsupportedInference,
    };
}

fn generateKlein(
    io: std.Io,
    allocator: std.mem.Allocator,
    request: Request,
) !Result {
    const validate = try progress.begin(io, allocator, "checking model files");
    try weights.validate(io, allocator, request.kind, request.weights_dir);
    try progress.done(io, allocator, validate);

    const cfg = try kleinConfig(request.kind);
    const load_stage = try progress.begin(io, allocator, "loading model");
    var rt = try zflux2_run.Runtime.init(io, allocator, cfg, request.weights_dir);
    defer rt.deinit(io, allocator);
    try progress.done(io, allocator, load_stage);

    const pol = kinds.policy(request.kind);
    const out = try rt.generate(io, allocator, .{
        .prompt = request.prompt,
        .width = request.width,
        .height = request.height,
        .steps = request.steps,
        .seed = request.seed,
        .guidance = if (request.guidance == 0) pol.default_guidance else request.guidance,
        .init_image = request.init_image,
        .strength = if (request.init_image.len > 0) request.strength else 1.0,
        .mask = request.mask,
        .ref_image = request.ref_image,
    });
    return .{ .pixels = out.pixels, .width = out.width, .height = out.height };
}

pub fn defaultKind() ModelKind {
    return kinds.defaultKind();
}

pub fn parseKind(name: []const u8) ?ModelKind {
    return kinds.parseKind(name);
}

pub const policy = kinds.policy;

pub fn cliName(kind: ModelKind) []const u8 {
    return kinds.cliName(kind);
}

pub fn kindName(kind: ModelKind) []const u8 {
    return kinds.kindName(kind);
}

test "every Klein variant packs to its own sidecar filename" {
    // Why this matters: see the pack_name contract in zflux2.zig.
    const kleins = [_]ModelKind{
        .flux2_klein_4b,
        .flux2_klein_9b,
        .flux2_klein_base_4b,
        .flux2_klein_base_9b,
        .flux2_klein_9b_kv,
    };
    for (kleins, 0..) |a, i| {
        const name_a = (try kleinConfig(a)).pack_name;
        try std.testing.expect(name_a.len > 0);
        for (kleins[i + 1 ..]) |b| {
            const name_b = (try kleinConfig(b)).pack_name;
            try std.testing.expect(!std.mem.eql(u8, name_a, name_b));
        }
    }
}

test "base variants keep their distilled sibling's dimensions" {
    const base4 = try kleinConfig(.flux2_klein_base_4b);
    const dist4 = try kleinConfig(.flux2_klein_4b);
    try std.testing.expectEqual(dist4.hidden, base4.hidden);
    try std.testing.expectEqual(dist4.double_layers, base4.double_layers);
    try std.testing.expectEqual(dist4.single_layers, base4.single_layers);
    try std.testing.expectEqual(dist4.joint_dim, base4.joint_dim);
}
