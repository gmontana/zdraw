//! Versioned execution plans for the private optimization control plane.
//!
//! The plan is data, not authority: Zig parses and validates every requested
//! route before execution. Python may propose plans but never bypass model,
//! quality, shape, lifetime, or hardware constraints.

const std = @import("std");

const model_kind = @import("model_kind.zig");

pub const schema_version: u32 = 1;
pub const max_regions: usize = 256;

pub const QualityContract = enum {
    exact,
    faithful,
    experimental,
};

pub const Phase = enum {
    text_encoder,
    denoiser,
    vae,
};

pub const Representation = enum {
    inherit,
    f32,
    f16,
    bf16,
    w8,
    w6,
    w4,
};

pub const Kernel = enum {
    inherit,
    metal,
    mps,
    mfa,
    steel,
    tensor_ops,
};

pub const Modality = enum {
    image,
    text,
};

pub const RegionLayout = enum {
    global,
    tile,
    stripe,
};

pub const AssignmentScope = enum {
    global,
    region_local,
};

pub const Selection = enum {
    facility_location,
};

pub const AlgorithmMode = enum {
    paper_spec,
    official_b578009,
};

pub const Assignment = enum {
    column_softmax,
};

pub const Unmerge = enum {
    paper_normalized_transpose,
    official_raw_transpose,
    pseudo_inverse,
};

pub const TokenCoarsen = struct {
    modality: Modality,
    source_tokens: u32,
    destination_tokens: u32,
    selection_layout: RegionLayout,
    assignment_scope: AssignmentScope,
    region_count: u32 = 1,
    selection: Selection = .facility_location,
    algorithm_mode: AlgorithmMode,
    assignment: Assignment = .column_softmax,
    unmerge: Unmerge = .paper_normalized_transpose,
    destination_refresh: u32 = 1,
    assignment_refresh: u32 = 1,
    assignment_scale: f32,
};

pub const HookBypass = struct {
    timesteps: []const u32,
};

pub const TokenScore = enum {
    feature_mean,
    zero,
};

pub const TokenReconstruction = enum {
    reference_scatter,
    cosine_residual_interpolate,
};

pub const TokenSelection = struct {
    semantic_program_sha256: []const u8,
    selection_program_sha256: []const u8,
    realization_sha256: []const u8,
    schedule_sha256: []const u8,
    source_sha256: []const u8,
    source_tokens: u32,
    feature_width: u32,
    selected_tokens: u32,
    layers: []const u32,
    timesteps: []const u32,
    score: TokenScore = .feature_mean,
    reconstruction: TokenReconstruction = .reference_scatter,
    reconstruction_weight: f32 = 0,
};

pub const Region = struct {
    id: []const u8,
    phase: Phase,
    layer_from: u32,
    layer_to: u32,
    representation: Representation = .inherit,
    kernel: Kernel = .inherit,
    token_coarsen: ?TokenCoarsen = null,
    hook_bypass: ?HookBypass = null,
    token_selection: ?TokenSelection = null,
};

pub const MemoryPolicy = struct {
    cross_phase_reuse: bool = false,
    max_peak_bytes: u64 = 0,
};

pub const ExecutionPlanV1 = struct {
    schema_version: u32,
    model: []const u8,
    quality_contract: QualityContract,
    source_candidate_sha256: ?[]const u8 = null,
    source_capsule_sha256: ?[]const u8 = null,
    regions: []const Region = &.{},
    memory: MemoryPolicy = .{},
};

pub const Capabilities = struct {
    mps: bool = true,
    tensor_ops: bool = false,
};

pub const Document = struct {
    parsed: std.json.Parsed(ExecutionPlanV1),
    source_sha256: [64]u8,

    pub fn deinit(self: *Document) void {
        self.parsed.deinit();
        self.* = undefined;
    }

    pub fn plan(self: *const Document) *const ExecutionPlanV1 {
        return &self.parsed.value;
    }
};

pub fn parse(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    capabilities: Capabilities,
) !Document {
    var parsed = try std.json.parseFromSlice(ExecutionPlanV1, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    });
    errdefer parsed.deinit();
    try validate(parsed.value, capabilities);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return .{
        .parsed = parsed,
        .source_sha256 = std.fmt.bytesToHex(digest, .lower),
    };
}

pub fn validate(plan: ExecutionPlanV1, capabilities: Capabilities) !void {
    if (plan.schema_version != schema_version) return error.UnsupportedPlanVersion;
    if (model_kind.parseKind(plan.model) == null) return error.UnsupportedModel;
    if (plan.source_candidate_sha256) |hash| {
        if (!isSha256(hash)) return error.InvalidCandidateHash;
    }
    if (plan.source_capsule_sha256) |hash| {
        if (!isSha256(hash)) return error.InvalidCapsuleHash;
    }
    if (plan.regions.len > max_regions) return error.TooManyRegions;
    for (plan.regions, 0..) |region, index| {
        if (region.id.len == 0) return error.EmptyRegionId;
        if (region.layer_from >= region.layer_to) return error.InvalidLayerRange;
        for (plan.regions[0..index]) |prior| {
            if (std.mem.eql(u8, prior.id, region.id)) return error.DuplicateRegionId;
        }
        if (region.kernel == .mps and !capabilities.mps) return error.UnsupportedKernel;
        if (region.kernel == .tensor_ops and !capabilities.tensor_ops) {
            return error.UnsupportedKernel;
        }
        if (plan.quality_contract == .exact and lossy(region.representation)) {
            return error.LossyExactPlan;
        }
        if (region.token_coarsen) |coarsen| {
            if (plan.quality_contract == .exact) return error.LossyExactPlan;
            if (region.phase != .denoiser) return error.InvalidCoarsenPhase;
            if (region.hook_bypass != null or region.token_selection != null) {
                return error.AmbiguousRegionTransform;
            }
            try validateCoarsen(coarsen);
        }
        if (region.hook_bypass) |bypass| {
            if (plan.quality_contract == .exact) return error.LossyExactPlan;
            if (region.phase != .denoiser) return error.InvalidBypassPhase;
            if (region.token_selection != null) return error.AmbiguousRegionTransform;
            try validateBypass(bypass);
        }
        if (region.token_selection) |selection| {
            if (plan.quality_contract == .exact) return error.LossyExactPlan;
            if (region.phase != .denoiser) return error.InvalidSelectionPhase;
            if (plan.source_candidate_sha256 == null or
                plan.source_capsule_sha256 == null)
            {
                return error.MissingCandidateCapsule;
            }
            try checkSelection(region, selection);
        }
    }
}

fn validateBypass(plan: HookBypass) !void {
    if (plan.timesteps.len == 0) return error.EmptyBypassTimesteps;
    for (plan.timesteps, 0..) |timestep, index| {
        if (index > 0 and plan.timesteps[index - 1] >= timestep) {
            return error.NonCanonicalBypassTimesteps;
        }
    }
}

fn checkSelection(region: Region, plan: TokenSelection) !void {
    if (!isSha256(plan.semantic_program_sha256) or
        !isSha256(plan.selection_program_sha256) or
        !isSha256(plan.realization_sha256) or
        !isSha256(plan.schedule_sha256) or
        !isSha256(plan.source_sha256))
    {
        return error.InvalidRealizationHash;
    }
    if (plan.source_tokens == 0 or
        plan.feature_width == 0 or
        plan.selected_tokens == 0 or
        plan.selected_tokens >= plan.source_tokens)
    {
        return error.InvalidTokenCount;
    }
    if (plan.source_tokens % 32 != 0 or plan.selected_tokens % 32 != 0) {
        return error.InvalidTokenAlignment;
    }
    if (!std.math.isFinite(plan.reconstruction_weight)) {
        return error.InvalidReconstructionWeight;
    }
    switch (plan.reconstruction) {
        .reference_scatter => if (plan.reconstruction_weight != 0) {
            return error.InvalidReconstructionWeight;
        },
        .cosine_residual_interpolate => if (plan.reconstruction_weight < 0 or
            plan.reconstruction_weight > 1)
        {
            return error.InvalidReconstructionWeight;
        },
    }
    if (plan.layers.len == 0 or plan.timesteps.len == 0) {
        return error.EmptySelectionDeployment;
    }
    for (plan.layers, 0..) |layer, index| {
        if (layer < region.layer_from or
            layer >= region.layer_to or
            (index > 0 and plan.layers[index - 1] >= layer))
        {
            return error.NonCanonicalSelectionLayers;
        }
    }
    if (plan.layers[0] != region.layer_from or
        plan.layers[plan.layers.len - 1] + 1 != region.layer_to)
    {
        return error.SelectionEnvelopeMismatch;
    }
    for (plan.timesteps, 0..) |timestep, index| {
        if (index > 0 and plan.timesteps[index - 1] >= timestep) {
            return error.NonCanonicalSelectionTimesteps;
        }
    }
}

fn validateCoarsen(plan: TokenCoarsen) !void {
    if (plan.source_tokens == 0 or plan.destination_tokens == 0) {
        return error.InvalidTokenCount;
    }
    if (plan.destination_tokens >= plan.source_tokens) return error.InvalidTokenCount;
    if (plan.region_count == 0) return error.InvalidRegionCount;
    if (plan.source_tokens % plan.region_count != 0) return error.InvalidRegionCount;
    if (plan.destination_tokens % plan.region_count != 0) {
        return error.InvalidRegionCount;
    }
    if (plan.destination_refresh == 0 or plan.assignment_refresh == 0) {
        return error.InvalidRefresh;
    }
    if (!std.math.isFinite(plan.assignment_scale) or plan.assignment_scale <= 0) {
        return error.InvalidAssignmentScale;
    }
    if (plan.algorithm_mode == .official_b578009 and
        (plan.selection_layout != .tile or plan.assignment_scope != .region_local or
            plan.unmerge != .official_raw_transpose))
    {
        return error.InvalidAlgorithmMode;
    }
    if (plan.algorithm_mode == .paper_spec and plan.unmerge == .official_raw_transpose) {
        return error.InvalidAlgorithmMode;
    }
    if (plan.selection_layout == .global and plan.region_count != 1) {
        return error.InvalidRegionCount;
    }
    if (plan.selection_layout == .tile) {
        const token_side = squareRoot(plan.source_tokens) orelse
            return error.InvalidTileGrid;
        const region_side = squareRoot(plan.region_count) orelse
            return error.InvalidTileGrid;
        if (token_side % region_side != 0) return error.InvalidTileGrid;
    }
}

fn lossy(representation: Representation) bool {
    return switch (representation) {
        .inherit, .f32 => false,
        .f16, .bf16, .w8, .w6, .w4 => true,
    };
}

fn squareRoot(value: u32) ?u32 {
    const root: u32 = @intFromFloat(@sqrt(@as(f64, @floatFromInt(value))));
    if (root * root == value) return root;
    return null;
}

fn isSha256(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |byte| {
        if (!std.ascii.isHex(byte)) return false;
    }
    return true;
}

test "parses and fingerprints a faithful token-coarsening plan" {
    const json =
        \\{
        \\  "schema_version": 1,
        \\  "model": "z-image-turbo",
        \\  "quality_contract": "faithful",
        \\  "regions": [{
        \\    "id": "image-middle",
        \\    "phase": "denoiser",
        \\    "layer_from": 12,
        \\    "layer_to": 24,
        \\    "representation": "f16",
        \\    "kernel": "metal",
        \\    "token_coarsen": {
        \\      "modality": "image",
        \\      "source_tokens": 4096,
        \\      "destination_tokens": 2048,
        \\      "selection_layout": "tile",
        \\      "assignment_scope": "global",
        \\      "region_count": 64,
        \\      "selection": "facility_location",
        \\      "algorithm_mode": "paper_spec",
        \\      "assignment": "column_softmax",
        \\      "unmerge": "paper_normalized_transpose",
        \\      "destination_refresh": 1,
        \\      "assignment_refresh": 1,
        \\      "assignment_scale": 1000
        \\    }
        \\  }]
        \\}
    ;
    var document = try parse(std.testing.allocator, json, .{});
    defer document.deinit();
    try std.testing.expectEqual(schema_version, document.plan().schema_version);
    try std.testing.expectEqual(@as(usize, 64), document.source_sha256.len);
    try std.testing.expectEqual(@as(u32, 2048), document.plan().regions[0]
        .token_coarsen.?.destination_tokens);
}

test "exact plans reject lossy representation and token coarsening" {
    const region = Region{
        .id = "bad",
        .phase = .denoiser,
        .layer_from = 0,
        .layer_to = 1,
        .representation = .f16,
    };
    try std.testing.expectError(error.LossyExactPlan, validate(.{
        .schema_version = schema_version,
        .model = "z-image-turbo",
        .quality_contract = .exact,
        .regions = &.{region},
    }, .{}));
}

test "tile coarsening requires square divisible grids" {
    const region = Region{
        .id = "bad-grid",
        .phase = .denoiser,
        .layer_from = 1,
        .layer_to = 2,
        .token_coarsen = .{
            .modality = .image,
            .source_tokens = 4096,
            .destination_tokens = 2048,
            .selection_layout = .tile,
            .assignment_scope = .global,
            .region_count = 63,
            .algorithm_mode = .paper_spec,
            .assignment_scale = 1000,
        },
    };
    try std.testing.expectError(error.InvalidRegionCount, validate(.{
        .schema_version = schema_version,
        .model = "z-image-turbo",
        .quality_contract = .faithful,
        .regions = &.{region},
    }, .{}));
}

test "tensor operations require an explicit hardware capability" {
    const region = Region{
        .id = "m5",
        .phase = .denoiser,
        .layer_from = 0,
        .layer_to = 1,
        .kernel = .tensor_ops,
    };
    const plan = ExecutionPlanV1{
        .schema_version = schema_version,
        .model = "flux2-klein-4b",
        .quality_contract = .faithful,
        .regions = &.{region},
    };
    try std.testing.expectError(error.UnsupportedKernel, validate(plan, .{}));
    try validate(plan, .{ .tensor_ops = true });
}

test "hook bypass requires explicit canonical timesteps and candidate identity" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const region = Region{
        .id = "searched-identity",
        .phase = .denoiser,
        .layer_from = 17,
        .layer_to = 19,
        .kernel = .metal,
        .hook_bypass = .{ .timesteps = &.{ 0, 1, 3 } },
    };
    try validate(.{
        .schema_version = schema_version,
        .model = "z-image-turbo",
        .quality_contract = .experimental,
        .source_candidate_sha256 = hash,
        .regions = &.{region},
    }, .{});
    var invalid = region;
    invalid.hook_bypass = .{ .timesteps = &.{ 1, 1 } };
    try std.testing.expectError(error.NonCanonicalBypassTimesteps, validate(.{
        .schema_version = schema_version,
        .model = "z-image-turbo",
        .quality_contract = .experimental,
        .regions = &.{invalid},
    }, .{}));
}

test "searched token selection binds semantics realization and deployment" {
    const candidate = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const program = "2fa6d407e70ae7630c9d6de825e52e5dbf973ee4eebc81b850bf507e7f2f12a5";
    const realization = "4471d3b66bc66b30e463042467bb9652ac366e608aeeb26e3381f8d5cdb852de";
    const source = "093f8d2d15867bbf5601abc18c9fc9d7d786a7d69445222cbe8804f510b70ac4";
    const region = Region{
        .id = "searched-token-selection",
        .phase = .denoiser,
        .layer_from = 0,
        .layer_to = 21,
        .kernel = .metal,
        .token_selection = .{
            .semantic_program_sha256 = program,
            .selection_program_sha256 = program,
            .realization_sha256 = realization,
            .schedule_sha256 = realization,
            .source_sha256 = source,
            .source_tokens = 64,
            .feature_width = 3840,
            .selected_tokens = 32,
            .layers = &.{ 0, 10, 20 },
            .timesteps = &.{ 0, 1, 2, 3 },
        },
    };
    try validate(.{
        .schema_version = schema_version,
        .model = "z-image-turbo",
        .quality_contract = .experimental,
        .source_candidate_sha256 = candidate,
        .source_capsule_sha256 = candidate,
        .regions = &.{region},
    }, .{});

    var invalid = region;
    invalid.token_selection.?.layers = &.{ 0, 20, 10 };
    try std.testing.expectError(error.NonCanonicalSelectionLayers, validate(.{
        .schema_version = schema_version,
        .model = "z-image-turbo",
        .quality_contract = .experimental,
        .source_candidate_sha256 = candidate,
        .source_capsule_sha256 = candidate,
        .regions = &.{invalid},
    }, .{}));
}
