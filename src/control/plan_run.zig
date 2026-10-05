//! Honest CLI boundary for optimization plans and evidence receipts.
//!
//! A parsed plan is not considered executable merely because its schema is
//! valid. This module admits only routes the engine can currently honor, then
//! records the actual process counters and output artifact. New transformations
//! become searchable only after their executor is added here.

const std = @import("std");

const execution_plan = @import("execution_plan.zig");
const execution_receipt = @import("execution_receipt.zig");
const safety = @import("../cli/safety.zig");
const version = @import("../cli/version.zig");
const hook_bypass = @import("hook_bypass_control.zig");
const image = @import("../cli/image.zig");
const metal_c = @import("../metal/metal_c.zig");
const metrics = @import("../metal/metrics.zig");
const model_kind = @import("../cli/model_kind.zig");
const profile = @import("../runtime/profile.zig");
const token_selection = @import("token_selection_control.zig");
const toma_config = @import("toma_config.zig");
const toma_control = @import("toma_control.zig");
const zseq = @import("../zimage/zseq.zig");

const max_plan_bytes = 1024 * 1024;

pub const Request = struct {
    kind: model_kind.ModelKind,
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    repeat: u32,
    has_seed_batch: bool,
    output_path: []const u8,
    plan_path: []const u8,
    receipt_path: []const u8,
    safety: bool = true,
};

pub const Run = struct {
    document: ?execution_plan.Document = null,
    request: Request,
    prompt_sha256: [64]u8,
    control_active: bool = false,
    bypass_active: bool = false,
    token_selection_active: bool = false,
    thermal_before: execution_receipt.ThermalState = .unknown,

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        request: Request,
    ) !Run {
        var prompt_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(request.prompt, &prompt_digest, .{});
        const prompt_sha256 = std.fmt.bytesToHex(prompt_digest, .lower);
        if (request.plan_path.len == 0) {
            return .{
                .request = request,
                .prompt_sha256 = prompt_sha256,
            };
        }

        const bytes = try std.Io.Dir.cwd().readFileAlloc(
            io,
            request.plan_path,
            allocator,
            .limited(max_plan_bytes),
        );
        defer allocator.free(bytes);
        const device = profile.detect();
        var run = Run{
            .document = try execution_plan.parse(
                allocator,
                bytes,
                capabilities(device),
            ),
            .request = request,
            .prompt_sha256 = prompt_sha256,
            .thermal_before = thermalState(),
        };
        errdefer run.deinit();
        validateRequest(run.document.?.plan().*, request) catch |err| {
            run.write(io, allocator, .rejected, 0, null, err) catch {};
            return err;
        };
        const control = try controlFor(run.document.?.plan().*, request);
        try installControl(control);
        run.control_active = true;
        run.bypass_active = switch (control) {
            .bypass => true,
            else => false,
        };
        run.token_selection_active = switch (control) {
            .token_selection => true,
            else => false,
        };
        return run;
    }

    pub fn deinit(self: *Run) void {
        if (self.token_selection_active) token_selection.deactivate();
        if (self.bypass_active) hook_bypass.deactivate();
        if (self.control_active) toma_control.deactivate();
        if (self.document) |*document| document.deinit();
        self.* = undefined;
    }

    pub fn completed(
        self: *const Run,
        io: std.Io,
        allocator: std.mem.Allocator,
        elapsed_ns: u64,
    ) !void {
        toma_control.verifyCompleted() catch |err| {
            try self.write(io, allocator, .failed, elapsed_ns, null, err);
            return err;
        };
        hook_bypass.verifyCompleted() catch |err| {
            try self.write(io, allocator, .failed, elapsed_ns, null, err);
            return err;
        };
        token_selection.verifyCompleted() catch |err| {
            try self.write(io, allocator, .failed, elapsed_ns, null, err);
            return err;
        };
        try self.write(io, allocator, .completed, elapsed_ns, self.request.output_path, null);
    }

    pub fn failed(
        self: *const Run,
        io: std.Io,
        allocator: std.mem.Allocator,
        elapsed_ns: u64,
        failure: anyerror,
    ) void {
        self.write(io, allocator, .failed, elapsed_ns, null, failure) catch {};
    }

    fn write(
        self: *const Run,
        io: std.Io,
        allocator: std.mem.Allocator,
        status: execution_receipt.Status,
        elapsed_ns: u64,
        output_path: ?[]const u8,
        failure: ?anyerror,
    ) !void {
        const document = if (self.document) |*value| value else return;
        var output_hash_storage: [64]u8 = undefined;
        const output_hash: ?[]const u8 = if (output_path) |path| hash: {
            output_hash_storage = try hashFile(io, path);
            break :hash &output_hash_storage;
        } else null;
        var timing_storage: [metrics.max_timing_labels + 1]execution_receipt.Timing = undefined;
        var timing_count: usize = 0;
        while (metrics.timingSummary(timing_count)) |summary| : (timing_count += 1) {
            timing_storage[timing_count] = .{
                .stage = summary.stage,
                .median_ns = summary.median_ns,
                .min_ns = summary.min_ns,
                .max_ns = summary.max_ns,
                .samples = summary.samples,
            };
        }
        if (elapsed_ns > 0) {
            timing_storage[timing_count] = .{
                .stage = "command_total",
                .median_ns = elapsed_ns,
                .min_ns = elapsed_ns,
                .max_ns = elapsed_ns,
                .samples = 1,
            };
            timing_count += 1;
        }
        const timings = timing_storage[0..timing_count];
        const failure_name: ?[]const u8 = if (failure) |err| @errorName(err) else null;
        var route_storage: [1]execution_receipt.Route = undefined;
        const routes = makeRoutes(document.plan().*, status, &route_storage);
        const receipt = makeReceipt(
            self,
            document,
            status,
            routes,
            timings,
            output_hash,
            failure_name,
        );
        const json = try execution_receipt.toJson(allocator, receipt);
        defer allocator.free(json);
        try image.ensureParent(io, self.request.receipt_path);
        const file = try std.Io.Dir.cwd().createFile(io, self.request.receipt_path, .{});
        defer file.close(io);
        var buffer: [4096]u8 = undefined;
        var writer = file.writerStreaming(io, &buffer);
        try writer.interface.writeAll(json);
        try writer.interface.writeByte('\n');
        try writer.interface.flush();
    }
};

/// The prompt stage ran (a blocked prompt never reaches a receipt) or was off.
fn safetyField(on: bool) execution_receipt.Safety {
    const word: []const u8 = if (on) "pass" else "off";
    return .{ .prompt = word, .image = safety.last_image_score, .action = word };
}

fn makeReceipt(
    run: *const Run,
    document: *const execution_plan.Document,
    status: execution_receipt.Status,
    routes: []const execution_receipt.Route,
    timings: []const execution_receipt.Timing,
    output_hash: ?[]const u8,
    failure_name: ?[]const u8,
) execution_receipt.ExecutionReceiptV1 {
    const device = profile.detect();
    const memory = metrics.memory();
    const counters = metrics.snapshot();
    const plan = document.plan();
    const binding = capsuleBinding(plan.*);
    const thermal_after = thermalState();
    return .{
        .schema_version = execution_receipt.schema_version,
        .plan_sha256 = &document.source_sha256,
        .engine_revision = engineRevision(),
        .source_candidate_sha256 = plan.source_candidate_sha256,
        .source_capsule_sha256 = binding.source_capsule_sha256,
        .realization_sha256 = binding.realization_sha256,
        .schedule_sha256 = binding.schedule_sha256,
        .source_artifact_sha256 = binding.source_artifact_sha256,
        .model = plan.model,
        .quality_contract = plan.quality_contract,
        .status = status,
        .device = receiptDevice(device),
        .workload = .{
            .width = run.request.width,
            .height = run.request.height,
            .steps = run.request.steps,
            .seed = run.request.seed,
            .prompt_sha256 = &run.prompt_sha256,
        },
        .routes = routes,
        .timings = timings,
        .memory = .{
            .peak_rss_bytes = memory.peak_rss_bytes,
            .physical_footprint_bytes = memory.phys_footprint_bytes,
            .peak_gpu_live_bytes = memory.peak_gpu_live_bytes,
            .weight_bytes = memory.weight_bytes,
        },
        .counters = .{
            .dispatches = counters.dispatch,
            .command_buffers = counters.command,
            .waits = metal_c.zdraw_metal_wait_count(),
            .readbacks = counters.readback,
            .fallbacks = counters.mps_fallback + counters.steel_fallback + counters.mpp_fallback,
            .gpu_active_ns = counters.gpu_ns,
        },
        .thermal_valid = thermalValid(run.thermal_before, thermal_after),
        .thermal_before = run.thermal_before,
        .thermal_after = thermal_after,
        .output_sha256 = output_hash,
        .safety = safetyField(run.request.safety),
        .failure = failure_name,
    };
}

const CapsuleBinding = struct {
    source_capsule_sha256: ?[]const u8 = null,
    realization_sha256: ?[]const u8 = null,
    schedule_sha256: ?[]const u8 = null,
    source_artifact_sha256: ?[]const u8 = null,
};

fn capsuleBinding(plan: execution_plan.ExecutionPlanV1) CapsuleBinding {
    if (plan.regions.len != 1) return .{};
    const selection = plan.regions[0].token_selection orelse return .{};
    return .{
        .source_capsule_sha256 = plan.source_capsule_sha256,
        .realization_sha256 = selection.realization_sha256,
        .schedule_sha256 = selection.schedule_sha256,
        .source_artifact_sha256 = selection.source_sha256,
    };
}

pub fn thermalState() execution_receipt.ThermalState {
    return std.enums.fromInt(
        execution_receipt.ThermalState,
        metal_c.zdraw_thermal_state(),
    ) orelse .unknown;
}

fn thermalValid(
    before: execution_receipt.ThermalState,
    after: execution_receipt.ThermalState,
) bool {
    return before == .nominal and after == .nominal;
}

fn validateRequest(plan: execution_plan.ExecutionPlanV1, request: Request) !void {
    if (!std.mem.eql(u8, plan.model, model_kind.cliName(request.kind))) {
        return error.PlanModelMismatch;
    }
    if (plan.quality_contract == .exact) {
        return error.UnsupportedQualityContract;
    }
    if (request.repeat != 1 or request.has_seed_batch) {
        return error.UnsupportedPlanWorkload;
    }
    if (plan.memory.cross_phase_reuse or plan.memory.max_peak_bytes != 0) {
        return error.UnsupportedPlanMemoryPolicy;
    }
    _ = try controlFor(plan, request);
}

const Control = union(enum) {
    baseline,
    toma: toma_control.Plan,
    bypass: hook_bypass.Plan,
    token_selection: token_selection.Plan,
};

fn installControl(control: Control) !void {
    switch (control) {
        .baseline => try toma_control.startBaseline(0),
        .toma => |plan| try toma_control.activateToma(plan),
        .bypass => |plan| {
            try toma_control.startBaseline(plan.expected_steps);
            errdefer toma_control.deactivate();
            try hook_bypass.activate(plan);
        },
        .token_selection => |plan| {
            try toma_control.startBaseline(plan.expected_steps);
            errdefer toma_control.deactivate();
            try token_selection.activate(plan);
        },
    }
}

fn controlFor(
    plan: execution_plan.ExecutionPlanV1,
    request: Request,
) !Control {
    if (plan.regions.len == 0) return .baseline;
    if (plan.regions.len != 1) return error.UnsupportedPlanTransform;
    if (request.kind != .z_image_turbo) return error.UnsupportedPlanTransform;
    const region = plan.regions[0];
    if (region.phase != .denoiser or region.representation != .inherit or
        region.kernel != .metal)
    {
        return error.UnsupportedPlanTransform;
    }
    if (region.hook_bypass) |bypass| {
        return bypassControl(region, bypass, request.steps);
    }
    if (region.token_selection) |selection| {
        return selectControl(selection, request.steps);
    }
    const coarsen = region.token_coarsen orelse return error.UnsupportedPlanTransform;
    return tomaControl(region, coarsen, request);
}

fn bypassControl(
    region: execution_plan.Region,
    bypass: execution_plan.HookBypass,
    steps: u32,
) !Control {
    for (bypass.timesteps) |timestep| {
        if (timestep >= steps) return error.PlanTimestepOutOfRange;
    }
    return .{ .bypass = .{
        .layer_from = region.layer_from,
        .layer_to = region.layer_to,
        .timesteps = bypass.timesteps,
        .expected_steps = steps,
    } };
}

fn selectControl(
    selection: execution_plan.TokenSelection,
    steps: u32,
) !Control {
    if (selection.feature_width != 3840) return error.PlanFeatureWidthMismatch;
    for (selection.timesteps) |timestep| {
        if (timestep >= steps) return error.PlanTimestepOutOfRange;
    }
    return .{ .token_selection = .{
        .semantic_program_sha256 = selection.semantic_program_sha256,
        .selection_program_sha256 = selection.selection_program_sha256,
        .realization_sha256 = selection.realization_sha256,
        .schedule_sha256 = selection.schedule_sha256,
        .source_sha256 = selection.source_sha256,
        .source_tokens = selection.source_tokens,
        .feature_width = selection.feature_width,
        .selected_tokens = selection.selected_tokens,
        .layers = selection.layers,
        .timesteps = selection.timesteps,
        .expected_steps = steps,
        .score = switch (selection.score) {
            .feature_mean => .feature_mean,
            .zero => .zero,
        },
        .reconstruction = switch (selection.reconstruction) {
            .reference_scatter => .reference_scatter,
            .cosine_residual_interpolate => .cosine_residual_interpolate,
        },
        .reconstruction_weight = selection.reconstruction_weight,
    } };
}

fn tomaControl(
    region: execution_plan.Region,
    coarsen: execution_plan.TokenCoarsen,
    request: Request,
) !Control {
    if (coarsen.modality != .image or coarsen.selection_layout != .tile or
        coarsen.selection != .facility_location or
        coarsen.assignment != .column_softmax)
    {
        return error.UnsupportedPlanTransform;
    }
    if (coarsen.destination_refresh != 1 or coarsen.assignment_refresh != 1) {
        return error.UnsupportedPlanRefresh;
    }
    const expected_tokens = try imageTokens(request.width, request.height);
    if (coarsen.source_tokens != expected_tokens) return error.PlanTokenCountMismatch;
    const mode: toma_config.Mode = switch (coarsen.algorithm_mode) {
        .paper_spec => .paper_spec,
        .official_b578009 => .official_b578009,
    };
    if (mode == .paper_spec and coarsen.unmerge != .paper_normalized_transpose) {
        return error.UnsupportedPlanTransform;
    }
    return .{ .toma = .{
        .source_tokens = coarsen.source_tokens,
        .layer_from = region.layer_from,
        .layer_to = region.layer_to,
        .config = .{
            .mode = mode,
            .destination_tokens = coarsen.destination_tokens,
            .region_count = coarsen.region_count,
            .selection_layout = coarsen.selection_layout,
            .assignment_scope = coarsen.assignment_scope,
            .unmerge = coarsen.unmerge,
            .assignment_scale = coarsen.assignment_scale,
        },
        .expected_steps = request.steps,
    } };
}

fn imageTokens(width: u32, height: u32) !u32 {
    if (width == 0 or height == 0 or width % 16 != 0 or height % 16 != 0) {
        return error.InvalidImageSize;
    }
    const raw = @as(usize, width / 16) * @as(usize, height / 16);
    return std.math.cast(u32, zseq.paddedLen(raw)) orelse error.InvalidImageSize;
}

fn makeRoutes(
    plan: execution_plan.ExecutionPlanV1,
    status: execution_receipt.Status,
    storage: *[1]execution_receipt.Route,
) []const execution_receipt.Route {
    if (plan.regions.len == 0) return &.{};
    const expected, const completed = if (plan.regions[0].hook_bypass != null) counts: {
        const counts = hook_bypass.counts();
        break :counts .{ counts.expected, counts.completed };
    } else if (plan.regions[0].token_selection != null) counts: {
        const counts = token_selection.counts();
        break :counts .{ counts.expected, counts.completed };
    } else counts: {
        const counts = toma_control.counts();
        break :counts .{ counts.expected, counts.completed };
    };
    const ran = completed > 0;
    const complete = status == .completed and completed == expected;
    storage[0] = .{
        .region_id = plan.regions[0].id,
        .requested_kernel = plan.regions[0].kernel,
        .actual_kernel = if (ran) .metal else .inherit,
        .fallback = !complete,
        .executions_expected = expected,
        .executions_completed = completed,
    };
    return storage;
}

fn capabilities(device: ?profile.Device) execution_plan.Capabilities {
    return .{
        .mps = true,
        .tensor_ops = if (device) |value|
            std.mem.indexOf(u8, value.chip(), "M5") != null
        else
            false,
    };
}

fn receiptDevice(device: ?profile.Device) execution_receipt.Device {
    if (device) |value| {
        return .{
            .name = value.chip(),
            .ram_bytes = value.ram_bytes,
            .os_major = value.os_major,
            .tensor_ops = capabilities(value).tensor_ops,
        };
    }
    return .{
        .name = "unknown-device",
        .ram_bytes = 0,
        .os_major = 0,
        .tensor_ops = false,
    };
}

fn engineRevision() []const u8 {
    const raw = std.c.getenv("ZDRAW_ENGINE_REVISION") orelse return version.revision;
    const value = std.mem.span(raw);
    return if (value.len > 0) value else version.revision;
}

pub fn hashFile(io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var read_buffer: [64 * 1024]u8 = undefined;
    var chunk: [64 * 1024]u8 = undefined;
    var reader = file.readerStreaming(io, &read_buffer);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    while (true) {
        const count = try reader.interface.readSliceShort(&chunk);
        if (count == 0) break;
        hasher.update(chunk[0..count]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

test "baseline and one faithful Z-Image ToMA region are executable" {
    const baseline = execution_plan.ExecutionPlanV1{
        .schema_version = execution_plan.schema_version,
        .model = "z-image-turbo",
        .quality_contract = .faithful,
    };
    const request = Request{
        .kind = .z_image_turbo,
        .prompt = "test prompt",
        .width = 1024,
        .height = 1024,
        .steps = 4,
        .seed = 42,
        .repeat = 1,
        .has_seed_batch = false,
        .output_path = "image.png",
        .plan_path = "plan.json",
        .receipt_path = "receipt.json",
    };
    try validateRequest(baseline, request);

    const coarsen = execution_plan.TokenCoarsen{
        .modality = .image,
        .source_tokens = 4096,
        .destination_tokens = 2048,
        .selection_layout = .tile,
        .assignment_scope = .region_local,
        .region_count = 64,
        .algorithm_mode = .paper_spec,
        .assignment_scale = 1000,
    };
    var transformed = baseline;
    transformed.regions = &.{.{
        .id = "middle",
        .phase = .denoiser,
        .layer_from = 12,
        .layer_to = 24,
        .kernel = .metal,
        .token_coarsen = coarsen,
    }};
    const control = try controlFor(transformed, request);
    try std.testing.expectEqual(@as(usize, 12), control.toma.layer_from);
    try std.testing.expectEqual(@as(usize, 2048), control.toma.config.destination_tokens);
    try std.testing.expectEqual(@as(u32, 4), control.toma.expected_steps);

    var unsupported = transformed;
    var stale = coarsen;
    stale.destination_refresh = 2;
    unsupported.regions = &.{.{
        .id = "stale",
        .phase = .denoiser,
        .layer_from = 12,
        .layer_to = 24,
        .kernel = .metal,
        .token_coarsen = stale,
    }};
    try std.testing.expectError(error.UnsupportedPlanRefresh, controlFor(unsupported, request));
}

test "searched hook bypass lowers to a scoped resident control" {
    const plan = execution_plan.ExecutionPlanV1{
        .schema_version = execution_plan.schema_version,
        .model = "z-image-turbo",
        .quality_contract = .experimental,
        .regions = &.{.{
            .id = "searched-identity",
            .phase = .denoiser,
            .layer_from = 17,
            .layer_to = 19,
            .kernel = .metal,
            .hook_bypass = .{ .timesteps = &.{ 0, 1, 3 } },
        }},
    };
    const request = Request{
        .kind = .z_image_turbo,
        .prompt = "test prompt",
        .width = 1024,
        .height = 1024,
        .steps = 4,
        .seed = 42,
        .repeat = 1,
        .has_seed_batch = false,
        .output_path = "image.png",
        .plan_path = "plan.json",
        .receipt_path = "receipt.json",
    };
    const control = try controlFor(plan, request);
    try std.testing.expectEqual(@as(usize, 17), control.bypass.layer_from);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3 }, control.bypass.timesteps);
}

test "searched token selection lowers to a capsule-bound resident control" {
    const candidate = "40928d655ee34f44d5d378636113a321dc55f5140550fd0c5eae6c61d87fe159";
    const semantic = "2fa6d407e70ae7630c9d6de825e52e5d" ++
        "bf973ee4eebc81b850bf507e7f2f12a5";
    const selection = "430e47182361505a741cd8456efd9e41c" ++
        "9d908b49262370954a2f0041a79e8f0";
    const realization = "7bd3a8bbf25bb1fef961dd46f6dcb95a" ++
        "b10a7d846f44f06863e1b1a802a0af22";
    const plan = execution_plan.ExecutionPlanV1{
        .schema_version = execution_plan.schema_version,
        .model = "z-image-turbo",
        .quality_contract = .experimental,
        .source_candidate_sha256 = candidate,
        .source_capsule_sha256 = candidate,
        .regions = &.{.{
            .id = "searched-token-selection",
            .phase = .denoiser,
            .layer_from = 0,
            .layer_to = 21,
            .kernel = .metal,
            .token_selection = .{
                .semantic_program_sha256 = semantic,
                .selection_program_sha256 = selection,
                .realization_sha256 = realization,
                .schedule_sha256 = "0c9132da9f1a57a8afe8e9666c6f6535fe2253935ff111f4be712d91e1cab68b",
                .source_sha256 = "53a8df0625ce3e033706d549a413aa67f1a67d8a3f0b85ea57519b14e08aae55",
                .source_tokens = 64,
                .feature_width = 3840,
                .selected_tokens = 32,
                .layers = &.{ 0, 10, 20 },
                .timesteps = &.{ 0, 1, 2, 3 },
            },
        }},
    };
    const request = Request{
        .kind = .z_image_turbo,
        .prompt = "test prompt",
        .width = 64,
        .height = 64,
        .steps = 4,
        .seed = 42,
        .repeat = 1,
        .has_seed_batch = false,
        .output_path = "image.png",
        .plan_path = "plan.json",
        .receipt_path = "receipt.json",
    };
    const control = try controlFor(plan, request);
    try std.testing.expectEqual(@as(usize, 64), control.token_selection.source_tokens);
    try std.testing.expectEqual(@as(usize, 32), control.token_selection.selected_tokens);
    try std.testing.expectEqualSlices(u32, &.{ 0, 10, 20 }, control.token_selection.layers);
}

test "plan request rejects ambiguous workloads and model mismatches" {
    const baseline = execution_plan.ExecutionPlanV1{
        .schema_version = execution_plan.schema_version,
        .model = "z-image-turbo",
        .quality_contract = .faithful,
    };
    var request = Request{
        .kind = .z_image_turbo,
        .prompt = "test prompt",
        .width = 1024,
        .height = 1024,
        .steps = 4,
        .seed = 42,
        .repeat = 2,
        .has_seed_batch = false,
        .output_path = "image.png",
        .plan_path = "plan.json",
        .receipt_path = "receipt.json",
    };
    try std.testing.expectError(
        error.UnsupportedPlanWorkload,
        validateRequest(baseline, request),
    );
    request.repeat = 1;
    request.kind = .flux2_klein_4b;
    try std.testing.expectError(
        error.PlanModelMismatch,
        validateRequest(baseline, request),
    );
}

test "thermal evidence is valid only when both samples are nominal" {
    try std.testing.expect(thermalValid(.nominal, .nominal));
    try std.testing.expect(!thermalValid(.nominal, .fair));
    try std.testing.expect(!thermalValid(.unknown, .nominal));
}
