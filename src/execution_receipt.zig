//! Deterministic evidence emitted after executing an optimization plan.
//!
//! Receipts report what actually ran. A requested route that fell back stays
//! visible so the private optimizer cannot learn from mislabeled experiments.

const std = @import("std");

const execution_plan = @import("execution_plan.zig");
const model_kind = @import("model_kind.zig");

pub const schema_version: u32 = 1;

pub const Status = enum {
    completed,
    rejected,
    failed,
};

pub const ThermalState = enum(i32) {
    unknown = -1,
    nominal = 0,
    fair = 1,
    serious = 2,
    critical = 3,
};

pub const Route = struct {
    region_id: []const u8,
    requested_kernel: execution_plan.Kernel,
    actual_kernel: execution_plan.Kernel,
    fallback: bool,
    executions_expected: u32,
    executions_completed: u32,
};

pub const Timing = struct {
    stage: []const u8,
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
    samples: u32,
};

pub const Memory = struct {
    peak_rss_bytes: u64,
    physical_footprint_bytes: u64,
    peak_gpu_live_bytes: u64,
    weight_bytes: u64,
};

pub const Counters = struct {
    dispatches: u64,
    command_buffers: u64,
    waits: u64,
    readbacks: u64,
    fallbacks: u64,
    gpu_active_ns: u64,
};

pub const Workload = struct {
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    prompt_sha256: ?[]const u8 = null,
};

pub const Device = struct {
    name: []const u8,
    ram_bytes: u64,
    os_major: u32,
    tensor_ops: bool,
};

/// The safety filter's outcome: prompt stage pass|blocked|off, image score
/// when the image stage ran, and the action taken.
pub const Safety = struct {
    prompt: []const u8,
    image: ?f64 = null,
    action: []const u8,
};

pub const Quality = struct {
    passed: bool,
    psnr: ?f64 = null,
    ssim: ?f64 = null,
    ms_ssim: ?f64 = null,
    edge_psnr: ?f64 = null,
    lpips: ?f64 = null,
    dists: ?f64 = null,
    dreamsim: ?f64 = null,
};

pub const ExecutionReceiptV1 = struct {
    schema_version: u32,
    plan_sha256: []const u8,
    engine_revision: []const u8,
    source_candidate_sha256: ?[]const u8 = null,
    source_capsule_sha256: ?[]const u8 = null,
    realization_sha256: ?[]const u8 = null,
    schedule_sha256: ?[]const u8 = null,
    source_artifact_sha256: ?[]const u8 = null,
    model: []const u8,
    quality_contract: execution_plan.QualityContract,
    status: Status,
    device: Device,
    workload: Workload,
    routes: []const Route,
    timings: []const Timing,
    memory: Memory,
    counters: Counters,
    thermal_valid: bool,
    thermal_before: ThermalState,
    thermal_after: ThermalState,
    output_sha256: ?[]const u8 = null,
    quality: ?Quality = null,
    safety: ?Safety = null,
    failure: ?[]const u8 = null,
};

pub fn validate(receipt: ExecutionReceiptV1) !void {
    if (receipt.schema_version != schema_version) return error.UnsupportedReceiptVersion;
    if (!isSha256(receipt.plan_sha256)) return error.InvalidPlanHash;
    if (receipt.output_sha256) |hash| {
        if (!isSha256(hash)) return error.InvalidOutputHash;
    }
    if (receipt.engine_revision.len == 0) return error.EmptyEngineRevision;
    if (receipt.source_candidate_sha256) |hash| {
        if (!isSha256(hash)) return error.InvalidCandidateHash;
    }
    try validateBinding(receipt);
    if (model_kind.parseKind(receipt.model) == null) return error.UnsupportedModel;
    if (receipt.device.name.len == 0) return error.EmptyDeviceName;
    if (receipt.workload.width == 0 or receipt.workload.height == 0) {
        return error.InvalidWorkload;
    }
    if (receipt.workload.steps == 0) return error.InvalidWorkload;
    if (receipt.workload.prompt_sha256) |hash| {
        if (!isSha256(hash)) return error.InvalidPromptHash;
    }
    try validateRoutes(receipt);
    try validateTimings(receipt);
    if (receipt.quality) |quality| try validateQuality(quality);
    const expected_thermal =
        receipt.thermal_before == .nominal and receipt.thermal_after == .nominal;
    if (receipt.thermal_valid != expected_thermal) {
        return error.InvalidThermalValidity;
    }
    if (receipt.status == .completed and receipt.failure != null) {
        return error.InconsistentStatus;
    }
    if (receipt.status != .completed and receipt.failure == null) {
        return error.InconsistentStatus;
    }
}

fn validateBinding(receipt: ExecutionReceiptV1) !void {
    inline for (.{
        receipt.source_capsule_sha256,
        receipt.realization_sha256,
        receipt.schedule_sha256,
        receipt.source_artifact_sha256,
    }) |optional_hash| {
        if (optional_hash) |hash| {
            if (!isSha256(hash)) return error.InvalidCapsuleBindingHash;
        }
    }
    const binding_count =
        @intFromBool(receipt.source_capsule_sha256 != null) +
        @intFromBool(receipt.realization_sha256 != null) +
        @intFromBool(receipt.schedule_sha256 != null) +
        @intFromBool(receipt.source_artifact_sha256 != null);
    if (binding_count != 0 and binding_count != 4) {
        return error.IncompleteCapsuleBinding;
    }
}

fn validateRoutes(receipt: ExecutionReceiptV1) !void {
    for (receipt.routes, 0..) |route, index| {
        if (route.region_id.len == 0) return error.EmptyRegionId;
        for (receipt.routes[0..index]) |prior| {
            if (std.mem.eql(u8, prior.region_id, route.region_id)) {
                return error.DuplicateRegionId;
            }
        }
        if (!route.fallback and route.requested_kernel != route.actual_kernel) {
            return error.UnreportedFallback;
        }
        if (route.executions_completed > route.executions_expected) {
            return error.InvalidExecutionCount;
        }
        if (receipt.status == .completed and
            route.executions_completed != route.executions_expected)
        {
            return error.IncompleteExecution;
        }
    }
}

fn validateTimings(receipt: ExecutionReceiptV1) !void {
    for (receipt.timings, 0..) |timing, index| {
        if (timing.stage.len == 0 or timing.samples == 0) return error.InvalidTiming;
        if (timing.min_ns > timing.median_ns or timing.median_ns > timing.max_ns) {
            return error.InvalidTiming;
        }
        for (receipt.timings[0..index]) |prior| {
            if (std.mem.eql(u8, prior.stage, timing.stage)) {
                return error.DuplicateTiming;
            }
        }
    }
}

pub fn toJson(
    allocator: std.mem.Allocator,
    receipt: ExecutionReceiptV1,
) ![]u8 {
    try validate(receipt);
    return std.json.Stringify.valueAlloc(allocator, receipt, .{
        .emit_null_optional_fields = false,
        .whitespace = .indent_2,
    });
}

fn validateQuality(quality: Quality) !void {
    inline for (@typeInfo(Quality).@"struct".fields) |field| {
        if (field.type != ?f64) continue;
        if (@field(quality, field.name)) |value| {
            if (!std.math.isFinite(value)) return error.InvalidQualityMetric;
        }
    }
}

fn isSha256(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |byte| {
        if (!std.ascii.isHex(byte)) return false;
    }
    return true;
}

test "serializes a validated receipt with explicit routes" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const routes = [_]Route{.{
        .region_id = "middle",
        .requested_kernel = .metal,
        .actual_kernel = .metal,
        .fallback = false,
        .executions_expected = 4,
        .executions_completed = 4,
    }};
    const timings = [_]Timing{.{
        .stage = "denoise",
        .median_ns = 20,
        .min_ns = 10,
        .max_ns = 30,
        .samples = 3,
    }};
    const receipt = ExecutionReceiptV1{
        .schema_version = schema_version,
        .plan_sha256 = hash,
        .engine_revision = "abc123",
        .model = "z-image-turbo",
        .quality_contract = .faithful,
        .status = .completed,
        .device = .{
            .name = "Apple M4 Max",
            .ram_bytes = 128 * 1024 * 1024 * 1024,
            .os_major = 26,
            .tensor_ops = false,
        },
        .workload = .{
            .width = 1024,
            .height = 1024,
            .steps = 4,
            .seed = 42,
            .prompt_sha256 = "0123456789abcdef0123456789abcdef" ++
                "0123456789abcdef0123456789abcdef",
        },
        .routes = &routes,
        .timings = &timings,
        .memory = .{
            .peak_rss_bytes = 9,
            .physical_footprint_bytes = 8,
            .peak_gpu_live_bytes = 7,
            .weight_bytes = 6,
        },
        .counters = .{
            .dispatches = 5,
            .command_buffers = 4,
            .waits = 3,
            .readbacks = 2,
            .fallbacks = 0,
            .gpu_active_ns = 1,
        },
        .thermal_valid = true,
        .thermal_before = .nominal,
        .thermal_after = .nominal,
        .output_sha256 = hash,
        .quality = .{ .passed = true, .psnr = 42.5, .ssim = 0.99 },
    };
    const json = try toJson(std.testing.allocator, receipt);
    defer std.testing.allocator.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    const object = parsed.value.object;
    try std.testing.expectEqualStrings("completed", object.get("status").?.string);
    try std.testing.expectEqualStrings("Apple M4 Max", object.get("device").?
        .object.get("name").?.string);
    try std.testing.expectEqualStrings(
        receipt.workload.prompt_sha256.?,
        object.get("workload").?.object.get("prompt_sha256").?.string,
    );
    try std.testing.expect(object.get("failure") == null);
}

test "rejects a route mismatch hidden as a non-fallback" {
    const route = Route{
        .region_id = "attention",
        .requested_kernel = .mfa,
        .actual_kernel = .metal,
        .fallback = false,
        .executions_expected = 1,
        .executions_completed = 1,
    };
    const receipt = minimalReceipt(&.{route});
    try std.testing.expectError(error.UnreportedFallback, validate(receipt));
}

test "failed receipts require a failure reason" {
    var receipt = minimalReceipt(&.{});
    receipt.status = .failed;
    try std.testing.expectError(error.InconsistentStatus, validate(receipt));
    receipt.failure = "compile failed";
    try validate(receipt);
}

fn minimalReceipt(routes: []const Route) ExecutionReceiptV1 {
    return .{
        .schema_version = schema_version,
        .plan_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .engine_revision = "test",
        .model = "z-image-turbo",
        .quality_contract = .faithful,
        .status = .completed,
        .device = .{
            .name = "test",
            .ram_bytes = 1,
            .os_major = 1,
            .tensor_ops = false,
        },
        .workload = .{ .width = 1, .height = 1, .steps = 1, .seed = 1 },
        .routes = routes,
        .timings = &.{},
        .memory = .{
            .peak_rss_bytes = 0,
            .physical_footprint_bytes = 0,
            .peak_gpu_live_bytes = 0,
            .weight_bytes = 0,
        },
        .counters = .{
            .dispatches = 0,
            .command_buffers = 0,
            .waits = 0,
            .readbacks = 0,
            .fallbacks = 0,
            .gpu_active_ns = 0,
        },
        .thermal_valid = true,
        .thermal_before = .nominal,
        .thermal_after = .nominal,
    };
}
