//! Immutable schema for offline semantic traces used by the discovery system.
//!
//! Trace runs are deliberately ineligible for performance claims: layer
//! capture forces Z-Image through its synchronized, CPU-visible path.

const std = @import("std");

pub const schema_version_v1: u32 = 1;
pub const schema_version_v2: u32 = 2;
pub const schema_version: u32 = 3;

pub const CaptureMode = enum {
    non_resident_layer_probe,
};

pub const FileArtifact = struct {
    path: []const u8,
    sha256: []const u8,
    byte_count: u64,
};

pub const TraceStepV1 = struct {
    index: u32,
    normalized_time: f32,
    layer_input: FileArtifact,
    layer_outputs: FileArtifact,
};

pub const TraceStep = struct {
    index: u32,
    normalized_time: f32,
    layer_input: FileArtifact,
    layer_outputs: FileArtifact,
    adaln: FileArtifact,
};

pub const TensorSpec = struct {
    dtype: []const u8,
    layer_input_shape: [2]usize,
    layer_output_shape: [3]usize,
};

pub const ReplaySpec = struct {
    adaln_dtype: []const u8,
    adaln_shape: [1]usize,
    position_dtype: []const u8,
    position_shape: [2]usize,
    positions: FileArtifact,
};

pub const RuntimeSetting = struct {
    name: []const u8,
    value: []const u8,
};

pub const Workload = struct {
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    prompt_sha256: []const u8,
};

pub const ArtifactV1 = struct {
    schema_version: u32,
    subject: []const u8,
    model: []const u8,
    model_revision: []const u8,
    engine_revision: []const u8,
    hook: []const u8,
    capture_mode: CaptureMode,
    performance_eligible: bool,
    workload: Workload,
    tensor: TensorSpec,
    trace_steps: []const TraceStepV1,
};

pub const ArtifactV2 = struct {
    schema_version: u32,
    subject: []const u8,
    model: []const u8,
    model_revision: []const u8,
    engine_revision: []const u8,
    hook: []const u8,
    capture_mode: CaptureMode,
    performance_eligible: bool,
    workload: Workload,
    tensor: TensorSpec,
    replay: ReplaySpec,
    trace_steps: []const TraceStep,
};

pub const ArtifactV3 = struct {
    schema_version: u32,
    subject: []const u8,
    model: []const u8,
    model_revision: []const u8,
    engine_revision: []const u8,
    hook: []const u8,
    capture_mode: CaptureMode,
    performance_eligible: bool,
    runtime: []const RuntimeSetting,
    workload: Workload,
    tensor: TensorSpec,
    replay: ReplaySpec,
    trace_steps: []const TraceStep,
};

pub fn validateV1(artifact: ArtifactV1) !void {
    if (artifact.schema_version != schema_version_v1) {
        return error.UnsupportedTraceVersion;
    }
    try checkIdentity(
        artifact.subject,
        artifact.model,
        artifact.model_revision,
        artifact.engine_revision,
        artifact.hook,
        artifact.performance_eligible,
    );
    try checkWorkload(artifact.workload, artifact.trace_steps.len);
    const sizes = try tensorSizes(artifact.tensor);
    for (artifact.trace_steps, 0..) |step, index| {
        if (step.index != index or !std.math.isFinite(step.normalized_time)) {
            return error.InvalidTraceStep;
        }
        try validateFile(step.layer_input, sizes.input);
        try validateFile(step.layer_outputs, sizes.output);
    }
}

pub fn validate(artifact: ArtifactV2) !void {
    if (artifact.schema_version != schema_version_v2) {
        return error.UnsupportedTraceVersion;
    }
    try checkIdentity(
        artifact.subject,
        artifact.model,
        artifact.model_revision,
        artifact.engine_revision,
        artifact.hook,
        artifact.performance_eligible,
    );
    try checkWorkload(artifact.workload, artifact.trace_steps.len);
    const sizes = try tensorSizes(artifact.tensor);
    if (!std.mem.eql(u8, artifact.replay.adaln_dtype, "f32-le") or
        !std.mem.eql(u8, artifact.replay.position_dtype, "u32-le"))
    {
        return error.UnsupportedTraceDtype;
    }
    if (artifact.replay.position_shape[0] != artifact.tensor.layer_input_shape[0] or
        artifact.replay.position_shape[1] != 3)
    {
        return error.InvalidTraceShape;
    }
    const adaln_bytes = try bytesFor(f32, &artifact.replay.adaln_shape);
    const position_bytes = try bytesFor(u32, &artifact.replay.position_shape);
    try validateFile(artifact.replay.positions, position_bytes);
    for (artifact.trace_steps, 0..) |step, index| {
        if (step.index != index or !std.math.isFinite(step.normalized_time)) {
            return error.InvalidTraceStep;
        }
        try validateFile(step.layer_input, sizes.input);
        try validateFile(step.layer_outputs, sizes.output);
        try validateFile(step.adaln, adaln_bytes);
    }
}

pub fn validateV3(artifact: ArtifactV3) !void {
    if (artifact.schema_version != schema_version) {
        return error.UnsupportedTraceVersion;
    }
    try validateRuntime(artifact.runtime);
    try validate(.{
        .schema_version = schema_version_v2,
        .subject = artifact.subject,
        .model = artifact.model,
        .model_revision = artifact.model_revision,
        .engine_revision = artifact.engine_revision,
        .hook = artifact.hook,
        .capture_mode = artifact.capture_mode,
        .performance_eligible = artifact.performance_eligible,
        .workload = artifact.workload,
        .tensor = artifact.tensor,
        .replay = artifact.replay,
        .trace_steps = artifact.trace_steps,
    });
}

fn validateRuntime(runtime: []const RuntimeSetting) !void {
    if (runtime.len == 0) return error.EmptyTraceRuntime;
    for (runtime, 0..) |setting, index| {
        if (setting.name.len == 0 or
            !std.mem.startsWith(u8, setting.name, "ZDRAW_"))
        {
            return error.InvalidTraceRuntime;
        }
        for (runtime[0..index]) |previous| {
            if (std.mem.eql(u8, previous.name, setting.name)) {
                return error.DuplicateTraceRuntimeSetting;
            }
        }
    }
}

const TensorSizes = struct {
    input: u64,
    output: u64,
};

fn checkIdentity(
    subject: []const u8,
    model: []const u8,
    model_revision: []const u8,
    engine_revision: []const u8,
    hook: []const u8,
    performance_eligible: bool,
) !void {
    if (subject.len == 0 or
        model.len == 0 or
        model_revision.len == 0 or
        engine_revision.len == 0 or
        hook.len == 0)
    {
        return error.EmptyTraceIdentity;
    }
    if (performance_eligible) return error.TraceCannotMeasurePerformance;
}

fn checkWorkload(workload: Workload, trace_steps: usize) !void {
    if (workload.width == 0 or
        workload.height == 0 or
        workload.steps == 0 or
        trace_steps != workload.steps)
    {
        return error.InvalidTraceWorkload;
    }
    if (!isSha256(workload.prompt_sha256)) return error.InvalidTraceHash;
}

fn tensorSizes(tensor: TensorSpec) !TensorSizes {
    if (!std.mem.eql(u8, tensor.dtype, "f32-le")) {
        return error.UnsupportedTraceDtype;
    }
    const input_elems = try product(&tensor.layer_input_shape);
    const output_elems = try product(&tensor.layer_output_shape);
    const input_bytes = try std.math.mul(u64, input_elems, @sizeOf(f32));
    const output_bytes = try std.math.mul(u64, output_elems, @sizeOf(f32));
    if (tensor.layer_input_shape[0] != tensor.layer_output_shape[1] or
        tensor.layer_input_shape[1] != tensor.layer_output_shape[2])
    {
        return error.InvalidTraceShape;
    }
    return .{ .input = input_bytes, .output = output_bytes };
}

pub fn toJsonV1(allocator: std.mem.Allocator, artifact: ArtifactV1) ![]u8 {
    try validateV1(artifact);
    return std.json.Stringify.valueAlloc(allocator, artifact, .{
        .whitespace = .indent_2,
    });
}

pub fn toJson(allocator: std.mem.Allocator, artifact: ArtifactV2) ![]u8 {
    try validate(artifact);
    return std.json.Stringify.valueAlloc(allocator, artifact, .{
        .whitespace = .indent_2,
    });
}

pub fn toJsonV3(allocator: std.mem.Allocator, artifact: ArtifactV3) ![]u8 {
    try validateV3(artifact);
    return std.json.Stringify.valueAlloc(allocator, artifact, .{
        .whitespace = .indent_2,
    });
}

fn bytesFor(comptime T: type, shape: []const usize) !u64 {
    return std.math.mul(u64, try product(shape), @sizeOf(T));
}

fn validateFile(file: FileArtifact, expected_bytes: u64) !void {
    if (file.path.len == 0 or file.path[0] == '/' or
        std.mem.indexOf(u8, file.path, "..") != null)
    {
        return error.InvalidTracePath;
    }
    if (!isSha256(file.sha256)) return error.InvalidTraceHash;
    if (file.byte_count != expected_bytes) return error.InvalidTraceByteCount;
}

fn product(values: []const usize) !u64 {
    var result: u64 = 1;
    for (values) |value| {
        if (value == 0) return error.InvalidTraceShape;
        result = try std.math.mul(u64, result, value);
    }
    return result;
}

fn isSha256(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |byte| {
        if (!std.ascii.isHex(byte)) return false;
    }
    return true;
}

test "semantic trace validates and serializes" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const steps = [_]TraceStepV1{.{
        .index = 0,
        .normalized_time = 1.0,
        .layer_input = .{
            .path = "step-000-input.f32le",
            .sha256 = hash,
            .byte_count = 32,
        },
        .layer_outputs = .{
            .path = "step-000-layers.f32le",
            .sha256 = hash,
            .byte_count = 64,
        },
    }};
    const artifact = ArtifactV1{
        .schema_version = schema_version_v1,
        .subject = "zdraw",
        .model = "z-image-turbo",
        .model_revision = "model-hash",
        .engine_revision = "engine-hash",
        .hook = "transformer.layer",
        .capture_mode = .non_resident_layer_probe,
        .performance_eligible = false,
        .workload = .{
            .width = 64,
            .height = 64,
            .steps = 1,
            .seed = 42,
            .prompt_sha256 = hash,
        },
        .tensor = .{
            .dtype = "f32-le",
            .layer_input_shape = .{ 4, 2 },
            .layer_output_shape = .{ 2, 4, 2 },
        },
        .trace_steps = &steps,
    };
    const json = try toJsonV1(std.testing.allocator, artifact);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"performance_eligible\": false") != null);
}

test "trace cannot claim performance evidence" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const artifact = ArtifactV1{
        .schema_version = schema_version_v1,
        .subject = "zdraw",
        .model = "z-image-turbo",
        .model_revision = "model-hash",
        .engine_revision = "engine-hash",
        .hook = "transformer.layer",
        .capture_mode = .non_resident_layer_probe,
        .performance_eligible = true,
        .workload = .{
            .width = 64,
            .height = 64,
            .steps = 1,
            .seed = 42,
            .prompt_sha256 = hash,
        },
        .tensor = .{
            .dtype = "f32-le",
            .layer_input_shape = .{ 4, 2 },
            .layer_output_shape = .{ 2, 4, 2 },
        },
        .trace_steps = &.{},
    };
    try std.testing.expectError(error.TraceCannotMeasurePerformance, validateV1(artifact));
}

test "replay trace binds adaln and portable positions" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const steps = [_]TraceStep{.{
        .index = 0,
        .normalized_time = 1.0,
        .layer_input = .{ .path = "step-0-input.f32le", .sha256 = hash, .byte_count = 32 },
        .layer_outputs = .{ .path = "step-0-layers.f32le", .sha256 = hash, .byte_count = 64 },
        .adaln = .{ .path = "step-0-adaln.f32le", .sha256 = hash, .byte_count = 8 },
    }};
    const artifact = ArtifactV2{
        .schema_version = schema_version_v2,
        .subject = "zdraw",
        .model = "z-image-turbo",
        .model_revision = "model-hash",
        .engine_revision = "engine-hash",
        .hook = "transformer.layer",
        .capture_mode = .non_resident_layer_probe,
        .performance_eligible = false,
        .workload = .{
            .width = 64,
            .height = 64,
            .steps = 1,
            .seed = 42,
            .prompt_sha256 = hash,
        },
        .tensor = .{
            .dtype = "f32-le",
            .layer_input_shape = .{ 4, 2 },
            .layer_output_shape = .{ 2, 4, 2 },
        },
        .replay = .{
            .adaln_dtype = "f32-le",
            .adaln_shape = .{2},
            .position_dtype = "u32-le",
            .position_shape = .{ 4, 3 },
            .positions = .{ .path = "positions.u32le", .sha256 = hash, .byte_count = 48 },
        },
        .trace_steps = &steps,
    };
    const json = try toJson(std.testing.allocator, artifact);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"schema_version\": 2") != null);
}

test "runtime-bound replay trace validates and serializes" {
    const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const steps = [_]TraceStep{.{
        .index = 0,
        .normalized_time = 1.0,
        .layer_input = .{ .path = "step-0-input.f32le", .sha256 = hash, .byte_count = 32 },
        .layer_outputs = .{ .path = "step-0-layers.f32le", .sha256 = hash, .byte_count = 64 },
        .adaln = .{ .path = "step-0-adaln.f32le", .sha256 = hash, .byte_count = 8 },
    }};
    const artifact = ArtifactV3{
        .schema_version = schema_version,
        .subject = "zdraw",
        .model = "z-image-turbo",
        .model_revision = "model-hash",
        .engine_revision = "engine-hash",
        .hook = "transformer.layer",
        .capture_mode = .non_resident_layer_probe,
        .performance_eligible = false,
        .runtime = &.{.{ .name = "ZDRAW_STACK_W16", .value = "1" }},
        .workload = .{
            .width = 64,
            .height = 64,
            .steps = 1,
            .seed = 42,
            .prompt_sha256 = hash,
        },
        .tensor = .{
            .dtype = "f32-le",
            .layer_input_shape = .{ 4, 2 },
            .layer_output_shape = .{ 2, 4, 2 },
        },
        .replay = .{
            .adaln_dtype = "f32-le",
            .adaln_shape = .{2},
            .position_dtype = "u32-le",
            .position_shape = .{ 4, 3 },
            .positions = .{ .path = "positions.u32le", .sha256 = hash, .byte_count = 48 },
        },
        .trace_steps = &steps,
    };
    const json = try toJsonV3(std.testing.allocator, artifact);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"schema_version\": 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ZDRAW_STACK_W16\"") != null);
}
