//! Subject-owned Z-Image layer-trace capture for offline discovery.
//!
//! Capture is explicit, fail-closed, and never performance eligible. Offline
//! analysis consumes immutable manifests and payloads outside this process.

const std = @import("std");

const discovery = @import("discovery_trace.zig");
const runtime_options = @import("../runtime/runtime_options.zig");
const zprobe = @import("../zimage/zprobe.zig");
const zrope = @import("../zimage/zrope.zig");
const zstep = @import("../zimage/zstep.zig");

const root_env = "ZDRAW_DISCOVERY_TRACE";
const model_env = "ZDRAW_MODEL_REVISION";
const engine_env = "ZDRAW_ENGINE_REVISION";
const positions_name = "positions.u32le";

pub const Workload = struct {
    prompt: []const u8,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    layers: usize,
};

const Spec = struct {
    root: []const u8,
    model_revision: []const u8,
    engine_revision: []const u8,
    workload: Workload,
};

const Shape = struct {
    tokens: usize,
    hidden: usize,
    adaln: usize,
};

const Record = struct {
    normalized_time: f32 = 0,
    input_name: [64]u8 = undefined,
    input_name_len: usize = 0,
    input_hash: [64]u8 = undefined,
    input_bytes: u64 = 0,
    output_name: [64]u8 = undefined,
    output_name_len: usize = 0,
    output_hash: [64]u8 = undefined,
    output_bytes: u64 = 0,
    adaln_name: [64]u8 = undefined,
    adaln_name_len: usize = 0,
    adaln_hash: [64]u8 = undefined,
    adaln_bytes: u64 = 0,
};

pub const Step = struct {
    allocator: std.mem.Allocator,
    before: []f32,
    after: []f32,
    adaln: []f32,
    positions: []zrope.Pos,

    fn init(
        allocator: std.mem.Allocator,
        shape: Shape,
        layers: usize,
    ) !Step {
        const state_len = try std.math.mul(usize, shape.tokens, shape.hidden);
        const before = try allocator.alloc(f32, state_len);
        errdefer allocator.free(before);
        const after = try allocator.alloc(
            f32,
            try std.math.mul(usize, layers, state_len),
        );
        errdefer allocator.free(after);
        const adaln = try allocator.alloc(f32, shape.adaln);
        errdefer allocator.free(adaln);
        const positions = try allocator.alloc(zrope.Pos, shape.tokens);
        return .{
            .allocator = allocator,
            .before = before,
            .after = after,
            .adaln = adaln,
            .positions = positions,
        };
    }

    pub fn trace(self: *Step) zstep.Trace {
        return .{
            .adaln = self.adaln,
            .positions = self.positions,
            .layer_probe = .{
                .before = self.before,
                .after = self.after,
                .state_len = self.before.len,
            },
        };
    }

    pub fn deinit(self: *Step) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.adaln);
        self.allocator.free(self.after);
        self.allocator.free(self.before);
        self.* = undefined;
    }
};

pub const Capture = struct {
    allocator: std.mem.Allocator,
    spec: Spec,
    records: []Record,
    runtime: []discovery.RuntimeSetting,
    next: usize = 0,
    shape: ?Shape = null,
    prompt_hash: [64]u8,
    positions_hash: [64]u8 = undefined,
    positions_bytes: u64 = 0,

    pub fn fromEnv(
        io: std.Io,
        allocator: std.mem.Allocator,
        workload: Workload,
    ) !?Capture {
        const root_raw = std.c.getenv(root_env) orelse return null;
        const model_raw = std.c.getenv(model_env) orelse
            return error.MissingTraceModelRevision;
        const engine_raw = std.c.getenv(engine_env) orelse
            return error.MissingTraceEngineRevision;
        return try init(io, allocator, .{
            .root = std.mem.span(root_raw),
            .model_revision = std.mem.span(model_raw),
            .engine_revision = std.mem.span(engine_raw),
            .workload = workload,
        });
    }

    fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        spec: Spec,
    ) !Capture {
        if (spec.root.len == 0 or
            spec.model_revision.len == 0 or
            spec.engine_revision.len == 0 or
            spec.workload.steps == 0 or
            spec.workload.layers == 0)
        {
            return error.InvalidTraceSpec;
        }
        const runtime = try captureRuntime(allocator);
        errdefer allocator.free(runtime);
        try std.Io.Dir.cwd().createDirPath(io, spec.root);
        try rejectComplete(io, allocator, spec.root);
        const records = try allocator.alloc(Record, spec.workload.steps);
        @memset(records, .{});
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(spec.workload.prompt, &digest, .{});
        return .{
            .allocator = allocator,
            .spec = spec,
            .records = records,
            .runtime = runtime,
            .prompt_hash = std.fmt.bytesToHex(digest, .lower),
        };
    }

    pub fn begin(
        self: *Capture,
        tokens: usize,
        hidden: usize,
        adaln: usize,
    ) !Step {
        if (self.next >= self.records.len) return error.TooManyTraceSteps;
        const shape = Shape{ .tokens = tokens, .hidden = hidden, .adaln = adaln };
        if (tokens == 0 or hidden == 0 or adaln == 0) {
            return error.InvalidTraceShape;
        }
        if (self.shape) |expected| {
            if (!std.meta.eql(expected, shape)) return error.TraceShapeChanged;
        } else {
            self.shape = shape;
        }
        return Step.init(self.allocator, shape, self.spec.workload.layers);
    }

    pub fn commit(
        self: *Capture,
        io: std.Io,
        normalized_time: f32,
        step: *const Step,
    ) !void {
        const shape = self.shape orelse return error.TraceNotStarted;
        try checkStep(step, shape, self.spec.workload.layers);
        if (!std.math.isFinite(normalized_time)) return error.InvalidTraceTime;
        const record = &self.records[self.next];
        record.normalized_time = normalized_time;
        record.input_name_len = try stepName(
            &record.input_name,
            self.next,
            "input.f32le",
        );
        record.output_name_len = try stepName(
            &record.output_name,
            self.next,
            "layers.f32le",
        );
        record.adaln_name_len = try stepName(
            &record.adaln_name,
            self.next,
            "adaln.f32le",
        );
        record.input_hash, record.input_bytes = try self.writeData(
            io,
            record.input_name[0..record.input_name_len],
            std.mem.sliceAsBytes(step.before),
        );
        record.output_hash, record.output_bytes = try self.writeData(
            io,
            record.output_name[0..record.output_name_len],
            std.mem.sliceAsBytes(step.after),
        );
        record.adaln_hash, record.adaln_bytes = try self.writeData(
            io,
            record.adaln_name[0..record.adaln_name_len],
            std.mem.sliceAsBytes(step.adaln),
        );
        try self.checkPositions(io, step.positions);
        self.next += 1;
    }

    pub fn finish(self: *Capture, io: std.Io) !void {
        if (self.next != self.records.len) return error.IncompleteTrace;
        const shape = self.shape orelse return error.TraceNotStarted;
        const steps = try self.allocator.alloc(discovery.TraceStep, self.records.len);
        defer self.allocator.free(steps);
        for (self.records, steps, 0..) |*record, *step, index| {
            step.* = .{
                .index = std.math.cast(u32, index) orelse
                    return error.TooManyTraceSteps,
                .normalized_time = record.normalized_time,
                .layer_input = artifact(
                    record.input_name[0..record.input_name_len],
                    &record.input_hash,
                    record.input_bytes,
                ),
                .layer_outputs = artifact(
                    record.output_name[0..record.output_name_len],
                    &record.output_hash,
                    record.output_bytes,
                ),
                .adaln = artifact(
                    record.adaln_name[0..record.adaln_name_len],
                    &record.adaln_hash,
                    record.adaln_bytes,
                ),
            };
        }
        const artifact_doc = self.document(shape, steps);
        const json = try discovery.toJsonV3(self.allocator, artifact_doc);
        defer self.allocator.free(json);
        try self.writeNamed(io, "manifest.json", json);
    }

    pub fn deinit(self: *Capture) void {
        self.allocator.free(self.runtime);
        self.allocator.free(self.records);
        self.* = undefined;
    }

    fn checkPositions(
        self: *Capture,
        io: std.Io,
        positions: []const zrope.Pos,
    ) !void {
        const encoded = try self.allocator.alloc(u32, positions.len * 3);
        defer self.allocator.free(encoded);
        for (positions, 0..) |position, index| {
            for (position, 0..) |value, axis| {
                encoded[index * 3 + axis] = std.math.cast(u32, value) orelse
                    return error.TracePositionOutOfRange;
            }
        }
        const bytes = std.mem.sliceAsBytes(encoded);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const hash = std.fmt.bytesToHex(digest, .lower);
        if (self.next == 0) {
            self.positions_hash = hash;
            self.positions_bytes = @intCast(bytes.len);
            try self.writeNamed(io, positions_name, bytes);
        } else if (!std.mem.eql(u8, &self.positions_hash, &hash)) {
            return error.TracePositionsChanged;
        }
    }

    fn writeData(
        self: *Capture,
        io: std.Io,
        name: []const u8,
        bytes: []const u8,
    ) !struct { [64]u8, u64 } {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        try self.writeNamed(io, name, bytes);
        return .{ std.fmt.bytesToHex(digest, .lower), @intCast(bytes.len) };
    }

    fn writeNamed(
        self: *Capture,
        io: std.Io,
        name: []const u8,
        bytes: []const u8,
    ) !void {
        const path = try std.fs.path.join(self.allocator, &.{ self.spec.root, name });
        defer self.allocator.free(path);
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buffer: [64 * 1024]u8 = undefined;
        var writer = file.writerStreaming(io, &buffer);
        try writer.interface.writeAll(bytes);
        try writer.interface.flush();
    }

    fn document(
        self: *const Capture,
        shape: Shape,
        steps: []const discovery.TraceStep,
    ) discovery.ArtifactV3 {
        return .{
            .schema_version = discovery.schema_version,
            .subject = "zdraw",
            .model = "z-image-turbo",
            .model_revision = self.spec.model_revision,
            .engine_revision = self.spec.engine_revision,
            .hook = "transformer.layer",
            .capture_mode = .non_resident_layer_probe,
            .performance_eligible = false,
            .runtime = self.runtime,
            .workload = .{
                .width = self.spec.workload.width,
                .height = self.spec.workload.height,
                .steps = self.spec.workload.steps,
                .seed = self.spec.workload.seed,
                .prompt_sha256 = &self.prompt_hash,
            },
            .tensor = .{
                .dtype = "f32-le",
                .layer_input_shape = .{ shape.tokens, shape.hidden },
                .layer_output_shape = .{
                    self.spec.workload.layers,
                    shape.tokens,
                    shape.hidden,
                },
            },
            .replay = .{
                .adaln_dtype = "f32-le",
                .adaln_shape = .{shape.adaln},
                .position_dtype = "u32-le",
                .position_shape = .{ shape.tokens, 3 },
                .positions = artifact(
                    positions_name,
                    &self.positions_hash,
                    self.positions_bytes,
                ),
            },
            .trace_steps = steps,
        };
    }
};

fn captureRuntime(allocator: std.mem.Allocator) ![]discovery.RuntimeSetting {
    const flags = runtime_options.commonFlags();
    const settings = try allocator.alloc(discovery.RuntimeSetting, flags.len);
    errdefer allocator.free(settings);
    for (flags, settings) |flag, *setting| {
        const raw = std.c.getenv(flag.name.ptr) orelse
            return error.MissingTraceRuntimeSetting;
        setting.* = .{
            .name = flag.name,
            .value = std.mem.span(raw),
        };
    }
    return settings;
}

fn checkStep(step: *const Step, shape: Shape, layers: usize) !void {
    const state_len = try std.math.mul(usize, shape.tokens, shape.hidden);
    if (step.before.len != state_len or
        step.after.len != try std.math.mul(usize, layers, state_len) or
        step.adaln.len != shape.adaln or
        step.positions.len != shape.tokens)
    {
        return error.InvalidTraceStep;
    }
}

fn artifact(
    name: []const u8,
    hash: *const [64]u8,
    bytes: u64,
) discovery.FileArtifact {
    return .{ .path = name, .sha256 = hash, .byte_count = bytes };
}

fn stepName(buffer: *[64]u8, index: usize, suffix: []const u8) !usize {
    const name = try std.fmt.bufPrint(buffer, "step-{d}-{s}", .{ index, suffix });
    return name.len;
}

fn rejectComplete(
    io: std.Io,
    allocator: std.mem.Allocator,
    root: []const u8,
) !void {
    const path = try std.fs.path.join(allocator, &.{ root, "manifest.json" });
    defer allocator.free(path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    file.close(io);
    return error.TraceAlreadyComplete;
}

test "capture writes a validated replay trace" {
    runtime_options.applyEnv(.strict, false);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/trace",
        .{tmp.sub_path},
    );
    defer std.testing.allocator.free(root);
    var capture = try Capture.init(std.testing.io, std.testing.allocator, .{
        .root = root,
        .model_revision = "model-revision",
        .engine_revision = "engine-revision",
        .workload = .{
            .prompt = "test prompt",
            .width = 64,
            .height = 64,
            .steps = 1,
            .seed = 7,
            .layers = 2,
        },
    });
    defer capture.deinit();
    var step = try capture.begin(4, 2, 2);
    defer step.deinit();
    @memset(step.before, 1);
    @memset(step.after, 2);
    @memset(step.adaln, 3);
    for (step.positions, 0..) |*position, index| {
        position.* = .{ index, index + 1, index + 2 };
    }
    try capture.commit(std.testing.io, 0.5, &step);
    try capture.finish(std.testing.io);

    const manifest = try std.fs.path.join(
        std.testing.allocator,
        &.{ root, "manifest.json" },
    );
    defer std.testing.allocator.free(manifest);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        manifest,
        std.testing.allocator,
        .limited(64 * 1024),
    );
    defer std.testing.allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(
        discovery.ArtifactV3,
        std.testing.allocator,
        bytes,
        .{},
    );
    defer parsed.deinit();
    try discovery.validateV3(parsed.value);
    try std.testing.expectEqual(@as(usize, 4), parsed.value.tensor.layer_input_shape[0]);
    try std.testing.expectEqual(runtime_options.commonFlags().len, parsed.value.runtime.len);
}
