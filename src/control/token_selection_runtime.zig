//! Resident owner for the searched token-selection Metal realization.
//!
//! The runtime borrows ZDraw's device and owns only compiled pipelines. The
//! chain pool owns activation storage; the execution plan owns policy and
//! identity. Every encode stays inside the caller's command buffer.

const std = @import("std");

const metal_c = @import("../metal/metal_c.zig");
const mpipe = @import("../metal/mpipe.zig");
const token_selection = @import("token_selection_control.zig");

const backend_id = "zdraw.token-selection.resident.v3";
const schedule_id = "zdraw.token-selection.schedule.v3";
pub const source: [:0]const u8 = @embedFile("token_selection_shader.metal");

const Params = extern struct {
    source_tokens: u32,
    feature_width: u32,
    selected_tokens: u32,
    reconstruction_weight: f32,
};

pub const Runtime = struct {
    device: *anyopaque,
    score_mean: *anyopaque,
    score_zero: *anyopaque,
    select: *anyopaque,
    gather_state: *anyopaque,
    gather_positions: *anyopaque,
    scatter_state: *anyopaque,
    normalize_source: *anyopaque,
    normalize_destination: *anyopaque,
    assign_max: *anyopaque,
    clear_inverse: *anyopaque,
    mark_inverse: *anyopaque,
    reconstruct_cosine_residual: *anyopaque,

    pub fn initBorrowed(device: *anyopaque) !Runtime {
        var compile_error: [1024]u8 = undefined;
        const score_mean = try compile(device, "zdraw_token_score_mean", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(score_mean);
        const score_zero = try compile(device, "zdraw_token_score_zero", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(score_zero);
        const select = try compile(device, "zdraw_token_select_topk", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(select);
        const gather_state =
            try compile(device, "zdraw_token_gather_state", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(gather_state);
        const gather_positions =
            try compile(device, "zdraw_token_gather_positions", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(gather_positions);
        const scatter_state =
            try compile(device, "zdraw_token_scatter_state", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(scatter_state);
        const normalize_source =
            try compile(device, "zdraw_token_normalize_source", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(normalize_source);
        const normalize_destination =
            try compile(device, "zdraw_token_normalize_destination", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(normalize_destination);
        const assign_max =
            try compile(device, "zdraw_token_assign_max", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(assign_max);
        const clear_inverse =
            try compile(device, "zdraw_token_clear_inverse", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(clear_inverse);
        const mark_inverse =
            try compile(device, "zdraw_token_mark_inverse", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(mark_inverse);
        const reconstruct_cosine_residual =
            try compile(
                device,
                "zdraw_token_reconstruct_cosine_residual",
                &compile_error,
            );
        return .{
            .device = device,
            .score_mean = score_mean,
            .score_zero = score_zero,
            .select = select,
            .gather_state = gather_state,
            .gather_positions = gather_positions,
            .scatter_state = scatter_state,
            .normalize_source = normalize_source,
            .normalize_destination = normalize_destination,
            .assign_max = assign_max,
            .clear_inverse = clear_inverse,
            .mark_inverse = mark_inverse,
            .reconstruct_cosine_residual = reconstruct_cosine_residual,
        };
    }

    pub fn deinit(self: *Runtime) void {
        metal_c.zdraw_metal_release_pipeline(self.reconstruct_cosine_residual);
        metal_c.zdraw_metal_release_pipeline(self.mark_inverse);
        metal_c.zdraw_metal_release_pipeline(self.clear_inverse);
        metal_c.zdraw_metal_release_pipeline(self.assign_max);
        metal_c.zdraw_metal_release_pipeline(self.normalize_destination);
        metal_c.zdraw_metal_release_pipeline(self.normalize_source);
        metal_c.zdraw_metal_release_pipeline(self.scatter_state);
        metal_c.zdraw_metal_release_pipeline(self.gather_positions);
        metal_c.zdraw_metal_release_pipeline(self.gather_state);
        metal_c.zdraw_metal_release_pipeline(self.select);
        metal_c.zdraw_metal_release_pipeline(self.score_zero);
        metal_c.zdraw_metal_release_pipeline(self.score_mean);
        self.* = undefined;
    }

    pub fn validate(self: *const Runtime, plan: token_selection.Plan) !void {
        _ = self;
        const actual_source = sourceSha256();
        if (!std.mem.eql(u8, &actual_source, plan.source_sha256)) {
            return error.TokenSelectionSourceMismatch;
        }
        const actual_realization = realizeSha256(plan);
        if (!std.mem.eql(u8, &actual_realization, plan.realization_sha256)) {
            return error.TokenSelectionRealizationMismatch;
        }
        const actual_schedule = scheduleSha256(plan);
        if (!std.mem.eql(u8, &actual_schedule, plan.schedule_sha256)) {
            return error.TokenSelectionScheduleMismatch;
        }
    }

    pub fn encodeSelect(
        self: *Runtime,
        batch: *anyopaque,
        full_state: *anyopaque,
        scores: *anyopaque,
        indices: *anyopaque,
        compact_state: *anyopaque,
        full_positions: *anyopaque,
        compact_positions: *anyopaque,
        plan: token_selection.Plan,
    ) !void {
        const params = try makeParams(plan);
        switch (plan.score) {
            .feature_mean => try encode(
                batch,
                self.score_mean,
                .{ full_state, scores, null, null, null },
                &params,
                2,
                plan.source_tokens,
            ),
            .zero => try encode(
                batch,
                self.score_zero,
                .{ scores, null, null, null, null },
                &params,
                1,
                plan.source_tokens,
            ),
        }
        try encodeTopK(
            batch,
            self.select,
            .{ scores, indices, null, null, null },
            &params,
            2,
        );
        try encode(
            batch,
            self.gather_state,
            .{ full_state, indices, compact_state, null, null },
            &params,
            3,
            try std.math.mul(usize, plan.selected_tokens, plan.feature_width),
        );
        try encode(
            batch,
            self.gather_positions,
            .{ full_positions, indices, compact_positions, null, null },
            &params,
            3,
            try std.math.mul(usize, plan.selected_tokens, 3),
        );
    }

    pub fn encodeScatter(
        self: *Runtime,
        batch: *anyopaque,
        compact_state: *anyopaque,
        indices: *anyopaque,
        full_state: *anyopaque,
        plan: token_selection.Plan,
    ) !void {
        const params = try makeParams(plan);
        try encode(
            batch,
            self.scatter_state,
            .{ compact_state, indices, full_state, null, null },
            &params,
            3,
            try std.math.mul(usize, plan.selected_tokens, plan.feature_width),
        );
    }

    pub fn encodeCosResid(
        self: *Runtime,
        batch: *anyopaque,
        compact_state: *anyopaque,
        indices: *anyopaque,
        full_state: *anyopaque,
        scores: *anyopaque,
        normalized_source: *anyopaque,
        normalized_destination: *anyopaque,
        assignments: *anyopaque,
        inverse: *anyopaque,
        plan: token_selection.Plan,
    ) !void {
        if (plan.reconstruction != .cosine_residual_interpolate) {
            return error.InvalidTokenSelectionPlan;
        }
        const params = try makeParams(plan);
        try encode(
            batch,
            self.normalize_source,
            .{ full_state, normalized_source, null, null, null },
            &params,
            2,
            plan.source_tokens,
        );
        try encode(
            batch,
            self.normalize_destination,
            .{ compact_state, normalized_destination, null, null, null },
            &params,
            2,
            plan.selected_tokens,
        );
        const gemm = metal_c.GemmParams{
            .m = try fitU32(plan.source_tokens),
            .k = try fitU32(plan.feature_width),
            .n = try fitU32(plan.selected_tokens),
            .dtype = 3,
            .mode = 2,
        };
        if (metal_c.zdraw_metal_run_gemm_mps_enc(
            batch,
            normalized_source,
            normalized_destination,
            scores,
            &gemm,
        ) != 0) return error.MetalDispatchFailed;
        try encode(
            batch,
            self.assign_max,
            .{ scores, assignments, null, null, null },
            &params,
            2,
            plan.source_tokens,
        );
        try encode(
            batch,
            self.clear_inverse,
            .{ inverse, null, null, null, null },
            &params,
            1,
            plan.source_tokens,
        );
        try encode(
            batch,
            self.mark_inverse,
            .{ indices, inverse, null, null, null },
            &params,
            2,
            plan.selected_tokens,
        );
        try encode(
            batch,
            self.reconstruct_cosine_residual,
            .{ compact_state, assignments, inverse, full_state, null },
            &params,
            4,
            try std.math.mul(usize, plan.source_tokens, plan.feature_width),
        );
    }
};

pub fn sourceSha256() [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn realizeSha256(plan: token_selection.Plan) [64]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(backend_id);
    hasher.update(&.{0});
    hasher.update(plan.semantic_program_sha256);
    hasher.update(&.{0});
    hasher.update(plan.selection_program_sha256);
    hasher.update(&.{0});
    const source_hash = sourceSha256();
    hasher.update(&source_hash);
    var dimensions = [_]u8{0} ** 24;
    std.mem.writeInt(u64, dimensions[0..8], plan.source_tokens, .little);
    std.mem.writeInt(u64, dimensions[8..16], plan.feature_width, .little);
    std.mem.writeInt(u64, dimensions[16..24], plan.selected_tokens, .little);
    hasher.update(&dimensions);
    hasher.update(scoreName(plan.score));
    hasher.update(&.{0});
    hasher.update(reconLabel(plan.reconstruction));
    var weight: [4]u8 = undefined;
    std.mem.writeInt(u32, &weight, @bitCast(plan.reconstruction_weight), .little);
    hasher.update(&weight);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn scheduleSha256(plan: token_selection.Plan) [64]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(schedule_id);
    hasher.update(&.{0});
    hasher.update(plan.realization_sha256);
    hasher.update(&.{0});
    var dimensions = [_]u8{0} ** 32;
    std.mem.writeInt(u64, dimensions[0..8], plan.source_tokens, .little);
    std.mem.writeInt(u64, dimensions[8..16], plan.feature_width, .little);
    std.mem.writeInt(u64, dimensions[16..24], plan.selected_tokens, .little);
    std.mem.writeInt(u64, dimensions[24..32], 32, .little);
    hasher.update(&dimensions);
    updateCount(&hasher, plan.layers.len);
    for (plan.layers) |layer| updateU32(&hasher, layer);
    updateCount(&hasher, plan.timesteps.len);
    for (plan.timesteps) |timestep| updateU32(&hasher, timestep);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn updateCount(hasher: *std.crypto.hash.sha2.Sha256, value: usize) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hasher.update(&bytes);
}

fn updateU32(hasher: *std.crypto.hash.sha2.Sha256, value: u32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, value, .little);
    hasher.update(&bytes);
}

fn makeParams(plan: token_selection.Plan) !Params {
    return .{
        .source_tokens = try fitU32(plan.source_tokens),
        .feature_width = try fitU32(plan.feature_width),
        .selected_tokens = try fitU32(plan.selected_tokens),
        .reconstruction_weight = plan.reconstruction_weight,
    };
}

fn scoreName(value: token_selection.Score) []const u8 {
    return switch (value) {
        .feature_mean => "feature_mean",
        .zero => "zero",
    };
}

fn reconLabel(value: token_selection.Reconstruction) []const u8 {
    return switch (value) {
        .reference_scatter => "reference_scatter",
        .cosine_residual_interpolate => "cosine_residual_interpolate",
    };
}

fn compile(
    device: *anyopaque,
    entry: [*:0]const u8,
    compile_error: *[1024]u8,
) !*anyopaque {
    return mpipe.required(device, source.ptr, entry, compile_error);
}

fn encode(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const Params,
    params_index: u32,
    grid: usize,
) !void {
    const threads = @min(
        @as(usize, 256),
        metal_c.zdraw_metal_pipeline_threads(pipeline),
    );
    if (metal_c.zdraw_metal_run_glue_enc(
        batch,
        pipeline,
        buffers[0],
        buffers[1],
        buffers[2],
        buffers[3],
        buffers[4],
        null,
        params,
        @sizeOf(Params),
        params_index,
        grid,
        threads,
        0,
    ) != 0) return error.MetalDispatchFailed;
}

fn encodeTopK(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const Params,
    params_index: u32,
) !void {
    const threads = 256;
    if (metal_c.zdraw_metal_pipeline_threads(pipeline) < threads) {
        return error.TokenSelectionThreadgroupTooSmall;
    }
    if (metal_c.zdraw_metal_run_glue_enc(
        batch,
        pipeline,
        buffers[0],
        buffers[1],
        buffers[2],
        buffers[3],
        buffers[4],
        null,
        params,
        @sizeOf(Params),
        params_index,
        threads,
        threads,
        0,
    ) != 0) return error.MetalDispatchFailed;
}

fn fitU32(value: usize) !u32 {
    return std.math.cast(u32, value) orelse error.InvalidTokenSelectionPlan;
}

test "resident realization identity binds program, source, and shape" {
    const hash = "430e47182361505a741cd8456efd9e41c" ++
        "9d908b49262370954a2f0041a79e8f0";
    const base = token_selection.Plan{
        .semantic_program_sha256 = hash,
        .selection_program_sha256 = hash,
        .realization_sha256 = hash,
        .schedule_sha256 = hash,
        .source_sha256 = hash,
        .source_tokens = 64,
        .feature_width = 3840,
        .selected_tokens = 32,
        .layers = &.{0},
        .timesteps = &.{0},
        .expected_steps = 1,
    };
    const first = realizeSha256(base);
    const second = realizeSha256(.{
        .semantic_program_sha256 = hash,
        .selection_program_sha256 = hash,
        .realization_sha256 = hash,
        .schedule_sha256 = hash,
        .source_sha256 = hash,
        .source_tokens = 64,
        .feature_width = 3840,
        .selected_tokens = 31,
        .layers = &.{0},
        .timesteps = &.{0},
        .expected_steps = 1,
    });
    const reconstructed = realizeSha256(.{
        .semantic_program_sha256 = hash,
        .selection_program_sha256 = hash,
        .realization_sha256 = hash,
        .schedule_sha256 = hash,
        .source_sha256 = hash,
        .source_tokens = 64,
        .feature_width = 3840,
        .selected_tokens = 32,
        .layers = &.{0},
        .timesteps = &.{0},
        .expected_steps = 1,
        .score = .zero,
        .reconstruction = .cosine_residual_interpolate,
        .reconstruction_weight = 0.75,
    });
    try std.testing.expect(!std.mem.eql(u8, &first, &second));
    try std.testing.expect(!std.mem.eql(u8, &first, &reconstructed));
    try std.testing.expectEqual(@as(usize, 64), sourceSha256().len);
}

test "resident schedule identity binds deployment and realization" {
    const semantic = "2fa6d407e70ae7630c9d6de825e52e5d" ++
        "bf973ee4eebc81b850bf507e7f2f12a5";
    const selection = "430e47182361505a741cd8456efd9e41c" ++
        "9d908b49262370954a2f0041a79e8f0";
    const plan = token_selection.Plan{
        .semantic_program_sha256 = semantic,
        .selection_program_sha256 = selection,
        .realization_sha256 = "7bd3a8bbf25bb1fef961dd46f6dcb95ab10a7d846f44f06863e1b1a802a0af22",
        .schedule_sha256 = "0c9132da9f1a57a8afe8e9666c6f6535fe2253935ff111f4be712d91e1cab68b",
        .source_sha256 = "53a8df0625ce3e033706d549a413aa67f1a67d8a3f0b85ea57519b14e08aae55",
        .source_tokens = 64,
        .feature_width = 3840,
        .selected_tokens = 32,
        .layers = &.{ 0, 10, 20 },
        .timesteps = &.{ 0, 1, 2, 3 },
        .expected_steps = 4,
    };
    const first = scheduleSha256(plan);
    var changed = plan;
    changed.timesteps = &.{ 0, 1, 2 };
    const second = scheduleSha256(changed);
    try std.testing.expect(!std.mem.eql(u8, &first, &second));
}

test "discovered cosine residual identity matches the capsule lowerer" {
    const plan = token_selection.Plan{
        .semantic_program_sha256 = "9724afed5868ceaf6987875fea1664e4" ++
            "37c4325194c298052f3192b9824fcf07",
        .selection_program_sha256 = "db68e8b62fbb4f7393982a7d55f4f3e" ++
            "ef84d7605ed6d48acc9106db3c583bb26",
        .realization_sha256 = "07bd8ece5de25911b6db94aa265372ccd" ++
            "a8be84cb98eeb238faf1acb47544d8f",
        .schedule_sha256 = "81fd777c060e36848796dc119bc49475c" ++
            "23f7f35720623f250e51a247da3a098",
        .source_sha256 = "f1ec29bcf6236709a4a8facee46310cad" ++
            "76ca53a07cf382594bb3a3e2e79ffb5",
        .source_tokens = 4128,
        .feature_width = 3840,
        .selected_tokens = 3584,
        .layers = &.{17},
        .timesteps = &.{3},
        .expected_steps = 4,
        .score = .zero,
        .reconstruction = .cosine_residual_interpolate,
        .reconstruction_weight = 0.75,
    };
    try std.testing.expectEqualStrings(plan.source_sha256, &sourceSha256());
    try std.testing.expectEqualStrings(plan.realization_sha256, &realizeSha256(plan));
    try std.testing.expectEqualStrings(plan.schedule_sha256, &scheduleSha256(plan));
}
