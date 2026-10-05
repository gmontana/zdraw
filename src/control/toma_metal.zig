//! GPU-resident exact F32 ToMA primitives.
//!
//! Pattern construction is one Metal command buffer with no intermediate CPU
//! readback. These first kernels prioritize oracle parity; later tuning may
//! replace individual stages without changing Pattern ownership or semantics.

const std = @import("std");

const execution_plan = @import("execution_plan.zig");
const mbuffer = @import("../metal/mbuffer.zig");
const metal_c = @import("../metal/metal_c.zig");
const mpipe = @import("../metal/mpipe.zig");
const shader = @import("toma_metal_shader.zig");
const toma_config = @import("toma_config.zig");
const toma_oracle = @import("toma_oracle.zig");

const PatternParams = extern struct {
    source_tokens: u32,
    feature_width: u32,
    destination_tokens: u32,
    region_count: u32,
    region_size: u32,
    destinations_per_region: u32,
    token_side: u32,
    region_side: u32,
    selection_layout: u32,
    assignment_scope: u32,
    mask_selected: u32,
    raw_zero_norm: u32,
    assignment_scale: f32,
};

const ApplyParams = extern struct {
    source_tokens: u32,
    destination_tokens: u32,
    width: u32,
    region_size: u32,
    destinations_per_region: u32,
    token_side: u32,
    region_side: u32,
    selection_layout: u32,
    assignment_scope: u32,
};

const CopyParams = extern struct {
    count: u32,
    source_offset: u32,
    destination_offset: u32,
};

const PositionParams = extern struct {
    image_source_tokens: u32,
    image_destination_tokens: u32,
    caption_tokens: u32,
};

pub const Pattern = struct {
    destinations: mbuffer.Buffer,
    assignment: mbuffer.Buffer,
    merge_matrix: mbuffer.Buffer,
    source_tokens: usize,
    destination_tokens: usize,
    unmerge: execution_plan.Unmerge,
    shape: toma_config.Shape,
    config: toma_config.Config,

    fn init(
        device: *anyopaque,
        shape: toma_config.Shape,
        config: toma_config.Config,
    ) !Pattern {
        var destinations = try mbuffer.Buffer.empty(
            device,
            config.destination_tokens * @sizeOf(u32),
        );
        errdefer destinations.deinit();
        const matrix_len = shape.source_tokens * config.destination_tokens;
        var assignment = try floatBuffer(device, matrix_len);
        errdefer assignment.deinit();
        const merge_matrix = try floatBuffer(device, matrix_len);
        return .{
            .destinations = destinations,
            .assignment = assignment,
            .merge_matrix = merge_matrix,
            .source_tokens = shape.source_tokens,
            .destination_tokens = config.destination_tokens,
            .unmerge = config.unmerge,
            .shape = shape,
            .config = config,
        };
    }

    pub fn deinit(self: *Pattern) void {
        self.destinations.deinit();
        self.assignment.deinit();
        self.merge_matrix.deinit();
        self.* = undefined;
    }

    pub fn read(
        self: *const Pattern,
        destinations: []u32,
        assignment: []f32,
        merge_matrix: []f32,
    ) !void {
        const matrix_len = self.source_tokens * self.destination_tokens;
        if (destinations.len != self.destination_tokens or
            assignment.len != matrix_len or merge_matrix.len != matrix_len)
        {
            return error.InvalidOutputShape;
        }
        metal_c.zdraw_metal_read_buffer(
            self.destinations.handle,
            std.mem.sliceAsBytes(destinations).ptr,
            std.mem.sliceAsBytes(destinations).len,
        );
        metal_c.zdraw_metal_read_buffer(
            self.assignment.handle,
            std.mem.sliceAsBytes(assignment).ptr,
            std.mem.sliceAsBytes(assignment).len,
        );
        metal_c.zdraw_metal_read_buffer(
            self.merge_matrix.handle,
            std.mem.sliceAsBytes(merge_matrix).ptr,
            std.mem.sliceAsBytes(merge_matrix).len,
        );
    }
};

/// Borrowed application buffers backed by Context scratch storage.
///
/// Pattern construction and merge/unmerge application do not overlap. Reusing
/// the dead pattern scratch avoids keeping a second full activation pair live.
/// The handles remain valid until Context is rebuilt for another shape or
/// deinitialized; their contents are invalidated by every pattern rebuild.
pub const ApplyBuffers = struct {
    merged: *anyopaque,
    unmerged: *anyopaque,
};

pub const Context = struct {
    device: *anyopaque,
    queue: *anyopaque,
    owns_runtime: bool,
    scratch: ?Scratch,
    normalize_pipe: *anyopaque,
    similarity_pipe: *anyopaque,
    facility_pipe: *anyopaque,
    clear_pipe: *anyopaque,
    assignment_pipe: *anyopaque,
    gather_pipe: *anyopaque,
    scores_mma_pipe: *anyopaque,
    score_softmax_pipe: *anyopaque,
    row_norm_pipe: *anyopaque,
    merge_pipe: *anyopaque,
    unmerge_pipe: *anyopaque,
    merge_mma_pipe: *anyopaque,
    unmerge_mma_pipe: *anyopaque,
    copy_pipe: *anyopaque,
    gather_positions_pipe: *anyopaque,

    pub fn init() !Context {
        const device = metal_c.zdraw_metal_create_device() orelse
            return error.MetalNotAvailable;
        errdefer metal_c.zdraw_metal_release_device(device);
        const queue = metal_c.zdraw_metal_create_queue(device) orelse
            return error.MetalQueueFailed;
        errdefer metal_c.zdraw_metal_release_queue(queue);
        return initOn(device, queue, true);
    }

    pub fn initBorrowed(device: *anyopaque, queue: *anyopaque) !Context {
        return initOn(device, queue, false);
    }

    fn initOn(device: *anyopaque, queue: *anyopaque, owns_runtime: bool) !Context {
        var compile_error: [1024]u8 = undefined;
        const normalize_pipe = try compile(device, "toma_normalize", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(normalize_pipe);
        const similarity_pipe = try compile(device, "toma_similarity", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(similarity_pipe);
        const facility_pipe = try compile(device, "toma_facility", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(facility_pipe);
        const clear_pipe = try compile(device, "toma_clear", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(clear_pipe);
        const assignment_pipe = try compile(device, "toma_assignment", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(assignment_pipe);
        const gather_pipe = try compile(device, "toma_gather", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(gather_pipe);
        const scores_mma_pipe = try compile(device, "toma_scores_mma", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(scores_mma_pipe);
        const score_softmax_pipe = try compile(device, "toma_score_softmax", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(score_softmax_pipe);
        const row_norm_pipe = try compile(device, "toma_row_norm", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(row_norm_pipe);
        const merge_pipe = try compile(device, "toma_merge", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(merge_pipe);
        const unmerge_pipe = try compile(device, "toma_unmerge", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(unmerge_pipe);
        const merge_mma_pipe = try compile(device, "toma_merge_mma", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(merge_mma_pipe);
        const unmerge_mma_pipe = try compile(device, "toma_unmerge_mma", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(unmerge_mma_pipe);
        const copy_pipe = try compile(device, "toma_copy_u32", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(copy_pipe);
        const gather_positions_pipe =
            try compile(device, "toma_gather_positions", &compile_error);
        errdefer metal_c.zdraw_metal_release_pipeline(gather_positions_pipe);
        return .{
            .device = device,
            .queue = queue,
            .owns_runtime = owns_runtime,
            .scratch = null,
            .normalize_pipe = normalize_pipe,
            .similarity_pipe = similarity_pipe,
            .facility_pipe = facility_pipe,
            .clear_pipe = clear_pipe,
            .assignment_pipe = assignment_pipe,
            .gather_pipe = gather_pipe,
            .scores_mma_pipe = scores_mma_pipe,
            .score_softmax_pipe = score_softmax_pipe,
            .row_norm_pipe = row_norm_pipe,
            .merge_pipe = merge_pipe,
            .unmerge_pipe = unmerge_pipe,
            .merge_mma_pipe = merge_mma_pipe,
            .unmerge_mma_pipe = unmerge_mma_pipe,
            .copy_pipe = copy_pipe,
            .gather_positions_pipe = gather_positions_pipe,
        };
    }

    pub fn deinit(self: *Context) void {
        if (self.scratch) |*scratch| scratch.deinit();
        metal_c.zdraw_metal_release_pipeline(self.gather_positions_pipe);
        metal_c.zdraw_metal_release_pipeline(self.copy_pipe);
        metal_c.zdraw_metal_release_pipeline(self.unmerge_mma_pipe);
        metal_c.zdraw_metal_release_pipeline(self.merge_mma_pipe);
        metal_c.zdraw_metal_release_pipeline(self.unmerge_pipe);
        metal_c.zdraw_metal_release_pipeline(self.merge_pipe);
        metal_c.zdraw_metal_release_pipeline(self.row_norm_pipe);
        metal_c.zdraw_metal_release_pipeline(self.score_softmax_pipe);
        metal_c.zdraw_metal_release_pipeline(self.scores_mma_pipe);
        metal_c.zdraw_metal_release_pipeline(self.gather_pipe);
        metal_c.zdraw_metal_release_pipeline(self.assignment_pipe);
        metal_c.zdraw_metal_release_pipeline(self.clear_pipe);
        metal_c.zdraw_metal_release_pipeline(self.facility_pipe);
        metal_c.zdraw_metal_release_pipeline(self.similarity_pipe);
        metal_c.zdraw_metal_release_pipeline(self.normalize_pipe);
        if (self.owns_runtime) {
            metal_c.zdraw_metal_release_queue(self.queue);
            metal_c.zdraw_metal_release_device(self.device);
        }
        self.* = undefined;
    }

    pub fn build(
        self: *Context,
        features: []const f32,
        source_tokens: usize,
        feature_width: usize,
        config: toma_config.Config,
    ) !Pattern {
        if (features.len != source_tokens * feature_width) {
            return error.InvalidInputShape;
        }
        var input = try mbuffer.Buffer.fromBytes(self.device, std.mem.sliceAsBytes(features));
        defer input.deinit();
        return self.buildBuffer(input.handle, source_tokens, feature_width, config);
    }

    pub fn buildBuffer(
        self: *Context,
        features: *anyopaque,
        source_tokens: usize,
        feature_width: usize,
        config: toma_config.Config,
    ) !Pattern {
        const shape = try toma_config.shape(source_tokens, feature_width, config);
        if (config.unmerge == .pseudo_inverse) return error.UnsupportedMetalUnmerge;
        var pattern = try Pattern.init(self.device, shape, config);
        errdefer pattern.deinit();
        try self.rebuildBuffer(features, &pattern);
        return pattern;
    }

    pub fn rebuildBuffer(
        self: *Context,
        features: *anyopaque,
        pattern: *Pattern,
    ) !void {
        const params = try patternParams(pattern.shape, pattern.config);
        const scratch = try self.scratchFor(pattern.shape, pattern.config);
        const batch = metal_c.zdraw_metal_batch_begin(self.queue) orelse
            return error.MetalDispatchFailed;
        var batch_open = true;
        errdefer {
            if (batch_open) _ = metal_c.zdraw_metal_batch_end(batch);
        }
        try self.encodePattern(batch, features, scratch, pattern, params);
        const result = metal_c.zdraw_metal_batch_end(batch);
        batch_open = false;
        if (result != 0) return error.MetalDispatchFailed;
    }

    pub fn merge(
        self: *Context,
        allocator: std.mem.Allocator,
        pattern: *const Pattern,
        input: []const f32,
        width: usize,
    ) ![]f32 {
        if (input.len != pattern.source_tokens * width) return error.InvalidInputShape;
        var input_buffer = try mbuffer.Buffer.fromBytes(
            self.device,
            std.mem.sliceAsBytes(input),
        );
        defer input_buffer.deinit();
        var output_buffer = try mbuffer.Buffer.empty(
            self.device,
            pattern.destination_tokens * width * @sizeOf(f32),
        );
        defer output_buffer.deinit();
        try self.runApply(pattern, input_buffer.handle, output_buffer.handle, width, true);
        const output = try allocator.alloc(f32, pattern.destination_tokens * width);
        metal_c.zdraw_metal_read_buffer(
            output_buffer.handle,
            std.mem.sliceAsBytes(output).ptr,
            std.mem.sliceAsBytes(output).len,
        );
        return output;
    }

    pub fn unmerge(
        self: *Context,
        allocator: std.mem.Allocator,
        pattern: *const Pattern,
        input: []const f32,
        width: usize,
    ) ![]f32 {
        if (input.len != pattern.destination_tokens * width) {
            return error.InvalidInputShape;
        }
        var input_buffer = try mbuffer.Buffer.fromBytes(
            self.device,
            std.mem.sliceAsBytes(input),
        );
        defer input_buffer.deinit();
        var output_buffer = try mbuffer.Buffer.empty(
            self.device,
            pattern.source_tokens * width * @sizeOf(f32),
        );
        defer output_buffer.deinit();
        try self.runApply(pattern, input_buffer.handle, output_buffer.handle, width, false);
        const output = try allocator.alloc(f32, pattern.source_tokens * width);
        metal_c.zdraw_metal_read_buffer(
            output_buffer.handle,
            std.mem.sliceAsBytes(output).ptr,
            std.mem.sliceAsBytes(output).len,
        );
        return output;
    }

    pub fn mergeBuffer(
        self: *Context,
        pattern: *const Pattern,
        input: *anyopaque,
        output: *anyopaque,
        width: usize,
    ) !void {
        try self.runApply(pattern, input, output, width, true);
    }

    pub fn unmergeBuffer(
        self: *Context,
        pattern: *const Pattern,
        input: *anyopaque,
        output: *anyopaque,
        width: usize,
    ) !void {
        try self.runApply(pattern, input, output, width, false);
    }

    /// Merge the image prefix, preserve the caption suffix, and gather the
    /// matching three-axis RoPE positions without leaving the command buffer.
    pub fn encodePackModal(
        self: *Context,
        batch: *anyopaque,
        pattern: *const Pattern,
        full_state: *anyopaque,
        full_positions: *anyopaque,
        reduced_state: *anyopaque,
        reduced_positions: *anyopaque,
        caption_tokens: usize,
        width: usize,
    ) !void {
        try self.encodeMerge(
            batch,
            pattern,
            full_state,
            reduced_state,
            caption_tokens,
            width,
        );
        try self.encodePos(
            batch,
            pattern,
            full_positions,
            reduced_positions,
            caption_tokens,
        );
    }

    pub fn encodeMerge(
        self: *Context,
        batch: *anyopaque,
        pattern: *const Pattern,
        full_state: *anyopaque,
        reduced_state: *anyopaque,
        caption_tokens: usize,
        width: usize,
    ) !void {
        try self.encodeApply(batch, pattern, full_state, reduced_state, width, true);
        const caption_words = try std.math.mul(usize, caption_tokens, width);
        try self.encodeCopy(
            batch,
            full_state,
            reduced_state,
            try std.math.mul(usize, pattern.source_tokens, width),
            try std.math.mul(usize, pattern.destination_tokens, width),
            caption_words,
        );
    }

    pub fn encodePos(
        self: *Context,
        batch: *anyopaque,
        pattern: *const Pattern,
        full_positions: *anyopaque,
        reduced_positions: *anyopaque,
        caption_tokens: usize,
    ) !void {
        const pos = PositionParams{
            .image_source_tokens = try u32Fit(pattern.source_tokens),
            .image_destination_tokens = try u32Fit(pattern.destination_tokens),
            .caption_tokens = try u32Fit(caption_tokens),
        };
        try rawEncode(
            batch,
            self.gather_positions_pipe,
            .{
                full_positions,
                pattern.destinations.handle,
                reduced_positions,
                null,
                null,
            },
            &pos,
            @sizeOf(PositionParams),
            3,
            (pattern.destination_tokens + caption_tokens) * 3,
        );
    }

    /// Restore the image prefix and copy the untouched caption suffix at a
    /// full-token residual boundary.
    pub fn encodeUnpack(
        self: *Context,
        batch: *anyopaque,
        pattern: *const Pattern,
        reduced_state: *anyopaque,
        full_state: *anyopaque,
        caption_tokens: usize,
        width: usize,
    ) !void {
        try self.encodeApply(
            batch,
            pattern,
            reduced_state,
            full_state,
            width,
            false,
        );
        const caption_words = try std.math.mul(usize, caption_tokens, width);
        try self.encodeCopy(
            batch,
            reduced_state,
            full_state,
            try std.math.mul(usize, pattern.destination_tokens, width),
            try std.math.mul(usize, pattern.source_tokens, width),
            caption_words,
        );
    }

    pub fn applyBuffers(
        self: *Context,
        pattern: *const Pattern,
        width: usize,
    ) !ApplyBuffers {
        if (width > pattern.shape.feature_width) return error.InvalidOutputShape;
        const scratch = try self.scratchFor(pattern.shape, pattern.config);
        return .{
            .merged = scratch.mergedHandle(),
            .unmerged = scratch.normalized.handle,
        };
    }

    fn encodePattern(
        self: *Context,
        batch: *anyopaque,
        features: *anyopaque,
        scratch: *Scratch,
        pattern: *Pattern,
        params: PatternParams,
    ) !void {
        try patternEncode(batch, self.normalize_pipe, .{
            features,
            scratch.normalized.handle,
            null,
            null,
            null,
        }, &params, 2, params.source_tokens);
        const selection = if (scratch.selection) |*buffer| second: {
            try patternEncode(batch, self.normalize_pipe, .{
                scratch.normalized.handle,
                buffer.handle,
                null,
                null,
                null,
            }, &params, 2, params.source_tokens);
            break :second buffer.handle;
        } else scratch.normalized.handle;
        try self.encodeMatrices(batch, selection, scratch, pattern, params);
    }

    fn encodeMatrices(
        self: *Context,
        batch: *anyopaque,
        selection: *anyopaque,
        scratch: *Scratch,
        pattern: *Pattern,
        params: PatternParams,
    ) !void {
        const similarity_count = params.region_count * params.region_size * params.region_size;
        try patternEncode(batch, self.similarity_pipe, .{
            selection,
            scratch.similarity.handle,
            null,
            null,
            null,
        }, &params, 2, similarity_count);
        try patternEncode(batch, self.facility_pipe, .{
            scratch.similarity.handle,
            pattern.destinations.handle,
            scratch.max_similarity.handle,
            scratch.selected.handle,
            null,
        }, &params, 4, params.region_count);
        try self.encodeAssign(batch, scratch, pattern, params);
        try patternEncode(batch, self.row_norm_pipe, .{
            pattern.assignment.handle,
            pattern.merge_matrix.handle,
            null,
            null,
            null,
        }, &params, 2, params.destination_tokens);
    }

    fn encodeAssign(
        self: *Context,
        batch: *anyopaque,
        scratch: *Scratch,
        pattern: *Pattern,
        params: PatternParams,
    ) !void {
        if (scratch.selected_features) |*selected_features| {
            try patternEncode(batch, self.gather_pipe, .{
                scratch.normalized.handle,
                pattern.destinations.handle,
                selected_features.handle,
                null,
                null,
            }, &params, 3, params.destination_tokens * params.feature_width);
            try patternMma(
                batch,
                self.scores_mma_pipe,
                .{
                    selected_features.handle,
                    scratch.normalized.handle,
                    pattern.assignment.handle,
                    null,
                    null,
                },
                &params,
                params.destination_tokens,
                params.source_tokens,
            );
            try patternEncode(batch, self.score_softmax_pipe, .{
                pattern.assignment.handle,
                null,
                null,
                null,
                null,
            }, &params, 1, params.source_tokens);
        } else {
            const matrix_count = params.destination_tokens * params.source_tokens;
            try patternEncode(batch, self.clear_pipe, .{
                pattern.assignment.handle,
                null,
                null,
                null,
                null,
            }, &params, 1, matrix_count);
            try patternEncode(batch, self.assignment_pipe, .{
                scratch.normalized.handle,
                pattern.destinations.handle,
                pattern.assignment.handle,
                null,
                null,
            }, &params, 3, params.source_tokens);
        }
    }

    fn runApply(
        self: *Context,
        pattern: *const Pattern,
        input: *anyopaque,
        output: *anyopaque,
        width: usize,
        is_merge: bool,
    ) !void {
        const batch = metal_c.zdraw_metal_batch_begin(self.queue) orelse
            return error.MetalDispatchFailed;
        var batch_open = true;
        errdefer {
            if (batch_open) _ = metal_c.zdraw_metal_batch_end(batch);
        }
        try self.encodeApply(batch, pattern, input, output, width, is_merge);
        const result = metal_c.zdraw_metal_batch_end(batch);
        batch_open = false;
        if (result != 0) return error.MetalDispatchFailed;
    }

    fn encodeApply(
        self: *Context,
        batch: *anyopaque,
        pattern: *const Pattern,
        input: *anyopaque,
        output: *anyopaque,
        width: usize,
        is_merge: bool,
    ) !void {
        const params = ApplyParams{
            .source_tokens = try u32Fit(pattern.source_tokens),
            .destination_tokens = try u32Fit(pattern.destination_tokens),
            .width = try u32Fit(width),
            .region_size = try u32Fit(pattern.shape.region_size),
            .destinations_per_region = try u32Fit(pattern.shape.destinations_per_region),
            .token_side = try u32Fit(pattern.shape.token_side),
            .region_side = try u32Fit(pattern.shape.region_side),
            .selection_layout = @intFromEnum(pattern.config.selection_layout),
            .assignment_scope = @intFromEnum(pattern.config.assignment_scope),
        };
        const matrix = if (is_merge or pattern.unmerge == .paper_normalized_transpose)
            pattern.merge_matrix.handle
        else
            pattern.assignment.handle;
        const inner = if (is_merge) pattern.source_tokens else pattern.destination_tokens;
        if (pattern.config.assignment_scope == .global and inner % 8 == 0) {
            const pipeline = if (is_merge) self.merge_mma_pipe else self.unmerge_mma_pipe;
            const rows = if (is_merge)
                pattern.destination_tokens
            else
                pattern.source_tokens;
            try applyMma(
                batch,
                pipeline,
                .{ matrix, input, output, null, null },
                &params,
                rows,
                width,
            );
        } else {
            const pipeline = if (is_merge) self.merge_pipe else self.unmerge_pipe;
            const grid = if (is_merge)
                pattern.destination_tokens * width
            else
                pattern.source_tokens * width;
            try applyEncode(
                batch,
                pipeline,
                .{ matrix, input, output, null, null },
                &params,
                grid,
            );
        }
    }

    fn encodeCopy(
        self: *Context,
        batch: *anyopaque,
        input: *anyopaque,
        output: *anyopaque,
        source_offset: usize,
        destination_offset: usize,
        count: usize,
    ) !void {
        if (count == 0) return;
        const params = CopyParams{
            .count = try u32Fit(count),
            .source_offset = try u32Fit(source_offset),
            .destination_offset = try u32Fit(destination_offset),
        };
        try rawEncode(
            batch,
            self.copy_pipe,
            .{ input, output, null, null, null },
            &params,
            @sizeOf(CopyParams),
            2,
            count,
        );
    }

    fn scratchFor(
        self: *Context,
        shape: toma_config.Shape,
        config: toma_config.Config,
    ) !*Scratch {
        if (self.scratch) |*scratch| {
            if (scratch.matches(shape, config)) return scratch;
            scratch.deinit();
            self.scratch = null;
        }
        self.scratch = try Scratch.init(self.device, shape, config);
        return &self.scratch.?;
    }
};

const Scratch = struct {
    normalized: mbuffer.Buffer,
    selection: ?mbuffer.Buffer,
    similarity: mbuffer.Buffer,
    max_similarity: mbuffer.Buffer,
    selected: mbuffer.Buffer,
    selected_features: ?mbuffer.Buffer,
    fallback_merge: ?mbuffer.Buffer,
    shape: toma_config.Shape,
    config: toma_config.Config,

    fn init(
        device: *anyopaque,
        shape: toma_config.Shape,
        config: toma_config.Config,
    ) !Scratch {
        var normalized = try floatBuffer(device, shape.source_tokens * shape.feature_width);
        errdefer normalized.deinit();
        var selection: ?mbuffer.Buffer = if (config.mode == .official_b578009)
            try floatBuffer(device, shape.source_tokens * shape.feature_width)
        else
            null;
        errdefer if (selection) |*buffer| buffer.deinit();
        var similarity = try floatBuffer(device, shape.source_tokens * shape.region_size);
        errdefer similarity.deinit();
        var max_similarity = try floatBuffer(device, shape.source_tokens);
        errdefer max_similarity.deinit();
        var selected = try mbuffer.Buffer.empty(device, shape.source_tokens);
        errdefer selected.deinit();
        var selected_features: ?mbuffer.Buffer =
            if (config.assignment_scope == .global and shape.feature_width % 8 == 0)
                try floatBuffer(
                    device,
                    config.destination_tokens * shape.feature_width,
                )
            else
                null;
        errdefer if (selected_features) |*buffer| buffer.deinit();
        const fallback_merge: ?mbuffer.Buffer =
            if (selection == null and selected_features == null)
                try floatBuffer(
                    device,
                    config.destination_tokens * shape.feature_width,
                )
            else
                null;
        return .{
            .normalized = normalized,
            .selection = selection,
            .similarity = similarity,
            .max_similarity = max_similarity,
            .selected = selected,
            .selected_features = selected_features,
            .fallback_merge = fallback_merge,
            .shape = shape,
            .config = config,
        };
    }

    fn matches(
        self: *const Scratch,
        shape: toma_config.Shape,
        config: toma_config.Config,
    ) bool {
        return std.meta.eql(self.shape, shape) and std.meta.eql(self.config, config);
    }

    fn mergedHandle(self: *Scratch) *anyopaque {
        if (self.selected_features) |*buffer| return buffer.handle;
        if (self.selection) |*buffer| return buffer.handle;
        return self.fallback_merge.?.handle;
    }

    fn deinit(self: *Scratch) void {
        if (self.fallback_merge) |*buffer| buffer.deinit();
        if (self.selected_features) |*buffer| buffer.deinit();
        self.selected.deinit();
        self.max_similarity.deinit();
        self.similarity.deinit();
        if (self.selection) |*buffer| buffer.deinit();
        self.normalized.deinit();
        self.* = undefined;
    }
};

fn compile(device: *anyopaque, entry: [*:0]const u8, err: *[1024]u8) !*anyopaque {
    return mpipe.required(device, shader.source.ptr, entry, err);
}

fn patternEncode(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const PatternParams,
    params_index: u32,
    grid: usize,
) !void {
    try rawEncode(
        batch,
        pipeline,
        buffers,
        params,
        @sizeOf(PatternParams),
        params_index,
        grid,
    );
}

fn applyEncode(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const ApplyParams,
    grid: usize,
) !void {
    try rawEncode(batch, pipeline, buffers, params, @sizeOf(ApplyParams), 3, grid);
}

fn applyMma(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const ApplyParams,
    rows: usize,
    columns: usize,
) !void {
    const groups = (rows + 31) / 32 * ((columns + 31) / 32);
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
        @sizeOf(ApplyParams),
        3,
        groups,
        32,
        1,
    ) != 0) return error.MetalDispatchFailed;
}

fn patternMma(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const PatternParams,
    rows: usize,
    columns: usize,
) !void {
    const groups = (rows + 31) / 32 * ((columns + 31) / 32);
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
        @sizeOf(PatternParams),
        3,
        groups,
        32,
        1,
    ) != 0) return error.MetalDispatchFailed;
}

fn rawEncode(
    batch: *anyopaque,
    pipeline: *anyopaque,
    buffers: [5]?*anyopaque,
    params: *const anyopaque,
    params_len: usize,
    params_index: u32,
    grid: usize,
) !void {
    const threads = @min(@as(usize, 256), metal_c.zdraw_metal_pipeline_threads(pipeline));
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
        params_len,
        params_index,
        grid,
        threads,
        0,
    ) != 0) return error.MetalDispatchFailed;
}

fn patternParams(shape: toma_config.Shape, config: toma_config.Config) !PatternParams {
    return .{
        .source_tokens = try u32Fit(shape.source_tokens),
        .feature_width = try u32Fit(shape.feature_width),
        .destination_tokens = try u32Fit(config.destination_tokens),
        .region_count = try u32Fit(config.region_count),
        .region_size = try u32Fit(shape.region_size),
        .destinations_per_region = try u32Fit(shape.destinations_per_region),
        .token_side = try u32Fit(shape.token_side),
        .region_side = try u32Fit(shape.region_side),
        .selection_layout = @intFromEnum(config.selection_layout),
        .assignment_scope = @intFromEnum(config.assignment_scope),
        .mask_selected = if (config.mode == .paper_spec) 1 else 0,
        .raw_zero_norm = if (config.mode == .official_b578009) 1 else 0,
        .assignment_scale = config.assignment_scale,
    };
}

fn floatBuffer(device: *anyopaque, count: usize) !mbuffer.Buffer {
    return mbuffer.Buffer.empty(device, try std.math.mul(usize, count, @sizeOf(f32)));
}

fn u32Fit(value: usize) !u32 {
    if (value > std.math.maxInt(u32)) return error.InvalidShape;
    return @intCast(value);
}

const fixture_features = [_]f32{
    1, 0, 0, 0.9, 0.1, 0, 1, 0, 0, 0.9, 0.1, 0,
    0, 1, 0, 0,   0,   1, 0, 1, 0, 0,   0,   1,
    1, 0, 0, 0.9, 0.1, 0, 1, 0, 0, 0.9, 0.1, 0,
    0, 1, 0, 0,   0,   1, 0, 1, 0, 0,   0,   1,
};

fn metalParity(config: toma_config.Config) !void {
    var context = Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => return error.SkipZigTest,
        else => return err,
    };
    defer context.deinit();
    var expected = try toma_oracle.build(
        std.testing.allocator,
        &fixture_features,
        16,
        3,
        config,
    );
    defer expected.deinit(std.testing.allocator);
    var actual = try context.build(&fixture_features, 16, 3, config);
    defer actual.deinit();
    var destinations: [8]u32 = undefined;
    var assignment: [8 * 16]f32 = undefined;
    var merge_matrix: [8 * 16]f32 = undefined;
    try actual.read(&destinations, &assignment, &merge_matrix);
    try std.testing.expectEqualSlices(u32, expected.destinations, &destinations);
    try expectClose(expected.assignment, &assignment, 2e-6);
    try expectClose(expected.merge_matrix, &merge_matrix, 2e-6);
    const expected_merge = try toma_oracle.applyMerge(
        std.testing.allocator,
        expected,
        &fixture_features,
        3,
    );
    defer std.testing.allocator.free(expected_merge);
    const actual_merge = try context.merge(
        std.testing.allocator,
        &actual,
        &fixture_features,
        3,
    );
    defer std.testing.allocator.free(actual_merge);
    try expectClose(expected_merge, actual_merge, 2e-5);
    const expected_unmerge = try toma_oracle.applyUnmerge(
        std.testing.allocator,
        expected,
        expected_merge,
        3,
    );
    defer std.testing.allocator.free(expected_unmerge);
    const actual_unmerge = try context.unmerge(
        std.testing.allocator,
        &actual,
        expected_merge,
        3,
    );
    defer std.testing.allocator.free(actual_unmerge);
    try expectClose(expected_unmerge, actual_unmerge, 2e-5);
}

test "Metal released mode matches the CPU oracle without stage readbacks" {
    try metalParity(.{
        .mode = .official_b578009,
        .destination_tokens = 8,
        .region_count = 4,
        .selection_layout = .tile,
        .assignment_scope = .region_local,
        .unmerge = .official_raw_transpose,
        .assignment_scale = 3,
    });
}

test "Metal paper mode matches tile selection with global assignment" {
    try metalParity(.{
        .mode = .paper_spec,
        .destination_tokens = 8,
        .region_count = 4,
        .selection_layout = .tile,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 3,
    });
}

test "Metal global score MMA matches the paper assignment" {
    var context = Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => return error.SkipZigTest,
        else => return err,
    };
    defer context.deinit();
    var features: [16 * 8]f32 = @splat(0);
    for (0..16) |token| {
        @memcpy(
            features[token * 8 ..][0..3],
            fixture_features[token * 3 ..][0..3],
        );
    }
    const config = toma_config.Config{
        .mode = .paper_spec,
        .destination_tokens = 8,
        .region_count = 4,
        .selection_layout = .tile,
        .assignment_scope = .global,
        .unmerge = .paper_normalized_transpose,
        .assignment_scale = 3,
    };
    var expected = try toma_oracle.build(std.testing.allocator, &features, 16, 8, config);
    defer expected.deinit(std.testing.allocator);
    var actual = try context.build(&features, 16, 8, config);
    defer actual.deinit();
    var destinations: [8]u32 = undefined;
    var assignment: [8 * 16]f32 = undefined;
    var merge_matrix: [8 * 16]f32 = undefined;
    try actual.read(&destinations, &assignment, &merge_matrix);
    try std.testing.expectEqualSlices(u32, expected.destinations, &destinations);
    try expectClose(expected.assignment, &assignment, 5e-6);
    try expectClose(expected.merge_matrix, &merge_matrix, 5e-6);
}

test "Metal modal pack keeps caption tokens and RoPE indices aligned" {
    var context = Context.init() catch |err| switch (err) {
        error.MetalNotAvailable => return error.SkipZigTest,
        else => return err,
    };
    defer context.deinit();
    const config = toma_config.Config{
        .mode = .official_b578009,
        .destination_tokens = 8,
        .region_count = 4,
        .selection_layout = .tile,
        .assignment_scope = .region_local,
        .unmerge = .official_raw_transpose,
        .assignment_scale = 3,
    };
    var oracle = try toma_oracle.build(
        std.testing.allocator,
        &fixture_features,
        16,
        3,
        config,
    );
    defer oracle.deinit(std.testing.allocator);
    var pattern = try context.build(&fixture_features, 16, 3, config);
    defer pattern.deinit();

    var full_state: [18 * 3]f32 = undefined;
    const caption = [_]f32{ 7, 8, 9, 10, 11, 12 };
    @memcpy(full_state[0..fixture_features.len], &fixture_features);
    @memcpy(full_state[fixture_features.len..], &caption);
    var full_positions: [18][3]usize = undefined;
    for (&full_positions, 0..) |*pos, token| pos.* = .{ token, token + 100, token + 200 };

    var state_buffer =
        try mbuffer.Buffer.fromBytes(context.device, std.mem.sliceAsBytes(&full_state));
    defer state_buffer.deinit();
    var pos_buffer =
        try mbuffer.Buffer.fromBytes(context.device, std.mem.sliceAsBytes(&full_positions));
    defer pos_buffer.deinit();
    var reduced_buffer =
        try floatBuffer(context.device, (8 + 2) * 3);
    defer reduced_buffer.deinit();
    var reduced_pos_buffer =
        try mbuffer.Buffer.empty(context.device, (8 + 2) * @sizeOf([3]usize));
    defer reduced_pos_buffer.deinit();
    var unpacked_buffer = try floatBuffer(context.device, full_state.len);
    defer unpacked_buffer.deinit();

    const pack = metal_c.zdraw_metal_batch_begin(context.queue) orelse
        return error.MetalDispatchFailed;
    try context.encodePackModal(
        pack,
        &pattern,
        state_buffer.handle,
        pos_buffer.handle,
        reduced_buffer.handle,
        reduced_pos_buffer.handle,
        2,
        3,
    );
    if (metal_c.zdraw_metal_batch_end(pack) != 0) return error.MetalDispatchFailed;

    var reduced: [10 * 3]f32 = undefined;
    metal_c.zdraw_metal_read_buffer(
        reduced_buffer.handle,
        std.mem.sliceAsBytes(&reduced).ptr,
        @sizeOf(@TypeOf(reduced)),
    );
    const expected_image = try toma_oracle.applyMerge(
        std.testing.allocator,
        oracle,
        &fixture_features,
        3,
    );
    defer std.testing.allocator.free(expected_image);
    try expectClose(expected_image, reduced[0 .. 8 * 3], 2e-5);
    try std.testing.expectEqualSlices(f32, full_state[16 * 3 ..], reduced[8 * 3 ..]);

    var reduced_positions: [10][3]usize = undefined;
    metal_c.zdraw_metal_read_buffer(
        reduced_pos_buffer.handle,
        std.mem.sliceAsBytes(&reduced_positions).ptr,
        @sizeOf(@TypeOf(reduced_positions)),
    );
    for (oracle.destinations, 0..) |source, destination| {
        try std.testing.expectEqual(
            full_positions[source],
            reduced_positions[destination],
        );
    }
    try std.testing.expectEqualSlices(
        [3]usize,
        full_positions[16..],
        reduced_positions[8..],
    );

    const unpack = metal_c.zdraw_metal_batch_begin(context.queue) orelse
        return error.MetalDispatchFailed;
    try context.encodeUnpack(
        unpack,
        &pattern,
        reduced_buffer.handle,
        unpacked_buffer.handle,
        2,
        3,
    );
    if (metal_c.zdraw_metal_batch_end(unpack) != 0) return error.MetalDispatchFailed;
    var unpacked: [18 * 3]f32 = undefined;
    metal_c.zdraw_metal_read_buffer(
        unpacked_buffer.handle,
        std.mem.sliceAsBytes(&unpacked).ptr,
        @sizeOf(@TypeOf(unpacked)),
    );
    const expected_unmerged = try toma_oracle.applyUnmerge(
        std.testing.allocator,
        oracle,
        expected_image,
        3,
    );
    defer std.testing.allocator.free(expected_unmerged);
    try expectClose(expected_unmerged, unpacked[0 .. 16 * 3], 2e-5);
    try std.testing.expectEqualSlices(f32, full_state[16 * 3 ..], unpacked[16 * 3 ..]);
}

fn expectClose(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, tolerance);
    }
}
