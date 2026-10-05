//! Rendering actions for the interactive zdraw session.

const std = @import("std");

const image = @import("image.zig");
const image_quality = @import("image_quality.zig");
const safety = @import("safety.zig");
const render = @import("render.zig");
const cmd = @import("session_cmd.zig");
const session_files = @import("session_files.zig");
const session_state = @import("session_state.zig");
const terminal_preview = @import("terminal_preview.zig");
const terminal = @import("terminal.zig");
const util = @import("session_util.zig");

const Frame = session_state.Frame;
const State = session_state.State;

pub const Intent = enum {
    new,
    refine,
    reroll,
    undo,
};

pub fn renderOne(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    intent: Intent,
) !void {
    if (!state.request.preview and state.request.safety) {
        try safety.gate(out.io, allocator, state.prompt);
    }
    state.count += 1;
    const seed = seedFor(state, state.count);

    try util.renderHeader(
        out,
        allocator,
        state.env,
        label(intent),
        state.request.width,
        state.request.height,
        state.request.steps,
    );
    const start = std.Io.Timestamp.now(out.io, .awake);
    const result = try makeFrameSeed(out.io, allocator, state, seed);
    try saveCandidate(out, allocator, state, .{
        .frame = result,
        .seed = seed,
        .start = start,
    });
}

/// A rendered frame; saveCandidate takes ownership of its pixels.
const Candidate = struct {
    frame: Frame,
    seed: u64,
    start: std.Io.Timestamp,
};

/// Reject or save a finished frame, update the last image, and print its result.
fn saveCandidate(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    cand: Candidate,
) !void {
    const result = cand.frame;
    var keep_pixels = false;
    defer if (!keep_pixels) allocator.free(result.pixels);
    if (try rejectBlank(out, allocator, result)) return;
    if (!state.request.preview and state.request.safety) {
        try safety.imageGate(
            out.io,
            allocator,
            state.env,
            result.pixels,
            result.width,
            result.height,
        );
    }
    const path = if (state.request.auto_save)
        try autoSave(out.io, allocator, state, result)
    else
        null;
    var path_owned = path;
    defer if (path_owned) |p| allocator.free(p);
    if (state.request.show) {
        try util.writeText(out, allocator, "\n");
        try terminal.writeImage(
            out.io,
            allocator,
            result.pixels,
            result.width,
            result.height,
            .{ .env = state.env },
        );
    }
    const elapsed_ns: u64 = @intCast(cand.start.untilNow(out.io, .awake).toNanoseconds());
    state.clearLastFrame(allocator);
    allocator.free(state.last_path);
    if (path) |p| {
        state.last_path = p;
        path_owned = null;
    } else {
        state.last_path = &.{};
        state.last_frame = result;
        keep_pixels = true;
    }
    try cmd.writeResult(
        out,
        allocator,
        state.request,
        cand.seed,
        elapsed_ns,
        path,
        state.stats,
    );
}

fn autoSave(io: std.Io, allocator: std.mem.Allocator, state: *State, frame: Frame) ![]u8 {
    const bytes = try image.encodePng(allocator, frame.pixels, frame.width, frame.height);
    defer allocator.free(bytes);
    return session_files.writeNumbered(io, allocator, state.request.output_dir, state.count, bytes);
}

fn rejectBlank(out: *util.Output, allocator: std.mem.Allocator, frame: Frame) !bool {
    const stats = try image_quality.analyzeRgb(frame.pixels, frame.width, frame.height);
    if (!image_quality.isBlankLike(stats)) return false;
    const text = try std.fmt.allocPrint(
        allocator,
        "rejected blank/low-content image; no file written (luma {d}-{d}, mean {d:.1})\n",
        .{ stats.min_luma, stats.max_luma, stats.mean_luma },
    );
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
    return true;
}

pub fn undo(out: *util.Output, allocator: std.mem.Allocator, state: *State) !void {
    const previous = state.history.pop() orelse {
        try util.writeText(out, allocator, "nothing to undo\n");
        return;
    };
    allocator.free(state.prompt);
    state.prompt = previous;
    try renderOne(out, allocator, state, .undo);
}

pub fn reroll(out: *util.Output, allocator: std.mem.Allocator, state: *State) !void {
    if (state.prompt.len == 0) {
        try util.writeText(out, allocator, "nothing to reroll\n");
        return;
    }
    try renderOne(out, allocator, state, .reroll);
}

pub fn save(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    path: []const u8,
) !void {
    if (state.last_path.len == 0) {
        if (state.last_frame) |frame| {
            try image.writePng(out.io, allocator, path, frame.pixels, frame.width, frame.height);
            try util.showPath(out, allocator, "saved ", path);
            return;
        }
        try util.writeText(out, allocator, "nothing to save\n");
        return;
    }
    try std.Io.Dir.copyFile(
        .cwd(),
        state.last_path,
        .cwd(),
        path,
        out.io,
        .{ .make_path = true },
    );
    try util.showPath(out, allocator, "saved ", path);
}

fn label(intent: Intent) []const u8 {
    return switch (intent) {
        .new => "new",
        .refine => "refine",
        .reroll => "reroll",
        .undo => "undo",
    };
}

/// Wrap at the maximum seed instead of overflowing in checked builds.
fn seedFor(state: *const State, index: u32) u64 {
    return state.request.seed +% index;
}

fn makeFrameSeed(
    io: std.Io,
    allocator: std.mem.Allocator,
    state: *State,
    seed: u64,
) !Frame {
    if (!state.request.preview) {
        const rt = if (state.rt) |*rt| rt else return error.UnsupportedInference;
        const show = state.request.show and state.request.progressive and
            state.request.kind != .z_image_turbo;
        var view = try terminal_preview.View.init(io, state.env, show);
        try view.start();
        defer view.deinit(io);
        const result = try rt.generate(io, allocator, .{
            .prompt = state.prompt,
            .width = state.request.width,
            .height = state.request.height,
            .steps = state.request.steps,
            .seed = seed,
            .guidance = state.request.guidance,
        });
        return .{ .pixels = result.pixels, .width = result.width, .height = result.height };
    }

    return .{
        .pixels = try render.renderPreview(
            allocator,
            state.prompt,
            state.request.width,
            state.request.height,
            seed,
        ),
        .width = state.request.width,
        .height = state.request.height,
    };
}
