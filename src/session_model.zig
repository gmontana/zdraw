//! Session model switching: release the old runtime before loading the next.
//! A failed load leaves the prompt loop running with no model. Sampling defaults
//! follow the new model; explicit settings are retained or rejected if unsupported.

const std = @import("std");

const args = @import("args.zig");
const model = @import("model.zig");
const model_kind = @import("model_kind.zig");
const model_paths = @import("model_paths.zig");
const checkpoint = @import("weights.zig");
const runtime = @import("session_runtime.zig");
const runtime_options = @import("runtime_options.zig");
const session_state = @import("session_state.zig");
const util = @import("session_util.zig");

const State = session_state.State;

/// Where each model's weights directory is configured. Exhaustive on
/// purpose: a new model kind must state its source instead of silently
/// having none.
fn weightsEnv(kind: model.ModelKind) [*:0]const u8 {
    return switch (kind) {
        .z_image_turbo => "ZDRAW_ZIMAGE_WEIGHTS",
        .flux2_klein_4b => "ZDRAW_KLEIN4B_WEIGHTS",
        .flux2_klein_9b => "ZDRAW_KLEIN9B_WEIGHTS",
        .flux2_klein_base_4b => "ZDRAW_KLEIN_BASE4B_WEIGHTS",
        .flux2_klein_base_9b => "ZDRAW_KLEIN_BASE9B_WEIGHTS",
        .flux2_klein_9b_kv => "ZDRAW_KLEIN9BKV_WEIGHTS",
    };
}

/// The configured weights directory for `kind`, or null when unset.
fn weightsFor(kind: model.ModelKind) ?[]const u8 {
    const raw = std.c.getenv(weightsEnv(kind)) orelse return null;
    const path = std.mem.span(raw);
    return if (path.len == 0) null else path;
}

/// Steps and guidance for `kind`, preserving whatever the user set by hand.
fn samplingFor(request: args.Session, kind: model.ModelKind) !struct { u32, f32 } {
    const steps: u32 = if (request.steps_explicit) request.steps else 0;
    const guidance: f32 = if (request.guidance_explicit) request.guidance else 0;
    return model_kind.sampling(kind, steps, guidance);
}

pub fn setModel(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    value: []const u8,
) !void {
    const name = util.clean(value);
    const kind = model_kind.parsePublic(name) orelse {
        try util.writeText(out, allocator, "unknown model; see `zdraw help` for the list\n");
        return;
    };
    if (state.request.preview) {
        try util.writeText(out, allocator, "preview mode runs no model\n");
        return;
    }
    args.checkDimensions(kind, state.request.width, state.request.height, false) catch {
        return util.writeText(out, allocator, "set a compatible size before switching models\n");
    };
    if (kind == state.request.kind and state.rt != null) {
        try util.showPath(out, allocator, "already on ", model.kindName(kind));
        return;
    }
    // A different model must never reuse the current model's directory.
    const explicit = weightsFor(kind) orelse
        if (kind == state.request.kind) state.request.weights_dir else "";
    const weights = model_paths.resolve(allocator, state.env, kind, explicit) catch |err| {
        return util.showPath(out, allocator, "could not resolve model directory: ", @errorName(err));
    };
    var keep = false;
    defer if (!keep) allocator.free(weights);
    checkpoint.validate(out.io, allocator, kind, weights) catch {
        const hint = try std.fmt.allocPrint(
            allocator,
            "model files missing; run zdraw fetch {s} --dir \"{s}\"\n",
            .{ name, weights },
        );
        defer allocator.free(hint);
        return util.writeText(out, allocator, hint);
    };
    const pair = samplingFor(state.request, kind) catch {
        try util.writeText(
            out,
            allocator,
            "that model cannot run at the guidance you set; change it first\n",
        );
        return;
    };

    if (!try swap(out, allocator, state, kind, weights)) return;
    if (state.weights_owned) |old| allocator.free(old);
    state.weights_owned = weights;
    keep = true;
    state.request.kind = kind;
    state.request.weights_dir = weights;
    state.request.steps = pair[0];
    state.request.guidance = pair[1];

    try showModel(out, allocator, state);
}

fn showModel(out: *util.Output, allocator: std.mem.Allocator, state: *State) !void {
    const kind = state.request.kind;
    const text = try std.fmt.allocPrint(
        allocator,
        "model {s} | {d}x{d} | {d} steps | guidance {d:.1}\n",
        .{
            model.kindName(kind),
            state.request.width,
            state.request.height,
            state.request.steps,
            state.request.guidance,
        },
    );
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
}

/// Tear the old runtime down and build the new one. Returns false when the
/// build failed, leaving the session alive with no model.
fn swap(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    kind: model.ModelKind,
    weights: []const u8,
) !bool {
    // Release before building: the two weight sets do not fit together.
    if (state.rt) |*rt| rt.deinit(out.io, allocator);
    state.rt = null;

    // Re-apply the execution profile for the target model, with the same
    // precedence as startup (an explicit --profile overwrites, the default
    // defers to exported variables).
    if (kind == .z_image_turbo) {
        runtime_options.applyEnv(state.request.profile, state.request.profile_explicit);
    }

    state.rt = runtime.Runtime.init(out.io, allocator, kind, weights) catch |err| {
        try util.showPath(out, allocator, "could not load ", @errorName(err));
        try util.writeText(out, allocator, "no model loaded; run `model <name>` again\n");
        return false;
    };
    return true;
}

test "an explicit step count survives a model swap" {
    var request = args.Session{ .kind = .flux2_klein_4b, .steps = 8, .steps_explicit = true };
    const pair = try samplingFor(request, .flux2_klein_base_4b);
    try std.testing.expectEqual(@as(u32, 8), pair[0]);
    try std.testing.expectEqual(@as(f32, 4.0), pair[1]);

    // Without an explicit value the target model's default applies, so a
    // base variant does not inherit the distilled 4 steps.
    request.steps_explicit = false;
    const defaulted = try samplingFor(request, .flux2_klein_base_4b);
    try std.testing.expectEqual(@as(u32, 50), defaulted[0]);
}

test "a guidance the target model cannot honour refuses the swap" {
    const request = args.Session{
        .kind = .flux2_klein_base_4b,
        .guidance = 4.0,
        .guidance_explicit = true,
    };
    try std.testing.expectError(error.GuidanceUnsupported, samplingFor(request, .flux2_klein_4b));
}
