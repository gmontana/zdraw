//! Interactive generation with prompt history, settings and numbered outputs.
//! Saves each result unless --no-auto-save is set.

const std = @import("std");
const stanza = @import("stanza");

const args = @import("args.zig");
const cmd = @import("session_cmd.zig");
const model = @import("../runtime/model.zig");
const runtime_options = @import("../runtime/runtime_options.zig");
const runtime = @import("session_runtime.zig");
const session_model = @import("session_model.zig");
const session_render = @import("session_render.zig");
const session_state = @import("session_state.zig");
const util = @import("session_util.zig");

const history_path = ".zdraw_history";
const prompt_text = "zdraw ) ";
const State = session_state.State;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    request: args.Session,
) !void {
    compactProgress();
    if (request.auto_save and request.output_dir.len != 0) {
        try std.Io.Dir.cwd().createDirPath(io, request.output_dir);
    }

    var out = util.Output{ .io = io };
    var state = try State.init(io, allocator, env, request);
    defer state.deinit(io, allocator);
    try printIntro(&out, allocator, &state);

    var ed = stanza.Editor.init(allocator, .{
        .editing = .emacs,
        .complete = cmd.complete,
        .complete_style = .menu,
        .hint = cmd.hint,
        .paint = if (util.colorEnabled(env)) cmd.paint else null,
        .install_resize_handler = true,
    });
    defer ed.deinit();
    out.editor = &ed;
    try loadHistory(&ed);
    defer saveHistory(&ed);
    try promptLoop(&out, allocator, &ed, &state);
}

fn printIntro(out: *util.Output, allocator: std.mem.Allocator, state: *State) !void {
    const request = state.request;
    if (request.preview) {
        try util.writeText(out, allocator, "preview mode\n\n");
    } else {
        // Same typed execution profile as `generate`, applied before any
        // Runtime/Metal init. Env overrides still win.
        if (request.kind == .z_image_turbo) runtime_options.applyEnv(request.profile, request.profile_explicit);
        state.rt = try runtime.Runtime.init(
            out.io,
            allocator,
            request.kind,
            request.weights_dir,
        );
        try util.writeText(out, allocator, "\n");
    }
    try readyLine(out, allocator, request);
}

fn readyLine(out: *util.Output, allocator: std.mem.Allocator, request: args.Session) !void {
    const label = if (request.preview) "preview" else "ready";
    const target = if (!request.auto_save)
        "auto-save off"
    else
        request.output_dir;
    const text = try std.fmt.allocPrint(
        allocator,
        "{s}  {s} | {d}x{d} | {d} steps | seed {d} | {s}\n" ++
            "help   commands and settings\n\n",
        .{
            label,
            model.kindName(request.kind),
            request.width,
            request.height,
            request.steps,
            request.seed,
            target,
        },
    );
    defer allocator.free(text);
    try util.writeText(out, allocator, text);
}

fn promptLoop(
    out: *util.Output,
    allocator: std.mem.Allocator,
    ed: *stanza.Editor,
    state: *State,
) !void {
    while (true) {
        const line = ed.prompt(prompt_text) catch |err| switch (err) {
            error.Eof => break,
            error.Interrupted => continue,
            else => return err,
        };
        defer allocator.free(line);
        const text = util.clean(line);
        if (text.len == 0) continue;
        if (util.isQuit(text)) break;
        try ed.history.add(text);
        handleLine(out, allocator, state, text) catch |err| {
            if (err == error.OutOfMemory) return err;
            if (err == error.SafetyBlocked) continue;
            try util.showPath(out, allocator, "command failed: ", @errorName(err));
        };
    }
}

fn handleLine(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    text: []const u8,
) !void {
    if (std.mem.eql(u8, text, "/help")) return util.help(out, allocator);
    if (std.mem.startsWith(u8, text, "/")) {
        return util.writeText(out, allocator, "unknown command; type help for session commands\n");
    }
    if (std.mem.eql(u8, text, "help")) return util.help(out, allocator);
    if (std.mem.eql(u8, text, "prompt")) return cmd.showPrompt(out, allocator, state.prompt);
    if (std.mem.eql(u8, text, "clear")) return clearPrompt(out, allocator, state);
    if (std.mem.eql(u8, text, "undo")) return session_render.undo(out, allocator, state);
    if (std.mem.eql(u8, text, "reroll")) return session_render.reroll(out, allocator, state);
    if (isNew(text)) return startNew(out, allocator, state, text);
    if (std.mem.startsWith(u8, text, "seed ")) {
        return cmd.setSeed(out, allocator, &state.request, text[5..]);
    }
    if (std.mem.startsWith(u8, text, "steps ")) {
        return cmd.setSteps(out, allocator, &state.request, text[6..]);
    }
    if (std.mem.startsWith(u8, text, "size ")) {
        return cmd.setSize(out, allocator, &state.request, text[5..]);
    }
    if (std.mem.startsWith(u8, text, "model ")) {
        return session_model.setModel(out, allocator, state, text[6..]);
    }
    if (isStats(text)) return cmd.setStats(out, allocator, &state.stats, text);
    if (util.saveTarget(text)) |path| return session_render.save(out, allocator, state, path);

    // A failed `model` swap leaves no runtime; refuse to render rather than
    // propagating an error out of the prompt loop and ending the session.
    if (state.rt == null and !state.request.preview) {
        return util.writeText(out, allocator, "no model loaded; run `model <name>`\n");
    }
    try refine(allocator, state, text);
    try session_render.renderOne(out, allocator, state, .refine);
}

fn refine(allocator: std.mem.Allocator, state: *State, text: []const u8) !void {
    if (state.prompt.len > 0) {
        try state.history.append(allocator, try allocator.dupe(u8, state.prompt));
    }

    const next = try util.merge(allocator, state.prompt, text, false);
    allocator.free(state.prompt);
    state.prompt = next;
}

fn startNew(
    out: *util.Output,
    allocator: std.mem.Allocator,
    state: *State,
    text: []const u8,
) !void {
    const part = if (std.mem.eql(u8, text, "new")) "" else util.clean(text[4..]);
    try resetPrompt(allocator, state, part);
    if (part.len == 0) {
        try util.writeText(out, allocator, "prompt cleared\n");
        return;
    }
    try session_render.renderOne(out, allocator, state, .new);
}

fn clearPrompt(out: *util.Output, allocator: std.mem.Allocator, state: *State) !void {
    try resetPrompt(allocator, state, "");
    try util.writeText(out, allocator, "prompt cleared\n");
}

fn resetPrompt(allocator: std.mem.Allocator, state: *State, text: []const u8) !void {
    const next = try allocator.dupe(u8, text);
    clearUndo(allocator, state);
    allocator.free(state.prompt);
    state.prompt = next;
}

fn clearUndo(allocator: std.mem.Allocator, state: *State) void {
    for (state.history.items) |item| allocator.free(item);
    state.history.clearRetainingCapacity();
}

fn compactProgress() void {
    _ = setenv("ZDRAW_PROGRESS", "compact", 1);
}

fn isNew(text: []const u8) bool {
    return std.mem.eql(u8, text, "new") or std.mem.startsWith(u8, text, "new ");
}

fn isStats(text: []const u8) bool {
    return std.mem.eql(u8, text, "stats") or std.mem.startsWith(u8, text, "stats ");
}

fn loadHistory(ed: *stanza.Editor) !void {
    ed.history.load(history_path) catch |err| {
        var buf: [128]u8 = undefined;
        // printAbove is stanza's one host-output call: above the live line when
        // a prompt is active, plain write otherwise — so session messages keep
        // working if this ever fires mid-edit. Rows end \r\n for raw mode.
        const msg = try std.fmt.bufPrint(
            &buf,
            "zdraw: history load skipped: {s}\r\n",
            .{@errorName(err)},
        );
        try ed.printAbove(msg);
    };
}

fn saveHistory(ed: *stanza.Editor) void {
    ed.history.appendNew(history_path) catch |err| {
        std.debug.print("zdraw: history save skipped: {s}\n", .{@errorName(err)});
    };
}
