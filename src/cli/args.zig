const std = @import("std");
const model = @import("model_kind.zig");
const runtime_options = @import("../runtime/runtime_options.zig");

pub const Command = union(enum) {
    generate: Generate,
    inspect: Inspect,
    preview: Preview,
    session: Session,
    doctor: Doctor,
    bench: Bench,
    fetch: Fetch,
    version,
    help: Help,
};

pub const Help = enum {
    overview,
    generate,
    session,
    fetch,
    bench,
    doctor,
    preview,
    inspect,
    version,
};
/// `zdraw bench`: the census case only (prompt, size, steps and seed are
/// fixed so cards stay comparable); --repeat adds warm runs.
pub const Bench = struct {
    kind: model.ModelKind = model.defaultKind(),
    weights_dir: []const u8,
    profile: runtime_options.Quality = .product,
    profile_explicit: bool = false,
    repeat: u32 = 1,
    output_path: []const u8 = "",
    card: bool = false,
    safety: bool = true,
};
/// `zdraw fetch <model> [--dir DIR] [--no-pack]`.
pub const Fetch = struct {
    kind: model.ModelKind,
    dir: []const u8 = "",
    no_pack: bool = false,
    /// `zdraw fetch nsfw-classifier`: the safety filter's image classifier.
    classifier: bool = false,
};

/// `zdraw doctor`: weights are optional, everything else is probed.
pub const Doctor = struct {
    kind: model.ModelKind = model.defaultKind(),
    weights_dir: []const u8 = "",
    json: bool = false,
};
pub const Generate = struct {
    kind: model.ModelKind = model.defaultKind(),
    weights_dir: []const u8,
    prompt: []const u8,
    output_path: []const u8 = "",
    width: u32 = 1024,
    height: u32 = 1024,
    // 0 = unset: main resolves the model's default (4 distilled, 50 base).
    steps: u32 = 0,
    seed: u64 = 42,
    // 0 = unset: main resolves the model's default (1.0 distilled, 4.0 base).
    guidance: f32 = 0,
    show: bool = false,
    progressive: bool = true,
    // Generate N times in one process (weights loaded once) with per-run
    // seconds: the warm in-session number competitor engines quote.
    repeat: u32 = 1,
    // Comma-separated seed list ("7,11,23"): one batched denoise produces one
    // image per seed (Klein only). Empty = the single --seed path.
    seeds_raw: []const u8 = "",
    /// The prompt-stage safety filter (`--safety off` disables it).
    safety: bool = true,
    /// img2img (Klein): start from this image's latent, `--strength` deep
    /// (the noise fraction to start from: 1 ignores it, 0.2 a gentle pass, 0 keeps it).
    init_image: []const u8 = "",
    strength: f32 = 0.6,
    /// "Fix a part": a mask image (white = redraw) with --init-image.
    mask: []const u8 = "",
    /// Instruction editing (Klein): the photo to change; the prompt says how.
    ref_image: []const u8 = "",
    // Typed production profile; defaults to product so normal `generate`
    // matches the benchmarked config. An explicit --profile wins over env;
    // the bare default defers to exported experiment variables.
    profile: runtime_options.Quality = .product,
    profile_explicit: bool = false,
    /// Whole-image f32 VAE reference decode (no strip streaming): the arm the
    /// strict tier is proven byte-identical to. Z-Image only; a lab knob.
    vae_reference: bool = false,
};
pub const Inspect = struct {
    path: []const u8,
};
pub const Preview = struct {
    prompt: []const u8,
    output_path: []const u8 = "",
    width: u32 = 1024,
    height: u32 = 1024,
    seed: u64 = 42,
    show: bool = false,
};
pub const Session = struct {
    kind: model.ModelKind = model.defaultKind(),
    weights_dir: []const u8 = "",
    safety: bool = true,
    output_dir: []const u8 = "",
    width: u32 = 512,
    height: u32 = 512,
    // 0 = unset: resolved from the model's policy (4 distilled / 50 base).
    steps: u32 = 0,
    seed: u64 = 42,
    guidance: f32 = 0,
    // Set when the value came from the user (flag or an in-session command),
    // so switching models re-resolves only the defaults.
    steps_explicit: bool = false,
    guidance_explicit: bool = false,
    show: bool = true,
    progressive: bool = true,
    preview: bool = false,
    auto_save: bool = true,
    // Same typed production profile as generate; product by default.
    profile: runtime_options.Quality = .product,
    profile_explicit: bool = false,
};

pub const ParseError = error{
    InvalidCommand,
    InvalidDimension,
    UnsupportedKleinResolution,
    InvalidModel,
    InvalidNumber,
    MissingCommand,
    MissingOptionValue,
    MissingPrompt,
    MissingModel,
    MissingOutput,
    MissingOutputDir,
    MissingPath,
    MissingWeights,
    HelpRequested,
    InvalidStrength,
    InvalidGuidance,
    EditUnsupported,
    ConflictingEdits,
    MissingInitImage,
    ConflictingSeeds,
    DuplicateSeed,
    EditSeedsUnsupported,
    ProfileUnsupported,
    SeedsUnsupported,
    UnknownOption,
};

pub const klein_resolution_hint = "Klein requires dimensions divisible by 32 " ++
    "and an area divisible by 8192 pixels, " ++
    "up to 1,048,576 pixels (1024x1024); try 128, 256, 512, 768 or 1024 square";

pub fn parse(iter: *std.process.Args.Iterator) ParseError!Command {
    _ = iter.next();
    const word = iter.next() orelse return .{ .help = .overview };
    if (isHelp(word) or std.mem.eql(u8, word, "help")) {
        const topic = if (iter.next()) |name| try helpTopic(name) else Help.overview;
        if (iter.next() != null) return error.UnknownOption;
        return .{ .help = topic };
    }
    return parseCommand(word, iter) catch |err| switch (err) {
        error.HelpRequested => .{ .help = try helpTopic(word) },
        else => return err,
    };
}

fn helpTopic(word: []const u8) ParseError!Help {
    const topic = std.meta.stringToEnum(Help, word) orelse return error.InvalidCommand;
    if (topic == .overview) return error.InvalidCommand;
    return topic;
}

fn isHelp(word: []const u8) bool {
    return std.mem.eql(u8, word, "--help") or std.mem.eql(u8, word, "-h");
}

fn parseCommand(word: []const u8, iter: *std.process.Args.Iterator) ParseError!Command {
    if (std.mem.eql(u8, word, "generate")) return .{ .generate = try parseGenerate(iter) };
    if (std.mem.eql(u8, word, "inspect")) return .{ .inspect = try parseInspect(iter) };
    if (std.mem.eql(u8, word, "preview")) return .{ .preview = try parsePreview(iter) };
    if (std.mem.eql(u8, word, "session")) return .{ .session = try parseSession(iter) };
    if (std.mem.eql(u8, word, "doctor")) return .{ .doctor = try parseDoctor(iter) };
    if (std.mem.eql(u8, word, "bench")) return .{ .bench = try parseBench(iter) };
    if (std.mem.eql(u8, word, "fetch")) return .{ .fetch = try parseFetch(iter) };
    if (std.mem.eql(u8, word, "version") or std.mem.eql(u8, word, "--version")) {
        if (iter.next()) |extra| {
            if (isHelp(extra)) return .{ .help = .version };
            return error.UnknownOption;
        }
        return .version;
    }
    return error.InvalidCommand;
}

/// Flags shared by generate and session (identical field names in both
/// structs). Returns true when arg was consumed.
fn sharedOption(
    comptime T: type,
    out: *T,
    iter: *std.process.Args.Iterator,
    arg: []const u8,
) ParseError!bool {
    if (std.mem.eql(u8, arg, "--model")) {
        out.kind = try parseKind(try needValue(iter));
    } else if (std.mem.eql(u8, arg, "--weights")) {
        out.weights_dir = try needValue(iter);
    } else if (std.mem.eql(u8, arg, "--width")) {
        out.width = try parseU32(try needValue(iter));
    } else if (std.mem.eql(u8, arg, "--height")) {
        out.height = try parseU32(try needValue(iter));
    } else if (std.mem.eql(u8, arg, "--steps")) {
        out.steps = try parseU32(try needValue(iter));
        if (out.steps == 0) return error.InvalidNumber;
        if (T == Session) out.steps_explicit = true;
    } else if (std.mem.eql(u8, arg, "--guidance")) {
        out.guidance = try parseF32(try needValue(iter));
        // 0 is the "use the model's default" sentinel; the per-model check
        // runs once the model is known (checkGuidance), since --guidance may
        // precede --model.
        if (out.guidance <= 0) return error.InvalidGuidance;
        if (T == Session) out.guidance_explicit = true;
    } else if (std.mem.eql(u8, arg, "--seed")) {
        out.seed = try parseU64(try needValue(iter));
    } else if (std.mem.eql(u8, arg, "--show")) {
        out.show = true;
    } else if (std.mem.eql(u8, arg, "--no-progressive")) {
        out.progressive = false;
    } else if (std.mem.eql(u8, arg, "--profile")) {
        const value = try needValue(iter);
        out.profile = runtime_options.Quality.parse(value) orelse return error.UnknownOption;
        out.profile_explicit = true;
    } else if (std.mem.eql(u8, arg, "--vae-reference")) {
        if (T != Generate) return error.UnknownOption;
        out.vae_reference = true;
    } else if (std.mem.eql(u8, arg, "--safety")) {
        out.safety = try parseOnOff(try needValue(iter));
    } else {
        return false;
    }
    return true;
}

fn parseOnOff(value: []const u8) ParseError!bool {
    if (std.mem.eql(u8, value, "on")) return true;
    if (std.mem.eql(u8, value, "off")) return false;
    return error.UnknownOption;
}

/// Validate --seeds at parse time so a typo fails before weights load.
fn checkSeeds(raw: []const u8) ParseError!void {
    var prefix_len: usize = 0;
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (trimmed.len == 0) return error.InvalidNumber;
        const seed = std.fmt.parseInt(u64, trimmed, 10) catch return error.InvalidNumber;
        var previous = std.mem.splitScalar(u8, raw[0..prefix_len], ',');
        while (previous.next()) |earlier| {
            if (earlier.len == 0) continue;
            const value = std.fmt.parseInt(u64, std.mem.trim(u8, earlier, " "), 10) catch
                return error.InvalidNumber;
            if (value == seed) return error.DuplicateSeed;
        }
        prefix_len += part.len + 1;
    }
}

fn parseSession(iter: *std.process.Args.Iterator) ParseError!Session {
    var out = Session{};

    while (iter.next()) |arg| {
        if (isHelp(arg)) return error.HelpRequested;
        if (try sharedOption(Session, &out, iter, arg)) continue;
        if (std.mem.eql(u8, arg, "--out-dir")) {
            out.output_dir = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--no-show")) {
            out.show = false;
        } else if (std.mem.eql(u8, arg, "--no-auto-save")) {
            out.auto_save = false;
        } else if (std.mem.eql(u8, arg, "--preview")) {
            out.preview = true;
        } else {
            return error.UnknownOption;
        }
    }

    if (out.auto_save and out.output_dir.len == 0) {
        return error.MissingOutputDir;
    }
    try checkGuidance(out.kind, out.guidance);
    try checkDimensions(out.kind, out.width, out.height, out.preview);
    if (out.profile_explicit and out.kind != .z_image_turbo) return error.ProfileUnsupported;
    return out;
}

fn parseF32(text: []const u8) ParseError!f32 {
    const value = std.fmt.parseFloat(f32, text) catch return error.InvalidNumber;
    if (!std.math.isFinite(value)) return error.InvalidNumber;
    return value;
}

fn parsePreview(iter: *std.process.Args.Iterator) ParseError!Preview {
    var out = Preview{ .prompt = "" };

    while (iter.next()) |arg| {
        if (isHelp(arg)) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--prompt")) {
            out.prompt = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out.output_path = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--width")) {
            out.width = try parseU32(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--height")) {
            out.height = try parseU32(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--seed")) {
            out.seed = try parseU64(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--show")) {
            out.show = true;
        } else {
            return error.UnknownOption;
        }
    }

    if (out.prompt.len == 0) return error.MissingPrompt;
    if (out.output_path.len == 0) return error.MissingOutput;
    if (out.width == 0 or out.height == 0) return error.InvalidDimension;
    return out;
}

fn parseGenerate(iter: *std.process.Args.Iterator) ParseError!Generate {
    var strength_set = false;
    var seed_set = false;
    var out = Generate{
        .weights_dir = "",
        .prompt = "",
    };

    while (iter.next()) |arg| {
        if (isHelp(arg)) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--seed")) seed_set = true;
        if (try sharedOption(Generate, &out, iter, arg)) continue;
        if (std.mem.eql(u8, arg, "--prompt")) {
            out.prompt = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out.output_path = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            out.repeat = try parseU32(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--seeds")) {
            out.seeds_raw = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--init-image")) {
            out.init_image = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--strength")) {
            out.strength = try parseF32(try needValue(iter));
            strength_set = true;
        } else if (std.mem.eql(u8, arg, "--mask")) {
            out.mask = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--edit")) {
            out.ref_image = try needValue(iter);
        } else {
            return error.UnknownOption;
        }
    }

    if (out.prompt.len == 0) return error.MissingPrompt;
    if (out.output_path.len == 0) return error.MissingOutput;
    try checkGuidance(out.kind, out.guidance);
    try checkDimensions(out.kind, out.width, out.height, false);
    if (out.repeat == 0) return error.InvalidNumber;
    try checkEdits(out, strength_set);
    if (seed_set and out.seeds_raw.len > 0) return error.ConflictingSeeds;
    if (out.seeds_raw.len > 0) {
        // Batched seeds are a Klein route; failing here beats a silent
        // single-image render at the default seed.
        if (out.kind == .z_image_turbo) return error.SeedsUnsupported;
        try checkSeeds(out.seeds_raw);
    }
    if (out.profile_explicit and out.kind != .z_image_turbo) return error.ProfileUnsupported;
    if (out.vae_reference and out.kind != .z_image_turbo) return error.ProfileUnsupported;
    return out;
}

fn checkEdits(out: Generate, strength_set: bool) ParseError!void {
    const edit = out.ref_image.len > 0;
    const init = out.init_image.len > 0;
    if (edit and init) return error.ConflictingEdits;
    if ((out.mask.len > 0 or strength_set) and !init) return error.MissingInitImage;
    if (out.strength < 0 or out.strength > 1) return error.InvalidStrength;
    if ((edit or init) and out.kind == .z_image_turbo) return error.EditUnsupported;
    if ((edit or init) and out.seeds_raw.len > 0) return error.EditSeedsUnsupported;
}

fn parseBench(iter: *std.process.Args.Iterator) ParseError!Bench {
    var out = Bench{ .weights_dir = "" };
    while (iter.next()) |arg| {
        if (isHelp(arg)) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--card")) {
            out.card = true;
        } else if (std.mem.eql(u8, arg, "--model")) {
            out.kind = try parseKind(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--weights")) {
            out.weights_dir = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--profile")) {
            out.profile = runtime_options.Quality.parse(try needValue(iter)) orelse
                return error.UnknownOption;
            out.profile_explicit = true;
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            out.repeat = try parseU32(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--out")) {
            out.output_path = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--safety")) {
            out.safety = try parseOnOff(try needValue(iter));
        } else return error.UnknownOption;
    }
    if (out.repeat == 0) return error.InvalidNumber;
    if (out.profile_explicit and out.kind != .z_image_turbo) return error.ProfileUnsupported;
    return out;
}

fn parseFetch(iter: *std.process.Args.Iterator) ParseError!Fetch {
    const name = iter.next() orelse return error.MissingModel;
    if (isHelp(name)) return error.HelpRequested;
    var out = if (std.mem.eql(u8, name, "nsfw-classifier"))
        Fetch{ .kind = model.defaultKind(), .classifier = true }
    else
        Fetch{ .kind = try parseKind(name) };
    while (iter.next()) |arg| {
        if (isHelp(arg)) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--dir")) {
            out.dir = try needValue(iter);
        } else if (std.mem.eql(u8, arg, "--no-pack")) {
            out.no_pack = true;
        } else return error.UnknownOption;
    }
    return out;
}

fn parseDoctor(iter: *std.process.Args.Iterator) ParseError!Doctor {
    var out = Doctor{};
    while (iter.next()) |arg| {
        if (isHelp(arg)) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--json")) {
            out.json = true;
        } else if (std.mem.eql(u8, arg, "--model")) {
            out.kind = try parseKind(try needValue(iter));
        } else if (std.mem.eql(u8, arg, "--weights")) {
            out.weights_dir = try needValue(iter);
        } else return error.UnknownOption;
    }
    return out;
}

fn parseInspect(iter: *std.process.Args.Iterator) ParseError!Inspect {
    const path = iter.next() orelse return error.MissingPath;
    if (isHelp(path)) return error.HelpRequested;
    if (iter.next() != null) return error.UnknownOption;
    return .{ .path = path };
}

fn needValue(iter: *std.process.Args.Iterator) ParseError![]const u8 {
    const value = iter.next() orelse return error.MissingOptionValue;
    if (std.mem.startsWith(u8, value, "--") or value.len == 0) return error.MissingOptionValue;
    return value;
}

fn parseU32(text: []const u8) ParseError!u32 {
    return std.fmt.parseInt(u32, text, 10) catch error.InvalidNumber;
}

fn parseU64(text: []const u8) ParseError!u64 {
    return std.fmt.parseInt(u64, text, 10) catch error.InvalidNumber;
}

fn parseKind(name: []const u8) ParseError!model.ModelKind {
    return model.parsePublic(name) orelse error.InvalidModel;
}

/// Reject a guidance the model cannot honour before loading weights: the
/// step-distilled models have no unconditional branch to steer with.
pub fn checkGuidance(kind: model.ModelKind, guidance: f32) ParseError!void {
    if (guidance == 0 or guidance == 1) return;
    if (!model.policy(kind).cfg_allowed) return error.InvalidGuidance;
}

/// Validate sizes before loading weights; preview has no model constraints.
pub fn checkDimensions(kind: model.ModelKind, width: u32, height: u32, preview: bool) ParseError!void {
    const multiple: u32 = if (preview) 1 else if (kind == .z_image_turbo) 16 else 32;
    if (width == 0 or height == 0 or width % multiple != 0 or height % multiple != 0) {
        return error.InvalidDimension;
    }
    if (!preview and kind != .z_image_turbo and !model.kleinDimsFit(width, height)) {
        return error.UnsupportedKleinResolution;
    }
}
